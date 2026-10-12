use std::fs;

use kvpack_core::{
    decode_authenticated_pack, decode_chunk, inspect_pack_header, ChunkObject, EncodedPack, Id32,
    ValidationContext,
};
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use crate::intent::IntentHasher;
use crate::telemetry::AuditOutcome;
use crate::{CacheLifecycle, PublishedArtifact, StoreError};

use super::publication::{write_immutable, DurableObjectKind};
use super::{
    audit::{self, AuditCapacity, AuditEventKey},
    fsync_dir, hex, vec_id, AuditEventKind, AuditObjectKind, LocalStore, QuarantinedUploadFile,
    UploadReservation, UploadState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthenticatedImportStatus {
    pub state: UploadState,
    pub manifest_id: Id32,
    pub next_chunk_ordinal: u64,
    pub publication_generation: u64,
}

impl LocalStore {
    /// Reserve a path-free authenticated-object import. The manifest identity
    /// is permanently bound to the idempotency key before any bytes arrive.
    pub fn begin_authenticated_import(
        &self,
        idempotency: &Id32,
        manifest_id: &Id32,
        expected_bytes: u64,
        publication_generation: u64,
    ) -> Result<AuthenticatedImportStatus, StoreError> {
        if manifest_id.iter().all(|byte| *byte == 0) {
            return Err(StoreError::Expectation("import manifest identity is zero"));
        }
        let mut intent = IntentHasher::new(b"kvpack/catalog/authenticated-import-intent/v1");
        intent.id(&self.tenant_namespace());
        intent.u64(self.key_epoch());
        intent.id(manifest_id);
        let state = self.reserve_upload(
            idempotency,
            UploadReservation {
                expected_bytes,
                publication_generation,
                intent_digest: intent.finish(),
                retention: super::RetentionInputs::conservative(expected_bytes, 1),
            },
        )?;
        {
            let connection = self.lock_catalog()?;
            let existing: Option<Vec<u8>> = connection
                .query_row(
                    "SELECT manifest_id FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                    params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                    |row| row.get(0),
                )
                .optional()?
                .flatten();
            if let Some(existing) = existing {
                if vec_id(existing)? != *manifest_id {
                    return Err(StoreError::Expectation(
                        "idempotency key was reused with a different manifest identity",
                    ));
                }
            } else {
                connection.execute(
                    "UPDATE uploads SET manifest_id=?3 WHERE tenant=?1 AND idempotency_key=?2 AND manifest_id IS NULL",
                    params![
                        self.tenant_namespace.as_slice(),
                        idempotency.as_slice(),
                        manifest_id.as_slice()
                    ],
                )?;
            }
        }
        if state != UploadState::Published {
            self.mark_receiving(idempotency)?;
        }
        Ok(AuthenticatedImportStatus {
            state: if state == UploadState::Reserved {
                UploadState::Receiving
            } else {
                state
            },
            manifest_id: *manifest_id,
            next_chunk_ordinal: self.import_next_chunk_ordinal(idempotency)?,
            publication_generation,
        })
    }

    pub fn authenticated_import_status(
        &self,
        idempotency: &Id32,
    ) -> Result<AuthenticatedImportStatus, StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<(String, Option<Vec<u8>>, u64, u64)> = connection
            .query_row(
                "SELECT state,manifest_id,next_chunk_ordinal,generation FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let (state, manifest_id, next_chunk_ordinal, publication_generation) =
            row.ok_or(StoreError::NotFound)?;
        Ok(AuthenticatedImportStatus {
            state: UploadState::parse(&state)?,
            manifest_id: vec_id(manifest_id.ok_or(StoreError::Authentication(
                "authenticated import is missing its manifest identity",
            ))?)?,
            next_chunk_ordinal,
            publication_generation,
        })
    }

    /// Seal an import after the authenticated manifest and every ordered chunk
    /// have arrived. Sealing is durable and observable; publication remains a
    /// separate catalog transaction.
    pub fn seal_authenticated_import(
        &self,
        idempotency: &Id32,
        context: &ValidationContext,
    ) -> Result<AuthenticatedImportStatus, StoreError> {
        let status = self.authenticated_import_status(idempotency)?;
        if status.state == UploadState::Published {
            return Ok(status);
        }
        if !matches!(status.state, UploadState::Receiving | UploadState::Verified) {
            return Err(StoreError::State("authenticated import cannot be sealed"));
        }
        let manifest = self.read_staged_manifest(idempotency, context)?;
        let expected_chunks = manifest.states.iter().try_fold(0u64, |sum, state| {
            sum.checked_add(state.chunks.len() as u64)
                .ok_or(StoreError::State("import chunk count overflow"))
        })?;
        if status.next_chunk_ordinal != expected_chunks {
            return Err(StoreError::State(
                "authenticated import has not received every ordered chunk",
            ));
        }

        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, expected_bytes, generation, stored_seal): (String, u64, u64, Vec<u8>) =
            transaction.query_row(
                "SELECT state,expected_bytes,generation,seal_digest FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;
        let rows = {
            let mut statement = transaction.prepare(
                "SELECT ordinal,object_key,object_digest FROM upload_chunks WHERE tenant=?1 AND idempotency_key=?2 ORDER BY ordinal",
            )?;
            let rows = statement
                .query_map(
                    params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                    |row| {
                        Ok((
                            row.get::<_, u64>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                        ))
                    },
                )?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        if rows.len() as u64 != expected_chunks
            || rows
                .iter()
                .enumerate()
                .any(|(ordinal, row)| row.0 != ordinal as u64)
        {
            return Err(StoreError::Authentication(
                "authenticated import chunk ledger is not contiguous",
            ));
        }
        let mut digest = IntentHasher::new(b"kvpack/catalog/authenticated-import-seal/v1");
        digest.id(&self.tenant_namespace);
        digest.id(idempotency);
        digest.id(&status.manifest_id);
        digest.u64(expected_bytes);
        digest.u64(generation);
        digest.u64(expected_chunks);
        for (ordinal, object_key, object_digest) in rows {
            digest.u64(ordinal);
            digest.id(&vec_id(object_key)?);
            digest.id(&vec_id(object_digest)?);
        }
        let seal = digest.finish();
        match state.as_str() {
            "VERIFIED" => {
                if stored_seal.as_slice() != seal {
                    return Err(StoreError::Authentication(
                        "authenticated import seal changed",
                    ));
                }
            }
            "RECEIVING" => {
                let changed = transaction.execute(
                    "UPDATE uploads SET state='VERIFIED',seal_digest=?3,updated_ns=?4 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING' AND next_chunk_ordinal=?5",
                    params![
                        self.tenant_namespace.as_slice(),
                        idempotency.as_slice(),
                        seal.as_slice(),
                        now_ns(),
                        expected_chunks,
                    ],
                )?;
                if changed != 1 {
                    return Err(StoreError::State(
                        "authenticated import seal transition raced",
                    ));
                }
            }
            _ => return Err(StoreError::State("authenticated import cannot be sealed")),
        }
        transaction.commit()?;
        drop(connection);
        self.authenticated_import_status(idempotency)
    }

    /// Authenticate and durably stage an exact `.kvpack` object. It remains
    /// undiscoverable until `commit_authenticated_import` publishes catalog
    /// and prefix rows in one transaction.
    pub fn stage_authenticated_manifest(
        &self,
        idempotency: &Id32,
        object: &[u8],
        context: &ValidationContext,
    ) -> Result<Id32, StoreError> {
        let status = self.authenticated_import_status(idempotency)?;
        if !matches!(
            status.state,
            UploadState::Receiving | UploadState::Verified | UploadState::Published
        ) {
            return Err(StoreError::State("authenticated import is not receiving"));
        }
        let header = inspect_pack_header(object)?;
        if header.manifest_id != status.manifest_id
            || header.tenant_namespace != self.tenant_namespace
            || header.key_epoch != self.key_epoch()
        {
            return Err(StoreError::Authentication(
                "imported manifest header is not bound to this store reservation",
            ));
        }
        let keys = self.schedule(header.key_epoch)?;
        let manifest = decode_authenticated_pack(object, &keys, context)?;
        if manifest.tenant_namespace != self.tenant_namespace {
            return Err(StoreError::Authentication(
                "imported manifest tenant namespace mismatch",
            ));
        }
        if status.state == UploadState::Published {
            let existing = self.read_authenticated_manifest_object(&status.manifest_id, context)?;
            if existing != object {
                return Err(StoreError::Authentication(
                    "published manifest retry changed object bytes",
                ));
            }
            return Ok(header.manifest_id);
        }
        let path = self.upload_manifest_path(idempotency);
        if path.exists() {
            let existing = fs::read(&path)
                .map_err(crate::error::io_error("read staged authenticated manifest"))?;
            if existing != object {
                self.quarantine_staged_import(
                    idempotency,
                    "authenticated manifest retry changed object bytes",
                )?;
                return Err(StoreError::Authentication(
                    "authenticated manifest retry changed object bytes",
                ));
            }
        } else {
            write_immutable(
                self,
                DurableObjectKind::UploadManifest,
                "partials",
                &path,
                object,
            )?;
        }
        Ok(header.manifest_id)
    }

    /// Verify a stored-object frame against its exact manifest ordinal before
    /// making the unreferenced CAS object available for atomic publication.
    pub fn put_authenticated_import_chunk(
        self: &std::sync::Arc<Self>,
        idempotency: &Id32,
        expected_ordinal: u64,
        object_key: &Id32,
        object: &[u8],
        context: &ValidationContext,
    ) -> Result<(), StoreError> {
        let status = self.authenticated_import_status(idempotency)?;
        if !matches!(
            status.state,
            UploadState::Receiving | UploadState::Verified | UploadState::Published
        ) {
            return Err(StoreError::State("authenticated import is not receiving"));
        }
        if expected_ordinal > status.next_chunk_ordinal {
            return Err(StoreError::Expectation(
                "chunk upload skipped the authenticated import cursor",
            ));
        }
        let manifest = if status.state == UploadState::Published {
            let pack = self.read_authenticated_manifest_object(&status.manifest_id, context)?;
            let header = inspect_pack_header(&pack)?;
            let keys = self.schedule(header.key_epoch)?;
            decode_authenticated_pack(&pack, &keys, context)?
        } else {
            self.read_staged_manifest(idempotency, context)?
        };
        let entry = manifest
            .states
            .iter()
            .zip(&manifest.realized_schema.states)
            .flat_map(|(state, schema)| {
                state
                    .chunks
                    .iter()
                    .zip(&schema.chunk_spans)
                    .map(move |(reference, span)| (&state.key, reference, span))
            })
            .enumerate()
            .find(|(ordinal, (_, reference, _))| {
                *ordinal as u64 == expected_ordinal && reference.object_key == *object_key
            })
            .map(|(_, entry)| entry)
            .ok_or(StoreError::Expectation(
                "chunk upload is not an exact staged-manifest reference",
            ))?;
        let (state_key, reference, span) = entry;
        let keys = self.schedule(reference.key_epoch)?;
        decode_chunk(
            object,
            reference,
            span,
            &manifest.tenant_namespace,
            &manifest.family,
            state_key,
            &keys,
        )?;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let cursor: u64 = transaction.query_row(
            "SELECT next_chunk_ordinal FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| row.get(0),
        )?;
        if expected_ordinal < cursor {
            let prior: Option<(Vec<u8>, Vec<u8>)> = transaction
                .query_row(
                    "SELECT object_key,object_digest FROM upload_chunks WHERE tenant=?1 AND idempotency_key=?2 AND ordinal=?3",
                    params![
                        self.tenant_namespace.as_slice(),
                        idempotency.as_slice(),
                        expected_ordinal
                    ],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if prior.as_ref().is_none_or(|(key, digest)| {
                key.as_slice() != reference.object_key
                    || digest.as_slice() != reference.object_digest
            }) {
                return Err(StoreError::Authentication(
                    "chunk retry disagrees with the durable import cursor",
                ));
            }
            transaction.commit()?;
            return Ok(());
        }
        if status.state != UploadState::Receiving {
            return Err(StoreError::State(
                "sealed import cannot accept another chunk ordinal",
            ));
        }
        if expected_ordinal != cursor {
            return Err(StoreError::Expectation(
                "chunk upload raced ahead of the authenticated import cursor",
            ));
        }
        drop(transaction);
        drop(connection);
        // The immutable chunk write, the chunk catalog rows, the import
        // ledger row, and the cursor advance commit in ONE catalog
        // transaction instead of two per imported chunk.
        let chunk_object = ChunkObject {
            chunk_id: reference.chunk_id,
            object_key: reference.object_key,
            object_digest: reference.object_digest,
            plaintext_bytes: reference.plaintext_bytes,
            bytes: object.to_vec(),
        };
        let stored = self.put_chunk_with_retention_and_ledger(
            &chunk_object,
            reference.key_epoch,
            super::RetentionInputs::conservative(
                object.len() as u64,
                reference.plaintext_bytes as u64,
            ),
            idempotency,
            expected_ordinal,
        )?;
        if stored.object_key != reference.object_key
            || stored.object_digest != reference.object_digest
            || stored.chunk_id != reference.chunk_id
        {
            return Err(StoreError::Authentication(
                "imported chunk catalog identity changed",
            ));
        }
        Ok(())
    }

    pub fn commit_authenticated_import(
        &self,
        idempotency: &Id32,
        context: &ValidationContext,
    ) -> Result<PublishedArtifact, StoreError> {
        let status = self.seal_authenticated_import(idempotency, context)?;
        if status.state == UploadState::Published {
            return self
                .published_upload(idempotency)?
                .ok_or(StoreError::Authentication(
                    "published import has no manifest catalog row",
                ));
        }
        let path = self.upload_manifest_path(idempotency);
        let bytes = fs::read(&path)
            .map_err(crate::error::io_error("read staged authenticated manifest"))?;
        let header = inspect_pack_header(&bytes)?;
        if header.manifest_id != status.manifest_id {
            return Err(StoreError::Authentication(
                "staged manifest identity changed before commit",
            ));
        }
        let keys = self.schedule(header.key_epoch)?;
        let manifest = decode_authenticated_pack(&bytes, &keys, context)?;
        let expected_chunks = manifest.states.iter().try_fold(0u64, |sum, state| {
            sum.checked_add(state.chunks.len() as u64)
                .ok_or(StoreError::State("import chunk count overflow"))
        })?;
        if status.next_chunk_ordinal != expected_chunks {
            return Err(StoreError::State(
                "authenticated import has not received every ordered chunk",
            ));
        }
        let encoded = EncodedPack {
            bytes,
            manifest_id: status.manifest_id,
        };
        let exact_node = kvpack_core::PrefixNode {
            token_count: manifest.input_cut.token_count,
            id: manifest.input_cut.token_root,
            reusable: manifest.input_cut.token_count % kvpack_core::PREFIX_BLOCK_TOKENS as u64 == 0,
        };
        let published = self.publish_manifest(
            idempotency,
            &encoded,
            &manifest,
            std::slice::from_ref(&exact_node),
        )?;
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "remove committed staged manifest",
                    source,
                });
            }
        }
        super::fsync_dir(&self.config.object_root.join("uploads"))?;
        Ok(published)
    }

    pub fn cancel_authenticated_import(&self, idempotency: &Id32) -> Result<(), StoreError> {
        let path = self.upload_manifest_path(idempotency);
        match fs::remove_file(&path) {
            Ok(()) => super::fsync_dir(&self.config.object_root.join("uploads"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "remove cancelled staged manifest",
                    source,
                });
            }
        }
        self.abort_upload(idempotency)
    }

    pub fn quarantine_authenticated_import(&self, idempotency: &Id32) -> Result<(), StoreError> {
        self.quarantine_staged_import(idempotency, "authenticated import quarantined by operator")
    }

    pub fn delete_prefix(&self, prefix_id: &Id32, catalog_epoch: u64) -> Result<u64, StoreError> {
        if prefix_id.iter().all(|byte| *byte == 0) || catalog_epoch != self.catalog_epoch() {
            return Err(StoreError::Authentication(
                "prefix deletion identity or catalog epoch is invalid",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Tombstoned,
            AuditObjectKind::Prefix,
            *prefix_id,
            catalog_epoch,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        let tombstoned = transaction.execute(
            "INSERT OR IGNORE INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'prefix',?2,?3,?4)",
            params![
                self.tenant_namespace.as_slice(),
                prefix_id.as_slice(),
                catalog_epoch,
                now_ns()
            ],
        )?;
        let deleted = transaction.execute(
            "DELETE FROM prefix_checkpoints WHERE tenant=?1 AND prefix_id=?2",
            params![self.tenant_namespace.as_slice(), prefix_id.as_slice()],
        )? as u64;
        let audit_enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            now_ns(),
        )?;
        transaction.commit()?;
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Tombstoned, tombstoned as u64);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        Ok(deleted)
    }

    fn upload_manifest_path(&self, idempotency: &Id32) -> std::path::PathBuf {
        self.config
            .object_root
            .join("uploads")
            .join(format!("{}.kvpack.partial", hex(idempotency)))
    }

    fn quarantine_staged_import(
        &self,
        idempotency: &Id32,
        reason: &'static str,
    ) -> Result<(), StoreError> {
        let source = self.upload_manifest_path(idempotency);
        let file = if source.exists() {
            let metadata = source.metadata().map_err(crate::error::io_error(
                "inspect quarantined upload manifest",
            ))?;
            if !metadata.is_file() {
                return Err(StoreError::State(
                    "staged upload manifest is not a regular file",
                ));
            }
            let mut entry_id = [0u8; 32];
            getrandom::fill(&mut entry_id)
                .map_err(|_| StoreError::State("quarantine identity entropy failed"))?;
            let path_token = format!("{}.upload.quarantine", hex(&entry_id));
            let destination = self.config.object_root.join("quarantine").join(&path_token);
            fs::hard_link(&source, &destination).map_err(crate::error::io_error(
                "link upload manifest into quarantine",
            ))?;
            fs::remove_file(&source)
                .map_err(crate::error::io_error("unlink quarantined upload manifest"))?;
            fsync_dir(source.parent().unwrap())?;
            fsync_dir(destination.parent().unwrap())?;
            Some(QuarantinedUploadFile {
                entry_id,
                path_token,
                file_bytes: metadata.len(),
            })
        } else {
            None
        };
        self.quarantine_upload(idempotency, reason, file.as_ref())
    }

    fn import_next_chunk_ordinal(&self, idempotency: &Id32) -> Result<u64, StoreError> {
        let connection = self.lock_catalog()?;
        Ok(connection.query_row(
            "SELECT next_chunk_ordinal FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| row.get(0),
        )?)
    }

    fn read_staged_manifest(
        &self,
        idempotency: &Id32,
        context: &ValidationContext,
    ) -> Result<kvpack_core::CutManifest, StoreError> {
        let bytes = fs::read(self.upload_manifest_path(idempotency))
            .map_err(crate::error::io_error("read staged authenticated manifest"))?;
        let header = inspect_pack_header(&bytes)?;
        let keys = self.schedule(header.key_epoch)?;
        let manifest = decode_authenticated_pack(&bytes, &keys, context)?;
        Ok(manifest)
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
