use kvpack_core::{decode_authenticated_pack, inspect_pack_header, Id32, ValidationContext};
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use crate::StoreError;

use super::{vec_id, LocalStore};

pub(crate) const MAX_SOURCE_LEASE_NS: u64 = 5 * 60 * 1_000_000_000;
const MAX_INVENTORY_PAGE: usize = 4_096;

type RemoteMutationRow = (Vec<u8>, String, Vec<u8>, Vec<u8>);
type SourceLeaseRow = (
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    u64,
    u64,
    String,
    u64,
    u64,
    u64,
    u64,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteImportFence {
    pub manifest_id: Id32,
    pub owner_id: Id32,
    pub attempt_epoch: u64,
    pub publication_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteMutation {
    Begin,
    ManifestPut,
    ChunkPut,
    Publish,
    Cancel,
    DeletePrefix,
    ReleaseLease,
}

impl RemoteMutation {
    fn name(self) -> &'static str {
        match self {
            Self::Begin => "BEGIN",
            Self::ManifestPut => "MANIFEST_PUT",
            Self::ChunkPut => "CHUNK_PUT",
            Self::Publish => "PUBLISH",
            Self::Cancel => "CANCEL",
            Self::DeletePrefix => "DELETE_PREFIX",
            Self::ReleaseLease => "RELEASE_LEASE",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationReplay {
    Fresh,
    ExactRetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLeaseState {
    Active,
    Uncertain,
    Released,
}

impl SourceLeaseState {
    fn parse(value: &str) -> Result<Self, StoreError> {
        match value {
            "ACTIVE" => Ok(Self::Active),
            "UNCERTAIN" => Ok(Self::Uncertain),
            "RELEASED" => Ok(Self::Released),
            _ => Err(StoreError::Authentication(
                "catalog contains an unknown source-lease state",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLeaseStatus {
    pub lease_id: Id32,
    pub manifest_id: Id32,
    pub owner_id: Id32,
    pub owner_incarnation: Id32,
    pub authority_term: u64,
    pub publication_generation: u64,
    pub state: SourceLeaseState,
    pub granted_ns: u64,
    pub maximum_duration_ns: u64,
    pub expires_ns: u64,
    pub object_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InventoryObjectKind {
    Manifest,
    Chunk,
}

impl InventoryObjectKind {
    fn ordinal(self) -> i64 {
        match self {
            Self::Manifest => 0,
            Self::Chunk => 1,
        }
    }

    fn parse(value: i64) -> Result<Self, StoreError> {
        match value {
            0 => Ok(Self::Manifest),
            1 => Ok(Self::Chunk),
            _ => Err(StoreError::Authentication(
                "catalog inventory contains an unknown object kind",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InventoryCursor {
    pub kind: InventoryObjectKind,
    pub object_id: Id32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventoryEntry {
    pub kind: InventoryObjectKind,
    pub object_id: Id32,
    pub object_digest: Id32,
    pub object_bytes: u64,
    pub publication_generation: u64,
    pub key_epoch: u64,
}

impl LocalStore {
    /// Permanently bind a remote upload idempotency key to one owner, attempt,
    /// manifest, and publication generation before transfer bytes are accepted.
    pub fn bind_remote_import(
        &self,
        idempotency_key: &Id32,
        fence: RemoteImportFence,
    ) -> Result<RemoteImportFence, StoreError> {
        require_id(idempotency_key, "remote import idempotency key is zero")?;
        require_id(
            &fence.manifest_id,
            "remote import manifest identity is zero",
        )?;
        require_id(&fence.owner_id, "remote import owner identity is zero")?;
        if fence.attempt_epoch == 0 || fence.publication_generation == 0 {
            return Err(StoreError::Expectation(
                "remote import attempt or publication generation is zero",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(Vec<u8>, Vec<u8>, u64, u64)> = transaction
            .query_row(
                "SELECT manifest_id,owner_id,attempt_epoch,generation FROM remote_upload_fences WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency_key.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        if let Some((manifest, owner, attempt, generation)) = existing {
            let existing = RemoteImportFence {
                manifest_id: vec_id(manifest)?,
                owner_id: vec_id(owner)?,
                attempt_epoch: attempt,
                publication_generation: generation,
            };
            if existing != fence {
                return Err(StoreError::Authentication(
                    "remote import idempotency key was rebound",
                ));
            }
            transaction.commit()?;
            return Ok(existing);
        }
        transaction.execute(
            "INSERT INTO remote_upload_fences(tenant,idempotency_key,manifest_id,owner_id,attempt_epoch,generation,created_ns) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                self.tenant_namespace.as_slice(),
                idempotency_key.as_slice(),
                fence.manifest_id.as_slice(),
                fence.owner_id.as_slice(),
                fence.attempt_epoch,
                fence.publication_generation,
                now_ns(),
            ],
        )?;
        transaction.commit()?;
        Ok(fence)
    }

    pub fn require_remote_import(
        &self,
        idempotency_key: &Id32,
        owner_id: &Id32,
        attempt_epoch: u64,
    ) -> Result<RemoteImportFence, StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<(Vec<u8>, Vec<u8>, u64, u64)> = connection
            .query_row(
                "SELECT manifest_id,owner_id,attempt_epoch,generation FROM remote_upload_fences WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency_key.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let (manifest, stored_owner, stored_attempt, generation) =
            row.ok_or(StoreError::NotFound)?;
        let fence = RemoteImportFence {
            manifest_id: vec_id(manifest)?,
            owner_id: vec_id(stored_owner)?,
            attempt_epoch: stored_attempt,
            publication_generation: generation,
        };
        if &fence.owner_id != owner_id || fence.attempt_epoch != attempt_epoch {
            return Err(StoreError::Authentication(
                "remote import owner or attempt fence mismatch",
            ));
        }
        Ok(fence)
    }

    /// Consume a derived mutation nonce durably. An exact retry is accepted;
    /// any changed request, operation, scope, or intent fails closed.
    pub fn consume_remote_mutation(
        &self,
        nonce: &Id32,
        request_id: &Id32,
        operation: RemoteMutation,
        scope_id: &Id32,
        intent_digest: &Id32,
    ) -> Result<MutationReplay, StoreError> {
        for (identity, message) in [
            (nonce, "remote mutation nonce is zero"),
            (request_id, "remote mutation request identity is zero"),
            (scope_id, "remote mutation scope identity is zero"),
            (intent_digest, "remote mutation intent digest is zero"),
        ] {
            require_id(identity, message)?;
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<RemoteMutationRow> = transaction
            .query_row(
                "SELECT request_id,operation,scope_id,intent_digest FROM remote_mutations WHERE tenant=?1 AND nonce=?2",
                params![self.tenant_namespace.as_slice(), nonce.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        if let Some((stored_request, stored_operation, stored_scope, stored_intent)) = existing {
            if stored_request.as_slice() != request_id
                || stored_operation != operation.name()
                || stored_scope.as_slice() != scope_id
                || stored_intent.as_slice() != intent_digest
            {
                return Err(StoreError::Authentication(
                    "remote mutation nonce was replayed with changed intent",
                ));
            }
            transaction.execute(
                "UPDATE remote_mutations SET last_seen_ns=?3 WHERE tenant=?1 AND nonce=?2",
                params![self.tenant_namespace.as_slice(), nonce.as_slice(), now_ns()],
            )?;
            transaction.commit()?;
            return Ok(MutationReplay::ExactRetry);
        }
        let timestamp = now_ns();
        transaction.execute(
            "INSERT INTO remote_mutations(tenant,nonce,request_id,operation,scope_id,intent_digest,first_seen_ns,last_seen_ns) VALUES(?1,?2,?3,?4,?5,?6,?7,?7)",
            params![
                self.tenant_namespace.as_slice(),
                nonce.as_slice(),
                request_id.as_slice(),
                operation.name(),
                scope_id.as_slice(),
                intent_digest.as_slice(),
                timestamp,
            ],
        )?;
        transaction.commit()?;
        Ok(MutationReplay::Fresh)
    }

    /// Acquire one source lease over an authenticated manifest and every
    /// immutable chunk it references. The catalog transaction rejects
    /// tombstoned or generation-mismatched sources.
    #[allow(clippy::too_many_arguments)]
    pub fn acquire_source_lease(
        &self,
        lease_id: &Id32,
        manifest_id: &Id32,
        owner_id: &Id32,
        owner_incarnation: &Id32,
        authority_term: u64,
        maximum_duration_ns: u64,
        context: &ValidationContext,
    ) -> Result<SourceLeaseStatus, StoreError> {
        for (identity, message) in [
            (lease_id, "source lease identity is zero"),
            (manifest_id, "source lease manifest identity is zero"),
            (owner_id, "source lease owner identity is zero"),
            (owner_incarnation, "source lease owner incarnation is zero"),
        ] {
            require_id(identity, message)?;
        }
        if authority_term != self.catalog_epoch()
            || maximum_duration_ns == 0
            || maximum_duration_ns > MAX_SOURCE_LEASE_NS
        {
            return Err(StoreError::Authentication(
                "source lease term or duration is invalid",
            ));
        }

        let pack = self.read_authenticated_manifest_object(manifest_id, context)?;
        let header = inspect_pack_header(&pack)?;
        let keys = self.schedule(header.key_epoch)?;
        let manifest = decode_authenticated_pack(&pack, &keys, context)?;
        let mut objects = manifest
            .states
            .iter()
            .flat_map(|state| state.chunks.iter().map(|chunk| chunk.object_key))
            .collect::<Vec<_>>();
        objects.sort_unstable();
        objects.dedup();

        let granted_ns = now_ns();
        let expires_ns = granted_ns
            .checked_add(maximum_duration_ns)
            .ok_or(StoreError::Expectation("source lease expiry overflows"))?;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(existing) = load_source_lease(&transaction, &self.tenant_namespace, lease_id)? {
            require_exact_source_lease(
                &existing,
                manifest_id,
                owner_id,
                owner_incarnation,
                authority_term,
                maximum_duration_ns,
            )?;
            transaction.commit()?;
            return Ok(existing);
        }
        let generation: Option<u64> = transaction
            .query_row(
                "SELECT generation FROM manifests m WHERE tenant=?1 AND manifest_id=?2 AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id)",
                params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
                |row| row.get(0),
            )
            .optional()?;
        let generation = generation.ok_or(StoreError::NotFound)?;
        transaction.execute(
            "INSERT INTO source_leases(tenant,lease_id,manifest_id,owner_id,owner_incarnation,authority_term,generation,state,granted_ns,maximum_duration_ns,expires_ns) VALUES(?1,?2,?3,?4,?5,?6,?7,'ACTIVE',?8,?9,?10)",
            params![
                self.tenant_namespace.as_slice(),
                lease_id.as_slice(),
                manifest_id.as_slice(),
                owner_id.as_slice(),
                owner_incarnation.as_slice(),
                authority_term,
                generation,
                granted_ns,
                maximum_duration_ns,
                expires_ns,
            ],
        )?;
        transaction.execute(
            "INSERT INTO source_lease_objects(tenant,lease_id,object_kind,object_id,ordinal) VALUES(?1,?2,'manifest',?3,0)",
            params![self.tenant_namespace.as_slice(), lease_id.as_slice(), manifest_id.as_slice()],
        )?;
        for (ordinal, object_id) in objects.iter().enumerate() {
            let available: Option<u8> = transaction
                .query_row(
                    "SELECT 1 FROM chunks c WHERE tenant=?1 AND object_key=?2 AND location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)",
                    params![self.tenant_namespace.as_slice(), object_id.as_slice()],
                    |row| row.get(0),
                )
                .optional()?;
            if available.is_none() {
                return Err(StoreError::NotFound);
            }
            transaction.execute(
                "INSERT INTO source_lease_objects(tenant,lease_id,object_kind,object_id,ordinal) VALUES(?1,?2,'chunk',?3,?4)",
                params![
                    self.tenant_namespace.as_slice(),
                    lease_id.as_slice(),
                    object_id.as_slice(),
                    ordinal as u64 + 1,
                ],
            )?;
        }
        transaction.commit()?;
        drop(connection);
        self.source_lease_status(lease_id)
    }

    pub fn source_lease_status(&self, lease_id: &Id32) -> Result<SourceLeaseStatus, StoreError> {
        let connection = self.lock_catalog()?;
        load_source_lease(&connection, &self.tenant_namespace, lease_id)?
            .ok_or(StoreError::NotFound)
    }

    pub fn reattach_source_lease(
        &self,
        lease_id: &Id32,
        owner_id: &Id32,
        owner_incarnation: &Id32,
        authority_term: u64,
        maximum_duration_ns: u64,
    ) -> Result<SourceLeaseStatus, StoreError> {
        if maximum_duration_ns == 0 || maximum_duration_ns > MAX_SOURCE_LEASE_NS {
            return Err(StoreError::Authentication(
                "source lease reattach duration is invalid",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = load_source_lease(&transaction, &self.tenant_namespace, lease_id)?
            .ok_or(StoreError::NotFound)?;
        if &current.owner_id != owner_id
            || &current.owner_incarnation != owner_incarnation
            || current.authority_term != authority_term
            || current.state == SourceLeaseState::Released
        {
            return Err(StoreError::Authentication(
                "source lease reattach fence mismatch",
            ));
        }
        let granted_ns = now_ns();
        let expires_ns = granted_ns
            .checked_add(maximum_duration_ns)
            .ok_or(StoreError::Expectation("source lease expiry overflows"))?;
        transaction.execute(
            "UPDATE source_leases SET state='ACTIVE',granted_ns=?3,maximum_duration_ns=?4,expires_ns=?5 WHERE tenant=?1 AND lease_id=?2",
            params![
                self.tenant_namespace.as_slice(),
                lease_id.as_slice(),
                granted_ns,
                maximum_duration_ns,
                expires_ns,
            ],
        )?;
        transaction.commit()?;
        drop(connection);
        self.source_lease_status(lease_id)
    }

    pub fn require_source_lease_object(
        &self,
        lease_id: &Id32,
        owner_id: &Id32,
        object_kind: InventoryObjectKind,
        object_id: &Id32,
    ) -> Result<(), StoreError> {
        let connection = self.lock_catalog()?;
        let present: Option<u8> = connection
            .query_row(
                "SELECT 1 FROM source_leases l JOIN source_lease_objects o ON o.tenant=l.tenant AND o.lease_id=l.lease_id WHERE l.tenant=?1 AND l.lease_id=?2 AND l.owner_id=?3 AND l.state='ACTIVE' AND l.expires_ns>=?4 AND o.object_kind=?5 AND o.object_id=?6",
                params![
                    self.tenant_namespace.as_slice(),
                    lease_id.as_slice(),
                    owner_id.as_slice(),
                    now_ns(),
                    match object_kind {
                        InventoryObjectKind::Manifest => "manifest",
                        InventoryObjectKind::Chunk => "chunk",
                    },
                    object_id.as_slice(),
                ],
                |row| row.get(0),
            )
            .optional()?;
        present.map(|_| ()).ok_or(StoreError::Authentication(
            "source lease does not authorize this object",
        ))
    }

    pub fn release_source_lease(
        &self,
        lease_id: &Id32,
        owner_id: &Id32,
        authority_term: u64,
    ) -> Result<SourceLeaseState, StoreError> {
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = load_source_lease(&transaction, &self.tenant_namespace, lease_id)?
            .ok_or(StoreError::NotFound)?;
        if &current.owner_id != owner_id || current.authority_term != authority_term {
            return Err(StoreError::Authentication(
                "source lease release fence mismatch",
            ));
        }
        if current.state != SourceLeaseState::Released {
            transaction.execute(
                "UPDATE source_leases SET state='RELEASED' WHERE tenant=?1 AND lease_id=?2 AND state IN ('ACTIVE','UNCERTAIN')",
                params![self.tenant_namespace.as_slice(), lease_id.as_slice()],
            )?;
        }
        transaction.commit()?;
        Ok(SourceLeaseState::Released)
    }

    pub fn release_uncertain_source_lease(
        &self,
        lease_id: &Id32,
        authority_term: u64,
        operator_confirmed: bool,
    ) -> Result<SourceLeaseState, StoreError> {
        if !operator_confirmed {
            return Err(StoreError::Authentication(
                "uncertain source lease release requires operator confirmation",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = load_source_lease(&transaction, &self.tenant_namespace, lease_id)?
            .ok_or(StoreError::NotFound)?;
        if current.authority_term != authority_term || current.state != SourceLeaseState::Uncertain
        {
            return Err(StoreError::Authentication(
                "uncertain source lease release fence mismatch",
            ));
        }
        transaction.execute(
            "UPDATE source_leases SET state='RELEASED' WHERE tenant=?1 AND lease_id=?2 AND state='UNCERTAIN'",
            params![self.tenant_namespace.as_slice(), lease_id.as_slice()],
        )?;
        transaction.commit()?;
        Ok(SourceLeaseState::Released)
    }

    pub fn inventory_page(
        &self,
        after: Option<InventoryCursor>,
        limit: usize,
    ) -> Result<Vec<InventoryEntry>, StoreError> {
        if limit == 0 || limit > MAX_INVENTORY_PAGE {
            return Err(StoreError::Expectation("inventory page bound is invalid"));
        }
        let cursor_kind = after.map_or(-1, |cursor| cursor.kind.ordinal());
        let cursor_id = after.map_or([0; 32], |cursor| cursor.object_id);
        let connection = self.lock_catalog()?;
        let mut statement = connection.prepare(
            "SELECT kind,object_id,object_digest,object_bytes,generation,key_epoch FROM (
               SELECT 0 AS kind,m.manifest_id AS object_id,m.manifest_id AS object_digest,m.file_bytes AS object_bytes,m.generation,m.key_epoch
               FROM manifests m WHERE m.tenant=?1 AND m.key_epoch BETWEEN ?5 AND ?6 AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id)
               UNION ALL
               SELECT 1 AS kind,c.object_key AS object_id,c.object_digest,c.object_bytes,0 AS generation,c.key_epoch
               FROM chunks c WHERE c.tenant=?1 AND c.key_epoch BETWEEN ?5 AND ?6 AND c.location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)
             ) WHERE kind>?2 OR (kind=?2 AND object_id>?3) ORDER BY kind,object_id LIMIT ?4",
        )?;
        let rows = statement.query_map(
            params![
                self.tenant_namespace.as_slice(),
                cursor_kind,
                cursor_id.as_slice(),
                limit as u64,
                self.minimum_readable_key_epoch(),
                self.key_epoch(),
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, u64>(4)?,
                    row.get::<_, u64>(5)?,
                ))
            },
        )?;
        rows.map(|row| {
            let (kind, object_id, object_digest, object_bytes, generation, key_epoch) = row?;
            Ok(InventoryEntry {
                kind: InventoryObjectKind::parse(kind)?,
                object_id: vec_id(object_id)?,
                object_digest: vec_id(object_digest)?,
                object_bytes,
                publication_generation: generation,
                key_epoch,
            })
        })
        .collect()
    }

    pub(super) fn mark_active_source_leases_uncertain(&self) -> Result<u64, StoreError> {
        let connection = self.lock_catalog()?;
        Ok(connection.execute(
            "UPDATE source_leases SET state='UNCERTAIN' WHERE tenant=?1 AND state='ACTIVE'",
            params![self.tenant_namespace.as_slice()],
        )? as u64)
    }
}

fn load_source_lease(
    connection: &rusqlite::Connection,
    tenant: &Id32,
    lease_id: &Id32,
) -> Result<Option<SourceLeaseStatus>, StoreError> {
    let row: Option<SourceLeaseRow> = connection
            .query_row(
                "SELECT manifest_id,owner_id,owner_incarnation,authority_term,generation,state,granted_ns,maximum_duration_ns,expires_ns,(SELECT COUNT(*) FROM source_lease_objects o WHERE o.tenant=source_leases.tenant AND o.lease_id=source_leases.lease_id) FROM source_leases WHERE tenant=?1 AND lease_id=?2",
                params![tenant.as_slice(), lease_id.as_slice()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                        row.get(9)?,
                    ))
                },
            )
            .optional()?;
    row.map(
        |(
            manifest,
            owner,
            incarnation,
            authority_term,
            generation,
            state,
            granted_ns,
            maximum_duration_ns,
            expires_ns,
            object_count,
        )| {
            Ok(SourceLeaseStatus {
                lease_id: *lease_id,
                manifest_id: vec_id(manifest)?,
                owner_id: vec_id(owner)?,
                owner_incarnation: vec_id(incarnation)?,
                authority_term,
                publication_generation: generation,
                state: SourceLeaseState::parse(&state)?,
                granted_ns,
                maximum_duration_ns,
                expires_ns,
                object_count,
            })
        },
    )
    .transpose()
}

fn require_exact_source_lease(
    existing: &SourceLeaseStatus,
    manifest_id: &Id32,
    owner_id: &Id32,
    owner_incarnation: &Id32,
    authority_term: u64,
    maximum_duration_ns: u64,
) -> Result<(), StoreError> {
    if &existing.manifest_id != manifest_id
        || &existing.owner_id != owner_id
        || &existing.owner_incarnation != owner_incarnation
        || existing.authority_term != authority_term
        || existing.maximum_duration_ns != maximum_duration_ns
    {
        return Err(StoreError::Authentication(
            "source lease identity was reused with changed bounds",
        ));
    }
    Ok(())
}

fn require_id(identity: &Id32, message: &'static str) -> Result<(), StoreError> {
    if identity.iter().all(|byte| *byte == 0) {
        return Err(StoreError::Expectation(message));
    }
    Ok(())
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}
