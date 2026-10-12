use super::*;

impl LocalStore {
    /// Resolve the staging disposition for one batch (typically one state) in
    /// a single catalog read, replacing the per-chunk session/cursor reads.
    pub(crate) fn begin_provisional_stage_batch(
        &self,
        idempotency: &Id32,
        token: u64,
    ) -> Result<ProvisionalStageMode, StoreError> {
        let (state, cursor, session_token, lease_expires_ns): (String, u64, i64, i64) = {
            let connection = self.lock_catalog()?;
            connection.query_row(
                "SELECT state,next_chunk_ordinal,session_token,lease_expires_ns FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?
        };
        if state == "PUBLISHED" {
            return Ok(ProvisionalStageMode::PublishedReplay);
        }
        if session_token as u64 != token {
            return Err(StoreError::State(
                "provisional upload session token is stale",
            ));
        }
        reject_expired_lease(lease_expires_ns)?;
        if state != "RECEIVING" {
            return Err(StoreError::State(
                "provisional upload is not accepting chunks",
            ));
        }
        Ok(ProvisionalStageMode::Receiving { cursor })
    }

    /// Stage one chunk's bytes (or reference an already-durable copy) without
    /// any catalog write.  The batch's ledger rows and the durable cursor
    /// advance commit in `commit_provisional_stage_batch`.
    pub(crate) fn stage_provisional_chunk_file(
        &self,
        idempotency: &Id32,
        mode: &ProvisionalStageMode,
        ordinal: u64,
        object: &ChunkObject,
    ) -> Result<ProvisionalStagedChunk, StoreError> {
        match mode {
            ProvisionalStageMode::PublishedReplay => {
                let reference = self.find_chunk(&object.chunk_id, self.key_epoch())?.ok_or(
                    StoreError::Authentication(
                        "provisional replay references an unavailable published chunk",
                    ),
                )?;
                validate_content_reference(&reference, object, self.key_epoch())?;
                Ok(ProvisionalStagedChunk {
                    reference,
                    staged_path: None,
                    staged_bytes: 0,
                    deduplicated_bytes: object.bytes.len() as u64,
                })
            }
            ProvisionalStageMode::Receiving { cursor } if ordinal < *cursor => {
                let reference = self.find_chunk(&object.chunk_id, self.key_epoch())?.ok_or(
                    StoreError::Authentication(
                        "provisional replay references an unavailable published chunk",
                    ),
                )?;
                validate_content_reference(&reference, object, self.key_epoch())?;
                self.verify_provisional_ledger_entry(idempotency, ordinal, &reference)?;
                Ok(ProvisionalStagedChunk {
                    reference,
                    staged_path: None,
                    staged_bytes: 0,
                    deduplicated_bytes: object.bytes.len() as u64,
                })
            }
            ProvisionalStageMode::Receiving { .. } => {
                if let Some(reference) = self.find_chunk(&object.chunk_id, self.key_epoch())? {
                    // Encrypted encodings carry fresh salt/nonce bytes. A
                    // cataloged object with the same authenticated content ID
                    // is the canonical dedup winner even when this attempt
                    // encoded different, equally valid ciphertext bytes.
                    validate_content_reference(&reference, object, self.key_epoch())?;
                    Ok(ProvisionalStagedChunk {
                        reference,
                        staged_path: None,
                        staged_bytes: 0,
                        deduplicated_bytes: object.bytes.len() as u64,
                    })
                } else {
                    let path = self.provisional_chunk_path(idempotency, ordinal);
                    write_or_verify_private(&path, &object.bytes)?;
                    Ok(ProvisionalStagedChunk {
                        reference: ChunkRef {
                            chunk_id: object.chunk_id,
                            object_key: object.object_key,
                            object_digest: object.object_digest,
                            key_epoch: self.key_epoch(),
                            plaintext_bytes: object.plaintext_bytes,
                            object_bytes: u32::try_from(object.bytes.len()).map_err(|_| {
                                StoreError::State("provisional chunk object exceeds u32")
                            })?,
                        },
                        staged_path: Some(path),
                        staged_bytes: object.bytes.len() as u64,
                        deduplicated_bytes: 0,
                    })
                }
            }
        }
    }

    /// Commit one staging batch's chunk-ledger rows and advance the durable
    /// upload cursor in ONE catalog transaction, replacing one transaction
    /// per staged chunk.  `staged` pairs are (ordinal, reference) for every
    /// chunk at or above the durable cursor, in ordinal order.
    pub(crate) fn commit_provisional_stage_batch(
        &self,
        idempotency: &Id32,
        token: u64,
        staged: &[(u64, ChunkRef)],
    ) -> Result<(), StoreError> {
        if staged.is_empty() {
            return Ok(());
        }
        let first = staged[0].0;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current: (String, u64, i64, i64) = transaction.query_row(
            "SELECT state,next_chunk_ordinal,session_token,lease_expires_ns FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        if current.2 as u64 != token {
            return Err(StoreError::State(
                "provisional upload session token is stale",
            ));
        }
        reject_expired_lease(current.3)?;
        if (current.0.as_str(), current.1) != ("RECEIVING", first) {
            return Err(StoreError::State(
                "provisional upload cursor raced while staging a chunk",
            ));
        }
        let now = now_ns();
        for (index, (ordinal, reference)) in staged.iter().enumerate() {
            if *ordinal != first + index as u64 {
                return Err(StoreError::State(
                    "provisional staging batch is not contiguous",
                ));
            }
            transaction.execute(
                "INSERT INTO upload_chunks(tenant,idempotency_key,ordinal,object_key,object_digest,verified_ns) VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    self.tenant_namespace.as_slice(),
                    idempotency.as_slice(),
                    ordinal,
                    reference.object_key.as_slice(),
                    reference.object_digest.as_slice(),
                    now,
                ],
            )?;
        }
        let changed = transaction.execute(
            "UPDATE uploads SET next_chunk_ordinal=next_chunk_ordinal+?4,lease_expires_ns=?5,updated_ns=?6 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING' AND next_chunk_ordinal=?3 AND session_token=?7",
            params![
                self.tenant_namespace.as_slice(),
                idempotency.as_slice(),
                first,
                staged.len() as u64,
                now.saturating_add(PROVISIONAL_UPLOAD_LEASE_NS) as i64,
                now as i64,
                token as i64,
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::State(
                "provisional upload cursor failed to advance",
            ));
        }
        transaction.commit()?;
        Ok(())
    }

    pub(crate) fn sync_provisional_state(&self, idempotency: &Id32) -> Result<(), StoreError> {
        fsync_dir(&self.provisional_upload_path(idempotency))
    }

    pub(crate) fn verify_provisional_ledger(
        &self,
        idempotency: &Id32,
        references: &[ChunkRef],
    ) -> Result<(), StoreError> {
        let connection = self.lock_catalog()?;
        let (state, cursor): (String, u64) = connection.query_row(
            "SELECT state,next_chunk_ordinal FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if state == "PUBLISHED" {
            return Ok(());
        }
        if !matches!(state.as_str(), "RECEIVING" | "VERIFIED") || cursor != references.len() as u64
        {
            return Err(StoreError::Authentication(
                "provisional chunk ledger state or count is incomplete",
            ));
        }
        let rows = {
            let mut statement = connection.prepare(
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
        if rows.len() != references.len() {
            return Err(StoreError::Authentication(
                "provisional chunk ledger has a noncontiguous inventory",
            ));
        }
        for (index, (ordinal, key, digest)) in rows.into_iter().enumerate() {
            if ordinal != index as u64
                || vec_id(key)? != references[index].object_key
                || vec_id(digest)? != references[index].object_digest
            {
                return Err(StoreError::Authentication(
                    "provisional chunk ledger identity changed",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn provisional_upload_path(&self, idempotency: &Id32) -> PathBuf {
        self.config
            .object_root
            .join("uploads")
            .join(hex(idempotency))
    }

    fn provisional_chunk_path(&self, idempotency: &Id32, ordinal: u64) -> PathBuf {
        self.provisional_upload_path(idempotency)
            .join(format!("{ordinal:020}.kvchunk"))
    }

    fn verify_provisional_ledger_entry(
        &self,
        idempotency: &Id32,
        ordinal: u64,
        reference: &ChunkRef,
    ) -> Result<(), StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<(Vec<u8>, Vec<u8>)> = connection
            .query_row(
                "SELECT object_key,object_digest FROM upload_chunks WHERE tenant=?1 AND idempotency_key=?2 AND ordinal=?3",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice(), ordinal],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if row.as_ref().is_none_or(|(key, digest)| {
            key.as_slice() != reference.object_key || digest.as_slice() != reference.object_digest
        }) {
            return Err(StoreError::Authentication(
                "provisional retry disagrees with its chunk ledger",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_content_reference(
    reference: &ChunkRef,
    object: &ChunkObject,
    key_epoch: u64,
) -> Result<(), StoreError> {
    if reference.chunk_id != object.chunk_id
        || reference.key_epoch != key_epoch
        || reference.plaintext_bytes != object.plaintext_bytes
    {
        return Err(StoreError::Authentication(
            "provisional dedup content identity or metadata changed",
        ));
    }
    Ok(())
}

fn write_or_verify_private(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    match options.open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(io_error("write provisional private object"))?;
            file.sync_all()
                .map_err(io_error("fsync provisional private object"))?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_file_bytes(path, bytes)
        }
        Err(source) => Err(StoreError::Io {
            op: "create provisional private object",
            source,
        }),
    }
}
