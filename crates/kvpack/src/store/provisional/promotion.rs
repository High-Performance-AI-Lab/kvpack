use super::staging::validate_content_reference;
use super::*;

impl LocalStore {
    pub(crate) fn promote_provisional_chunk(
        &self,
        idempotency: &Id32,
        staged: &ProvisionalStagedChunk,
        object: &ChunkObject,
        retention: RetentionInputs,
    ) -> Result<(ChunkRef, ProvisionalPromotedChunk), StoreError> {
        validate_reference(&staged.reference, object, self.key_epoch())?;
        let target = self.chunk_path(&object.object_key);
        if let Some(parent) = target.parent() {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            builder
                .create(parent)
                .map_err(io_error("create provisional CAS shard"))?;
        }
        let created_target = if let Some(source) = staged.staged_path.as_ref() {
            match fs::hard_link(source, &target) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    verify_file_bytes(&target, &object.bytes)?;
                    false
                }
                Err(source) => {
                    return Err(StoreError::Io {
                        op: "promote provisional chunk with no-replace link",
                        source,
                    });
                }
            }
        } else {
            verify_file_bytes(&target, &object.bytes)?;
            false
        };
        if created_target {
            fsync_dir(target.parent().expect("chunk target has a shard parent"))?;
        }
        // `created_target` means this operation hard-linked the exact staged
        // bytes the caller re-hashed immediately before this call; carry that
        // proof so the publish step skips the same-inode dedup re-read.  The
        // pre-existing dedup branches (no staged source, or the link lost a
        // race) keep verifying the target bytes.
        let retention = retention.with_physical_bytes(object.bytes.len() as u64)?;
        let reference = match if created_target {
            self.put_export_chunk_with_retention_verified(
                object,
                self.key_epoch(),
                retention,
                crate::store::publication::AlreadyVerifiedTarget,
            )
        } else {
            self.put_export_chunk_with_retention(object, self.key_epoch(), retention)
        } {
            Ok(reference) => reference,
            Err(error) => {
                if created_target {
                    self.cleanup_one_provisional_promotion(
                        idempotency,
                        source_from(staged)?,
                        &target,
                        &object.object_key,
                    )?;
                }
                return Err(error);
            }
        };
        validate_content_reference(&reference, object, self.key_epoch())?;
        let created_target = created_target && reference.object_key == object.object_key;
        if !created_target
            && staged.staged_path.is_some()
            && reference.object_key != object.object_key
        {
            self.cleanup_one_provisional_promotion(
                idempotency,
                source_from(staged)?,
                &target,
                &object.object_key,
            )?;
        } else if created_target {
            validate_reference(&reference, object, self.key_epoch())?;
        }
        Ok((
            reference,
            ProvisionalPromotedChunk {
                object_key: object.object_key,
                staged_path: staged.staged_path.clone(),
                created_target,
            },
        ))
    }

    pub(crate) fn cleanup_provisional_promotion(
        &self,
        idempotency: &Id32,
        promoted: &[ProvisionalPromotedChunk],
    ) -> Result<(), StoreError> {
        for chunk in promoted.iter().rev().filter(|chunk| chunk.created_target) {
            let Some(staged) = chunk.staged_path.as_ref() else {
                continue;
            };
            let target = self.chunk_path(&chunk.object_key);
            self.cleanup_one_provisional_promotion(
                idempotency,
                staged,
                &target,
                &chunk.object_key,
            )?;
        }
        Ok(())
    }

    fn cleanup_one_provisional_promotion(
        &self,
        idempotency: &Id32,
        staged: &Path,
        target: &Path,
        object_key: &Id32,
    ) -> Result<(), StoreError> {
        let staged_metadata = match fs::metadata(staged) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(StoreError::Io {
                    op: "inspect provisional staged cleanup link",
                    source,
                });
            }
        };
        let target_metadata = match fs::metadata(target) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(StoreError::Io {
                    op: "inspect provisional CAS cleanup link",
                    source,
                });
            }
        };
        if staged_metadata.dev() != target_metadata.dev()
            || staged_metadata.ino() != target_metadata.ino()
        {
            return Ok(());
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let safe: u64 = transaction.query_row(
                "SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM chunks c WHERE c.tenant=?1 AND c.object_key=?2 AND c.refcount<>0) AND NOT EXISTS(SELECT 1 FROM manifest_chunks m WHERE m.tenant=?1 AND m.object_key=?2) AND NOT EXISTS(SELECT 1 FROM upload_chunks uc JOIN uploads u ON u.tenant=uc.tenant AND u.idempotency_key=uc.idempotency_key WHERE uc.tenant=?1 AND uc.object_key=?2 AND uc.idempotency_key<>?3 AND u.state IN ('INIT','RESERVED','RECEIVING','VERIFIED')) THEN 1 ELSE 0 END",
                params![
                    self.tenant_namespace.as_slice(),
                    object_key.as_slice(),
                    idempotency.as_slice(),
                ],
                |row| row.get(0),
            )?;
        if safe == 0 {
            transaction.commit()?;
            return Ok(());
        }
        fs::remove_file(target).map_err(io_error("unlink exclusive provisional CAS link"))?;
        transaction.execute(
            "DELETE FROM locations WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2",
            params![self.tenant_namespace.as_slice(), object_key.as_slice()],
        )?;
        transaction.execute(
            "DELETE FROM policy_objects WHERE tenant=?1 AND object_key=?2",
            params![self.tenant_namespace.as_slice(), object_key.as_slice()],
        )?;
        transaction.execute(
            "DELETE FROM chunks WHERE tenant=?1 AND object_key=?2 AND refcount=0",
            params![self.tenant_namespace.as_slice(), object_key.as_slice()],
        )?;
        if let Err(error) = transaction.commit() {
            let _ = fs::hard_link(staged, target);
            return Err(error.into());
        }
        fsync_dir(target.parent().expect("chunk target has a shard parent"))?;
        Ok(())
    }
}

fn validate_reference(
    reference: &ChunkRef,
    object: &ChunkObject,
    key_epoch: u64,
) -> Result<(), StoreError> {
    if reference.chunk_id != object.chunk_id {
        return Err(StoreError::Authentication(
            "provisional chunk content identity changed during staging or promotion",
        ));
    }
    if reference.object_key != object.object_key || reference.object_digest != object.object_digest
    {
        return Err(StoreError::Authentication(
            "provisional encrypted object identity changed during staging or promotion",
        ));
    }
    if reference.key_epoch != key_epoch
        || reference.plaintext_bytes != object.plaintext_bytes
        || reference.object_bytes as usize != object.bytes.len()
    {
        return Err(StoreError::Authentication(
            "provisional chunk metadata changed during staging or promotion",
        ));
    }
    Ok(())
}

fn source_from(staged: &ProvisionalStagedChunk) -> Result<&Path, StoreError> {
    staged.staged_path.as_deref().ok_or(StoreError::State(
        "created provisional CAS link has no private source",
    ))
}
