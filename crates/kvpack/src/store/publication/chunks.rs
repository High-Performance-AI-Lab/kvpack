use super::immutable::{
    write_immutable, write_immutable_batch_mode, write_immutable_deferred_cleanup,
};
use super::*;

/// One already-published chunk object awaiting its catalog rows inside a
/// shared transaction.
struct ChunkCatalogWrite<'a> {
    object: &'a ChunkObject,
    path: PathBuf,
    published: bool,
    retention: crate::RetentionInputs,
}

impl LocalStore {
    pub(crate) fn put_chunk(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
    ) -> Result<ChunkRef, StoreError> {
        self.put_chunk_with_retention(
            object,
            key_epoch,
            crate::RetentionInputs::conservative(
                object.bytes.len() as u64,
                object.plaintext_bytes as u64,
            ),
        )
    }

    pub(crate) fn put_chunk_with_retention(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
        retention: crate::RetentionInputs,
    ) -> Result<ChunkRef, StoreError> {
        self.put_chunk_with_retention_mode(object, key_epoch, retention, false, None)
    }

    pub(crate) fn put_export_chunk_with_retention(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
        retention: crate::RetentionInputs,
    ) -> Result<ChunkRef, StoreError> {
        self.put_chunk_with_retention_mode(object, key_epoch, retention, true, None)
    }

    /// Export-path chunk put carrying proof that the caller already verified
    /// the exact bytes behind the object's immutable target in this operation
    /// (provisional promotion), so the same-inode dedup re-read is skipped.
    pub(in crate::store) fn put_export_chunk_with_retention_verified(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
        retention: crate::RetentionInputs,
        verified: immutable::AlreadyVerifiedTarget,
    ) -> Result<ChunkRef, StoreError> {
        self.put_chunk_with_retention_mode(object, key_epoch, retention, true, Some(verified))
    }

    fn put_chunk_with_retention_mode(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
        retention: crate::RetentionInputs,
        defer_partial_cleanup: bool,
        verified: Option<immutable::AlreadyVerifiedTarget>,
    ) -> Result<ChunkRef, StoreError> {
        if let Some(existing) = self.find_chunk(&object.chunk_id, key_epoch)? {
            return Ok(existing);
        }
        let retention = retention.with_physical_bytes(object.bytes.len() as u64)?;
        let path = self.chunk_path(&object.object_key);
        let published = if defer_partial_cleanup {
            write_immutable_deferred_cleanup(
                self,
                DurableObjectKind::Chunk,
                "partials",
                &path,
                &object.bytes,
                verified,
            )?
        } else {
            write_immutable(
                self,
                DurableObjectKind::Chunk,
                "partials",
                &path,
                &object.bytes,
            )?
        };
        let writes = [ChunkCatalogWrite {
            object,
            path,
            published,
            retention,
        }];
        self.catalog_chunks(&writes, key_epoch, None)?
            .pop()
            .ok_or(StoreError::State(
                "chunk catalog write returned no reference",
            ))
    }

    /// Batched chunk put: one dedup lookup per object, one staged-immutable
    /// batch write (per-file fsync, ONE directory sync set), and ONE catalog
    /// transaction for every row, mirroring the
    /// `write_immutable_batch`/`publish_manifest_batch_inner` pattern.
    /// References come back in input order.
    pub(crate) fn put_chunks_batch_with_retention(
        &self,
        objects: &[&ChunkObject],
        key_epoch: u64,
        retention: crate::RetentionInputs,
        defer_partial_cleanup: bool,
    ) -> Result<Vec<ChunkRef>, StoreError> {
        let mut references: Vec<Option<ChunkRef>> = Vec::with_capacity(objects.len());
        references.resize_with(objects.len(), || None);
        let mut missing = Vec::with_capacity(objects.len());
        for (index, object) in objects.iter().enumerate() {
            if let Some(existing) = self.find_chunk(&object.chunk_id, key_epoch)? {
                references[index] = Some(existing);
            } else {
                missing.push(index);
            }
        }
        if missing.is_empty() {
            return references
                .into_iter()
                .map(|reference| {
                    reference.ok_or(StoreError::State("batched chunk put lost a reference"))
                })
                .collect();
        }
        let mut writes = Vec::with_capacity(missing.len());
        for &index in &missing {
            let object = objects[index];
            writes.push((self.chunk_path(&object.object_key), object.bytes.as_slice()));
        }
        let published = write_immutable_batch_mode(
            self,
            DurableObjectKind::Chunk,
            "partials",
            &writes,
            defer_partial_cleanup,
        )?;
        let mut catalog_writes = Vec::with_capacity(missing.len());
        for (&index, published) in missing.iter().zip(&published) {
            let object = objects[index];
            catalog_writes.push(ChunkCatalogWrite {
                object,
                path: self.chunk_path(&object.object_key),
                published: *published,
                retention: retention.with_physical_bytes(object.bytes.len() as u64)?,
            });
        }
        let resolved = self.catalog_chunks(&catalog_writes, key_epoch, None)?;
        for (&index, reference) in missing.iter().zip(resolved) {
            references[index] = Some(reference);
        }
        references
            .into_iter()
            .map(|reference| {
                reference.ok_or(StoreError::State("batched chunk put lost a reference"))
            })
            .collect()
    }

    /// Authenticated-import chunk put: the immutable write is cataloged in the
    /// SAME transaction as the upload-ledger row and cursor advance, instead
    /// of two catalog commits per imported chunk.
    pub(in crate::store) fn put_chunk_with_retention_and_ledger(
        &self,
        object: &ChunkObject,
        key_epoch: u64,
        retention: crate::RetentionInputs,
        idempotency: &Id32,
        ordinal: u64,
    ) -> Result<ChunkRef, StoreError> {
        let retention = retention.with_physical_bytes(object.bytes.len() as u64)?;
        let path = self.chunk_path(&object.object_key);
        let published = write_immutable(
            self,
            DurableObjectKind::Chunk,
            "partials",
            &path,
            &object.bytes,
        )?;
        let writes = [ChunkCatalogWrite {
            object,
            path,
            published,
            retention,
        }];
        self.catalog_chunks(&writes, key_epoch, Some((*idempotency, ordinal)))?
            .pop()
            .ok_or(StoreError::State(
                "chunk catalog write returned no reference",
            ))
    }

    /// ONE catalog transaction for a batch of already-published chunk
    /// objects: chunk rows, location rows, policy rows, and optionally the
    /// single authenticated-import ledger row plus cursor advance.
    fn catalog_chunks(
        &self,
        writes: &[ChunkCatalogWrite<'_>],
        key_epoch: u64,
        ledger: Option<(Id32, u64)>,
    ) -> Result<Vec<ChunkRef>, StoreError> {
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let timestamp = now_ns();
        let mut outcomes = Vec::with_capacity(writes.len());
        for write in writes {
            let object = write.object;
            let inserted = transaction.execute("INSERT OR IGNORE INTO chunks(tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,location_state,created_ns,last_access_ns,retention_segment,frequency_estimate) VALUES(?1,?2,?3,?4,?5,?6,?7,'AVAILABLE',?8,?8,'PROBATIONARY',1)", params![self.tenant_namespace.as_slice(), object.object_key.as_slice(), object.chunk_id.as_slice(), object.object_digest.as_slice(), key_epoch, object.plaintext_bytes, object.bytes.len() as u64, timestamp])?;
            let canonical: (Vec<u8>, Vec<u8>, u64, u32, u32) = transaction.query_row("SELECT object_key,object_digest,key_epoch,plaintext_bytes,object_bytes FROM chunks WHERE tenant=?1 AND chunk_id=?2 AND key_epoch=?3", params![self.tenant_namespace.as_slice(), object.chunk_id.as_slice(), key_epoch], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))?;
            if inserted == 1 {
                transaction.execute("INSERT OR REPLACE INTO locations(tenant,object_kind,object_id,tier,state,locator) VALUES(?1,'chunk',?2,'local','AVAILABLE',?3)", params![self.tenant_namespace.as_slice(), object.object_key.as_slice(), write.path.to_string_lossy()])?;
                transaction.execute("INSERT INTO policy_objects(tenant,object_key,frequency,segment,score,last_access_ns,last_persisted_epoch) VALUES(?1,?2,1,'PROBATIONARY',0,?3,0)", params![self.tenant_namespace.as_slice(), object.object_key.as_slice(), timestamp])?;
            }
            outcomes.push((inserted == 1, canonical));
        }
        if let Some((idempotency, ordinal)) = ledger {
            let cursor: u64 = transaction.query_row(
                "SELECT next_chunk_ordinal FROM uploads WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING'",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| row.get(0),
            )?;
            if ordinal != cursor {
                return Err(StoreError::Expectation(
                    "chunk upload raced ahead of the authenticated import cursor",
                ));
            }
            let object = writes[0].object;
            transaction.execute("INSERT INTO upload_chunks(tenant,idempotency_key,ordinal,object_key,object_digest,verified_ns) VALUES(?1,?2,?3,?4,?5,?6)", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), ordinal, object.object_key.as_slice(), object.object_digest.as_slice(), now_ns()])?;
            let changed = transaction.execute("UPDATE uploads SET next_chunk_ordinal=next_chunk_ordinal+1,updated_ns=?4 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING' AND next_chunk_ordinal=?3", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), ordinal, now_ns()])?;
            if changed != 1 {
                return Err(StoreError::State(
                    "authenticated import cursor failed to advance",
                ));
            }
        }
        transaction.commit()?;
        let mut references = Vec::with_capacity(writes.len());
        for (write, (inserted, canonical)) in writes.iter().zip(outcomes) {
            let object = write.object;
            if inserted {
                let _ = self
                    .telemetry
                    .add_bytes(ByteCounter::DurableWritten, object.bytes.len() as u64);
                self.policy
                    .lock()
                    .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
                    .register_with_retention(object.object_key, write.retention, timestamp);
            }
            let key = vec_id(canonical.0)?;
            if !inserted && write.published && key != object.object_key {
                quarantine_object(self, &write.path)?;
            }
            references.push(ChunkRef {
                chunk_id: object.chunk_id,
                object_key: key,
                object_digest: vec_id(canonical.1)?,
                key_epoch: canonical.2,
                plaintext_bytes: canonical.3,
                object_bytes: canonical.4,
            });
        }
        Ok(references)
    }

    pub(crate) fn sync_export_partial_cleanup(&self) -> Result<(), StoreError> {
        fsync_dir(&self.config.object_root.join("partials"))
    }

    pub(crate) fn find_chunk(
        &self,
        chunk_id: &Id32,
        key_epoch: u64,
    ) -> Result<Option<ChunkRef>, StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<StoredChunkRow> = connection.query_row("SELECT object_key,object_digest,key_epoch,plaintext_bytes,object_bytes FROM chunks WHERE tenant=?1 AND chunk_id=?2 AND key_epoch=?3 AND location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=chunks.tenant AND t.object_kind='chunk' AND t.object_id=chunks.object_key)", params![self.tenant_namespace.as_slice(), chunk_id.as_slice(), key_epoch], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))).optional()?;
        row.map(
            |(object_key, digest, key_epoch, plaintext_bytes, object_bytes)| {
                Ok(ChunkRef {
                    chunk_id: *chunk_id,
                    object_key: vec_id(object_key)?,
                    object_digest: vec_id(digest)?,
                    key_epoch,
                    plaintext_bytes,
                    object_bytes,
                })
            },
        )
        .transpose()
    }
}
