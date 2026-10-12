use super::*;

const EVICT_MANIFEST_SQL: &str = "SELECT m.manifest_id,m.file_bytes,m.generation
 FROM manifests m
 WHERE m.tenant=?1
 AND EXISTS(SELECT 1 FROM locations ml WHERE ml.tenant=m.tenant AND ml.object_kind='manifest' AND ml.object_id=m.manifest_id AND ml.tier='local' AND ml.state='AVAILABLE')
 AND NOT EXISTS(SELECT 1 FROM manifests child WHERE child.tenant=m.tenant AND child.parent_id=m.manifest_id)
 AND NOT EXISTS(SELECT 1 FROM leases l WHERE l.tenant=m.tenant AND l.object_kind='manifest' AND l.object_id=m.manifest_id AND l.state='ACTIVE' AND l.expires_ns>?2)
 AND NOT EXISTS(SELECT 1 FROM source_lease_objects slo JOIN source_leases sl ON sl.tenant=slo.tenant AND sl.lease_id=slo.lease_id WHERE slo.tenant=m.tenant AND slo.object_kind='manifest' AND slo.object_id=m.manifest_id AND (sl.state='UNCERTAIN' OR (sl.state='ACTIVE' AND sl.expires_ns>?2)))
 AND NOT EXISTS(SELECT 1 FROM manifest_chunks mc JOIN pins p ON p.tenant=mc.tenant AND p.object_key=mc.object_key WHERE mc.tenant=m.tenant AND mc.manifest_id=m.manifest_id)
 AND NOT EXISTS(SELECT 1 FROM manifest_chunks mc JOIN leases l ON l.tenant=mc.tenant AND l.object_kind='chunk' AND l.object_id=mc.object_key AND l.state='ACTIVE' AND l.expires_ns>?2 WHERE mc.tenant=m.tenant AND mc.manifest_id=m.manifest_id)
 AND NOT EXISTS(SELECT 1 FROM manifest_chunks mc JOIN source_lease_objects slo ON slo.tenant=mc.tenant AND slo.object_kind='chunk' AND slo.object_id=mc.object_key JOIN source_leases sl ON sl.tenant=slo.tenant AND sl.lease_id=slo.lease_id WHERE mc.tenant=m.tenant AND mc.manifest_id=m.manifest_id AND (sl.state='UNCERTAIN' OR (sl.state='ACTIVE' AND sl.expires_ns>?2)))
 AND NOT EXISTS(SELECT 1 FROM uploads u WHERE u.tenant=m.tenant AND u.state IN ('INIT','RESERVED','RECEIVING','VERIFIED'))
 ORDER BY CASE WHEN EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id) THEN 0 ELSE 1 END,COALESCE((
   SELECT CAST(COALESCE(SUM(COALESCE(p.score,0)),0) AS REAL) /
     MAX(1,m.file_bytes+COALESCE(SUM(CASE WHEN c.refcount=1 THEN c.object_bytes ELSE 0 END),0))
   FROM (SELECT DISTINCT object_key FROM manifest_chunks WHERE tenant=m.tenant AND manifest_id=m.manifest_id) d
   JOIN chunks c ON c.tenant=m.tenant AND c.object_key=d.object_key
   LEFT JOIN policy_objects p ON p.tenant=c.tenant AND p.object_key=c.object_key
 ),0),m.published_ns,m.manifest_id
 LIMIT 1";

const GC_CHUNK_BATCH_SQL: &str = "SELECT c.object_key,c.key_epoch,c.location_state
 FROM chunks c
 LEFT JOIN policy_objects p ON p.tenant=c.tenant AND p.object_key=c.object_key
 WHERE c.tenant=?1 AND c.refcount=0 AND c.location_state IN ('AVAILABLE','TOMBSTONED')
 AND NOT EXISTS(SELECT 1 FROM pins pin WHERE pin.tenant=c.tenant AND pin.object_key=c.object_key)
 AND NOT EXISTS(SELECT 1 FROM leases l WHERE l.tenant=c.tenant AND l.object_kind='chunk' AND l.object_id=c.object_key AND l.state='ACTIVE' AND l.expires_ns>?2)
 AND NOT EXISTS(SELECT 1 FROM source_lease_objects slo JOIN source_leases sl ON sl.tenant=slo.tenant AND sl.lease_id=slo.lease_id WHERE slo.tenant=c.tenant AND slo.object_kind='chunk' AND slo.object_id=c.object_key AND (sl.state='UNCERTAIN' OR (sl.state='ACTIVE' AND sl.expires_ns>?2)))
 AND NOT EXISTS(SELECT 1 FROM uploads u WHERE u.tenant=c.tenant AND u.state IN ('INIT','RESERVED','RECEIVING','VERIFIED'))
 ORDER BY CASE WHEN EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key) THEN 0 ELSE 1 END,
 CASE COALESCE(p.segment,c.retention_segment) WHEN 'PROBATIONARY' THEN 0 ELSE 1 END,
 COALESCE(p.score,0),COALESCE(p.last_access_ns,c.last_access_ns),c.object_bytes DESC,c.object_key
 LIMIT ?3";

impl LocalStore {
    /// Tombstone and remove one leaf manifest, releasing its chunk references.
    /// Parents remain until all authenticated children have been evicted.
    pub fn evict_manifest_one(&self) -> Result<bool, StoreError> {
        Ok(self.evict_manifest_one_inner()?.is_some())
    }

    /// `evict_manifest_one` returning the freed durable byte count, so the
    /// capacity loop can track utilization without a fresh stat() scan.
    pub(super) fn evict_manifest_one_inner(&self) -> Result<Option<u64>, StoreError> {
        let (manifest_id, file_bytes, tombstoned, audit_enqueued) = {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let row: Option<(Vec<u8>, u64, u64)> = transaction
                .query_row(
                    EVICT_MANIFEST_SQL,
                    params![self.tenant_namespace.as_slice(), now_ns()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?;
            let Some((raw, file_bytes, generation)) = row else {
                return Ok(None);
            };
            let manifest_id = vec_id(raw)?;
            let audit_events = [AuditEventKey::new(
                AuditEventKind::Tombstoned,
                AuditObjectKind::Manifest,
                manifest_id,
                generation,
            )];
            if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
                == AuditCapacity::Backpressured
            {
                transaction.commit()?;
                let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
                return Err(StoreError::Busy);
            }
            let objects = {
                let mut statement = transaction.prepare("SELECT DISTINCT object_key FROM manifest_chunks WHERE tenant=?1 AND manifest_id=?2")?;
                let rows = statement.query_map(
                    params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
                    |row| row.get::<_, Vec<u8>>(0),
                )?;
                rows.collect::<Result<Vec<_>, _>>()?
            };
            let tombstoned = transaction.execute("INSERT OR IGNORE INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'manifest',?2,?3,?4)", params![self.tenant_namespace.as_slice(), manifest_id.as_slice(), self.catalog_epoch(), now_ns()])?;
            transaction.execute(
                "DELETE FROM prefix_checkpoints WHERE tenant=?1 AND manifest_id=?2",
                params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
            )?;
            transaction.execute("UPDATE locations SET state='EVICTING' WHERE tenant=?1 AND object_kind='manifest' AND object_id=?2 AND tier='local'", params![self.tenant_namespace.as_slice(), manifest_id.as_slice()])?;
            for object in objects {
                transaction.execute("UPDATE chunks SET refcount=MAX(0,refcount-1) WHERE tenant=?1 AND object_key=?2", params![self.tenant_namespace.as_slice(), object])?;
            }
            transaction.execute(
                "DELETE FROM manifest_chunks WHERE tenant=?1 AND manifest_id=?2",
                params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
            )?;
            let audit_enqueued = audit::append_events(
                &transaction,
                &self.tenant_namespace,
                &audit_events,
                now_ns(),
            )?;
            transaction.commit()?;
            (manifest_id, file_bytes, tombstoned as u64, audit_enqueued)
        };
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Tombstoned, tombstoned);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        let freed = self.finish_manifest_eviction(manifest_id, file_bytes)?;
        Ok(Some(freed))
    }

    /// Evict one unreferenced, unpinned CAS object using the transactional
    /// AVAILABLE→EVICTING→TRASHED→ABSENT sequence.
    pub fn gc_one(&self) -> Result<bool, StoreError> {
        Ok(self.gc_chunk_batch(1)?.1 > 0)
    }

    /// Evict up to `limit` unreferenced, unpinned CAS objects. Victims are
    /// selected and tombstoned in ONE catalog transaction and their
    /// EVICTING→TRASHED→ABSENT completion runs in ONE more, instead of two
    /// transactions per object.  Returns (freed durable bytes, evicted count).
    /// Tombstone-rung rows (`location_state='TOMBSTONED'`) release no durable
    /// bytes here: demotion already accounted for them.
    pub(super) fn gc_chunk_batch(&self, limit: usize) -> Result<(u64, u64), StoreError> {
        let selected: Vec<(kvpack_core::Id32, bool)> = {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let victims = {
                let mut statement = transaction.prepare(GC_CHUNK_BATCH_SQL)?;
                let rows = statement.query_map(
                    params![self.tenant_namespace.as_slice(), now_ns(), limit as u64],
                    |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, u64>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )?;
                rows.collect::<Result<Vec<_>, _>>()?
            };
            if victims.is_empty() {
                return Ok((0, 0));
            }
            let mut keyed = Vec::with_capacity(victims.len());
            let mut audit_events = Vec::with_capacity(victims.len());
            for (raw, key_epoch, location_state) in victims {
                if location_state != "AVAILABLE" && location_state != "TOMBSTONED" {
                    return Err(StoreError::State(
                        "chunk eviction victim has an invalid catalog state",
                    ));
                }
                let id = vec_id(raw)?;
                audit_events.push(AuditEventKey::new(
                    AuditEventKind::Tombstoned,
                    AuditObjectKind::Chunk,
                    id,
                    key_epoch,
                ));
                keyed.push((id, location_state == "TOMBSTONED"));
            }
            if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
                == AuditCapacity::Backpressured
            {
                transaction.commit()?;
                let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
                return Err(StoreError::Busy);
            }
            let mut selected = Vec::with_capacity(keyed.len());
            let mut selected_events = Vec::with_capacity(keyed.len());
            let mut tombstoned = 0u64;
            for ((id, was_tombstoned), event) in keyed.into_iter().zip(audit_events) {
                tombstoned = tombstoned.saturating_add(transaction.execute("INSERT OR IGNORE INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'chunk',?2,?3,?4)", params![self.tenant_namespace.as_slice(), id.as_slice(), self.catalog_epoch(), now_ns()])? as u64);
                let changed = transaction.execute("UPDATE chunks SET location_state='EVICTING' WHERE tenant=?1 AND object_key=?2 AND location_state IN ('AVAILABLE','TOMBSTONED')", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                if changed == 1 {
                    selected.push((id, was_tombstoned));
                    selected_events.push(event);
                }
            }
            let audit_enqueued = if selected_events.is_empty() {
                0
            } else {
                audit::append_events(
                    &transaction,
                    &self.tenant_namespace,
                    &selected_events,
                    now_ns(),
                )?
            };
            transaction.commit()?;
            let _ = self
                .telemetry
                .record_lifecycle_count(CacheLifecycle::Tombstoned, tombstoned);
            let _ = self
                .telemetry
                .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
            selected
        };
        let mut moved: Vec<(kvpack_core::Id32, u64, bool)> = Vec::with_capacity(selected.len());
        for (id, was_tombstoned) in &selected {
            let object_bytes = {
                let connection = self.lock_catalog()?;
                let row: Option<(u64, u64, String)> = connection
                    .query_row(
                        "SELECT object_bytes,refcount,location_state FROM chunks WHERE tenant=?1 AND object_key=?2",
                        params![self.tenant_namespace.as_slice(), id.as_slice()],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                let Some((object_bytes, refcount, state)) = row else {
                    continue;
                };
                if refcount != 0 || (state != "EVICTING" && state != "TRASHED") {
                    return Err(StoreError::State(
                        "chunk eviction recovery found an invalid catalog transition",
                    ));
                }
                object_bytes
            };
            let source = self.chunk_path(id);
            let trash = self
                .config
                .object_root
                .join("trash")
                .join(format!("{}.trash", hex(id)));
            move_and_unlink(
                &source,
                &trash,
                "move evicting chunk to trash",
                "unlink trashed chunk",
            )?;
            moved.push((*id, object_bytes, *was_tombstoned));
        }
        if moved.is_empty() {
            return Ok((0, 0));
        }
        let mut freed = 0u64;
        let mut deleted_events: Vec<AuditEventKey> = Vec::with_capacity(moved.len());
        let audit_enqueued = {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let collected_events: Vec<AuditEventKey> = moved
                .iter()
                .map(|(id, _, _)| {
                    AuditEventKey::new(AuditEventKind::Collected, AuditObjectKind::Chunk, *id, 0)
                })
                .collect();
            if audit::preflight_events(&transaction, &self.tenant_namespace, &collected_events)?
                == AuditCapacity::Backpressured
            {
                transaction.commit()?;
                let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
                return Err(StoreError::Busy);
            }
            for (id, object_bytes, was_tombstoned) in &moved {
                transaction.execute("UPDATE chunks SET location_state='TRASHED' WHERE tenant=?1 AND object_key=?2 AND location_state='EVICTING'", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                transaction.execute("UPDATE locations SET state='TRASHED' WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2 AND state IN ('AVAILABLE','EVICTING')", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                transaction.execute(
                    "DELETE FROM locations WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2",
                    params![self.tenant_namespace.as_slice(), id.as_slice()],
                )?;
                transaction.execute(
                    "DELETE FROM policy_objects WHERE tenant=?1 AND object_key=?2",
                    params![self.tenant_namespace.as_slice(), id.as_slice()],
                )?;
                let deleted = transaction.execute("DELETE FROM chunks WHERE tenant=?1 AND object_key=?2 AND location_state='TRASHED' AND refcount=0", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                if deleted == 1 {
                    // Tombstone-rung rows released their durable bytes at
                    // demotion; only byte-backed rows free capacity here.
                    if !was_tombstoned {
                        freed = freed
                            .checked_add(*object_bytes)
                            .ok_or(StoreError::State("evicted byte total overflow"))?;
                    }
                    deleted_events.push(AuditEventKey::new(
                        AuditEventKind::Collected,
                        AuditObjectKind::Chunk,
                        *id,
                        0,
                    ));
                }
            }
            if freed > 0 {
                transaction.execute(
                    "UPDATE tenants SET durable_bytes=MAX(0,durable_bytes-?2) WHERE namespace=?1",
                    params![self.tenant_namespace.as_slice(), freed],
                )?;
            }
            let audit_enqueued = if deleted_events.is_empty() {
                0
            } else {
                audit::append_events(
                    &transaction,
                    &self.tenant_namespace,
                    &deleted_events,
                    now_ns(),
                )?
            };
            transaction.commit()?;
            audit_enqueued
        };
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Collected, deleted_events.len() as u64);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        for (id, _) in &selected {
            self.policy
                .lock()
                .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
                .remove(id);
        }
        Ok((freed, moved.len() as u64))
    }

    pub(in crate::store) fn reconcile_evictions(&self, bound: usize) -> Result<usize, StoreError> {
        let mut completed = 0usize;
        while completed < bound {
            let manifest: Option<(Vec<u8>, u64)> = {
                let connection = self.lock_catalog()?;
                connection
                    .query_row(
                        "SELECT m.manifest_id,m.file_bytes FROM manifests m JOIN locations l ON l.tenant=m.tenant AND l.object_kind='manifest' AND l.object_id=m.manifest_id WHERE m.tenant=?1 AND l.tier='local' AND l.state IN ('EVICTING','TRASHED') ORDER BY m.manifest_id LIMIT 1",
                        [self.tenant_namespace.as_slice()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?
            };
            if let Some((manifest_id, file_bytes)) = manifest {
                self.finish_manifest_eviction(vec_id(manifest_id)?, file_bytes)?;
                completed += 1;
                continue;
            }
            let chunk: Option<Vec<u8>> = {
                let connection = self.lock_catalog()?;
                connection
                    .query_row(
                        "SELECT object_key FROM chunks WHERE tenant=?1 AND location_state IN ('EVICTING','TRASHED') ORDER BY object_key LIMIT 1",
                        [self.tenant_namespace.as_slice()],
                        |row| row.get(0),
                    )
                    .optional()?
            };
            let Some(chunk) = chunk else {
                break;
            };
            self.finish_chunk_eviction(vec_id(chunk)?)?;
            completed += 1;
        }
        Ok(completed)
    }

    fn finish_manifest_eviction(
        &self,
        manifest_id: kvpack_core::Id32,
        file_bytes: u64,
    ) -> Result<u64, StoreError> {
        let source = self.manifest_path(&manifest_id);
        let trash = self
            .config
            .object_root
            .join("trash")
            .join(format!("{}.manifest.trash", hex(&manifest_id)));
        move_and_unlink(
            &source,
            &trash,
            "move evicting manifest to trash",
            "unlink trashed manifest",
        )?;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Collected,
            AuditObjectKind::Manifest,
            manifest_id,
            0,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        transaction.execute("UPDATE locations SET state='TRASHED' WHERE tenant=?1 AND object_kind='manifest' AND object_id=?2 AND state='EVICTING'", params![self.tenant_namespace.as_slice(), manifest_id.as_slice()])?;
        transaction.execute(
            "DELETE FROM locations WHERE tenant=?1 AND object_kind='manifest' AND object_id=?2",
            params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
        )?;
        let deleted = transaction.execute(
            "DELETE FROM manifests WHERE tenant=?1 AND manifest_id=?2",
            params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
        )?;
        if deleted == 1 {
            transaction.execute(
                "UPDATE tenants SET durable_bytes=MAX(0,durable_bytes-?2) WHERE namespace=?1",
                params![self.tenant_namespace.as_slice(), file_bytes],
            )?;
        }
        let audit_enqueued = if deleted == 1 {
            audit::append_events(
                &transaction,
                &self.tenant_namespace,
                &audit_events,
                now_ns(),
            )?
        } else {
            0
        };
        transaction.commit()?;
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Collected, deleted as u64);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        Ok(if deleted == 1 { file_bytes } else { 0 })
    }

    fn finish_chunk_eviction(&self, object_key: kvpack_core::Id32) -> Result<(), StoreError> {
        let (object_bytes, rung_tombstoned) = {
            let connection = self.lock_catalog()?;
            let row: Option<(u64, u64, String, u64)> = connection
                .query_row(
                    "SELECT object_bytes,refcount,location_state,fidelity_rung FROM chunks WHERE tenant=?1 AND object_key=?2",
                    params![self.tenant_namespace.as_slice(), object_key.as_slice()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            let Some((object_bytes, refcount, state, fidelity_rung)) = row else {
                return Ok(());
            };
            if refcount != 0 || (state != "EVICTING" && state != "TRASHED") {
                return Err(StoreError::State(
                    "chunk eviction recovery found an invalid catalog transition",
                ));
            }
            (object_bytes, fidelity_rung == 2)
        };
        let source = self.chunk_path(&object_key);
        let trash = self
            .config
            .object_root
            .join("trash")
            .join(format!("{}.trash", hex(&object_key)));
        move_and_unlink(
            &source,
            &trash,
            "move evicting chunk to trash",
            "unlink trashed chunk",
        )?;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Collected,
            AuditObjectKind::Chunk,
            object_key,
            0,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        transaction.execute("UPDATE chunks SET location_state='TRASHED' WHERE tenant=?1 AND object_key=?2 AND location_state='EVICTING'", params![self.tenant_namespace.as_slice(), object_key.as_slice()])?;
        transaction.execute("UPDATE locations SET state='TRASHED' WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2 AND state IN ('AVAILABLE','EVICTING')", params![self.tenant_namespace.as_slice(), object_key.as_slice()])?;
        transaction.execute(
            "DELETE FROM locations WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2",
            params![self.tenant_namespace.as_slice(), object_key.as_slice()],
        )?;
        transaction.execute(
            "DELETE FROM policy_objects WHERE tenant=?1 AND object_key=?2",
            params![self.tenant_namespace.as_slice(), object_key.as_slice()],
        )?;
        let deleted = transaction.execute("DELETE FROM chunks WHERE tenant=?1 AND object_key=?2 AND location_state='TRASHED' AND refcount=0", params![self.tenant_namespace.as_slice(), object_key.as_slice()])?;
        if deleted == 1 && !rung_tombstoned {
            transaction.execute(
                "UPDATE tenants SET durable_bytes=MAX(0,durable_bytes-?2) WHERE namespace=?1",
                params![self.tenant_namespace.as_slice(), object_bytes],
            )?;
        }
        let audit_enqueued = if deleted == 1 {
            audit::append_events(
                &transaction,
                &self.tenant_namespace,
                &audit_events,
                now_ns(),
            )?
        } else {
            0
        };
        transaction.commit()?;
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Collected, deleted as u64);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        self.policy
            .lock()
            .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
            .remove(&object_key);
        Ok(())
    }
}

pub(super) fn move_and_unlink(
    source: &std::path::Path,
    trash: &std::path::Path,
    move_op: &'static str,
    unlink_op: &'static str,
) -> Result<(), StoreError> {
    match fs::rename(source, trash) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && trash.exists() => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(StoreError::Io {
                op: move_op,
                source,
            });
        }
    }
    fsync_dir(
        source
            .parent()
            .ok_or(StoreError::State("eviction source has no parent"))?,
    )?;
    fsync_dir(
        trash
            .parent()
            .ok_or(StoreError::State("eviction trash has no parent"))?,
    )?;
    match fs::remove_file(trash) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(StoreError::Io {
                op: unlink_op,
                source,
            });
        }
    }
    fsync_dir(
        trash
            .parent()
            .ok_or(StoreError::State("eviction trash has no parent"))?,
    )?;
    Ok(())
}
