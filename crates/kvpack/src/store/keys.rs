use rusqlite::{params, TransactionBehavior};

use crate::telemetry::AuditOutcome;
use crate::{CacheLifecycle, StoreError};

use super::{
    audit::{self, AuditCapacity, AuditEventKey},
    vec_id, AuditEventKind, AuditObjectKind, LocalStore,
};

const MAX_RETIREMENT_BATCH: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyEpochRetirementReport {
    pub minimum_readable_key_epoch: u64,
    pub manifests_tombstoned: u64,
    pub chunks_tombstoned: u64,
    pub remaining_objects: u64,
}

impl KeyEpochRetirementReport {
    pub fn complete(self) -> bool {
        self.remaining_objects == 0
    }
}

impl LocalStore {
    /// Tombstone a bounded batch of objects that require an epoch below the
    /// store's configured readable minimum. The configured minimum is already
    /// forward-only in the catalog; callers repeat this operation until the
    /// returned report is complete, then ordinary GC reclaims physical bytes.
    pub fn retire_key_epochs_before(
        &self,
        minimum_readable_key_epoch: u64,
        maximum_objects: usize,
    ) -> Result<KeyEpochRetirementReport, StoreError> {
        if minimum_readable_key_epoch != self.minimum_readable_key_epoch() {
            return Err(StoreError::State(
                "retirement epoch must equal the configured readable minimum",
            ));
        }
        if maximum_objects == 0 || maximum_objects > MAX_RETIREMENT_BATCH {
            return Err(StoreError::State(
                "key-epoch retirement batch bound is invalid",
            ));
        }

        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let manifests = {
            let mut statement = transaction.prepare(
                "SELECT m.manifest_id,m.generation FROM manifests m
                 WHERE m.tenant=?1
                 AND (m.key_epoch<?2 OR EXISTS(
                   SELECT 1 FROM manifest_chunks mc
                   JOIN chunks c ON c.tenant=mc.tenant AND c.object_key=mc.object_key
                   WHERE mc.tenant=m.tenant AND mc.manifest_id=m.manifest_id AND c.key_epoch<?2
                 ))
                 AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id)
                 ORDER BY m.key_epoch,m.manifest_id LIMIT ?3",
            )?;
            let rows = statement
                .query_map(
                    params![
                        self.tenant_namespace.as_slice(),
                        minimum_readable_key_epoch,
                        maximum_objects
                    ],
                    |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, u64>(1)?)),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        let chunk_limit = maximum_objects.saturating_sub(manifests.len());
        let chunks = if chunk_limit == 0 {
            Vec::new()
        } else {
            let mut statement = transaction.prepare(
                "SELECT c.object_key,c.key_epoch FROM chunks c
                 WHERE c.tenant=?1 AND c.key_epoch<?2
                 AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)
                 ORDER BY c.key_epoch,c.object_key LIMIT ?3",
            )?;
            let rows = statement
                .query_map(
                    params![
                        self.tenant_namespace.as_slice(),
                        minimum_readable_key_epoch,
                        chunk_limit
                    ],
                    |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, u64>(1)?)),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        let mut audit_events = Vec::with_capacity(manifests.len().saturating_add(chunks.len()));
        for (raw, generation) in &manifests {
            audit_events.push(AuditEventKey::new(
                AuditEventKind::Tombstoned,
                AuditObjectKind::Manifest,
                vec_id(raw.clone())?,
                *generation,
            ));
        }
        for (raw, key_epoch) in &chunks {
            audit_events.push(AuditEventKey::new(
                AuditEventKind::Tombstoned,
                AuditObjectKind::Chunk,
                vec_id(raw.clone())?,
                *key_epoch,
            ));
        }
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }

        let timestamp = now_ns();
        for (raw, _) in &manifests {
            let manifest_id = vec_id(raw.clone())?;
            transaction.execute(
                "INSERT OR IGNORE INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'manifest',?2,?3,?4)",
                params![
                    self.tenant_namespace.as_slice(),
                    manifest_id.as_slice(),
                    self.catalog_epoch(),
                    timestamp
                ],
            )?;
            transaction.execute(
                "DELETE FROM prefix_checkpoints WHERE tenant=?1 AND manifest_id=?2",
                params![self.tenant_namespace.as_slice(), manifest_id.as_slice()],
            )?;
        }
        for (raw, _) in &chunks {
            let object_key = vec_id(raw.clone())?;
            transaction.execute(
                "INSERT OR IGNORE INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'chunk',?2,?3,?4)",
                params![
                    self.tenant_namespace.as_slice(),
                    object_key.as_slice(),
                    self.catalog_epoch(),
                    timestamp
                ],
            )?;
        }

        let remaining_manifests: u64 = transaction.query_row(
            "SELECT COUNT(*) FROM manifests m
             WHERE m.tenant=?1
             AND (m.key_epoch<?2 OR EXISTS(
               SELECT 1 FROM manifest_chunks mc
               JOIN chunks c ON c.tenant=mc.tenant AND c.object_key=mc.object_key
               WHERE mc.tenant=m.tenant AND mc.manifest_id=m.manifest_id AND c.key_epoch<?2
             ))
             AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id)",
            params![self.tenant_namespace.as_slice(), minimum_readable_key_epoch],
            |row| row.get(0),
        )?;
        let remaining_chunks: u64 = transaction.query_row(
            "SELECT COUNT(*) FROM chunks c WHERE c.tenant=?1 AND c.key_epoch<?2
             AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)",
            params![self.tenant_namespace.as_slice(), minimum_readable_key_epoch],
            |row| row.get(0),
        )?;
        let remaining_objects = remaining_manifests
            .checked_add(remaining_chunks)
            .ok_or(StoreError::State("retired object count overflow"))?;
        let tombstoned = (manifests.len() as u64)
            .checked_add(chunks.len() as u64)
            .ok_or(StoreError::State("retired tombstone count overflow"))?;
        let audit_enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            timestamp,
        )?;
        transaction.commit()?;
        let _ = self
            .telemetry
            .record_lifecycle_count(CacheLifecycle::Tombstoned, tombstoned);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
        Ok(KeyEpochRetirementReport {
            minimum_readable_key_epoch,
            manifests_tombstoned: manifests.len() as u64,
            chunks_tombstoned: chunks.len() as u64,
            remaining_objects,
        })
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
