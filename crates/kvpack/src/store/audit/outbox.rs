use super::*;

const DELIVERED_AUDIT_RETENTION_NS: u64 = 7 * 24 * 60 * 60 * 1_000_000_000;

pub(in crate::store) fn preflight_events(
    transaction: &Transaction<'_>,
    tenant: &Id32,
    events: &[AuditEventKey],
) -> Result<AuditCapacity, StoreError> {
    let unique = unique_events(events);
    if unique.iter().any(|event| {
        event.object_id.iter().all(|byte| *byte == 0) || event.generation > i64::MAX as u64
    }) {
        return Err(StoreError::State(
            "audit event identity or generation is invalid",
        ));
    }
    let mut needed = 0u64;
    for event in &unique {
        let exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND event_kind=?2 AND object_kind=?3 AND object_id=?4 AND generation=?5)",
            params![
                tenant.as_slice(),
                event.event.as_str(),
                event.object.as_str(),
                event.object_id.as_slice(),
                event.generation
            ],
            |row| row.get(0),
        )?;
        needed = needed.saturating_add(u64::from(!exists));
    }
    let pending: u64 = transaction.query_row(
        "SELECT COUNT(*) FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND delivered_ns IS NULL",
        [tenant.as_slice()],
        |row| row.get(0),
    )?;
    if pending.saturating_add(needed) > MAX_PENDING_AUDIT_RECORDS {
        let changed = transaction.execute(
            "UPDATE audit_state SET backpressure_events=backpressure_events+1 WHERE tenant=?1 AND stream='publication'",
            [tenant.as_slice()],
        )?;
        if changed != 1 {
            return Err(StoreError::State("publication audit stream is missing"));
        }
        return Ok(AuditCapacity::Backpressured);
    }
    Ok(AuditCapacity::Ready)
}

pub(in crate::store) fn append_events(
    transaction: &Transaction<'_>,
    tenant: &Id32,
    events: &[AuditEventKey],
    occurred_unix_ns: u64,
) -> Result<u64, StoreError> {
    if occurred_unix_ns == 0 {
        return Err(StoreError::State("audit event timestamp is invalid"));
    }
    let unique = unique_events(events);
    let mut next_sequence: u64 = transaction.query_row(
        "SELECT next_sequence FROM audit_state WHERE tenant=?1 AND stream='publication'",
        [tenant.as_slice()],
        |row| row.get(0),
    )?;
    let mut inserted = 0u64;
    for event in unique {
        let changed = transaction.execute(
            "INSERT OR IGNORE INTO audit_outbox(tenant,stream,sequence,event_kind,object_kind,object_id,generation,occurred_ns) VALUES(?1,'publication',?2,?3,?4,?5,?6,?7)",
            params![
                tenant.as_slice(),
                next_sequence,
                event.event.as_str(),
                event.object.as_str(),
                event.object_id.as_slice(),
                event.generation,
                occurred_unix_ns
            ],
        )?;
        if changed == 1 {
            inserted = inserted.saturating_add(1);
            next_sequence = next_sequence
                .checked_add(1)
                .ok_or(StoreError::State("audit sequence exhausted"))?;
        }
    }
    transaction.execute(
        "UPDATE audit_state SET next_sequence=?2 WHERE tenant=?1 AND stream='publication'",
        params![tenant.as_slice(), next_sequence],
    )?;
    Ok(inserted)
}

fn unique_events(events: &[AuditEventKey]) -> Vec<AuditEventKey> {
    let mut seen = BTreeSet::new();
    events
        .iter()
        .copied()
        .filter(|event| seen.insert(*event))
        .collect()
}

impl LocalStore {
    pub fn audit_status(&self) -> Result<AuditStatus, StoreError> {
        let connection = self.lock_catalog()?;
        load_status(&connection, &self.tenant_namespace)
    }

    pub fn export_audit_batch(
        &self,
        exporter: &dyn AuditExporter,
        maximum_records: usize,
    ) -> Result<AuditExportReport, StoreError> {
        if maximum_records == 0 || maximum_records > MAX_AUDIT_BATCH_RECORDS {
            return Err(StoreError::State("audit batch bound is invalid"));
        }
        let _serial = self
            .audit_export_serial
            .lock()
            .map_err(|_| StoreError::State("audit export mutex poisoned"))?;
        let batch = {
            let connection = self.lock_catalog()?;
            load_batch(&connection, &self.tenant_namespace, maximum_records)?
        };
        if batch.records.is_empty() {
            return Ok(AuditExportReport {
                exported_records: 0,
                retention_pruned_records: 0,
                status: self.audit_status()?,
            });
        }
        if let Err(error) = exporter.export(&batch) {
            self.record_delivery_failure()?;
            let _ = self.telemetry.record_audit(AuditOutcome::ExportRetry);
            return Err(error);
        }
        let delivered = now_ns();
        let pruned = {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for record in &batch.records {
                let changed = transaction.execute(
                    "UPDATE audit_outbox SET delivered_ns=?3,delivery_attempts=delivery_attempts+1 WHERE tenant=?1 AND stream='publication' AND sequence=?2 AND delivered_ns IS NULL",
                    params![self.tenant_namespace.as_slice(), record.sequence, delivered],
                )?;
                if changed != 1 {
                    return Err(StoreError::State("audit batch acknowledgement changed"));
                }
            }
            transaction.execute(
                "UPDATE audit_state SET last_flushed_ns=?2 WHERE tenant=?1 AND stream='publication'",
                params![self.tenant_namespace.as_slice(), delivered],
            )?;
            let pruned = prune_delivered(&transaction, &self.tenant_namespace, delivered)?;
            transaction.commit()?;
            pruned
        };
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Exported, batch.records.len() as u64);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::RetentionPruned, pruned);
        Ok(AuditExportReport {
            exported_records: batch.records.len() as u64,
            retention_pruned_records: pruned,
            status: self.audit_status()?,
        })
    }

    fn record_delivery_failure(&self) -> Result<(), StoreError> {
        let connection = self.lock_catalog()?;
        let changed = connection.execute(
            "UPDATE audit_state SET delivery_failures=delivery_failures+1 WHERE tenant=?1 AND stream='publication'",
            [self.tenant_namespace.as_slice()],
        )?;
        if changed != 1 {
            return Err(StoreError::State("publication audit stream is missing"));
        }
        Ok(())
    }
}

pub(super) fn load_batch(
    connection: &MutexGuard<'_, rusqlite::Connection>,
    tenant: &Id32,
    maximum_records: usize,
) -> Result<AuditBatch, StoreError> {
    let raw = {
        let mut statement = connection.prepare(
            "SELECT sequence,event_kind,object_kind,object_id,generation,occurred_ns FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND delivered_ns IS NULL ORDER BY sequence LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![tenant.as_slice(), maximum_records as u64], |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, u64>(4)?,
                    row.get::<_, u64>(5)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        rows
    };
    let records = raw
        .into_iter()
        .map(
            |(sequence, event, object, object_id, generation, occurred_unix_ns)| {
                Ok(AuditRecord {
                    sequence,
                    event: AuditEventKind::parse(&event)?,
                    object: AuditObjectKind::parse(&object)?,
                    object_id: vec_id(object_id)?,
                    generation,
                    occurred_unix_ns,
                })
            },
        )
        .collect::<Result<Vec<_>, StoreError>>()?;
    if records
        .windows(2)
        .any(|pair| pair[1].sequence != pair[0].sequence.saturating_add(1))
    {
        return Err(StoreError::State(
            "publication audit sequence contains a gap",
        ));
    }
    Ok(AuditBatch { records })
}

fn load_status(
    connection: &rusqlite::Connection,
    tenant: &Id32,
) -> Result<AuditStatus, StoreError> {
    let (next_sequence, last_flushed_unix_ns, backpressure_events, delivery_failures, retention_pruned_records, lost_records) = connection.query_row(
        "SELECT next_sequence,last_flushed_ns,backpressure_events,delivery_failures,retention_pruned_records,lost_records FROM audit_state WHERE tenant=?1 AND stream='publication'",
        [tenant.as_slice()],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
    )?;
    let (pending_records, retained_delivered_records) = connection.query_row(
        "SELECT COALESCE(SUM(delivered_ns IS NULL),0),COALESCE(SUM(delivered_ns IS NOT NULL),0) FROM audit_outbox WHERE tenant=?1 AND stream='publication'",
        [tenant.as_slice()],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(AuditStatus {
        pending_records,
        retained_delivered_records,
        next_sequence,
        last_flushed_unix_ns,
        backpressure_events,
        delivery_failures,
        retention_pruned_records,
        lost_records,
    })
}

fn prune_delivered(
    transaction: &Transaction<'_>,
    tenant: &Id32,
    now: u64,
) -> Result<u64, StoreError> {
    let cutoff = now.saturating_sub(DELIVERED_AUDIT_RETENTION_NS);
    let mut deleted = transaction.execute(
        "DELETE FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND delivered_ns IS NOT NULL AND delivered_ns<?2",
        params![tenant.as_slice(), cutoff],
    )? as u64;
    let threshold: Option<u64> = transaction
        .query_row(
            "SELECT sequence FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND delivered_ns IS NOT NULL ORDER BY sequence DESC LIMIT 1 OFFSET ?2",
            params![
                tenant.as_slice(),
                MAX_RETAINED_DELIVERED_AUDIT_RECORDS.saturating_sub(1)
            ],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(threshold) = threshold {
        deleted = deleted.saturating_add(transaction.execute(
            "DELETE FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND delivered_ns IS NOT NULL AND sequence<?2",
            params![tenant.as_slice(), threshold],
        )? as u64);
    }
    if deleted > 0 {
        transaction.execute(
            "UPDATE audit_state SET retention_pruned_records=retention_pruned_records+?2 WHERE tenant=?1 AND stream='publication'",
            params![tenant.as_slice(), deleted],
        )?;
    }
    Ok(deleted)
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
