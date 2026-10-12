use super::*;

const QUARANTINE_MAINTENANCE_BOUND: usize = 1024;

impl LocalStore {
    pub fn prune_quarantine(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
    ) -> Result<u64, StoreError> {
        self.prune_quarantine_bounded(tier_capacity_bytes, policy, usize::MAX)
            .map(|(removed, _, _)| removed)
    }

    pub(crate) fn enforce_quarantine_cap(&self) -> Result<(), StoreError> {
        let (_, remaining, limit) = self.prune_quarantine_bounded(
            self.config.quota_bytes,
            UtilizationPolicy::default(),
            QUARANTINE_MAINTENANCE_BOUND,
        )?;
        if remaining > limit {
            return Err(StoreError::Quota(
                "quarantine capacity requires bounded maintenance",
            ));
        }
        Ok(())
    }

    pub(crate) fn maintain_quarantine_cap(&self) -> Result<(), StoreError> {
        self.prune_quarantine_bounded(
            self.config.quota_bytes,
            UtilizationPolicy::default(),
            QUARANTINE_MAINTENANCE_BOUND,
        )?;
        Ok(())
    }

    fn prune_quarantine_bounded(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
        maximum_entries: usize,
    ) -> Result<(u64, u64, u64), StoreError> {
        let policy = policy.validate()?;
        if tier_capacity_bytes == 0 || maximum_entries == 0 {
            return Err(StoreError::State(
                "quarantine capacity and maintenance bound must be nonzero",
            ));
        }
        let fraction_million = (policy.quarantine_fraction * 1_000_000.0).round() as u64;
        let limit = ((tier_capacity_bytes as u128).saturating_mul(fraction_million as u128)
            / 1_000_000)
            .min(u64::MAX as u128) as u64;
        let directory = self.config.object_root.join("quarantine");
        let now = now_ns();
        let mut removed = 0u64;
        let mut operations = 0usize;
        loop {
            let (total, candidate) = {
                let connection = self.lock_catalog()?;
                let total: u64 = connection.query_row(
                    "SELECT COALESCE(SUM(file_bytes),0) FROM quarantine_entries WHERE tenant=?1",
                    [self.tenant_namespace.as_slice()],
                    |row| row.get(0),
                )?;
                let candidate: Option<(Vec<u8>, String, u64, u64)> = if total > limit {
                    connection
                        .query_row(
                            "SELECT entry_id,path_token,file_bytes,expires_ns FROM quarantine_entries WHERE tenant=?1 ORDER BY CASE WHEN expires_ns<=?2 THEN 0 ELSE 1 END,created_ns,entry_id LIMIT 1",
                            params![self.tenant_namespace.as_slice(), now],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                        )
                        .optional()?
                } else {
                    connection
                        .query_row(
                            "SELECT entry_id,path_token,file_bytes,expires_ns FROM quarantine_entries WHERE tenant=?1 AND expires_ns<=?2 ORDER BY expires_ns,created_ns,entry_id LIMIT 1",
                            params![self.tenant_namespace.as_slice(), now],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                        )
                        .optional()?
                };
                (total, candidate)
            };
            let Some((entry_id, path_token, bytes, expires_ns)) = candidate else {
                return Ok((removed, total, limit));
            };
            if operations >= maximum_entries || (total <= limit && expires_ns > now) {
                return Ok((removed, total, limit));
            }
            let entry_id = vec_id(entry_id)?;
            let path = quarantine_path(&directory, &path_token)?;
            match fs::remove_file(&path) {
                Ok(()) => fsync_dir(&directory)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(StoreError::Io {
                        op: "prune quarantine entry",
                        source,
                    });
                }
            }
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let audit_events = [AuditEventKey::new(
                AuditEventKind::Collected,
                AuditObjectKind::Quarantine,
                entry_id,
                0,
            )];
            if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
                == AuditCapacity::Backpressured
            {
                transaction.commit()?;
                let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
                return Err(StoreError::Busy);
            }
            let deleted = transaction.execute(
                "DELETE FROM quarantine_entries WHERE tenant=?1 AND entry_id=?2",
                params![self.tenant_namespace.as_slice(), entry_id.as_slice()],
            )?;
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
            if deleted == 1 {
                removed = removed.saturating_add(bytes);
                operations += 1;
                let _ = self.telemetry.record_lifecycle(CacheLifecycle::Collected);
                let _ = self
                    .telemetry
                    .record_audit_count(AuditOutcome::Enqueued, audit_enqueued);
            } else {
                break;
            }
        }
        let remaining = self.stat()?.quarantine_bytes;
        Ok((removed, remaining, limit))
    }
}

fn quarantine_path(directory: &Path, token: &str) -> Result<std::path::PathBuf, StoreError> {
    let mut components = Path::new(token).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) if !name.is_empty() => Ok(directory.join(name)),
        _ => Err(StoreError::State(
            "catalog quarantine entry contains an invalid path token",
        )),
    }
}
