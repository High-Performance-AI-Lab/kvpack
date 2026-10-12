use super::*;

impl LocalStore {
    pub fn stat(&self) -> Result<CatalogStat, StoreError> {
        let result = self.stat_inner();
        let _ = self
            .telemetry
            .set_health(HealthComponent::Catalog, result.is_ok());
        result
    }

    fn stat_inner(&self) -> Result<CatalogStat, StoreError> {
        self.reap_expired_provisional_uploads()?;
        let connection = self.lock_catalog()?;
        let (durable_bytes, reserved_bytes, quota_bytes, staging_quota_bytes): (
            u64,
            u64,
            u64,
            u64,
        ) = connection.query_row(
            "SELECT durable_bytes,reserved_bytes,quota_bytes,staging_quota_bytes FROM tenants WHERE namespace=?1",
            [self.tenant_namespace.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        let quarantine_bytes = connection.query_row(
            "SELECT COALESCE(SUM(file_bytes),0) FROM quarantine_entries WHERE tenant=?1",
            [self.tenant_namespace.as_slice()],
            |row| row.get(0),
        )?;
        // The catalog has no persisted dedup ledger (provisional receipts
        // track deduplicated bytes only in memory), so the honest store-wide
        // figure is derived from chunk refcounts: every reference beyond the
        // first is a physical copy that was never written.
        let deduplicated_bytes = connection.query_row(
            "SELECT COALESCE(SUM((refcount-1)*object_bytes),0) FROM chunks WHERE refcount>1",
            [],
            |row| row.get(0),
        )?;
        let manifests = count(&connection, "manifests", &self.tenant_namespace)?;
        let chunks = count(&connection, "chunks", &self.tenant_namespace)?;
        let pins = count(&connection, "pins", &self.tenant_namespace)?;
        let active_uploads = connection.query_row(
            "SELECT COUNT(*) FROM uploads WHERE tenant=?1 AND state IN ('INIT','RESERVED','RECEIVING','VERIFIED')",
            [self.tenant_namespace.as_slice()],
            |row| row.get(0),
        )?;
        let active_grants = connection.query_row(
            "SELECT COUNT(*) FROM grants WHERE tenant=?1 AND state='ACTIVE'",
            [self.tenant_namespace.as_slice()],
            |row| row.get(0),
        )?;
        let active_leases = connection.query_row(
            "SELECT COUNT(*) FROM leases WHERE tenant=?1 AND state='ACTIVE'",
            [self.tenant_namespace.as_slice()],
            |row| row.get(0),
        )?;
        let active_source_leases = connection.query_row(
            "SELECT COUNT(*) FROM source_leases WHERE tenant=?1 AND state IN ('ACTIVE','UNCERTAIN')",
            [self.tenant_namespace.as_slice()],
            |row| row.get(0),
        )?;
        drop(connection);
        let active_restores = self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?
            .len() as u64;
        let mut provisional_directories = 0u64;
        for entry in fs::read_dir(self.config.object_root.join("uploads"))
            .map_err(io_error("read upload work directory for catalog status"))?
        {
            let entry = entry.map_err(io_error("read upload work status entry"))?;
            let file_type = entry
                .file_type()
                .map_err(io_error("inspect upload work status entry"))?;
            if file_type.is_symlink() {
                return Err(StoreError::State(
                    "upload work directory contains a symlink",
                ));
            }
            if file_type.is_dir() {
                provisional_directories = provisional_directories
                    .checked_add(1)
                    .ok_or(StoreError::State("provisional directory count overflow"))?;
            }
        }
        let _ = self
            .telemetry
            .set_resource(ResourceGauge::DurableBytes, durable_bytes);
        let _ = self
            .telemetry
            .set_resource(ResourceGauge::StagingReservedBytes, reserved_bytes);
        let _ = self
            .telemetry
            .set_resource(ResourceGauge::QuarantineBytes, quarantine_bytes);
        let _ = self
            .telemetry
            .set_resource(ResourceGauge::CatalogPins, pins);
        Ok(CatalogStat {
            active_grants,
            active_leases,
            active_restores,
            active_source_leases,
            active_uploads,
            deduplicated_bytes,
            durable_bytes,
            reserved_bytes,
            quota_bytes,
            staging_quota_bytes,
            quarantine_bytes,
            manifests,
            chunks,
            pins,
            provisional_directories,
        })
    }
}

fn count(connection: &Connection, table: &str, tenant: &Id32) -> Result<u64, StoreError> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE tenant=?1");
    Ok(connection.query_row(&sql, [tenant.as_slice()], |row| row.get(0))?)
}
