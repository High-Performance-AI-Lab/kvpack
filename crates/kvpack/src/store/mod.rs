use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kvpack_core::{namespace_id, CutManifest, EncodedPack, Id32, PrefixNode};
use rusqlite::{Connection, OpenFlags};
use sha2::{Digest, Sha256};

use crate::error::io_error;
use crate::restore::{
    charge_within_limits, HeldRestoreResources, RestoreLimits, RestoreResourceCharge,
};
use crate::telemetry::{
    FallbackReason, HealthComponent, LookupResult, OperationalTelemetry, ResourceGauge,
    ServiceComponent, SpanOutcome, TraceContext, TracePhase,
};
use crate::{StoreError, StoreKey};

mod audit;
mod fill;
mod fsck;
mod gc;
mod holds;
mod import;
mod keys;
mod policy;
mod provisional;
mod publication;
mod read;
mod recovery;
mod remote;
mod resolve;
mod schema;
mod stats;

pub use audit::{
    AuditBatch, AuditDirectoryExporter, AuditDirectoryPolicy, AuditEventKind, AuditExportReport,
    AuditExporter, AuditObjectKind, AuditRecord, AuditStatus, AUDIT_SCHEMA_VERSION,
    MAX_AUDIT_BATCH_RECORDS, MAX_PENDING_AUDIT_RECORDS, MAX_RETAINED_DELIVERED_AUDIT_RECORDS,
};
pub use fill::{FillCancellation, SingleflightFill};
pub use fsck::{FsckBounds, FsckReport};
pub use gc::{CachePressure, DemotionReport, EvictionReport, UtilizationPolicy};
pub use import::AuthenticatedImportStatus;
pub use keys::KeyEpochRetirementReport;
pub use policy::{
    access_flush_schedule, replay_cache_policy, replay_offline_oracle, replay_tinylfu,
    retention_value, AccessFlushSchedule, AdmissionDecision, CacheReplayConfig, CacheReplayOutcome,
    CacheReplayPolicy, CacheReplayResult, CacheReplayTier, CacheTraceEvent, OfflineOracleBounds,
    OfflineOracleResult, PolicyReplayEvent, PolicyReplayResult, ReplayTierProfile, RetentionInputs,
    RetentionSegment, TinyLfuConfig, TinyLfuPolicy, MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS,
    MAX_OFFLINE_ORACLE_EVENTS, MAX_OFFLINE_ORACLE_STATES, MAX_OFFLINE_ORACLE_TRANSITIONS,
};
pub(crate) use provisional::{
    ProvisionalPromotedChunk, ProvisionalStageMode, ProvisionalStagedChunk,
};
pub use provisional::{ProvisionalProvenance, ProvisionalUploadMetadata};
pub(crate) use read::{release_restore_pin_batch, RetainedPin};
pub use read::{AuthenticatedPublicationChunk, AuthenticatedPublicationSource};
pub mod direct {
    //! Runtime-gated cache-bypass chunk read fast path (see
    //! [`direct_read_enabled`](self::direct_read_enabled)).
    pub use super::read::direct::*;
}
pub use recovery::{
    CatalogBackupBounds, CatalogBackupReport, CatalogReconciliationReport, InventorySnapshotBounds,
    InventorySnapshotReport, ReconciliationBounds, VerifiedInventorySnapshot,
};
pub use remote::{
    InventoryCursor, InventoryEntry, InventoryObjectKind, MutationReplay, RemoteImportFence,
    RemoteMutation, SourceLeaseState, SourceLeaseStatus,
};

pub const CATALOG_SCHEMA_VERSION: i64 = 11;

/// Bound on buffered chunk bytes between batched chunk puts. Writer sessions
/// flush their pending batch at this threshold and always at state end, so
/// peak memory stays bounded while the per-chunk fsync storm collapses into
/// one directory sync set and one catalog transaction per batch.
pub(crate) const CHUNK_PUT_BATCH_BYTES: u64 = 64 * 1024 * 1024;

/// Derive the tenant namespace needed by service identity maps without opening
/// or mutating the store catalog.
pub fn derive_store_tenant_namespace(
    operator_tenant_id: &[u8],
    key_epoch: u64,
    key: &StoreKey,
) -> Result<Id32, StoreError> {
    if operator_tenant_id.is_empty() || key_epoch == 0 || !key.supports_epoch(key_epoch) {
        return Err(StoreError::State(
            "tenant namespace inputs or key epoch are invalid",
        ));
    }
    let tenant_seed: Id32 = Sha256::digest(operator_tenant_id).into();
    let bootstrap = key.schedule(&tenant_seed, key_epoch)?;
    Ok(namespace_id(bootstrap.namespace_key(), operator_tenant_id)?)
}

#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub object_root: PathBuf,
    pub catalog_path: PathBuf,
    pub operator_tenant_id: Vec<u8>,
    pub key_epoch: u64,
    pub minimum_readable_key_epoch: u64,
    pub catalog_epoch: u64,
    pub quota_bytes: u64,
    pub staging_quota_bytes: u64,
    pub endurance_bytes_per_five_minutes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadState {
    Init,
    Reserved,
    Receiving,
    Verified,
    Published,
    Aborted,
    Quarantined,
}

impl UploadState {
    pub(crate) fn parse(value: &str) -> Result<Self, StoreError> {
        match value {
            "INIT" => Ok(Self::Init),
            "RESERVED" => Ok(Self::Reserved),
            "RECEIVING" => Ok(Self::Receiving),
            "VERIFIED" => Ok(Self::Verified),
            "PUBLISHED" => Ok(Self::Published),
            "ABORTED" => Ok(Self::Aborted),
            "QUARANTINED" => Ok(Self::Quarantined),
            _ => Err(StoreError::State(
                "catalog contains an unknown upload state",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixHit {
    pub manifest_id: Id32,
    pub token_count: u64,
    pub recompute_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogStat {
    pub active_grants: u64,
    pub active_leases: u64,
    pub active_restores: u64,
    pub active_source_leases: u64,
    pub active_uploads: u64,
    /// Store-wide physical dedup savings: bytes that additional manifest
    /// references to already-stored chunks did NOT cost, aggregated across
    /// every tenant's published manifests as
    /// `SUM((refcount-1)*object_bytes)` over multiply-referenced chunks.
    pub deduplicated_bytes: u64,
    pub durable_bytes: u64,
    pub reserved_bytes: u64,
    pub quota_bytes: u64,
    pub staging_quota_bytes: u64,
    pub quarantine_bytes: u64,
    pub manifests: u64,
    pub chunks: u64,
    pub pins: u64,
    pub provisional_directories: u64,
}

pub struct LocalStore {
    pub(super) config: StoreConfig,
    pub(super) tenant_namespace: Id32,
    pub(super) key: StoreKey,
    pub(super) catalog: Mutex<Connection>,
    pub(super) backup_catalog: Mutex<Connection>,
    pub(crate) restore_holds: Mutex<BTreeMap<Id32, RestoreHold>>,
    pub(super) manifest_cache: Mutex<read::ManifestLru>,
    pub(crate) policy: Mutex<TinyLfuPolicy>,
    pub(crate) access_flush_scheduled: AtomicBool,
    pub(crate) access_flush_last_attempt_ns: AtomicU64,
    pub(crate) telemetry: Arc<OperationalTelemetry>,
    pub(crate) audit_export_serial: Mutex<()>,
    #[cfg(test)]
    durability_fault: Mutex<Option<publication::DurabilityFaultPoint>>,
}

pub(crate) struct RestoreHold {
    pins: Vec<RetainedPin>,
    resources: RestoreResourceCharge,
}

pub(crate) struct PendingManifest<'a> {
    pub encoded: &'a EncodedPack,
    pub manifest: &'a CutManifest,
    pub prefix_node: PrefixNode,
    pub exact_final: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UploadReservation {
    pub expected_bytes: u64,
    pub publication_generation: u64,
    pub intent_digest: Id32,
    pub retention: RetentionInputs,
}

pub(crate) struct QuarantinedUploadFile {
    pub entry_id: Id32,
    pub path_token: String,
    pub file_bytes: u64,
}

impl std::fmt::Debug for LocalStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalStore")
            .field("object_root", &self.config.object_root)
            .field("catalog_path", &self.config.catalog_path)
            .field("tenant_namespace", &hex(&self.tenant_namespace))
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl LocalStore {
    pub fn open(config: StoreConfig, key: StoreKey) -> Result<Self, StoreError> {
        if config.key_epoch == 0
            || config.minimum_readable_key_epoch == 0
            || config.minimum_readable_key_epoch > config.key_epoch
            || config
                .key_epoch
                .saturating_sub(config.minimum_readable_key_epoch)
                >= 64
            || config.catalog_epoch == 0
            || config.quota_bytes == 0
            || config.staging_quota_bytes == 0
        {
            return Err(StoreError::State(
                "store epochs, readable window, or durable/staging quotas are invalid",
            ));
        }
        if !(config.minimum_readable_key_epoch..=config.key_epoch)
            .all(|epoch| key.supports_epoch(epoch))
        {
            return Err(StoreError::State(
                "store key does not contain the complete readable epoch window",
            ));
        }
        create_private_dir(&config.object_root)?;
        for name in [
            "chunks",
            "manifests",
            "partials",
            "uploads",
            "trash",
            "quarantine",
        ] {
            create_private_dir(&config.object_root.join(name))?;
        }
        if let Some(parent) = config.catalog_path.parent() {
            create_private_dir(parent)?;
        }
        let tenant_namespace =
            derive_store_tenant_namespace(&config.operator_tenant_id, config.key_epoch, &key)?;
        let mut connection = Connection::open(&config.catalog_path)?;
        schema::migrate(&mut connection)?;
        schema::register_tenant(&mut connection, &config, &tenant_namespace)?;
        let policy = load_policy(&connection, &tenant_namespace, config.quota_bytes)?;
        // Keep the online-backup reader open for the store lifetime. Besides
        // avoiding connection setup in every backup, this gives the descriptor
        // explicit ownership on platforms where SQLite defers closing a second
        // database FD while another connection holds POSIX locks.
        let backup_connection = Connection::open_with_flags(
            &config.catalog_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        backup_connection.busy_timeout(Duration::from_secs(5))?;
        backup_connection.pragma_update(None, "trusted_schema", "OFF")?;
        let telemetry = Arc::new(OperationalTelemetry::default());
        let store = Self {
            config,
            tenant_namespace,
            key,
            catalog: Mutex::new(connection),
            backup_catalog: Mutex::new(backup_connection),
            restore_holds: Mutex::new(BTreeMap::new()),
            manifest_cache: Mutex::new(read::ManifestLru::new()),
            policy: Mutex::new(policy),
            access_flush_scheduled: AtomicBool::new(false),
            access_flush_last_attempt_ns: AtomicU64::new(0),
            telemetry,
            audit_export_serial: Mutex::new(()),
            #[cfg(test)]
            durability_fault: Mutex::new(None),
        };
        store.mark_active_source_leases_uncertain()?;
        store.reconcile_work_dirs(10_000)?;
        store.enforce_quarantine_cap()?;
        let _ = store.telemetry.set_health(HealthComponent::Store, true);
        let _ = store.telemetry.set_health(HealthComponent::Catalog, true);
        Ok(store)
    }

    pub fn tenant_namespace(&self) -> Id32 {
        self.tenant_namespace
    }
    pub fn key_epoch(&self) -> u64 {
        self.config.key_epoch
    }
    pub fn minimum_readable_key_epoch(&self) -> u64 {
        self.config.minimum_readable_key_epoch
    }
    pub fn catalog_epoch(&self) -> u64 {
        self.config.catalog_epoch
    }
    pub fn telemetry(&self) -> Arc<OperationalTelemetry> {
        Arc::clone(&self.telemetry)
    }
    pub fn prometheus_metrics(&self) -> Result<String, StoreError> {
        self.stat()?;
        self.telemetry.prometheus_text()
    }
    pub fn derive_input_cut(
        &self,
        semantic_model: &kvpack_core::SemanticModelId,
        family: &kvpack_core::RepresentationFamilyId,
        tokens: &[u32],
        auxiliary_inputs: &[kvpack_core::AuxiliaryInputId],
    ) -> Result<(kvpack_core::InputCutId, Vec<PrefixNode>), StoreError> {
        let keys = self.schedule(self.config.key_epoch)?;
        Ok(kvpack_core::derive_input_cut(
            keys.prefix_key(),
            &self.tenant_namespace,
            semantic_model,
            family,
            tokens,
            auxiliary_inputs,
        )?)
    }
    pub(crate) fn schedule(&self, epoch: u64) -> Result<kvpack_core::KeySchedule, StoreError> {
        if epoch < self.config.minimum_readable_key_epoch || epoch > self.config.key_epoch {
            return Err(StoreError::State(
                "key epoch is outside the readable window",
            ));
        }
        self.key.schedule(&self.tenant_namespace, epoch)
    }
    pub(crate) fn chunk_path(&self, object_key: &Id32) -> PathBuf {
        let name = hex(object_key);
        self.config
            .object_root
            .join("chunks")
            .join(&name[..2])
            .join(format!("{name}.kvchunk"))
    }
    pub(crate) fn manifest_path(&self, manifest_id: &Id32) -> PathBuf {
        let name = hex(manifest_id);
        self.config
            .object_root
            .join("manifests")
            .join(&name[..2])
            .join(format!("{name}.kvpack"))
    }
    pub(crate) fn lock_catalog(&self) -> Result<std::sync::MutexGuard<'_, Connection>, StoreError> {
        self.catalog
            .lock()
            .map_err(|_| StoreError::State("catalog mutex poisoned"))
    }
}

fn load_policy(
    connection: &Connection,
    tenant: &Id32,
    capacity_bytes: u64,
) -> Result<TinyLfuPolicy, StoreError> {
    let mut policy = TinyLfuPolicy::new(TinyLfuConfig::production(capacity_bytes)?)?;
    let mut statement = connection.prepare("SELECT c.object_key,c.object_bytes,COALESCE(p.frequency,c.frequency_estimate),COALESCE(p.score,0),COALESCE(p.segment,c.retention_segment),COALESCE(p.last_access_ns,c.last_access_ns) FROM chunks c LEFT JOIN policy_objects p ON p.tenant=c.tenant AND p.object_key=c.object_key WHERE c.tenant=?1 AND c.location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key) ORDER BY COALESCE(p.last_access_ns,c.last_access_ns),c.object_key")?;
    let rows = statement.query_map([tenant.as_slice()], |row| {
        Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, u64>(1)?,
            row.get::<_, u64>(2)?,
            row.get::<_, u64>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, u64>(5)?,
        ))
    })?;
    for row in rows {
        let (key, bytes, frequency, score, segment, last_access_ns) = row?;
        policy.restore_entry(
            vec_id(key)?,
            bytes,
            frequency,
            score,
            RetentionSegment::parse(&segment)?,
            last_access_ns,
        );
    }
    Ok(policy)
}

pub(crate) fn create_private_dir(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder
        .create(path)
        .map_err(crate::error::io_error("create private store directory"))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(crate::error::io_error(
        "set private store directory permissions",
    ))
}

pub(crate) fn fsync_dir(path: &Path) -> Result<(), StoreError> {
    fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(crate::error::io_error("fsync store directory"))
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    const NIBBLES: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(NIBBLES[(byte >> 4) as usize] as char);
        out.push(NIBBLES[(byte & 0x0f) as usize] as char);
    }
    out
}
pub(crate) fn vec_id(bytes: Vec<u8>) -> Result<Id32, StoreError> {
    bytes
        .try_into()
        .map_err(|_| StoreError::Authentication("catalog identity has invalid length"))
}

pub(crate) fn semantic_digest(value: &kvpack_core::SemanticModelId) -> Id32 {
    kvpack_core::semantic_model_id(value)
}
pub(crate) fn family_digest(
    value: &kvpack_core::RepresentationFamilyId,
) -> Result<Id32, StoreError> {
    Ok(kvpack_core::representation_family_id(value)?)
}
