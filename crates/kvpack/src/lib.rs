//! Authenticated production-v1 content-addressed store.
//!
//! The public restore surface is semantic and store-based.  Legacy path and
//! record-index readers are intentionally not exported.

#![deny(unsafe_code)]

#[cfg(test)]
#[allow(dead_code)]
mod artifact;
mod capacity;
mod error;
mod export;
mod gguf_layout;
mod inspection;
mod intent;
mod mla;
mod prefill;
mod restore;
mod sink;
mod store;
mod store_key;
mod telemetry;
#[cfg_attr(not(test), allow(dead_code))]
mod writer;

#[cfg(test)]
pub(crate) use artifact::{
    ArtifactLocator, AuthenticatedArtifact, OpenExpectations, RestoreSelection,
};
pub use capacity::{
    CapacityPlanningError, CapacityWorksheet, CapacityWorksheetInput,
    CAPACITY_WORKSHEET_SCHEMA_VERSION, MAX_CAPACITY_BACKUP_GENERATIONS,
    MAX_CAPACITY_CATALOG_BYTES_PER_ROW, MAX_CAPACITY_CONCURRENCY, MAX_CAPACITY_GATEWAY_INSTANCES,
    MAX_CAPACITY_NETWORK_UTILIZATION_MILLIS, MAX_CAPACITY_WRITE_AMPLIFICATION_MILLIS,
    PRODUCTION_CAPACITY_RATIO_MILLIS, PRODUCTION_DURABLE_TARGET_MILLIS,
    PRODUCTION_GATEWAY_DATA_STREAMS, PRODUCTION_GATEWAY_IN_FLIGHT_BYTES,
    PRODUCTION_GATEWAY_RECEIVE_BUFFERS_PER_STREAM, PRODUCTION_GATEWAY_RECEIVE_BUFFER_BYTES,
    PRODUCTION_INTERNAL_TARGET_MILLIS, PRODUCTION_QUARANTINE_MILLIS,
    PRODUCTION_WRITE_AMPLIFICATION_LIMIT_MILLIS,
};
pub use error::StoreError;
pub use export::{
    ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateBounds, ExportStateDeclaration,
    ExportStateWriter, ProvisionalExportDeclaration, ProvisionalExportReceipt,
    ProvisionalExportSeal, ProvisionalExportSession, ProvisionalIoIntervalV1,
    ProvisionalStateReceipt, PublishedCut, PublishedCutSet,
};
pub use gguf_layout::{
    derive_layout_from_gguf, derive_layout_from_metadata, derive_layout_from_sidecar,
    parse_layout_sidecar, read_gguf_metadata, GgufMetadata, GgufValue, OwnedLayoutClassV2,
    OwnedLayoutV2, RopeConvention,
};
pub use inspection::{inspect_untrusted, InspectionBounds, UntrustedInspection};
pub use mla::{
    derive_mla_layout_from_metadata, expand_mla_latents, parse_mla_expansion_descriptor,
    MlaExpandedKv, MlaExpansionDescriptor, MlaTargetLayout, MLA_EXPANSION_DESCRIPTOR_SCHEMA_V1,
    MLA_LATENT_LAYOUT_CLASS, MLA_LATENT_STATE_NAME,
};
pub use prefill::{
    bind_weights_scalar_math_v2, derive_portable_prefill_descriptor_v1,
    derive_portable_prefill_descriptor_v2, derive_portable_prefill_descriptor_v2_from_layout,
    muse_session_resume_preconditions, muse_session_tail_shortfalls,
    place_windowed_tail_into_engine_ring, portable_prefill_geometry_v1,
    portable_prefill_layout_name_v2, portable_prefill_layout_v2, portable_prefill_token_ids_sha256,
    relocate_plane_bytes, relocate_session_planes, verify_nope_planes_require_no_rotation,
    ArtifactTailCoverage, MuseSessionArtifact, MuseSessionArtifactReceipt, MuseSessionPlaneWriter,
    MuseSessionWriter, PortablePrefillDescriptorInputV1, PortablePrefillDescriptorInputV2,
    PortablePrefillDescriptorV1, PortablePrefillGeometryV1, PortablePrefillLayoutClassV2,
    PortablePrefillLayoutV2, PositionDelta, PreRopeKernelPinV1, RelocateAction,
    SessionRelocateReport, TailCoverageShortfall, WeightsScalarMathV1, MUSE_EXACT_LOGITS_LAYER,
    MUSE_EXACT_LOGITS_STATE, PORTABLE_PREFILL_ABI_V1, PORTABLE_PREFILL_ABI_V2,
    PORTABLE_PREFILL_ABI_V2_PREROPE, PORTABLE_PREFILL_GEOMETRIES_V1,
    PORTABLE_PREFILL_GEOMETRY_QWEN25_05B_V1, PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1,
    PORTABLE_PREFILL_HEAD_DIM_V1, PORTABLE_PREFILL_KV_HEADS_V1, PORTABLE_PREFILL_LAYERS_V1,
    PORTABLE_PREFILL_LAYOUTS_V2, PORTABLE_PREFILL_MAX_CONTEXT_V1,
};
pub use restore::{
    AuthenticatedRestorePlan, AuthenticatedScatterDescriptor, HeldRestoreResources,
    InstalledRestore, PinnedScatterBatch, PreparedScatterTransfer, RestoreAvailableSource,
    RestoreCancellation, RestoreCandidate, RestoreLimits, RestoreRequest,
    RestoreResourceRequirements, RestoreTier, ShadowRestoreHandle, MAX_SCATTER_FDS_PER_BATCH,
};
pub use sink::{RestoreStatePlan, VerifiedRestoreSink};
pub use store::direct;
pub use store::{
    access_flush_schedule, derive_store_tenant_namespace, replay_cache_policy,
    replay_offline_oracle, replay_tinylfu, retention_value, AccessFlushSchedule, AdmissionDecision,
    AuditBatch, AuditDirectoryExporter, AuditDirectoryPolicy, AuditEventKind, AuditExportReport,
    AuditExporter, AuditObjectKind, AuditRecord, AuditStatus, AuthenticatedImportStatus,
    AuthenticatedPublicationChunk, AuthenticatedPublicationSource, CachePressure,
    CacheReplayConfig, CacheReplayOutcome, CacheReplayPolicy, CacheReplayResult, CacheReplayTier,
    CacheTraceEvent, CatalogBackupBounds, CatalogBackupReport, CatalogReconciliationReport,
    CatalogStat, DemotionReport, EvictionReport, FillCancellation, FsckBounds, FsckReport,
    InventoryCursor, InventoryEntry, InventoryObjectKind, InventorySnapshotBounds,
    InventorySnapshotReport, KeyEpochRetirementReport, LocalStore, MutationReplay,
    OfflineOracleBounds, OfflineOracleResult, PolicyReplayEvent, PolicyReplayResult, PrefixHit,
    ProvisionalProvenance, ProvisionalUploadMetadata, ReconciliationBounds, RemoteImportFence,
    RemoteMutation, ReplayTierProfile, RetentionInputs, RetentionSegment, SingleflightFill,
    SourceLeaseState, SourceLeaseStatus, StoreConfig, TinyLfuConfig, TinyLfuPolicy, UploadState,
    UtilizationPolicy, VerifiedInventorySnapshot, AUDIT_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION,
    MAX_AUDIT_BATCH_RECORDS, MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS, MAX_OFFLINE_ORACLE_EVENTS,
    MAX_OFFLINE_ORACLE_STATES, MAX_OFFLINE_ORACLE_TRANSITIONS, MAX_PENDING_AUDIT_RECORDS,
    MAX_RETAINED_DELIVERED_AUDIT_RECORDS,
};
pub use store_key::{
    create_store_key_random, load_store_key, load_store_key_from_provider, FileKeyProvider,
    InMemoryKeyProvider, KeyEpochWindow, KeyProvider, KeyProviderQualification,
    LinuxOsKeyStoreProvider, MacOsKeychainProvider, StoreKey,
};
pub use telemetry::{
    AdmissionResult, AuditOutcome, ByteCounter, CacheLifecycle, FallbackReason, HealthComponent,
    LookupResult, OpaqueSpanId, OperationalTelemetry, OtlpExportOutcome, OtlpHttpExporter,
    OtlpSpanExporter, OtlpSpanRecord, ResourceGauge, ServiceComponent, SpanOutcome, TraceCarrier,
    TraceContext, TracePhase, WorkClass, MAX_OTLP_BATCH_SPANS, MAX_OTLP_HTTP_REQUEST_BYTES,
    MAX_OTLP_HTTP_RESPONSE_BYTES, MAX_OTLP_HTTP_TIMEOUT, MAX_PROMETHEUS_TEXT_BYTES,
    MAX_TRACE_BUFFER_SPANS, TRACE_CARRIER_BYTES, TRACE_CARRIER_HEX_BYTES, TRACE_CARRIER_VERSION,
};
#[cfg(test)]
pub(crate) use writer::ArtifactWriter;
pub use writer::{PublishedArtifact, WritePolicy};

pub use kvpack_core as wire;
pub use kvpack_core::{Id32, PackError};

/// Version of bounded-cardinality metric names, labels, and trace field names.
/// This operational schema is independent from all durable wire versions.
pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
extern crate self as kvpack;

#[cfg(test)]
#[path = "../tests/cut_export/main.rs"]
mod cut_export_tests;
#[cfg(test)]
#[path = "../tests/fidelity/main.rs"]
mod fidelity_tests;
#[cfg(test)]
#[path = "../tests/production_v1/main.rs"]
mod production_v1_tests;
#[cfg(test)]
#[path = "../tests/restore_plan/main.rs"]
mod restore_plan_tests;
