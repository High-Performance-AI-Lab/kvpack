macro_rules! closed_labels {
    ($name:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub const COUNT: usize = Self::ALL.len();

            pub const fn label(self) -> &'static str {
                match self {
                    $(Self::$variant => $label),+
                }
            }
        }
    };
}

closed_labels!(AdmissionResult {
    Admitted => "admitted",
    Queued => "queued",
    RejectedCapacity => "rejected_capacity",
    RejectedPolicy => "rejected_policy",
    Cancelled => "cancelled",
});

closed_labels!(WorkClass {
    Demand => "demand",
    Publication => "publication",
    Maintenance => "maintenance",
});

closed_labels!(LookupResult {
    Exact => "exact",
    Ancestor => "ancestor",
    Miss => "miss",
    Stale => "stale",
    Unavailable => "unavailable",
    IntegrityError => "integrity_error",
});

closed_labels!(ByteCounter {
    SourceRead => "source_read",
    DurableWritten => "durable_written",
    Restored => "restored",
    Transferred => "transferred",
    Recomputed => "recomputed",
});

closed_labels!(TracePhase {
    Request => "request",
    ExactLookup => "exact_lookup",
    Plan => "plan",
    Transfer => "transfer",
    Verification => "verification",
    SuffixWork => "suffix_work",
    Install => "install",
    Publication => "publication",
    Release => "release",
});

closed_labels!(ResourceGauge {
    DurableBytes => "durable_bytes",
    StagingReservedBytes => "staging_reserved_bytes",
    AdmissionReservedBytes => "admission_reserved_bytes",
    QuarantineBytes => "quarantine_bytes",
    InFlightBytes => "in_flight_bytes",
    CatalogPins => "catalog_pins",
    RestorePins => "restore_pins",
    Grants => "grants",
    ActiveRestores => "active_restores",
    OpenDescriptors => "open_descriptors",
});

closed_labels!(CacheLifecycle {
    Reserved => "reserved",
    Receiving => "receiving",
    Verified => "verified",
    Published => "published",
    Aborted => "aborted",
    Quarantined => "quarantined",
    Tombstoned => "tombstoned",
    Collected => "collected",
});

closed_labels!(HealthComponent {
    Store => "store",
    Catalog => "catalog",
    Agent => "agent",
    Gateway => "gateway",
    ObjectTier => "object_tier",
    Transport => "transport",
});

closed_labels!(FallbackReason {
    NoExactCut => "no_exact_cut",
    StaleLocation => "stale_location",
    ResourceInfeasible => "resource_infeasible",
    DataPlaneUnavailable => "data_plane_unavailable",
    CapabilityExpired => "capability_expired",
    CatalogUnavailable => "catalog_unavailable",
});

closed_labels!(ServiceComponent {
    Store => "kvpack-store",
    Agent => "kvpack-agent",
    Gateway => "kvpack-gateway",
    CacheControlPlane => "kvpack-cache",
    Engine => "inference-engine",
});

closed_labels!(SpanOutcome {
    Ok => "ok",
    Miss => "miss",
    Rejected => "rejected",
    Unavailable => "unavailable",
    Cancelled => "cancelled",
    IntegrityError => "integrity_error",
});

closed_labels!(AuditOutcome {
    Enqueued => "enqueued",
    Exported => "exported",
    ExportRetry => "export_retry",
    Backpressure => "backpressure",
    RetentionPruned => "retention_pruned",
    Lost => "lost",
});
