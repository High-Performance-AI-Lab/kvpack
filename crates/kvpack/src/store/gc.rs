use std::fs;
use std::path::{Component, Path};

use rusqlite::{params, OptionalExtension, TransactionBehavior};

use crate::telemetry::AuditOutcome;
use crate::{CacheLifecycle, StoreError};

use super::{
    audit::{self, AuditCapacity, AuditEventKey},
    fsync_dir, hex, vec_id, AdmissionDecision, AuditEventKind, AuditObjectKind, LocalStore,
};

mod capacity;
mod demotion;
mod eviction;
mod quarantine;

pub use capacity::{CachePressure, EvictionReport, UtilizationPolicy};
pub use demotion::DemotionReport;

/// Victim query bound per eviction round: one bounded query plus one catalog
/// transaction per phase per batch instead of two transactions per object.
const EVICTION_VICTIM_BATCH: usize = 256;

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
