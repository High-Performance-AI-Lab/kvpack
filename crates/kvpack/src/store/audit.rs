use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::MutexGuard;
use std::time::{Duration, SystemTime};

use kvpack_core::Id32;
use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};

use crate::error::io_error;
use crate::telemetry::AuditOutcome;
use crate::StoreError;

use super::{create_private_dir, fsync_dir, hex, vec_id, LocalStore};

mod outbox;
mod segments;

#[cfg(test)]
use outbox::load_batch;
pub(super) use outbox::{append_events, preflight_events};
#[cfg(test)]
use segments::MIN_AUDIT_SEGMENT_BYTES;
pub use segments::{AuditDirectoryExporter, AuditDirectoryPolicy};

pub const AUDIT_SCHEMA_VERSION: u32 = 1;
pub const MAX_AUDIT_BATCH_RECORDS: usize = 512;
pub const MAX_PENDING_AUDIT_RECORDS: u64 = 65_536;
pub const MAX_RETAINED_DELIVERED_AUDIT_RECORDS: u64 = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditEventKind {
    Reserved,
    Receiving,
    Verified,
    Published,
    Aborted,
    Quarantined,
    Tombstoned,
    Collected,
}

impl AuditEventKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Reserved => "reserved",
            Self::Receiving => "receiving",
            Self::Verified => "verified",
            Self::Published => "published",
            Self::Aborted => "aborted",
            Self::Quarantined => "quarantined",
            Self::Tombstoned => "tombstoned",
            Self::Collected => "collected",
        }
    }

    fn parse(value: &str) -> Result<Self, StoreError> {
        match value {
            "reserved" => Ok(Self::Reserved),
            "receiving" => Ok(Self::Receiving),
            "verified" => Ok(Self::Verified),
            "published" => Ok(Self::Published),
            "aborted" => Ok(Self::Aborted),
            "quarantined" => Ok(Self::Quarantined),
            "tombstoned" => Ok(Self::Tombstoned),
            "collected" => Ok(Self::Collected),
            _ => Err(StoreError::State("catalog contains an unknown audit event")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditObjectKind {
    Upload,
    Prefix,
    Manifest,
    Chunk,
    Quarantine,
}

impl AuditObjectKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Prefix => "prefix",
            Self::Manifest => "manifest",
            Self::Chunk => "chunk",
            Self::Quarantine => "quarantine",
        }
    }

    fn parse(value: &str) -> Result<Self, StoreError> {
        match value {
            "upload" => Ok(Self::Upload),
            "prefix" => Ok(Self::Prefix),
            "manifest" => Ok(Self::Manifest),
            "chunk" => Ok(Self::Chunk),
            "quarantine" => Ok(Self::Quarantine),
            _ => Err(StoreError::State(
                "catalog contains an unknown audit object kind",
            )),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AuditEventKey {
    event: AuditEventKind,
    object: AuditObjectKind,
    object_id: Id32,
    generation: u64,
}

impl AuditEventKey {
    pub(super) const fn new(
        event: AuditEventKind,
        object: AuditObjectKind,
        object_id: Id32,
        generation: u64,
    ) -> Self {
        Self {
            event,
            object,
            object_id,
            generation,
        }
    }
}

impl fmt::Debug for AuditEventKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuditEventKey")
            .field("event", &self.event)
            .field("object", &self.object)
            .field("object_id", &"[opaque]")
            .field("generation", &self.generation)
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AuditRecord {
    sequence: u64,
    event: AuditEventKind,
    object: AuditObjectKind,
    object_id: Id32,
    generation: u64,
    occurred_unix_ns: u64,
}

impl AuditRecord {
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn event(&self) -> AuditEventKind {
        self.event
    }

    pub const fn object(&self) -> AuditObjectKind {
        self.object
    }

    pub const fn object_id(&self) -> &Id32 {
        &self.object_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn occurred_unix_ns(&self) -> u64 {
        self.occurred_unix_ns
    }
}

impl fmt::Debug for AuditRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuditRecord")
            .field("sequence", &self.sequence)
            .field("event", &self.event)
            .field("object", &self.object)
            .field("object_id", &"[opaque]")
            .field("generation", &self.generation)
            .field("occurred_unix_ns", &self.occurred_unix_ns)
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AuditBatch {
    records: Vec<AuditRecord>,
}

impl AuditBatch {
    pub fn records(&self) -> &[AuditRecord] {
        &self.records
    }

    pub fn first_sequence(&self) -> u64 {
        self.records.first().map_or(0, AuditRecord::sequence)
    }

    pub fn last_sequence(&self) -> u64 {
        self.records.last().map_or(0, AuditRecord::sequence)
    }
}

impl fmt::Debug for AuditBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuditBatch")
            .field("records", &self.records.len())
            .field("first_sequence", &self.first_sequence())
            .field("last_sequence", &self.last_sequence())
            .finish()
    }
}

pub trait AuditExporter: Send + Sync {
    /// Persist one ordered batch. Implementations must make exact retries
    /// idempotent because a crash can occur after export and before catalog ack.
    fn export(&self, batch: &AuditBatch) -> Result<(), StoreError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditStatus {
    pub pending_records: u64,
    pub retained_delivered_records: u64,
    pub next_sequence: u64,
    pub last_flushed_unix_ns: u64,
    pub backpressure_events: u64,
    pub delivery_failures: u64,
    pub retention_pruned_records: u64,
    pub lost_records: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditExportReport {
    pub exported_records: u64,
    pub retention_pruned_records: u64,
    pub status: AuditStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AuditCapacity {
    Ready,
    Backpressured,
}

#[cfg(test)]
include!("audit/tests/inline.rs");
