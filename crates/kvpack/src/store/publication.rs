use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use kvpack_core::{ChunkObject, ChunkRef, CutManifest, EncodedPack, Id32, PrefixNode};
use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};

use crate::error::io_error;
use crate::telemetry::{
    AuditOutcome, ByteCounter, CacheLifecycle, ServiceComponent, SpanOutcome, TraceContext,
    TracePhase,
};
use crate::{PublishedArtifact, StoreError};

use super::{
    audit::{self, AuditCapacity, AuditEventKey},
    family_digest, fsync_dir, hex, semantic_digest, vec_id, AdmissionDecision, AuditEventKind,
    AuditObjectKind, LocalStore, PendingManifest, QuarantinedUploadFile, UploadReservation,
    UploadState, UtilizationPolicy,
};

type StoredChunkRow = (Vec<u8>, Vec<u8>, u64, u32, u32);
type StoredManifestRow = (u64, u64, u64, Vec<u8>, Vec<u8>, u64, Option<Vec<u8>>, u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DurableObjectKind {
    Chunk,
    Manifest,
    UploadManifest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ImmutableFaultPhase {
    Create,
    Write,
    FileSync,
    NoReplace,
    TargetDirectorySync,
    PartialDirectorySync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DurabilityFaultPoint {
    Immutable(DurableObjectKind, ImmutableFaultPhase),
    CatalogBegin,
    CatalogCommit,
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

mod batch;
mod chunks;
mod immutable;
mod manifests;
mod uploads;
use batch::validate_and_write_manifest_batch;
use immutable::{durability_fault, quarantine_object, transition};
pub(super) use immutable::{write_immutable, AlreadyVerifiedTarget};

#[cfg(test)]
include!("publication/tests/inline.rs");
