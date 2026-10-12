use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use kvpack_core::{ChunkObject, ChunkRef, Id32};
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use crate::error::io_error;
use crate::StoreError;

use super::{fsync_dir, hex, vec_id, LocalStore, RetentionInputs};

mod promotion;
mod reconcile;
mod session;
mod staging;

/// Lease attached to every provisional upload session. A producer dying
/// mid-upload stops refreshing the lease; once it lapses, `reserve_upload`
/// and `stat` reap the reservation exactly like an abort.
pub(crate) const PROVISIONAL_UPLOAD_LEASE_NS: u64 = 15 * 60 * 1_000_000_000;

/// Per-directory entry bound applied during restart reconciliation. A
/// maximum bundle stages 32,768 / 256 prefix blocks x 24 layers x 2 states
/// = 3,072 chunk files, so 8,192 leaves ample headroom.
pub(crate) const PROVISIONAL_DIRECTORY_ENTRY_BOUND: usize = 8_192;

/// Provenance captured when a provisional export session begins. It is
/// persisted on the upload row and hashed into the seal digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvisionalProvenance {
    pub source_wall_clock_ns: u64,
    pub clock_offset_ns: Option<u64>,
    pub quiesced: bool,
}

/// Seal-time metadata persisted on the upload row, retrievable by the
/// decode side after publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvisionalUploadMetadata {
    pub boundary_token_id: Option<u32>,
    pub provenance: ProvisionalProvenance,
}

#[derive(Debug, Clone)]
pub(crate) struct ProvisionalStagedChunk {
    pub reference: ChunkRef,
    pub staged_path: Option<PathBuf>,
    pub staged_bytes: u64,
    pub deduplicated_bytes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct ProvisionalPromotedChunk {
    pub object_key: Id32,
    pub staged_path: Option<PathBuf>,
    pub created_target: bool,
}

/// Disposition of a provisional staging batch, resolved once per batch by
/// `begin_provisional_stage_batch` and re-validated under the write
/// transaction by `commit_provisional_stage_batch`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProvisionalStageMode {
    /// The upload is already published: every chunk validates against the
    /// published catalog and stages nothing.
    PublishedReplay,
    /// The upload is receiving: ordinals below the durable cursor validate
    /// against the chunk ledger; ordinals from the cursor on stage new bytes
    /// and ledger rows.
    Receiving { cursor: u64 },
}

fn verify_file_bytes(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let metadata = fs::symlink_metadata(path).map_err(io_error("inspect immutable collision"))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != bytes.len() as u64
    {
        return Err(StoreError::Authentication(
            "immutable collision metadata mismatch",
        ));
    }
    let mut file = File::open(path).map_err(io_error("open immutable collision"))?;
    let mut offset = 0usize;
    let mut buffer = [0u8; 64 * 1024];
    while offset < bytes.len() {
        let count = file
            .read(&mut buffer)
            .map_err(io_error("read immutable collision"))?;
        if count == 0 || bytes[offset..].get(..count) != Some(&buffer[..count]) {
            return Err(StoreError::Authentication(
                "immutable collision bytes mismatch",
            ));
        }
        offset += count;
    }
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(io_error("finish immutable collision read"))?
        != 0
    {
        return Err(StoreError::Authentication(
            "immutable collision length changed",
        ));
    }
    Ok(())
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

fn new_session_token() -> Result<u64, StoreError> {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes)
        .map_err(|_| StoreError::State("provisional session token generation failed"))?;
    // SQLite INTEGER is signed; keep the token in 1..=i64::MAX.
    Ok((u64::from_le_bytes(bytes) & (i64::MAX as u64)).max(1))
}

fn reject_expired_lease(lease_expires_ns: i64) -> Result<(), StoreError> {
    if lease_expires_ns != 0 && now_ns() > lease_expires_ns as u64 {
        return Err(StoreError::State("provisional upload lease expired"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "provisional_tests.rs"]
mod tests;
