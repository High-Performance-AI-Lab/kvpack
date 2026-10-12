use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::sync::Arc;

use kvpack_core::{
    decode_authenticated_pack, decode_chunk, inspect_pack_header, validate_manifest, ChunkRef,
    ChunkSpan, CutManifest, Id32, RepresentationFamilyId, StateKey, ValidationContext,
    MAX_MANIFEST_BYTES, PACK_FOOTER_BYTES, PACK_HEADER_BYTES,
};
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use rustix::fs::{Mode, OFlags};
use sha2::{Digest, Sha256};

use crate::error::io_error;
use crate::StoreError;

use super::remote::MAX_SOURCE_LEASE_NS;
use super::{fsync_dir, hex, InventoryObjectKind, LocalStore, SourceLeaseState};

mod chunks;
pub mod direct;
mod manifests;
mod pins;
mod reconcile;

pub use chunks::{AuthenticatedPublicationChunk, AuthenticatedPublicationSource};
pub(crate) use manifests::ManifestLru;
pub(crate) use pins::{release_restore_pin_batch, RetainedPin};

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

fn random_nonzero_id(message: &'static str) -> Result<Id32, StoreError> {
    let mut id = [0u8; 32];
    getrandom::fill(&mut id).map_err(|_| StoreError::State(message))?;
    if id == [0; 32] {
        id[31] = 1;
    }
    Ok(id)
}

fn process_identity() -> Id32 {
    use sha2::{Digest, Sha256};
    static ID: std::sync::OnceLock<Id32> = std::sync::OnceLock::new();
    *ID.get_or_init(|| {
        let mut nonce = [0u8; 32];
        let _ = getrandom::fill(&mut nonce);
        let mut hash = Sha256::new();
        hash.update(b"kvpack/v1/process-instance\0");
        hash.update(std::process::id().to_le_bytes());
        hash.update(nonce);
        hash.finalize().into()
    })
}
