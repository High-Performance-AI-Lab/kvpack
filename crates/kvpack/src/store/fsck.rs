use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use kvpack_core::{
    decode_authenticated_pack, decode_chunk, inspect_pack_header, ChunkObject, EncodedPack,
    ValidationContext,
};

use crate::error::io_error;
use crate::{StoreConfig, StoreError, StoreKey};

use super::{fsync_dir, LocalStore, UploadReservation};

#[derive(Debug, Clone, Copy)]
pub struct FsckBounds {
    pub maximum_manifests: usize,
    pub maximum_chunks: usize,
    pub maximum_scan_bytes: u64,
}
impl Default for FsckBounds {
    fn default() -> Self {
        Self {
            maximum_manifests: 10_000_000,
            maximum_chunks: 100_000_000,
            maximum_scan_bytes: 1024 * 1024 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsckReport {
    pub manifests: usize,
    pub chunks: usize,
    pub scanned_bytes: u64,
}

impl LocalStore {
    /// Bounded offline payload scan.  A complete replacement catalog is built
    /// beside the old database, checkpointed, then atomically renamed.  The
    /// former DB/WAL/SHM are retained as timestamped recovery evidence.
    pub fn fsck_rebuild_catalog(
        mut config: StoreConfig,
        key: StoreKey,
        bounds: FsckBounds,
        context: ValidationContext,
    ) -> Result<FsckReport, StoreError> {
        let parent = config
            .catalog_path
            .parent()
            .ok_or(StoreError::State("catalog path has no parent"))?
            .to_path_buf();
        let mut nonce = [0u8; 8];
        getrandom::fill(&mut nonce)
            .map_err(|_| StoreError::State("fsck filename entropy failed"))?;
        let suffix: String = nonce.iter().map(|byte| format!("{byte:02x}")).collect();
        let original_catalog = config.catalog_path.clone();
        let scratch = parent.join(format!(".kvpack-catalog-rebuild-{suffix}.sqlite"));
        config.catalog_path = scratch.clone();
        config.quota_bytes = i64::MAX as u64;
        config.staging_quota_bytes = i64::MAX as u64;
        config.endurance_bytes_per_five_minutes = i64::MAX as u64;
        let store = Arc::new(LocalStore::open(config.clone(), key)?);
        let mut manifests = Vec::new();
        let mut scanned = 0u64;
        for path in object_files(
            &config.object_root.join("manifests"),
            ".kvpack",
            bounds.maximum_manifests,
        )? {
            let bytes = bounded_read(&path, bounds.maximum_scan_bytes.saturating_sub(scanned))?;
            scanned = scanned
                .checked_add(bytes.len() as u64)
                .ok_or(StoreError::State("fsck scan byte overflow"))?;
            if scanned > bounds.maximum_scan_bytes {
                return Err(StoreError::Quota("fsck scan byte bound exceeded"));
            }
            let header = inspect_pack_header(&bytes)?;
            if header.tenant_namespace != store.tenant_namespace() {
                continue;
            }
            let keys = store.schedule(header.key_epoch)?;
            let manifest = decode_authenticated_pack(&bytes, &keys, &context)?;
            let id = kvpack_core::manifest_id(&manifest.encode_canonical()?);
            manifests.push((manifest.realized_schema.kind.depth(), id, bytes, manifest));
        }
        manifests.sort_by_key(|entry| (entry.0, entry.1));
        let mut chunks_seen = BTreeSet::new();
        for (_, manifest_id, pack_bytes, manifest) in &manifests {
            for (state, schema) in manifest.states.iter().zip(&manifest.realized_schema.states) {
                for (reference, span) in state.chunks.iter().zip(&schema.chunk_spans) {
                    if chunks_seen.insert(reference.object_key) {
                        if chunks_seen.len() > bounds.maximum_chunks {
                            return Err(StoreError::Quota("fsck chunk count bound exceeded"));
                        }
                        let bytes = bounded_read(
                            &store.chunk_path(&reference.object_key),
                            bounds.maximum_scan_bytes.saturating_sub(scanned),
                        )?;
                        scanned = scanned
                            .checked_add(bytes.len() as u64)
                            .ok_or(StoreError::State("fsck scan byte overflow"))?;
                        let keys = store.schedule(reference.key_epoch)?;
                        decode_chunk(
                            &bytes,
                            reference,
                            span,
                            &manifest.tenant_namespace,
                            &manifest.family,
                            &state.key,
                            &keys,
                        )?;
                        store.put_chunk(
                            &ChunkObject {
                                chunk_id: reference.chunk_id,
                                object_key: reference.object_key,
                                object_digest: reference.object_digest,
                                plaintext_bytes: reference.plaintext_bytes,
                                bytes,
                            },
                            reference.key_epoch,
                        )?;
                    }
                }
            }
            let expected = manifest
                .states
                .iter()
                .flat_map(|state| &state.chunks)
                .map(|chunk| chunk.object_bytes as u64)
                .sum::<u64>()
                .saturating_add(pack_bytes.len() as u64)
                .saturating_add(4096);
            store.reserve_upload(
                manifest_id,
                UploadReservation {
                    expected_bytes: expected,
                    publication_generation: 1,
                    intent_digest: *manifest_id,
                    retention: super::RetentionInputs::conservative(expected, 1),
                },
            )?;
            store.mark_receiving(manifest_id)?;
            let exact_node = kvpack_core::PrefixNode {
                token_count: manifest.input_cut.token_count,
                id: manifest.input_cut.token_root,
                reusable: manifest.input_cut.token_count % kvpack_core::PREFIX_BLOCK_TOKENS as u64
                    == 0,
            };
            store.publish_manifest(
                manifest_id,
                &EncodedPack {
                    bytes: pack_bytes.clone(),
                    manifest_id: *manifest_id,
                },
                manifest,
                std::slice::from_ref(&exact_node),
            )?;
        }
        {
            let connection = store.lock_catalog()?;
            connection
                .execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;")?;
        }
        let report = FsckReport {
            manifests: manifests.len(),
            chunks: chunks_seen.len(),
            scanned_bytes: scanned,
        };
        drop(store);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        for suffix_name in ["", "-wal", "-shm"] {
            let source = PathBuf::from(format!(
                "{}{suffix_name}",
                original_catalog.to_string_lossy()
            ));
            if source.exists() {
                let backup = PathBuf::from(format!(
                    "{}.backup-{timestamp}{suffix_name}",
                    original_catalog.to_string_lossy()
                ));
                fs::rename(source, backup).map_err(io_error("backup old fsck catalog"))?;
            }
        }
        fs::rename(&scratch, &original_catalog)
            .map_err(io_error("publish rebuilt fsck catalog"))?;
        fsync_dir(&parent)?;
        Ok(report)
    }
}

fn object_files(
    root: &std::path::Path,
    extension: &str,
    bound: usize,
) -> Result<Vec<PathBuf>, StoreError> {
    let mut result = Vec::new();
    let expected_extension = extension.trim_start_matches('.');
    for shard in fs::read_dir(root).map_err(io_error("scan fsck object root"))? {
        let shard = shard.map_err(io_error("scan fsck shard"))?;
        if !shard
            .file_type()
            .map_err(io_error("inspect fsck shard"))?
            .is_dir()
        {
            continue;
        }
        for entry in fs::read_dir(shard.path()).map_err(io_error("scan fsck shard entries"))? {
            let entry = entry.map_err(io_error("scan fsck object"))?;
            if entry
                .file_type()
                .map_err(io_error("inspect fsck object"))?
                .is_file()
                && entry.path().extension().and_then(|value| value.to_str())
                    == Some(expected_extension)
            {
                result.push(entry.path());
                if result.len() > bound {
                    return Err(StoreError::Quota("fsck object count bound exceeded"));
                }
            }
        }
    }
    result.sort();
    Ok(result)
}
fn bounded_read(path: &std::path::Path, remaining: u64) -> Result<Vec<u8>, StoreError> {
    let mut file = fs::File::open(path).map_err(io_error("open fsck object"))?;
    let metadata = file.metadata().map_err(io_error("inspect fsck object"))?;
    if metadata.len() > remaining {
        return Err(StoreError::Quota("fsck scan byte bound exceeded"));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    (&mut file)
        .take(remaining.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(io_error("read fsck object"))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(StoreError::Authentication(
            "fsck object changed during scan",
        ));
    }
    Ok(bytes)
}
