use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use kvpack_core::{
    decode_authenticated_pack, inspect_pack_header, manifest_id, Id32, ValidationContext,
};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::io_error;
use crate::{StoreConfig, StoreError, StoreKey};

use super::{
    create_private_dir, derive_store_tenant_namespace, fsync_dir, vec_id, InventoryCursor,
    InventoryEntry, InventoryObjectKind, LocalStore, CATALOG_SCHEMA_VERSION,
};

type HmacSha256 = Hmac<Sha256>;
type SecretKey = Zeroizing<[u8; 32]>;

const BACKUP_MAGIC: &[u8; 8] = b"KVCATB01";
const INVENTORY_MAGIC: &[u8; 8] = b"KVINVS01";
const RECOVERY_VERSION: u16 = 1;
const BACKUP_FLAGS: u16 = 0x0003;
const INVENTORY_FLAGS: u16 = 0x0001;
const BACKUP_HEADER_BYTES: usize = 192;
const INVENTORY_HEADER_BYTES: usize = 176;
const INVENTORY_ENTRY_BYTES: usize = 96;
const SIGNATURE_BYTES: u64 = 32;
const BACKUP_BLOCK_BYTES: usize = 4 * 1024 * 1024;
const AEAD_TAG_BYTES: u64 = 16;
const RECORD_HEADER_BYTES: u64 = 8;
const BACKUP_MAC_DOMAIN: &[u8] = b"kvpack/catalog-backup/v1/mac\0";
const BACKUP_AEAD_DOMAIN: &[u8] = b"kvpack/catalog-backup/v1/block\0";
const INVENTORY_MAC_DOMAIN: &[u8] = b"kvpack/inventory-snapshot/v1/mac\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogBackupBounds {
    pub maximum_plaintext_bytes: u64,
    pub maximum_backup_bytes: u64,
    pub maximum_blocks: u64,
}

impl Default for CatalogBackupBounds {
    fn default() -> Self {
        Self {
            maximum_plaintext_bytes: 1024 * 1024 * 1024 * 1024,
            maximum_backup_bytes: 1025 * 1024 * 1024 * 1024,
            maximum_blocks: 262_144,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogBackupReport {
    pub catalog_schema: u64,
    pub catalog_epoch: u64,
    pub key_epoch: u64,
    pub created_ns: u64,
    pub plaintext_bytes: u64,
    pub backup_bytes: u64,
    pub block_count: u64,
    pub plaintext_digest: Id32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventorySnapshotBounds {
    pub maximum_entries: usize,
    pub maximum_snapshot_bytes: u64,
}

impl Default for InventorySnapshotBounds {
    fn default() -> Self {
        Self {
            maximum_entries: 10_000_000,
            maximum_snapshot_bytes: 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventorySnapshotReport {
    pub catalog_schema: u64,
    pub catalog_epoch: u64,
    pub key_epoch: u64,
    pub created_ns: u64,
    pub entries: u64,
    pub snapshot_bytes: u64,
    pub payload_digest: Id32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedInventorySnapshot {
    pub report: InventorySnapshotReport,
    pub entries: Vec<InventoryEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconciliationBounds {
    pub maximum_objects: usize,
    pub maximum_scan_bytes: u64,
}

impl Default for ReconciliationBounds {
    fn default() -> Self {
        Self {
            maximum_objects: 110_000_000,
            maximum_scan_bytes: 1024 * 1024 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogReconciliationReport {
    pub catalog_objects: u64,
    pub present_objects: u64,
    pub checked_bytes: u64,
    pub missing_objects: Vec<InventoryCursor>,
    pub corrupt_objects: Vec<InventoryCursor>,
}

#[derive(Debug, Clone, Copy)]
struct BackupHeader {
    tenant: Id32,
    catalog_schema: u64,
    catalog_epoch: u64,
    key_epoch: u64,
    created_ns: u64,
    plaintext_bytes: u64,
    block_count: u64,
    plaintext_digest: Id32,
    salt: [u8; 16],
    nonce_prefix: [u8; 4],
}

#[derive(Debug, Clone, Copy)]
struct InventoryHeader {
    tenant: Id32,
    catalog_schema: u64,
    catalog_epoch: u64,
    key_epoch: u64,
    created_ns: u64,
    entry_count: u64,
    payload_bytes: u64,
    payload_digest: Id32,
    salt: [u8; 16],
}

struct BackupKeys {
    encryption: SecretKey,
    authentication: SecretKey,
}

struct AuthenticatedBackup {
    header: BackupHeader,
    encoded_header: [u8; BACKUP_HEADER_BYTES],
    backup_bytes: u64,
    cipher: ChaCha20Poly1305,
}

mod backup;
mod helpers;
mod inventory;
mod reconcile;

use helpers::{
    authenticate_backup, backup_block_aad, block_nonce, create_private_temporary,
    decode_inventory_entry, decode_inventory_header, derive_backup_keys, derive_inventory_key,
    encode_backup_header, encode_inventory_entry, encode_inventory_header, encode_record_header,
    expected_backup_bytes, get_u32, hash_file, load_inventory_entries, now_ns, publish_temporary,
    read_exact_or_truncated, recovery_tenant, require_bare_catalog_path, require_new_destination,
    validate_backup_bounds, validate_inventory_bounds, validate_restored_catalog,
    validate_snapshot_identity, verify_file_mac, write_mac, ObjectCheck,
};
