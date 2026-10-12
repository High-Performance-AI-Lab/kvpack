use super::*;

mod framing;
mod io;

pub(super) use framing::{
    backup_block_aad, block_nonce, decode_backup_header, decode_inventory_entry,
    decode_inventory_header, encode_backup_header, encode_inventory_entry, encode_inventory_header,
    encode_record_header, expected_backup_bytes, get_u32,
};
pub(super) use io::{
    hash_file, now_ns, read_exact_or_truncated, require_bare_catalog_path, require_new_destination,
    verify_file_mac, write_mac,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum ObjectCheck {
    Present,
    Missing,
    Corrupt,
}

pub(super) fn authenticate_backup(
    config: &StoreConfig,
    key: &StoreKey,
    backup_path: &Path,
    bounds: CatalogBackupBounds,
) -> Result<AuthenticatedBackup, StoreError> {
    let mut file = fs::File::open(backup_path).map_err(io_error("open catalog backup"))?;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect catalog backup"))?;
    if !metadata.is_file() || metadata.len() > bounds.maximum_backup_bytes {
        return Err(StoreError::Quota("catalog backup exceeds its bound"));
    }
    let mut encoded_header = [0u8; BACKUP_HEADER_BYTES];
    read_exact_or_truncated(
        &mut file,
        &mut encoded_header,
        "catalog backup header is truncated",
    )?;
    let header = decode_backup_header(&encoded_header)?;
    if header.catalog_schema != CATALOG_SCHEMA_VERSION as u64
        || header.plaintext_bytes == 0
        || header.plaintext_bytes > bounds.maximum_plaintext_bytes
        || header.block_count == 0
        || header.block_count > bounds.maximum_blocks
        || header.block_count != header.plaintext_bytes.div_ceil(BACKUP_BLOCK_BYTES as u64)
    {
        return Err(StoreError::Codec(
            "catalog backup schema or bounds are invalid",
        ));
    }
    let expected_bytes = expected_backup_bytes(header.plaintext_bytes, header.block_count)?;
    if metadata.len() != expected_bytes {
        return Err(StoreError::Codec(
            "catalog backup has trailing or truncated bytes",
        ));
    }
    let expected_tenant = recovery_tenant(config, key, header.key_epoch)?;
    let schedule = key.schedule(&expected_tenant, header.key_epoch)?;
    let keys = derive_backup_keys(&schedule, &header.salt)?;
    verify_file_mac(
        backup_path,
        expected_bytes,
        BACKUP_MAC_DOMAIN,
        keys.authentication.as_ref(),
    )?;
    validate_snapshot_identity(config, header.tenant, header.catalog_epoch, expected_tenant)?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(keys.encryption.as_ref()));
    Ok(AuthenticatedBackup {
        header,
        encoded_header,
        backup_bytes: expected_bytes,
        cipher,
    })
}

pub(super) fn recovery_tenant(
    config: &StoreConfig,
    key: &StoreKey,
    key_epoch: u64,
) -> Result<Id32, StoreError> {
    if key_epoch < config.minimum_readable_key_epoch || key_epoch > config.key_epoch {
        return Err(StoreError::Authentication(
            "recovery artifact key epoch is outside the readable window",
        ));
    }
    derive_store_tenant_namespace(&config.operator_tenant_id, config.key_epoch, key)
}

pub(super) fn validate_snapshot_identity(
    config: &StoreConfig,
    tenant: Id32,
    catalog_epoch: u64,
    expected_tenant: Id32,
) -> Result<(), StoreError> {
    if tenant != expected_tenant || catalog_epoch != config.catalog_epoch {
        return Err(StoreError::Authentication(
            "recovery artifact tenant or catalog epoch does not match the store",
        ));
    }
    Ok(())
}

pub(super) fn validate_restored_catalog(
    path: &Path,
    config: &StoreConfig,
    header: BackupHeader,
) -> Result<(), StoreError> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.pragma_update(None, "trusted_schema", "OFF")?;
    let integrity: String = connection.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if integrity != "ok" {
        return Err(StoreError::Authentication(
            "restored catalog failed SQLite integrity_check",
        ));
    }
    let foreign_key_error: Option<String> = connection
        .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
        .optional()?;
    if foreign_key_error.is_some() {
        return Err(StoreError::Authentication(
            "restored catalog failed foreign-key validation",
        ));
    }
    let versions = {
        let mut statement =
            connection.prepare("SELECT version FROM schema_migrations ORDER BY version")?;
        let versions = statement
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        versions
    };
    if versions.len() != CATALOG_SCHEMA_VERSION as usize
        || versions
            .iter()
            .enumerate()
            .any(|(index, value)| *value != index as i64 + 1)
    {
        return Err(StoreError::Authentication(
            "restored catalog migration sequence is invalid",
        ));
    }
    let tenants: u64 =
        connection.query_row("SELECT COUNT(*) FROM tenants", [], |row| row.get(0))?;
    if tenants != 1 {
        return Err(StoreError::Authentication(
            "restored catalog tenant cardinality is invalid",
        ));
    }
    let row: (Vec<u8>, u64, u64, u64) = connection.query_row(
        "SELECT namespace,key_epoch,minimum_readable_key_epoch,catalog_epoch FROM tenants",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    if vec_id(row.0)? != header.tenant
        || row.1 != header.key_epoch
        || row.2 > row.1
        || row.3 != header.catalog_epoch
        || row.1 > config.key_epoch
        || row.2 < config.minimum_readable_key_epoch
    {
        return Err(StoreError::Authentication(
            "restored catalog metadata disagrees with its envelope",
        ));
    }
    Ok(())
}

pub(super) fn load_inventory_entries(
    connection: &Connection,
    tenant: &Id32,
    minimum_key_epoch: u64,
    maximum_key_epoch: u64,
    maximum_entries: usize,
) -> Result<Vec<InventoryEntry>, StoreError> {
    if maximum_entries == 0 || maximum_entries >= i64::MAX as usize {
        return Err(StoreError::Expectation(
            "inventory snapshot entry bound is invalid",
        ));
    }
    let query_limit = maximum_entries
        .checked_add(1)
        .ok_or(StoreError::Expectation("inventory entry bound overflow"))?;
    let mut statement = connection.prepare(
        "SELECT kind,object_id,object_digest,object_bytes,generation,key_epoch FROM (
           SELECT 0 AS kind,m.manifest_id AS object_id,m.manifest_id AS object_digest,m.file_bytes AS object_bytes,m.generation,m.key_epoch
           FROM manifests m WHERE m.tenant=?1 AND m.key_epoch BETWEEN ?2 AND ?3 AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=m.tenant AND t.object_kind='manifest' AND t.object_id=m.manifest_id)
           UNION ALL
           SELECT 1 AS kind,c.object_key AS object_id,c.object_digest,c.object_bytes,0 AS generation,c.key_epoch
           FROM chunks c WHERE c.tenant=?1 AND c.key_epoch BETWEEN ?2 AND ?3 AND c.location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)
         ) ORDER BY kind,object_id LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            tenant.as_slice(),
            minimum_key_epoch,
            maximum_key_epoch,
            query_limit as u64,
        ],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, u64>(3)?,
                row.get::<_, u64>(4)?,
                row.get::<_, u64>(5)?,
            ))
        },
    )?;
    let mut entries = Vec::new();
    for row in rows {
        let (kind, object_id, object_digest, object_bytes, generation, key_epoch) = row?;
        let kind = match kind {
            0 => InventoryObjectKind::Manifest,
            1 => InventoryObjectKind::Chunk,
            _ => {
                return Err(StoreError::Authentication(
                    "catalog inventory contains an unknown object kind",
                ));
            }
        };
        entries.push(InventoryEntry {
            kind,
            object_id: vec_id(object_id)?,
            object_digest: vec_id(object_digest)?,
            object_bytes,
            publication_generation: generation,
            key_epoch,
        });
        if entries.len() > maximum_entries {
            return Err(StoreError::Quota(
                "catalog inventory exceeds its entry bound",
            ));
        }
    }
    Ok(entries)
}

pub(super) fn derive_backup_keys(
    schedule: &kvpack_core::KeySchedule,
    salt: &[u8; 16],
) -> Result<BackupKeys, StoreError> {
    let encryption = Hkdf::<Sha256>::new(Some(salt), schedule.manifest_encryption_key());
    let authentication = Hkdf::<Sha256>::new(Some(salt), schedule.manifest_auth_key());
    let mut encryption_key = Zeroizing::new([0u8; 32]);
    let mut authentication_key = Zeroizing::new([0u8; 32]);
    encryption
        .expand(
            b"kvpack/catalog-backup/v1/encryption",
            encryption_key.as_mut(),
        )
        .map_err(|_| StoreError::Authentication("catalog backup key derivation failed"))?;
    authentication
        .expand(
            b"kvpack/catalog-backup/v1/authentication",
            authentication_key.as_mut(),
        )
        .map_err(|_| StoreError::Authentication("catalog backup key derivation failed"))?;
    Ok(BackupKeys {
        encryption: encryption_key,
        authentication: authentication_key,
    })
}

pub(super) fn derive_inventory_key(
    schedule: &kvpack_core::KeySchedule,
    salt: &[u8; 16],
) -> Result<SecretKey, StoreError> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), schedule.manifest_auth_key());
    let mut key = Zeroizing::new([0u8; 32]);
    hkdf.expand(b"kvpack/inventory-snapshot/v1/authentication", key.as_mut())
        .map_err(|_| StoreError::Authentication("inventory key derivation failed"))?;
    Ok(key)
}

pub(super) fn validate_backup_bounds(bounds: CatalogBackupBounds) -> Result<(), StoreError> {
    if bounds.maximum_plaintext_bytes == 0
        || bounds.maximum_backup_bytes <= BACKUP_HEADER_BYTES as u64 + SIGNATURE_BYTES
        || bounds.maximum_blocks == 0
    {
        return Err(StoreError::Expectation("catalog backup bounds are invalid"));
    }
    Ok(())
}

pub(super) fn validate_inventory_bounds(bounds: InventorySnapshotBounds) -> Result<(), StoreError> {
    if bounds.maximum_entries == 0
        || bounds.maximum_snapshot_bytes < INVENTORY_HEADER_BYTES as u64 + SIGNATURE_BYTES
    {
        return Err(StoreError::Expectation(
            "inventory snapshot bounds are invalid",
        ));
    }
    Ok(())
}

pub(super) struct TemporaryFile {
    pub(super) path: PathBuf,
    armed: bool,
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(super) fn create_private_temporary(
    parent: &Path,
    label: &str,
    extension: &str,
) -> Result<(TemporaryFile, fs::File), StoreError> {
    let metadata = fs::metadata(parent).map_err(io_error("inspect recovery temporary parent"))?;
    if !metadata.is_dir() {
        return Err(StoreError::State(
            "recovery temporary parent is not a directory",
        ));
    }
    for _ in 0..16 {
        let mut random = [0u8; 16];
        getrandom::fill(&mut random)
            .map_err(|_| StoreError::State("recovery filename entropy failed"))?;
        let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let path = parent.join(format!(".{label}-{suffix}.{extension}"));
        let mut options = fs::OpenOptions::new();
        options.write(true).read(true).create_new(true).mode(0o600);
        match options.open(&path) {
            Ok(file) => {
                return Ok((TemporaryFile { path, armed: true }, file));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(StoreError::Io {
                    op: "create private recovery temporary",
                    source,
                });
            }
        }
    }
    Err(StoreError::Busy)
}

pub(super) fn publish_temporary(
    temporary: &mut TemporaryFile,
    destination: &Path,
) -> Result<(), StoreError> {
    fs::hard_link(&temporary.path, destination).map_err(|source| StoreError::Io {
        op: "publish create-new recovery artifact",
        source,
    })?;
    fs::remove_file(&temporary.path).map_err(io_error("unlink recovery temporary"))?;
    temporary.armed = false;
    fsync_dir(
        destination
            .parent()
            .ok_or(StoreError::State("recovery destination has no parent"))?,
    )
}
