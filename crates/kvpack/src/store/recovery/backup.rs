use super::*;

impl LocalStore {
    /// Create a consistent online SQLite snapshot and publish an encrypted,
    /// authenticated backup without buffering the catalog in memory. The
    /// destination is create-new and the only plaintext temporary lives beside
    /// the internal catalog.
    pub fn write_catalog_backup(
        &self,
        destination: &Path,
        bounds: CatalogBackupBounds,
    ) -> Result<CatalogBackupReport, StoreError> {
        validate_backup_bounds(bounds)?;
        require_new_destination(destination)?;
        let catalog_parent = self
            .config
            .catalog_path
            .parent()
            .ok_or(StoreError::State("catalog path has no parent"))?;
        let (snapshot_guard, snapshot_file) =
            create_private_temporary(catalog_parent, "catalog-backup-plaintext", "sqlite")?;
        drop(snapshot_file);
        {
            // A WAL snapshot does not need the process-wide catalog mutex.
            // Holding that mutex across SQLite's complete backup loop stalls
            // exact lookup/pin traffic behind backup I/O. A separate read-only
            // connection gets the same transactionally consistent snapshot
            // while leaving the live connection available to demand work.
            let source = self
                .backup_catalog
                .lock()
                .map_err(|_| StoreError::State("backup catalog mutex poisoned"))?;
            let mut destination_connection = Connection::open(&snapshot_guard.path)?;
            let backup = rusqlite::backup::Backup::new(&source, &mut destination_connection)?;
            backup.run_to_completion(256, Duration::ZERO, None)?;
            drop(backup);
            destination_connection.execute_batch(
                "PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA optimize;",
            )?;
        }
        fs::File::open(&snapshot_guard.path)
            .and_then(|file| file.sync_all())
            .map_err(io_error("fsync plaintext catalog snapshot"))?;
        let (plaintext_bytes, plaintext_digest) =
            hash_file(&snapshot_guard.path, bounds.maximum_plaintext_bytes)?;
        if plaintext_bytes == 0 {
            return Err(StoreError::Authentication(
                "catalog backup snapshot is empty",
            ));
        }
        let block_count = plaintext_bytes.div_ceil(BACKUP_BLOCK_BYTES as u64);
        if block_count == 0 || block_count > bounds.maximum_blocks {
            return Err(StoreError::Quota(
                "catalog backup block count exceeds its bound",
            ));
        }
        let backup_bytes = expected_backup_bytes(plaintext_bytes, block_count)?;
        if backup_bytes > bounds.maximum_backup_bytes {
            return Err(StoreError::Quota(
                "encrypted catalog backup exceeds its bound",
            ));
        }
        let mut salt = [0u8; 16];
        let mut nonce_prefix = [0u8; 4];
        getrandom::fill(&mut salt)
            .map_err(|_| StoreError::State("catalog backup salt generation failed"))?;
        getrandom::fill(&mut nonce_prefix)
            .map_err(|_| StoreError::State("catalog backup nonce generation failed"))?;
        let header = BackupHeader {
            tenant: self.tenant_namespace,
            catalog_schema: CATALOG_SCHEMA_VERSION as u64,
            catalog_epoch: self.catalog_epoch(),
            key_epoch: self.key_epoch(),
            created_ns: now_ns(),
            plaintext_bytes,
            block_count,
            plaintext_digest,
            salt,
            nonce_prefix,
        };
        let encoded_header = encode_backup_header(header);
        let schedule = self.schedule(header.key_epoch)?;
        let keys = derive_backup_keys(&schedule, &salt)?;
        let cipher = ChaCha20Poly1305::new(Key::from_slice(keys.encryption.as_ref()));
        let mut mac = <HmacSha256 as Mac>::new_from_slice(keys.authentication.as_ref())
            .map_err(|_| StoreError::Authentication("catalog backup MAC key is invalid"))?;
        mac.update(BACKUP_MAC_DOMAIN);

        let output_parent = destination.parent().ok_or(StoreError::State(
            "catalog backup destination has no parent",
        ))?;
        let (mut output_guard, mut output) =
            create_private_temporary(output_parent, "catalog-backup", "partial")?;
        write_mac(&mut output, &mut mac, &encoded_header)?;
        let mut source = fs::File::open(&snapshot_guard.path)
            .map_err(io_error("open plaintext catalog snapshot"))?;
        let mut buffer = vec![0u8; BACKUP_BLOCK_BYTES];
        let mut remaining = plaintext_bytes;
        for ordinal in 0..block_count {
            let expected = remaining.min(BACKUP_BLOCK_BYTES as u64) as usize;
            read_exact_or_truncated(
                &mut source,
                &mut buffer[..expected],
                "plaintext catalog snapshot was truncated",
            )?;
            let stored_bytes = expected
                .checked_add(AEAD_TAG_BYTES as usize)
                .ok_or(StoreError::State("catalog backup record length overflow"))?;
            let record_header = encode_record_header(expected as u32, stored_bytes as u32);
            let nonce = block_nonce(&nonce_prefix, ordinal);
            let aad = backup_block_aad(
                &encoded_header,
                ordinal,
                expected as u32,
                stored_bytes as u32,
            );
            let ciphertext = cipher
                .encrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: &buffer[..expected],
                        aad: &aad,
                    },
                )
                .map_err(|_| StoreError::Authentication("catalog backup encryption failed"))?;
            write_mac(&mut output, &mut mac, &record_header)?;
            write_mac(&mut output, &mut mac, &ciphertext)?;
            remaining -= expected as u64;
        }
        if remaining != 0
            || source
                .read(&mut [0u8; 1])
                .map_err(io_error("finish catalog snapshot read"))?
                != 0
        {
            return Err(StoreError::Authentication(
                "plaintext catalog snapshot changed during backup",
            ));
        }
        output
            .write_all(&mac.finalize().into_bytes())
            .map_err(io_error("write catalog backup signature"))?;
        output
            .sync_all()
            .map_err(io_error("fsync catalog backup partial"))?;
        let actual_bytes = output
            .metadata()
            .map_err(io_error("inspect catalog backup partial"))?
            .len();
        if actual_bytes != backup_bytes {
            return Err(StoreError::Authentication(
                "catalog backup length disagrees with its envelope",
            ));
        }
        drop(output);
        publish_temporary(&mut output_guard, destination)?;
        Ok(CatalogBackupReport {
            catalog_schema: header.catalog_schema,
            catalog_epoch: header.catalog_epoch,
            key_epoch: header.key_epoch,
            created_ns: header.created_ns,
            plaintext_bytes,
            backup_bytes,
            block_count,
            plaintext_digest,
        })
    }

    /// Restore a verified backup only into a bare catalog location. The caller
    /// must hold the service's external singleton/drain fence; this API never
    /// replaces a live or existing database.
    pub fn restore_catalog_backup(
        config: &StoreConfig,
        key: &StoreKey,
        backup_path: &Path,
        bounds: CatalogBackupBounds,
    ) -> Result<CatalogBackupReport, StoreError> {
        validate_backup_bounds(bounds)?;
        let catalog_parent = config
            .catalog_path
            .parent()
            .ok_or(StoreError::State("catalog restore path has no parent"))?;
        create_private_dir(catalog_parent)?;
        require_bare_catalog_path(&config.catalog_path)?;
        let (mut restored_guard, mut restored) =
            create_private_temporary(catalog_parent, "catalog-restore", "sqlite")?;
        let authenticated = authenticate_backup(config, key, backup_path, bounds)?;
        let header = authenticated.header;
        let mut source =
            fs::File::open(backup_path).map_err(io_error("open authenticated catalog backup"))?;
        source
            .seek(SeekFrom::Start(BACKUP_HEADER_BYTES as u64))
            .map_err(io_error("seek catalog backup records"))?;
        let mut plaintext_hash = Sha256::new();
        let mut remaining = header.plaintext_bytes;
        for ordinal in 0..header.block_count {
            let mut record_header = [0u8; RECORD_HEADER_BYTES as usize];
            read_exact_or_truncated(
                &mut source,
                &mut record_header,
                "catalog backup record header is truncated",
            )?;
            let plaintext_bytes = get_u32(&record_header, 0) as u64;
            let stored_bytes = get_u32(&record_header, 4) as u64;
            let expected_plaintext = remaining.min(BACKUP_BLOCK_BYTES as u64);
            if plaintext_bytes != expected_plaintext
                || stored_bytes != plaintext_bytes + AEAD_TAG_BYTES
            {
                return Err(StoreError::Codec(
                    "catalog backup record has noncanonical bounds",
                ));
            }
            let mut ciphertext = vec![0u8; stored_bytes as usize];
            read_exact_or_truncated(
                &mut source,
                &mut ciphertext,
                "catalog backup ciphertext is truncated",
            )?;
            let nonce = block_nonce(&header.nonce_prefix, ordinal);
            let aad = backup_block_aad(
                &authenticated.encoded_header,
                ordinal,
                plaintext_bytes as u32,
                stored_bytes as u32,
            );
            let plaintext = authenticated
                .cipher
                .decrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    },
                )
                .map_err(|_| {
                    StoreError::Authentication("catalog backup block authentication failed")
                })?;
            if plaintext.len() as u64 != plaintext_bytes {
                return Err(StoreError::Authentication(
                    "catalog backup plaintext length mismatch",
                ));
            }
            plaintext_hash.update(&plaintext);
            restored
                .write_all(&plaintext)
                .map_err(io_error("write restored catalog snapshot"))?;
            remaining -= plaintext_bytes;
        }
        if remaining != 0 {
            return Err(StoreError::Authentication(
                "catalog backup ended before its declared plaintext",
            ));
        }
        let actual_digest: Id32 = plaintext_hash.finalize().into();
        if actual_digest != header.plaintext_digest {
            return Err(StoreError::Authentication(
                "catalog backup plaintext digest mismatch",
            ));
        }
        restored
            .sync_all()
            .map_err(io_error("fsync restored catalog snapshot"))?;
        drop(restored);
        validate_restored_catalog(&restored_guard.path, config, header)?;
        publish_temporary(&mut restored_guard, &config.catalog_path)?;
        Ok(CatalogBackupReport {
            catalog_schema: header.catalog_schema,
            catalog_epoch: header.catalog_epoch,
            key_epoch: header.key_epoch,
            created_ns: header.created_ns,
            plaintext_bytes: header.plaintext_bytes,
            backup_bytes: authenticated.backup_bytes,
            block_count: header.block_count,
            plaintext_digest: header.plaintext_digest,
        })
    }
}
