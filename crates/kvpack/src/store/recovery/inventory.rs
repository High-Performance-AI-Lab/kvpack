use super::*;

impl LocalStore {
    /// Write a complete, bounded catalog-derived inventory. Entries contain
    /// only tenant-scoped opaque identities and are canonically ordered.
    pub fn write_signed_inventory_snapshot(
        &self,
        destination: &Path,
        bounds: InventorySnapshotBounds,
    ) -> Result<InventorySnapshotReport, StoreError> {
        validate_inventory_bounds(bounds)?;
        require_new_destination(destination)?;
        let entries = {
            let connection = self.lock_catalog()?;
            load_inventory_entries(
                &connection,
                &self.tenant_namespace,
                self.minimum_readable_key_epoch(),
                self.key_epoch(),
                bounds.maximum_entries,
            )?
        };
        let entry_count = entries.len() as u64;
        let payload_bytes = entry_count
            .checked_mul(INVENTORY_ENTRY_BYTES as u64)
            .ok_or(StoreError::State("inventory payload length overflow"))?;
        let snapshot_bytes = (INVENTORY_HEADER_BYTES as u64)
            .checked_add(payload_bytes)
            .and_then(|value| value.checked_add(SIGNATURE_BYTES))
            .ok_or(StoreError::State("inventory snapshot length overflow"))?;
        if snapshot_bytes > bounds.maximum_snapshot_bytes {
            return Err(StoreError::Quota(
                "signed inventory snapshot exceeds its byte bound",
            ));
        }
        let mut payload_hash = Sha256::new();
        for entry in &entries {
            payload_hash.update(encode_inventory_entry(entry));
        }
        let payload_digest: Id32 = payload_hash.finalize().into();
        let mut salt = [0u8; 16];
        getrandom::fill(&mut salt)
            .map_err(|_| StoreError::State("inventory snapshot salt generation failed"))?;
        let header = InventoryHeader {
            tenant: self.tenant_namespace,
            catalog_schema: CATALOG_SCHEMA_VERSION as u64,
            catalog_epoch: self.catalog_epoch(),
            key_epoch: self.key_epoch(),
            created_ns: now_ns(),
            entry_count,
            payload_bytes,
            payload_digest,
            salt,
        };
        let encoded_header = encode_inventory_header(header);
        let schedule = self.schedule(header.key_epoch)?;
        let authentication_key = derive_inventory_key(&schedule, &salt)?;
        let mut mac = <HmacSha256 as Mac>::new_from_slice(authentication_key.as_ref())
            .map_err(|_| StoreError::Authentication("inventory MAC key is invalid"))?;
        mac.update(INVENTORY_MAC_DOMAIN);
        let output_parent = destination
            .parent()
            .ok_or(StoreError::State("inventory destination has no parent"))?;
        let (mut output_guard, mut output) =
            create_private_temporary(output_parent, "inventory-snapshot", "partial")?;
        write_mac(&mut output, &mut mac, &encoded_header)?;
        for entry in &entries {
            write_mac(&mut output, &mut mac, &encode_inventory_entry(entry))?;
        }
        output
            .write_all(&mac.finalize().into_bytes())
            .map_err(io_error("write inventory signature"))?;
        output
            .sync_all()
            .map_err(io_error("fsync inventory snapshot partial"))?;
        if output
            .metadata()
            .map_err(io_error("inspect inventory snapshot partial"))?
            .len()
            != snapshot_bytes
        {
            return Err(StoreError::Authentication(
                "inventory snapshot length disagrees with its envelope",
            ));
        }
        drop(output);
        publish_temporary(&mut output_guard, destination)?;
        Ok(InventorySnapshotReport {
            catalog_schema: header.catalog_schema,
            catalog_epoch: header.catalog_epoch,
            key_epoch: header.key_epoch,
            created_ns: header.created_ns,
            entries: entry_count,
            snapshot_bytes,
            payload_digest,
        })
    }

    pub fn verify_signed_inventory_snapshot(
        config: &StoreConfig,
        key: &StoreKey,
        snapshot_path: &Path,
        bounds: InventorySnapshotBounds,
    ) -> Result<VerifiedInventorySnapshot, StoreError> {
        validate_inventory_bounds(bounds)?;
        let mut file =
            fs::File::open(snapshot_path).map_err(io_error("open signed inventory snapshot"))?;
        let metadata = file
            .metadata()
            .map_err(io_error("inspect signed inventory snapshot"))?;
        if !metadata.is_file() || metadata.len() > bounds.maximum_snapshot_bytes {
            return Err(StoreError::Quota(
                "signed inventory snapshot exceeds its bound",
            ));
        }
        let mut encoded_header = [0u8; INVENTORY_HEADER_BYTES];
        read_exact_or_truncated(
            &mut file,
            &mut encoded_header,
            "inventory snapshot header is truncated",
        )?;
        let header = decode_inventory_header(&encoded_header)?;
        if header.catalog_schema != CATALOG_SCHEMA_VERSION as u64
            || header.entry_count > bounds.maximum_entries as u64
            || header.payload_bytes
                != header
                    .entry_count
                    .checked_mul(INVENTORY_ENTRY_BYTES as u64)
                    .ok_or(StoreError::Codec("inventory payload length overflow"))?
        {
            return Err(StoreError::Codec(
                "inventory snapshot schema or bounds are invalid",
            ));
        }
        let expected_bytes = (INVENTORY_HEADER_BYTES as u64)
            .checked_add(header.payload_bytes)
            .and_then(|value| value.checked_add(SIGNATURE_BYTES))
            .ok_or(StoreError::Codec("inventory snapshot length overflow"))?;
        if metadata.len() != expected_bytes {
            return Err(StoreError::Codec(
                "inventory snapshot has trailing or truncated bytes",
            ));
        }
        let expected_tenant = recovery_tenant(config, key, header.key_epoch)?;
        let schedule = key.schedule(&expected_tenant, header.key_epoch)?;
        let authentication_key = derive_inventory_key(&schedule, &header.salt)?;
        verify_file_mac(
            snapshot_path,
            expected_bytes,
            INVENTORY_MAC_DOMAIN,
            authentication_key.as_ref(),
        )?;
        validate_snapshot_identity(config, header.tenant, header.catalog_epoch, expected_tenant)?;

        file.seek(SeekFrom::Start(INVENTORY_HEADER_BYTES as u64))
            .map_err(io_error("seek inventory payload"))?;
        let mut payload_hash = Sha256::new();
        let mut entries = Vec::with_capacity(header.entry_count as usize);
        let mut previous: Option<InventoryCursor> = None;
        for _ in 0..header.entry_count {
            let mut encoded = [0u8; INVENTORY_ENTRY_BYTES];
            read_exact_or_truncated(&mut file, &mut encoded, "inventory entry is truncated")?;
            payload_hash.update(encoded);
            let entry = decode_inventory_entry(&encoded)?;
            let cursor = InventoryCursor {
                kind: entry.kind,
                object_id: entry.object_id,
            };
            if previous.is_some_and(|value| value >= cursor) {
                return Err(StoreError::Codec(
                    "inventory entries are not in canonical order",
                ));
            }
            previous = Some(cursor);
            entries.push(entry);
        }
        let actual_digest: Id32 = payload_hash.finalize().into();
        if actual_digest != header.payload_digest {
            return Err(StoreError::Authentication(
                "inventory payload digest mismatch",
            ));
        }
        Ok(VerifiedInventorySnapshot {
            report: InventorySnapshotReport {
                catalog_schema: header.catalog_schema,
                catalog_epoch: header.catalog_epoch,
                key_epoch: header.key_epoch,
                created_ns: header.created_ns,
                entries: header.entry_count,
                snapshot_bytes: expected_bytes,
                payload_digest: header.payload_digest,
            },
            entries,
        })
    }
}
