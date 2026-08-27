use super::*;

impl LocalStore {
    /// Compare catalog-listed immutable objects with their expected files.
    /// This is an explicit bounded recovery operation, not a startup scan, and
    /// it never enumerates unrelated or unknown files on the object tier.
    pub fn reconcile_catalog_objects(
        &self,
        bounds: ReconciliationBounds,
    ) -> Result<CatalogReconciliationReport, StoreError> {
        if bounds.maximum_objects == 0 || bounds.maximum_scan_bytes == 0 {
            return Err(StoreError::Expectation(
                "catalog reconciliation bounds are invalid",
            ));
        }
        let entries = {
            let connection = self.lock_catalog()?;
            load_inventory_entries(
                &connection,
                &self.tenant_namespace,
                self.minimum_readable_key_epoch(),
                self.key_epoch(),
                bounds.maximum_objects,
            )?
        };
        let mut report = CatalogReconciliationReport {
            catalog_objects: entries.len() as u64,
            present_objects: 0,
            checked_bytes: 0,
            missing_objects: Vec::new(),
            corrupt_objects: Vec::new(),
        };
        for entry in entries {
            report.checked_bytes =
                report
                    .checked_bytes
                    .checked_add(entry.object_bytes)
                    .ok_or(StoreError::Quota(
                        "catalog reconciliation byte total overflow",
                    ))?;
            if report.checked_bytes > bounds.maximum_scan_bytes {
                return Err(StoreError::Quota(
                    "catalog reconciliation scan byte bound exceeded",
                ));
            }
            let cursor = InventoryCursor {
                kind: entry.kind,
                object_id: entry.object_id,
            };
            match self.check_inventory_object(&entry) {
                Ok(ObjectCheck::Present) => report.present_objects += 1,
                Ok(ObjectCheck::Missing) => report.missing_objects.push(cursor),
                Ok(ObjectCheck::Corrupt) => report.corrupt_objects.push(cursor),
                Err(error) => return Err(error),
            }
        }
        Ok(report)
    }

    fn check_inventory_object(&self, entry: &InventoryEntry) -> Result<ObjectCheck, StoreError> {
        let path = match entry.kind {
            InventoryObjectKind::Manifest => self.manifest_path(&entry.object_id),
            InventoryObjectKind::Chunk => self.chunk_path(&entry.object_id),
        };
        let mut file = match rustix::fs::open(
            &path,
            rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC | rustix::fs::OFlags::NOFOLLOW,
            rustix::fs::Mode::empty(),
        ) {
            Ok(fd) => fs::File::from(fd),
            Err(rustix::io::Errno::NOENT) => return Ok(ObjectCheck::Missing),
            Err(errno) => {
                return Err(StoreError::Io {
                    op: "open catalog reconciliation object",
                    source: std::io::Error::from(errno),
                });
            }
        };
        let metadata = file
            .metadata()
            .map_err(io_error("inspect catalog reconciliation object"))?;
        if !metadata.is_file() || metadata.len() != entry.object_bytes {
            return Ok(ObjectCheck::Corrupt);
        }
        let expected_len = usize::try_from(entry.object_bytes)
            .map_err(|_| StoreError::Quota("catalog object does not fit address space"))?;
        let mut bytes = Vec::with_capacity(expected_len);
        (&mut file)
            .take(entry.object_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(io_error("read catalog reconciliation object"))?;
        if bytes.len() != expected_len {
            return Ok(ObjectCheck::Corrupt);
        }
        match entry.kind {
            InventoryObjectKind::Chunk => {
                let digest: Id32 = Sha256::digest(&bytes).into();
                if digest != entry.object_digest {
                    return Ok(ObjectCheck::Corrupt);
                }
            }
            InventoryObjectKind::Manifest => {
                let result = (|| {
                    let header = inspect_pack_header(&bytes)?;
                    if header.tenant_namespace != self.tenant_namespace
                        || header.key_epoch != entry.key_epoch
                    {
                        return Err(StoreError::Authentication(
                            "reconciled manifest envelope identity mismatch",
                        ));
                    }
                    let keys = self.schedule(header.key_epoch)?;
                    let manifest =
                        decode_authenticated_pack(&bytes, &keys, &ValidationContext::default())?;
                    if manifest_id(&manifest.encode_canonical()?) != entry.object_id {
                        return Err(StoreError::Authentication(
                            "reconciled manifest identity mismatch",
                        ));
                    }
                    Ok::<(), StoreError>(())
                })();
                if result.is_err() {
                    return Ok(ObjectCheck::Corrupt);
                }
            }
        }
        Ok(ObjectCheck::Present)
    }
}
