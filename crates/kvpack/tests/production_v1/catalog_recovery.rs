use super::common::*;

#[test]
fn signed_inventory_and_encrypted_catalog_backup_restore_a_bare_cache_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let tenant = b"tenant-catalog-recovery";
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let config = StoreConfig {
        object_root: temp.path().join("objects"),
        catalog_path: temp.path().join("catalog/catalog.sqlite"),
        operator_tenant_id: tenant.to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: 1024 * 1024 * 1024,
        staging_quota_bytes: 1024 * 1024 * 1024,
        endurance_bytes_per_five_minutes: 1024 * 1024 * 1024,
    };
    let store = Arc::new(
        LocalStore::open(
            config.clone(),
            kvpack::load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap(),
    );
    let (published, family, schema, input_cut) = publish(Arc::clone(&store), Codec::Lossless);

    let inventory_path = temp.path().join("inventory.snapshot");
    let inventory_bounds = InventorySnapshotBounds {
        maximum_entries: 16,
        maximum_snapshot_bytes: 16 * 1024,
    };
    let inventory_report = store
        .write_signed_inventory_snapshot(&inventory_path, inventory_bounds)
        .unwrap();
    assert_eq!(inventory_report.entries, 3);
    let verification_key = kvpack::load_store_key(&key_path, temp.path()).unwrap();
    let verified = LocalStore::verify_signed_inventory_snapshot(
        &config,
        &verification_key,
        &inventory_path,
        inventory_bounds,
    )
    .unwrap();
    assert_eq!(verified.report, inventory_report);
    assert_eq!(verified.entries.len(), 3);

    let corrupt_inventory = temp.path().join("inventory-corrupt.snapshot");
    let mut inventory_bytes = std::fs::read(&inventory_path).unwrap();
    let inventory_mutation = inventory_bytes.len() / 2;
    inventory_bytes[inventory_mutation] ^= 0x80;
    std::fs::write(&corrupt_inventory, inventory_bytes).unwrap();
    assert!(matches!(
        LocalStore::verify_signed_inventory_snapshot(
            &config,
            &verification_key,
            &corrupt_inventory,
            inventory_bounds,
        ),
        Err(StoreError::Authentication(_))
    ));
    drop(verification_key);

    let backup_path = temp.path().join("catalog.backup");
    let backup_bounds = CatalogBackupBounds {
        maximum_plaintext_bytes: 64 * 1024 * 1024,
        maximum_backup_bytes: 65 * 1024 * 1024,
        maximum_blocks: 16,
    };
    let backup_report = store
        .write_catalog_backup(&backup_path, backup_bounds)
        .unwrap();
    assert!(backup_report.plaintext_bytes > 0);
    assert_eq!(backup_report.block_count, 1);
    let backup_bytes = std::fs::read(&backup_path).unwrap();
    assert_ne!(&backup_bytes[..16], b"SQLite format 3\0");
    assert!(!backup_bytes
        .windows(b"SQLite format 3\0".len())
        .any(|window| window == b"SQLite format 3\0"));
    assert!(store
        .write_catalog_backup(&backup_path, backup_bounds)
        .is_err());

    let live_reconciliation = store
        .reconcile_catalog_objects(ReconciliationBounds {
            maximum_objects: 16,
            maximum_scan_bytes: 64 * 1024 * 1024,
        })
        .unwrap();
    assert_eq!(live_reconciliation.catalog_objects, 3);
    assert_eq!(live_reconciliation.present_objects, 3);
    assert!(live_reconciliation.missing_objects.is_empty());
    assert!(live_reconciliation.corrupt_objects.is_empty());
    drop(store);

    for suffix in ["", "-wal", "-shm"] {
        let path =
            std::path::PathBuf::from(format!("{}{suffix}", config.catalog_path.to_string_lossy()));
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
    }
    let restore_key = kvpack::load_store_key(&key_path, temp.path()).unwrap();
    let restored_report =
        LocalStore::restore_catalog_backup(&config, &restore_key, &backup_path, backup_bounds)
            .unwrap();
    assert_eq!(restored_report, backup_report);
    drop(restore_key);
    assert!(LocalStore::restore_catalog_backup(
        &config,
        &kvpack::load_store_key(&key_path, temp.path()).unwrap(),
        &backup_path,
        backup_bounds,
    )
    .is_err());

    let restored = Arc::new(
        LocalStore::open(
            config.clone(),
            kvpack::load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap(),
    );
    AuthenticatedArtifact::open(
        Arc::clone(&restored),
        open_expectations(published, family, schema, input_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();

    let chunk = restored
        .inventory_page(None, 16)
        .unwrap()
        .into_iter()
        .find(|entry| entry.kind == InventoryObjectKind::Chunk)
        .unwrap();
    let name: String = chunk
        .object_id
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let chunk_path = config
        .object_root
        .join("chunks")
        .join(&name[..2])
        .join(format!("{name}.kvchunk"));
    let held_path = chunk_path.with_extension("kvchunk.held-for-reconciliation");
    std::fs::rename(&chunk_path, &held_path).unwrap();
    let missing = restored
        .reconcile_catalog_objects(ReconciliationBounds {
            maximum_objects: 16,
            maximum_scan_bytes: 64 * 1024 * 1024,
        })
        .unwrap();
    assert_eq!(missing.missing_objects.len(), 1);
    assert_eq!(missing.corrupt_objects.len(), 0);
    std::fs::rename(&held_path, &chunk_path).unwrap();

    let mut corrupt_backup_bytes = backup_bytes;
    let backup_mutation = corrupt_backup_bytes.len() / 2;
    corrupt_backup_bytes[backup_mutation] ^= 1;
    let corrupt_backup = temp.path().join("catalog-corrupt.backup");
    std::fs::write(&corrupt_backup, corrupt_backup_bytes).unwrap();
    let mut corrupt_target = config.clone();
    corrupt_target.catalog_path = temp.path().join("corrupt-target/catalog.sqlite");
    assert!(matches!(
        LocalStore::restore_catalog_backup(
            &corrupt_target,
            &kvpack::load_store_key(&key_path, temp.path()).unwrap(),
            &corrupt_backup,
            backup_bounds,
        ),
        Err(StoreError::Authentication(_))
    ));
    assert!(!corrupt_target.catalog_path.exists());

    let mut substituted_tenant = config.clone();
    substituted_tenant.operator_tenant_id = b"tenant-catalog-substitution".to_vec();
    substituted_tenant.catalog_path = temp.path().join("substituted-target/catalog.sqlite");
    assert!(matches!(
        LocalStore::restore_catalog_backup(
            &substituted_tenant,
            &kvpack::load_store_key(&key_path, temp.path()).unwrap(),
            &backup_path,
            backup_bounds,
        ),
        Err(StoreError::Authentication(_))
    ));
    assert!(!substituted_tenant.catalog_path.exists());

    let wrong_key_path = temp.path().join("wrong-keys/root.key");
    kvpack::create_store_key_random(&wrong_key_path, temp.path()).unwrap();
    let mut lost_key_target = config.clone();
    lost_key_target.catalog_path = temp.path().join("lost-key-target/catalog.sqlite");
    assert!(matches!(
        LocalStore::restore_catalog_backup(
            &lost_key_target,
            &kvpack::load_store_key(&wrong_key_path, temp.path()).unwrap(),
            &backup_path,
            backup_bounds,
        ),
        Err(StoreError::Authentication(_))
    ));
    assert!(!lost_key_target.catalog_path.exists());
}
