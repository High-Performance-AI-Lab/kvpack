use super::common::*;

#[test]
fn encrypted_lossless_round_trip_installs_only_after_all_states_verify() {
    let fixture = fixture(b"tenant-a");
    let (published, family, schema, input_cut) =
        publish(Arc::clone(&fixture.store), Codec::Lossless);
    let artifact = AuthenticatedArtifact::open(
        Arc::clone(&fixture.store),
        open_expectations(published, family, schema, input_cut),
    )
    .unwrap();
    let selection = RestoreSelection::new([StateKey::new(0, "k"), StateKey::new(0, "v")]).unwrap();
    let mut sink = ShadowSink::default();
    artifact.restore_selected(&selection, &mut sink).unwrap();
    assert!(sink.committed);
    assert!(!sink.aborted);
    assert_eq!(sink.installed[&StateKey::new(0, "k")], vec![11; 32]);
    assert_eq!(sink.installed[&StateKey::new(0, "v")], vec![22; 32]);
    artifact.scrub_full().unwrap();
}

#[test]
fn offline_fsck_rebuilds_a_bare_catalog_from_authenticated_objects() {
    let fixture = fixture(b"tenant-offline-fsck");
    let configuration = StoreConfig {
        object_root: fixture._temp.path().join("objects"),
        catalog_path: fixture._temp.path().join("catalog/catalog.sqlite"),
        operator_tenant_id: b"tenant-offline-fsck".to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: 1024 * 1024 * 1024,
        staging_quota_bytes: 1024 * 1024 * 1024,
        endurance_bytes_per_five_minutes: 1024 * 1024 * 1024,
    };
    let (published, family, schema, input_cut) =
        publish(Arc::clone(&fixture.store), Codec::Lossless);
    let before = fixture.store.stat().unwrap();
    drop(fixture.store);
    for suffix in ["", "-wal", "-shm"] {
        let path = std::path::PathBuf::from(format!(
            "{}{suffix}",
            configuration.catalog_path.to_string_lossy()
        ));
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("remove temporary catalog: {error}"),
        }
    }

    let report = LocalStore::fsck_rebuild_catalog(
        configuration.clone(),
        kvpack::load_store_key(
            &fixture._temp.path().join("keys/root.key"),
            fixture._temp.path(),
        )
        .unwrap(),
        FsckBounds {
            maximum_manifests: 4,
            maximum_chunks: 4,
            maximum_scan_bytes: 1024 * 1024,
        },
        kvpack::wire::ValidationContext::default(),
    )
    .unwrap();
    assert_eq!(report.manifests, 1);
    assert_eq!(report.chunks, 2);
    assert!(report.scanned_bytes > 0);

    let rebuilt = Arc::new(
        LocalStore::open(
            configuration,
            kvpack::load_store_key(
                &fixture._temp.path().join("keys/root.key"),
                fixture._temp.path(),
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let after = rebuilt.stat().unwrap();
    assert_eq!(after.durable_bytes, before.durable_bytes);
    assert_eq!((after.manifests, after.chunks), (1, 2));
    AuthenticatedArtifact::open(
        rebuilt,
        open_expectations(published, family, schema, input_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
}

#[test]
fn selection_cannot_split_an_atomic_group() {
    let fixture = fixture(b"tenant-a");
    let (published, family, schema, input_cut) = publish(Arc::clone(&fixture.store), Codec::Raw);
    let artifact = AuthenticatedArtifact::open(
        Arc::clone(&fixture.store),
        open_expectations(published, family, schema, input_cut),
    )
    .unwrap();
    let selection = RestoreSelection::new([StateKey::new(0, "k")]).unwrap();
    let mut sink = ShadowSink::default();
    assert!(matches!(
        artifact.restore_selected(&selection, &mut sink),
        Err(StoreError::Expectation(_))
    ));
    assert!(!sink.committed);
}
