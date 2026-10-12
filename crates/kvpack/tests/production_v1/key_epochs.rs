use super::common::*;

#[test]
fn provider_identity_is_not_durable_and_development_keys_fail_readiness() {
    let tenant = b"tenant-provider-parity";
    let development = InMemoryKeyProvider::from_master(id(70)).unwrap();
    assert!(matches!(
        kvpack::load_store_key_from_provider(
            &development,
            tenant,
            KeyEpochWindow {
                minimum_readable: 1,
                active: 1,
            },
            true,
        ),
        Err(StoreError::State(
            "development key provider is ineligible for production readiness"
        ))
    ));
    let file_provider = FileKeyProvider::new(
        std::path::PathBuf::from("/development/root.key"),
        std::path::PathBuf::from("/development"),
    );
    assert_eq!(
        file_provider.qualification(),
        KeyProviderQualification::DevelopmentOnly
    );
    assert_eq!(
        MacOsKeychainProvider::new("kvpack".into(), "tenant".into())
            .unwrap()
            .qualification(),
        KeyProviderQualification::Production
    );
    assert_eq!(
        LinuxOsKeyStoreProvider::new("kvpack".into())
            .unwrap()
            .qualification(),
        KeyProviderQualification::Production
    );

    let provider_a = TestKeyProvider {
        provider_name: "test-kms-a",
        stable_root: id(71),
        epoch_roots: vec![(1, id(72))],
    };
    let provider_b = TestKeyProvider {
        provider_name: "test-kms-b",
        ..provider_a.clone()
    };
    let incomplete_provider = TestKeyProvider {
        provider_name: "test-kms-incomplete",
        stable_root: id(71),
        epoch_roots: vec![(2, id(73))],
    };
    assert!(kvpack::load_store_key_from_provider(
        &incomplete_provider,
        tenant,
        KeyEpochWindow {
            minimum_readable: 1,
            active: 2,
        },
        true,
    )
    .is_err());
    assert_ne!(provider_a.name(), provider_b.name());
    let root_a = tempfile::tempdir().unwrap();
    let root_b = tempfile::tempdir().unwrap();
    let store_a = Arc::new(
        LocalStore::open(
            epoch_config(root_a.path(), "catalog.sqlite", tenant, 1, 1),
            provider_key(&provider_a, tenant, 1, 1),
        )
        .unwrap(),
    );
    let store_b = Arc::new(
        LocalStore::open(
            epoch_config(root_b.path(), "catalog.sqlite", tenant, 1, 1),
            provider_key(&provider_b, tenant, 1, 1),
        )
        .unwrap(),
    );
    assert_eq!(store_a.tenant_namespace(), store_b.tenant_namespace());

    let (published_a, _, _, _) =
        publish_with_options(Arc::clone(&store_a), Codec::Raw, [1, 2, 3, 4], 21, false);
    let (published_b, _, _, _) =
        publish_with_options(Arc::clone(&store_b), Codec::Raw, [1, 2, 3, 4], 21, false);
    assert_eq!(published_a.manifest_id, published_b.manifest_id);
    let context = kvpack::wire::ValidationContext::default();
    assert_eq!(
        store_a
            .read_authenticated_manifest_object(&published_a.manifest_id, &context)
            .unwrap(),
        store_b
            .read_authenticated_manifest_object(&published_b.manifest_id, &context)
            .unwrap()
    );
}

#[test]
fn rotation_retirement_and_backup_key_recovery_preserve_new_cache_entries() {
    let tenant = b"tenant-key-rotation";
    let root = tempfile::tempdir().unwrap();
    let full_provider = TestKeyProvider {
        provider_name: "test-kms-full-backup",
        stable_root: id(80),
        epoch_roots: vec![(1, id(81)), (2, id(82))],
    };
    let epoch_two_provider = TestKeyProvider {
        provider_name: "test-kms-active",
        stable_root: id(80),
        epoch_roots: vec![(2, id(82))],
    };
    let primary_epoch_one = epoch_config(root.path(), "primary.sqlite", tenant, 1, 1);
    let store = Arc::new(
        LocalStore::open(
            primary_epoch_one,
            provider_key(&full_provider, tenant, 1, 1),
        )
        .unwrap(),
    );
    let namespace = store.tenant_namespace();
    let (old, old_family, old_schema, old_cut) =
        publish_with_options(Arc::clone(&store), Codec::Raw, [10, 20, 30, 40], 31, true);
    assert_eq!(store.stat().unwrap().chunks, 2);
    drop(store);

    let primary_overlap = epoch_config(root.path(), "primary.sqlite", tenant, 1, 2);
    let store = Arc::new(
        LocalStore::open(primary_overlap, provider_key(&full_provider, tenant, 1, 2)).unwrap(),
    );
    assert_eq!(store.tenant_namespace(), namespace);
    AuthenticatedArtifact::open(
        Arc::clone(&store),
        open_expectations(old, old_family.clone(), old_schema.clone(), old_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
    let (new, new_family, new_schema, new_cut) =
        publish_with_options(Arc::clone(&store), Codec::Raw, [50, 60, 70, 80], 32, true);
    assert_ne!(old.manifest_id, new.manifest_id);
    assert_eq!(store.stat().unwrap().chunks, 4);
    drop(store);

    let primary_retired = epoch_config(root.path(), "primary.sqlite", tenant, 2, 2);
    let store = Arc::new(
        LocalStore::open(
            primary_retired.clone(),
            provider_key(&epoch_two_provider, tenant, 2, 2),
        )
        .unwrap(),
    );
    assert_eq!(store.tenant_namespace(), namespace);
    assert!(matches!(
        AuthenticatedArtifact::open(
            Arc::clone(&store),
            open_expectations(old, old_family.clone(), old_schema.clone(), old_cut,),
        ),
        Err(StoreError::NotFound)
    ));
    AuthenticatedArtifact::open(
        Arc::clone(&store),
        open_expectations(new, new_family.clone(), new_schema.clone(), new_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();

    let (_, old_nodes) = store
        .derive_input_cut(&semantic(), &old_family, &[10, 20, 30, 40], &auxiliary())
        .unwrap();
    assert!(store
        .resolve_prefix(
            &old_nodes,
            &kvpack::wire::semantic_model_id(&semantic()),
            &kvpack::wire::representation_family_id(&old_family).unwrap(),
            8,
        )
        .unwrap()
        .is_none());

    let mut tombstoned_manifests = 0;
    let mut tombstoned_chunks = 0;
    for _ in 0..4 {
        let report = store.retire_key_epochs_before(2, 1).unwrap();
        tombstoned_manifests += report.manifests_tombstoned;
        tombstoned_chunks += report.chunks_tombstoned;
        if report.complete() {
            break;
        }
    }
    assert_eq!(tombstoned_manifests, 1);
    assert_eq!(tombstoned_chunks, 2);
    assert!(store.retire_key_epochs_before(2, 1).unwrap().complete());

    let recovered_config = epoch_config(root.path(), "recovered.sqlite", tenant, 1, 2);
    let report = LocalStore::fsck_rebuild_catalog(
        recovered_config.clone(),
        provider_key(&full_provider, tenant, 1, 2),
        FsckBounds::default(),
        kvpack::wire::ValidationContext::default(),
    )
    .unwrap();
    assert_eq!(report.manifests, 2);
    assert_eq!(report.chunks, 4);
    let recovered = Arc::new(
        LocalStore::open(recovered_config, provider_key(&full_provider, tenant, 1, 2)).unwrap(),
    );
    AuthenticatedArtifact::open(
        Arc::clone(&recovered),
        open_expectations(old, old_family, old_schema, old_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
    AuthenticatedArtifact::open(
        Arc::clone(&recovered),
        open_expectations(new, new_family.clone(), new_schema.clone(), new_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
    drop(recovered);

    let backward = epoch_config(root.path(), "primary.sqlite", tenant, 1, 2);
    assert!(matches!(
        LocalStore::open(backward, provider_key(&full_provider, tenant, 1, 2)),
        Err(StoreError::State(
            "key epochs cannot move backward in an existing catalog"
        ))
    ));

    assert!(store.evict_manifest_one().unwrap());
    assert!(store.gc_one().unwrap());
    assert!(store.gc_one().unwrap());
    assert!(!store.gc_one().unwrap());
    let final_stat = store.stat().unwrap();
    assert_eq!(final_stat.manifests, 1);
    assert_eq!(final_stat.chunks, 2);
    AuthenticatedArtifact::open(
        Arc::clone(&store),
        open_expectations(new, new_family, new_schema, new_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
}
