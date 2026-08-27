use super::common::*;

#[test]
fn source_lease_protects_authenticated_manifest_and_inventory_is_catalog_only() {
    let fixture = fixture(b"tenant-source-lease");
    let (published, family, schema, input_cut) =
        publish(Arc::clone(&fixture.store), Codec::Lossless);
    let context = kvpack::wire::ValidationContext::default();
    let artifact = AuthenticatedArtifact::open(
        Arc::clone(&fixture.store),
        open_expectations(published, family, schema, input_cut),
    )
    .unwrap();
    let first_chunk = artifact.manifest().states[0].chunks[0].object_key;
    let lease_id = id(91);
    let owner = id(92);
    let incarnation = id(93);
    let lease = fixture
        .store
        .acquire_source_lease(
            &lease_id,
            &published.manifest_id,
            &owner,
            &incarnation,
            fixture.store.catalog_epoch(),
            60_000_000_000,
            &context,
        )
        .unwrap();
    assert_eq!(lease.state, SourceLeaseState::Active);
    assert_eq!(lease.object_count, 3);
    assert_eq!(
        fixture
            .store
            .acquire_source_lease(
                &lease_id,
                &published.manifest_id,
                &owner,
                &incarnation,
                fixture.store.catalog_epoch(),
                60_000_000_000,
                &context,
            )
            .unwrap(),
        lease
    );
    fixture
        .store
        .require_source_lease_object(
            &lease_id,
            &owner,
            InventoryObjectKind::Manifest,
            &published.manifest_id,
        )
        .unwrap();
    fixture
        .store
        .require_source_lease_object(&lease_id, &owner, InventoryObjectKind::Chunk, &first_chunk)
        .unwrap();
    assert!(fixture
        .store
        .require_source_lease_object(&lease_id, &id(94), InventoryObjectKind::Chunk, &first_chunk,)
        .is_err());
    assert!(!fixture.store.evict_manifest_one().unwrap());

    let inventory = fixture.store.inventory_page(None, 16).unwrap();
    assert_eq!(
        inventory
            .iter()
            .filter(|entry| entry.kind == InventoryObjectKind::Manifest)
            .count(),
        1
    );
    assert_eq!(
        inventory
            .iter()
            .filter(|entry| entry.kind == InventoryObjectKind::Chunk)
            .count(),
        2
    );
    assert_eq!(
        fixture
            .store
            .release_source_lease(&lease_id, &owner, fixture.store.catalog_epoch())
            .unwrap(),
        SourceLeaseState::Released
    );
    assert!(fixture.store.evict_manifest_one().unwrap());
}

#[test]
fn authenticated_publication_source_is_ordered_bounded_and_lease_protected() {
    let fixture = fixture(b"tenant-publication-source");
    let (published, _, _, _) = publish(Arc::clone(&fixture.store), Codec::Lossless);
    let context = kvpack::wire::ValidationContext::default();
    let mut source = fixture
        .store
        .authenticated_publication_source(&published.manifest_id, &context)
        .unwrap();
    assert_eq!(source.manifest_id(), published.manifest_id);
    assert!(!source.manifest_object().is_empty());
    assert_eq!(source.chunk_count(), 2);

    let mut expected_bytes = source.manifest_object().len() as u64;
    for ordinal in 0..source.chunk_count() {
        let metadata = source.chunk(ordinal).unwrap();
        assert_eq!(metadata.ordinal(), ordinal);
        assert_ne!(metadata.object_key(), [0; 32]);
        let (read_metadata, object) = source.read_chunk_object(ordinal).unwrap();
        assert_eq!(read_metadata, metadata);
        assert_eq!(object.len(), metadata.object_bytes() as usize);
        expected_bytes += object.len() as u64;
    }
    assert_eq!(source.expected_bytes(), expected_bytes);
    assert!(source.chunk(source.chunk_count()).is_none());
    assert!(source.read_chunk_object(source.chunk_count()).is_err());
    assert!(!fixture.store.evict_manifest_one().unwrap());

    source.renew().unwrap();
    source.release().unwrap();
    source.release().unwrap();
    assert!(source.read_chunk_object(0).is_err());
    assert!(fixture.store.evict_manifest_one().unwrap());
}
