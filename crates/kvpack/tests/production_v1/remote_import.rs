use super::common::*;

#[test]
fn authenticated_object_import_publishes_only_after_all_chunks_verify() {
    let source = fixture(b"tenant-import");
    let destination_key = kvpack::load_store_key(
        &source._temp.path().join("keys/root.key"),
        source._temp.path(),
    )
    .unwrap();
    let destination_config = StoreConfig {
        object_root: source._temp.path().join("destination-objects"),
        catalog_path: source._temp.path().join("destination/catalog.sqlite"),
        operator_tenant_id: b"tenant-import".to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: 1024 * 1024 * 1024,
        staging_quota_bytes: 1024 * 1024 * 1024,
        endurance_bytes_per_five_minutes: 1024 * 1024 * 1024,
    };
    let destination =
        Arc::new(LocalStore::open(destination_config.clone(), destination_key).unwrap());
    let (published, family, schema, input_cut) =
        publish(Arc::clone(&source.store), Codec::Lossless);
    let source_artifact = AuthenticatedArtifact::open(
        Arc::clone(&source.store),
        open_expectations(published, family.clone(), schema.clone(), input_cut),
    )
    .unwrap();
    let context = kvpack::wire::ValidationContext::default();
    let idempotency = id(77);
    destination
        .begin_authenticated_import(&idempotency, &published.manifest_id, 16 * 1024 * 1024, 1)
        .unwrap();
    assert!(matches!(
        destination.begin_authenticated_import(
            &idempotency,
            &published.manifest_id,
            16 * 1024 * 1024,
            2,
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different publication generation"
        ))
    ));
    assert!(matches!(
        destination.begin_authenticated_import(
            &idempotency,
            &published.manifest_id,
            16 * 1024 * 1024 + 1,
            1,
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with different write bounds"
        ))
    ));
    assert!(matches!(
        destination.begin_authenticated_import(&idempotency, &id(78), 16 * 1024 * 1024, 1),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different immutable declaration"
        ))
    ));
    let pack = source
        .store
        .read_authenticated_manifest_object(&published.manifest_id, &context)
        .unwrap();
    destination
        .stage_authenticated_manifest(&idempotency, &pack, &context)
        .unwrap();
    assert_eq!(destination.stat().unwrap().manifests, 0);
    drop(destination);

    let destination_key = kvpack::load_store_key(
        &source._temp.path().join("keys/root.key"),
        source._temp.path(),
    )
    .unwrap();
    let destination =
        Arc::new(LocalStore::open(destination_config.clone(), destination_key).unwrap());
    let status = destination
        .authenticated_import_status(&idempotency)
        .unwrap();
    assert_eq!(status.state, kvpack::UploadState::Receiving);
    assert_eq!(status.next_chunk_ordinal, 0);
    assert_eq!(status.publication_generation, 1);
    let first_reference = &source_artifact.manifest().states[0].chunks[0];
    let first_object = source
        .store
        .read_authenticated_chunk_object(
            &published.manifest_id,
            &first_reference.object_key,
            0,
            &context,
        )
        .unwrap();
    assert!(matches!(
        destination.put_authenticated_import_chunk(
            &idempotency,
            99,
            &first_reference.object_key,
            &first_object,
            &context,
        ),
        Err(StoreError::Expectation(_))
    ));
    destination
        .put_authenticated_import_chunk(
            &idempotency,
            0,
            &first_reference.object_key,
            &first_object,
            &context,
        )
        .unwrap();
    assert_eq!(
        destination
            .authenticated_import_status(&idempotency)
            .unwrap()
            .next_chunk_ordinal,
        1
    );
    destination
        .put_authenticated_import_chunk(
            &idempotency,
            0,
            &first_reference.object_key,
            &first_object,
            &context,
        )
        .unwrap();
    assert_eq!(
        destination
            .authenticated_import_status(&idempotency)
            .unwrap()
            .next_chunk_ordinal,
        1
    );
    assert!(matches!(
        destination.put_authenticated_import_chunk(
            &idempotency,
            2,
            &first_reference.object_key,
            &first_object,
            &context,
        ),
        Err(StoreError::Expectation(
            "chunk upload skipped the authenticated import cursor"
        ))
    ));
    drop(destination);

    let destination_key = kvpack::load_store_key(
        &source._temp.path().join("keys/root.key"),
        source._temp.path(),
    )
    .unwrap();
    let destination = Arc::new(LocalStore::open(destination_config, destination_key).unwrap());
    let status = destination
        .authenticated_import_status(&idempotency)
        .unwrap();
    assert_eq!(status.state, kvpack::UploadState::Receiving);
    assert_eq!(status.next_chunk_ordinal, 1);
    assert_eq!(status.publication_generation, 1);
    for (ordinal, reference) in source_artifact
        .manifest()
        .states
        .iter()
        .flat_map(|state| &state.chunks)
        .enumerate()
        .skip(1)
    {
        let object = source
            .store
            .read_authenticated_chunk_object(
                &published.manifest_id,
                &reference.object_key,
                ordinal as u64,
                &context,
            )
            .unwrap();
        destination
            .put_authenticated_import_chunk(
                &idempotency,
                ordinal as u64,
                &reference.object_key,
                &object,
                &context,
            )
            .unwrap();
    }
    let sealed = destination
        .seal_authenticated_import(&idempotency, &context)
        .unwrap();
    assert_eq!(sealed.state, kvpack::UploadState::Verified);
    let final_reference = source_artifact
        .manifest()
        .states
        .last()
        .unwrap()
        .chunks
        .last()
        .unwrap();
    let final_ordinal = sealed.next_chunk_ordinal - 1;
    let final_object = source
        .store
        .read_authenticated_chunk_object(
            &published.manifest_id,
            &final_reference.object_key,
            final_ordinal,
            &context,
        )
        .unwrap();
    destination
        .put_authenticated_import_chunk(
            &idempotency,
            final_ordinal,
            &final_reference.object_key,
            &final_object,
            &context,
        )
        .unwrap();
    assert_eq!(
        destination
            .seal_authenticated_import(&idempotency, &context)
            .unwrap()
            .state,
        kvpack::UploadState::Verified
    );
    let imported = destination
        .commit_authenticated_import(&idempotency, &context)
        .unwrap();
    assert_eq!(imported.manifest_id, published.manifest_id);
    assert_eq!(imported.publication_generation, 1);
    assert_eq!(
        destination
            .authenticated_import_status(&idempotency)
            .unwrap()
            .next_chunk_ordinal,
        source_artifact
            .manifest()
            .states
            .iter()
            .map(|state| state.chunks.len() as u64)
            .sum::<u64>()
    );
    assert_eq!(
        destination
            .authenticated_import_status(&idempotency)
            .unwrap()
            .publication_generation,
        1
    );
    let retry = destination
        .begin_authenticated_import(&idempotency, &published.manifest_id, 16 * 1024 * 1024, 1)
        .unwrap();
    assert_eq!(retry.state, kvpack::UploadState::Published);
    assert_eq!(retry.manifest_id, published.manifest_id);
    assert_eq!(retry.publication_generation, 1);
    destination
        .stage_authenticated_manifest(&idempotency, &pack, &context)
        .unwrap();
    assert_eq!(
        std::fs::read_dir(source._temp.path().join("destination-objects/uploads"))
            .unwrap()
            .count(),
        0
    );
    let root: [u8; 32] = std::fs::read(source._temp.path().join("keys/root.key"))
        .unwrap()
        .try_into()
        .unwrap();
    let keys = kvpack::wire::KeySchedule::derive(
        &root,
        &destination.tenant_namespace(),
        destination.key_epoch(),
    )
    .unwrap();
    let changed_pack =
        kvpack::wire::encode_authenticated_pack(source_artifact.manifest(), &keys, true, &context)
            .unwrap()
            .bytes;
    assert_ne!(changed_pack, pack);
    assert!(matches!(
        destination.stage_authenticated_manifest(&idempotency, &changed_pack, &context),
        Err(StoreError::Authentication(
            "published manifest retry changed object bytes"
        ))
    ));
    assert_eq!(destination.stat().unwrap().manifests, 1);
    AuthenticatedArtifact::open(
        Arc::clone(&destination),
        open_expectations(imported, family, schema, input_cut),
    )
    .unwrap()
    .scrub_full()
    .unwrap();
}

#[test]
fn remote_fences_and_mutation_nonces_accept_only_exact_retries() {
    let fixture = fixture(b"tenant-remote-fence");
    let (published, _, _, _) = publish(Arc::clone(&fixture.store), Codec::Raw);
    let idempotency = id(81);
    let fence = RemoteImportFence {
        manifest_id: published.manifest_id,
        owner_id: id(82),
        attempt_epoch: 7,
        publication_generation: published.publication_generation,
    };
    assert_eq!(
        fixture
            .store
            .bind_remote_import(&idempotency, fence)
            .unwrap(),
        fence
    );
    assert_eq!(
        fixture
            .store
            .bind_remote_import(&idempotency, fence)
            .unwrap(),
        fence
    );
    assert!(fixture
        .store
        .bind_remote_import(
            &idempotency,
            RemoteImportFence {
                attempt_epoch: 8,
                ..fence
            }
        )
        .is_err());
    assert_eq!(
        fixture
            .store
            .require_remote_import(&idempotency, &fence.owner_id, 7)
            .unwrap(),
        fence
    );
    assert!(fixture
        .store
        .require_remote_import(&idempotency, &id(83), 7)
        .is_err());

    let nonce = id(84);
    let request = id(85);
    let scope = id(86);
    let intent = id(87);
    assert_eq!(
        fixture
            .store
            .consume_remote_mutation(&nonce, &request, RemoteMutation::ChunkPut, &scope, &intent,)
            .unwrap(),
        MutationReplay::Fresh
    );
    assert_eq!(
        fixture
            .store
            .consume_remote_mutation(&nonce, &request, RemoteMutation::ChunkPut, &scope, &intent,)
            .unwrap(),
        MutationReplay::ExactRetry
    );
    for changed in [
        (nonce, id(88), RemoteMutation::ChunkPut, scope, intent),
        (nonce, request, RemoteMutation::Publish, scope, intent),
        (nonce, request, RemoteMutation::ChunkPut, id(88), intent),
        (nonce, request, RemoteMutation::ChunkPut, scope, id(88)),
    ] {
        assert!(fixture
            .store
            .consume_remote_mutation(&changed.0, &changed.1, changed.2, &changed.3, &changed.4)
            .is_err());
    }
}

#[test]
fn authenticated_import_retry_conflict_is_terminally_quarantined() {
    let fixture = fixture(b"tenant-import-quarantine");
    let (published, _, _, _) = publish(Arc::clone(&fixture.store), Codec::Lossless);
    let context = kvpack::wire::ValidationContext::default();
    let idempotency = id(79);
    fixture
        .store
        .begin_authenticated_import(&idempotency, &published.manifest_id, 16 * 1024 * 1024, 3)
        .unwrap();
    let pack = fixture
        .store
        .read_authenticated_manifest_object(&published.manifest_id, &context)
        .unwrap();
    fixture
        .store
        .stage_authenticated_manifest(&idempotency, &pack, &context)
        .unwrap();

    let root: [u8; 32] = std::fs::read(fixture._temp.path().join("keys/root.key"))
        .unwrap()
        .try_into()
        .unwrap();
    let keys = kvpack::wire::KeySchedule::derive(
        &root,
        &fixture.store.tenant_namespace(),
        fixture.store.key_epoch(),
    )
    .unwrap();
    let manifest = kvpack::wire::decode_authenticated_pack(&pack, &keys, &context).unwrap();
    let alternate = kvpack::wire::encode_authenticated_pack(&manifest, &keys, true, &context)
        .unwrap()
        .bytes;
    assert_ne!(alternate, pack);
    assert!(matches!(
        fixture
            .store
            .stage_authenticated_manifest(&idempotency, &alternate, &context),
        Err(StoreError::Authentication(
            "authenticated manifest retry changed object bytes"
        ))
    ));

    let status = fixture
        .store
        .authenticated_import_status(&idempotency)
        .unwrap();
    assert_eq!(status.state, kvpack::UploadState::Quarantined);
    assert_eq!(status.publication_generation, 3);
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
    assert_eq!(
        std::fs::read_dir(fixture._temp.path().join("objects/uploads"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        std::fs::read_dir(fixture._temp.path().join("objects/quarantine"))
            .unwrap()
            .count(),
        1
    );
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let (path_token, file_bytes, lifetime, reason): (String, u64, u64, String) = connection
        .query_row(
            "SELECT path_token,file_bytes,expires_ns-created_ns,reason FROM quarantine_entries WHERE tenant=?1 AND object_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                idempotency.as_slice()
            ],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert!(!path_token.contains('/'));
    assert_eq!(file_bytes, pack.len() as u64);
    assert_eq!(lifetime, 24 * 60 * 60 * 1_000_000_000);
    assert_eq!(reason, "authenticated manifest retry changed object bytes");
    drop(connection);

    fixture
        .store
        .quarantine_authenticated_import(&idempotency)
        .unwrap();
    fixture
        .store
        .cancel_authenticated_import(&idempotency)
        .unwrap();
    assert_eq!(
        fixture
            .store
            .authenticated_import_status(&idempotency)
            .unwrap()
            .state,
        kvpack::UploadState::Quarantined
    );
    assert!(matches!(
        fixture.store.begin_authenticated_import(
            &idempotency,
            &published.manifest_id,
            16 * 1024 * 1024,
            3,
        ),
        Err(StoreError::State("upload is not reserved"))
    ));
}
