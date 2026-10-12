use super::common::*;

#[test]
fn parent_splice_and_ancestor_or_final_substitution_fail_closed() {
    let fixture = fixture(b"restore-parent-splice");
    let (root, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let wrong_parent_cut = fixture
        .store
        .derive_input_cut(
            &semantic(),
            &family(),
            &(1..=256u32).collect::<Vec<_>>(),
            &auxiliary(),
        )
        .unwrap()
        .0;
    assert!(matches!(
        ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            declaration(
                512,
                ManifestKind::Delta {
                    parent: root.manifest_id,
                    parent_cut: wrong_parent_cut,
                    depth: 1,
                },
                256,
                256,
            ),
            WritePolicy::exact_qualified(id(97), semantic(), &family()).unwrap(),
        ),
        Err(StoreError::Expectation(
            "delta parent cut does not identify the exact child prefix"
        ))
    ));

    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let root_path = manifest_path(&fixture, root.manifest_id);
    let child_path = manifest_path(&fixture, child.manifest_id);
    let root_bytes = std::fs::read(&root_path).unwrap();
    let child_bytes = std::fs::read(&child_path).unwrap();
    std::fs::write(&root_path, &child_bytes).unwrap();
    assert!(matches!(
        AuthenticatedRestorePlan::build(
            Arc::clone(&fixture.store),
            &candidate,
            RestoreLimits::default(),
        ),
        Err(StoreError::Authentication(_))
    ));
    std::fs::write(&root_path, &root_bytes).unwrap();
    std::fs::write(&child_path, &root_bytes).unwrap();
    assert!(matches!(
        AuthenticatedRestorePlan::build(
            Arc::clone(&fixture.store),
            &candidate,
            RestoreLimits::default(),
        ),
        Err(StoreError::Authentication(_))
    ));
}

#[test]
fn missing_locations_are_clean_misses_but_corruption_is_integrity_failure() {
    let missing = fixture(b"restore-missing");
    let (_, child) = publish_delta_chain(Arc::clone(&missing.store));
    let missing_path = manifest_path(&missing, child.manifest_id);
    let repaired_bytes = std::fs::read(&missing_path).unwrap();
    std::fs::remove_file(&missing_path).unwrap();
    let candidates = missing.store.restore_candidates(request(600, 1)).unwrap();
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].matched_cut().token_count, 256);
    assert_eq!(candidates[1].tier(), RestoreTier::Recompute);
    std::fs::write(&missing_path, repaired_bytes).unwrap();
    let repaired = missing.store.restore_candidates(request(600, 1)).unwrap();
    assert_eq!(repaired[0].manifest_id(), Some(child.manifest_id));
    assert_eq!(repaired[0].matched_cut().token_count, 512);

    let missing_chunk = fixture(b"restore-missing-chunk");
    let (root, child) = publish_delta_chain(Arc::clone(&missing_chunk.store));
    let connection =
        rusqlite::Connection::open(missing_chunk.temp.path().join("catalog/catalog.sqlite"))
            .unwrap();
    let object_key: Vec<u8> = connection
        .query_row(
            "SELECT object_key FROM manifest_chunks WHERE tenant=?1 AND manifest_id=?2 ORDER BY ordinal LIMIT 1",
            rusqlite::params![
                missing_chunk.store.tenant_namespace().as_slice(),
                child.manifest_id.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    let object_name = object_key
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let chunk_path = missing_chunk
        .temp
        .path()
        .join("objects/chunks")
        .join(&object_name[..2])
        .join(format!("{object_name}.kvchunk"));
    let chunk_bytes = std::fs::read(&chunk_path).unwrap();
    std::fs::remove_file(&chunk_path).unwrap();
    let fallback = missing_chunk
        .store
        .restore_candidates(request(600, 1))
        .unwrap();
    assert_eq!(fallback[0].manifest_id(), Some(root.manifest_id));
    assert_eq!(fallback[1].tier(), RestoreTier::Recompute);
    std::fs::write(&chunk_path, chunk_bytes).unwrap();
    let repaired = missing_chunk
        .store
        .restore_candidates(request(600, 1))
        .unwrap();
    assert_eq!(repaired[0].manifest_id(), Some(child.manifest_id));

    let corrupt = fixture(b"restore-corrupt-manifest");
    let (_, child) = publish_delta_chain(Arc::clone(&corrupt.store));
    let path = manifest_path(&corrupt, child.manifest_id);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(path, bytes).unwrap();
    assert!(matches!(
        corrupt.store.restore_candidates(request(600, 1)),
        Err(StoreError::Pack(_)) | Err(StoreError::Authentication(_))
    ));
}

#[test]
fn sequential_and_parallel_paths_report_the_same_first_integrity_error() {
    let fixture = fixture(b"restore-error-order");
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();
    let path = first_chunk_path(&fixture);
    let mut object = std::fs::read(&path).unwrap();
    object[0] ^= 1;
    std::fs::write(path, object).unwrap();

    let mut sequential = ShadowSink::default();
    let sequential_error = plan
        .restore_sequential(&mut sequential, &RestoreCancellation::default())
        .unwrap_err();
    let mut parallel = ShadowSink::default();
    let parallel_error = plan
        .restore_parallel(&mut parallel, &RestoreCancellation::default(), 2)
        .unwrap_err();
    assert_eq!(error_category(&sequential_error), "integrity");
    assert_eq!(
        error_category(&sequential_error),
        error_category(&parallel_error)
    );
    assert!(sequential.aborted && parallel.aborted);
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
}

#[test]
fn short_reads_and_wrong_chunk_ordinals_abort_the_complete_shadow() {
    for wrong_ordinal in [false, true] {
        let fixture = fixture(if wrong_ordinal {
            b"restore-wrong-ordinal"
        } else {
            b"restore-short-read"
        });
        publish_delta_chain(Arc::clone(&fixture.store));
        let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
        let plan = AuthenticatedRestorePlan::build(
            Arc::clone(&fixture.store),
            &candidate,
            RestoreLimits::default(),
        )
        .unwrap();
        let paths = chunk_paths(&fixture);
        assert_eq!(paths.len(), 2);
        if wrong_ordinal {
            let replacement = std::fs::read(&paths[1]).unwrap();
            std::fs::write(&paths[0], replacement).unwrap();
        } else {
            let bytes = std::fs::read(&paths[0]).unwrap();
            std::fs::write(&paths[0], &bytes[..bytes.len() - 1]).unwrap();
        }
        let mut sink = ShadowSink::default();
        assert!(matches!(
            plan.restore_sequential(&mut sink, &RestoreCancellation::default()),
            Err(StoreError::Authentication(_)) | Err(StoreError::Pack(_))
        ));
        assert!(sink.aborted);
        assert!(sink.installed.is_empty());
        assert_eq!(fixture.store.stat().unwrap().pins, 0);
        assert_eq!(
            fixture.store.held_restore_resources().unwrap(),
            kvpack::HeldRestoreResources::default()
        );
    }
}

#[test]
fn chunk_bytes_bound_their_state_and_atomic_group() {
    let fixture = fixture(b"restore-wrong-state");
    let family = two_state_family();
    let mut writer = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        two_state_declaration(256),
        WritePolicy::exact_qualified(id(98), semantic(), &family).unwrap(),
    )
    .unwrap();
    for (name, value) in [("k", 1u8), ("v", 2u8)] {
        let mut state = writer.next_state(StateKey::new(0, name)).unwrap();
        state.write_all(&vec![value; 256 * 4]).unwrap();
        state.finish().unwrap();
    }
    let artifact = writer.commit().unwrap();
    let candidates = fixture
        .store
        .restore_candidates(RestoreRequest {
            semantic_model: semantic(),
            family,
            input_tokens: (0..256u32).collect(),
            auxiliary_inputs: auxiliary(),
            minimum_key_epoch: 1,
            maximum_candidates: 8,
        })
        .unwrap();
    assert_eq!(candidates[0].manifest_id(), Some(artifact.manifest_id));
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidates[0],
        RestoreLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.states().len(), 2);
    assert!(plan.states().iter().all(|state| state.atomic_group == 7));

    let connection =
        rusqlite::Connection::open(fixture.temp.path().join("catalog/catalog.sqlite")).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT state_name,object_key FROM manifest_chunks WHERE tenant=?1 AND manifest_id=?2 ORDER BY state_name",
        )
        .unwrap();
    let rows = statement
        .query_map(
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                artifact.manifest_id.as_slice()
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, "k");
    assert_eq!(rows[1].0, "v");
    let object_path = |raw: &[u8]| {
        let name = raw
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        fixture
            .temp
            .path()
            .join("objects/chunks")
            .join(&name[..2])
            .join(format!("{name}.kvchunk"))
    };
    let key_path = object_path(&rows[0].1);
    let value_bytes = std::fs::read(object_path(&rows[1].1)).unwrap();
    std::fs::write(key_path, value_bytes).unwrap();
    let mut sink = ShadowSink::default();
    assert!(matches!(
        plan.restore_sequential(&mut sink, &RestoreCancellation::default()),
        Err(StoreError::Authentication(_)) | Err(StoreError::Pack(_))
    ));
    assert!(sink.aborted);
    assert!(sink.installed.is_empty());
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
}
