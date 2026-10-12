use super::common::*;

#[test]
fn eighth_append_compacts_references_without_rereading_prior_payloads() {
    let fixture = fixture(b"reference-only-compaction");
    let family = family();
    let mut root = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration(1, ManifestKind::Full, 0, 1),
        WritePolicy::exact_qualified(id(70), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = root.next_state(StateKey::new(0, "k")).unwrap();
    source.write_all(&[1; 4]).unwrap();
    source.finish().unwrap();
    let mut parent = root.commit().unwrap();
    let mut before_compaction = BTreeMap::new();
    let mut seventh_delta = None;

    for child_count in 2..=9usize {
        if child_count == 9 {
            seventh_delta = Some(parent);
            before_compaction = chunk_snapshot(&fixture);
            assert_eq!(before_compaction.len(), 8);
        }
        let parent_tokens = (0..(child_count - 1) as u32).collect::<Vec<_>>();
        let parent_cut = fixture
            .store
            .derive_input_cut(&semantic(), &family, &parent_tokens, &auxiliary())
            .unwrap()
            .0;
        let mut child = ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            declaration(
                child_count,
                ManifestKind::Delta {
                    parent: parent.manifest_id,
                    parent_cut,
                    depth: 99,
                },
                (child_count - 1) as u64,
                1,
            ),
            WritePolicy::exact_qualified(id(69 + child_count as u8), semantic(), &family).unwrap(),
        )
        .unwrap();
        let mut source = child.next_state(StateKey::new(0, "k")).unwrap();
        source.write_all(&[child_count as u8; 4]).unwrap();
        source.finish().unwrap();
        parent = child.commit().unwrap();
    }

    assert_eq!(fixture.store.stat().unwrap().manifests, 9);
    assert_eq!(fixture.store.stat().unwrap().chunks, 9);
    for (path, bytes) in before_compaction {
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
    let connection =
        rusqlite::Connection::open(fixture.temp.path().join("catalog/catalog.sqlite")).unwrap();
    let final_depth: u64 = connection
        .query_row(
            "SELECT parent_depth FROM manifests WHERE tenant=?1 AND manifest_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                parent.manifest_id.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(final_depth, 0);

    let before_replay = fixture.store.stat().unwrap();
    let parent_cut = fixture
        .store
        .derive_input_cut(
            &semantic(),
            &family,
            &(0..8u32).collect::<Vec<_>>(),
            &auxiliary(),
        )
        .unwrap()
        .0;
    let replay = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration(
            9,
            ManifestKind::Delta {
                parent: seventh_delta.unwrap().manifest_id,
                parent_cut,
                depth: 0,
            },
            8,
            1,
        ),
        WritePolicy::exact_qualified(id(78), semantic(), &family).unwrap(),
    )
    .unwrap();
    assert_eq!(replay.published_replay(), Some(parent));
    assert_eq!(replay.commit().unwrap(), parent);
    assert_eq!(fixture.store.stat().unwrap(), before_replay);

    let candidate = fixture.store.restore_candidates(request(9, 1)).unwrap()[0].clone();
    assert_eq!(candidate.manifest_id(), Some(parent.manifest_id));
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.states()[0].chunk_count, 9);
    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    let expected = (1..=9u8).flat_map(|value| [value; 4]).collect::<Vec<_>>();
    assert_eq!(sink.installed[&StateKey::new(0, "k")], expected);
    installed.engine_free().unwrap();
}

#[test]
fn capacity_maintenance_retains_pinned_chains_then_evicts_leaf_first() {
    let fixture = fixture(b"restore-gc-pins");
    let (root, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();
    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    let before = fixture.store.stat().unwrap().durable_bytes;
    let connection =
        rusqlite::Connection::open(fixture.temp.path().join("catalog/catalog.sqlite")).unwrap();
    let child_manifest_bytes: u64 = connection
        .query_row(
            "SELECT file_bytes FROM manifests WHERE tenant=?1 AND manifest_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                child.manifest_id.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    let child_chunk_bytes: u64 = connection
        .query_row(
            "SELECT SUM(c.object_bytes) FROM manifest_chunks mc JOIN chunks c ON c.tenant=mc.tenant AND c.object_key=mc.object_key WHERE mc.tenant=?1 AND mc.manifest_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                child.manifest_id.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    let retained_parent_bytes = before
        .saturating_sub(child_manifest_bytes)
        .saturating_sub(child_chunk_bytes);
    let capacity = retained_parent_bytes.saturating_mul(4).saturating_add(2) / 3;

    let held = fixture
        .store
        .maintain_capacity(capacity, UtilizationPolicy::default(), 64)
        .unwrap();
    assert!(held.blocked);
    assert_eq!(held.before_bytes, held.after_bytes);
    assert_eq!(held.manifests_evicted, 0);
    assert_eq!(held.chunks_evicted, 0);

    installed.engine_free().unwrap();
    let reclaimed = fixture
        .store
        .maintain_capacity(capacity, UtilizationPolicy::default(), 64)
        .unwrap();
    assert!(!reclaimed.blocked);
    assert!(reclaimed.manifests_evicted >= 1);
    assert!(reclaimed.chunks_evicted >= 1);
    assert!(reclaimed.after_bytes <= capacity.saturating_mul(75) / 100);
    assert_eq!(reclaimed.after_bytes, retained_parent_bytes);
    assert!(manifest_path(&fixture, root.manifest_id).exists());
    assert!(!manifest_path(&fixture, child.manifest_id).exists());
    let metrics = fixture.store.prometheus_metrics().unwrap();
    assert!(!metrics.contains("kvpack_lifecycle_total{transition=\"tombstoned\"} 0\n"));
    assert!(!metrics.contains("kvpack_lifecycle_total{transition=\"collected\"} 0\n"));
    let candidates = fixture.store.restore_candidates(request(600, 1)).unwrap();
    assert_eq!(candidates[0].manifest_id(), Some(root.manifest_id));
    assert_eq!(candidates[1].tier(), RestoreTier::Recompute);
}
