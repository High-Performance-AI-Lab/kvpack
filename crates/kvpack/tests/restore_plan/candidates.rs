use super::common::*;

#[test]
fn candidates_are_bounded_ancestor_aware_and_include_cut_zero() {
    let fixture = fixture(b"restore-candidates");
    let (root, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let candidates = fixture.store.restore_candidates(request(600, 1)).unwrap();
    assert_eq!(candidates.len(), 3);
    assert_eq!(candidates[0].tier(), RestoreTier::Local);
    assert_eq!(candidates[0].manifest_id(), Some(child.manifest_id));
    assert_eq!(candidates[0].matched_cut().token_count, 512);
    assert_eq!(candidates[0].requested_cut().token_count, 600);
    assert_eq!(candidates[0].suffix_tokens(), 88);
    assert!(candidates[0].chain_identity().is_some());
    assert_eq!(candidates[1].manifest_id(), Some(root.manifest_id));
    assert_eq!(candidates[1].suffix_tokens(), 344);
    assert_eq!(candidates[2].tier(), RestoreTier::Recompute);
    assert_eq!(candidates[2].matched_cut().token_count, 0);
    assert_eq!(candidates[2].suffix_tokens(), 600);

    let stale = fixture.store.restore_candidates(request(600, 2)).unwrap();
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].tier(), RestoreTier::Recompute);

    let metrics = fixture.store.prometheus_metrics().unwrap();
    assert!(
        metrics.contains("kvpack_lookup_total{result=\"ancestor\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("kvpack_lookup_total{result=\"miss\"} 1"),
        "{metrics}"
    );
    assert!(metrics.contains("kvpack_phase_latency_seconds_count{phase=\"exact_lookup\"} 2"));
    assert!(fixture.store.telemetry().pending_spans().unwrap() >= 2);
}

#[test]
fn resident_local_gateway_and_recompute_plans_are_exact_bounded_and_unscored() {
    let fixture = fixture(b"restore-all-tiers");
    let (_, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let baseline = fixture.store.restore_candidates(request(600, 1)).unwrap();
    let local = &baseline[0];
    let cut = local.matched_cut();
    let restored_bytes = local.restored_bytes();
    let mut gateway_resources = local.resources();
    gateway_resources.staging_bytes = restored_bytes;
    gateway_resources.receive_window_bytes = 8 * 1024 * 1024;
    let resident_resources = RestoreResourceRequirements {
        pinned_source_bytes: restored_bytes,
        safety_margin_bytes: 4096,
        source_pins: 1,
        ..RestoreResourceRequirements::default()
    };
    let foreign = super::common::fixture(b"restore-all-tiers-foreign");
    let foreign_cut = input_cut(&foreign.store, 512);
    let sources = vec![
        RestoreAvailableSource::Gateway {
            matched_cut: cut,
            manifest_id: child.manifest_id,
            chain_identity: local.chain_identity().unwrap(),
            key_epoch: 1,
            restored_bytes,
            resources: gateway_resources,
        },
        RestoreAvailableSource::Resident {
            matched_cut: cut,
            resident_identity: id(90),
            restored_bytes,
            resources: resident_resources,
        },
        RestoreAvailableSource::Resident {
            matched_cut: foreign_cut,
            resident_identity: id(91),
            restored_bytes,
            resources: resident_resources,
        },
        RestoreAvailableSource::Gateway {
            matched_cut: cut,
            manifest_id: id(92),
            chain_identity: id(93),
            key_epoch: 0,
            restored_bytes,
            resources: gateway_resources,
        },
    ];
    let candidates = fixture
        .store
        .restore_candidates_with_sources(request(600, 1), &sources)
        .unwrap();
    assert_eq!(candidates.len(), 5);
    assert_eq!(
        candidates
            .iter()
            .map(|item| item.tier())
            .collect::<Vec<_>>(),
        [
            RestoreTier::Resident,
            RestoreTier::Local,
            RestoreTier::Gateway,
            RestoreTier::Local,
            RestoreTier::Recompute,
        ]
    );
    assert_eq!(candidates[0].source_identity(), Some(id(90)));
    assert_eq!(candidates[0].manifest_id(), None);
    assert_eq!(candidates[2].manifest_id(), Some(child.manifest_id));
    assert_eq!(candidates[2].source_key_epoch(), Some(1));
    assert_eq!(candidates[2].suffix_tokens(), 88);
    assert!(matches!(
        AuthenticatedRestorePlan::build(
            Arc::clone(&fixture.store),
            &candidates[2],
            RestoreLimits::default(),
        ),
        Err(StoreError::Expectation(_))
    ));

    let mut bounded_request = request(600, 1);
    bounded_request.maximum_candidates = 3;
    let bounded = fixture
        .store
        .restore_candidates_with_sources(bounded_request, &sources)
        .unwrap();
    assert_eq!(bounded.len(), 3);
    assert_eq!(bounded[0].tier(), RestoreTier::Resident);
    assert_eq!(bounded[1].tier(), RestoreTier::Local);
    assert_eq!(bounded[2].tier(), RestoreTier::Recompute);
}

#[test]
fn exact_boundary_and_partial_final_candidates_name_their_real_cut() {
    let fixture = fixture(b"restore-exact-boundaries");
    let (root, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let exact_root = fixture.store.restore_candidates(request(256, 1)).unwrap();
    assert_eq!(exact_root[0].manifest_id(), Some(root.manifest_id));
    assert_eq!(exact_root[0].matched_cut(), exact_root[0].requested_cut());
    assert_eq!(exact_root[0].suffix_tokens(), 0);
    let exact_child = fixture.store.restore_candidates(request(512, 1)).unwrap();
    assert_eq!(exact_child[0].manifest_id(), Some(child.manifest_id));
    assert_eq!(exact_child[0].matched_cut(), exact_child[0].requested_cut());
    assert_eq!(exact_child[0].suffix_tokens(), 0);
    let one_token_suffix = fixture.store.restore_candidates(request(257, 1)).unwrap();
    assert_eq!(one_token_suffix[0].manifest_id(), Some(root.manifest_id));
    assert_eq!(one_token_suffix[0].suffix_tokens(), 1);

    let family = family();
    let mut partial = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration(300, ManifestKind::Full, 0, 300),
        WritePolicy::exact_qualified(id(94), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut state = partial.next_state(StateKey::new(0, "k")).unwrap();
    state.write_all(&vec![3; 300 * 4]).unwrap();
    state.finish().unwrap();
    let partial = partial.commit().unwrap();
    let candidates = fixture.store.restore_candidates(request(300, 1)).unwrap();
    assert_eq!(candidates[0].manifest_id(), Some(partial.manifest_id));
    assert_eq!(candidates[0].matched_cut(), candidates[0].requested_cut());
    assert_eq!(candidates[0].suffix_tokens(), 0);
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidates[0],
        RestoreLimits::default(),
    )
    .unwrap();
    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    assert_eq!(sink.installed[&StateKey::new(0, "k")], vec![3; 300 * 4]);
    installed.engine_free().unwrap();
}

#[test]
fn cross_family_candidates_and_stale_external_sources_fall_back_to_recompute() {
    let fixture = fixture(b"restore-cross-family");
    publish_delta_chain(Arc::clone(&fixture.store));
    let original_cut = input_cut(&fixture.store, 512);
    let mut other_family = family();
    other_family.engine_cache_abi = id(95);
    let request = RestoreRequest {
        semantic_model: semantic(),
        family: other_family,
        input_tokens: (0..600u32).collect(),
        auxiliary_inputs: auxiliary(),
        minimum_key_epoch: 1,
        maximum_candidates: 8,
    };
    let source = RestoreAvailableSource::Resident {
        matched_cut: original_cut,
        resident_identity: id(96),
        restored_bytes: 2048,
        resources: RestoreResourceRequirements {
            pinned_source_bytes: 2048,
            source_pins: 1,
            ..RestoreResourceRequirements::default()
        },
    };
    let candidates = fixture
        .store
        .restore_candidates_with_sources(request, &[source])
        .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].tier(), RestoreTier::Recompute);
    assert_eq!(candidates[0].matched_cut().token_count, 0);
    assert_eq!(candidates[0].suffix_tokens(), 600);
}
