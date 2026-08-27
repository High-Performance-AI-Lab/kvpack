use super::common::*;

fn publish_and_plan(fixture: &Fixture) -> AuthenticatedRestorePlan {
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap()
}

#[test]
fn shadow_promote_with_exact_key_installs_without_a_second_read() {
    let fixture = fixture(b"shadow-promote-no-reread");
    let plan = publish_and_plan(&fixture);
    let shadow = plan
        .prestage_shadow(&RestoreCancellation::default())
        .unwrap();
    assert_eq!(shadow.manifest_id(), plan.manifest_id());
    assert_eq!(shadow.matched_cut().token_count, 512);
    assert!(shadow.staged_bytes() > 0);
    assert_eq!(fixture.store.held_restore_count().unwrap(), 1);

    // Overwrite every source chunk on disk after staging. Any re-read during
    // promotion would now fail integrity or install garbage; a true
    // memory-only transfer is unaffected.
    for path in chunk_paths(&fixture) {
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, vec![0xA5; bytes.len()]).unwrap();
    }

    let mut sink = ShadowSink::default();
    let installed = shadow
        .promote_if(
            plan.manifest_id(),
            &mut sink,
            &RestoreCancellation::default(),
        )
        .unwrap();
    assert_eq!(
        sink.installed[&StateKey::new(0, "k")],
        [vec![1; 256 * 4], vec![2; 256 * 4]].concat()
    );
    installed.engine_free().unwrap();
    assert_eq!(fixture.store.held_restore_count().unwrap(), 0);
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
}

#[test]
fn shadow_promote_with_wrong_key_aborts_and_fails_closed() {
    let fixture = fixture(b"shadow-promote-wrong-key");
    let plan = publish_and_plan(&fixture);
    let shadow = plan
        .prestage_shadow(&RestoreCancellation::default())
        .unwrap();
    let restore_id = shadow.restore_id();

    let mut sink = ShadowSink::default();
    let error = shadow
        .promote_if(id(77), &mut sink, &RestoreCancellation::default())
        .unwrap_err();
    assert_eq!(error_category(&error), "integrity");
    assert!(!sink.begun);
    // The abort released the reservation and source pins completely.
    assert_eq!(fixture.store.held_restore_count().unwrap(), 0);
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
    assert!(!fixture.store.acknowledge_engine_free(&restore_id).unwrap());

    // The normal exact restore path is unaffected by the aborted shadow.
    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    assert_eq!(
        sink.installed[&StateKey::new(0, "k")],
        [vec![1; 256 * 4], vec![2; 256 * 4]].concat()
    );
    installed.engine_free().unwrap();
}

#[test]
fn shadow_abort_discards_everything_and_exact_restore_is_unaffected() {
    let fixture = fixture(b"shadow-abort-clean");
    let plan = publish_and_plan(&fixture);
    let shadow = plan
        .prestage_shadow(&RestoreCancellation::default())
        .unwrap();
    assert_eq!(fixture.store.held_restore_count().unwrap(), 1);
    let restore_id = shadow.restore_id();

    shadow.abort().unwrap();
    assert_eq!(fixture.store.held_restore_count().unwrap(), 0);
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
    // The reservation is gone; a late acknowledgement is rejected.
    assert!(!fixture.store.acknowledge_engine_free(&restore_id).unwrap());

    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    assert_eq!(
        sink.installed[&StateKey::new(0, "k")],
        [vec![1; 256 * 4], vec![2; 256 * 4]].concat()
    );
    installed.engine_free().unwrap();
}

#[test]
fn shadow_promote_cancelled_aborts_instead_of_committing() {
    let fixture = fixture(b"shadow-promote-cancelled");
    let plan = publish_and_plan(&fixture);
    let shadow = plan
        .prestage_shadow(&RestoreCancellation::default())
        .unwrap();

    let cancellation = RestoreCancellation::default();
    cancellation.cancel();
    let mut sink = ShadowSink::default();
    assert!(matches!(
        shadow.promote_if(plan.manifest_id(), &mut sink, &cancellation),
        Err(StoreError::Cancelled)
    ));
    assert!(!sink.begun);
    assert_eq!(fixture.store.held_restore_count().unwrap(), 0);
}

#[test]
fn shadow_prestage_respects_cancellation_and_quota() {
    let fixture = fixture(b"shadow-prestage-gates");
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();

    let cancellation = RestoreCancellation::default();
    cancellation.cancel();
    assert!(matches!(
        plan.prestage_shadow(&cancellation),
        Err(StoreError::Cancelled)
    ));
    assert_eq!(fixture.store.held_restore_count().unwrap(), 0);

    // One shadow holds the full reservation; a second concurrent pre-stage
    // under exact-fit limits is denied by the existing quota gate.
    let resources = candidate.resources();
    let tight = RestoreLimits {
        maximum_shadow_bytes: resources.shadow_bytes,
        maximum_pinned_source_bytes: resources.pinned_source_bytes,
        maximum_scratch_bytes: resources.scratch_bytes_per_task,
        maximum_staging_bytes: resources.staging_bytes,
        maximum_receive_window_bytes: resources.receive_window_bytes,
        maximum_safety_margin_bytes: resources.safety_margin_bytes,
        maximum_source_pins: resources.source_pins,
        maximum_source_fds: resources.source_fds,
        maximum_parallelism: 1,
    };
    let tight_plan =
        AuthenticatedRestorePlan::build(Arc::clone(&fixture.store), &candidate, tight).unwrap();
    let shadow = tight_plan
        .prestage_shadow(&RestoreCancellation::default())
        .unwrap();
    assert!(matches!(
        tight_plan.prestage_shadow(&RestoreCancellation::default()),
        Err(StoreError::Quota(
            "concurrent restore resources exceed declared limits"
        ))
    ));
    shadow.abort().unwrap();
    assert_eq!(
        fixture.store.held_restore_resources().unwrap(),
        kvpack::HeldRestoreResources::default()
    );
}
