use super::common::*;

#[test]
fn full_delta_plan_restores_identically_sequential_and_parallel() {
    let fixture = fixture(b"restore-full-delta");
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.matched_cut().token_count, 512);
    assert_eq!(plan.requested_cut().token_count, 600);
    assert_eq!(plan.states()[0].chunk_count, 2);

    let mut sequential = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sequential, &RestoreCancellation::default())
        .unwrap();
    assert_eq!(fixture.store.stat().unwrap().pins, 2);
    assert_eq!(fixture.store.held_restore_count().unwrap(), 1);
    installed.engine_free().unwrap();
    assert_eq!(fixture.store.stat().unwrap().pins, 0);

    let mut parallel = ShadowSink::default();
    let installed = plan
        .restore_parallel(&mut parallel, &RestoreCancellation::default(), 2)
        .unwrap();
    assert_eq!(parallel.installed, sequential.installed);
    assert_eq!(
        parallel.installed[&StateKey::new(0, "k")],
        [vec![1; 256 * 4], vec![2; 256 * 4]].concat()
    );
    let restore_id = installed.restore_id();
    assert_eq!(fixture.store.flush_access_epochs().unwrap(), 2);
    let policy_connection =
        rusqlite::Connection::open(fixture.temp.path().join("catalog/catalog.sqlite")).unwrap();
    let protected: u64 = policy_connection
        .query_row(
            "SELECT COUNT(*) FROM policy_objects WHERE tenant=?1 AND segment='PROTECTED' AND last_persisted_epoch>0",
            [fixture.store.tenant_namespace().as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(protected, 2);
    drop(installed);
    assert_eq!(fixture.store.held_restore_count().unwrap(), 1);
    assert!(fixture.store.acknowledge_engine_free(&restore_id).unwrap());
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
}

#[test]
fn complete_restore_resources_stay_charged_until_engine_free() {
    let fixture = fixture(b"restore-resource-holds");
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let resources = candidate.resources();
    assert!(resources.shadow_bytes > 0);
    assert!(resources.pinned_source_bytes > 0);
    assert!(resources.scratch_bytes_per_task > 0);
    assert!(resources.safety_margin_bytes > 0);
    assert_eq!(resources.source_pins, 2);
    assert_eq!(resources.source_fds, 2);
    let limits = RestoreLimits {
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
    let plan =
        AuthenticatedRestorePlan::build(Arc::clone(&fixture.store), &candidate, limits).unwrap();
    let mut first_sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut first_sink, &RestoreCancellation::default())
        .unwrap();
    let held = fixture.store.held_restore_resources().unwrap();
    assert_eq!(held.active_restores, 1);
    assert_eq!(held.shadow_bytes, resources.shadow_bytes);
    assert_eq!(held.pinned_source_bytes, resources.pinned_source_bytes);
    assert_eq!(held.scratch_bytes, resources.scratch_bytes_per_task);
    assert_eq!(held.staging_bytes, resources.staging_bytes);
    assert_eq!(held.receive_window_bytes, resources.receive_window_bytes);
    assert_eq!(held.safety_margin_bytes, resources.safety_margin_bytes);
    assert_eq!(held.source_pins, resources.source_pins);
    assert_eq!(held.source_fds, resources.source_fds);

    let mut denied_sink = ShadowSink::default();
    assert!(matches!(
        plan.restore_sequential(&mut denied_sink, &RestoreCancellation::default()),
        Err(StoreError::Quota(
            "concurrent restore resources exceed declared limits"
        ))
    ));
    assert!(!denied_sink.begun);
    installed.engine_free().unwrap();
    assert_eq!(
        fixture.store.held_restore_resources().unwrap(),
        kvpack::HeldRestoreResources::default()
    );

    let parallel_limits = RestoreLimits {
        maximum_scratch_bytes: resources.scratch_bytes_per_task * 2,
        maximum_parallelism: 2,
        ..limits
    };
    let parallel_plan =
        AuthenticatedRestorePlan::build(Arc::clone(&fixture.store), &candidate, parallel_limits)
            .unwrap();
    let mut parallel_sink = ShadowSink::default();
    let installed = parallel_plan
        .restore_parallel(&mut parallel_sink, &RestoreCancellation::default(), 2)
        .unwrap();
    assert_eq!(
        fixture
            .store
            .held_restore_resources()
            .unwrap()
            .scratch_bytes,
        resources.scratch_bytes_per_task * 2
    );
    installed.engine_free().unwrap();
}

#[test]
fn cancellation_sink_failure_and_resource_denial_leave_no_install_or_pin() {
    let fixture = fixture(b"restore-failure-atomicity");
    publish_delta_chain(Arc::clone(&fixture.store));
    let candidate = fixture.store.restore_candidates(request(600, 1)).unwrap()[0].clone();
    let limits = RestoreLimits {
        maximum_shadow_bytes: 1,
        ..RestoreLimits::default()
    };
    assert!(matches!(
        AuthenticatedRestorePlan::build(Arc::clone(&fixture.store), &candidate, limits),
        Err(StoreError::Quota(_))
    ));
    let plan = AuthenticatedRestorePlan::build(
        Arc::clone(&fixture.store),
        &candidate,
        RestoreLimits::default(),
    )
    .unwrap();

    let cancellation = RestoreCancellation::default();
    cancellation.cancel();
    let mut cancelled = ShadowSink::default();
    assert!(matches!(
        plan.restore_sequential(&mut cancelled, &cancellation),
        Err(StoreError::Cancelled)
    ));
    assert!(cancelled.installed.is_empty());
    assert_eq!(fixture.store.stat().unwrap().pins, 0);

    let mut failed = ShadowSink {
        fail_write: Some(2),
        ..ShadowSink::default()
    };
    assert!(matches!(
        plan.restore_sequential(&mut failed, &RestoreCancellation::default()),
        Err(StoreError::State("injected scatter failure"))
    ));
    assert!(failed.aborted);
    assert!(failed.installed.is_empty());
    assert_eq!(fixture.store.stat().unwrap().pins, 0);

    let mut begin_failed = ShadowSink {
        fail_begin: true,
        ..ShadowSink::default()
    };
    assert!(matches!(
        plan.restore_sequential(&mut begin_failed, &RestoreCancellation::default()),
        Err(StoreError::State("injected begin failure"))
    ));
    assert!(begin_failed.aborted);
    assert!(begin_failed.installed.is_empty());

    for parallel in [false, true] {
        let midstream = RestoreCancellation::default();
        let mut sink = ShadowSink {
            cancel_after_write: Some((1, midstream.clone())),
            ..ShadowSink::default()
        };
        let result = if parallel {
            plan.restore_parallel(&mut sink, &midstream, 2)
        } else {
            plan.restore_sequential(&mut sink, &midstream)
        };
        assert!(matches!(result, Err(StoreError::Cancelled)));
        assert!(sink.aborted);
        assert!(sink.installed.is_empty());
    }

    let mut commit_failed = ShadowSink {
        fail_commit: true,
        ..ShadowSink::default()
    };
    assert!(matches!(
        plan.restore_sequential(&mut commit_failed, &RestoreCancellation::default()),
        Err(StoreError::State("injected commit failure"))
    ));
    assert!(commit_failed.reset);
    assert!(commit_failed.installed.is_empty());
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
    assert_eq!(
        fixture.store.held_restore_resources().unwrap(),
        kvpack::HeldRestoreResources::default()
    );
}
