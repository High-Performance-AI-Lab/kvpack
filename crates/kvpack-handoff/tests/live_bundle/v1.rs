use super::*;

#[test]
fn python_and_rust_token_digest_fixture_matches() {
    assert_eq!(
        token_ids_sha256(&[10, 20, 30]),
        "2a805bb79d9d3c5c21f55c868215ff419aea7dbcea73527d26db38c481c4dbe2"
    );
}

#[test]
fn staged_bundle_becomes_visible_only_after_a_valid_seal() {
    let (begin, planes, seal, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin.clone(), limits.clone()).unwrap();
    assert!(!final_path.exists());
    for (header, bytes) in planes {
        stager.ingest(header, &bytes).unwrap();
        assert!(!final_path.exists());
    }
    let committed = stager.seal(seal.clone()).unwrap();
    assert_eq!(
        committed,
        std::fs::canonicalize(temp.path())
            .unwrap()
            .join("ready-bundle")
    );

    let verified = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(verified.begin(), &begin);
    assert_eq!(verified.seal(), &seal);
    assert_eq!(verified.planes().len(), 2);
}

#[test]
fn tamper_and_noncanonical_json_fail_closed() {
    let (begin, planes, seal, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin, limits.clone()).unwrap();
    for (header, bytes) in planes {
        stager.ingest(header, &bytes).unwrap();
    }
    stager.seal(seal).unwrap();

    std::fs::write(final_path.join("layers/00000-k.f16le"), vec![9u8; 8]).unwrap();
    assert!(VerifiedBundle::open(&final_path, &limits).is_err());

    let bytes = std::fs::read(final_path.join("begin.json")).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["unexpected"] = serde_json::json!(true);
    std::fs::write(
        final_path.join("begin.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    assert!(VerifiedBundle::open(&final_path, &limits).is_err());
}

#[test]
fn abort_removes_only_run_owned_partial_state() {
    let (begin, planes, _seal, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin, limits).unwrap();
    let staging = stager.staging_path().to_owned();
    stager.ingest(planes[0].0.clone(), &planes[0].1).unwrap();
    assert!(staging.exists());
    assert!(!final_path.exists());
    stager.abort().unwrap();
    assert!(!staging.exists());
    assert!(!final_path.exists());
}

#[test]
fn canonical_seal_round_trips_with_flattened_core() {
    let (_, _, seal, _) = fixture();
    let bytes = canonical_json(&seal).unwrap();
    let parsed: SealManifestV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, seal);
}

#[test]
fn incremental_verifier_rolls_back_after_payload_failure() {
    let (begin, planes, seal, limits) = fixture();
    let mut verifier = IncrementalVerifierV1::new(begin, limits).unwrap();
    assert!(verifier
        .verify_plane(planes[0].0.clone(), vec![0u8; 8])
        .is_err());
    assert_eq!(verifier.next_sequence(), 0);
    assert_eq!(verifier.payload_bytes(), 0);
    for (header, bytes) in planes {
        verifier.verify_plane(header, bytes).unwrap();
    }
    assert_eq!(verifier.next_sequence(), 2);
    verifier.verify_seal(seal).unwrap();
}

#[test]
fn coordinator_emits_once_after_v_and_prepare_is_not_publication() {
    let (begin, planes, seal, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut coordinator =
        StreamingCoordinatorV1::create(&final_path, begin, limits.clone()).unwrap();
    let pool = LayerPermitPoolV1::experiment_v2();
    let permit = pool.acquire().unwrap();
    assert!(coordinator
        .ingest_plane(planes[0].0.clone(), planes[0].1.clone(), Some(permit))
        .unwrap()
        .is_none());
    assert_eq!(coordinator.ready_layers(), 0);
    let ready = coordinator
        .ingest_plane(planes[1].0.clone(), planes[1].1.clone(), None)
        .unwrap()
        .expect("V completes exactly one layer event");
    assert_eq!(ready.layer(), 0);
    assert_eq!(coordinator.ready_layers(), 1);
    assert_eq!(pool.in_use().unwrap(), 1);
    drop(ready);
    assert_eq!(pool.in_use().unwrap(), 0);

    coordinator.verify_and_prepare_seal(seal).unwrap();
    assert!(!final_path.exists());
    assert!(!coordinator.staging_path().join("READY").exists());
    coordinator.publish().unwrap();
    VerifiedBundle::open(&final_path, &limits).unwrap();
}
