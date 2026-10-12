use super::*;

// Failure-mode evidence (kvpack-spark-prefill docs/FAILURE_MODES.md): these
// cases pin the receiver half of F8/F9/F11/F18 at the same gates the live
// receiver arms from.

#[test]
fn partial_delivery_then_premature_seal_is_rejected_and_abort_cleans() {
    // F8: only the first of two layer planes arrives; a seal attempted at
    // that point must be rejected (frame chain incomplete), and the staging
    // dir must disappear on abort without ever creating the final bundle.
    let (begin, planes, seal, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin, limits).unwrap();
    let staging = stager.staging_path().to_owned();
    stager.ingest(planes[0].0.clone(), &planes[0].1).unwrap();

    assert!(stager.seal(seal).is_err(), "seal over a partial stream");
    assert!(!final_path.exists());
    stager.abort().unwrap();
    assert!(!staging.exists());
    assert!(!final_path.exists());
}

#[test]
fn manifest_clock_window_is_enforced() {
    // F11: the 30 s forward-only clock-skew window (fixture uses a 20 ms
    // window): created too far ahead, a non-positive lifetime, and an
    // already-expired deadline are all rejected at arm time; the boundary
    // values are admitted.
    let (mut begin, _, _, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bundle");

    // Boundary: created exactly now + skew is admitted (now 110, skew 20).
    begin.created_unix_ms = 130;
    begin.deadline_unix_ms = 200;
    BundleStager::create(&path, begin.clone(), limits.clone())
        .unwrap()
        .abort()
        .unwrap();

    let mut future = begin.clone();
    future.created_unix_ms = 131;
    assert!(BundleStager::create(&path, future, limits.clone()).is_err());

    let mut inverted = begin.clone();
    inverted.deadline_unix_ms = inverted.created_unix_ms;
    assert!(BundleStager::create(&path, inverted, limits.clone()).is_err());

    let mut expired = begin.clone();
    expired.created_unix_ms = 90;
    expired.deadline_unix_ms = 105; // now == 110 > 105 -> session already over
    assert!(BundleStager::create(&path, expired, limits).is_err());
}

#[test]
fn weights_precision_closed_set_is_exact_match() {
    // F18: the receiver admits only the closed release set; anything else is
    // rejected before any state exists.
    let (mut begin, _, _, limits) = fixture();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bundle");

    begin.precision.weights = "nvfp4".into();
    BundleStager::create(&path, begin.clone(), limits.clone())
        .unwrap()
        .abort()
        .unwrap();

    begin.precision.weights = "q4_k_xl".into();
    BundleStager::create(&path, begin.clone(), limits.clone())
        .unwrap()
        .abort()
        .unwrap();

    begin.precision.weights = "fp8_e4m3".into();
    assert!(BundleStager::create(&path, begin, limits).is_err());
}
