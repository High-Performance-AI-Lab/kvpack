use super::*;
use kvpack_handoff::{CanaryRecord, PORTABLE_KV_ABI_V2_PREROPE};

// ── Protocol v2 pre-RoPE capture family (canonical-kv-prerope-v2) ──────
// Single-class fixture: layers 0..2, kv_heads 1, head_dim 4, cached 3
// tokens. Under the pre-RoPE ABI the Key planes are f32-LE (48 bytes per
// plane) and the Value planes stay f16-LE (24 bytes per plane): 4 frames,
// 2 * (48 + 24) = 144 payload bytes.

const PREROPE_KEY_PLANE_BYTES: usize = 3 * 4 * 4;
const PREROPE_VALUE_PLANE_BYTES: usize = 3 * 4 * 2;

fn prerope_fixture() -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
    let tokens = vec![10, 20, 30, 40];
    let limits = ValidationLimits {
        max_cached_tokens: 32,
        max_clock_skew_ms: 20,
        max_context_tokens: 64,
        max_frame_bytes: 1024,
        max_layers: 64,
        max_session_ms: 1000,
        max_total_bytes: 4096,
        now_unix_ms: 110,
    };
    let class = LayoutClassV2 {
        class: "gqa-full".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 0,
        head_dim: 4,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 2,
        window_tokens: 0,
    };
    let begin = BeginManifestV1 {
        cached_token_count: 3,
        created_unix_ms: 100,
        deadline_unix_ms: 200,
        endpoints: EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-qwen25-f16-v1".into(),
            consumer_node: "mac-m3-ultra".into(),
            producer_engine_abi: "ferrite-qwen25-f16-v1".into(),
            producer_node: "mac-m3-ultra".into(),
            trust_domain: "lab-prefill".into(),
        },
        expected_layer_frames: 4,
        expected_payload_bytes: (2 * (PREROPE_KEY_PLANE_BYTES + PREROPE_VALUE_PLANE_BYTES)) as u64,
        geometry: GeometryV1 {
            head_dim: 4,
            max_context_tokens: 32,
            num_kv_heads: 1,
            num_layers: 2,
        },
        identity: ExactIdentityV1 {
            adapter_sha256: "0".repeat(64),
            chat_template_sha256: "1".repeat(64),
            context_policy_sha256: "2".repeat(64),
            model_revision: "repo@revision".into(),
            model_sha256: "3".repeat(64),
            tokenizer_revision: "tokenizer@revision".into(),
            tokenizer_sha256: "4".repeat(64),
        },
        portable_abi: PORTABLE_KV_ABI_V2_PREROPE.into(),
        precision: PrecisionV1 {
            compute: "float16".into(),
            kv: "float16".into(),
            weights: "q4_k_m".into(),
        },
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: "c".repeat(64),
        layout_table: vec![class],
        schedule: None,
        hmac_key_id: None,
    };
    let mut planes = Vec::new();
    let mut sequence = 0u32;
    for layer in 0..2 {
        for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
            let (bytes, dtype) = if role == TensorRoleV1::Key {
                (
                    vec![5u8; PREROPE_KEY_PLANE_BYTES],
                    Some("float32".to_string()),
                )
            } else {
                (vec![9u8; PREROPE_VALUE_PLANE_BYTES], None)
            };
            planes.push((
                LayerHeaderV1 {
                    byte_length: bytes.len() as u64,
                    layer,
                    logical_token_end: 3,
                    logical_token_start: 0,
                    role,
                    schema_version: LIVE_HANDOFF_SCHEMA_V1,
                    sequence,
                    sha256: sha256_hex(&bytes),
                    shape: [3, 1, 4],
                    transfer_id: begin.transfer_id.clone(),
                    dtype,
                    layout_class: Some("gqa-full".into()),
                },
                bytes,
            ));
            sequence += 1;
        }
    }
    let headers = planes
        .iter()
        .map(|(header, _)| header.clone())
        .collect::<Vec<_>>();
    let mut payload_hash = Sha256::new();
    for (_, bytes) in &planes {
        payload_hash.update(bytes);
    }
    let core = SealCoreV1 {
        completed_unix_ms: 150,
        descriptor_chain_sha256: descriptor_chain_sha256(&headers).unwrap(),
        frame_count: 4,
        payload_bytes: begin.expected_payload_bytes,
        payload_sha256: hex::encode(payload_hash.finalize()),
        prompt_token_ids: tokens.clone(),
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: begin.transfer_id.clone(),
        canary: Some(CanaryRecord {
            sample_token_start: 0,
            sample_token_count: 1,
            post_rope_k_sha256: "d".repeat(64),
            post_rope_v_sha256: "e".repeat(64),
        }),
    };
    let seal = SealManifestV1 {
        artifact_sha256: artifact_sha256(&begin, &headers, &core).unwrap(),
        artifact_hmac_sha256: None,
        core,
    };
    (begin, planes, seal, limits)
}

#[test]
fn prerope_begin_validates_and_planes_stage_and_open() {
    let (begin, planes, seal, limits) = prerope_fixture();
    assert!(begin.is_prerope_v2());
    begin.validate(&limits).expect("pre-rope begin is valid");
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin.clone(), limits.clone()).unwrap();
    for (header, bytes) in planes {
        stager.ingest(header, &bytes).unwrap();
    }
    stager.seal(seal.clone()).unwrap();
    // The staged K payloads carry the f32 extension; V keeps f16le.
    let entries = std::fs::read_dir(final_path.join("layers"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(entries.iter().any(|name| name == "00000-k.f32le"));
    assert!(entries.iter().any(|name| name == "00001-k.f32le"));
    assert!(entries.iter().any(|name| name == "00000-v.f16le"));
    assert!(!entries.iter().any(|name| name.ends_with("-k.f16le")));
    let verified = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(verified.begin(), &begin);
    assert_eq!(verified.seal(), &seal);
    assert_eq!(verified.planes().len(), 4);
}

#[test]
fn prerope_begin_rejects_post_rope_byte_accounting() {
    let (begin, _, _, limits) = prerope_fixture();
    // All-f16 accounting undercounts the f32 Key planes and must fail.
    let mut drifted = begin.clone();
    drifted.expected_payload_bytes = 2 * (24 + 24);
    assert!(drifted.validate(&limits).is_err());
    // So does a frame-count drift.
    let mut drifted = begin.clone();
    drifted.expected_layer_frames = 3;
    assert!(drifted.validate(&limits).is_err());
}

#[test]
fn prerope_key_frames_require_the_float32_tag_and_width() {
    let (begin, planes, _, _) = prerope_fixture();
    let key = &planes[0].0;

    // Absent tag keeps meaning "float16" — never acceptable for a pre-RoPE
    // Key plane.
    let mut drifted = key.clone();
    drifted.dtype = None;
    assert!(drifted.validate_for(&begin, 0).is_err());
    // An explicit float16 tag is just as wrong.
    let mut drifted = key.clone();
    drifted.dtype = Some("float16".into());
    assert!(drifted.validate_for(&begin, 0).is_err());
    // f16 byte width on a Key frame fails the f32 byte bound.
    let mut drifted = key.clone();
    drifted.byte_length = PREROPE_VALUE_PLANE_BYTES as u64;
    assert!(drifted.validate_for(&begin, 0).is_err());
    // The honest frame passes.
    key.validate_for(&begin, 0).unwrap();
    // Value planes keep post-RoPE rules: absent tag means float16.
    let value = &planes[1].0;
    value.validate_for(&begin, 1).unwrap();
    let mut drifted = value.clone();
    drifted.dtype = Some("float32".into());
    assert!(drifted.validate_for(&begin, 1).is_err());
}

#[test]
fn unknown_prerope_abi_labels_fail_closed() {
    let (begin, _, _, limits) = prerope_fixture();
    // A newer/older pre-RoPE label this build does not know is rejected
    // exactly like any unknown ABI.
    let mut drifted = begin.clone();
    drifted.portable_abi = "canonical-kv-prerope-v3".into();
    assert!(drifted.validate(&limits).is_err());
    // The post-RoPE v2 label on a pre-RoPE-accounted begin also fails: the
    // declared payload bytes no longer match the role accounting.
    let mut drifted = begin.clone();
    drifted.portable_abi = PORTABLE_KV_ABI_V2.into();
    assert!(drifted.validate(&limits).is_err());
}

// ── F4/F7: finite gate and adversarial mutation for the pre-RoPE family ──
// The plain SHA-256 authenticates bytes, not values; these cases pin the
// fail-closed gates a forged or corrupted pre-RoPE plane must hit.

/// Build a fresh verifier over the fixture begin and run one plane through
/// `verify_plane`, returning the outcome so each case can assert its gate.
fn verify_one_key_plane(
    header: LayerHeaderV1,
    payload: Vec<u8>,
) -> kvpack_handoff::Result<kvpack_handoff::VerifiedPlaneV1> {
    let (begin, _, _, limits) = prerope_fixture();
    let mut verifier = IncrementalVerifierV1::new(begin, limits).unwrap();
    verifier.verify_plane(header, payload)
}

#[test]
fn prerope_key_plane_with_nan_is_rejected_by_the_finite_gate() {
    // F4: a NaN in an authenticated f32 Key plane (producer bug or an
    // in-plane bit-flip) must not flow into the f16 cache. The header's
    // SHA-256 is recomputed over the corrupted bytes so the hash gate
    // passes and the value gate is what rejects.
    let (_, planes, _, _) = prerope_fixture();
    let (header, _) = &planes[0];
    let mut bytes = vec![0u8; PREROPE_KEY_PLANE_BYTES];
    // f32 quiet NaN = 0x7fc00000, little-endian.
    bytes[..4].copy_from_slice(&0x7fc00000u32.to_le_bytes());
    let mut corrupted = header.clone();
    corrupted.sha256 = sha256_hex(&bytes);
    let err = verify_one_key_plane(corrupted, bytes).unwrap_err();
    assert!(
        matches!(err, kvpack_handoff::HandoffError::Validation(ref m) if m.contains("non-finite")),
        "got {err:?}"
    );
}

#[test]
fn prerope_key_plane_with_inf_is_rejected_by_the_finite_gate() {
    let (_, planes, _, _) = prerope_fixture();
    let (header, _) = &planes[0];
    let mut bytes = vec![0u8; PREROPE_KEY_PLANE_BYTES];
    // f32 +Inf = 0x7f800000, little-endian.
    bytes[..4].copy_from_slice(&0x7f800000u32.to_le_bytes());
    let mut corrupted = header.clone();
    corrupted.sha256 = sha256_hex(&bytes);
    let err = verify_one_key_plane(corrupted, bytes).unwrap_err();
    assert!(
        matches!(err, kvpack_handoff::HandoffError::Validation(ref m) if m.contains("non-finite")),
        "got {err:?}"
    );
}

#[test]
fn prerope_value_planes_are_not_subject_to_the_key_finite_gate() {
    // F4 is scoped to Key planes (the f32 family); a Value plane (f16) is
    // never scanned by the finite gate. Ingest the Key plane first, then
    // the Value plane at its own sequence — both pass, confirming the
    // gate never inspects Value bytes.
    let (begin, planes, _, limits) = prerope_fixture();
    let mut verifier = IncrementalVerifierV1::new(begin, limits).unwrap();
    let (key_header, key_bytes) = &planes[0];
    assert_eq!(key_header.role, TensorRoleV1::Key);
    verifier
        .verify_plane(key_header.clone(), key_bytes.clone())
        .unwrap();
    let (value_header, value_bytes) = &planes[1];
    assert_eq!(value_header.role, TensorRoleV1::Value);
    verifier
        .verify_plane(value_header.clone(), value_bytes.clone())
        .unwrap();
}

#[test]
fn prerope_role_swap_is_rejected() {
    // F7: a Key plane presented under a Value role (or vice versa) must
    // fail closed at the descriptor gate — the layout table pins the role
    // at each sequence.
    let (_, planes, _, _) = prerope_fixture();
    let (mut header, bytes) = planes[0].clone();
    assert_eq!(header.role, TensorRoleV1::Key);
    header.role = TensorRoleV1::Value;
    header.dtype = None;
    assert!(verify_one_key_plane(header, bytes).is_err());
}

#[test]
fn prerope_oversized_key_plane_is_rejected_by_the_shape_gate() {
    // F7: a Key plane claiming a wider shape (head_dim 8 vs 4) is rejected
    // by the exact-shape check, not admitted by byte-count alone.
    let (_, planes, _, _) = prerope_fixture();
    let (mut header, _) = planes[0].clone();
    let oversized = vec![0u8; 3 * 8 * 4];
    header.byte_length = oversized.len() as u64;
    header.shape = [3, 1, 8];
    header.sha256 = sha256_hex(&oversized);
    assert!(verify_one_key_plane(header, oversized).is_err());
}

#[test]
fn prerope_bundle_with_an_extra_layer_file_is_rejected() {
    // F7 (duplicated plane): a second copy of a layer file inside the
    // sealed bundle must fail the exact-entry-set reopen check.
    let (begin, planes, seal, limits) = prerope_fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin.clone(), limits.clone()).unwrap();
    for (header, bytes) in planes {
        stager.ingest(header, &bytes).unwrap();
    }
    stager.seal(seal.clone()).unwrap();
    // Drop an extra, fully-authenticated-looking duplicate into the layers
    // directory. It is not in the declared layout, so reopen rejects it.
    let extra = final_path.join("layers").join("00000-k.dup");
    std::fs::write(&extra, b"stray").unwrap();
    assert!(VerifiedBundle::open(&final_path, &limits).is_err());
}

// ── F2/F7: canary enforcement for the pre-RoPE family ──

#[test]
fn prerope_seal_without_a_canary_is_rejected() {
    // F2: the contract claims the family is canary-gated; a pre-RoPE seal
    // with no canary record must fail `validate_for` even when its
    // artifact hash is otherwise consistent.
    let (begin, planes, seal, _) = prerope_fixture();
    let headers: Vec<LayerHeaderV1> = planes.iter().map(|(h, _)| h.clone()).collect();
    let mut core = seal.core.clone();
    core.canary = None;
    let unkeyed = SealManifestV1 {
        artifact_sha256: artifact_sha256(&begin, &headers, &core).unwrap(),
        artifact_hmac_sha256: None,
        core,
    };
    let err = unkeyed
        .validate_for(
            &begin,
            &headers,
            begin.expected_payload_bytes,
            &seal.core.payload_sha256,
        )
        .unwrap_err();
    assert!(
        matches!(err, kvpack_handoff::HandoffError::Validation(ref m)
            if m.contains("does not authenticate")),
        "got {err:?}"
    );
}

#[test]
fn prerope_canary_with_a_bad_digest_or_window_is_rejected() {
    let (begin, planes, seal, _) = prerope_fixture();
    let headers: Vec<LayerHeaderV1> = planes.iter().map(|(h, _)| h.clone()).collect();
    let payload_sha = seal.core.payload_sha256.clone();

    // Malformed digest.
    let mut bad_digest = seal.core.clone();
    bad_digest.canary = Some(CanaryRecord {
        sample_token_start: 0,
        sample_token_count: 1,
        post_rope_k_sha256: "z".repeat(64),
        post_rope_v_sha256: "e".repeat(64),
    });
    let bad_digest_seal = SealManifestV1 {
        artifact_sha256: artifact_sha256(&begin, &headers, &bad_digest).unwrap(),
        artifact_hmac_sha256: None,
        core: bad_digest,
    };
    assert!(bad_digest_seal
        .validate_for(&begin, &headers, begin.expected_payload_bytes, &payload_sha)
        .is_err());

    // Window outside the cached token range (cached_token_count = 3).
    let mut bad_window = seal.core.clone();
    bad_window.canary = Some(CanaryRecord {
        sample_token_start: 3,
        sample_token_count: 1,
        post_rope_k_sha256: "d".repeat(64),
        post_rope_v_sha256: "e".repeat(64),
    });
    let bad_window_seal = SealManifestV1 {
        artifact_sha256: artifact_sha256(&begin, &headers, &bad_window).unwrap(),
        artifact_hmac_sha256: None,
        core: bad_window,
    };
    assert!(bad_window_seal
        .validate_for(&begin, &headers, begin.expected_payload_bytes, &payload_sha)
        .is_err());
}

#[test]
fn prerope_canary_verify_against_matches_and_rejects() {
    // F2 engine gate scaffold: the consumer compares its pinned-kernel
    // row digests to the producer-recorded canary.
    let record = CanaryRecord {
        sample_token_start: 0,
        sample_token_count: 1,
        post_rope_k_sha256: "d".repeat(64),
        post_rope_v_sha256: "e".repeat(64),
    };
    record
        .verify_against(&"d".repeat(64), &"e".repeat(64))
        .unwrap();
    assert!(record
        .verify_against(&"d".repeat(64), &"f".repeat(64))
        .is_err());
    assert!(record
        .verify_against(&"0".repeat(64), &"e".repeat(64))
        .is_err());
}

// ── F1/F7: keyed HMAC authentication of the artifact ──

#[test]
fn prerope_artifact_hmac_round_trip_and_forge_rejection() {
    // F1: a deployment that arms a tenant MAC key stamps the seal and the
    // consumer verifies under the same key. A wrong key, a stripped tag,
    // or a tampered begin all fail closed.
    let (begin, planes, seal, _) = prerope_fixture();
    let headers: Vec<LayerHeaderV1> = planes.iter().map(|(h, _)| h.clone()).collect();
    let core = seal.core.clone();
    let key = MacKey::from_hex(&"a".repeat(64)).unwrap();
    let tag = artifact_hmac_sha256(&begin, &headers, &core, &key).unwrap();

    let mut sealed = seal.clone();
    sealed.artifact_hmac_sha256 = Some(tag);
    // Right key accepts; a plain SHA-256 is never a valid tag.
    sealed.authenticate_hmac(&begin, &headers, &key).unwrap();
    assert!(sealed
        .authenticate_hmac(
            &begin,
            &headers,
            &MacKey::from_hex(&hex::encode([0u8; 32])).unwrap()
        )
        .is_err());

    // Wrong key rejects the producer's tag.
    let other = MacKey::from_hex(&"b".repeat(64)).unwrap();
    assert!(sealed.authenticate_hmac(&begin, &headers, &other).is_err());

    // Stripped tag rejects.
    let mut stripped = sealed.clone();
    stripped.artifact_hmac_sha256 = None;
    assert!(stripped.authenticate_hmac(&begin, &headers, &key).is_err());

    // Tampered begin rejects (the tag is bound to the authenticated begin).
    let mut tampered = begin.clone();
    tampered.transfer_id = "9".repeat(64);
    assert!(sealed.authenticate_hmac(&tampered, &headers, &key).is_err());
}

#[test]
fn prerope_seal_with_mismatched_hmac_shape_is_rejected_at_validation() {
    // F1: an hmac_key_id on the begin with no tag on the seal (or a
    // malformed tag) fails the shape check inside `validate_for`, so an
    // integrity-only verifier never needs a key to refuse the mismatch.
    let (begin, planes, seal, _) = prerope_fixture();
    let headers: Vec<LayerHeaderV1> = planes.iter().map(|(h, _)| h.clone()).collect();
    let mut untagged = seal.clone();
    untagged.artifact_hmac_sha256 = None;
    let mut keyed_begin = begin.clone();
    keyed_begin.hmac_key_id = Some("tenant-a".into());
    // Recompute the artifact hash so the only failing clause is the HMAC
    // shape rule.
    untagged.artifact_sha256 = artifact_sha256(&keyed_begin, &headers, &untagged.core).unwrap();
    assert!(untagged
        .validate_for(
            &keyed_begin,
            &headers,
            keyed_begin.expected_payload_bytes,
            &seal.core.payload_sha256
        )
        .is_err());
}
