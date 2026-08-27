use super::*;

/// Two-class fixture with the FULL class declared first and the windowed
/// class second, so `decode-priority` genuinely permutes the walk:
/// declared order streams layers 0,1 (full history) before 2,3 (newest
/// cuts), while `decode-priority` streams the windowed class first. Per
/// layer the roles stay K-then-V as declared.
fn v2_schedule_fixture(
    schedule: Option<&str>,
) -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
    let tokens = vec![10, 20, 30, 40, 50];
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
    let class_full = LayoutClassV2 {
        class: "gqa-full".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 0,
        head_dim: 2,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 2,
        window_tokens: 0,
    };
    let class_windowed = LayoutClassV2 {
        class: "gqa-windowed".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 2,
        head_dim: 2,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 4,
        window_tokens: 2,
    };
    let mut begin = BeginManifestV1 {
        cached_token_count: 4,
        created_unix_ms: 100,
        deadline_unix_ms: 200,
        endpoints: EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-gemma4-f16-v1".into(),
            consumer_node: "mac-m3-ultra".into(),
            producer_engine_abi: "llamacpp-gb10-v1".into(),
            producer_node: "producer-node".into(),
            trust_domain: "lab-prefill".into(),
        },
        expected_layer_frames: 8,
        expected_payload_bytes: 96,
        geometry: GeometryV1 {
            head_dim: 0,
            max_context_tokens: 32,
            num_kv_heads: 0,
            num_layers: 4,
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
        portable_abi: PORTABLE_KV_ABI_V2.into(),
        precision: PrecisionV1 {
            compute: "float16".into(),
            kv: "float16".into(),
            weights: "bf16".into(),
        },
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: "c".repeat(64),
        layout_table: vec![class_full, class_windowed],
        schedule: None,
        hmac_key_id: None,
    };
    // Plane order follows the schedule exactly as the producer would send
    // it: decode-priority puts the windowed (newest-cut) class first;
    // anything else is the declared class order. Ranges: full planes cover
    // 0..4, windowed planes cover the trailing 2..4.
    let decode_priority = schedule == Some(WIRE_SCHEDULE_DECODE_PRIORITY);
    begin.schedule = schedule.map(str::to_string);
    let (first, second) = if decode_priority {
        (
            (
                begin.layout_table[1].clone(),
                2u32..4u32,
                2u32,
                4u32,
                8usize,
            ),
            (
                begin.layout_table[0].clone(),
                0u32..2u32,
                0u32,
                4u32,
                16usize,
            ),
        )
    } else {
        (
            (
                begin.layout_table[0].clone(),
                0u32..2u32,
                0u32,
                4u32,
                16usize,
            ),
            (
                begin.layout_table[1].clone(),
                2u32..4u32,
                2u32,
                4u32,
                8usize,
            ),
        )
    };
    let mut planes = Vec::new();
    let mut sequence = 0u32;
    for (class, layers, start, end, len) in [first, second] {
        for layer in layers {
            for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
                let bytes = vec![7u8; len];
                planes.push((
                    LayerHeaderV1 {
                        byte_length: bytes.len() as u64,
                        layer,
                        logical_token_end: end,
                        logical_token_start: start,
                        role,
                        schema_version: LIVE_HANDOFF_SCHEMA_V1,
                        sequence,
                        sha256: sha256_hex(&bytes),
                        shape: [end - start, class.kv_heads, class.head_dim],
                        transfer_id: begin.transfer_id.clone(),
                        dtype: Some(class.dtype.clone()),
                        layout_class: Some(class.class.clone()),
                    },
                    bytes,
                ));
                sequence += 1;
            }
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
        frame_count: 8,
        payload_bytes: 96,
        payload_sha256: hex::encode(payload_hash.finalize()),
        prompt_token_ids: tokens.clone(),
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: begin.transfer_id.clone(),
        canary: None,
    };
    let seal = SealManifestV1 {
        artifact_sha256: artifact_sha256(&begin, &headers, &core).unwrap(),
        artifact_hmac_sha256: None,
        core,
    };
    (begin, planes, seal, limits)
}

/// Stream, publish, and reopen the fixture; returns the reopened layer
/// order. Per-plane exact-match verification runs inside the coordinator.
fn stream_publish_reopen(
    begin: &BeginManifestV1,
    planes: &[(LayerHeaderV1, Vec<u8>)],
    seal: &SealManifestV1,
    limits: &ValidationLimits,
) -> Vec<u32> {
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut coordinator =
        StreamingCoordinatorV1::create(&final_path, begin.clone(), limits.clone()).unwrap();
    let pool = LayerPermitPoolV1::experiment_v2();
    for (header, bytes) in planes {
        match header.role {
            TensorRoleV1::Key => {
                let permit = pool.acquire().unwrap();
                coordinator
                    .ingest_plane(header.clone(), bytes.clone(), Some(permit))
                    .unwrap();
            }
            TensorRoleV1::Value => {
                coordinator
                    .ingest_plane(header.clone(), bytes.clone(), None)
                    .unwrap();
            }
        }
    }
    coordinator.verify_and_prepare_seal(seal.clone()).unwrap();
    coordinator.publish().unwrap();
    let verified = VerifiedBundle::open_materialized(&final_path, limits).unwrap();
    assert_eq!(verified.begin(), begin);
    assert_eq!(verified.seal(), seal);
    verified
        .planes()
        .iter()
        .map(|plane| plane.header.layer)
        .collect()
}

#[test]
fn v2_decode_priority_streams_newest_cuts_first_and_reopens() {
    let (begin, planes, seal, limits) = v2_schedule_fixture(Some(WIRE_SCHEDULE_DECODE_PRIORITY));
    begin
        .validate(&limits)
        .expect("decode-priority begin is valid");
    // Exact-match verification derives the same scheduled walk: the first
    // frame must be the windowed class's layer-2 K plane; the declared
    // order's first plane (full class layer 0) fails closed at sequence 0.
    let verifier = IncrementalVerifierV1::new(begin.clone(), limits.clone()).unwrap();
    verifier
        .validate_next_header(&planes[0].0)
        .expect("windowed layer 2 K is expected first under decode-priority");
    let (begin_default, default_planes, _, _) = v2_schedule_fixture(None);
    let verifier_default = IncrementalVerifierV1::new(begin_default, limits.clone()).unwrap();
    verifier_default
        .validate_next_header(&default_planes[0].0)
        .expect("declared order accepts the full class first");
    verifier
        .validate_next_header(&default_planes[0].0)
        .expect_err("a declared-order plane mismatches the decode-priority walk");
    let reopened = stream_publish_reopen(&begin, &planes, &seal, &limits);
    assert_eq!(reopened, vec![2, 2, 3, 3, 0, 0, 1, 1]);
}

#[test]
fn v2_absent_and_layer_order_schedules_are_identical() {
    let (begin_absent, planes_absent, seal_absent, limits) = v2_schedule_fixture(None);
    let (begin_explicit, planes_explicit, seal_explicit, _) =
        v2_schedule_fixture(Some(WIRE_SCHEDULE_LAYER_ORDER));
    begin_absent.validate(&limits).unwrap();
    begin_explicit.validate(&limits).unwrap();
    // Absent schedule: canonical bytes carry no `schedule` key at all, so
    // pre-schedule begin bytes stay valid and hash-identical.
    let canonical = canonical_json(&begin_absent).unwrap();
    assert!(!canonical
        .windows(b"schedule".len())
        .any(|window| window == b"schedule"));
    // Declared order is the walk either way — same headers, same seal
    // core (the artifact hash legitimately differs: it authenticates the
    // begin, which carries the explicit schedule field), same reopen.
    assert_eq!(planes_absent, planes_explicit);
    assert_eq!(seal_absent.core, seal_explicit.core);
    let declared = vec![0, 0, 1, 1, 2, 2, 3, 3];
    assert_eq!(
        stream_publish_reopen(&begin_absent, &planes_absent, &seal_absent, &limits),
        declared
    );
    assert_eq!(
        stream_publish_reopen(&begin_explicit, &planes_explicit, &seal_explicit, &limits),
        declared
    );
}

#[test]
fn v2_begin_rejects_unknown_or_misplaced_schedule() {
    let (begin, _, _, limits) = v2_schedule_fixture(Some("k-first"));
    begin
        .validate(&limits)
        .expect_err("unknown schedule value fails closed");
    // A schedule without a v2 layout table is rejected too: there is no
    // walk to schedule on a v1 begin.
    let (v1_begin, _, _, v1_limits) = fixture();
    let mut v1_begin = v1_begin;
    v1_begin.schedule = Some(WIRE_SCHEDULE_LAYER_ORDER.into());
    v1_begin
        .validate(&v1_limits)
        .expect_err("schedule on a v1 begin fails closed");
}
