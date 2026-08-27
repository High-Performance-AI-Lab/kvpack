use super::*;
use kvpack_handoff::HandoffError;

// ── Protocol v2: descriptor-driven layout tables (docs/PROTOCOL_V2_DESIGN.md)
// Gemma-shaped fixture: class A "swa" (layers 0..4, hd 2, kvh 1, window 1
// token) + class B "full" (layers 4..6, hd 4, kvh 2, full range), cached =
// 3 tokens. Proves mixed head_dim, SWA token ranges, dtype tags, and the
// declared-order walk, with v1 semantics preserved when the table is empty.

fn v2_fixture() -> (
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
    let class_a = LayoutClassV2 {
        class: "gqa-windowed".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 0,
        head_dim: 2,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 4,
        window_tokens: 1,
    };
    let class_b = LayoutClassV2 {
        class: "gqa-full".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 4,
        head_dim: 4,
        kv_heads: 2,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 6,
        window_tokens: 0,
    };
    let begin = BeginManifestV1 {
        cached_token_count: 3,
        created_unix_ms: 100,
        deadline_unix_ms: 200,
        endpoints: EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-gemma4-f16-v1".into(),
            consumer_node: "mac-m3-ultra".into(),
            producer_engine_abi: "vllm-0.21.0-gb10-v1".into(),
            producer_node: "producer-node".into(),
            trust_domain: "lab-prefill".into(),
        },
        expected_layer_frames: 12,
        expected_payload_bytes: 224,
        geometry: GeometryV1 {
            head_dim: 0,
            max_context_tokens: 32,
            num_kv_heads: 0,
            num_layers: 6,
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
            weights: "q4_k_m".into(),
        },
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: "b".repeat(64),
        layout_table: vec![class_a, class_b],
        schedule: None,
        hmac_key_id: None,
    };
    let mut planes = Vec::new();
    let mut sequence = 0u32;
    let mut push = |sequence: &mut u32,
                    begin: &BeginManifestV1,
                    layer: u32,
                    class: &LayoutClassV2,
                    role: TensorRoleV1,
                    start: u32,
                    end: u32,
                    bytes: Vec<u8>| {
        planes.push((
            LayerHeaderV1 {
                byte_length: bytes.len() as u64,
                layer,
                logical_token_end: end,
                logical_token_start: start,
                role,
                schema_version: LIVE_HANDOFF_SCHEMA_V1,
                sequence: *sequence,
                sha256: sha256_hex(&bytes),
                shape: [end - start, class.kv_heads, class.head_dim],
                transfer_id: begin.transfer_id.clone(),
                dtype: Some(class.dtype.clone()),
                layout_class: Some(class.class.clone()),
            },
            bytes,
        ));
        *sequence += 1;
    };
    let (class_a, class_b) = (begin.layout_table[0].clone(), begin.layout_table[1].clone());
    for layer in 0..4 {
        for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
            push(
                &mut sequence,
                &begin,
                layer,
                &class_a,
                role,
                2,
                3,
                vec![7u8; 4],
            );
        }
    }
    for layer in 4..6 {
        for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
            push(
                &mut sequence,
                &begin,
                layer,
                &class_b,
                role,
                0,
                3,
                vec![9u8; 48],
            );
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
        frame_count: 12,
        payload_bytes: 224,
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

#[test]
fn v2_mixed_head_dim_and_windowed_ranges_stage_and_open() {
    let (begin, planes, seal, limits) = v2_fixture();
    begin.validate(&limits).expect("v2 begin is valid");
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut stager = BundleStager::create(&final_path, begin.clone(), limits.clone()).unwrap();
    for (header, bytes) in planes {
        stager.ingest(header, &bytes).unwrap();
    }
    stager.seal(seal.clone()).unwrap();
    let verified = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(verified.begin(), &begin);
    assert_eq!(verified.seal(), &seal);
    assert_eq!(verified.planes().len(), 12);
}

#[test]
fn v2_rejects_shape_range_dtype_and_class_drift() {
    let (begin, planes, _, _) = v2_fixture();

    // Shape drift on a class-A plane (hd 2 -> 4).
    let mut drifted = planes[0].0.clone();
    drifted.shape = [1, 1, 4];
    assert!(drifted.validate_for(&begin, 0).is_err());

    // Range drift: windowed plane must carry exactly the last token.
    let mut drifted = planes[0].0.clone();
    drifted.logical_token_start = 1;
    assert!(drifted.validate_for(&begin, 0).is_err());
    let mut drifted = planes[0].0.clone();
    drifted.logical_token_start = 0;
    assert!(drifted.validate_for(&begin, 0).is_err());

    // dtype tag drift; absent tag is fine only because the class is float16.
    let mut drifted = planes[0].0.clone();
    drifted.dtype = Some("float32".into());
    assert!(drifted.validate_for(&begin, 0).is_err());
    let mut ok = planes[0].0.clone();
    ok.dtype = None;
    ok.validate_for(&begin, 0).unwrap();

    // Wrong class tag; missing class tag.
    let mut drifted = planes[0].0.clone();
    drifted.layout_class = Some("gqa-full".into());
    assert!(drifted.validate_for(&begin, 0).is_err());
    let mut drifted = planes[0].0.clone();
    drifted.layout_class = None;
    assert!(drifted.validate_for(&begin, 0).is_err());

    // Out-of-order: class-B plane presented at sequence 0.
    assert!(planes[8].0.validate_for(&begin, 0).is_err());
}

#[test]
fn v2_begin_rejects_flat_geometry_and_abi_disagreement() {
    let (begin, _, _, limits) = v2_fixture();

    // Multi-class table requires zeroed flat kv_heads/head_dim.
    let mut drifted = begin.clone();
    drifted.geometry.num_kv_heads = 1;
    assert!(drifted.validate(&limits).is_err());

    // Single-class table must match the flat geometry exactly. The table
    // covers layers 0..4, so the declared layer count shrinks to 4 to keep
    // the coverage requirement (union == 0..num_layers) satisfied.
    let mut single = begin.clone();
    single.layout_table.truncate(1);
    single.expected_layer_frames = 8;
    single.expected_payload_bytes = 32;
    single.geometry.num_layers = 4;
    single.geometry.num_kv_heads = 1;
    single.geometry.head_dim = 2;
    single.validate(&limits).unwrap();
    let mut drifted = single.clone();
    drifted.geometry.head_dim = 4;
    assert!(drifted.validate(&limits).is_err());

    // v1 abi with a table present is rejected; v2 abi required.
    let mut drifted = begin.clone();
    drifted.portable_abi = PORTABLE_KV_ABI_V1.into();
    assert!(drifted.validate(&limits).is_err());

    // Declared frame/payload bounds must match the table walk.
    let mut drifted = begin.clone();
    drifted.expected_layer_frames = 11;
    assert!(drifted.validate(&limits).is_err());
    let mut drifted = begin.clone();
    drifted.expected_payload_bytes = 223;
    assert!(drifted.validate(&limits).is_err());

    // Overlapping classes are rejected.
    let mut drifted = begin.clone();
    drifted.layout_table[1].from = 3;
    assert!(drifted.validate(&limits).is_err());

    // v1 semantics preserved: empty table + v1 abi stays valid.
    let (v1_begin, _, _, v1_limits) = fixture();
    v1_begin.validate(&v1_limits).unwrap();
}

#[test]
fn v2_begin_rejects_unbounded_until_before_any_allocation() {
    // F1: `until: u32::MAX` must fail at validation without materializing
    // the claimed range (~16 GiB before this fix). The test passing at all
    // in a constrained CI process is the allocation proof.
    let (begin, _, _, limits) = v2_fixture();
    let mut drifted = begin.clone();
    drifted.layout_table[0].until = u32::MAX;
    let started = std::time::Instant::now();
    assert!(drifted.validate(&limits).is_err());
    assert!(started.elapsed() < std::time::Duration::from_secs(5));

    // `step: 0` is rejected on the same pre-materialization path.
    let mut drifted = begin.clone();
    drifted.layout_table[0].step = 0;
    assert!(drifted.validate(&limits).is_err());

    // An unvalidated class with step == 0 never materializes a range:
    // layers() is checked count-then-collect with an empty result for
    // invalid ranges. (The u32::MAX case is never handed to layers(): the
    // validator's until bound above fires first.)
    let mut hostile = begin.layout_table[0].clone();
    hostile.step = 0;
    assert!(hostile.layers().is_empty());
}

#[test]
fn v2_begin_enforces_per_class_frame_bound_not_average() {
    let (begin, _, _, limits) = v2_fixture();
    // Fixture per-plane bytes: class A 1*1*2*2 = 4, class B 3*2*4*2 = 48;
    // the cross-class average is 224/12 ~= 18.7.
    assert!(begin.validate(&limits).is_ok(), "control: 1024 cap passes");

    // A cap between the average and class B's per-plane size: the average
    // check passes, the per-class bound must reject at arm.
    let mut tight = limits.clone();
    tight.max_frame_bytes = 32;
    let error = begin.validate(&tight).expect_err("class B exceeds the cap");
    assert!(
        matches!(&error, HandoffError::Validation(message) if message.contains("per-frame")),
        "unexpected error: {error}"
    );

    // A cap above every class's per-plane size passes both directions.
    let mut loose = limits.clone();
    loose.max_frame_bytes = 48;
    begin.validate(&loose).unwrap();
}

#[test]
fn v2_begin_requires_full_layer_coverage() {
    let (begin, _, _, limits) = v2_fixture();

    // A holey table (layers 4..6 uncovered) must be rejected: it seals a
    // cache that is missing layers.
    let mut holey = begin.clone();
    holey.layout_table.truncate(1);
    holey.expected_layer_frames = 8;
    holey.expected_payload_bytes = 32;
    holey.geometry.num_kv_heads = 1;
    holey.geometry.head_dim = 2;
    let error = holey.validate(&limits).expect_err("holey table must fail");
    assert!(
        matches!(&error, HandoffError::Validation(message) if message.contains("cover every declared layer")),
        "unexpected error: {error}"
    );

    // A table covering beyond the declared count fails on the same rule.
    let mut over = begin.clone();
    over.layout_table[1].until = 7;
    assert!(over.validate(&limits).is_err());

    // Full exact coverage passes (the fixture itself).
    begin.validate(&limits).unwrap();
}

#[test]
fn v2_begin_rejects_out_of_range_or_duplicate_except() {
    let (begin, _, _, limits) = v2_fixture();

    // An except entry outside [from, until) is dead wiring; reject it.
    let mut drifted = begin.clone();
    drifted.layout_table[0].except = vec![9];
    assert!(drifted.validate(&limits).is_err());

    // Duplicates are rejected too.
    let mut drifted = begin.clone();
    drifted.layout_table[0].except = vec![1, 1];
    assert!(drifted.validate(&limits).is_err());

    // In-range, unique, coverage-preserving except entries pass.
    let mut ok = begin.clone();
    ok.layout_table[0].except = vec![1];
    ok.layout_table[1].from = 1;
    ok.layout_table[1].until = 2;
    // Class B now covers only layer 1 plus 4..6; split it so coverage
    // stays exact: A covers {0,2,3}, B {1}, plus a third class for 4..6.
    ok.layout_table[1].class = "gqa-full-mid".into();
    let mut tail = begin.layout_table[1].clone();
    tail.class = "gqa-full-tail".into();
    tail.from = 4;
    tail.until = 6;
    ok.layout_table.push(tail);
    let window_plane = 4; // 1 token * 1 kvh * 2 hd * 2 bytes
    let full_plane = 48; // 3 tokens * 2 kvh * 4 hd * 2 bytes
    ok.expected_layer_frames = 3 * 2 + 2 + 2 * 2;
    ok.expected_payload_bytes = (3 * 2 * window_plane + 3 * 2 * full_plane) as u64;
    ok.validate(&limits).unwrap();
}

#[test]
fn v2_begin_requires_exact_key_value_role_order() {
    let (begin, _, _, limits) = v2_fixture();

    let mut reversed = begin.clone();
    reversed.layout_table[0].roles = vec![TensorRoleV1::Value, TensorRoleV1::Key];
    assert!(reversed.validate(&limits).is_err());

    let mut key_only = begin.clone();
    key_only.layout_table[0].roles = vec![TensorRoleV1::Key];
    assert!(key_only.validate(&limits).is_err());

    let mut empty = begin.clone();
    empty.layout_table[0].roles = Vec::new();
    assert!(empty.validate(&limits).is_err());

    begin.validate(&limits).unwrap();
}
