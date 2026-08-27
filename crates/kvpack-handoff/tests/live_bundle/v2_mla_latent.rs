use super::*;

/// mla-latent fixture: 4 layers, each covered by a single-role class
/// declaring one packed latent plane per layer (roles [key], kv_heads 1).
fn mla_fixture() -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
    let class_mla = LayoutClassV2 {
        class: "mla-latent".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 0,
        head_dim: 2,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key],
        step: 1,
        until: 4,
        window_tokens: 0,
    };
    mla_like_fixture(
        class_mla,
        GeometryV1 {
            head_dim: 2,
            max_context_tokens: 32,
            num_kv_heads: 1,
            num_layers: 4,
        },
    )
}

/// Mixed-table fixture: a gqa-full K/V class (layers 0,1) plus an
/// mla-latent single-role class (layers 2,3); flat geometry zeroed.
fn mixed_fixture() -> (
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
    let class_mla = LayoutClassV2 {
        class: "mla-latent".into(),
        dtype: "float16".into(),
        except: Vec::new(),
        from: 2,
        head_dim: 2,
        kv_heads: 1,
        roles: vec![TensorRoleV1::Key],
        step: 1,
        until: 4,
        window_tokens: 0,
    };
    mla_like_fixture_multi(
        vec![class_full, class_mla],
        GeometryV1 {
            head_dim: 0,
            max_context_tokens: 32,
            num_kv_heads: 0,
            num_layers: 4,
        },
        limits,
        tokens,
    )
}

fn mla_like_fixture(
    class: LayoutClassV2,
    geometry: GeometryV1,
) -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
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
    mla_like_fixture_multi(vec![class], geometry, limits, vec![10, 20, 30, 40, 50])
}

fn mla_like_fixture_multi(
    classes: Vec<LayoutClassV2>,
    geometry: GeometryV1,
    limits: ValidationLimits,
    tokens: Vec<u32>,
) -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
    let mut begin = BeginManifestV1 {
        cached_token_count: 4,
        created_unix_ms: 100,
        deadline_unix_ms: 200,
        endpoints: EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-dsv4-f16-v1".into(),
            consumer_node: "mac-m3-ultra".into(),
            producer_engine_abi: "llamacpp-gb10-v1".into(),
            producer_node: "producer-node".into(),
            trust_domain: "lab-prefill".into(),
        },
        expected_layer_frames: 0,
        expected_payload_bytes: 0,
        geometry,
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
        transfer_id: "d".repeat(64),
        layout_table: classes,
        schedule: None,
        hmac_key_id: None,
    };
    // Planes follow the declared walk: classes in order, layers ascending,
    // roles in each class's declared order (one frame per mla-latent layer).
    let mut planes = Vec::new();
    let mut sequence = 0u32;
    let mut payload_bytes = 0u64;
    for class_idx in 0..begin.layout_table.len() {
        let class = begin.layout_table[class_idx].clone();
        for layer in class.from..class.until {
            for role in &class.roles {
                let bytes = vec![9u8; 16];
                payload_bytes += bytes.len() as u64;
                planes.push((
                    LayerHeaderV1 {
                        byte_length: bytes.len() as u64,
                        layer,
                        logical_token_end: 4,
                        logical_token_start: 0,
                        role: *role,
                        schema_version: LIVE_HANDOFF_SCHEMA_V1,
                        sequence,
                        sha256: sha256_hex(&bytes),
                        shape: [4, class.kv_heads, class.head_dim],
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
    begin.expected_layer_frames = sequence;
    begin.expected_payload_bytes = payload_bytes;
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
        frame_count: begin.expected_layer_frames,
        payload_bytes,
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

/// One observed ready event, summarized before the event (and its permit)
/// is dropped — the capacity-2 pool deadlocks if ready events accumulate.
type ReadySummary = (u32, bool, u64, bool);

/// Stream the fixture through the coordinator, recording each ready event
/// as `(layer, has value plane, canonical bytes, has value files)` and
/// dropping it immediately; seals, publishes, and returns the bundle path.
fn stream_publish(
    begin: &BeginManifestV1,
    planes: &[(LayerHeaderV1, Vec<u8>)],
    seal: &SealManifestV1,
    limits: &ValidationLimits,
) -> (tempfile::TempDir, std::path::PathBuf, Vec<ReadySummary>) {
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut coordinator =
        StreamingCoordinatorV1::create(&final_path, begin.clone(), limits.clone()).unwrap();
    let pool = LayerPermitPoolV1::experiment_v2();
    let mut ready = Vec::new();
    for (header, bytes) in planes {
        let permit = match header.role {
            TensorRoleV1::Key => Some(pool.acquire().unwrap()),
            TensorRoleV1::Value => None,
        };
        if let Some(event) = coordinator
            .ingest_plane(header.clone(), bytes.clone(), permit)
            .unwrap()
        {
            ready.push((
                event.layer(),
                event.pair().value().is_some(),
                event.canonical_bytes(),
                event.files().value_header().is_some() && event.files().value_payload().is_some(),
            ));
            // The event drops here, releasing its layer permit back to the
            // capacity-2 pool before the next K-plane acquire.
        }
    }
    // (e) Seal succeeds with no pending pair left open.
    coordinator.verify_and_prepare_seal(seal.clone()).unwrap();
    coordinator.publish().unwrap();
    (temp, final_path, ready)
}

#[test]
fn v2_mla_latent_single_role_layers_stream_seal_and_reopen() {
    let (begin, planes, seal, limits) = mla_fixture();
    begin.validate(&limits).expect("mla-latent begin is valid");
    assert_eq!(begin.expected_layer_frames, 4);
    assert_eq!(begin.expected_payload_bytes, 64);
    let (_temp, final_path, ready) = stream_publish(&begin, &planes, &seal, &limits);
    // (a) Every layer completed immediately as a single-plane ready event:
    // no pair, no value files, the K-plane permit accounting unchanged.
    assert_eq!(
        ready,
        vec![
            (0, false, 16, false),
            (1, false, 16, false),
            (2, false, 16, false),
            (3, false, 16, false),
        ]
    );
    // Reopen re-verifies every plane against the one-frame-per-layer walk,
    // both streaming and materialized.
    let reopened = VerifiedBundle::open(&final_path, &limits).unwrap();
    assert_eq!(reopened.begin(), &begin);
    assert_eq!(reopened.seal(), &seal);
    let materialized = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(
        materialized
            .planes()
            .iter()
            .map(|plane| (plane.header.layer, plane.header.role))
            .collect::<Vec<_>>(),
        (0..4)
            .map(|layer| (layer, TensorRoleV1::Key))
            .collect::<Vec<_>>()
    );
}

#[test]
fn v2_mixed_gqa_full_and_mla_latent_stream_and_reopen() {
    let (begin, planes, seal, limits) = mixed_fixture();
    begin.validate(&limits).expect("mixed begin is valid");
    assert_eq!(begin.expected_layer_frames, 6);
    assert_eq!(begin.expected_payload_bytes, 96);
    let (_temp, final_path, ready) = stream_publish(&begin, &planes, &seal, &limits);
    // Two K/V pairs plus two single-role layers, in walk order.
    assert_eq!(
        ready
            .iter()
            .map(|&(layer, has_value, _, _)| (layer, has_value))
            .collect::<Vec<_>>(),
        vec![(0, true), (1, true), (2, false), (3, false)]
    );
    let materialized = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(
        materialized
            .planes()
            .iter()
            .map(|plane| (plane.header.layer, plane.header.role))
            .collect::<Vec<_>>(),
        vec![
            (0, TensorRoleV1::Key),
            (0, TensorRoleV1::Value),
            (1, TensorRoleV1::Key),
            (1, TensorRoleV1::Value),
            (2, TensorRoleV1::Key),
            (3, TensorRoleV1::Key),
        ]
    );
}

#[test]
fn v2_single_role_on_non_mla_latent_class_fails_validation() {
    let (mut begin, _, _, limits) = mla_fixture();
    begin.layout_table[0].class = "gqa-latent".into();
    begin
        .validate(&limits)
        .expect_err("roles [key] on a non-mla-latent class fails closed");
}

#[test]
fn v2_consecutive_mla_latent_key_planes_do_not_error() {
    let (begin, planes, seal, limits) = mla_fixture();
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut coordinator =
        StreamingCoordinatorV1::create(&final_path, begin.clone(), limits.clone()).unwrap();
    let pool = LayerPermitPoolV1::experiment_v2();
    // (d) Two consecutive mla K planes: the first completed its layer
    // immediately, so the second is not "another K before its V". The ready
    // events drop at once, releasing their permits before the next acquire.
    let first = coordinator
        .ingest_plane(
            planes[0].0.clone(),
            planes[0].1.clone(),
            Some(pool.acquire().unwrap()),
        )
        .unwrap();
    assert!(first.is_some());
    drop(first);
    let second = coordinator
        .ingest_plane(
            planes[1].0.clone(),
            planes[1].1.clone(),
            Some(pool.acquire().unwrap()),
        )
        .unwrap();
    assert!(second.is_some());
    drop(second);
    assert_eq!(coordinator.ready_layers(), 2);
    // Sealing with only single-role planes streamed leaves no pending pair.
    for (header, bytes) in &planes[2..] {
        let ready = coordinator
            .ingest_plane(header.clone(), bytes.clone(), Some(pool.acquire().unwrap()))
            .unwrap();
        assert!(ready.is_some());
    }
    coordinator.verify_and_prepare_seal(seal).unwrap();
    coordinator.publish().unwrap();
    VerifiedBundle::open(&final_path, &limits).unwrap();
    // A K/V-class K plane followed by another K still fails closed.
    let (mixed_begin, mixed_planes, _, mixed_limits) = mixed_fixture();
    let temp2 = tempfile::tempdir().unwrap();
    let mut mixed_coordinator = StreamingCoordinatorV1::create(
        temp2.path().join("mixed-bundle"),
        mixed_begin,
        mixed_limits,
    )
    .unwrap();
    mixed_coordinator
        .ingest_plane(
            mixed_planes[0].0.clone(),
            mixed_planes[0].1.clone(),
            Some(pool.acquire().unwrap()),
        )
        .unwrap();
    mixed_coordinator
        .ingest_plane(
            mixed_planes[2].0.clone(),
            mixed_planes[2].1.clone(),
            Some(pool.acquire().unwrap()),
        )
        .expect_err("a pair-class K before its V still fails closed");
}
