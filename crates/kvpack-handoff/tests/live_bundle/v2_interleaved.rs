use super::*;

/// Gemma-shaped interleave: the declared walk is NOT ascending in layer
/// order (windowed layers 0,1,3 stream before full-attention layer 2). The
/// v1 `sequence/2` parity derivation used to break both the coordinator
/// cursor and the sealed-bundle reopen here.
fn v2_interleaved_fixture() -> (
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
        except: vec![2],
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
        from: 2,
        head_dim: 4,
        kv_heads: 2,
        roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
        step: 1,
        until: 3,
        window_tokens: 0,
    };
    let begin = BeginManifestV1 {
        cached_token_count: 3,
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
        expected_payload_bytes: 120,
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
        transfer_id: "b".repeat(64),
        layout_table: vec![class_a, class_b],
        schedule: None,
        hmac_key_id: None,
    };
    let mut planes = Vec::new();
    let mut sequence = 0u32;
    let mut push = |begin: &BeginManifestV1,
                    sequence: &mut u32,
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
    for layer in [0, 1, 3] {
        for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
            push(
                &begin,
                &mut sequence,
                layer,
                &class_a,
                role,
                2,
                3,
                vec![7u8; 4],
            );
        }
    }
    for role in [TensorRoleV1::Key, TensorRoleV1::Value] {
        push(
            &begin,
            &mut sequence,
            2,
            &class_b,
            role,
            0,
            3,
            vec![9u8; 48],
        );
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
        payload_bytes: 120,
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
fn v2_non_ascending_walk_streams_publishes_and_reopens() {
    let (begin, planes, seal, limits) = v2_interleaved_fixture();
    begin
        .validate(&limits)
        .expect("interleaved v2 begin is valid");
    let temp = tempfile::tempdir().unwrap();
    let final_path = temp.path().join("ready-bundle");
    let mut coordinator =
        StreamingCoordinatorV1::create(&final_path, begin.clone(), limits.clone()).unwrap();
    let pool = LayerPermitPoolV1::experiment_v2();
    for (header, bytes) in &planes {
        match header.role {
            TensorRoleV1::Key => {
                let permit = pool.acquire().unwrap();
                assert!(coordinator
                    .ingest_plane(header.clone(), bytes.clone(), Some(permit))
                    .unwrap()
                    .is_none());
            }
            TensorRoleV1::Value => {
                let ready = coordinator
                    .ingest_plane(header.clone(), bytes.clone(), None)
                    .unwrap()
                    .expect("V completes exactly one layer event");
                assert_eq!(ready.layer(), header.layer);
                drop(ready);
            }
        }
    }
    assert_eq!(coordinator.ready_layers(), 4);
    coordinator.verify_and_prepare_seal(seal.clone()).unwrap();
    coordinator.publish().unwrap();
    let verified = VerifiedBundle::open_materialized(&final_path, &limits).unwrap();
    assert_eq!(verified.begin(), &begin);
    assert_eq!(verified.seal(), &seal);
    let reopened_layers = verified
        .planes()
        .iter()
        .map(|plane| plane.header.layer)
        .collect::<Vec<_>>();
    assert_eq!(reopened_layers, vec![0, 0, 1, 1, 3, 3, 2, 2]);
}
