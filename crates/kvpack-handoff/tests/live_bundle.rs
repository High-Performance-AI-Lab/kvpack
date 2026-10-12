use sha2::{Digest, Sha256};

use kvpack_handoff::{
    artifact_hmac_sha256, artifact_sha256, canonical_json, descriptor_chain_sha256, sha256_hex,
    token_ids_sha256, BeginManifestV1, BundleStager, EndpointIdentityV1, ExactIdentityV1,
    GeometryV1, HandoffStrategyV1, IncrementalVerifierV1, LayerHeaderV1, LayerPermitPoolV1,
    LayoutClassV2, MacKey, PrecisionV1, SealCoreV1, SealManifestV1, StreamingCoordinatorV1,
    TensorRoleV1, ValidationLimits, VerifiedBundle, LIVE_HANDOFF_PROTOCOL_V1,
    LIVE_HANDOFF_SCHEMA_V1, PORTABLE_KV_ABI_V1, PORTABLE_KV_ABI_V2, WIRE_SCHEDULE_DECODE_PRIORITY,
    WIRE_SCHEDULE_LAYER_ORDER,
};

#[path = "live_bundle/failure_modes.rs"]
mod failure_modes;
#[path = "live_bundle/v1.rs"]
mod v1;
#[path = "live_bundle/v2.rs"]
mod v2;
#[path = "live_bundle/v2_interleaved.rs"]
mod v2_interleaved;
#[path = "live_bundle/v2_mla_latent.rs"]
mod v2_mla_latent;
#[path = "live_bundle/v2_prerope.rs"]
mod v2_prerope;
#[path = "live_bundle/v2_schedule.rs"]
mod v2_schedule;

fn fixture() -> (
    BeginManifestV1,
    Vec<(LayerHeaderV1, Vec<u8>)>,
    SealManifestV1,
    ValidationLimits,
) {
    let tokens = vec![10, 20, 30];
    let limits = ValidationLimits {
        max_cached_tokens: 32,
        max_clock_skew_ms: 20,
        max_context_tokens: 64,
        max_frame_bytes: 1024,
        max_layers: 4,
        max_session_ms: 1000,
        max_total_bytes: 4096,
        now_unix_ms: 110,
    };
    let begin = BeginManifestV1 {
        cached_token_count: 2,
        created_unix_ms: 100,
        deadline_unix_ms: 200,
        endpoints: EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-qwen25-f16-v1".into(),
            consumer_node: "mac-m3-ultra".into(),
            producer_engine_abi: "vllm-0.21.0-gb10-v1".into(),
            producer_node: "producer-node".into(),
            trust_domain: "lab-prefill".into(),
        },
        expected_layer_frames: 2,
        expected_payload_bytes: 16,
        geometry: GeometryV1 {
            head_dim: 2,
            max_context_tokens: 32,
            num_kv_heads: 1,
            num_layers: 1,
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
        portable_abi: PORTABLE_KV_ABI_V1.into(),
        precision: PrecisionV1 {
            compute: "float16".into(),
            kv: "float16".into(),
            weights: "q4_k_m".into(),
        },
        protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
        schema_version: LIVE_HANDOFF_SCHEMA_V1,
        strategy: HandoffStrategyV1::ConsumerLastPromptToken,
        token_ids_sha256: token_ids_sha256(&tokens),
        transfer_id: "a".repeat(64),
        layout_table: Vec::new(),
        schedule: None,
        hmac_key_id: None,
    };
    let payloads = [vec![1u8; 8], vec![2u8; 8]];
    let planes = payloads
        .into_iter()
        .enumerate()
        .map(|(sequence, bytes)| {
            let role = if sequence == 0 {
                TensorRoleV1::Key
            } else {
                TensorRoleV1::Value
            };
            (
                LayerHeaderV1 {
                    byte_length: bytes.len() as u64,
                    layer: 0,
                    logical_token_end: 2,
                    logical_token_start: 0,
                    role,
                    schema_version: LIVE_HANDOFF_SCHEMA_V1,
                    sequence: sequence as u32,
                    sha256: sha256_hex(&bytes),
                    shape: [2, 1, 2],
                    transfer_id: begin.transfer_id.clone(),
                    dtype: None,
                    layout_class: None,
                },
                bytes,
            )
        })
        .collect::<Vec<_>>();
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
        frame_count: 2,
        payload_bytes: 16,
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
