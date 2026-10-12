use super::*;

pub const PORTABLE_PREFILL_ABI_V1: &str = "canonical-kv-f16-le-v1";
pub const PORTABLE_PREFILL_LAYERS_V1: u32 = 24;
pub const PORTABLE_PREFILL_KV_HEADS_V1: u32 = 2;
pub const PORTABLE_PREFILL_HEAD_DIM_V1: u32 = 64;
pub const PORTABLE_PREFILL_MAX_CONTEXT_V1: u32 = 32_768;

/// One closed, fail-closed geometry tuple admitted by the portable prefill
/// gate. Anything outside `PORTABLE_PREFILL_GEOMETRIES_V1` hard-errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortablePrefillGeometryV1 {
    pub name: &'static str,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

pub const PORTABLE_PREFILL_GEOMETRY_QWEN25_05B_V1: PortablePrefillGeometryV1 =
    PortablePrefillGeometryV1 {
        name: "qwen2.5-0.5b",
        num_layers: PORTABLE_PREFILL_LAYERS_V1,
        num_kv_heads: PORTABLE_PREFILL_KV_HEADS_V1,
        head_dim: PORTABLE_PREFILL_HEAD_DIM_V1,
    };

pub const PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1: PortablePrefillGeometryV1 =
    PortablePrefillGeometryV1 {
        name: "qwen2.5-7b",
        num_layers: 28,
        num_kv_heads: 4,
        head_dim: 128,
    };

pub const PORTABLE_PREFILL_GEOMETRIES_V1: &[PortablePrefillGeometryV1] = &[
    PORTABLE_PREFILL_GEOMETRY_QWEN25_05B_V1,
    PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1,
];

/// Resolve the closed geometry tuple for one exact (layers, KV heads, head
/// dimension) triple. Unknown geometry is an error, never a guess.
pub fn portable_prefill_geometry_v1(
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<&'static PortablePrefillGeometryV1, StoreError> {
    PORTABLE_PREFILL_GEOMETRIES_V1
        .iter()
        .find(|geometry| {
            geometry.num_layers == num_layers
                && geometry.num_kv_heads == num_kv_heads
                && geometry.head_dim == head_dim
        })
        .ok_or(StoreError::Expectation(
            "portable prefill geometry is outside the qualified closed tuples",
        ))
}

/// Validated BEGIN fields needed to derive durable identities. This type is
/// independent of any handoff transport or frame grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePrefillDescriptorInputV1 {
    pub model_sha256: Id32,
    pub adapter_sha256: Id32,
    pub tokenizer_sha256: Id32,
    pub chat_template_sha256: Id32,
    pub context_policy_sha256: Id32,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub producer_engine_abi: String,
    pub consumer_engine_abi: String,
    pub portable_abi: String,
    pub compute_precision: String,
    pub kv_precision: String,
    pub weight_precision: String,
    pub cached_token_count: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_context_tokens: u32,
}

pub fn derive_portable_prefill_descriptor_v1(
    input: &PortablePrefillDescriptorInputV1,
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    portable_prefill_geometry_v1(input.num_layers, input.num_kv_heads, input.head_dim)?;
    if input.cached_token_count == 0
        || input.cached_token_count >= PORTABLE_PREFILL_MAX_CONTEXT_V1
        || input.max_context_tokens != PORTABLE_PREFILL_MAX_CONTEXT_V1
        || input.portable_abi != PORTABLE_PREFILL_ABI_V1
        || input.compute_precision != "float16"
        || input.kv_precision != "float16"
        || input.weight_precision != "q4_k_m"
    {
        return Err(StoreError::Expectation(
            "portable prefill descriptor is outside the qualified Qwen2.5 tuple",
        ));
    }
    for value in [
        &input.model_revision,
        &input.tokenizer_revision,
        &input.producer_engine_abi,
        &input.consumer_engine_abi,
    ] {
        if value.is_empty() || value.len() > 1024 || !value.is_ascii() {
            return Err(StoreError::Expectation(
                "portable prefill identity strings must be bounded nonempty ASCII",
            ));
        }
    }

    let semantic_model = SemanticModelId {
        weights_config: domain_id(
            b"kvpack/spark-prefill/weights-config/v1\0",
            &[
                &input.model_sha256,
                input.model_revision.as_bytes(),
                input.weight_precision.as_bytes(),
            ],
        ),
        adapters: domain_id(
            b"kvpack/spark-prefill/adapters/v1\0",
            &[&input.adapter_sha256],
        ),
        tokenizer_template: domain_id(
            b"kvpack/spark-prefill/tokenizer-template/v1\0",
            &[
                &input.tokenizer_sha256,
                &input.chat_template_sha256,
                input.tokenizer_revision.as_bytes(),
            ],
        ),
        position_semantics: domain_id(
            b"kvpack/spark-prefill/position-semantics/v1\0",
            &[
                &input.context_policy_sha256,
                &input.max_context_tokens.to_le_bytes(),
            ],
        ),
        qualified_math: domain_id(
            b"kvpack/spark-prefill/qualified-math/v1\0",
            &[
                input.producer_engine_abi.as_bytes(),
                input.consumer_engine_abi.as_bytes(),
                input.compute_precision.as_bytes(),
                input.kv_precision.as_bytes(),
            ],
        ),
    };
    let geometry = [
        input.num_layers.to_le_bytes(),
        input.num_kv_heads.to_le_bytes(),
        input.head_dim.to_le_bytes(),
        input.max_context_tokens.to_le_bytes(),
    ];
    let engine_cache_abi = domain_id(
        b"kvpack/spark-prefill/engine-cache-abi/v1\0",
        &[
            input.portable_abi.as_bytes(),
            input.consumer_engine_abi.as_bytes(),
            &geometry[0],
            &geometry[1],
            &geometry[2],
            &geometry[3],
        ],
    );
    let topology = domain_id(
        b"kvpack/spark-prefill/topology/v1\0",
        &[&geometry[0], &geometry[1], &geometry[2], &geometry[3]],
    );
    let shard_map = domain_id(
        b"kvpack/spark-prefill/shard-map/v1\0",
        &[b"single-rank-layer-ascending-k-then-v"],
    );
    let elements_per_token = u64::from(input.num_kv_heads)
        .checked_mul(u64::from(input.head_dim))
        .ok_or(StoreError::State("prefill elements-per-token overflow"))?;
    let mut family_states = Vec::with_capacity((input.num_layers * 2) as usize);
    let mut states = Vec::with_capacity((input.num_layers * 2) as usize);
    for layer in 0..input.num_layers {
        for state_name in ["attn.k", "attn.v"] {
            let key = StateKey::new(layer, state_name);
            family_states.push(FamilyState {
                key: key.clone(),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::F16,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token,
                dimensions: vec![
                    StaticDimension::Token,
                    StaticDimension::Fixed(u64::from(input.num_kv_heads)),
                    StaticDimension::Fixed(u64::from(input.head_dim)),
                ],
                dependencies: Vec::new(),
            });
            states.push(ExportStateDeclaration {
                key,
                strides: vec![elements_per_token, u64::from(input.head_dim), 1],
                atomic_group: layer + 1,
            });
        }
    }
    let family = RepresentationFamilyId {
        engine_cache_abi,
        mode: RepresentationMode::Portable,
        page_size_tokens: PREFIX_BLOCK_TOKENS as u32,
        topology,
        shard_map,
        states: family_states,
    };
    validate_family(&family)?;
    let bytes_per_state = u64::from(input.cached_token_count)
        .checked_mul(elements_per_token)
        .and_then(|value| value.checked_mul(2))
        .ok_or(StoreError::State("prefill state byte bound overflow"))?;
    let restored_bytes = bytes_per_state
        .checked_mul(u64::from(input.num_layers) * 2)
        .ok_or(StoreError::State("prefill restored-byte bound overflow"))?;
    Ok(PortablePrefillDescriptorV1 {
        semantic_model,
        family,
        states,
        bytes_per_state,
        restored_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PortablePrefillDescriptorInputV1 {
        PortablePrefillDescriptorInputV1 {
            model_sha256: [1; 32],
            adapter_sha256: [2; 32],
            tokenizer_sha256: [3; 32],
            chat_template_sha256: [4; 32],
            context_policy_sha256: [5; 32],
            model_revision: "model@revision".into(),
            tokenizer_revision: "tokenizer@revision".into(),
            producer_engine_abi: "vllm-0.21".into(),
            consumer_engine_abi: "ferrite-v1".into(),
            portable_abi: PORTABLE_PREFILL_ABI_V1.into(),
            compute_precision: "float16".into(),
            kv_precision: "float16".into(),
            weight_precision: "q4_k_m".into(),
            cached_token_count: 16_383,
            num_layers: PORTABLE_PREFILL_LAYERS_V1,
            num_kv_heads: PORTABLE_PREFILL_KV_HEADS_V1,
            head_dim: PORTABLE_PREFILL_HEAD_DIM_V1,
            max_context_tokens: PORTABLE_PREFILL_MAX_CONTEXT_V1,
        }
    }

    #[test]
    fn derives_the_closed_48_state_family_and_bounds() {
        let descriptor = derive_portable_prefill_descriptor_v1(&input()).unwrap();
        assert_eq!(descriptor.states.len(), 48);
        assert_eq!(descriptor.states[0].key, StateKey::new(0, "attn.k"));
        assert_eq!(descriptor.states[47].key, StateKey::new(23, "attn.v"));
        assert_eq!(descriptor.states[0].strides, [128, 64, 1]);
        assert_eq!(descriptor.states[0].atomic_group, 1);
        assert_eq!(descriptor.states[1].atomic_group, 1);
        assert_eq!(descriptor.bytes_per_state, 16_383 * 256);
        assert_eq!(descriptor.restored_bytes, 16_383 * 256 * 48);
    }

    #[test]
    fn rejects_geometry_or_precision_drift() {
        let mut changed = input();
        changed.num_layers = 23;
        assert!(derive_portable_prefill_descriptor_v1(&changed).is_err());
        changed = input();
        changed.kv_precision = "float32".into();
        assert!(derive_portable_prefill_descriptor_v1(&changed).is_err());
    }

    #[test]
    fn derives_the_closed_7b_56_state_family() {
        let mut changed = input();
        changed.num_layers = PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1.num_layers;
        changed.num_kv_heads = PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1.num_kv_heads;
        changed.head_dim = PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1.head_dim;
        let descriptor = derive_portable_prefill_descriptor_v1(&changed).unwrap();
        assert_eq!(descriptor.states.len(), 56);
        assert_eq!(descriptor.states[0].key, StateKey::new(0, "attn.k"));
        assert_eq!(descriptor.states[55].key, StateKey::new(27, "attn.v"));
        assert_eq!(descriptor.states[0].strides, [512, 128, 1]);
        assert_eq!(descriptor.bytes_per_state, 16_383 * 512 * 2,);
        assert_eq!(descriptor.restored_bytes, 16_383 * 512 * 2 * 56);
    }

    #[test]
    fn rejects_unregistered_geometry() {
        let mut changed = input();
        changed.num_layers = 28;
        changed.num_kv_heads = 8;
        changed.head_dim = 128;
        assert!(derive_portable_prefill_descriptor_v1(&changed).is_err());
        changed = input();
        changed.num_layers = 36;
        assert!(derive_portable_prefill_descriptor_v1(&changed).is_err());
    }
}
