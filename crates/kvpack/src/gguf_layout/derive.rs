use super::*;

/// RoPE configuration parsed from GGUF metadata, ready to attach to a
/// derived class. Crate-internal plumbing shared by the uniform layout
/// derivation and the MLA derivation (crate::mla).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RopeFieldsV2 {
    pub freq_base_bits: u64,
    pub dimension_count: u32,
    pub scaling: String,
    pub convention: RopeConvention,
}

/// Architectures whose GGUF metadata derives a uniform full-attention
/// single-class layout. Anything else — hybrid, MLA, SSM, conv state —
/// hard-errors so the exception is visible at arm time.
const DERIVABLE_ARCHITECTURES: &[&str] = &["llama", "qwen2", "gpt-oss"];

/// Derive a v2 layout from a GGUF file on disk. Thin wrapper over
/// `derive_layout_from_metadata`.
pub fn derive_layout_from_gguf(path: &Path) -> Result<OwnedLayoutV2, StoreError> {
    let metadata = read_gguf_metadata(path)?;
    derive_layout_from_metadata(&metadata)
}

/// Derive a v2 layout from parsed GGUF metadata. Pure function: the harness
/// may feed it from any metadata source. Fail-closed — underivable geometry
/// is an error, never a guess.
pub fn derive_layout_from_metadata(metadata: &GgufMetadata) -> Result<OwnedLayoutV2, StoreError> {
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .ok_or(StoreError::Expectation(
            "gguf layout underivable: general.architecture is missing",
        ))?;
    // Hybrid sliding-window geometry (Gemma-style `_swa` keys, sliding_window,
    // or a per-layer head_count_kv array that is not uniform) cannot be
    // derived: GGUF carries no per-class KV-head counts, and guessing them is
    // exactly what the fail-closed posture forbids. A sidecar descriptor is
    // required. This check runs before the architecture allowlist so a hybrid
    // model fails with the actionable sidecar error, not a bare refusal.
    // Per-class rope variants (`{arch}.rope.*_swa`, e.g. gemma4's
    // freq_base_swa/dimension_count_swa) are hybrid geometry too: one rope
    // config cannot describe both classes, so they fail toward the sidecar.
    let attention_prefix = format!("{architecture}.attention.");
    let rope_prefix = format!("{architecture}.rope.");
    let has_swa_keys = metadata.keys().any(|name| {
        (name.starts_with(&attention_prefix)
            && (name.ends_with("_swa")
                || name.ends_with("sliding_window")
                || name.ends_with("sliding_window_pattern")))
            || (name.starts_with(&rope_prefix) && name.ends_with("_swa"))
    });
    if has_swa_keys {
        return Err(StoreError::Expectation(
            "gguf layout underivable: hybrid sliding-window geometry requires a sidecar descriptor",
        ));
    }
    if !DERIVABLE_ARCHITECTURES.contains(&architecture) {
        return Err(StoreError::Expectation(
            "gguf layout underivable: architecture is outside the derivation allowlist",
        ));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");

    let num_layers = required_u64(metadata, &key("block_count"))?;
    let head_count = required_u64(metadata, &key("attention.head_count"))?;
    if head_count == 0 {
        return Err(StoreError::Expectation(
            "gguf layout underivable: attention.head_count is zero",
        ));
    }
    let head_count_kv = match metadata.get(&key("attention.head_count_kv")) {
        Some(GgufValue::Array(counts)) => {
            let mut distinct = BTreeSet::new();
            for count in counts {
                distinct.insert(count.as_u64().ok_or(StoreError::Expectation(
                    "gguf layout underivable: attention.head_count_kv is not an integer array",
                ))?);
            }
            if distinct.len() != 1 {
                return Err(StoreError::Expectation(
                    "gguf layout underivable: per-layer attention.head_count_kv requires a sidecar descriptor",
                ));
            }
            *distinct.iter().next().unwrap_or(&0)
        }
        Some(value) => value.as_u64().ok_or(StoreError::Expectation(
            "gguf layout underivable: attention.head_count_kv is not an integer",
        ))?,
        None => {
            return Err(StoreError::Expectation(
                "gguf layout underivable: attention.head_count_kv is missing",
            ));
        }
    };
    if head_count_kv == 0 {
        return Err(StoreError::Expectation(
            "gguf layout underivable: attention.head_count_kv is zero",
        ));
    }

    let key_length = optional_u64(metadata, &key("attention.key_length"))?;
    let value_length = optional_u64(metadata, &key("attention.value_length"))?;
    let head_dim = match key_length {
        Some(length) => length,
        None => {
            let embedding_length = required_u64(metadata, &key("embedding_length"))?;
            if embedding_length % head_count != 0 {
                return Err(StoreError::Expectation(
                    "gguf layout underivable: embedding_length is not divisible by attention.head_count",
                ));
            }
            embedding_length / head_count
        }
    };
    if let Some(value_length) = value_length {
        if value_length != head_dim {
            return Err(StoreError::Expectation(
                "gguf layout underivable: key_length and value_length differ",
            ));
        }
    }
    if head_dim == 0
        || num_layers == 0
        || num_layers > MAX_LAYOUT_LAYERS
        || head_count_kv > u64::from(u32::MAX)
        || head_dim > u64::from(u32::MAX)
    {
        return Err(StoreError::Expectation(
            "gguf layout underivable: geometry is outside the v2 bounds",
        ));
    }

    let name = metadata
        .get("general.name")
        .and_then(GgufValue::as_str)
        .unwrap_or(architecture)
        .to_string();
    let rope = rope_fields_from_gguf(metadata, architecture, head_dim, head_dim)?;
    Ok(OwnedLayoutV2 {
        name,
        num_layers: num_layers as u32,
        classes: vec![OwnedLayoutClassV2 {
            class: "gqa-full".to_string(),
            from: 0,
            until: num_layers as u32,
            step: 1,
            except: Vec::new(),
            kv_heads: head_count_kv as u32,
            head_dim: head_dim as u32,
            window_tokens: 0,
            rope_freq_base_bits: rope.freq_base_bits,
            rope_dimension_count: rope.dimension_count,
            rope_scaling: rope.scaling,
            rope_convention: rope.convention,
        }],
    })
}

/// Parse the RoPE configuration of one uniform-geometry class from GGUF
/// metadata (docs/KV_ALGEBRA_2026-08-09.md, item 1): `{arch}.rope.freq_base`
/// (required float — a known arch without it fails closed, never defaults),
/// `{arch}.rope.dimension_count` (optional; `default_dimension_count` when
/// absent, matching the GGUF full-rotary convention), and
/// `{arch}.rope.scaling.type` + `.factor` canonicalized to the closed label
/// set. The pairing convention comes from the pinned per-architecture table
/// (`pinned_rope_convention`) because GGUF carries no convention key.
pub(crate) fn rope_fields_from_gguf(
    metadata: &GgufMetadata,
    architecture: &str,
    head_dim: u64,
    default_dimension_count: u64,
) -> Result<RopeFieldsV2, StoreError> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let freq_base = match metadata.get(&key("rope.freq_base")) {
        Some(GgufValue::Float32(value)) => f64::from(*value),
        Some(GgufValue::Float64(value)) => *value,
        Some(_) => {
            return Err(StoreError::Expectation(
                "gguf rope config underivable: rope.freq_base is not a float",
            ))
        }
        None => {
            return Err(StoreError::Expectation(
                "gguf rope config underivable: rope.freq_base is missing",
            ))
        }
    };
    if !freq_base.is_finite() || freq_base <= 1.0 {
        return Err(StoreError::Expectation(
            "gguf rope config underivable: rope.freq_base is outside the supported bounds",
        ));
    }
    let dimension_count = match metadata.get(&key("rope.dimension_count")) {
        None => default_dimension_count,
        Some(value) => value.as_u64().ok_or(StoreError::Expectation(
            "gguf rope config underivable: rope.dimension_count is not an integer",
        ))?,
    };
    if dimension_count < 2
        || dimension_count % 2 != 0
        || dimension_count > head_dim
        || dimension_count > u64::from(u32::MAX)
    {
        return Err(StoreError::Expectation(
            "gguf rope config underivable: rope.dimension_count is outside the v2 bounds",
        ));
    }
    let scaling = match metadata.get(&key("rope.scaling.type")) {
        None => {
            if metadata.contains_key(&key("rope.scaling.factor")) {
                return Err(StoreError::Expectation(
                    "gguf rope config underivable: rope.scaling.factor requires a scaling type",
                ));
            }
            "none".to_string()
        }
        Some(GgufValue::String(kind)) => {
            let factor = match metadata.get(&key("rope.scaling.factor")) {
                Some(GgufValue::Float32(value)) => f64::from(*value),
                Some(GgufValue::Float64(value)) => *value,
                Some(_) => {
                    return Err(StoreError::Expectation(
                        "gguf rope config underivable: rope.scaling.factor is not a float",
                    ))
                }
                None => {
                    return Err(StoreError::Expectation(
                        "gguf rope config underivable: rope.scaling.factor is missing",
                    ))
                }
            };
            rope_scaling_label(kind, factor).ok_or(StoreError::Expectation(
                "gguf rope config underivable: rope.scaling is outside the closed canonical set",
            ))?
        }
        Some(_) => {
            return Err(StoreError::Expectation(
                "gguf rope config underivable: rope.scaling.type is not a string",
            ))
        }
    };
    let convention = pinned_rope_convention(architecture).ok_or(StoreError::Expectation(
        "gguf rope config underivable: architecture has no pinned rope convention",
    ))?;
    Ok(RopeFieldsV2 {
        freq_base_bits: freq_base.to_bits(),
        dimension_count: dimension_count as u32,
        scaling,
        convention,
    })
}

fn required_u64(metadata: &GgufMetadata, key: &str) -> Result<u64, StoreError> {
    optional_u64(metadata, key)?.ok_or(StoreError::Expectation(
        "gguf layout underivable: a required geometry key is missing or not an integer",
    ))
}

fn optional_u64(metadata: &GgufMetadata, key: &str) -> Result<Option<u64>, StoreError> {
    match metadata.get(key) {
        None => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or(StoreError::Expectation(
            "gguf layout underivable: a geometry key is not an integer",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen25_7b_metadata() -> GgufMetadata {
        let mut metadata = GgufMetadata::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        );
        metadata.insert("qwen2.block_count".to_string(), GgufValue::Uint32(28));
        metadata.insert(
            "qwen2.attention.head_count".to_string(),
            GgufValue::Uint32(28),
        );
        metadata.insert(
            "qwen2.attention.head_count_kv".to_string(),
            GgufValue::Uint32(4),
        );
        metadata.insert(
            "qwen2.embedding_length".to_string(),
            GgufValue::Uint32(3_584),
        );
        metadata.insert(
            "qwen2.context_length".to_string(),
            GgufValue::Uint32(32_768),
        );
        metadata.insert(
            "qwen2.rope.freq_base".to_string(),
            GgufValue::Float32(1_000_000.0),
        );
        metadata
    }

    #[test]
    fn derives_qwen25_7b_from_the_real_gguf() {
        let Ok(model) = std::env::var("KVPACK_QWEN25_MODEL") else {
            return;
        };
        let path = Path::new(&model);
        if !path.exists() {
            return;
        }
        let derived = derive_layout_from_gguf(path).unwrap();
        assert_layout_matches_registry(&derived, "qwen2.5-7b");
    }

    #[test]
    fn refuses_gemma4_31b_real_gguf_without_a_sidecar() {
        let Ok(model) = std::env::var("KVPACK_GEMMA4_MODEL") else {
            return;
        };
        let path = Path::new(&model);
        if !path.exists() {
            return;
        }
        let error = derive_layout_from_gguf(path).unwrap_err();
        assert!(error.to_string().contains("sidecar"));
    }

    #[test]
    fn derives_qwen25_7b_from_synthetic_metadata() {
        let derived = derive_layout_from_metadata(&qwen25_7b_metadata()).unwrap();
        assert_layout_matches_registry(&derived, "qwen2.5-7b");
        // The rope config rides the derived class: full rotary at the GGUF
        // freq_base, no scaling, the pinned NEOX convention.
        let class = &derived.classes[0];
        assert_eq!(class.rope_freq_base_bits, 1_000_000.0f64.to_bits());
        assert_eq!(class.rope_dimension_count, 128);
        assert_eq!(class.rope_scaling, "none");
        assert_eq!(class.rope_convention, RopeConvention::Neox);
    }

    #[test]
    fn freq_base_is_required_and_bounded() {
        // Missing rope.freq_base fails closed for a known arch — never a
        // default base.
        let mut metadata = qwen25_7b_metadata();
        metadata.remove("qwen2.rope.freq_base");
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf rope config underivable: rope.freq_base is missing"
        );
        // Wrong type, non-finite, and out-of-bounds bases are all refused.
        for bad in [
            GgufValue::String("1000000".to_string()),
            GgufValue::Uint32(1_000_000),
            GgufValue::Float32(f32::NAN),
            GgufValue::Float32(1.0),
            GgufValue::Float32(0.5),
        ] {
            let mut metadata = qwen25_7b_metadata();
            metadata.insert("qwen2.rope.freq_base".to_string(), bad);
            assert!(derive_layout_from_metadata(&metadata).is_err());
        }
    }

    #[test]
    fn rope_dimension_count_defaults_and_validates() {
        // Absent means full rotary (the GGUF convention); an explicit
        // partial-rotary width is honored.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.dimension_count".to_string(),
            GgufValue::Uint32(64),
        );
        let derived = derive_layout_from_metadata(&metadata).unwrap();
        assert_eq!(derived.classes[0].rope_dimension_count, 64);
        // Odd, zero, over-wide, or non-integer widths fail closed.
        for bad in [
            GgufValue::Uint32(0),
            GgufValue::Uint32(3),
            GgufValue::Uint32(256),
            GgufValue::Float32(64.0),
        ] {
            let mut metadata = qwen25_7b_metadata();
            metadata.insert("qwen2.rope.dimension_count".to_string(), bad);
            assert!(derive_layout_from_metadata(&metadata).is_err());
        }
    }

    #[test]
    fn rope_scaling_canonicalizes_and_fails_closed() {
        // yarn factor 32 canonicalizes to the closed label form.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.scaling.type".to_string(),
            GgufValue::String("yarn".to_string()),
        );
        metadata.insert(
            "qwen2.rope.scaling.factor".to_string(),
            GgufValue::Float32(32.0),
        );
        let derived = derive_layout_from_metadata(&metadata).unwrap();
        assert_eq!(derived.classes[0].rope_scaling, "yarn:32");
        // A scaling factor without a type is a malformed-metadata smell.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.scaling.factor".to_string(),
            GgufValue::Float32(32.0),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf rope config underivable: rope.scaling.factor requires a scaling type"
        );
        // A type outside the closed set, or a missing factor, fails closed.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.scaling.type".to_string(),
            GgufValue::String("dynamic".to_string()),
        );
        metadata.insert(
            "qwen2.rope.scaling.factor".to_string(),
            GgufValue::Float32(32.0),
        );
        assert!(derive_layout_from_metadata(&metadata).is_err());
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.scaling.type".to_string(),
            GgufValue::String("yarn".to_string()),
        );
        assert!(derive_layout_from_metadata(&metadata).is_err());
    }

    #[test]
    fn rope_swa_keys_fail_closed_toward_a_sidecar() {
        // Per-class rope variants (gemma4-style) are hybrid geometry even
        // when the attention keys look uniform.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.rope.freq_base_swa".to_string(),
            GgufValue::Float32(10_000.0),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: hybrid sliding-window geometry requires a sidecar descriptor"
        );
    }

    #[test]
    fn ggufs_differing_only_in_freq_base_derive_different_identities() {
        let input = crate::prefill::PortablePrefillDescriptorInputV2 {
            model_sha256: [1; 32],
            adapter_sha256: [2; 32],
            tokenizer_sha256: [3; 32],
            chat_template_sha256: [4; 32],
            context_policy_sha256: [5; 32],
            model_revision: "model@revision".into(),
            tokenizer_revision: "tokenizer@revision".into(),
            producer_engine_abi: "vllm-0.21".into(),
            consumer_engine_abi: "ferrite-v1".into(),
            portable_abi: "canonical-kv-v2".into(),
            compute_precision: "float16".into(),
            kv_precision: "float16".into(),
            weight_precision: "q4_k_m".into(),
            cached_token_count: 1_024,
            max_context_tokens: 32_768,
            layout_name: "derived:qwen2-fixture".into(),
            transform: None,
            prerope_kernel_pin: None,
        };
        let base = derive_layout_from_metadata(&qwen25_7b_metadata()).unwrap();
        let mut changed_metadata = qwen25_7b_metadata();
        changed_metadata.insert(
            "qwen2.rope.freq_base".to_string(),
            GgufValue::Float32(500_000.0),
        );
        let changed = derive_layout_from_metadata(&changed_metadata).unwrap();
        // Identical numeric geometry; only the rope base differs.
        assert_eq!(base.classes[0].kv_heads, changed.classes[0].kv_heads);
        assert_ne!(
            base.classes[0].rope_freq_base_bits,
            changed.classes[0].rope_freq_base_bits
        );
        // The changed rope config no longer matches the registry oracle.
        assert!(crate::prefill::portable_prefill_layout_name_v2(&changed).is_none());
        let base_descriptor =
            crate::prefill::derive_portable_prefill_descriptor_v2_from_layout(&input, &base)
                .unwrap();
        let changed_descriptor =
            crate::prefill::derive_portable_prefill_descriptor_v2_from_layout(&input, &changed)
                .unwrap();
        assert_ne!(
            base_descriptor.family.engine_cache_abi,
            changed_descriptor.family.engine_cache_abi
        );
        assert_eq!(
            base_descriptor.semantic_model,
            changed_descriptor.semantic_model
        );
    }

    #[test]
    fn refuses_gpt_oss_120b_metadata_without_a_sidecar() {
        // The real gpt-oss-120b GGUF metadata key set: block_count 36,
        // scalar head_count_kv 8, key/value_length 64, and
        // `gpt-oss.attention.sliding_window` 128 — the model alternates
        // sliding-window(128) and full-attention layers (HF config
        // `layer_types`). The windowed class's existence makes the geometry
        // underivable from GGUF alone: derivation must fail closed toward a
        // sidecar, and the "gpt-oss-120b" REGISTRY entry is what describes
        // the real geometry (see prefill::v2).
        //
        // DEFERRED: a real-GGUF fixture test is not possible — no gpt-oss
        // GGUF file is available on disk in this environment.
        let mut metadata = GgufMetadata::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("gpt-oss".to_string()),
        );
        metadata.insert("gpt-oss.block_count".to_string(), GgufValue::Uint32(36));
        metadata.insert(
            "gpt-oss.attention.head_count".to_string(),
            GgufValue::Uint32(64),
        );
        metadata.insert(
            "gpt-oss.attention.head_count_kv".to_string(),
            GgufValue::Uint32(8),
        );
        metadata.insert(
            "gpt-oss.attention.key_length".to_string(),
            GgufValue::Uint32(64),
        );
        metadata.insert(
            "gpt-oss.attention.value_length".to_string(),
            GgufValue::Uint32(64),
        );
        metadata.insert(
            "gpt-oss.attention.sliding_window".to_string(),
            GgufValue::Uint32(128),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: hybrid sliding-window geometry requires a sidecar descriptor"
        );
    }

    #[test]
    fn refuses_gemma4_style_swa_metadata_without_a_sidecar() {
        // Mirrors the real gemma-4-31B shard metadata: block_count 60, scalar
        // head_count_kv 4 (full layers only), `key_length{,_swa}` 512/256,
        // sliding_window 1024. The windowed class's 16 KV heads are not in
        // the GGUF at all — derivation must fail closed and ask for a
        // sidecar, never guess.
        let mut metadata = GgufMetadata::new();
        let entries: &[(&str, GgufValue)] = &[
            (
                "general.architecture",
                GgufValue::String("gemma4".to_string()),
            ),
            ("gemma4.block_count", GgufValue::Uint32(60)),
            ("gemma4.attention.head_count", GgufValue::Uint32(32)),
            ("gemma4.attention.head_count_kv", GgufValue::Uint32(4)),
            ("gemma4.attention.key_length", GgufValue::Uint32(512)),
            ("gemma4.attention.key_length_swa", GgufValue::Uint32(256)),
            ("gemma4.attention.value_length", GgufValue::Uint32(512)),
            ("gemma4.attention.value_length_swa", GgufValue::Uint32(256)),
            ("gemma4.attention.sliding_window", GgufValue::Uint32(1_024)),
        ];
        for (key, value) in entries {
            metadata.insert((*key).to_string(), value.clone());
        }
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: hybrid sliding-window geometry requires a sidecar descriptor"
        );
    }

    #[test]
    fn refuses_swa_keys_on_an_allowlisted_arch() {
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.attention.key_length_swa".to_string(),
            GgufValue::Uint32(64),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: hybrid sliding-window geometry requires a sidecar descriptor"
        );
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.attention.sliding_window".to_string(),
            GgufValue::Uint32(1_024),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("sidecar"));
    }

    #[test]
    fn refuses_geometry_fields_above_the_v2_bounds() {
        // 2^32 layers: must error before any `as u32` truncation.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.block_count".to_string(),
            GgufValue::Uint64(1u64 << 32),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: geometry is outside the v2 bounds"
        );

        // Above the 4,096-layer derivation cap.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert("qwen2.block_count".to_string(), GgufValue::Uint64(4_097));
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: geometry is outside the v2 bounds"
        );

        // 2^32 KV heads: must not truncate to zero.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.attention.head_count_kv".to_string(),
            GgufValue::Uint64(1u64 << 32),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: geometry is outside the v2 bounds"
        );

        // 2^32 head_dim via key_length: must not truncate to zero.
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.attention.key_length".to_string(),
            GgufValue::Uint64(1u64 << 32),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: geometry is outside the v2 bounds"
        );
    }

    #[test]
    fn refuses_unknown_architectures() {
        for arch in ["mamba2", "qwen3next"] {
            let mut metadata = GgufMetadata::new();
            metadata.insert(
                "general.architecture".to_string(),
                GgufValue::String(arch.to_string()),
            );
            let error = derive_layout_from_metadata(&metadata).unwrap_err();
            assert_eq!(
                error.to_string(),
                "gguf layout underivable: architecture is outside the derivation allowlist"
            );
        }
    }

    #[test]
    fn refuses_missing_head_count_kv() {
        let mut metadata = qwen25_7b_metadata();
        metadata.remove("qwen2.attention.head_count_kv");
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: attention.head_count_kv is missing"
        );
    }

    #[test]
    fn refuses_indivisible_embedding_length() {
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.embedding_length".to_string(),
            GgufValue::Uint32(3_585),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf layout underivable: embedding_length is not divisible by attention.head_count"
        );
    }

    #[test]
    fn refuses_nonuniform_head_count_kv_array() {
        let mut metadata = qwen25_7b_metadata();
        metadata.insert(
            "qwen2.attention.head_count_kv".to_string(),
            GgufValue::Array(vec![GgufValue::Uint32(4), GgufValue::Uint32(8)]),
        );
        let error = derive_layout_from_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("sidecar"));
    }
}
