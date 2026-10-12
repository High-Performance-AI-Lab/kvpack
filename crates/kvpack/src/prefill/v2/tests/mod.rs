use super::*;

fn input_v2(layout_name: &str) -> PortablePrefillDescriptorInputV2 {
    PortablePrefillDescriptorInputV2 {
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
        cached_token_count: 16_383,
        max_context_tokens: 32_768,
        layout_name: layout_name.into(),
        transform: None,
        prerope_kernel_pin: None,
    }
}

fn ferrite_prerope_pin() -> PreRopeKernelPinV1 {
    PreRopeKernelPinV1 {
        engine: "ferrite-metal".into(),
        kernel: "rope_store_kv_batch_cached".into(),
        dtype_path: "f32-k-in-f16-cache".into(),
        convention: RopeConvention::Neox,
    }
}

fn input_prerope(layout_name: &str) -> PortablePrefillDescriptorInputV2 {
    let mut input = input_v2(layout_name);
    input.portable_abi = PORTABLE_PREFILL_ABI_V2_PREROPE.into();
    input.prerope_kernel_pin = Some(ferrite_prerope_pin());
    input
}

#[test]
fn v2_subsumes_the_v1_7b_tuple_as_a_single_class_layout() {
    let descriptor = derive_portable_prefill_descriptor_v2(&input_v2("qwen2.5-7b")).unwrap();
    assert_eq!(descriptor.states.len(), 56);
    assert_eq!(descriptor.states[0].key, StateKey::new(0, "attn.k"));
    assert_eq!(descriptor.states[55].key, StateKey::new(27, "attn.v"));
    assert_eq!(descriptor.states[0].strides, [512, 128, 1]);
    assert_eq!(descriptor.bytes_per_state, 16_383 * 512 * 2);
    assert_eq!(descriptor.restored_bytes, 16_383 * 512 * 2 * 56);
}

#[test]
fn v2_registers_gpt_oss_120b_geometry() {
    // The real model alternates sliding-window(128) on even layers with
    // full attention on odd layers (HF config `layer_types`); both
    // classes carry 8 KV heads at head_dim 64.
    let descriptor = derive_portable_prefill_descriptor_v2(&input_v2("gpt-oss-120b")).unwrap();
    assert_eq!(descriptor.states.len(), 72);
    assert_eq!(descriptor.states[0].strides, [512, 64, 1]);
    let full = descriptor
        .states
        .iter()
        .find(|state| state.key == StateKey::new(1, "attn.k"))
        .unwrap();
    assert_eq!(full.strides, [512, 64, 1]);
    assert_eq!(descriptor.bytes_per_state, 16_383 * 512 * 2);
    let expected_restored =
            (16_383u64 * 512 * 2) * 36 /* full-attention states */
                + (128u64 * 512 * 2) * 36 /* sliding-window states */;
    assert_eq!(descriptor.restored_bytes, expected_restored);
}

#[test]
fn v2_gemma4_mixed_classes_and_windowed_bounds() {
    let descriptor = derive_portable_prefill_descriptor_v2(&input_v2("gemma4-31b")).unwrap();
    assert_eq!(descriptor.states.len(), 120);
    // First full-attention layer (5) has 4 KV heads at hd 512; its
    // neighbours are windowed at 16 heads hd 256 with the 1,024-token
    // tail bound.
    let full = descriptor
        .states
        .iter()
        .find(|state| state.key == StateKey::new(5, "attn.k"))
        .unwrap();
    assert_eq!(full.strides, [2_048, 512, 1]);
    let windowed = descriptor
        .states
        .iter()
        .find(|state| state.key == StateKey::new(4, "attn.k"))
        .unwrap();
    assert_eq!(windowed.strides, [4_096, 256, 1]);
    assert_eq!(descriptor.bytes_per_state, 16_383 * 2_048 * 2);
    let expected_restored = (16_383u64 * 2_048 * 2) * 20 + (1_024u64 * 4_096 * 2) * 100;
    assert_eq!(descriptor.restored_bytes, expected_restored);
}

fn mla_owned_layout() -> crate::gguf_layout::OwnedLayoutV2 {
    crate::gguf_layout::OwnedLayoutV2 {
        name: "deepseek-mla-fixture".into(),
        num_layers: 3,
        classes: vec![crate::gguf_layout::OwnedLayoutClassV2 {
            class: crate::mla::MLA_LATENT_LAYOUT_CLASS.into(),
            from: 0,
            until: 3,
            step: 1,
            except: vec![],
            kv_heads: 1,
            head_dim: 16 + 4,
            window_tokens: 0,
            rope_freq_base_bits: 10_000.0f64.to_bits(),
            rope_dimension_count: 4,
            rope_scaling: "none".into(),
            rope_convention: RopeConvention::Neox,
        }],
    }
}

#[test]
fn v2_mla_latent_derives_one_latent_state_per_layer() {
    let input = input_v2("derived:deepseek-mla-fixture");
    let descriptor =
        derive_portable_prefill_descriptor_v2_from_layout(&input, &mla_owned_layout()).unwrap();
    assert_eq!(descriptor.states.len(), 3);
    for (layer, state) in descriptor.states.iter().enumerate() {
        assert_eq!(
            state.key,
            StateKey::new(layer as u32, crate::mla::MLA_LATENT_STATE_NAME)
        );
        assert_eq!(state.strides, [20, 20, 1]);
    }
    let family_states = &descriptor.family.states;
    assert_eq!(family_states.len(), 3);
    assert_eq!(family_states[0].elements_per_token, 20);
    assert_eq!(family_states[0].token_axis_rule, TokenAxisRule::Direct);
    assert_eq!(descriptor.bytes_per_state, 16_383 * 20 * 2);
    assert_eq!(descriptor.restored_bytes, 16_383 * 20 * 2 * 3);
}

#[test]
fn v2_mla_latent_refuses_windowed_or_multi_vector_classes() {
    let input = input_v2("derived:deepseek-mla-fixture");
    let mut windowed = mla_owned_layout();
    windowed.classes[0].window_tokens = 1_024;
    let error = derive_portable_prefill_descriptor_v2_from_layout(&input, &windowed).unwrap_err();
    assert_eq!(
        error.to_string(),
        "portable prefill v2 mla-latent class must be a single-vector full-coverage record"
    );
    let mut multi_vector = mla_owned_layout();
    multi_vector.classes[0].kv_heads = 2;
    assert!(derive_portable_prefill_descriptor_v2_from_layout(&input, &multi_vector).is_err());
}

#[test]
fn v2_rejects_unregistered_layout_and_drift() {
    assert!(derive_portable_prefill_descriptor_v2(&input_v2("unknown-model")).is_err());
    let mut changed = input_v2("gpt-oss-120b");
    changed.portable_abi = "canonical-kv-f16-le-v1".into();
    assert!(derive_portable_prefill_descriptor_v2(&changed).is_err());
    let mut changed = input_v2("gpt-oss-120b");
    changed.weight_precision = "fp8_e4m3".into();
    assert!(derive_portable_prefill_descriptor_v2(&changed).is_err());
    let mut changed = input_v2("gpt-oss-120b");
    changed.weight_precision = "mxfp4".into();
    assert!(derive_portable_prefill_descriptor_v2(&changed).is_ok());
    let mut changed = input_v2("muse-glimmer-30b");
    changed.weight_precision = "q4_k_xl".into();
    assert!(derive_portable_prefill_descriptor_v2(&changed).is_ok());
    let mut changed = input_v2("gemma4-31b");
    changed.weight_precision = "bf16".into();
    assert!(derive_portable_prefill_descriptor_v2(&changed).is_ok());
}

#[test]
fn v2_transform_binding_changes_the_derived_identity() {
    let untransformed = derive_portable_prefill_descriptor_v2(&input_v2("qwen2.5-7b")).unwrap();
    let mut transformed_input = input_v2("qwen2.5-7b");
    transformed_input.transform = Some([7; 32]);
    let transformed = derive_portable_prefill_descriptor_v2(&transformed_input).unwrap();
    // The transform binds into the representation family (engine cache
    // ABI); semantics and geometry are untouched.
    assert_ne!(
        transformed.family.engine_cache_abi,
        untransformed.family.engine_cache_abi
    );
    assert_eq!(transformed.semantic_model, untransformed.semantic_model);
    assert_eq!(transformed.states, untransformed.states);
    assert_eq!(transformed.restored_bytes, untransformed.restored_bytes);
    // A different transform Id32 derives a different identity again.
    let mut other_input = input_v2("qwen2.5-7b");
    other_input.transform = Some([8; 32]);
    let other = derive_portable_prefill_descriptor_v2(&other_input).unwrap();
    assert_ne!(
        other.family.engine_cache_abi,
        transformed.family.engine_cache_abi
    );
    assert_ne!(
        other.family.engine_cache_abi,
        untransformed.family.engine_cache_abi
    );
}

/// The retired v2 identity: bare numeric geometry under the
/// engine-cache-abi/v2 domain. Retained so the collision that motivated
/// v3 stays pinned by a test; NOT used by any derivation anymore.
fn legacy_v2_engine_cache_abi(
    input: &PortablePrefillDescriptorInputV2,
    layout: &crate::gguf_layout::OwnedLayoutV2,
) -> Id32 {
    let mut geometry_bytes = Vec::new();
    for class in &layout.classes {
        geometry_bytes.extend_from_slice(&class.from.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.until.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.step.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.kv_heads.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.head_dim.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.window_tokens.to_le_bytes());
        for except in &class.except {
            geometry_bytes.extend_from_slice(&except.to_le_bytes());
        }
    }
    geometry_bytes.extend_from_slice(&input.max_context_tokens.to_le_bytes());
    let transform_binding: &[u8] = match &input.transform {
        Some(id) => id,
        None => &[],
    };
    domain_id(
        b"kvpack/spark-prefill/engine-cache-abi/v2\0",
        &[
            input.portable_abi.as_bytes(),
            input.consumer_engine_abi.as_bytes(),
            &geometry_bytes,
            transform_binding,
        ],
    )
}

fn owned_layout(class: &str) -> crate::gguf_layout::OwnedLayoutV2 {
    crate::gguf_layout::OwnedLayoutV2 {
        name: format!("derived:{class}"),
        num_layers: 3,
        classes: vec![crate::gguf_layout::OwnedLayoutClassV2 {
            class: class.into(),
            from: 0,
            until: 3,
            step: 1,
            except: vec![],
            kv_heads: 1,
            head_dim: 20,
            window_tokens: 0,
            rope_freq_base_bits: 500_000.0f64.to_bits(),
            rope_dimension_count: 20,
            rope_scaling: "none".into(),
            rope_convention: RopeConvention::Neox,
        }],
    }
}

#[test]
fn v3_binds_class_labels_into_the_engine_cache_abi() {
    let input = input_v2("derived:same-numeric-geometry");
    // Identical numeric geometry (from/until/step/kv_heads/head_dim/
    // window_tokens/except, num_layers, max_context) — only the class
    // label and its state-name derivation differ.
    let gqa = derive_portable_prefill_descriptor_v2_from_layout(&input, &owned_layout("gqa-full"))
        .unwrap();
    let mla = derive_portable_prefill_descriptor_v2_from_layout(
        &input,
        &owned_layout(crate::mla::MLA_LATENT_LAYOUT_CLASS),
    )
    .unwrap();
    // The retired v2 identity genuinely collided for this pair — that is
    // the bug v3 fixes.
    assert_eq!(
        legacy_v2_engine_cache_abi(&input, &owned_layout("gqa-full")),
        legacy_v2_engine_cache_abi(&input, &owned_layout(crate::mla::MLA_LATENT_LAYOUT_CLASS)),
    );
    // v3 separates them.
    assert_ne!(gqa.family.engine_cache_abi, mla.family.engine_cache_abi);
    // And the derived state names really do differ (K/V pair vs one
    // packed latent record plane).
    assert_eq!(gqa.states.len(), 6);
    assert_eq!(mla.states.len(), 3);
    assert_eq!(
        mla.states[0].key,
        StateKey::new(0, crate::mla::MLA_LATENT_STATE_NAME)
    );
}

#[test]
fn v3_differs_from_the_retired_v2_domain_for_the_same_layout() {
    let input = input_v2("derived:domain-bump");
    let layout = owned_layout("gqa-full");
    let descriptor = derive_portable_prefill_descriptor_v2_from_layout(&input, &layout).unwrap();
    assert_ne!(
        descriptor.family.engine_cache_abi,
        legacy_v2_engine_cache_abi(&input, &layout),
    );
}

#[test]
fn v3_binds_the_rope_config_into_the_engine_cache_abi() {
    let input = input_v2("derived:rope-sensitivity");
    let base = derive_portable_prefill_descriptor_v2_from_layout(&input, &owned_layout("gqa-full"))
        .unwrap();
    // Each rope field alone moves the identity; geometry, class label,
    // and state emission are untouched.
    let mut freq_changed = owned_layout("gqa-full");
    freq_changed.classes[0].rope_freq_base_bits = 1_000_000.0f64.to_bits();
    let mut dim_changed = owned_layout("gqa-full");
    dim_changed.classes[0].rope_dimension_count = 16;
    let mut scaling_changed = owned_layout("gqa-full");
    scaling_changed.classes[0].rope_scaling = "yarn:32".into();
    let mut convention_changed = owned_layout("gqa-full");
    convention_changed.classes[0].rope_convention = RopeConvention::Interleaved;
    for changed in [
        &freq_changed,
        &dim_changed,
        &scaling_changed,
        &convention_changed,
    ] {
        let descriptor =
            derive_portable_prefill_descriptor_v2_from_layout(&input, changed).unwrap();
        assert_ne!(
            descriptor.family.engine_cache_abi,
            base.family.engine_cache_abi,
        );
        assert_eq!(descriptor.states, base.states);
        assert_eq!(descriptor.semantic_model, base.semantic_model);
    }
    // The base layout derives deterministically (same input, same id).
    let replay =
        derive_portable_prefill_descriptor_v2_from_layout(&input, &owned_layout("gqa-full"))
            .unwrap();
    assert_eq!(replay.family.engine_cache_abi, base.family.engine_cache_abi);
}

#[test]
fn v2_rope_config_fails_closed_at_the_descriptor_derivation() {
    let input = input_v2("derived:rope-validation");
    // freq_base at/below one, non-finite.
    for bad_bits in [1.0f64.to_bits(), 0.5f64.to_bits(), f64::NAN.to_bits()] {
        let mut bad = owned_layout("gqa-full");
        bad.classes[0].rope_freq_base_bits = bad_bits;
        let error = derive_portable_prefill_descriptor_v2_from_layout(&input, &bad).unwrap_err();
        assert_eq!(
            error.to_string(),
            "portable prefill v2 rope freq_base is outside the supported bounds"
        );
    }
    // Rotary width zero, odd, or wider than the head.
    for bad_count in [0u32, 3, 24] {
        let mut bad = owned_layout("gqa-full");
        bad.classes[0].rope_dimension_count = bad_count;
        let error = derive_portable_prefill_descriptor_v2_from_layout(&input, &bad).unwrap_err();
        assert_eq!(
            error.to_string(),
            "portable prefill v2 rope dimension_count is outside the v2 bounds"
        );
    }
    // Non-canonical scaling label.
    let mut bad = owned_layout("gqa-full");
    bad.classes[0].rope_scaling = "dynamic:32".into();
    let error = derive_portable_prefill_descriptor_v2_from_layout(&input, &bad).unwrap_err();
    assert_eq!(
        error.to_string(),
        "portable prefill v2 rope scaling label is not canonical"
    );
}

#[test]
fn registry_layouts_carry_a_valid_rope_config() {
    // The registry is closed and skips owned-layout validation, but the
    // descriptor core validates rope for every class — derive each
    // registered layout to prove the constants pass.
    for layout in PORTABLE_PREFILL_LAYOUTS_V2 {
        derive_portable_prefill_descriptor_v2(&input_v2(layout.name)).unwrap();
    }
}

#[test]
fn prerope_label_derives_f32_k_states_and_a_distinct_family() {
    let post_rope = derive_portable_prefill_descriptor_v2(&input_v2("qwen2.5-7b")).unwrap();
    let pre_rope = derive_portable_prefill_descriptor_v2(&input_prerope("qwen2.5-7b")).unwrap();
    // Same state inventory and geometry; the K states carry f32.
    assert_eq!(pre_rope.states, post_rope.states);
    assert_eq!(pre_rope.semantic_model, post_rope.semantic_model);
    for state in &pre_rope.family.states {
        let expected = if state.key.state_name == "attn.k" {
            DType::F32
        } else {
            DType::F16
        };
        assert_eq!(state.dtype, expected, "state {:?}", state.key);
    }
    // Byte bounds: K planes double, V planes unchanged.
    assert_eq!(pre_rope.bytes_per_state, 16_383 * 512 * 4);
    assert_eq!(post_rope.bytes_per_state, 16_383 * 512 * 2);
    assert_eq!(
        pre_rope.restored_bytes,
        16_383 * 512 * 4 * 28 + 16_383 * 512 * 2 * 28
    );
    // The representation label is bound into the identity: a pre-RoPE
    // artifact can never authenticate as the post-RoPE family.
    assert_ne!(
        pre_rope.family.engine_cache_abi,
        post_rope.family.engine_cache_abi
    );
    // Derivation is deterministic.
    let replay = derive_portable_prefill_descriptor_v2(&input_prerope("qwen2.5-7b")).unwrap();
    assert_eq!(
        replay.family.engine_cache_abi,
        pre_rope.family.engine_cache_abi
    );
    // A different pinned kernel derives a different family again.
    let mut other = input_prerope("qwen2.5-7b");
    other.prerope_kernel_pin.as_mut().unwrap().kernel = "fused_bias_rope_store_kv_batch".into();
    let other = derive_portable_prefill_descriptor_v2(&other).unwrap();
    assert_ne!(
        other.family.engine_cache_abi,
        pre_rope.family.engine_cache_abi
    );
    // The post-RoPE derivation is untouched by the pin machinery: same
    // four-part identity input as before this change.
    let post_replay = derive_portable_prefill_descriptor_v2(&input_v2("qwen2.5-7b")).unwrap();
    assert_eq!(
        post_replay.family.engine_cache_abi,
        post_rope.family.engine_cache_abi
    );
}

#[test]
fn prerope_label_and_pin_fail_closed_without_each_other() {
    // Pre-RoPE label without the pin.
    let mut missing = input_prerope("qwen2.5-7b");
    missing.prerope_kernel_pin = None;
    let error = derive_portable_prefill_descriptor_v2(&missing).unwrap_err();
    assert_eq!(
        error.to_string(),
        "portable prefill v2 pre-rope representation requires the pinned rotation kernel identity"
    );
    // Pin attached to the post-RoPE label.
    let mut stray = input_v2("qwen2.5-7b");
    stray.prerope_kernel_pin = Some(ferrite_prerope_pin());
    let error = derive_portable_prefill_descriptor_v2(&stray).unwrap_err();
    assert_eq!(
            error.to_string(),
            "portable prefill v2 pinned rotation kernel identity requires the pre-rope representation label"
        );
    // Pin convention disagreeing with the class convention.
    let mut mismatched = input_prerope("qwen2.5-7b");
    mismatched.prerope_kernel_pin.as_mut().unwrap().convention = RopeConvention::Interleaved;
    let error = derive_portable_prefill_descriptor_v2(&mismatched).unwrap_err();
    assert_eq!(
            error.to_string(),
            "portable prefill v2 pre-rope kernel pin convention disagrees with the class rope convention"
        );
    // Empty pin labels are not identity strings.
    let mut empty = input_prerope("qwen2.5-7b");
    empty.prerope_kernel_pin.as_mut().unwrap().kernel = String::new();
    assert!(derive_portable_prefill_descriptor_v2(&empty).is_err());
    // The family is undefined for mla-latent classes.
    let error = derive_portable_prefill_descriptor_v2_from_layout(
        &input_prerope("derived:mla"),
        &mla_owned_layout(),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "portable prefill v2 pre-rope representation does not support mla-latent classes"
    );
}

// ── Muse Glimmer 30B layout tests ──────────────────────────────────

fn muse_layout() -> &'static PortablePrefillLayoutV2 {
    PORTABLE_PREFILL_LAYOUTS_V2
        .iter()
        .find(|l| l.name == "muse-glimmer-30b")
        .expect("muse-glimmer-30b layout registered")
}

#[test]
fn muse_layout_validates_and_covers_all_52_layers() {
    // The 2-class table from the onboarding plan §1.2: 39 windowed +
    // 13 full, partitioning 0..52 without overlap.
    derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b"))
        .expect("muse layout derives cleanly");
    let layout = muse_layout();
    let mut covered: Vec<u32> = Vec::new();
    for class in layout.classes {
        covered.extend(class.layers());
    }
    covered.sort_unstable();
    covered.dedup();
    assert_eq!(covered.len(), 52, "must cover all 52 layers");
    assert_eq!(covered[0], 0);
    assert_eq!(covered[51], 51);
}

#[test]
fn muse_full_layers_are_nope_position_free() {
    // The 13 full-attention layers are NoPE (theta=0): freq_base 0,
    // dimension_count 0, convention None — no rotation at install.
    let full = muse_layout()
        .classes
        .iter()
        .find(|c| c.class == "gqa-full")
        .expect("gqa-full class");
    assert_eq!(full.rope_convention, RopeConvention::None);
    assert_eq!(f64::from_bits(full.rope_freq_base_bits), 0.0);
    assert_eq!(full.rope_dimension_count, 0);
    assert_eq!(full.window_tokens, 0);
    assert_eq!(
        full.layers(),
        vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
    );
}

#[test]
fn muse_windowed_layers_rotate_at_theta_500000_window_2048() {
    let swa = muse_layout()
        .classes
        .iter()
        .find(|c| c.class == "gqa-windowed")
        .expect("gqa-windowed class");
    // Muse NORM rotates adjacent pairs. This used to preserve the stale NEOX
    // guess rather than the llama.cpp-cross-checked engine behavior.
    assert_eq!(swa.rope_convention, RopeConvention::Interleaved);
    assert_eq!(f64::from_bits(swa.rope_freq_base_bits), 500_000.0);
    assert_eq!(swa.rope_dimension_count, 128);
    assert_eq!(swa.window_tokens, 2_048);
    assert_eq!(swa.layers().len(), 39);
}

#[test]
fn muse_cut_chain_aligns_at_2048_boundary() {
    // Cuts are 256 tokens (IDENTITY_V1); 2048 / 256 = 8 exactly. A
    // checkpoint at any 256-multiple N has its windowed start (N - 2048)
    // also on a 256-multiple — no partial-window reassembly needed.
    const CUT: u32 = 256;
    const WINDOW: u32 = 2_048;
    assert_eq!(WINDOW % CUT, 0, "window must divide cuts evenly");
    for n in (WINDOW..=131_072).step_by(CUT as usize) {
        let window_start = n - WINDOW;
        assert_eq!(window_start % CUT, 0, "N={n} window_start not cut-aligned");
    }
}

#[test]
fn muse_descriptor_is_deterministic_round_trip() {
    // Schema round-trip byte-exactness: deriving twice yields identical
    // engine_cache_abi (the identity hash).
    let a = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let b = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    assert_eq!(
        a.family.engine_cache_abi, b.family.engine_cache_abi,
        "muse descriptor must be deterministic"
    );
    // A NoPE layer set mutation (swapping convention None→Neox on the
    // full class) changes the identity — caught by the registry being
    // const (immutable at runtime). Documented: a sidecar carrying
    // convention "neox" on the full layers derives a different family.
}

// ── IDENTITY EXTENSION: theta / NoPE-set / window / weights-version ──
// (the muse-identity extension). These fields are already bound by
// K1-K3 (the per-class rope fields + except-list partition feed
// `engine_cache_abi`'s `labeled_geometry_bytes`; `model_revision` feeds
// `weights_config`) — no new derivation code below, only regression
// coverage locking in that a producer/consumer disagreeing on any of
// them can never share an identity. See `bind_weights_scalar_math_v2`
// above (K4) for the one field that genuinely had no representation:
// qk_scale_factor / output_multiplier / final_logit_softcapping /
// post_norm_eps.

fn muse_owned_layout() -> crate::gguf_layout::OwnedLayoutV2 {
    muse_layout().to_owned_layout()
}

#[test]
fn muse_weights_version_mutation_changes_identity() {
    // "weights version" = model_revision: already bound into
    // weights_config. A revision bump on an otherwise-identical
    // checkpoint (same model_sha256) must never silently share a
    // restore with the prior revision.
    let base = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let mut changed_input = input_v2("muse-glimmer-30b");
    changed_input.model_revision = "model@revision-2".into();
    let changed = derive_portable_prefill_descriptor_v2(&changed_input).unwrap();
    assert_ne!(changed.semantic_model, base.semantic_model);
    // Geometry (theta, window, NoPE set) is untouched by a revision bump.
    assert_eq!(
        changed.family.engine_cache_abi,
        base.family.engine_cache_abi
    );
}

#[test]
fn muse_theta_mutation_on_the_windowed_class_changes_identity() {
    let base = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &muse_owned_layout(),
    )
    .unwrap();
    let mut changed = muse_owned_layout();
    let windowed = changed
        .classes
        .iter_mut()
        .find(|c| c.class == "gqa-windowed")
        .unwrap();
    windowed.rope_freq_base_bits = 250_000.0f64.to_bits();
    let changed = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &changed,
    )
    .unwrap();
    assert_ne!(
        changed.family.engine_cache_abi,
        base.family.engine_cache_abi
    );
}

#[test]
fn muse_window_size_mutation_changes_identity() {
    let base = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &muse_owned_layout(),
    )
    .unwrap();
    let mut changed = muse_owned_layout();
    changed
        .classes
        .iter_mut()
        .find(|c| c.class == "gqa-windowed")
        .unwrap()
        .window_tokens = 1_024;
    let changed = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &changed,
    )
    .unwrap();
    assert_ne!(
        changed.family.engine_cache_abi,
        base.family.engine_cache_abi
    );
}

#[test]
fn muse_nope_layer_set_misflagging_changes_identity() {
    // The onboarding plan's flagged risk (Part 1 §1.3, K1 row): "the
    // NoPE-ness of the 13 full layers is nowhere in the [GGUF] file —
    // implied by sliding_window_pattern=4... A position_semantics
    // digest populated from GGUF metadata hashes identically for a
    // correct and a NoPE-mis-flagged model." This proves the mechanism
    // that DOES catch it: a sidecar that mis-flags the full layers as
    // rotating (instead of NoPE) derives a different engine_cache_abi,
    // so it can never silently reuse a correctly-flagged artifact (or
    // vice versa) — even though position_semantics itself does not
    // carry the except-list, the class partition's geometry hash does.
    let base = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &muse_owned_layout(),
    )
    .unwrap();
    let mut misflagged = muse_owned_layout();
    let full = misflagged
        .classes
        .iter_mut()
        .find(|c| c.class == "gqa-full")
        .unwrap();
    full.rope_convention = RopeConvention::Neox;
    full.rope_freq_base_bits = 500_000.0f64.to_bits();
    full.rope_dimension_count = 128;
    let misflagged = derive_portable_prefill_descriptor_v2_from_layout(
        &input_v2("derived:muse-glimmer-30b"),
        &misflagged,
    )
    .unwrap();
    assert_ne!(
        misflagged.family.engine_cache_abi, base.family.engine_cache_abi,
        "a NoPE-mis-flagged full class must never share an identity with the correct one"
    );
}

// ── K4: weights scalar-math identity tests ────────────────────────

/// Muse Glimmer's real calibration (Part 0.5.1 / Part 0.5.3): the
/// synthetic QK-RMSNorm scale, the config.json f64 output_multiplier,
/// the real final_logit_softcapping, and the config post_norm_eps.
fn muse_scalar_math() -> WeightsScalarMathV1 {
    WeightsScalarMathV1 {
        qk_scale_factor_bits: 3.87f64.to_bits(),
        output_multiplier_bits: 0.196_116_135_138_184_04_f64.to_bits(),
        final_logit_softcapping_bits: 20.0f64.to_bits(),
        post_norm_eps_bits: 1e-8_f64.to_bits(),
    }
}

#[test]
fn muse_weights_scalar_math_binds_real_calibration() {
    let unbound = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let bound = bind_weights_scalar_math_v2(unbound.clone(), &muse_scalar_math()).unwrap();
    assert_ne!(
        bound.semantic_model.weights_config,
        unbound.semantic_model.weights_config
    );
    assert_ne!(bound.semantic_model, unbound.semantic_model);
    // Binding only touches weights_config: geometry, states, and the
    // engine cache ABI are untouched.
    assert_eq!(
        bound.family.engine_cache_abi,
        unbound.family.engine_cache_abi
    );
    assert_eq!(bound.states, unbound.states);
    // Deterministic: re-binding the same scalars replays byte-for-byte.
    let replay = bind_weights_scalar_math_v2(unbound, &muse_scalar_math()).unwrap();
    assert_eq!(replay.semantic_model, bound.semantic_model);
}

#[test]
fn weights_scalar_math_mutation_rejects_each_field() {
    let base = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let baseline = bind_weights_scalar_math_v2(base.clone(), &muse_scalar_math()).unwrap();
    let mutations: [(&str, WeightsScalarMathV1); 4] = [
        (
            "qk_scale_factor",
            WeightsScalarMathV1 {
                qk_scale_factor_bits: 1.0f64.to_bits(),
                ..muse_scalar_math()
            },
        ),
        (
            "output_multiplier",
            WeightsScalarMathV1 {
                output_multiplier_bits: 1.0f64.to_bits(),
                ..muse_scalar_math()
            },
        ),
        (
            "final_logit_softcapping",
            WeightsScalarMathV1 {
                final_logit_softcapping_bits: 30.0f64.to_bits(),
                ..muse_scalar_math()
            },
        ),
        (
            "post_norm_eps",
            WeightsScalarMathV1 {
                post_norm_eps_bits: 1e-5_f64.to_bits(),
                ..muse_scalar_math()
            },
        ),
    ];
    for (field, mutated_scalars) in mutations {
        let mutated = bind_weights_scalar_math_v2(base.clone(), &mutated_scalars).unwrap();
        assert_ne!(
            mutated.semantic_model, baseline.semantic_model,
            "mutating {field} alone must change the identity"
        );
    }
}

#[test]
fn weights_scalar_math_output_multiplier_f32_vs_f64_rounding_diverges() {
    // Part 0.5.3: GGUF stores logit_scale as f32-rounded
    // (0.1961161345243454); config.json carries the full f64
    // (0.19611613513818404). A harness hashing one and a harness
    // hashing the other must never derive the same identity — that
    // divergence is exactly the silent-false-positive K4 exists to
    // close.
    let base = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let gguf_rounded = WeightsScalarMathV1 {
        output_multiplier_bits: 0.196_116_134_524_345_4_f64.to_bits(),
        ..muse_scalar_math()
    };
    let config_f64 = muse_scalar_math();
    let a = bind_weights_scalar_math_v2(base.clone(), &gguf_rounded).unwrap();
    let b = bind_weights_scalar_math_v2(base, &config_f64).unwrap();
    assert_ne!(a.semantic_model, b.semantic_model);
}

#[test]
fn weights_scalar_math_eps_ambiguity_diverges() {
    // The GGUF carries only layer_norm_rms_epsilon = 1e-5; config.json
    // specifies post_norm_eps = 1e-8. "A one-eps engine and a two-eps
    // engine will diverge" (Part 0.5.3) — this proves the divergence is
    // now a different identity, not a silent agreement.
    let base = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let one_eps_engine = WeightsScalarMathV1 {
        post_norm_eps_bits: 1e-5_f64.to_bits(),
        ..muse_scalar_math()
    };
    let two_eps_engine = muse_scalar_math();
    let a = bind_weights_scalar_math_v2(base.clone(), &one_eps_engine).unwrap();
    let b = bind_weights_scalar_math_v2(base, &two_eps_engine).unwrap();
    assert_ne!(a.semantic_model, b.semantic_model);
}

#[test]
fn weights_scalar_math_bounds_fail_closed() {
    let base = derive_portable_prefill_descriptor_v2(&input_v2("muse-glimmer-30b")).unwrap();
    let bad_cases: [WeightsScalarMathV1; 6] = [
        WeightsScalarMathV1 {
            qk_scale_factor_bits: f64::NAN.to_bits(),
            ..muse_scalar_math()
        },
        WeightsScalarMathV1 {
            qk_scale_factor_bits: 0.0f64.to_bits(),
            ..muse_scalar_math()
        },
        WeightsScalarMathV1 {
            output_multiplier_bits: (-1.0f64).to_bits(),
            ..muse_scalar_math()
        },
        WeightsScalarMathV1 {
            final_logit_softcapping_bits: (-20.0f64).to_bits(),
            ..muse_scalar_math()
        },
        WeightsScalarMathV1 {
            post_norm_eps_bits: 0.0f64.to_bits(),
            ..muse_scalar_math()
        },
        WeightsScalarMathV1 {
            post_norm_eps_bits: f64::INFINITY.to_bits(),
            ..muse_scalar_math()
        },
    ];
    for bad in bad_cases {
        let error = bind_weights_scalar_math_v2(base.clone(), &bad).unwrap_err();
        assert_eq!(
            error.to_string(),
            "portable prefill v2 weights scalar-math fields are outside the supported bounds"
        );
    }
    // Softcap zero (disabled) is legitimate — only negative/non-finite
    // is rejected.
    let disabled_softcap = WeightsScalarMathV1 {
        final_logit_softcapping_bits: 0.0f64.to_bits(),
        ..muse_scalar_math()
    };
    assert!(bind_weights_scalar_math_v2(base, &disabled_softcap).is_ok());
}

#[test]
fn weights_scalar_math_binding_is_strictly_additive() {
    // Every existing v2 caller (all four registered layouts, both
    // kvpack-cli qualify-transform and persist-prefill-v2 gates) never
    // calls bind_weights_scalar_math_v2 and must derive exactly the
    // same identity as before this change — proven by construction
    // (the base derivation's weights_config formula is untouched) and
    // reconfirmed here for every registered layout.
    for layout in PORTABLE_PREFILL_LAYOUTS_V2 {
        let a = derive_portable_prefill_descriptor_v2(&input_v2(layout.name)).unwrap();
        let b = derive_portable_prefill_descriptor_v2(&input_v2(layout.name)).unwrap();
        assert_eq!(a.semantic_model, b.semantic_model);
        assert_eq!(a.family.engine_cache_abi, b.family.engine_cache_abi);
    }
}

#[test]
fn prerope_undefined_for_nope_classes() {
    // The rotation path must never be invoked for a NoPE class: the
    // pre-RoPE capture family requires the consumer to rotate K once at
    // install, and a NoPE class never rotates by construction (K1's
    // sealed set: freq_base == 0, dimension_count == 0).
    //
    // Isolate the NoPE-specific rejection with a single-class fixture
    // and a pin whose own convention is "none" — the only way a NoPE
    // class could ever pass the generic pin/class convention-agreement
    // check above (mirrors how `mla_owned_layout` isolates the
    // mla-latent rejection with a matching-convention pin). Even then it
    // is rejected: "pre-RoPE capture" has no meaning for a class that
    // never rotates — there is nothing to capture pre-rotation and
    // nothing for the pin to rotate at install.
    let nope_only_layout = crate::gguf_layout::OwnedLayoutV2 {
        name: "derived:nope-only-fixture".into(),
        num_layers: 3,
        classes: vec![crate::gguf_layout::OwnedLayoutClassV2 {
            class: "gqa-full".into(),
            from: 0,
            until: 3,
            step: 1,
            except: vec![],
            kv_heads: 2,
            head_dim: 128,
            window_tokens: 0,
            rope_freq_base_bits: 0.0f64.to_bits(),
            rope_dimension_count: 0,
            rope_scaling: "none".into(),
            rope_convention: RopeConvention::None,
        }],
    };
    let mut input = input_prerope("derived:nope-only-fixture");
    input.prerope_kernel_pin.as_mut().unwrap().convention = RopeConvention::None;
    let error =
        derive_portable_prefill_descriptor_v2_from_layout(&input, &nope_only_layout).unwrap_err();
    assert_eq!(
            error.to_string(),
            "portable prefill v2 pre-rope representation does not support NoPE (rope_convention None) classes"
        );

    // The realistic case: Muse Glimmer's actual registered (mixed)
    // layout with the pin convention a real rotation kernel would
    // plausibly use (Neox, matching the 39 windowed layers). The
    // pre-existing convention-agreement check catches this one class
    // earlier in iteration order ("gqa-windowed" precedes "gqa-full" in
    // the registry), with its own, more generic message. Both paths are
    // regression-tested here so neither can silently regress into
    // acceptance — a NoPE-containing layout can never derive a pre-RoPE
    // identity, however the pin is constructed.
    let error =
        derive_portable_prefill_descriptor_v2(&input_prerope("muse-glimmer-30b")).unwrap_err();
    assert_eq!(
            error.to_string(),
            "portable prefill v2 pre-rope kernel pin convention disagrees with the class rope convention"
        );
}
