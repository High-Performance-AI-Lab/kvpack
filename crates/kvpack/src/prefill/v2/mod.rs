use super::*;

// ── v2: registered layout tables (docs/PROTOCOL_V2_DESIGN.md) ────────
// The unit of admission is a named layout — per-class geometry with
// optional sliding-window truncation — not a uniform model tuple. The
// two v1 tuples are subsumed as single-class layouts; anything outside
// the registry still hard-errors.

/// The post-RoPE v2 portable ABI label. Mirrored by
/// `kvpack_handoff::PORTABLE_KV_ABI_V2`; the kvpack-cli tests pin the
/// equality so the two can never drift.
pub const PORTABLE_PREFILL_ABI_V2: &str = "canonical-kv-v2";
/// The pre-RoPE capture family label (docs/PREROPE_CAPTURE_CONTRACT.md):
/// K planes persist as pre-RoPE post-bias f32 and the consumer rotates
/// once at install inside its pinned kernel. Mirrored by
/// `kvpack_handoff::PORTABLE_KV_ABI_V2_PREROPE`.
pub const PORTABLE_PREFILL_ABI_V2_PREROPE: &str = "canonical-kv-prerope-v2";

/// The pinned install-time rotation kernel identity of the pre-RoPE
/// capture family (docs/PREROPE_CAPTURE_CONTRACT.md §3). Mandatory with
/// the pre-RoPE label, forbidden without it; bound into the
/// engine-cache-abi/v3 identity so two different pins can never share a
/// representation family. Supplied at arm time, never from the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreRopeKernelPinV1 {
    /// Engine + device class, e.g. `ferrite-metal`.
    pub engine: String,
    /// Exact rotation kernel, e.g. `rope_store_kv_batch_cached`.
    pub kernel: String,
    /// Dtype path, e.g. `f32-k-in-f16-cache`.
    pub dtype_path: String,
    /// Rotation convention; must equal every class's `rope_convention`.
    pub convention: RopeConvention,
}

impl PreRopeKernelPinV1 {
    /// Canonical byte serialization bound into the identity hash:
    /// length-prefixed fields in declaration order.
    fn identity_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for field in [&self.engine, &self.kernel, &self.dtype_path] {
            bytes.extend_from_slice(&(field.len() as u32).to_le_bytes());
            bytes.extend_from_slice(field.as_bytes());
        }
        let convention = self.convention.as_str();
        bytes.extend_from_slice(&(convention.len() as u32).to_le_bytes());
        bytes.extend_from_slice(convention.as_bytes());
        bytes
    }
}

/// One layer class inside a registered v2 layout. Layers are
/// `from..until` stepped by `step`, minus `except`; `window_tokens > 0`
/// exports only the trailing in-window tokens per plane. The `rope_*`
/// fields pin the class's RoPE configuration — bound into the
/// engine-cache-abi/v3 identity (docs/KV_ALGEBRA_2026-08-09.md, item 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePrefillLayoutClassV2 {
    pub class: &'static str,
    pub from: u32,
    pub until: u32,
    pub step: u32,
    pub except: &'static [u32],
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window_tokens: u32,
    /// RoPE frequency base as exact f64 bits — the identity binds the bits,
    /// never a formatted float.
    pub rope_freq_base_bits: u64,
    /// Rotary width (full `head_dim` for full-rotary classes).
    pub rope_dimension_count: u32,
    /// Canonical scaling label: `none` or `{linear|ntk|yarn}:{factor}`.
    pub rope_scaling: &'static str,
    /// Pairing convention of the cached K bytes.
    pub rope_convention: RopeConvention,
}

impl PortablePrefillLayoutClassV2 {
    pub fn layers(&self) -> Vec<u32> {
        (self.from..self.until)
            .step_by(self.step.max(1) as usize)
            .filter(|layer| !self.except.contains(layer))
            .collect()
    }
}

/// A named, registered v2 layout table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePrefillLayoutV2 {
    pub name: &'static str,
    pub num_layers: u32,
    pub classes: &'static [PortablePrefillLayoutClassV2],
}

// Qwen2.5 7B: rope_theta 1e6 (GGUF `qwen2.rope.freq_base`), full rotary
// (no rope.dimension_count key), no scaling; the qualified engines' cache
// bytes are NEOX-paired (llama.cpp measured in docs/GPU_PHASE_2026-08-09.md).
const QWEN25_7B_V2_CLASSES: &[PortablePrefillLayoutClassV2] = &[PortablePrefillLayoutClassV2 {
    class: "gqa-full",
    from: 0,
    until: 28,
    step: 1,
    except: &[],
    kv_heads: 4,
    head_dim: 128,
    window_tokens: 0,
    rope_freq_base_bits: 1_000_000.0f64.to_bits(),
    rope_dimension_count: 128,
    rope_scaling: "none",
    rope_convention: RopeConvention::Neox,
}];

// gpt-oss 120B: 36 layers alternating sliding-window(128) and full
// attention, 8 KV heads at head_dim 64. Parity verified against the real
// model config (huggingface.co/openai/gpt-oss-120b/raw/main/config.json):
// layer_types[0] == "sliding_attention" and alternates, so EVEN layers are
// windowed and ODD layers are full attention; sliding_window == 128. The
// real GGUF carries `gpt-oss.attention.sliding_window`, so metadata
// derivation fails closed toward a sidecar — this registry entry is what
// describes the real geometry. Rope: theta 150000 with yarn factor 32 (the
// same config.json), full rotary at head_dim 64, NEOX pairing; both
// classes share one rope config.
const GPT_OSS_120B_CLASSES: &[PortablePrefillLayoutClassV2] = &[
    PortablePrefillLayoutClassV2 {
        class: "gqa-windowed",
        from: 0,
        until: 36,
        step: 2,
        except: &[],
        kv_heads: 8,
        head_dim: 64,
        window_tokens: 128,
        rope_freq_base_bits: 150_000.0f64.to_bits(),
        rope_dimension_count: 64,
        rope_scaling: "yarn:32",
        rope_convention: RopeConvention::Neox,
    },
    PortablePrefillLayoutClassV2 {
        class: "gqa-full",
        from: 1,
        until: 36,
        step: 2,
        except: &[],
        kv_heads: 8,
        head_dim: 64,
        window_tokens: 0,
        rope_freq_base_bits: 150_000.0f64.to_bits(),
        rope_dimension_count: 64,
        rope_scaling: "yarn:32",
        rope_convention: RopeConvention::Neox,
    },
];

// Gemma 4 31B: 60 layers; every 6th (5, 11, …, 59) is full attention with
// 4 KV heads at head_dim 512; the remaining 50 are sliding-window-1024 with
// 16 KV heads at head_dim 256. Geometry verified against the GGUF metadata
// (attention.head_count_kv array, key_length{,_swa}) and a live llama.cpp
// session export (plane rows 4096 / 8192 bytes). The rope config is
// per-class, also from the GGUF (`gemma4.rope.freq_base{,_swa}`,
// `dimension_count{,_swa}`): the windowed layers rotate at base 1e4 over
// 256 dims, the full-attention layers at base 1e6 over 512 dims.
const GEMMA4_31B_FULL_EXCEPT: &[u32] = &[5, 11, 17, 23, 29, 35, 41, 47, 53, 59];
const GEMMA4_31B_CLASSES: &[PortablePrefillLayoutClassV2] = &[
    PortablePrefillLayoutClassV2 {
        class: "gqa-windowed",
        from: 0,
        until: 60,
        step: 1,
        except: GEMMA4_31B_FULL_EXCEPT,
        kv_heads: 16,
        head_dim: 256,
        window_tokens: 1_024,
        rope_freq_base_bits: 10_000.0f64.to_bits(),
        rope_dimension_count: 256,
        rope_scaling: "none",
        rope_convention: RopeConvention::Neox,
    },
    PortablePrefillLayoutClassV2 {
        class: "gqa-full",
        from: 5,
        until: 60,
        step: 6,
        except: &[],
        kv_heads: 4,
        head_dim: 512,
        window_tokens: 0,
        rope_freq_base_bits: 1_000_000.0f64.to_bits(),
        rope_dimension_count: 512,
        rope_scaling: "none",
        rope_convention: RopeConvention::Neox,
    },
];

// Muse Glimmer 30B: 52 layers = [SWA,SWA,SWA,full]×13. 39 SWA layers:
// window 2048, RoPE theta 500000, permanently ring-bounded (2 MiB/layer at
// 1024 B/token BF16). 13 full layers: NoPE (theta 0), unbounded growth,
// position-free by construction. GQA 2 KV heads × head_dim 128.
// Geometry from the muse-glimmer onboarding plan (ferrite-research),
// §0.1 + Part 1 §1.2.
//
// Pairing convention corrected 2026-08-11: Ferrite's CPU reference forward
// pass, transcribed from llama.cpp's Muse implementation and checked for
// token equality, applies LLAMA_ROPE_TYPE_NORM to adjacent pairs
// `(x[2i], x[2i+1])`. The independent GPU routing audit likewise rejects
// NEOX-only kernels for Muse NORM layout. The windowed Muse planes therefore
// use interleaved pairing; the earlier NEOX value was an unverified guess.
const MUSE_GLIMMER_30B_FULL_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51];
const MUSE_GLIMMER_30B_CLASSES: &[PortablePrefillLayoutClassV2] = &[
    PortablePrefillLayoutClassV2 {
        class: "gqa-windowed",
        from: 0,
        until: 52,
        step: 1,
        except: MUSE_GLIMMER_30B_FULL_LAYERS,
        kv_heads: 2,
        head_dim: 128,
        window_tokens: 2_048,
        rope_freq_base_bits: 500_000.0f64.to_bits(),
        rope_dimension_count: 128,
        rope_scaling: "none",
        rope_convention: RopeConvention::Interleaved,
    },
    PortablePrefillLayoutClassV2 {
        class: "gqa-full",
        from: 3,
        until: 52,
        step: 4,
        except: &[],
        kv_heads: 2,
        head_dim: 128,
        window_tokens: 0,
        rope_freq_base_bits: 0.0f64.to_bits(),
        rope_dimension_count: 0,
        rope_scaling: "none",
        rope_convention: RopeConvention::None,
    },
];

pub const PORTABLE_PREFILL_LAYOUTS_V2: &[PortablePrefillLayoutV2] = &[
    PortablePrefillLayoutV2 {
        name: "qwen2.5-7b",
        num_layers: 28,
        classes: QWEN25_7B_V2_CLASSES,
    },
    PortablePrefillLayoutV2 {
        name: "gpt-oss-120b",
        num_layers: 36,
        classes: GPT_OSS_120B_CLASSES,
    },
    PortablePrefillLayoutV2 {
        name: "gemma4-31b",
        num_layers: 60,
        classes: GEMMA4_31B_CLASSES,
    },
    PortablePrefillLayoutV2 {
        name: "muse-glimmer-30b",
        num_layers: 52,
        classes: MUSE_GLIMMER_30B_CLASSES,
    },
];

/// Resolve a registered v2 layout by name. Unknown layout is an error,
/// never a guess — the registry replaces the v1 closed tuples without
/// loosening the fail-closed posture.
pub fn portable_prefill_layout_v2(
    name: &str,
) -> Result<&'static PortablePrefillLayoutV2, StoreError> {
    PORTABLE_PREFILL_LAYOUTS_V2
        .iter()
        .find(|layout| layout.name == name)
        .ok_or(StoreError::Expectation(
            "portable prefill layout is outside the registered v2 layouts",
        ))
}

/// Validated BEGIN fields for the v2 descriptor derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePrefillDescriptorInputV2 {
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
    pub max_context_tokens: u32,
    pub layout_name: String,
    /// Optional authenticated engine-ABI translation (M4): the canonical
    /// transform descriptor's Id32. `None` is the untransformed artifact;
    /// when set it binds into the representation family exactly like the
    /// other derived fields, so the transformed artifact's identity differs.
    pub transform: Option<Id32>,
    /// The pinned install-time rotation kernel identity. Mandatory iff
    /// `portable_abi` is the pre-RoPE capture label — the derivation fails
    /// closed on a missing pin and on a pin attached to any other label.
    pub prerope_kernel_pin: Option<PreRopeKernelPinV1>,
}

/// One borrowed layer-class view consumed by the v2 descriptor core. Both
/// the `'static` registry classes and derived owned classes lower into it.
#[derive(Debug, Clone, Copy)]
struct LayoutClassView<'a> {
    class: &'a str,
    from: u32,
    until: u32,
    step: u32,
    except: &'a [u32],
    kv_heads: u32,
    head_dim: u32,
    window_tokens: u32,
    rope_freq_base_bits: u64,
    rope_dimension_count: u32,
    rope_scaling: &'a str,
    rope_convention: RopeConvention,
}

impl LayoutClassView<'_> {
    fn layers(&self) -> Vec<u32> {
        (self.from..self.until)
            .step_by(self.step.max(1) as usize)
            .filter(|layer| !self.except.contains(layer))
            .collect()
    }
}

/// State names emitted per layer of a class: mla-latent layers carry one
/// packed latent record plane; everything else emits the K/V pair. This
/// derivation feeds both the descriptor state emission and the v3 identity
/// binding, so the two can never drift apart. `pub(crate)` so the session
/// module (`prefill::session`) can compute the same per-state keys a saved
/// session artifact must cover, without duplicating this table.
pub(crate) fn class_state_names(class: &str) -> &'static [&'static str] {
    if class == crate::mla::MLA_LATENT_LAYOUT_CLASS {
        &[crate::mla::MLA_LATENT_STATE_NAME]
    } else {
        &["attn.k", "attn.v"]
    }
}

/// The number of trailing tokens a class's plane covers for a given
/// `cached_token_count`: the full prefix for an unbounded (`window_tokens ==
/// 0`) class, otherwise the lesser of the window and what has been cached so
/// far. Shared by the descriptor's per-state byte-bound derivation below and
/// by `prefill::session`'s resume-precondition tail-coverage check, so the
/// two can never disagree about how many tokens a windowed plane must carry.
pub(crate) fn effective_window_tokens(window_tokens: u32, cached_token_count: u32) -> u32 {
    if window_tokens == 0 {
        cached_token_count
    } else {
        window_tokens.min(cached_token_count)
    }
}

impl From<&PortablePrefillLayoutClassV2> for LayoutClassView<'_> {
    fn from(class: &PortablePrefillLayoutClassV2) -> Self {
        Self {
            class: class.class,
            from: class.from,
            until: class.until,
            step: class.step,
            except: class.except,
            kv_heads: class.kv_heads,
            head_dim: class.head_dim,
            window_tokens: class.window_tokens,
            rope_freq_base_bits: class.rope_freq_base_bits,
            rope_dimension_count: class.rope_dimension_count,
            rope_scaling: class.rope_scaling,
            rope_convention: class.rope_convention,
        }
    }
}

impl PortablePrefillLayoutV2 {
    /// Lower the registered layout into the owned derivation type.
    pub fn to_owned_layout(&self) -> OwnedLayoutV2 {
        OwnedLayoutV2 {
            name: self.name.to_string(),
            num_layers: self.num_layers,
            classes: self
                .classes
                .iter()
                .map(|class| crate::gguf_layout::OwnedLayoutClassV2 {
                    class: class.class.to_string(),
                    from: class.from,
                    until: class.until,
                    step: class.step,
                    except: class.except.to_vec(),
                    kv_heads: class.kv_heads,
                    head_dim: class.head_dim,
                    window_tokens: class.window_tokens,
                    rope_freq_base_bits: class.rope_freq_base_bits,
                    rope_dimension_count: class.rope_dimension_count,
                    rope_scaling: class.rope_scaling.to_string(),
                    rope_convention: class.rope_convention,
                })
                .collect(),
        }
    }
}

/// Resolve a derived owned layout against the registry oracle: the
/// registered name when the geometry matches exactly (name ignored —
/// derived names are provenance), otherwise `None`.
pub fn portable_prefill_layout_name_v2(layout: &OwnedLayoutV2) -> Option<&'static str> {
    PORTABLE_PREFILL_LAYOUTS_V2
        .iter()
        .find(|entry| layout.matches_registry(entry))
        .map(|entry| entry.name)
}

/// Fail-closed validation of an owned layout about to drive descriptor
/// derivation: sane per-class bounds and an exact, overlap-free partition
/// of `0..num_layers`. Mirrors the sidecar validation; the registry is
/// closed and skips it.
fn validate_owned_layout_v2(layout: &OwnedLayoutV2) -> Result<(), StoreError> {
    if layout.num_layers == 0 || layout.classes.is_empty() {
        return Err(StoreError::Expectation(
            "portable prefill v2 owned layout has no layers or no classes",
        ));
    }
    let mut covered = std::collections::BTreeSet::new();
    for class in &layout.classes {
        if class.from >= class.until
            || class.until > layout.num_layers
            || class.step == 0
            || class.kv_heads == 0
            || class.head_dim == 0
            || class.class.is_empty()
            || class
                .except
                .iter()
                .any(|layer| *layer < class.from || *layer >= class.until)
        {
            return Err(StoreError::Expectation(
                "portable prefill v2 owned layout class bounds are outside the v2 bounds",
            ));
        }
        for layer in class.layers() {
            if !covered.insert(layer) {
                return Err(StoreError::Expectation(
                    "portable prefill v2 owned layout classes overlap",
                ));
            }
        }
        // mla-latent carries the packed per-token record (c_KV ‖ k_rope) as a
        // single full-coverage plane: exactly one vector per token, never a
        // sliding window (crate::mla).
        if class.class == crate::mla::MLA_LATENT_LAYOUT_CLASS
            && (class.kv_heads != 1 || class.window_tokens != 0)
        {
            return Err(StoreError::Expectation(
                "portable prefill v2 mla-latent class must be a single-vector full-coverage record",
            ));
        }
    }
    if covered.len() != layout.num_layers as usize {
        return Err(StoreError::Expectation(
            "portable prefill v2 owned layout classes do not partition the layer range",
        ));
    }
    Ok(())
}

/// Derive the descriptor family from a registered v2 layout: per-class
/// geometry, per-class token axis (full prefix vs trailing window), and
/// per-class byte bounds. `bytes_per_state` is the max single-state
/// bound; `restored_bytes` is the exact total.
pub fn derive_portable_prefill_descriptor_v2(
    input: &PortablePrefillDescriptorInputV2,
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    let layout = portable_prefill_layout_v2(&input.layout_name)?;
    let classes: Vec<LayoutClassView> = layout.classes.iter().map(Into::into).collect();
    derive_portable_prefill_descriptor_v2_core(input, &classes)
}

/// Sibling of `derive_portable_prefill_descriptor_v2` for layouts derived
/// at arm time (GGUF metadata or a JSON sidecar) instead of the registry.
/// The owned layout is validated fail-closed before it drives anything;
/// `input.layout_name` is a provenance label only — it is not resolved
/// and never enters the derived identities.
pub fn derive_portable_prefill_descriptor_v2_from_layout(
    input: &PortablePrefillDescriptorInputV2,
    layout: &OwnedLayoutV2,
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    validate_owned_layout_v2(layout)?;
    let classes: Vec<LayoutClassView> = layout
        .classes
        .iter()
        .map(|class| LayoutClassView {
            class: &class.class,
            from: class.from,
            until: class.until,
            step: class.step,
            except: &class.except,
            kv_heads: class.kv_heads,
            head_dim: class.head_dim,
            window_tokens: class.window_tokens,
            rope_freq_base_bits: class.rope_freq_base_bits,
            rope_dimension_count: class.rope_dimension_count,
            rope_scaling: &class.rope_scaling,
            rope_convention: class.rope_convention,
        })
        .collect();
    derive_portable_prefill_descriptor_v2_core(input, &classes)
}

/// Fail-closed validation of one class's RoPE configuration before it binds
/// into the identity. Runs for registry and derived classes alike — a bad
/// registry constant fails the same way a bad sidecar does.
fn validate_class_rope_v2(class: &LayoutClassView) -> Result<(), StoreError> {
    let freq_base = f64::from_bits(class.rope_freq_base_bits);
    // NoPE / theta-zero: position-free layers (Muse Glimmer's 13 full-attention
    // layers). freq_base == 0, dimension_count == 0, convention None. No
    // rotation at install; the plane is byte-identical producer→consumer.
    // Validated first and separately so that the standard-RoPE bounds below
    // are never relaxed for a non-NoPE class.
    if class.rope_convention == RopeConvention::None {
        if freq_base != 0.0 || class.rope_dimension_count != 0 {
            return Err(StoreError::Expectation(
                "NoPE class (rope_convention None) must have freq_base 0 and dimension_count 0",
            ));
        }
        return Ok(());
    }
    if !freq_base.is_finite() || freq_base <= 1.0 {
        return Err(StoreError::Expectation(
            "portable prefill v2 rope freq_base is outside the supported bounds",
        ));
    }
    if class.rope_dimension_count < 2
        || class.rope_dimension_count % 2 != 0
        || class.rope_dimension_count > class.head_dim
    {
        return Err(StoreError::Expectation(
            "portable prefill v2 rope dimension_count is outside the v2 bounds",
        ));
    }
    if !crate::gguf_layout::is_canonical_rope_scaling(class.rope_scaling) {
        return Err(StoreError::Expectation(
            "portable prefill v2 rope scaling label is not canonical",
        ));
    }
    Ok(())
}

fn derive_portable_prefill_descriptor_v2_core(
    input: &PortablePrefillDescriptorInputV2,
    classes: &[LayoutClassView],
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    let prerope = input.portable_abi == PORTABLE_PREFILL_ABI_V2_PREROPE;
    if input.cached_token_count == 0
        || input.cached_token_count >= input.max_context_tokens
        || input.max_context_tokens > 131_072
        || !(input.portable_abi == PORTABLE_PREFILL_ABI_V2 || prerope)
        || input.compute_precision != "float16"
        || input.kv_precision != "float16"
        || !matches!(
            input.weight_precision.as_str(),
            "q4_k_m" | "q4_k_xl" | "nvfp4" | "mxfp4" | "bf16"
        )
    {
        return Err(StoreError::Expectation(
            "portable prefill v2 descriptor is outside the qualified bounds",
        ));
    }
    // The pre-RoPE label and the pinned rotation kernel identity are
    // inseparable: each fails closed without the other, so a pre-RoPE
    // artifact can never derive a pin-less identity and a pin can never
    // attach to a post-RoPE family.
    match (prerope, &input.prerope_kernel_pin) {
        (true, None) => {
            return Err(StoreError::Expectation(
                "portable prefill v2 pre-rope representation requires the pinned rotation kernel identity",
            ));
        }
        (false, Some(_)) => {
            return Err(StoreError::Expectation(
                "portable prefill v2 pinned rotation kernel identity requires the pre-rope representation label",
            ));
        }
        _ => {}
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
    if let Some(pin) = &input.prerope_kernel_pin {
        for value in [&pin.engine, &pin.kernel, &pin.dtype_path] {
            if value.is_empty() || value.len() > 1024 || !value.is_ascii() {
                return Err(StoreError::Expectation(
                    "portable prefill identity strings must be bounded nonempty ASCII",
                ));
            }
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
    let mut geometry_bytes = Vec::new();
    let mut labeled_geometry_bytes = Vec::new();
    for class in classes {
        validate_class_rope_v2(class)?;
        // The pin's convention must agree with every class's cached-byte
        // convention — a pinned NEOX kernel can never legitimately rotate
        // interleaved-paired content, and vice versa.
        if let Some(pin) = &input.prerope_kernel_pin {
            if pin.convention != class.rope_convention {
                return Err(StoreError::Expectation(
                    "portable prefill v2 pre-rope kernel pin convention disagrees with the class rope convention",
                ));
            }
            // The pre-RoPE family stores K planes for rotation at install;
            // an mla-latent class has no K plane in that sense — its packed
            // record already carries its rotated share — so the family is
            // undefined for it and fails closed.
            if class.class == crate::mla::MLA_LATENT_LAYOUT_CLASS {
                return Err(StoreError::Expectation(
                    "portable prefill v2 pre-rope representation does not support mla-latent classes",
                ));
            }
            // Same reasoning applies to a NoPE class (rope_convention None —
            // Muse Glimmer's 13 full-attention layers): the pre-RoPE family
            // exists so the consumer can rotate K once, at install, inside
            // its pinned kernel. A NoPE class never rotates — the K1 sealed
            // set forces freq_base and dimension_count to zero — so
            // "pre-RoPE" and "post-RoPE" bytes are identical for it and
            // there is no install-time rotation for any pin to describe.
            // Undefined, same as mla-latent; fails closed rather than
            // silently forcing an F32 capture + kernel pin that would never
            // run (this is the "rotation path is not invoked for NoPE
            // classes" invariant, enforced at derivation time, not just by
            // convention).
            if class.rope_convention == RopeConvention::None {
                return Err(StoreError::Expectation(
                    "portable prefill v2 pre-rope representation does not support NoPE (rope_convention None) classes",
                ));
            }
        }
        geometry_bytes.extend_from_slice(&class.from.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.until.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.step.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.kv_heads.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.head_dim.to_le_bytes());
        geometry_bytes.extend_from_slice(&class.window_tokens.to_le_bytes());
        for except in class.except {
            geometry_bytes.extend_from_slice(&except.to_le_bytes());
        }
        // v3 serialization: same numeric geometry, plus the class label and
        // the state-name derivation, length-prefixed. The bare v2 bytes make
        // an mla-latent layout and a gqa layout with identical numeric
        // geometry collide in engine_cache_abi; the label + state names bind
        // the layout SEMANTICS into the identity, not just its numbers.
        labeled_geometry_bytes.extend_from_slice(
            &geometry_bytes[geometry_bytes.len() - (6 + class.except.len()) * 4..],
        );
        labeled_geometry_bytes.extend_from_slice(&(class.class.len() as u32).to_le_bytes());
        labeled_geometry_bytes.extend_from_slice(class.class.as_bytes());
        let state_names = class_state_names(class.class);
        labeled_geometry_bytes.extend_from_slice(&(state_names.len() as u32).to_le_bytes());
        for name in state_names {
            labeled_geometry_bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
            labeled_geometry_bytes.extend_from_slice(name.as_bytes());
        }
        // RoPE configuration (docs/KV_ALGEBRA_2026-08-09.md, item 1): the
        // authenticated identity commits to the exact rotary parameters of
        // the cached K bytes — frequency base as f64 bits, rotary width,
        // canonical scaling label, and pairing convention — so a silent rope
        // mismatch between producer and consumer derives a different family
        // and never reuses. Same length-prefixed part pattern as the class
        // label above, inside the existing engine-cache-abi/v3 domain.
        labeled_geometry_bytes.extend_from_slice(&class.rope_freq_base_bits.to_le_bytes());
        labeled_geometry_bytes.extend_from_slice(&class.rope_dimension_count.to_le_bytes());
        labeled_geometry_bytes.extend_from_slice(&(class.rope_scaling.len() as u32).to_le_bytes());
        labeled_geometry_bytes.extend_from_slice(class.rope_scaling.as_bytes());
        let convention = class.rope_convention.as_str();
        labeled_geometry_bytes.extend_from_slice(&(convention.len() as u32).to_le_bytes());
        labeled_geometry_bytes.extend_from_slice(convention.as_bytes());
    }
    geometry_bytes.extend_from_slice(&input.max_context_tokens.to_le_bytes());
    labeled_geometry_bytes.extend_from_slice(&input.max_context_tokens.to_le_bytes());
    // Domain-separated transform binding: absent and present identities
    // can never collide, and the transform Id32 flows into the engine cache
    // ABI exactly like the layout geometry bytes above.
    let transform_binding: &[u8] = match &input.transform {
        Some(id) => id,
        None => &[],
    };
    // engine-cache-abi/v3 supersedes v2: it hashes the labeled geometry
    // bytes, so same-numeric-geometry layouts of different classes derive
    // different identities. This is a deliberate identity break from v2 —
    // recorded in the FORMAT.md drift register. The labeled bytes also carry
    // the per-class rope config (above), so rope-different layouts of
    // identical numeric geometry and class labels derive different
    // identities too (drift register D8).
    //
    // The pre-RoPE kernel pin is a CONDITIONAL fifth part: it is appended
    // only for the pre-RoPE label, so existing post-RoPE derivations keep
    // the exact four-part hash input (and their identities) unchanged while
    // two different pins can never share a pre-RoPE family.
    let pin_identity = input
        .prerope_kernel_pin
        .as_ref()
        .map(PreRopeKernelPinV1::identity_bytes);
    let mut engine_cache_abi_parts: Vec<&[u8]> = vec![
        input.portable_abi.as_bytes(),
        input.consumer_engine_abi.as_bytes(),
        &labeled_geometry_bytes,
        transform_binding,
    ];
    if let Some(pin_identity) = &pin_identity {
        engine_cache_abi_parts.push(pin_identity);
    }
    let engine_cache_abi = domain_id(
        b"kvpack/spark-prefill/engine-cache-abi/v3\0",
        &engine_cache_abi_parts,
    );
    let topology = domain_id(b"kvpack/spark-prefill/topology/v2\0", &[&geometry_bytes]);
    let shard_map = domain_id(
        b"kvpack/spark-prefill/shard-map/v2\0",
        &[b"single-rank-layout-table-declared-order"],
    );

    let mut family_states = Vec::new();
    let mut states = Vec::new();
    let mut bytes_per_state = 0u64;
    let mut restored_bytes = 0u64;
    // Descriptor states must be in canonical layer-ascending order; the
    // wire walk (class order) is independent of this emission order.
    let mut ordered: Vec<(u32, &LayoutClassView)> = classes
        .iter()
        .flat_map(|class| class.layers().into_iter().map(move |layer| (layer, class)))
        .collect();
    ordered.sort_by_key(|(layer, _)| *layer);
    for (layer, class) in ordered {
        let elements_per_token = u64::from(class.kv_heads)
            .checked_mul(u64::from(class.head_dim))
            .ok_or(StoreError::State("prefill elements-per-token overflow"))?;
        let window = effective_window_tokens(class.window_tokens, input.cached_token_count);
        let token_axis_rule = if class.window_tokens == 0 {
            TokenAxisRule::Direct
        } else {
            TokenAxisRule::TailWindow
        };
        // mla-latent layers emit one state — the packed latent record plane —
        // instead of the K/V pair (crate::mla).
        let state_names = class_state_names(class.class);
        for &state_name in state_names {
            // Pre-RoPE family: K planes persist as f32 (4-byte elements,
            // rotated at install inside the pinned kernel); V planes stay
            // f16 exactly as in the post-RoPE family.
            let (state_dtype, element_bytes) = if prerope && state_name == "attn.k" {
                (DType::F32, 4u64)
            } else {
                (DType::F16, 2u64)
            };
            let state_bytes = u64::from(window)
                .checked_mul(elements_per_token)
                .and_then(|value| value.checked_mul(element_bytes))
                .ok_or(StoreError::State("prefill state byte bound overflow"))?;
            bytes_per_state = bytes_per_state.max(state_bytes);
            let key = StateKey::new(layer, state_name);
            family_states.push(FamilyState {
                key: key.clone(),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: state_dtype,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule,
                token_axis: 0,
                elements_per_token,
                dimensions: vec![
                    StaticDimension::Token,
                    StaticDimension::Fixed(u64::from(class.kv_heads)),
                    StaticDimension::Fixed(u64::from(class.head_dim)),
                ],
                dependencies: Vec::new(),
            });
            states.push(ExportStateDeclaration {
                key,
                strides: vec![elements_per_token, u64::from(class.head_dim), 1],
                atomic_group: layer + 1,
            });
            restored_bytes = restored_bytes
                .checked_add(state_bytes)
                .ok_or(StoreError::State("prefill restored-byte bound overflow"))?;
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
    Ok(PortablePrefillDescriptorV1 {
        semantic_model,
        family,
        states,
        bytes_per_state,
        restored_bytes,
    })
}

// ── K4: checkpoint-intrinsic scalar-math identity coverage ───────────
// (the ferrite-research onboarding plan,
// Part 0.5.1 + Part 0.5.3, format/identity change list item K4). Muse
// Glimmer surfaced the gap — its `qk_scale_factor = 3.87` is really a
// synthetic QK-RMSNorm weight, not a bare scalar multiply, and the artifact
// exposed two silent-divergence traps: GGUF's f32-rounded
// `output_multiplier` (0.1961161345243454) vs config.json's f64 value
// (0.19611613513818404), and GGUF's `layer_norm_rms_epsilon` (1e-5) vs
// config's `post_norm_eps` (1e-8) — "a one-eps engine and a two-eps engine
// will diverge" on byte-identical weights. None of the four scalars below
// had any structured representation before this change.

/// FREEZE-SENSITIVE (Z1-reserved). Checkpoint-intrinsic scalar constants
/// that parameterize the forward math exactly like `weight_precision`
/// parameterizes storage encoding — `qk_scale_factor`, `output_multiplier`,
/// `final_logit_softcapping`, `post_norm_eps`. The identity binds the raw
/// f64 bits, never a formatted float, matching the rope fields' own
/// convention (D8): two checkpoints agreeing on every other field but
/// disagreeing on one bit pattern here must never share an identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightsScalarMathV1 {
    pub qk_scale_factor_bits: u64,
    pub output_multiplier_bits: u64,
    pub final_logit_softcapping_bits: u64,
    pub post_norm_eps_bits: u64,
}

/// Bind `WeightsScalarMathV1` into an already-derived v2 descriptor's
/// `weights_config` identity (K4), re-deriving `semantic_model.weights_config`
/// as a domain-separated hash of the prior digest plus the four scalar
/// fields — the same digest-of-digest composition `spec/IDENTITY_V1.md`
/// already uses to fold `semantic_digest`/`family_digest` into the token
/// prefix chain's `context` value, applied one layer down.
///
/// FREEZE-SENSITIVE (Z1-reserved), and deliberately a **strictly additive
/// new entry point** rather than new required fields on
/// `PortablePrefillDescriptorInputV2`: every existing caller of
/// `derive_portable_prefill_descriptor_v2` / `_from_layout` —
/// including kvpack-cli's `qualify-transform` and `persist-prefill-v2`
/// gates — is completely unaffected and needs no change (see
/// `weights_scalar_math_is_additive_existing_identities_unchanged` below).
/// A model that opts in (Muse Glimmer, see the registry tests) gets the
/// full guarantee: any of the four scalars disagreeing between producer and
/// consumer derives a different `weights_config`, hence a different
/// `semantic_model` digest (`kvpack_core::semantic_model_id`), hence a
/// restore miss rather than a silent wrong answer. A model that does not
/// opt in keeps exactly its pre-K4 identity, unchanged bit for bit.
///
/// **MANDATORY for Muse Glimmer 30B and any layout with NoPE
/// (rope_convention None) classes.** Muse's `qk_scale_factor` is a
/// QK-RMSNorm weight (not a bare multiply); omitting the binder means
/// the identity does not cover it — a silent wrong-restore risk. Callers
/// producing a Muse descriptor MUST call this before pack/restore.
pub fn bind_weights_scalar_math_v2(
    mut descriptor: PortablePrefillDescriptorV1,
    scalars: &WeightsScalarMathV1,
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    let qk_scale_factor = f64::from_bits(scalars.qk_scale_factor_bits);
    let output_multiplier = f64::from_bits(scalars.output_multiplier_bits);
    let final_logit_softcapping = f64::from_bits(scalars.final_logit_softcapping_bits);
    let post_norm_eps = f64::from_bits(scalars.post_norm_eps_bits);
    // qk_scale_factor and output_multiplier are multiplicative scales: zero,
    // negative, or non-finite is never a legitimate calibration. Softcap is
    // legitimately zero (disabled — Muse itself has no attn_logit_softcapping,
    // only a nonzero final_logit_softcapping); eps must be a small positive
    // number by definition.
    if !qk_scale_factor.is_finite()
        || qk_scale_factor <= 0.0
        || !output_multiplier.is_finite()
        || output_multiplier <= 0.0
        || !final_logit_softcapping.is_finite()
        || final_logit_softcapping < 0.0
        || !post_norm_eps.is_finite()
        || post_norm_eps <= 0.0
    {
        return Err(StoreError::Expectation(
            "portable prefill v2 weights scalar-math fields are outside the supported bounds",
        ));
    }
    descriptor.semantic_model.weights_config = domain_id(
        b"kvpack/spark-prefill/weights-config-scalar-math/v1\0",
        &[
            &descriptor.semantic_model.weights_config,
            &scalars.qk_scale_factor_bits.to_le_bytes(),
            &scalars.output_multiplier_bits.to_le_bytes(),
            &scalars.final_logit_softcapping_bits.to_le_bytes(),
            &scalars.post_norm_eps_bits.to_le_bytes(),
        ],
    );
    Ok(descriptor)
}

#[cfg(test)]
mod tests;
