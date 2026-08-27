//! GGUF-metadata → v2 layout-table derivation (docs/MODEL_DERIVED_LAYOUTS.md).
//!
//! The derivation compiler replaces hand-maintained registry entries as the
//! way new models enter the portable-prefill lanes. It parses GGUF metadata
//! (header + key/value pairs only, never tensor data) and emits the same
//! layout-table structure the registry carries. The posture is fail-closed:
//! a missing key, an unrecognized architecture, or a geometry the table
//! cannot express is an error, never a guess. What GGUF cannot express
//! (per-class KV-head geometry on hybrid sliding-window models) goes through
//! the JSON sidecar escape hatch. Every class — derived or registered —
//! also carries its RoPE configuration (frequency base as f64 bits, rotary
//! width, canonical scaling label, pairing convention), bound into the
//! engine-cache-abi/v3 identity (docs/KV_ALGEBRA_2026-08-09.md, item 1).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::io_error;
use crate::StoreError;

mod derive;
mod parse;
mod sidecar;

pub(crate) use derive::rope_fields_from_gguf;
pub use derive::{derive_layout_from_gguf, derive_layout_from_metadata};
pub use parse::read_gguf_metadata;
pub use sidecar::{derive_layout_from_sidecar, parse_layout_sidecar};

/// Hard cap on the layer count any v2 layout may declare, in both the GGUF
/// and sidecar derivation paths. Real models are two orders of magnitude
/// below this; the cap keeps per-layer expansion (`layers()`, descriptor
/// state emission) bounded no matter what the input metadata claims.
pub(crate) const MAX_LAYOUT_LAYERS: u64 = 4_096;

/// One GGUF metadata value (GGUF v3, value types 0-12). Tensor data is never
/// read; arrays cannot nest per the format.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(value) => Some(u64::from(*value)),
            Self::Int8(value) => u64::try_from(*value).ok(),
            Self::Uint16(value) => Some(u64::from(*value)),
            Self::Int16(value) => u64::try_from(*value).ok(),
            Self::Uint32(value) => Some(u64::from(*value)),
            Self::Int32(value) => u64::try_from(*value).ok(),
            Self::Uint64(value) => Some(*value),
            Self::Int64(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }
}

/// Parsed GGUF metadata: the key/value section only.
pub type GgufMetadata = BTreeMap<String, GgufValue>;

/// RoPE rotation-pairing convention of the cached K bytes: GPT-NeoX
/// half-split pairs `(i, i + d/2)` vs GPT-J interleaved pairs `(2i, 2i+1)`.
/// The closed label set is bound into the v2 identity; anything else fails
/// closed at parse time (docs/KV_ALGEBRA_2026-08-09.md, item 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeConvention {
    Neox,
    Interleaved,
    /// NoPE / theta-zero: no rotation at install. The plane is position-free
    /// by construction (freq_base == 0, dimension_count == 0). Muse Glimmer's
    /// 13 full-attention layers use this convention.
    None,
}

impl RopeConvention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neox => "neox",
            Self::Interleaved => "interleaved",
            Self::None => "none",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "neox" => Some(Self::Neox),
            "interleaved" => Some(Self::Interleaved),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

/// Per-architecture RoPE pairing convention for the GGUF derivation lanes.
/// GGUF carries no convention key; the pinned value records the measured
/// pairing of the qualified engines' cache bytes for that architecture
/// (llama.cpp's kernel was measured NEOX — pairs (d, d + 64) at head_dim 128
/// — in docs/GPU_PHASE_2026-08-09.md; the vLLM producer lane is neox-style
/// for these architectures). An architecture outside the table derives no
/// convention and fails closed; the sidecar `convention` field is the
/// explicit escape hatch.
pub(crate) fn pinned_rope_convention(architecture: &str) -> Option<RopeConvention> {
    Some(match architecture {
        "llama" | "qwen2" | "gpt-oss" | "deepseek2" => RopeConvention::Neox,
        _ => return None,
    })
}

/// Closed rope-scaling type set for the canonical label
/// (docs/KV_ALGEBRA_2026-08-09.md: none/yarn/ntk, plus linear).
pub(crate) const ROPE_SCALING_TYPES: &[&str] = &["linear", "ntk", "yarn"];

/// Canonical scaling label for a closed-set type and factor:
/// `{type}:{factor}` with the factor in shortest round-trip form, so equal
/// f64 bits produce equal labels. Anything outside the closed set is `None`.
pub(crate) fn rope_scaling_label(kind: &str, factor: f64) -> Option<String> {
    if !ROPE_SCALING_TYPES.contains(&kind) || !factor.is_finite() || factor <= 0.0 {
        return None;
    }
    Some(format!("{kind}:{factor}"))
}

/// A canonical scaling label is exactly `none` or `{type}:{factor}` with a
/// closed-set type and a positive finite factor in shortest round-trip form
/// (the label must re-emit byte-identically, so `yarn:32.0` is refused in
/// favor of `yarn:32`).
pub(crate) fn is_canonical_rope_scaling(label: &str) -> bool {
    if label == "none" {
        return true;
    }
    let Some((kind, factor)) = label.split_once(':') else {
        return false;
    };
    let Ok(factor) = factor.parse::<f64>() else {
        return false;
    };
    rope_scaling_label(kind, factor).as_deref() == Some(label)
}

/// One owned layer class inside a derived v2 layout. Mirrors
/// `crate::prefill::PortablePrefillLayoutClassV2` without the `'static`
/// registry lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedLayoutClassV2 {
    pub class: String,
    pub from: u32,
    pub until: u32,
    pub step: u32,
    pub except: Vec<u32>,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window_tokens: u32,
    /// RoPE frequency base as exact f64 bits (`{arch}.rope.freq_base`) —
    /// the identity binds the bits, never a formatted float.
    pub rope_freq_base_bits: u64,
    /// Rotary width (`{arch}.rope.dimension_count`; the full `head_dim`
    /// when the GGUF carries no partial-rotary key).
    pub rope_dimension_count: u32,
    /// Canonical scaling label (`is_canonical_rope_scaling`).
    pub rope_scaling: String,
    /// Pairing convention of the cached K bytes.
    pub rope_convention: RopeConvention,
}

impl OwnedLayoutClassV2 {
    pub fn layers(&self) -> Vec<u32> {
        (self.from..self.until)
            .step_by(self.step.max(1) as usize)
            .filter(|layer| !self.except.contains(layer))
            .collect()
    }
}

/// An owned, derived v2 layout table. The `name` is provenance only — the
/// geometry (`num_layers` + `classes`) is what must byte-match the registry
/// oracle while the registry exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedLayoutV2 {
    pub name: String,
    pub num_layers: u32,
    pub classes: Vec<OwnedLayoutClassV2>,
}

impl OwnedLayoutV2 {
    /// Field-by-field geometry equality with a registered layout. The name
    /// is ignored: derived names are provenance, not identity.
    pub fn matches_registry(&self, registry: &crate::prefill::PortablePrefillLayoutV2) -> bool {
        self.num_layers == registry.num_layers
            && self.classes.len() == registry.classes.len()
            && self
                .classes
                .iter()
                .zip(registry.classes.iter())
                .all(|(derived, registered)| {
                    derived.class == registered.class
                        && derived.from == registered.from
                        && derived.until == registered.until
                        && derived.step == registered.step
                        && derived.except == registered.except
                        && derived.kv_heads == registered.kv_heads
                        && derived.head_dim == registered.head_dim
                        && derived.window_tokens == registered.window_tokens
                        && derived.rope_freq_base_bits == registered.rope_freq_base_bits
                        && derived.rope_dimension_count == registered.rope_dimension_count
                        && derived.rope_scaling == registered.rope_scaling
                        && derived.rope_convention == registered.rope_convention
                })
    }
}

#[cfg(test)]
pub(crate) fn assert_layout_matches_registry(derived: &OwnedLayoutV2, registry_name: &str) {
    use crate::prefill::{portable_prefill_layout_v2, PortablePrefillLayoutV2};

    let registry: &PortablePrefillLayoutV2 = portable_prefill_layout_v2(registry_name).unwrap();
    assert_eq!(derived.num_layers, registry.num_layers);
    assert_eq!(derived.classes.len(), registry.classes.len());
    for (derived_class, registry_class) in derived.classes.iter().zip(registry.classes) {
        assert_eq!(derived_class.class, registry_class.class);
        assert_eq!(derived_class.from, registry_class.from);
        assert_eq!(derived_class.until, registry_class.until);
        assert_eq!(derived_class.step, registry_class.step);
        assert_eq!(derived_class.except, registry_class.except);
        assert_eq!(derived_class.kv_heads, registry_class.kv_heads);
        assert_eq!(derived_class.head_dim, registry_class.head_dim);
        assert_eq!(derived_class.window_tokens, registry_class.window_tokens);
        assert_eq!(
            derived_class.rope_freq_base_bits,
            registry_class.rope_freq_base_bits
        );
        assert_eq!(
            derived_class.rope_dimension_count,
            registry_class.rope_dimension_count
        );
        assert_eq!(derived_class.rope_scaling, registry_class.rope_scaling);
        assert_eq!(
            derived_class.rope_convention,
            registry_class.rope_convention
        );
    }
}
