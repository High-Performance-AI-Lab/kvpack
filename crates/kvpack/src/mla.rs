//! MLA-native latent layout class (docs/KV_IMPROVEMENT_RESEARCH_2026-08-02.md,
//! "MLA latent-native storage"): the exact model state of a multi-head latent
//! attention layer is the per-token record `c_KV ‖ k_rope`, so the portable
//! layout stores the latent and expands at read time. Everything here is
//! fail-closed: a GGUF without the MLA geometry keys is not an MLA model, a
//! descriptor with unknown fields or bad geometry is refused, and the
//! expansion refuses dimension mismatches — never a guess.

use serde::{Deserialize, Serialize};

use crate::error::StoreError;
use crate::gguf_layout::{GgufMetadata, GgufValue, OwnedLayoutClassV2, OwnedLayoutV2};

/// Layout-table class label for MLA latent records.
pub const MLA_LATENT_LAYOUT_CLASS: &str = "mla-latent";
/// Descriptor state name for the per-layer latent plane (c_KV ‖ k_rope).
pub const MLA_LATENT_STATE_NAME: &str = "attn.kv_latent";
/// Schema version of [`MlaExpansionDescriptor`].
pub const MLA_EXPANSION_DESCRIPTOR_SCHEMA_V1: u32 = 1;

// Fail-closed geometry bounds: these mirror the GGUF parse-bound posture —
// any dimension outside them is an error, never a clamp.
const MLA_MAX_LATENT_DIM: u32 = 65_536;
const MLA_MAX_ROPE_DIM: u32 = 1_024;
const MLA_MAX_HEADS: u32 = 1_024;
const MLA_MAX_HEAD_DIM: u32 = 1_024;

/// Where the expanded (or unexpanded) latents are consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlaTargetLayout {
    /// Materialize per-head K and V (the vLLM/SGLang "naive" MLA path).
    #[serde(rename = "naive-per-head")]
    NaivePerHead,
    /// Consume the latent directly with absorbed attention (MQA-style); no
    /// expansion is performed, so `expand_mla_latents` refuses this target.
    #[serde(rename = "absorbed-mqa")]
    AbsorbedMqa,
}

/// Canonical, authenticated description of the latent→per-head expansion.
/// The W_KVb weight itself is engine tensor data and never enters this
/// crate; its SHA-256 binds the exact matrix the expansion was qualified
/// against. Canonical W_KVb layout: row-major `[latent_dim, 2 * num_heads *
/// head_dim]`, columns `0..num_heads*head_dim` are the K up-projection
/// (W_UK, head `h` at columns `h*head_dim..`), columns `num_heads*head_dim..`
/// are the V up-projection (W_UV, same per-head stride).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlaExpansionDescriptor {
    pub schema_version: u32,
    pub w_kvb_sha256: String,
    pub rope_config_sha256: String,
    pub latent_dim: u32,
    pub rope_dim: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub target_layout: MlaTargetLayout,
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl MlaExpansionDescriptor {
    /// Fail-closed validation: schema version, bounded nonzero geometry, and
    /// well-formed hashes. Overflow in the derived W_KVb element count is an
    /// error, not a wrap.
    pub fn validate(&self) -> Result<(), StoreError> {
        if self.schema_version != MLA_EXPANSION_DESCRIPTOR_SCHEMA_V1 {
            return Err(StoreError::Expectation(
                "mla expansion descriptor schema version is unsupported",
            ));
        }
        if !valid_sha256_hex(&self.w_kvb_sha256) || !valid_sha256_hex(&self.rope_config_sha256) {
            return Err(StoreError::Expectation(
                "mla expansion descriptor hashes must be 64 lowercase hexadecimal digits",
            ));
        }
        if self.latent_dim == 0
            || self.latent_dim > MLA_MAX_LATENT_DIM
            || self.rope_dim == 0
            || self.rope_dim > MLA_MAX_ROPE_DIM
            || self.num_heads == 0
            || self.num_heads > MLA_MAX_HEADS
            || self.head_dim == 0
            || self.head_dim > MLA_MAX_HEAD_DIM
        {
            return Err(StoreError::Expectation(
                "mla expansion descriptor geometry is outside the bounded nonzero dims",
            ));
        }
        self.w_kvb_elements()?;
        Ok(())
    }

    /// Elements of the canonical W_KVb matrix: `latent_dim × 2·num_heads·head_dim`.
    pub fn w_kvb_elements(&self) -> Result<u64, StoreError> {
        u64::from(self.latent_dim)
            .checked_mul(2)
            .and_then(|value| value.checked_mul(u64::from(self.num_heads)))
            .and_then(|value| value.checked_mul(u64::from(self.head_dim)))
            .ok_or(StoreError::Expectation(
                "mla expansion descriptor W_KVb element count overflows",
            ))
    }

    /// Elements of one per-token latent record: `latent_dim + rope_dim`.
    pub fn record_elements(&self) -> u64 {
        u64::from(self.latent_dim) + u64::from(self.rope_dim)
    }
}

/// Parse and validate an expansion descriptor from JSON. Unknown fields are
/// denied (`deny_unknown_fields`); a parse or validation failure is an error,
/// never a default.
pub fn parse_mla_expansion_descriptor(json: &str) -> Result<MlaExpansionDescriptor, StoreError> {
    let descriptor: MlaExpansionDescriptor = serde_json::from_str(json)
        .map_err(|_| StoreError::Expectation("mla expansion descriptor is not valid json"))?;
    descriptor.validate()?;
    Ok(descriptor)
}

/// The expanded naive-per-head KV of a batch of latent records.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaExpandedKv {
    /// Per-head K without the rope component: `[token][head][head_dim]`.
    pub k_nope: Vec<f32>,
    /// Per-head V: `[token][head][head_dim]`.
    pub v: Vec<f32>,
    /// Rope-path keys passed through unchanged: `[token][rope_dim]`.
    pub k_rope: Vec<f32>,
}

/// Expand latent records into naive-per-head K/V: `c_KV @ W_KVb` with the
/// rope path (`k_rope`) passed through untouched — rope is applied upstream
/// of the cache, so the stored `k_rope` is already in its final form.
/// `latents` is the packed fp32 record stream `[token][latent_dim + rope_dim]`.
/// Pure function, no dependencies; every dimension mismatch is an error.
pub fn expand_mla_latents(
    descriptor: &MlaExpansionDescriptor,
    w_kvb: &[f32],
    latents: &[f32],
) -> Result<MlaExpandedKv, StoreError> {
    descriptor.validate()?;
    if descriptor.target_layout != MlaTargetLayout::NaivePerHead {
        return Err(StoreError::Expectation(
            "mla expansion target absorbed-mqa consumes latents without expansion",
        ));
    }
    let latent_dim = descriptor.latent_dim as usize;
    let rope_dim = descriptor.rope_dim as usize;
    let num_heads = descriptor.num_heads as usize;
    let head_dim = descriptor.head_dim as usize;
    let record = latent_dim + rope_dim;
    if latents.is_empty() || latents.len() % record != 0 {
        return Err(StoreError::Expectation(
            "mla latent stream is empty or not a whole number of per-token records",
        ));
    }
    let tokens = latents.len() / record;
    let expected_w_kvb = usize::try_from(descriptor.w_kvb_elements()?).map_err(|_| {
        StoreError::Expectation("mla W_KVb element count exceeds the addressable bounds")
    })?;
    if w_kvb.len() != expected_w_kvb {
        return Err(StoreError::Expectation(
            "mla W_KVb element count does not match the descriptor geometry",
        ));
    }
    let kv_cols = num_heads * head_dim;
    let mut k_nope = vec![0.0f32; tokens * kv_cols];
    let mut v = vec![0.0f32; tokens * kv_cols];
    let mut k_rope = vec![0.0f32; tokens * rope_dim];
    for token in 0..tokens {
        let record_base = token * record;
        let c_kv = &latents[record_base..record_base + latent_dim];
        k_rope[token * rope_dim..(token + 1) * rope_dim]
            .copy_from_slice(&latents[record_base + latent_dim..record_base + record]);
        for head in 0..num_heads {
            for dim in 0..head_dim {
                let column = head * head_dim + dim;
                let mut k_sum = 0.0f32;
                let mut v_sum = 0.0f32;
                for (row, &c) in c_kv.iter().enumerate() {
                    let weight_base = row * 2 * kv_cols;
                    k_sum += c * w_kvb[weight_base + column];
                    v_sum += c * w_kvb[weight_base + kv_cols + column];
                }
                k_nope[token * kv_cols + column] = k_sum;
                v[token * kv_cols + column] = v_sum;
            }
        }
    }
    Ok(MlaExpandedKv { k_nope, v, k_rope })
}

fn mla_metadata_u64(metadata: &GgufMetadata, key: &str) -> Result<Option<u64>, StoreError> {
    match metadata.get(key) {
        None => Ok(None),
        Some(value) => {
            let integer = match value {
                GgufValue::Uint8(value) => Some(u64::from(*value)),
                GgufValue::Int8(value) => u64::try_from(*value).ok(),
                GgufValue::Uint16(value) => Some(u64::from(*value)),
                GgufValue::Int16(value) => u64::try_from(*value).ok(),
                GgufValue::Uint32(value) => Some(u64::from(*value)),
                GgufValue::Int32(value) => u64::try_from(*value).ok(),
                GgufValue::Uint64(value) => Some(*value),
                GgufValue::Int64(value) => u64::try_from(*value).ok(),
                _ => None,
            };
            integer.map(Some).ok_or(StoreError::Expectation(
                "gguf mla layout underivable: a geometry key is not an integer",
            ))
        }
    }
}

/// Derive the `mla-latent` v2 layout from GGUF metadata carrying the
/// MLA/deepseek-style geometry keys (`{arch}.attention.kv_lora_rank`,
/// `{arch}.attention.qk_rope_head_dim`). A GGUF without those keys is not an
/// MLA model and is refused; the non-MLA derivation allowlist is untouched.
/// The class encodes the packed per-token record: `kv_heads = 1`,
/// `head_dim = kv_lora_rank + qk_rope_head_dim`, `window_tokens = 0`.
pub fn derive_mla_layout_from_metadata(
    metadata: &GgufMetadata,
) -> Result<OwnedLayoutV2, StoreError> {
    let architecture = metadata
        .get("general.architecture")
        .and_then(|value| match value {
            GgufValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or(StoreError::Expectation(
            "gguf mla layout underivable: general.architecture is missing",
        ))?;
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let latent_dim = mla_metadata_u64(metadata, &key("attention.kv_lora_rank"))?.ok_or(
        StoreError::Expectation(
            "gguf mla layout underivable: not an MLA model (attention.kv_lora_rank is missing)",
        ),
    )?;
    let rope_dim =
        mla_metadata_u64(metadata, &key("attention.qk_rope_head_dim"))?
            .ok_or(StoreError::Expectation(
            "gguf mla layout underivable: not an MLA model (attention.qk_rope_head_dim is missing)",
        ))?;
    let num_layers = mla_metadata_u64(metadata, &key("block_count"))?.ok_or(
        StoreError::Expectation("gguf mla layout underivable: block_count is missing"),
    )?;
    let record = latent_dim
        .checked_add(rope_dim)
        .ok_or(StoreError::Expectation(
            "gguf mla layout underivable: latent record dim overflows",
        ))?;
    if latent_dim == 0
        || latent_dim > u64::from(MLA_MAX_LATENT_DIM)
        || rope_dim == 0
        || rope_dim > u64::from(MLA_MAX_ROPE_DIM)
        || num_layers == 0
        || num_layers > u64::from(u32::MAX)
        || record > u64::from(u32::MAX)
    {
        return Err(StoreError::Expectation(
            "gguf mla layout underivable: geometry is outside the v2 bounds",
        ));
    }
    let name = metadata
        .get("general.name")
        .and_then(|value| match value {
            GgufValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .unwrap_or(architecture)
        .to_string();
    // The rotary applies to the k_rope sub-dimension of the packed record,
    // so an absent rope.dimension_count defaults to qk_rope_head_dim (not
    // the record width). Same fail-closed rope parsing as the uniform lane.
    let rope = crate::gguf_layout::rope_fields_from_gguf(metadata, architecture, record, rope_dim)?;
    Ok(OwnedLayoutV2 {
        name,
        num_layers: num_layers as u32,
        classes: vec![OwnedLayoutClassV2 {
            class: MLA_LATENT_LAYOUT_CLASS.to_string(),
            from: 0,
            until: num_layers as u32,
            step: 1,
            except: Vec::new(),
            kv_heads: 1,
            head_dim: record as u32,
            window_tokens: 0,
            rope_freq_base_bits: rope.freq_base_bits,
            rope_dimension_count: rope.dimension_count,
            rope_scaling: rope.scaling,
            rope_convention: rope.convention,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_descriptor() -> MlaExpansionDescriptor {
        MlaExpansionDescriptor {
            schema_version: MLA_EXPANSION_DESCRIPTOR_SCHEMA_V1,
            w_kvb_sha256: "a".repeat(64),
            rope_config_sha256: "b".repeat(64),
            latent_dim: 16,
            rope_dim: 4,
            num_heads: 2,
            head_dim: 8,
            target_layout: MlaTargetLayout::NaivePerHead,
        }
    }

    fn fixture_w_kvb(descriptor: &MlaExpansionDescriptor) -> Vec<f32> {
        let elements = descriptor.w_kvb_elements().unwrap() as usize;
        (0..elements)
            .map(|index| ((index * 7 + 3) % 17) as f32 * 0.125 - 1.0)
            .collect()
    }

    fn fixture_latents(descriptor: &MlaExpansionDescriptor, tokens: usize) -> Vec<f32> {
        let record = descriptor.record_elements() as usize;
        (0..tokens * record)
            .map(|index| ((index * 5 + 1) % 23) as f32 * 0.0625 - 0.5)
            .collect()
    }

    #[test]
    fn expansion_matches_an_independent_direct_computation() {
        let descriptor = fixture_descriptor();
        let w_kvb = fixture_w_kvb(&descriptor);
        let tokens = 3;
        let latents = fixture_latents(&descriptor, tokens);
        let expanded = expand_mla_latents(&descriptor, &w_kvb, &latents).unwrap();

        // Independent direct formula: per token, K/V are plain matvecs of
        // c_KV against the K and V column blocks of W_KVb; k_rope passes
        // through verbatim.
        let latent_dim = descriptor.latent_dim as usize;
        let rope_dim = descriptor.rope_dim as usize;
        let num_heads = descriptor.num_heads as usize;
        let head_dim = descriptor.head_dim as usize;
        let record = latent_dim + rope_dim;
        let kv_cols = num_heads * head_dim;
        let mut direct_k = vec![0.0f32; tokens * kv_cols];
        let mut direct_v = vec![0.0f32; tokens * kv_cols];
        for token in 0..tokens {
            let c_kv = &latents[token * record..token * record + latent_dim];
            for column in 0..kv_cols {
                for (row, &c) in c_kv.iter().enumerate() {
                    direct_k[token * kv_cols + column] += c * w_kvb[row * 2 * kv_cols + column];
                    direct_v[token * kv_cols + column] +=
                        c * w_kvb[row * 2 * kv_cols + kv_cols + column];
                }
            }
            for rope in 0..rope_dim {
                assert_eq!(
                    expanded.k_rope[token * rope_dim + rope],
                    latents[token * record + latent_dim + rope]
                );
            }
        }
        for (actual, expected) in expanded.k_nope.iter().zip(&direct_k) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "k: {actual} != {expected}"
            );
        }
        for (actual, expected) in expanded.v.iter().zip(&direct_v) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "v: {actual} != {expected}"
            );
        }
        // The expansion is not degenerate: c_KV actually moved the outputs.
        assert!(expanded.k_nope.iter().any(|value| value.abs() > 1e-3));
        assert!(expanded.v.iter().any(|value| value.abs() > 1e-3));
    }

    #[test]
    fn expansion_refuses_dimension_mismatches() {
        let descriptor = fixture_descriptor();
        let w_kvb = fixture_w_kvb(&descriptor);
        let latents = fixture_latents(&descriptor, 2);

        let error =
            expand_mla_latents(&descriptor, &w_kvb[..w_kvb.len() - 1], &latents).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla W_KVb element count does not match the descriptor geometry"
        );
        let error =
            expand_mla_latents(&descriptor, &w_kvb, &latents[..latents.len() - 1]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla latent stream is empty or not a whole number of per-token records"
        );
        let error = expand_mla_latents(&descriptor, &w_kvb, &[]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla latent stream is empty or not a whole number of per-token records"
        );
    }

    #[test]
    fn expansion_refuses_the_absorbed_target_and_bad_descriptors() {
        let descriptor = fixture_descriptor();
        let w_kvb = fixture_w_kvb(&descriptor);
        let latents = fixture_latents(&descriptor, 2);

        let mut absorbed = descriptor.clone();
        absorbed.target_layout = MlaTargetLayout::AbsorbedMqa;
        let error = expand_mla_latents(&absorbed, &w_kvb, &latents).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla expansion target absorbed-mqa consumes latents without expansion"
        );

        let mut zero_dim = descriptor.clone();
        zero_dim.latent_dim = 0;
        assert!(expand_mla_latents(&zero_dim, &w_kvb, &latents).is_err());
        let mut bad_hash = descriptor.clone();
        bad_hash.w_kvb_sha256 = "not-hex".into();
        assert!(expand_mla_latents(&bad_hash, &w_kvb, &latents).is_err());
        let mut bad_schema = descriptor.clone();
        bad_schema.schema_version = 2;
        assert!(expand_mla_latents(&bad_schema, &w_kvb, &latents).is_err());
    }

    #[test]
    fn descriptor_validation_matrix() {
        let descriptor = fixture_descriptor();
        descriptor.validate().unwrap();
        assert_eq!(descriptor.record_elements(), 20);
        assert_eq!(descriptor.w_kvb_elements().unwrap(), 16 * 2 * 2 * 8);

        let mutations: &[fn(&mut MlaExpansionDescriptor)] = &[
            |descriptor| descriptor.schema_version = 0,
            |descriptor| descriptor.schema_version = 2,
            |descriptor| descriptor.w_kvb_sha256 = "a".repeat(63),
            |descriptor| descriptor.w_kvb_sha256 = "A".repeat(64),
            |descriptor| descriptor.rope_config_sha256.clear(),
            |descriptor| descriptor.latent_dim = 0,
            |descriptor| descriptor.rope_dim = 0,
            |descriptor| descriptor.num_heads = 0,
            |descriptor| descriptor.head_dim = 0,
            |descriptor| descriptor.latent_dim = MLA_MAX_LATENT_DIM + 1,
            |descriptor| descriptor.rope_dim = MLA_MAX_ROPE_DIM + 1,
        ];
        for mutate in mutations {
            let mut changed = descriptor.clone();
            mutate(&mut changed);
            assert!(
                changed.validate().is_err(),
                "mutation accepted: {changed:?}"
            );
        }
    }

    #[test]
    fn descriptor_json_is_canonical_and_denies_unknown_fields() {
        let descriptor = fixture_descriptor();
        let json = serde_json::to_string(&descriptor).unwrap();
        let parsed = parse_mla_expansion_descriptor(&json).unwrap();
        assert_eq!(parsed, descriptor);

        let with_unknown = json.replace(
            "\"latent_dim\":16",
            "\"latent_dim\":16,\"speculative_field\":true",
        );
        assert!(with_unknown.contains("speculative_field"));
        let error = parse_mla_expansion_descriptor(&with_unknown).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla expansion descriptor is not valid json"
        );

        let error = parse_mla_expansion_descriptor("{ not json").unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla expansion descriptor is not valid json"
        );
        // Valid JSON, invalid descriptor: geometry validation still gates.
        let mut zero_dim = descriptor.clone();
        zero_dim.rope_dim = 0;
        let error =
            parse_mla_expansion_descriptor(&serde_json::to_string(&zero_dim).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "mla expansion descriptor geometry is outside the bounded nonzero dims"
        );
    }

    fn deepseek_style_metadata() -> GgufMetadata {
        let mut metadata = GgufMetadata::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "general.name".to_string(),
            GgufValue::String("deepseek-mla-fixture".to_string()),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::Uint32(3));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::Uint32(16),
        );
        metadata.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::Uint32(512),
        );
        metadata.insert(
            "deepseek2.attention.qk_rope_head_dim".to_string(),
            GgufValue::Uint32(64),
        );
        metadata.insert(
            "deepseek2.rope.freq_base".to_string(),
            GgufValue::Float32(10_000.0),
        );
        metadata
    }

    #[test]
    fn derives_the_mla_latent_class_from_deepseek_style_metadata() {
        let derived = derive_mla_layout_from_metadata(&deepseek_style_metadata()).unwrap();
        assert_eq!(derived.name, "deepseek-mla-fixture");
        assert_eq!(derived.num_layers, 3);
        assert_eq!(derived.classes.len(), 1);
        let class = &derived.classes[0];
        assert_eq!(class.class, MLA_LATENT_LAYOUT_CLASS);
        assert_eq!(class.from, 0);
        assert_eq!(class.until, 3);
        assert_eq!(class.step, 1);
        assert!(class.except.is_empty());
        assert_eq!(class.kv_heads, 1);
        assert_eq!(class.head_dim, 512 + 64);
        assert_eq!(class.window_tokens, 0);
        // The rotary width defaults to the k_rope sub-dimension, not the
        // packed record width.
        assert_eq!(class.rope_freq_base_bits, 10_000.0f64.to_bits());
        assert_eq!(class.rope_dimension_count, 64);
        assert_eq!(class.rope_scaling, "none");
        assert_eq!(
            class.rope_convention,
            crate::gguf_layout::RopeConvention::Neox
        );
    }

    #[test]
    fn refuses_mla_metadata_without_a_rope_freq_base() {
        let mut metadata = deepseek_style_metadata();
        metadata.remove("deepseek2.rope.freq_base");
        let error = derive_mla_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf rope config underivable: rope.freq_base is missing"
        );
    }

    #[test]
    fn refuses_gguf_without_the_mla_keys_as_not_an_mla_model() {
        // A plain GQA model (qwen2-style keys only) is not an MLA model.
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
        let error = derive_mla_layout_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf mla layout underivable: not an MLA model (attention.kv_lora_rank is missing)"
        );

        // Only one of the two MLA keys is still not an MLA model.
        let mut partial = deepseek_style_metadata();
        partial.remove("deepseek2.attention.qk_rope_head_dim");
        let error = derive_mla_layout_from_metadata(&partial).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf mla layout underivable: not an MLA model (attention.qk_rope_head_dim is missing)"
        );
    }

    #[test]
    fn refuses_mla_geometry_outside_the_bounds() {
        let mut zero_rank = deepseek_style_metadata();
        zero_rank.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::Uint32(0),
        );
        assert!(derive_mla_layout_from_metadata(&zero_rank).is_err());

        let mut zero_rope = deepseek_style_metadata();
        zero_rope.insert(
            "deepseek2.attention.qk_rope_head_dim".to_string(),
            GgufValue::Uint32(0),
        );
        assert!(derive_mla_layout_from_metadata(&zero_rope).is_err());

        let mut no_layers = deepseek_style_metadata();
        no_layers.remove("deepseek2.block_count");
        let error = derive_mla_layout_from_metadata(&no_layers).unwrap_err();
        assert_eq!(
            error.to_string(),
            "gguf mla layout underivable: block_count is missing"
        );

        let mut oversized = deepseek_style_metadata();
        oversized.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::Uint32(MLA_MAX_LATENT_DIM + 1),
        );
        assert!(derive_mla_layout_from_metadata(&oversized).is_err());
    }
}
