use serde::{Deserialize, Serialize};

use crate::{HandoffError, Result};

mod validation;
pub(crate) use validation::validate_prerope_key_plane_finite;
pub use validation::{
    artifact_hmac_sha256, artifact_mac_stream, artifact_sha256, descriptor_chain_sha256,
};

pub const LIVE_HANDOFF_SCHEMA_V1: u32 = 1;
pub const LIVE_HANDOFF_PROTOCOL_V1: &str = "kvpack-live-f16-le-v1";
pub const PORTABLE_KV_ABI_V1: &str = "canonical-kv-f16-le-v1";
pub const PORTABLE_KV_ABI_V2: &str = "canonical-kv-v2";
/// v2 pre-RoPE capture representation family
/// (docs/PREROPE_CAPTURE_CONTRACT.md in the kvpack repo): Key planes cross
/// as pre-RoPE post-bias f32-LE and the consumer rotates once at install
/// inside its own pinned rotation kernel; Value planes stay f16-LE exactly
/// as in the post-RoPE family. Consumers that do not know this label keep
/// rejecting it here — the unknown-label case stays fail-closed.
pub const PORTABLE_KV_ABI_V2_PREROPE: &str = "canonical-kv-prerope-v2";
/// v2 wire schedules (`BeginManifestV1::schedule`). `layer-order` is the
/// default declared-order walk; `decode-priority` streams windowed
/// classes (the newest cuts) before full-history classes.
pub const WIRE_SCHEDULE_LAYER_ORDER: &str = "layer-order";
pub const WIRE_SCHEDULE_DECODE_PRIORITY: &str = "decode-priority";
pub const FRAME_MAGIC: [u8; 4] = *b"KVHF";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffStrategyV1 {
    ConsumerLastPromptToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRoleV1 {
    Key,
    Value,
}

impl TensorRoleV1 {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Key => "k",
            Self::Value => "v",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointIdentityV1 {
    pub consumer_engine_abi: String,
    pub consumer_node: String,
    pub producer_engine_abi: String,
    pub producer_node: String,
    pub trust_domain: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactIdentityV1 {
    pub adapter_sha256: String,
    pub chat_template_sha256: String,
    pub context_policy_sha256: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub tokenizer_revision: String,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryV1 {
    pub head_dim: u32,
    pub max_context_tokens: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionV1 {
    pub compute: String,
    pub kv: String,
    pub weights: String,
}

/// v2: one named layer class in a begin's layout table. Layers are
/// `from..until` stepped by `step`, minus `except` — compact enough for
/// uniform models (one class) and for interleaved schedules (Gemma's
/// every-6th-layer full attention). A class with `window_tokens > 0`
/// ships only the trailing in-window tokens of each plane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutClassV2 {
    pub class: String,
    pub dtype: String,
    pub except: Vec<u32>,
    pub from: u32,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub roles: Vec<TensorRoleV1>,
    pub step: u32,
    pub until: u32,
    pub window_tokens: u32,
}

impl LayoutClassV2 {
    /// Layers covered by this class, ascending, range minus `except`.
    ///
    /// Checked count-then-collect: `step == 0` or an empty/inverted range
    /// yields no layers (validation rejects both), and the range length is
    /// computed before anything is collected. This is still only bounded
    /// once `until <= geometry.num_layers` has been proven —
    /// `validate_layout_table` enforces that bound BEFORE any walk
    /// materializes a class's layers, so a hostile begin with
    /// `until: u32::MAX` is rejected without allocating the range it
    /// claims (F1 pre-validation OOM).
    pub fn layers(&self) -> Vec<u32> {
        if self.step == 0 || self.from >= self.until {
            return Vec::new();
        }
        let count = (self.until - self.from).div_ceil(self.step) as usize;
        let mut layers = Vec::with_capacity(count);
        layers.extend(
            (self.from..self.until)
                .step_by(self.step as usize)
                .filter(|layer| !self.except.contains(layer)),
        );
        layers
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeginManifestV1 {
    pub cached_token_count: u32,
    pub created_unix_ms: u64,
    pub deadline_unix_ms: u64,
    pub endpoints: EndpointIdentityV1,
    pub expected_layer_frames: u32,
    pub expected_payload_bytes: u64,
    pub geometry: GeometryV1,
    pub identity: ExactIdentityV1,
    pub portable_abi: String,
    pub precision: PrecisionV1,
    pub protocol: String,
    pub schema_version: u32,
    pub strategy: HandoffStrategyV1,
    pub token_ids_sha256: String,
    pub transfer_id: String,
    /// v2: per-class layout table. Empty means v1 semantics exactly —
    /// v1 begin bytes stay valid and hash-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layout_table: Vec<LayoutClassV2>,
    /// v2: wire schedule variant. Absent means `layer-order` (the
    /// declared-order walk) exactly — v2 begin bytes without the field
    /// stay valid and hash-identical. `decode-priority` streams windowed
    /// classes (newest cuts) before full-history classes; derivation is
    /// deterministic from the layout table at both ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    /// F1: identifier of the tenant MAC key that authenticated this
    /// artifact (`artifact_hmac_sha256` on the seal). Absent means
    /// integrity-only and keeps begin bytes hash-identical; a consumer
    /// armed with a [`crate::MacKey`] requires it. The id is advisory (it names
    /// the key the producer used); the tag's validity under the armed key
    /// is what authenticates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hmac_key_id: Option<String>,
}

/// The (class, layer, role) expected at one sequence position when
/// walking a begin's layout table in declared order.
pub struct ExpectedPlane<'a> {
    pub class: &'a LayoutClassV2,
    pub layer: u32,
    pub role: TensorRoleV1,
    pub range_start: u32,
    pub range_end: u32,
}

impl BeginManifestV1 {
    pub(crate) fn is_v2(&self) -> bool {
        !self.layout_table.is_empty()
    }

    /// v2 pre-RoPE capture family (`canonical-kv-prerope-v2`): Key planes
    /// are f32-LE pre-RoPE post-bias content; Value planes are f16-LE
    /// exactly as in the post-RoPE family.
    pub fn is_prerope_v2(&self) -> bool {
        self.is_v2() && self.portable_abi == PORTABLE_KV_ABI_V2_PREROPE
    }

    /// Element width in bytes of one role's planes under this begin's ABI:
    /// 4 for pre-RoPE Key planes, 2 (float16) for everything else.
    fn plane_element_bytes_v2(&self, role: TensorRoleV1) -> u64 {
        if self.is_prerope_v2() && role == TensorRoleV1::Key {
            4
        } else {
            2
        }
    }

    /// The dtype tag a plane frame must carry for one role under this
    /// begin's ABI. An absent frame tag keeps meaning the class dtype
    /// ("float16"), so pre-RoPE Key frames must be tagged `float32`
    /// explicitly.
    fn expected_plane_dtype_v2<'a>(&self, class: &'a LayoutClassV2, role: TensorRoleV1) -> &'a str {
        if self.is_prerope_v2() && role == TensorRoleV1::Key {
            "float32"
        } else {
            class.dtype.as_str()
        }
    }

    /// Payload file extension for one role's staged plane inside a sealed
    /// bundle: `f32le` for pre-RoPE Key planes, `f16le` otherwise.
    pub(crate) fn plane_payload_extension(&self, role: TensorRoleV1) -> &'static str {
        if self.is_prerope_v2() && role == TensorRoleV1::Key {
            "f32le"
        } else {
            "f16le"
        }
    }

    /// Flat v2 layout walk: the (class index, layer, role) expected at
    /// every sequence position — classes in schedule order (declared
    /// order for `layer-order`/absent, windowed-first for
    /// `decode-priority`), layers ascending within a class, roles in each
    /// class's declared order. Validation sessions build it once and then
    /// index it in O(1) instead of re-walking the table (and rebuilding
    /// per-class layer vectors) for every plane.
    pub(crate) fn layout_walk_v2(&self) -> Vec<(u32, u32, TensorRoleV1)> {
        let mut walk = Vec::new();
        for class_idx in self.schedule_class_order() {
            let class = &self.layout_table[class_idx];
            for layer in class.layers() {
                for role in &class.roles {
                    walk.push((class_idx as u32, layer, *role));
                }
            }
        }
        walk
    }

    /// Deterministic class traversal order, derived from the schedule
    /// field and the layout table only. `decode-priority` is a stable
    /// partition: windowed classes (`window_tokens > 0`, i.e. the classes
    /// streaming the newest cuts) first, full-history classes after, each
    /// group keeping its declared relative order. Anything else
    /// (including an absent schedule) is the declared order.
    fn schedule_class_order(&self) -> Vec<usize> {
        let order: Vec<usize> = (0..self.layout_table.len()).collect();
        match self.schedule.as_deref() {
            Some(WIRE_SCHEDULE_DECODE_PRIORITY) => {
                let mut windowed: Vec<usize> = Vec::new();
                let mut full: Vec<usize> = Vec::new();
                for idx in order {
                    if self.layout_table[idx].window_tokens > 0 {
                        windowed.push(idx);
                    } else {
                        full.push(idx);
                    }
                }
                windowed.extend(full);
                windowed
            }
            _ => order,
        }
    }

    /// Resolve one precomputed walk entry to its full expectation: class
    /// membership plus the canonical token range.
    pub(crate) fn expected_from_walk_entry(
        &self,
        entry: (u32, u32, TensorRoleV1),
    ) -> ExpectedPlane<'_> {
        let (class_idx, layer, role) = entry;
        let class = &self.layout_table[class_idx as usize];
        let window = if class.window_tokens == 0 {
            self.cached_token_count
        } else {
            class.window_tokens.min(self.cached_token_count)
        };
        ExpectedPlane {
            class,
            layer,
            role,
            range_start: self.cached_token_count.saturating_sub(window),
            range_end: self.cached_token_count,
        }
    }

    /// Expected (class, layer, role, range) at `sequence` in the v2
    /// walk. Single-plane callers build the walk here; per-plane hot
    /// paths precompute [`BeginManifestV1::layout_walk_v2`] once and
    /// resolve entries through [`BeginManifestV1::expected_from_walk_entry`].
    pub(crate) fn expected_plane_at(&self, sequence: u32) -> Result<ExpectedPlane<'_>> {
        let entry = self
            .layout_walk_v2()
            .get(sequence as usize)
            .copied()
            .ok_or_else(|| {
                HandoffError::Validation(format!(
                    "layer frame {sequence} is outside the declared layout table"
                ))
            })?;
        Ok(self.expected_from_walk_entry(entry))
    }

    fn v2_frame_counts(&self) -> Result<(u32, u64)> {
        let mut frames = 0u64;
        let mut bytes = 0u64;
        for class in &self.layout_table {
            let window = if class.window_tokens == 0 {
                self.cached_token_count
            } else {
                class.window_tokens.min(self.cached_token_count)
            };
            let layers = class.layers().len() as u64;
            // Per-role accounting: under the pre-RoPE ABI the Key planes
            // are f32 (4-byte elements), Value planes stay f16.
            for role in &class.roles {
                let per_plane = u64::from(window)
                    .checked_mul(u64::from(class.kv_heads))
                    .and_then(|value| value.checked_mul(u64::from(class.head_dim)))
                    .and_then(|value| value.checked_mul(self.plane_element_bytes_v2(*role)))
                    .ok_or_else(|| {
                        HandoffError::Validation("declared payload size overflows u64".into())
                    })?;
                frames = frames
                    .checked_add(layers)
                    .ok_or_else(|| HandoffError::Validation("layer-frame count overflow".into()))?;
                bytes = bytes
                    .checked_add(layers.checked_mul(per_plane).ok_or_else(|| {
                        HandoffError::Validation("declared payload size overflows u64".into())
                    })?)
                    .ok_or_else(|| {
                        HandoffError::Validation("declared payload size overflows u64".into())
                    })?;
            }
        }
        let frames = u32::try_from(frames)
            .map_err(|_| HandoffError::Validation("layer-frame count overflow".into()))?;
        Ok((frames, bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerHeaderV1 {
    pub byte_length: u64,
    pub layer: u32,
    pub logical_token_end: u32,
    pub logical_token_start: u32,
    pub role: TensorRoleV1,
    pub schema_version: u32,
    pub sequence: u32,
    pub sha256: String,
    pub shape: [u32; 3],
    pub transfer_id: String,
    /// v2: plane dtype tag. Absent means "float16" (v1 bytes stay valid
    /// and descriptor-chain hashes identical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    /// v2: layout class this plane belongs to (must exist in the
    /// begin's layout table).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_class: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealCoreV1 {
    pub completed_unix_ms: u64,
    pub descriptor_chain_sha256: String,
    pub frame_count: u32,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub protocol: String,
    pub schema_version: u32,
    pub strategy: HandoffStrategyV1,
    pub token_ids_sha256: String,
    pub transfer_id: String,
    /// F2: mandatory authenticated canary for pre-RoPE begins; absent for
    /// every other family. The producer records a digest of the
    /// consumer-equivalent post-RoPE rows for a fixed position sample
    /// (docs/PREROPE_CAPTURE_CONTRACT.md §6); the consumer recomputes those
    /// rows with its pinned rotation kernel and checks them through
    /// [`CanaryRecord::verify_against`] before committing the install. The
    /// handoff layer authenticates the record and requires its presence
    /// for a pre-RoPE begin; the numerical gate itself is the engine's.
    /// Absent means "no canary" and keeps v1/core bytes hash-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canary: Option<CanaryRecord>,
}

/// F2 (red-team 2026-08-09): the per-artifact canary a pre-RoPE bundle
/// must carry. A residual-carrying (1-2 f16-ulp K) restore must never be
/// committed unconditionally: the producer stamps the post-RoPE K/V row
/// digests for a fixed position sample, and the consumer compares the
/// digests its own pinned rotation kernel produces. See
/// docs/PREROPE_CAPTURE_CONTRACT.md §6 (the contract claims the family is
/// canary-gated; this record plus [`BeginManifestV1`] presence
/// enforcement make that claim true at the data layer).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryRecord {
    /// Inclusive start of the sampled token window (absolute position).
    pub sample_token_start: u32,
    /// Number of token positions in the sample window (`> 0`).
    pub sample_token_count: u32,
    /// SHA-256 of the canonical f16-LE post-RoPE K rows the consumer's
    /// pinned kernel produces for the sample window, over every layer in
    /// declared order (domain-separated by the producer).
    pub post_rope_k_sha256: String,
    /// SHA-256 of the canonical f16-LE V rows for the sample window.
    pub post_rope_v_sha256: String,
}

impl CanaryRecord {
    /// F2 engine gate: the consumer passes the post-RoPE K/V row digests
    /// its pinned rotation kernel produced for the recorded sample window;
    /// any mismatch fails closed (the residual exceeded the recorded
    /// canary, or the producer's kernel diverged). Run this before
    /// committing the install.
    pub fn verify_against(&self, produced_k_sha256: &str, produced_v_sha256: &str) -> Result<()> {
        if self.post_rope_k_sha256 != produced_k_sha256
            || self.post_rope_v_sha256 != produced_v_sha256
        {
            return Err(HandoffError::Validation(
                "pre-RoPE canary does not match the consumer's pinned-kernel rows".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealManifestV1 {
    pub artifact_sha256: String,
    /// F1: optional keyed HMAC-SHA256 over the same stream as
    /// `artifact_sha256` (domain-separated via [`artifact_mac_stream`]).
    /// Absent means integrity-only and keeps seal bytes hash-identical;
    /// a consumer armed with a [`crate::MacKey`] requires it and verifies it
    /// through [`SealManifestV1::authenticate_hmac`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_hmac_sha256: Option<String>,
    #[serde(flatten)]
    pub core: SealCoreV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AckManifestV1 {
    pub artifact_sha256: String,
    pub protocol: String,
    pub schema_version: u32,
    pub status: String,
    pub transfer_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbortManifestV1 {
    pub code: String,
    pub protocol: String,
    pub schema_version: u32,
    pub transfer_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationLimits {
    pub max_cached_tokens: u32,
    pub max_clock_skew_ms: u64,
    pub max_context_tokens: u32,
    pub max_frame_bytes: u64,
    pub max_layers: u32,
    pub max_session_ms: u64,
    pub max_total_bytes: u64,
    pub now_unix_ms: u64,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_cached_tokens: 32_767,
            max_clock_skew_ms: 30_000,
            max_context_tokens: 32_768,
            max_frame_bytes: 64 * 1024 * 1024,
            max_layers: 64,
            max_session_ms: 15 * 60 * 1000,
            max_total_bytes: 4 * 1024 * 1024 * 1024,
            now_unix_ms: 1,
        }
    }
}
