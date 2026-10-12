use sha2::{Digest, Sha256};

use super::{
    AckManifestV1, BeginManifestV1, ExpectedPlane, HandoffStrategyV1, LayerHeaderV1, SealCoreV1,
    SealManifestV1, TensorRoleV1, ValidationLimits, LIVE_HANDOFF_PROTOCOL_V1,
    LIVE_HANDOFF_SCHEMA_V1, PORTABLE_KV_ABI_V1, PORTABLE_KV_ABI_V2, PORTABLE_KV_ABI_V2_PREROPE,
    WIRE_SCHEDULE_DECODE_PRIORITY, WIRE_SCHEDULE_LAYER_ORDER,
};
use crate::{canonical_json, token_ids_sha256, HandoffError, MacKey, Result};

fn checked_payload_bytes(begin: &BeginManifestV1) -> Result<u64> {
    u64::from(begin.geometry.num_layers)
        .checked_mul(2)
        .and_then(|value| value.checked_mul(u64::from(begin.cached_token_count)))
        .and_then(|value| value.checked_mul(u64::from(begin.geometry.num_kv_heads)))
        .and_then(|value| value.checked_mul(u64::from(begin.geometry.head_dim)))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| HandoffError::Validation("declared payload size overflows u64".into()))
}

fn valid_label(value: &str, max_len: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !matches!(byte, b'\\' | b'/' | b'\"'))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// F4 (red-team 2026-08-09): a pre-RoPE Key plane is f32-LE content the
/// consumer rotates into its f16 cache. SHA-256 authenticates bytes, not
/// values, so a producer bug or an in-plane bit-flip inside an otherwise
/// authenticated plane can push a NaN/Inf silently into the cache. Every
/// f32 element of such a plane must therefore be finite — this is the
/// fail-closed value gate run after the descriptor/hash check on every
/// pre-RoPE Key plane.
pub(crate) fn validate_prerope_key_plane_finite(payload: &[u8]) -> Result<()> {
    if payload.len() % 4 != 0 {
        return Err(HandoffError::Validation(format!(
            "pre-RoPE Key plane length {} is not a multiple of the f32 element width",
            payload.len()
        )));
    }
    for element in payload.chunks_exact(4) {
        let value = f32::from_le_bytes(element.try_into().expect("chunks_exact(4) yields 4 bytes"));
        if !value.is_finite() {
            return Err(HandoffError::Validation(
                "pre-RoPE Key plane contains a non-finite (NaN/Inf) f32 element".into(),
            ));
        }
    }
    Ok(())
}

impl BeginManifestV1 {
    pub fn validate(&self, limits: &ValidationLimits) -> Result<()> {
        // `portable_abi` is a closed set per table shape: v1 semantics for
        // an empty layout table, the post-RoPE or pre-RoPE v2 label for a
        // declared one. Anything else — including a label this build does
        // not know — fails closed here, before any payload moves.
        let abi_ok = if self.is_v2() {
            matches!(
                self.portable_abi.as_str(),
                PORTABLE_KV_ABI_V2 | PORTABLE_KV_ABI_V2_PREROPE
            )
        } else {
            self.portable_abi == PORTABLE_KV_ABI_V1
        };
        if self.schema_version != LIVE_HANDOFF_SCHEMA_V1
            || self.protocol != LIVE_HANDOFF_PROTOCOL_V1
            || !abi_ok
            || self.strategy != HandoffStrategyV1::ConsumerLastPromptToken
        {
            return Err(HandoffError::Validation(
                "unsupported live handoff schema, protocol, ABI, or strategy".into(),
            ));
        }
        if self.cached_token_count == 0
            || self.cached_token_count > limits.max_cached_tokens
            || self.geometry.max_context_tokens > limits.max_context_tokens
            || self.cached_token_count >= self.geometry.max_context_tokens
            || self.geometry.num_layers == 0
            || self.geometry.num_layers > limits.max_layers
        {
            return Err(HandoffError::Validation(
                "geometry or cached-token count is outside configured bounds".into(),
            ));
        }
        let (expected_frames, expected_bytes) = if self.is_v2() {
            self.validate_layout_table(limits)?;
            self.validate_schedule()?;
            self.v2_frame_counts()?
        } else {
            if self.schedule.is_some() {
                return Err(HandoffError::Validation(
                    "wire schedule requires a v2 layout table".into(),
                ));
            }
            if self.geometry.num_kv_heads == 0 || self.geometry.head_dim == 0 {
                return Err(HandoffError::Validation(
                    "geometry or cached-token count is outside configured bounds".into(),
                ));
            }
            let frames = self
                .geometry
                .num_layers
                .checked_mul(2)
                .ok_or_else(|| HandoffError::Validation("layer-frame count overflow".into()))?;
            (frames, checked_payload_bytes(self)?)
        };
        let one_frame = expected_bytes / u64::from(expected_frames);
        if self.expected_layer_frames != expected_frames
            || self.expected_payload_bytes != expected_bytes
            || expected_bytes > limits.max_total_bytes
            || one_frame > limits.max_frame_bytes
        {
            return Err(HandoffError::Validation(
                "declared frame or payload bounds do not match geometry".into(),
            ));
        }
        if self.precision.compute != "float16"
            || self.precision.kv != "float16"
            // Closed weights set: q4_k_m/q4_k_xl (GGUF lanes), nvfp4 (modelopt FP4
            // producer + deterministic F16-dequant consumer, B12), mxfp4
            // (gpt-oss producer; consumer runs a matched artifact, survey),
            // bf16 (byte-matched BF16 GGUF both ends, Gemma 4 31B lane).
            // The receiver still exact-matches the label against its
            // configured expectation, so an unexpected label stays fail-closed.
            || !matches!(self.precision.weights.as_str(), "q4_k_m" | "q4_k_xl" | "nvfp4" | "mxfp4" | "bf16")
        {
            return Err(HandoffError::Validation(
                "v1 requires q4_k_m, q4_k_xl, nvfp4, mxfp4, or bf16 weights with float16 compute and KV"
                    .into(),
            ));
        }
        let strings = [
            self.endpoints.consumer_engine_abi.as_str(),
            self.endpoints.consumer_node.as_str(),
            self.endpoints.producer_engine_abi.as_str(),
            self.endpoints.producer_node.as_str(),
            self.endpoints.trust_domain.as_str(),
            self.identity.model_revision.as_str(),
            self.identity.tokenizer_revision.as_str(),
        ];
        if strings.iter().any(|value| !valid_label(value, 256))
            || !valid_sha256(&self.transfer_id)
            || !valid_sha256(&self.token_ids_sha256)
            || !valid_sha256(&self.identity.adapter_sha256)
            || !valid_sha256(&self.identity.chat_template_sha256)
            || !valid_sha256(&self.identity.context_policy_sha256)
            || !valid_sha256(&self.identity.model_sha256)
            || !valid_sha256(&self.identity.tokenizer_sha256)
            || self
                .hmac_key_id
                .as_ref()
                .is_some_and(|id| !valid_label(id, 64))
        {
            return Err(HandoffError::Validation(
                "identity contains an invalid label or SHA-256".into(),
            ));
        }
        let latest_created = limits.now_unix_ms.saturating_add(limits.max_clock_skew_ms);
        let latest_deadline = self
            .created_unix_ms
            .checked_add(limits.max_session_ms)
            .ok_or_else(|| HandoffError::Validation("session deadline overflow".into()))?;
        if self.created_unix_ms == 0
            || self.created_unix_ms > latest_created
            || self.deadline_unix_ms <= self.created_unix_ms
            || self.deadline_unix_ms > latest_deadline
            || limits.now_unix_ms > self.deadline_unix_ms
        {
            return Err(HandoffError::Validation(
                "handoff creation/deadline is invalid or expired".into(),
            ));
        }
        Ok(())
    }

    pub fn expected_plane_bytes(&self) -> Result<u64> {
        checked_payload_bytes(self).map(|bytes| bytes / u64::from(self.expected_layer_frames))
    }

    /// v2: the wire schedule is a closed set; an unknown value fails
    /// closed at validation (never a guessed walk). Absent is the
    /// `layer-order` default.
    fn validate_schedule(&self) -> Result<()> {
        match self.schedule.as_deref() {
            None | Some(WIRE_SCHEDULE_LAYER_ORDER | WIRE_SCHEDULE_DECODE_PRIORITY) => Ok(()),
            Some(_) => Err(HandoffError::Validation(
                "layout table declares an unknown wire schedule".into(),
            )),
        }
    }

    /// v2: the layout table must be internally consistent and agree with
    /// the flat geometry block (which v1 peers and tooling still read).
    fn validate_layout_table(&self, limits: &ValidationLimits) -> Result<()> {
        let mut covered: Vec<u32> = Vec::new();
        for class in &self.layout_table {
            // Bounds BEFORE any layer materialization: a hostile
            // `until: u32::MAX` is rejected here without allocating the
            // range it claims (F1 pre-validation OOM). `geometry.num_layers`
            // is already proven `1..=max_layers` by the geometry check in
            // `validate`.
            if class.step == 0
                || class.from >= class.until
                || class.until > self.geometry.num_layers
            {
                return Err(HandoffError::Validation(
                    "layout table contains an invalid layer class".into(),
                ));
            }
            // Per-class per-plane byte bound at arm: the largest single
            // frame a class can emit must fit the frame cap. Checking only
            // the cross-class average let mixed-class layouts (Gemma-style
            // windowed + full classes) pass arm and die mid-stream on the
            // first oversized plane.
            let window = if class.window_tokens == 0 {
                self.cached_token_count
            } else {
                class.window_tokens.min(self.cached_token_count)
            };
            // The widest role's plane bounds the class (pre-RoPE ABI: Key
            // planes are f32, double the f16 width).
            let max_element_bytes = class
                .roles
                .iter()
                .map(|role| self.plane_element_bytes_v2(*role))
                .max()
                .unwrap_or(2);
            let per_plane = u64::from(window)
                .checked_mul(u64::from(class.kv_heads))
                .and_then(|value| value.checked_mul(u64::from(class.head_dim)))
                .and_then(|value| value.checked_mul(max_element_bytes))
                .ok_or_else(|| {
                    HandoffError::Validation("declared payload size overflows u64".into())
                })?;
            if per_plane > limits.max_frame_bytes {
                return Err(HandoffError::Validation(
                    "layout table class exceeds the per-frame byte bound".into(),
                ));
            }
            // `except` entries must live inside the class range and be
            // unique; anything else is silently dead wiring on the producer
            // walk, so it fails closed here.
            let mut except = class.except.clone();
            except.sort_unstable();
            except.dedup();
            if except.len() != class.except.len()
                || class
                    .except
                    .iter()
                    .any(|layer| *layer < class.from || *layer >= class.until)
            {
                return Err(HandoffError::Validation(
                    "layout table contains an out-of-range or duplicate except entry".into(),
                ));
            }
            // Roles are exactly [key, value] until the pair cursor
            // generalizes; any other order or arity passes validation only
            // to abort mid-stream. Sole exception: mla-latent classes, whose
            // single packed latent plane per layer declares roles [key].
            let roles_ok = class.roles == [TensorRoleV1::Key, TensorRoleV1::Value]
                || (class.class == "mla-latent" && class.roles == [TensorRoleV1::Key]);
            if !roles_ok {
                return Err(HandoffError::Validation(
                    "layout table classes must declare roles [key, value]".into(),
                ));
            }
            let layers = class.layers();
            if !valid_label(&class.class, 64)
                || layers.is_empty()
                || class.kv_heads == 0
                || class.head_dim == 0
                || class.dtype != "float16"
            {
                return Err(HandoffError::Validation(
                    "layout table contains an invalid layer class".into(),
                ));
            }
            covered.extend(layers);
        }
        let mut dedup = covered.clone();
        dedup.sort_unstable();
        dedup.dedup();
        if dedup.len() != covered.len() {
            return Err(HandoffError::Validation(
                "layout table assigns a layer to more than one class".into(),
            ));
        }
        // Coverage: the deduped union must be exactly 0..num_layers. A
        // holey table used to seal successfully, authenticating a cache
        // that is missing layers.
        if dedup.len() != self.geometry.num_layers as usize
            || dedup
                .iter()
                .enumerate()
                .any(|(index, layer)| *layer != index as u32)
        {
            return Err(HandoffError::Validation(
                "layout table must cover every declared layer exactly once".into(),
            ));
        }
        // Flat-geometry agreement: a single-class table must spell the
        // same heads/dim; a multi-class table must zero them (="see table").
        if self.layout_table.len() == 1 {
            let class = &self.layout_table[0];
            if self.geometry.num_kv_heads != class.kv_heads
                || self.geometry.head_dim != class.head_dim
            {
                return Err(HandoffError::Validation(
                    "flat geometry disagrees with the single-class layout table".into(),
                ));
            }
        } else if self.geometry.num_kv_heads != 0 || self.geometry.head_dim != 0 {
            return Err(HandoffError::Validation(
                "multi-class layout table requires zero flat kv_heads/head_dim".into(),
            ));
        }
        Ok(())
    }
}

impl LayerHeaderV1 {
    pub fn validate_for(&self, begin: &BeginManifestV1, next_sequence: u32) -> Result<()> {
        if begin.is_v2() {
            return self.validate_for_v2(begin, next_sequence);
        }
        let expected_layer = next_sequence / 2;
        let expected_role = if next_sequence % 2 == 0 {
            TensorRoleV1::Key
        } else {
            TensorRoleV1::Value
        };
        if self.schema_version != LIVE_HANDOFF_SCHEMA_V1
            || self.transfer_id != begin.transfer_id
            || self.sequence != next_sequence
            || self.layer != expected_layer
            || self.role != expected_role
            || self.logical_token_start != 0
            || self.logical_token_end != begin.cached_token_count
            || self.shape
                != [
                    begin.cached_token_count,
                    begin.geometry.num_kv_heads,
                    begin.geometry.head_dim,
                ]
            || self.byte_length != begin.expected_plane_bytes()?
            || !valid_sha256(&self.sha256)
            || self.dtype.is_some()
            || self.layout_class.is_some()
        {
            return Err(HandoffError::Validation(format!(
                "layer frame {next_sequence} does not match the ordered canonical tensor contract"
            )));
        }
        Ok(())
    }

    /// v2: validate against the begin's layout table — class membership,
    /// declared order, per-class shape, token range, dtype tag.
    fn validate_for_v2(&self, begin: &BeginManifestV1, next_sequence: u32) -> Result<()> {
        let expected = begin.expected_plane_at(next_sequence)?;
        self.validate_for_v2_expected(begin, next_sequence, &expected)
    }

    /// v2 check against an already-resolved layout expectation. The
    /// incremental verifier resolves expectations from the flat walk it
    /// precomputed when the begin was validated.
    pub(crate) fn validate_for_v2_expected(
        &self,
        begin: &BeginManifestV1,
        next_sequence: u32,
        expected: &ExpectedPlane<'_>,
    ) -> Result<()> {
        let range_len = self
            .logical_token_end
            .checked_sub(self.logical_token_start)
            .ok_or_else(|| {
                HandoffError::Validation(format!(
                    "layer frame {next_sequence} has an inverted token range"
                ))
            })?;
        let expected_bytes = u64::from(range_len)
            .checked_mul(u64::from(expected.class.kv_heads))
            .and_then(|value| value.checked_mul(u64::from(expected.class.head_dim)))
            .and_then(|value| value.checked_mul(begin.plane_element_bytes_v2(expected.role)))
            .ok_or_else(|| {
                HandoffError::Validation(format!("layer frame {next_sequence} byte bound overflow"))
            })?;
        // The expected tag is role-derived: pre-RoPE Key planes must be
        // tagged float32; an absent tag keeps meaning "float16", so it can
        // never stand in for a pre-RoPE plane.
        let expected_dtype = begin.expected_plane_dtype_v2(expected.class, expected.role);
        let dtype_ok = match &self.dtype {
            Some(tag) => tag == expected_dtype,
            None => expected_dtype == "float16",
        };
        if self.schema_version != LIVE_HANDOFF_SCHEMA_V1
            || self.transfer_id != begin.transfer_id
            || self.sequence != next_sequence
            || self.layer != expected.layer
            || self.role != expected.role
            || self.layout_class.as_deref() != Some(expected.class.class.as_str())
            || self.logical_token_start != expected.range_start
            || self.logical_token_end != expected.range_end
            || self.shape != [range_len, expected.class.kv_heads, expected.class.head_dim]
            || self.byte_length != expected_bytes
            || !dtype_ok
            || !valid_sha256(&self.sha256)
        {
            return Err(HandoffError::Validation(format!(
                "layer frame {next_sequence} does not match the v2 layout-table contract"
            )));
        }
        Ok(())
    }

    pub fn file_name(&self) -> String {
        format!("{:05}-{}.f16le", self.layer, self.role.suffix())
    }
}

pub fn descriptor_chain_sha256(headers: &[LayerHeaderV1]) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"kvpack-live-descriptor-chain-v1\0");
    for header in headers {
        hash.update(canonical_json(header)?);
        hash.update(b"\n");
    }
    Ok(hex::encode(hash.finalize()))
}

pub fn artifact_sha256(
    begin: &BeginManifestV1,
    headers: &[LayerHeaderV1],
    core: &SealCoreV1,
) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"kvpack-live-artifact-v1\0");
    hash.update(canonical_json(begin)?);
    hash.update(b"\n");
    for header in headers {
        hash.update(canonical_json(header)?);
        hash.update(b"\n");
    }
    hash.update(canonical_json(core)?);
    Ok(hex::encode(hash.finalize()))
}

/// F1: the byte stream an artifact HMAC tags. This is the same begin +
/// headers + core stream as [`artifact_sha256`], but prefixed with a
/// distinct domain tag (`kvpack-live-artifact-mac-v1`) so a plain
/// SHA-256 digest can never be replayed as a keyed tag.
pub fn artifact_mac_stream(
    begin: &BeginManifestV1,
    headers: &[LayerHeaderV1],
    core: &SealCoreV1,
) -> Result<Vec<u8>> {
    let mut stream = Vec::new();
    stream.extend_from_slice(b"kvpack-live-artifact-mac-v1\0");
    stream.extend_from_slice(&canonical_json(begin)?);
    stream.push(b'\n');
    for header in headers {
        stream.extend_from_slice(&canonical_json(header)?);
        stream.push(b'\n');
    }
    stream.extend_from_slice(&canonical_json(core)?);
    Ok(stream)
}

/// F1 (producer): the keyed HMAC-SHA256 tag a producer stamps into
/// `SealManifestV1::artifact_hmac_sha256` under the tenant [`MacKey`].
pub fn artifact_hmac_sha256(
    begin: &BeginManifestV1,
    headers: &[LayerHeaderV1],
    core: &SealCoreV1,
    key: &MacKey,
) -> Result<String> {
    key.tag_hex(&artifact_mac_stream(begin, headers, core)?)
}

impl SealManifestV1 {
    pub fn validate_for(
        &self,
        begin: &BeginManifestV1,
        headers: &[LayerHeaderV1],
        payload_bytes: u64,
        payload_sha256: &str,
    ) -> Result<()> {
        let core = &self.core;
        // F2: a pre-RoPE family is canary-gated — the bundle MUST carry an
        // authenticated canary record whose window fits inside the cached
        // token range and whose digests are well-formed. Presence is the
        // data-layer gate; the engine verifies the recorded digest against
        // its own pinned rotation kernel before committing the install
        // (contract §6). Any other family MUST NOT carry one (no spurious
        // authenticated fields).
        let canary_ok = if begin.is_prerope_v2() {
            match &core.canary {
                Some(canary) => {
                    canary.sample_token_count > 0
                        && canary
                            .sample_token_start
                            .saturating_add(canary.sample_token_count)
                            <= begin.cached_token_count
                        && valid_sha256(&canary.post_rope_k_sha256)
                        && valid_sha256(&canary.post_rope_v_sha256)
                }
                None => false,
            }
        } else {
            core.canary.is_none()
        };
        // F1: when the begin declares an HMAC key id, the seal MUST carry a
        // well-formed tag (domain-separated hex SHA-256). Verifying the tag
        // under the armed key is [`SealManifestV1::authenticate_hmac`]; the
        // integrity verifier only checks shape here so the unkeyed verifier
        // never needs a key.
        let hmac_shape_ok = match (&begin.hmac_key_id, &self.artifact_hmac_sha256) {
            (Some(_), Some(tag)) => valid_sha256(tag),
            (Some(_), None) => false,
            (None, None) => true,
            (None, Some(_)) => false,
        };
        if core.schema_version != LIVE_HANDOFF_SCHEMA_V1
            || core.protocol != LIVE_HANDOFF_PROTOCOL_V1
            || core.strategy != begin.strategy
            || core.transfer_id != begin.transfer_id
            || core.frame_count != begin.expected_layer_frames
            || usize::try_from(core.frame_count).ok() != Some(headers.len())
            || core.payload_bytes != begin.expected_payload_bytes
            || core.payload_bytes != payload_bytes
            || core.payload_sha256 != payload_sha256
            || core.descriptor_chain_sha256 != descriptor_chain_sha256(headers)?
            || core.token_ids_sha256 != begin.token_ids_sha256
            || core.token_ids_sha256 != token_ids_sha256(&core.prompt_token_ids)
            || core.prompt_token_ids.len()
                != usize::try_from(begin.cached_token_count)
                    .ok()
                    .and_then(|count| count.checked_add(1))
                    .unwrap_or(usize::MAX)
            || core.completed_unix_ms < begin.created_unix_ms
            || core.completed_unix_ms > begin.deadline_unix_ms
            || self.artifact_sha256 != artifact_sha256(begin, headers, core)?
            || !canary_ok
            || !hmac_shape_ok
        {
            return Err(HandoffError::Validation(
                "terminal seal does not authenticate the complete ordered handoff".into(),
            ));
        }
        Ok(())
    }

    /// F1 (consumer): authenticate this artifact under the armed tenant
    /// [`MacKey`]. The integrity verifier already proved the seal matches
    /// begin + headers + core, so this recomputes the keyed tag over those
    /// authenticated manifests and constant-time-compares it to the
    /// stamped tag. A bundle that reached the engine outside the
    /// authenticated transport (local file, locality hop) is forgeable
    /// without the key; this is the gate that refuses it.
    pub fn authenticate_hmac(
        &self,
        begin: &BeginManifestV1,
        headers: &[LayerHeaderV1],
        key: &MacKey,
    ) -> Result<()> {
        let tag = self.artifact_hmac_sha256.as_ref().ok_or_else(|| {
            HandoffError::Validation("sealed artifact carries no HMAC tag to authenticate".into())
        })?;
        key.verify_hex(&artifact_mac_stream(begin, headers, &self.core)?, tag)
    }
}

impl AckManifestV1 {
    pub fn committed(begin: &BeginManifestV1, seal: &SealManifestV1) -> Self {
        Self {
            artifact_sha256: seal.artifact_sha256.clone(),
            protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            status: "committed".into(),
            transfer_id: begin.transfer_id.clone(),
        }
    }
}
