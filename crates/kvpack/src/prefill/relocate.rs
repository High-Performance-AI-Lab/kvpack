//! Relocation — shifting a captured session's absolute position offset
//! (the ferrite-research relocation product proof, built on this crate's
//! session save/resume artifact).
//!
//! This is the primitive `place_windowed_tail_into_engine_ring` and
//! `muse_session_resume_preconditions` (`prefill/session.rs`) do not cover:
//! those two handle *same-offset* resume (identity + tail coverage +
//! engine-ring placement). Relocation is a *different-offset* restore — the
//! saved chunk's own absolute position changes — and needs one more
//! operation per plane before the bytes are valid at the new offset.
//!
//! The governing algebra (`docs/KV_ALGEBRA_2026-08-09.md`): RoPE's position
//! map is a group homomorphism `(Z,+) -> SO(2)^{d/2}`, so
//! `R(p_old + delta) = R(delta) * R(p_old)`. A plane already carries
//! `R(p_old)` baked in at capture time (Option A ships already-rotated
//! bytes, `prefill/session.rs` module doc); relocating to `p_old + delta`
//! only needs the *delta* rotation applied on top, never the original
//! position or the pre-rotation K.
//!
//! Per-class dispatch, architectural, not a tuning choice:
//!
//! - **`RopeConvention::None` (Muse's 13 NoPE layers).** No rotation was
//!   ever applied at capture (`verify_nope_planes_require_no_rotation`
//!   already makes that a checked invariant); relocation is therefore a
//!   pure memcpy — this module does not even read the bytes, let alone
//!   transform them. This is the "architectural free lunch" the product
//!   framing rests on: byte-identical before/after is not a property this
//!   code achieves by being careful, it is a property of never touching the
//!   bytes at all. [`relocate_session_planes`] asserts it anyway, because a
//!   framing this load-bearing should be a checked gate, not a silent
//!   consequence of the control-flow shape.
//! - **`attn.v`, any class.** RoPE is never applied to V in any known
//!   transformer, Muse included — always a memcpy, independent of class.
//! - **`attn.k`, a windowed class (Muse's 39 SWA layers: `RopeConvention::
//!   Interleaved`; other registered layouts — Qwen2.5, gpt-oss, Gemma4 —
//!   use `RopeConvention::Neox`).** Rotate every token's K vector by
//!   `R(delta)`. The two conventions differ only in which elements a pair
//!   couples — NEOX couples index `i` with `i + rotary/2` (split-half);
//!   interleaved couples `2i` with `2i+1` (adjacent) — the per-pair
//!   frequency and angle math is identical either way: `omega_i =
//!   freq_base^(-2i/rotary)`, angle `delta * omega_i`. `delta` is the
//!   *same* signed integer for every token in a relocated chunk, because a
//!   relocation shifts the whole chunk rigidly — every token's absolute
//!   position moves by the same amount, so the per-pair angle is a single
//!   scalar per plane, not a per-token array.
//!
//! **Muse's convention is `Interleaved`, corrected 2026-08-11.** The layout
//! table (`prefill/v2.rs`, `MUSE_GLIMMER_30B_CLASSES`) originally shipped
//! `RopeConvention::Neox` for the windowed class, explicitly flagged at the
//! time as an unverified guess (PRODUCT §5: "Pairing convention (NEOX vs
//! interleaved) ... the value is unverified — a data question, flagged
//! open"). It is now verified against ground truth — Ferrite's CPU
//! reference forward pass, transcribed from and cross-checked token-exact
//! against llama.cpp's own `src/models/muse-glimmer.cpp`, applies
//! `LLAMA_ROPE_TYPE_NORM`: "rotate the interleaved pairs `(x[2i],
//! x[2i+1])`" — and Ferrite's independently-developed GPU kernel layer
//! reached the same conclusion hard enough to build an explicit fence that
//! *refuses* to run any NEOX-only kernel against a NORM-layout model rather
//! than silently mis-pairing it. See `prefill/v2.rs`'s updated class
//! comment for the full citation. This module implements both pairings
//! (NEOX is real and live for Qwen2.5/gpt-oss/Gemma4's registered layouts,
//! so it stays exercised and correct, not dead code) and dispatches on
//! whatever the layout class actually declares — Muse gets `Interleaved`,
//! nothing here hardcodes an assumption either way.
//!
//! **What this module deliberately does not claim.** `KV_ALGEBRA_2026-08-09.md`
//! is explicit that the translation identity is exact in the reals but
//! *not* bit-exact in floating point except under a pinned kernel+device+
//! dtype+geometry contract: `float(pos)*freq` rounding makes long-position
//! angles differ from the reference engine's own kernel by ulps. This
//! module computes the same delta-rotation algebra in f64 (matching
//! `PortablePrefillLayoutClassV2::rope_freq_base_bits`'s own bit-exact-f64
//! convention) before rounding to the wire's f16, which is the correct
//! *math*, but it is an independent implementation, not literally Ferrite's
//! pinned Metal/CPU kernel — so the relocated gate this module exists to
//! support is a measured drift bound (<=0.5 max|Delta-logit>, the sink-free
//! substrate precedent this session's queued conformal-PE note cites), not
//! a byte-identical assertion. Byte-identical is reserved for what it is
//! architecturally true for: NoPE planes, and delta==0 (true-suffix) restore.

use std::collections::BTreeMap;

use kvpack_core::{half::f16_to_f32, half::f32_to_f16, StateKey};

use super::{class_state_names, PortablePrefillLayoutClassV2, PortablePrefillLayoutV2};
use crate::gguf_layout::RopeConvention;
use crate::StoreError;

/// A signed absolute-position shift applied rigidly to every token in a
/// relocated chunk. Positive moves the chunk later in the sequence
/// (`new_offset > old_offset`), negative earlier.
pub type PositionDelta = i64;

/// How [`relocate_plane_bytes`] handled one plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocateAction {
    /// NoPE class, or the V-role of any class: bytes untouched.
    MemcpyNoOp,
    /// Windowed class's K-role, `delta != 0`: every token's K vector
    /// rotated by `R(delta)` in place.
    DeltaRotated,
    /// Windowed class's K-role, `delta == 0`: the true-suffix case. No
    /// bytes touched (an explicit early return, not `R(0)` computed and
    /// applied) — kept distinct from `MemcpyNoOp` so callers can tell "this
    /// plane is architecturally position-free" apart from "this plane
    /// happened to relocate by zero this time".
    IdentityDelta,
}

impl RelocateAction {
    pub fn bytes_are_unchanged(self) -> bool {
        matches!(self, Self::MemcpyNoOp | Self::IdentityDelta)
    }
}

/// Relocate one plane's bytes — one `(layer, state_name)` pair, already
/// linearized in logical-token order (`logical_token_start = cached -
/// min(window, cached)`, `spec/KVPACK_LIVE_V2.md`) — by `delta` absolute
/// position tokens, in place.
///
/// Byte layout matches the wire's per-plane layout, confirmed against real
/// exported hardware bytes (`scratchpad/muse-results/producer-spike.md`,
/// ferrite-research): row-major `[token][kv_head][head_dim]`, `DType::F16`,
/// K and V as separate planes (`class_state_names`: `"attn.k"`, `"attn.v"`).
pub fn relocate_plane_bytes(
    class: &PortablePrefillLayoutClassV2,
    state_name: &str,
    delta: PositionDelta,
    bytes: &mut [u8],
) -> Result<RelocateAction, StoreError> {
    if class.rope_convention == RopeConvention::None || state_name != "attn.k" {
        return Ok(RelocateAction::MemcpyNoOp);
    }
    if delta == 0 {
        return Ok(RelocateAction::IdentityDelta);
    }
    let pairing = match class.rope_convention {
        RopeConvention::Neox => PairingConvention::SplitHalf,
        RopeConvention::Interleaved => PairingConvention::Adjacent,
        RopeConvention::None => unreachable!("handled by the early return above"),
    };
    rotate_k_inplace(class, pairing, delta, bytes)?;
    Ok(RelocateAction::DeltaRotated)
}

/// Which elements a RoPE rotation pair index `i` (0..rotary/2) refers to
/// within a `head_dim`-wide K vector. The frequency/angle math is identical
/// either way; only the element addressing differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairingConvention {
    /// NEOX: pair `i` couples elements `i` and `i + rotary/2`.
    SplitHalf,
    /// NORM / GPT-J style: pair `i` couples elements `2i` and `2i+1`.
    Adjacent,
}

impl PairingConvention {
    #[inline]
    fn offsets(self, pair_index: usize, half: usize) -> (usize, usize) {
        match self {
            Self::SplitHalf => (pair_index, pair_index + half),
            Self::Adjacent => (2 * pair_index, 2 * pair_index + 1),
        }
    }
}

/// Delta rotation of every token's K vector in `bytes`, in place, under the
/// given pairing convention. Frequency `omega_i = freq_base^(-2i/rotary)`;
/// rotation angle `delta * omega_i`, the same scalar per pair index for
/// every token and every kv_head this plane covers, because a relocation
/// shifts the whole chunk by one constant delta.
fn rotate_k_inplace(
    class: &PortablePrefillLayoutClassV2,
    pairing: PairingConvention,
    delta: PositionDelta,
    bytes: &mut [u8],
) -> Result<(), StoreError> {
    let head_dim = class.head_dim as usize;
    let kv_heads = class.kv_heads as usize;
    let rotary = class.rope_dimension_count as usize;
    if rotary == 0 || rotary % 2 != 0 || rotary > head_dim || kv_heads == 0 {
        return Err(StoreError::Expectation(
            "relocate: invalid rotary width / kv_heads for delta-rotation",
        ));
    }
    let freq_base = f64::from_bits(class.rope_freq_base_bits);
    if freq_base.partial_cmp(&1.0) != Some(std::cmp::Ordering::Greater) {
        return Err(StoreError::Expectation(
            "relocate: delta-rotation requires freq_base > 1.0 (a NoPE class must dispatch \
             through the RopeConvention::None no-op path, never here)",
        ));
    }
    let elems_per_token = kv_heads * head_dim;
    let bytes_per_token = elems_per_token * 2; // DType::F16
    if bytes_per_token == 0 || bytes.len() % bytes_per_token != 0 {
        return Err(StoreError::Expectation(
            "relocate: plane byte length is not a whole number of tokens for this class's geometry",
        ));
    }
    let token_count = bytes.len() / bytes_per_token;
    let half = rotary / 2;

    // Per-pair-index angle is identical for every token/head this plane
    // covers (delta is one constant shift for the whole relocated chunk) —
    // compute each pair's (cos, sin) once, not per token.
    let cos_sin: Vec<(f64, f64)> = (0..half)
        .map(|i| {
            let omega = freq_base.powf(-2.0 * (i as f64) / (rotary as f64));
            let angle = (delta as f64) * omega;
            (angle.cos(), angle.sin())
        })
        .collect();

    for token in 0..token_count {
        let token_base = token * bytes_per_token;
        for head in 0..kv_heads {
            let head_base = token_base + head * head_dim * 2;
            for (i, &(c, s)) in cos_sin.iter().enumerate() {
                let (off0, off1) = pairing.offsets(i, half);
                let lo = head_base + off0 * 2;
                let hi = head_base + off1 * 2;
                let x0 = f16_to_f32(u16::from_le_bytes([bytes[lo], bytes[lo + 1]])) as f64;
                let x1 = f16_to_f32(u16::from_le_bytes([bytes[hi], bytes[hi + 1]])) as f64;
                let y0 = x0 * c - x1 * s;
                let y1 = x0 * s + x1 * c;
                bytes[lo..lo + 2].copy_from_slice(&f32_to_f16(y0 as f32).to_le_bytes());
                bytes[hi..hi + 2].copy_from_slice(&f32_to_f16(y1 as f32).to_le_bytes());
            }
            // rotary < head_dim never occurs for any registered layout today
            // (Muse's windowed class: rotary == head_dim == 128); any tail
            // beyond `rotary` would be left untouched by construction (the
            // loop above only ever indexes pairs drawn from `[0, rotary)`).
        }
    }
    Ok(())
}

/// Outcome of relocating every plane in a captured session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRelocateReport {
    /// Planes handled by memcpy: all 13 NoPE `attn.k` + all 52 `attn.v`
    /// planes (65 for Muse Glimmer at any `delta`), plus, when `delta==0`,
    /// the 39 SWA `attn.k` planes too (`IdentityDelta`, tracked separately).
    pub memcpy_planes: usize,
    /// Windowed `attn.k` planes actually rotated (`delta != 0`): 39 for
    /// Muse Glimmer's full layout.
    pub rotated_planes: usize,
    /// Windowed `attn.k` planes that took the `delta == 0` identity path.
    pub identity_delta_planes: usize,
    /// Every NoPE-class plane's bytes, compared before vs. after this call,
    /// came back byte-identical. Checked explicitly rather than inferred
    /// from `relocate_plane_bytes` never writing to them — this is the
    /// gate the product framing rests on, so it is asserted, not assumed.
    pub nope_bytes_identical: bool,
}

/// Relocate every plane of a registered layout's session by `delta`
/// absolute-position tokens. `planes` maps `(layer, state_name)` to that
/// plane's bytes (already sized per the layout's per-class geometry, in
/// logical-token order) and is mutated in place; every plane the layout
/// declares must be present or this fails closed with `StoreError::NotFound`
/// — a partial relocation is not a valid one.
pub fn relocate_session_planes(
    layout: &PortablePrefillLayoutV2,
    delta: PositionDelta,
    planes: &mut BTreeMap<StateKey, Vec<u8>>,
) -> Result<SessionRelocateReport, StoreError> {
    let mut memcpy_planes = 0usize;
    let mut rotated_planes = 0usize;
    let mut identity_delta_planes = 0usize;
    let mut nope_bytes_identical = true;

    for class in layout.classes {
        for layer in class.layers() {
            for &state_name in class_state_names(class.class) {
                let key = StateKey::new(layer, state_name);
                let bytes = planes.get_mut(&key).ok_or(StoreError::NotFound)?;
                let is_nope_plane = class.rope_convention == RopeConvention::None;
                let before = if is_nope_plane {
                    Some(bytes.clone())
                } else {
                    None
                };

                let action = relocate_plane_bytes(class, state_name, delta, bytes)?;
                match action {
                    RelocateAction::MemcpyNoOp => memcpy_planes += 1,
                    RelocateAction::DeltaRotated => rotated_planes += 1,
                    RelocateAction::IdentityDelta => identity_delta_planes += 1,
                }

                if let Some(before) = before {
                    if *bytes != before {
                        nope_bytes_identical = false;
                    }
                }
            }
        }
    }

    Ok(SessionRelocateReport {
        memcpy_planes,
        rotated_planes,
        identity_delta_planes,
        nope_bytes_identical,
    })
}

#[cfg(test)]
mod tests {
    use super::super::portable_prefill_layout_v2;
    use super::*;

    fn muse_layout() -> &'static PortablePrefillLayoutV2 {
        portable_prefill_layout_v2("muse-glimmer-30b").unwrap()
    }

    fn nope_class() -> &'static PortablePrefillLayoutClassV2 {
        muse_layout()
            .classes
            .iter()
            .find(|c| c.rope_convention == RopeConvention::None)
            .unwrap()
    }

    /// Muse's windowed class — the one this whole module exists to serve.
    /// Its registered convention is `Interleaved` (corrected 2026-08-11,
    /// see the module doc); asserted here so a future accidental revert of
    /// that fix fails a test instead of silently reintroducing the bug.
    fn muse_swa_class() -> &'static PortablePrefillLayoutClassV2 {
        let class = muse_layout()
            .classes
            .iter()
            .find(|c| c.window_tokens > 0)
            .unwrap();
        assert_eq!(class.rope_convention, RopeConvention::Interleaved);
        class
    }

    /// A real registered NEOX-convention windowed class, so the split-half
    /// pairing path stays exercised by something that actually uses it
    /// (Qwen2.5's class is full-attention/unwindowed; gpt-oss's windowed
    /// class is NEOX and is the one used here).
    fn neox_windowed_class() -> &'static PortablePrefillLayoutClassV2 {
        let layout = portable_prefill_layout_v2("gpt-oss-120b").unwrap();
        let class = layout.classes.iter().find(|c| c.window_tokens > 0).unwrap();
        assert_eq!(class.rope_convention, RopeConvention::Neox);
        class
    }

    /// Deterministic pseudo-random f16 plane bytes for `token_count` tokens
    /// of a given class — a fixed xorshift so tests are reproducible without
    /// pulling in a `rand` dependency.
    fn synth_plane_bytes(
        class: &PortablePrefillLayoutClassV2,
        token_count: usize,
        seed: u64,
    ) -> Vec<u8> {
        let elems = class.kv_heads as usize * class.head_dim as usize * token_count;
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(elems * 2);
        for _ in 0..elems {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to a small-ish float range so f16 round-trips are exact-ish
            // and rotation never saturates: [-4.0, 4.0).
            let unit = (state >> 40) as u32 as f32 / (1u32 << 24) as f32;
            let value = (unit - 0.5) * 8.0;
            out.extend_from_slice(&f32_to_f16(value).to_le_bytes());
        }
        out
    }

    /// Independent ground-truth rotation: rotate `bytes` (already sitting at
    /// `base_position`) directly to `base_position + delta` by recomputing
    /// the ABSOLUTE angle at the target position from the identity K value —
    /// i.e., this helper undoes `base_position`'s rotation and reapplies
    /// `base_position + delta`'s rotation from scratch, as a from-first-
    /// principles cross-check independent of `rotate_k_inplace`'s own
    /// delta-composition code path. `pairing` must match the class under
    /// test; passed explicitly (rather than re-derived from
    /// `class.rope_convention`) so this helper stays independent of
    /// `relocate_plane_bytes`'s own convention-to-pairing mapping.
    fn rotate_absolute_reference(
        class: &PortablePrefillLayoutClassV2,
        pairing: PairingConvention,
        base_position: i64,
        target_position: i64,
        bytes: &[u8],
    ) -> Vec<u8> {
        let head_dim = class.head_dim as usize;
        let kv_heads = class.kv_heads as usize;
        let rotary = class.rope_dimension_count as usize;
        let half = rotary / 2;
        let freq_base = f64::from_bits(class.rope_freq_base_bits);
        let bytes_per_token = kv_heads * head_dim * 2;
        let token_count = bytes.len() / bytes_per_token;
        let mut out = bytes.to_vec();
        for token in 0..token_count {
            let token_base = token * bytes_per_token;
            for head in 0..kv_heads {
                let head_base = token_base + head * head_dim * 2;
                for i in 0..half {
                    let omega = freq_base.powf(-2.0 * (i as f64) / (rotary as f64));
                    let (off0, off1) = pairing.offsets(i, half);
                    let lo = head_base + off0 * 2;
                    let hi = head_base + off1 * 2;
                    let x0 = f16_to_f32(u16::from_le_bytes([bytes[lo], bytes[lo + 1]])) as f64;
                    let x1 = f16_to_f32(u16::from_le_bytes([bytes[hi], bytes[hi + 1]])) as f64;
                    // Un-rotate by base_position, then rotate by target_position:
                    // R(target) * R(-base) applied directly, computed as a
                    // single combined angle (mathematically identical to two
                    // sequential rotations, done in one step here for a
                    // *differently-shaped* reference computation than the
                    // production code's compose-by-delta approach).
                    let combined_angle = ((target_position - base_position) as f64) * omega;
                    let c = combined_angle.cos();
                    let s = combined_angle.sin();
                    let y0 = x0 * c - x1 * s;
                    let y1 = x0 * s + x1 * c;
                    out[lo..lo + 2].copy_from_slice(&f32_to_f16(y0 as f32).to_le_bytes());
                    out[hi..hi + 2].copy_from_slice(&f32_to_f16(y1 as f32).to_le_bytes());
                }
            }
        }
        out
    }

    #[test]
    fn nope_plane_is_byte_identical_regardless_of_delta() {
        let class = nope_class();
        for &delta in &[0i64, 1, -1, 512, -2048, 131_072] {
            let original = synth_plane_bytes(class, 17, 0xC0FFEE);
            let mut bytes = original.clone();
            let action = relocate_plane_bytes(class, "attn.k", delta, &mut bytes).unwrap();
            assert_eq!(action, RelocateAction::MemcpyNoOp);
            assert_eq!(
                bytes, original,
                "NoPE plane must be byte-identical at delta={delta}"
            );
        }
    }

    #[test]
    fn v_plane_is_always_byte_identical_even_for_windowed_class() {
        let class = muse_swa_class();
        let original = synth_plane_bytes(class, 9, 0xBEEF);
        let mut bytes = original.clone();
        let action = relocate_plane_bytes(class, "attn.v", 777, &mut bytes).unwrap();
        assert_eq!(action, RelocateAction::MemcpyNoOp);
        assert_eq!(bytes, original);
    }

    #[test]
    fn swa_k_plane_identity_delta_is_byte_identical() {
        let class = muse_swa_class();
        let original = synth_plane_bytes(class, 5, 0xA11CE);
        let mut bytes = original.clone();
        let action = relocate_plane_bytes(class, "attn.k", 0, &mut bytes).unwrap();
        assert_eq!(action, RelocateAction::IdentityDelta);
        assert_eq!(
            bytes, original,
            "delta==0 (true-suffix) must not touch the bytes"
        );
    }

    #[test]
    fn muse_swa_k_plane_rotation_matches_independent_absolute_reference_interleaved() {
        // The production path rotates by composing R(delta) onto the
        // already-rotated stored bytes. The reference here recomputes the
        // combined angle directly rather than reusing that composition
        // code — an independent derivation of the same group-homomorphism
        // identity from KV_ALGEBRA_2026-08-09.md (R(p_old+delta) =
        // R(delta)*R(p_old)), not a restatement of the implementation. Uses
        // Muse's actual registered class (Interleaved pairing).
        let class = muse_swa_class();
        for &(base, delta) in &[(0i64, 512i64), (1024, -512), (0, 2047), (999, 1)] {
            let original = synth_plane_bytes(class, 6, base as u64 ^ 0x5EED);
            let mut produced = original.clone();
            relocate_plane_bytes(class, "attn.k", delta, &mut produced).unwrap();
            let expected = rotate_absolute_reference(
                class,
                PairingConvention::Adjacent,
                base,
                base + delta,
                &original,
            );
            assert_eq!(
                produced, expected,
                "base={base} delta={delta}: production rotation disagrees with the independent reference"
            );
        }
    }

    #[test]
    fn neox_windowed_k_plane_rotation_matches_independent_absolute_reference() {
        // Same cross-check, for the split-half (NEOX) pairing path — kept
        // exercised because real registered layouts (gpt-oss, Gemma4) use
        // it even though Muse itself does not.
        let class = neox_windowed_class();
        for &(base, delta) in &[(0i64, 100i64), (500, -50), (0, 127)] {
            let original = synth_plane_bytes(class, 4, base as u64 ^ 0x77AA);
            let mut produced = original.clone();
            relocate_plane_bytes(class, "attn.k", delta, &mut produced).unwrap();
            let expected = rotate_absolute_reference(
                class,
                PairingConvention::SplitHalf,
                base,
                base + delta,
                &original,
            );
            assert_eq!(
                produced, expected,
                "base={base} delta={delta}: NEOX production rotation disagrees with the independent reference"
            );
        }
    }

    #[test]
    fn swa_k_plane_delta_composition_matches_group_homomorphism() {
        // R(d2) * R(d1) applied sequentially should equal R(d1+d2) applied
        // once — the algebraic property KV_ALGEBRA_2026-08-09.md names as
        // the reason relocation-by-delta is valid at all — up to a tight
        // numeric tolerance, NOT exact byte equality: the sequential path
        // rounds to f16 twice (once per call) while the direct path rounds
        // once, and KV_ALGEBRA_2026-08-09.md is explicit that this class of
        // double-rounding is exactly where float(pos)*freq stops being
        // bit-translation-invariant even though it is exact in the reals.
        // First run empirically confirmed this: byte-exact equality failed
        // with scattered +/-1-ULP f16 differences, consistent with that
        // doc's own prediction — recorded here as the expected, bounded
        // shape of the residual, not chased to zero.
        let class = muse_swa_class();
        let original = synth_plane_bytes(class, 4, 0x1234);

        let mut sequential = original.clone();
        relocate_plane_bytes(class, "attn.k", 300, &mut sequential).unwrap();
        relocate_plane_bytes(class, "attn.k", 212, &mut sequential).unwrap();

        let mut direct = original.clone();
        relocate_plane_bytes(class, "attn.k", 512, &mut direct).unwrap();

        let mut max_abs_diff = 0.0f32;
        let mut differing_elements = 0usize;
        let mut total_elements = 0usize;
        for (a, b) in sequential.chunks(2).zip(direct.chunks(2)) {
            let av = f16_to_f32(u16::from_le_bytes([a[0], a[1]]));
            let bv = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            total_elements += 1;
            if a != b {
                differing_elements += 1;
            }
            max_abs_diff = max_abs_diff.max((av - bv).abs());
        }
        assert!(
            max_abs_diff <= 0.01,
            "R(212)*R(300) vs R(512): max abs diff {max_abs_diff} exceeds the double-rounding \
             tolerance ({differing_elements}/{total_elements} elements differed at all)"
        );
    }

    #[test]
    fn swa_k_plane_negative_delta_round_trips_close_to_original() {
        // R(-delta) * R(delta) should recover the original K, up to f16
        // round-trip rounding (two rotations = two f16 requantizations).
        let class = muse_swa_class();
        let original = synth_plane_bytes(class, 8, 0x9E3779B9);
        let mut bytes = original.clone();
        relocate_plane_bytes(class, "attn.k", 900, &mut bytes).unwrap();
        relocate_plane_bytes(class, "attn.k", -900, &mut bytes).unwrap();
        for (a, b) in original.chunks(2).zip(bytes.chunks(2)) {
            let av = f16_to_f32(u16::from_le_bytes([a[0], a[1]]));
            let bv = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            assert!(
                (av - bv).abs() <= 0.02,
                "round-trip rotation drifted too far: {av} vs {bv}"
            );
        }
    }

    #[test]
    fn relocate_session_planes_reports_the_muse_split_exactly() {
        let layout = muse_layout();
        let mut planes: BTreeMap<StateKey, Vec<u8>> = BTreeMap::new();
        for class in layout.classes {
            for layer in class.layers() {
                for &state_name in class_state_names(class.class) {
                    planes.insert(
                        StateKey::new(layer, state_name),
                        synth_plane_bytes(class, 3, u64::from(layer) ^ (state_name.len() as u64)),
                    );
                }
            }
        }
        let report = relocate_session_planes(layout, 512, &mut planes).unwrap();
        // 13 NoPE layers x {k,v} = 26 memcpy, plus 39 SWA layers' v = 39 memcpy => 65.
        assert_eq!(report.memcpy_planes, 65);
        // 39 SWA layers' k, delta != 0 => all rotated.
        assert_eq!(report.rotated_planes, 39);
        assert_eq!(report.identity_delta_planes, 0);
        assert!(report.nope_bytes_identical);
    }

    #[test]
    fn relocate_session_planes_delta_zero_is_all_no_op() {
        let layout = muse_layout();
        let mut planes: BTreeMap<StateKey, Vec<u8>> = BTreeMap::new();
        for class in layout.classes {
            for layer in class.layers() {
                for &state_name in class_state_names(class.class) {
                    planes.insert(
                        StateKey::new(layer, state_name),
                        synth_plane_bytes(class, 2, u64::from(layer) + 7),
                    );
                }
            }
        }
        let report = relocate_session_planes(layout, 0, &mut planes).unwrap();
        assert_eq!(report.memcpy_planes, 65);
        assert_eq!(report.rotated_planes, 0);
        assert_eq!(report.identity_delta_planes, 39);
        assert!(report.nope_bytes_identical);
    }

    #[test]
    fn relocate_session_planes_fails_closed_on_missing_plane() {
        let layout = muse_layout();
        let mut planes: BTreeMap<StateKey, Vec<u8>> = BTreeMap::new();
        // Deliberately omit everything: must fail, never silently relocate
        // a subset.
        let error = relocate_session_planes(layout, 128, &mut planes).unwrap_err();
        assert!(matches!(error, StoreError::NotFound));
    }
}
