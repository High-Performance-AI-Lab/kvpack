//! Session save/resume artifact — START (the ferrite-research onboarding
//! plan, Part 1 §1.5-§1.6, the PRODUCT lane's session SAVE/RESUME spec). Built on the v2 portable-prefill descriptor
//! (K1-K3, K4 above) and Muse Glimmer's registered layout; the two layer
//! classes are handled exactly as the plan decides:
//!
//! - **13 NoPE planes (`gqa-full`)**: shipped as-is, full prefix coverage,
//!   no rotation call ever — restore is pure I/O (plan §1.5: "Restore is
//!   pure I/O; correctness does not depend on being restored at the
//!   capture offset"). `verify_nope_planes_require_no_rotation` makes this
//!   a checked invariant rather than a convention; `derive_portable_prefill_
//!   descriptor_v2*`'s prerope+NoPE guard (this branch's parent commit)
//!   makes the stronger derivation-time version of the same guarantee.
//! - **39 SWA planes (`gqa-windowed`)**: **Option A** — ship the windowed
//!   KV byte planes themselves (already-computed, already-rotated bytes,
//!   `window_tokens`-bounded), restored by plain copy. This is the plan's
//!   explicit decision (§1.5): *"Decision: ship the windowed planes
//!   (Option A). Do not ship a raw tail and recompute (Option B)."*
//!   Option B — a raw *token* tail replayed through a forward pass on
//!   restore — is explicitly reserved for cold-tier compaction only (V2,
//!   gated by E-KV-3), not the resume path implemented here. See the note
//!   below on the assignment text's wording for why this module does not
//!   implement a recompute path.
//!
//! **Wording note.** The originating assignment described the windowed
//! layers as "ship-the-2048-token-raw-tail with rebuild-on-restore." The
//! plan text is unambiguous that the *shipped* artifact must be the
//! windowed KV byte planes, not a raw token tail replayed through the
//! model (Option B, above) — so this module ships bytes, never tokens, for
//! the windowed classes. Read charitably, "raw tail" / "rebuild" also fits
//! Option A: the shipped bytes are a literal ("raw") trailing ("tail")
//! window of KV, and "rebuild on restore" is the mechanical placement of
//! those linear bytes back into the *consumer's* physical KV ring buffer —
//! exactly the SWA ring cell -> logical-position mapping stubbed below
//! (`place_windowed_tail_into_engine_ring`). Under either reading, no
//! forward-pass recompute happens here. `producer-spike.md` landed
//! mid-way through this branch's work and proved the placement mapping
//! for the trivial no-wraparound regime only, on real hardware — see that
//! function's doc comment for the details and what remains open.
//!
//! **Resume preconditions fail closed** (plan §1.6, PRODUCT §5: *"any
//! mismatch is a miss, never a best-effort restore"*): identity match
//! (`muse_session_resume_preconditions` compares the canonical
//! `semantic_model_id` / `representation_family_id` digests — the same
//! functions `kvpack-core` already uses for the public digest, per
//! `spec/IDENTITY_V1.md`) and full tail coverage for every windowed-class
//! state. Either failing is a plain miss, not a partial/best-effort
//! restore.

use std::collections::BTreeMap;

use kvpack_core::{representation_family_id, semantic_model_id};

use super::*;

/// Per-state coverage actually present in a candidate session artifact:
/// how many trailing tokens of real bytes are shipped for each windowed
/// state. This module does not read pack/manifest bytes itself — the
/// caller supplies this from wherever the artifact's manifest records
/// per-state extents (e.g. a `RealizedCutSchemaId`'s per-state byte
/// bounds, divided by the state's `elements_per_token * dtype width`).
pub type ArtifactTailCoverage = BTreeMap<StateKey, u32>;

/// One windowed-class state whose required trailing-token coverage is not
/// fully present in a candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailCoverageShortfall {
    pub key: StateKey,
    pub required_tokens: u32,
    pub present_tokens: u32,
}

/// Every windowed-class state (the 39 SWA layers' `attn.k`/`attn.v` planes)
/// whose required trailing-token coverage is not fully present in
/// `artifact_tail_coverage`, for a session at `cached_token_count`. Empty
/// means every windowed plane's tail is fully shipped. NoPE / unbounded
/// classes never appear here — they carry the full prefix, not a trailing
/// window (plan §1.5: "Restore is pure I/O").
pub fn muse_session_tail_shortfalls(
    layout: &PortablePrefillLayoutV2,
    cached_token_count: u32,
    artifact_tail_coverage: &ArtifactTailCoverage,
) -> Vec<TailCoverageShortfall> {
    let mut shortfalls = Vec::new();
    for class in layout.classes {
        if class.window_tokens == 0 {
            continue;
        }
        let required = effective_window_tokens(class.window_tokens, cached_token_count);
        for layer in class.layers() {
            for &state_name in class_state_names(class.class) {
                let key = StateKey::new(layer, state_name);
                let present = artifact_tail_coverage.get(&key).copied().unwrap_or(0);
                if present < required {
                    shortfalls.push(TailCoverageShortfall {
                        key,
                        required_tokens: required,
                        present_tokens: present,
                    });
                }
            }
        }
    }
    shortfalls
}

/// Fail-closed resume gate (plan §1.6 / PRODUCT §5): `stored` is the
/// descriptor authenticated at save time (read back from the artifact's
/// manifest); `runtime` is freshly derived from the resuming engine's own
/// claimed inputs. Any of the eight `CacheIdentity`-equivalent fields
/// disagreeing — weights/config, adapters, tokenizer/template, position
/// semantics, qualified math (the five `SemanticModelId` fields, compared
/// via the same canonical digest `spec/IDENTITY_V1.md` defines), or the
/// engine/geometry `RepresentationFamilyId` — is a miss, never a
/// best-effort restore. So is a windowed-class artifact that does not
/// carry its full required tail.
pub fn muse_session_resume_preconditions(
    stored: &PortablePrefillDescriptorV1,
    runtime: &PortablePrefillDescriptorV1,
    layout: &PortablePrefillLayoutV2,
    cached_token_count: u32,
    artifact_tail_coverage: &ArtifactTailCoverage,
) -> Result<(), StoreError> {
    if semantic_model_id(&stored.semantic_model) != semantic_model_id(&runtime.semantic_model) {
        return Err(StoreError::Authentication(
            "muse session resume: semantic model identity mismatch (weights/adapters/tokenizer/\
             position-semantics/qualified-math digest disagrees) — fail closed, never a \
             best-effort restore",
        ));
    }
    if representation_family_id(&stored.family)? != representation_family_id(&runtime.family)? {
        return Err(StoreError::Authentication(
            "muse session resume: representation family identity mismatch (engine cache ABI / \
             geometry disagrees) — fail closed, never a best-effort restore",
        ));
    }
    if !muse_session_tail_shortfalls(layout, cached_token_count, artifact_tail_coverage).is_empty()
    {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

/// Save-time invariant for the 13 NoPE planes: full-prefix (`Direct`) axis,
/// plain `DType::F16` — never the F32 pre-RoPE-capture representation.
/// `derive_portable_prefill_descriptor_v2*`'s prerope+NoPE guard already
/// makes the analogous derivation-time check a hard error (a NoPE class
/// cannot even be derived under the pre-RoPE label); this is the
/// complementary save-time check over an *already-derived* descriptor,
/// useful when the descriptor was constructed by a caller this module does
/// not control (e.g. read back from a stored manifest rather than derived
/// fresh). Fails closed rather than panicking: a violation here means the
/// artifact about to be shipped is wrong, not a test-only invariant.
pub fn verify_nope_planes_require_no_rotation(
    descriptor: &PortablePrefillDescriptorV1,
    layout: &PortablePrefillLayoutV2,
) -> Result<(), StoreError> {
    for class in layout.classes {
        if class.rope_convention != RopeConvention::None {
            continue;
        }
        for layer in class.layers() {
            for &state_name in class_state_names(class.class) {
                let key = StateKey::new(layer, state_name);
                let state = descriptor
                    .family
                    .states
                    .iter()
                    .find(|entry| entry.key == key)
                    .ok_or(StoreError::Expectation(
                        "muse session save: a NoPE-class state is missing from the descriptor",
                    ))?;
                if state.token_axis_rule != TokenAxisRule::Direct || state.dtype != DType::F16 {
                    return Err(StoreError::Expectation(
                        "muse session save: a NoPE-class plane carries a windowed axis or a \
                         non-F16 dtype — the rotation / pre-rotation-capture path must never be \
                         invoked for a NoPE class",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// STUB — the SWA ring cell -> logical-position mapping for the 39
/// windowed layers is still unproven for the case this function needs.
///
/// `results/muse-20260810/producer-spike.md` (ferrite-research) landed
/// mid-way through this branch's work (it did not exist when GLM's parent
/// K1/K3 commit bc07a44 flagged the dependency, nor for most of this
/// session module's own development — confirmed by direct repeated checks
/// of the path). Its headline finding corroborates this stub's framing
/// exactly, from real hardware, not static analysis: *"there is no
/// closed-form `cell = position mod window` formula anywhere in
/// llama.cpp. The physical cell<->position relationship is
/// allocator-emergent, not computed — any exporter must read per-cell
/// position metadata, never assume an index formula."* Concretely (per the
/// spike): llama.cpp's SWA physical ring is `GGML_PAD(min(n_ctx, n_swa *
/// n_seq_max + n_ubatch), 256)` cells — sized off `n_ubatch`, a
/// **producer-runtime** parameter, not a model constant — and a cell
/// becomes reusable via a FIFO distance test evaluated per candidate cell
/// during allocation, not a precomputed index. For a single
/// monotonically-growing sequence this *degenerates* to `cell_index ==
/// position mod physical_capacity`, but that is an observed property of
/// the access pattern, not a fact this function is entitled to assume.
///
/// What the spike proved on hardware was only the **no-wraparound**
/// regime (cached tokens within the physical capacity: cell positions
/// read back as a contiguous, ascending `[0..cached)` range — the trivial
/// identity mapping). It explicitly did **not** exercise the
/// wraparound/eviction regime engaged once cached tokens exceed physical
/// capacity — the regime that matters once a session runs past roughly
/// one window's worth of tokens, i.e. the regime this function exists
/// for — calling it out as "the next thing to prove." This stub therefore
/// still reflects the real state of the art, not a citation that has
/// simply gone stale.
///
/// This is the step between `muse_session_resume_preconditions` passing
/// (identity + tail coverage both verified) and actually installing the
/// shipped windowed-plane bytes into a *live* engine's physical KV ring
/// buffer. `kvpack`'s wire artifact itself is unaffected by this gap —
/// planes are already linearized by logical token order
/// (`logical_token_start = cached - min(window_tokens, cached)`,
/// `spec/KVPACK_LIVE_V2.md`, corroborated by the spike: *"kvpack's
/// `window_tokens` field ... must key off the model's fixed logical
/// window, never the engine's physical capacity, which is a
/// producer-runtime artifact"*) — this stub is for the ENGINE-side
/// placement step that runs after a `muse_session_resume_preconditions`
/// pass, not a gap in the artifact format K1-K4 authenticate.
///
/// `_linear_tail_tokens` is the shipped plane's bytes in logical-token
/// order (token `cached - window` .. `cached`, ascending); the real
/// implementation will scatter them into the consumer engine's physical
/// ring cells per the pending mapping, reading per-cell position metadata
/// exactly as the spike's own `llamacpp_session_send.py` parser already
/// does, rather than computing an index. Update this implementation once
/// the wraparound regime is proven.
pub fn place_windowed_tail_into_engine_ring(
    _layer: u32,
    _state_name: &str,
    _linear_tail_tokens: &[u8],
) -> Result<(), StoreError> {
    Err(StoreError::Expectation(
        "TODO(producer-spike): SWA ring cell->position mapping is not yet available — see \
         results/muse-20260810/producer-spike.md (ferrite-research); wraparound remains \
         unproven, so this stays stubbed and is not required for the wire artifact itself",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_layout::OwnedLayoutV2;

    fn muse_layout() -> &'static PortablePrefillLayoutV2 {
        portable_prefill_layout_v2("muse-glimmer-30b").unwrap()
    }

    fn muse_input() -> PortablePrefillDescriptorInputV2 {
        PortablePrefillDescriptorInputV2 {
            model_sha256: [0x82; 32],
            adapter_sha256: [0; 32],
            tokenizer_sha256: [0x11; 32],
            chat_template_sha256: [0x22; 32],
            context_policy_sha256: [0x33; 32],
            model_revision: "muse-glimmer-30b@Q4_K_XL".into(),
            tokenizer_revision: "muse-glimmer-30b-tokenizer@1".into(),
            producer_engine_abi: "llama.cpp-pr26841".into(),
            consumer_engine_abi: "ferrite-metal-v1".into(),
            portable_abi: PORTABLE_PREFILL_ABI_V2.into(),
            compute_precision: "float16".into(),
            kv_precision: "float16".into(),
            weight_precision: "q4_k_m".into(),
            cached_token_count: 4_096,
            max_context_tokens: 131_072,
            layout_name: "muse-glimmer-30b".into(),
            transform: None,
            prerope_kernel_pin: None,
        }
    }

    fn muse_scalar_math() -> WeightsScalarMathV1 {
        WeightsScalarMathV1 {
            qk_scale_factor_bits: 3.87f64.to_bits(),
            output_multiplier_bits: 0.196_116_135_138_184_04_f64.to_bits(),
            final_logit_softcapping_bits: 20.0f64.to_bits(),
            post_norm_eps_bits: 1e-8_f64.to_bits(),
        }
    }

    /// Full, correct tail coverage for `muse_input()`'s `cached_token_count`
    /// (4,096): every windowed-class state carries its full 2,048-token
    /// window.
    fn full_tail_coverage(cached_token_count: u32) -> ArtifactTailCoverage {
        let mut coverage = BTreeMap::new();
        for class in muse_layout().classes {
            if class.window_tokens == 0 {
                continue;
            }
            let required = effective_window_tokens(class.window_tokens, cached_token_count);
            for layer in class.layers() {
                for &state_name in class_state_names(class.class) {
                    coverage.insert(StateKey::new(layer, state_name), required);
                }
            }
        }
        coverage
    }

    #[test]
    fn resume_accepts_matching_identity_and_full_tail_coverage() {
        let stored = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let stored = bind_weights_scalar_math_v2(stored, &muse_scalar_math()).unwrap();
        let runtime = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let runtime = bind_weights_scalar_math_v2(runtime, &muse_scalar_math()).unwrap();
        let coverage = full_tail_coverage(muse_input().cached_token_count);
        assert!(muse_session_resume_preconditions(
            &stored,
            &runtime,
            muse_layout(),
            muse_input().cached_token_count,
            &coverage,
        )
        .is_ok());
    }

    #[test]
    fn resume_rejects_semantic_model_mismatch() {
        // Mutation rejection: flip model_sha256 (a SemanticModelId input)
        // on the runtime side only.
        let stored = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let mut mutated_input = muse_input();
        mutated_input.model_sha256 = [0xAB; 32];
        let runtime = derive_portable_prefill_descriptor_v2(&mutated_input).unwrap();
        let coverage = full_tail_coverage(muse_input().cached_token_count);
        let error = muse_session_resume_preconditions(
            &stored,
            &runtime,
            muse_layout(),
            muse_input().cached_token_count,
            &coverage,
        )
        .unwrap_err();
        assert!(matches!(error, StoreError::Authentication(_)));
    }

    #[test]
    fn resume_rejects_weights_scalar_math_mismatch() {
        // Mutation rejection: qk_scale_factor (K4) disagrees between the
        // artifact's authenticated calibration and the resuming engine's.
        let base = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let stored = bind_weights_scalar_math_v2(base.clone(), &muse_scalar_math()).unwrap();
        let wrong_scale = WeightsScalarMathV1 {
            qk_scale_factor_bits: 1.0f64.to_bits(),
            ..muse_scalar_math()
        };
        let runtime = bind_weights_scalar_math_v2(base, &wrong_scale).unwrap();
        let coverage = full_tail_coverage(muse_input().cached_token_count);
        let error = muse_session_resume_preconditions(
            &stored,
            &runtime,
            muse_layout(),
            muse_input().cached_token_count,
            &coverage,
        )
        .unwrap_err();
        assert!(matches!(error, StoreError::Authentication(_)));
    }

    #[test]
    fn resume_rejects_engine_cache_abi_mismatch() {
        // Mutation rejection: theta on the windowed class disagrees — an
        // engine_cache_abi-level mismatch (geometry/rope), not a
        // semantic_model-level one.
        let owned = muse_layout().to_owned_layout();
        let stored =
            derive_portable_prefill_descriptor_v2_from_layout(&muse_input(), &owned).unwrap();
        let mut mutated: OwnedLayoutV2 = owned;
        mutated
            .classes
            .iter_mut()
            .find(|c| c.class == "gqa-windowed")
            .unwrap()
            .rope_freq_base_bits = 250_000.0f64.to_bits();
        let runtime =
            derive_portable_prefill_descriptor_v2_from_layout(&muse_input(), &mutated).unwrap();
        let coverage = full_tail_coverage(muse_input().cached_token_count);
        let error = muse_session_resume_preconditions(
            &stored,
            &runtime,
            muse_layout(),
            muse_input().cached_token_count,
            &coverage,
        )
        .unwrap_err();
        assert!(matches!(error, StoreError::Authentication(_)));
    }

    #[test]
    fn resume_rejects_on_missing_or_short_tail_coverage() {
        let descriptor = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let cached = muse_input().cached_token_count;
        // Whole state absent.
        let mut short = full_tail_coverage(cached);
        short.remove(&StateKey::new(0, "attn.k"));
        let shortfalls = muse_session_tail_shortfalls(muse_layout(), cached, &short);
        assert_eq!(
            shortfalls,
            vec![TailCoverageShortfall {
                key: StateKey::new(0, "attn.k"),
                required_tokens: 2_048,
                present_tokens: 0,
            }]
        );
        let error = muse_session_resume_preconditions(
            &descriptor,
            &descriptor,
            muse_layout(),
            cached,
            &short,
        )
        .unwrap_err();
        assert!(matches!(error, StoreError::NotFound));

        // State present but truncated (a partially-written tail).
        let mut truncated = full_tail_coverage(cached);
        truncated.insert(StateKey::new(0, "attn.v"), 1_000);
        let error = muse_session_resume_preconditions(
            &descriptor,
            &descriptor,
            muse_layout(),
            cached,
            &truncated,
        )
        .unwrap_err();
        assert!(matches!(error, StoreError::NotFound));

        // NoPE-class states never require tail coverage — omitting them
        // entirely from the coverage map must not trip a shortfall.
        let mut only_windowed = BTreeMap::new();
        for class in muse_layout().classes {
            if class.window_tokens == 0 {
                continue;
            }
            for layer in class.layers() {
                for &state_name in class_state_names(class.class) {
                    only_windowed.insert(
                        StateKey::new(layer, state_name),
                        effective_window_tokens(class.window_tokens, cached),
                    );
                }
            }
        }
        assert!(muse_session_tail_shortfalls(muse_layout(), cached, &only_windowed).is_empty());
    }

    #[test]
    fn resume_cut_chain_alignment_at_2048() {
        // The required windowed tail is well-defined and cut-aligned at
        // every 256-token checkpoint boundary from the window size up to
        // full context — the same fact K1/K3 established for the layout
        // table (2048 / 256 == 8 exactly), reconfirmed here at the session
        // level: the tail-coverage requirement itself never asks for a
        // partial-cut window.
        const CUT: u32 = 256;
        for cached in (2_048..=131_072u32).step_by(CUT as usize) {
            for class in muse_layout().classes {
                if class.window_tokens == 0 {
                    continue;
                }
                let required = effective_window_tokens(class.window_tokens, cached);
                assert_eq!(
                    required % CUT,
                    0,
                    "required tail at cached={cached} is not cut-aligned"
                );
            }
        }
    }

    #[test]
    fn nope_planes_require_no_rotation_and_full_axis() {
        let descriptor = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        assert!(verify_nope_planes_require_no_rotation(&descriptor, muse_layout()).is_ok());

        // Adversarial mutation: force a NoPE-class state to look like it
        // went through the pre-RoPE F32 capture path (a windowed axis and
        // an F32 dtype it should never carry). Must be rejected, not
        // silently accepted.
        let mut broken = descriptor.clone();
        let target = broken
            .family
            .states
            .iter_mut()
            .find(|state| state.key == StateKey::new(3, "attn.k"))
            .expect("layer 3 is the first NoPE layer");
        assert_eq!(target.token_axis_rule, TokenAxisRule::Direct);
        target.token_axis_rule = TokenAxisRule::TailWindow;
        target.dtype = DType::F32;
        let error = verify_nope_planes_require_no_rotation(&broken, muse_layout()).unwrap_err();
        assert!(matches!(error, StoreError::Expectation(_)));
    }

    #[test]
    fn ring_mapping_stub_documents_pending_dependency() {
        let error = place_windowed_tail_into_engine_ring(0, "attn.k", &[]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("producer-spike.md"));
        assert!(message.contains("TODO"));
    }

    #[test]
    fn session_family_schema_round_trip_byte_exact() {
        // Schema round-trip byte-exactness over the session artifact's
        // authenticated family object: encode, decode, and re-encode must
        // reproduce the identical bytes (RepresentationFamilyId::
        // decode_canonical already self-checks this internally; asserting
        // it here pins the behavior for the session artifact specifically,
        // including the K4-bound weights_config).
        let descriptor = derive_portable_prefill_descriptor_v2(&muse_input()).unwrap();
        let descriptor = bind_weights_scalar_math_v2(descriptor, &muse_scalar_math()).unwrap();
        let encoded = descriptor.family.encode_canonical().unwrap();
        let decoded = RepresentationFamilyId::decode_canonical(&encoded).unwrap();
        assert_eq!(decoded, descriptor.family);
        let re_encoded = decoded.encode_canonical().unwrap();
        assert_eq!(re_encoded, encoded);
    }
}
