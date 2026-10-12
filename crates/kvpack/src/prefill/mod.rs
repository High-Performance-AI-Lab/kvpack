//! Protocol-independent descriptor derivation for the qualified portable prefill.

use kvpack_core::{
    validate_family, CacheKind, Codec, DType, FamilyState, Id32, Layout, RepresentationFamilyId,
    RepresentationMode, SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
    PREFIX_BLOCK_TOKENS,
};
use sha2::{Digest, Sha256};

use crate::gguf_layout::{OwnedLayoutV2, RopeConvention};
use crate::{ExportStateDeclaration, StoreError};

mod relocate;
mod session;
mod session_artifact;
mod v1;
mod v2;

pub use v1::{
    derive_portable_prefill_descriptor_v1, portable_prefill_geometry_v1,
    PortablePrefillDescriptorInputV1, PortablePrefillGeometryV1, PORTABLE_PREFILL_ABI_V1,
    PORTABLE_PREFILL_GEOMETRIES_V1, PORTABLE_PREFILL_GEOMETRY_QWEN25_05B_V1,
    PORTABLE_PREFILL_GEOMETRY_QWEN25_7B_V1, PORTABLE_PREFILL_HEAD_DIM_V1,
    PORTABLE_PREFILL_KV_HEADS_V1, PORTABLE_PREFILL_LAYERS_V1, PORTABLE_PREFILL_MAX_CONTEXT_V1,
};
pub use v2::{
    bind_weights_scalar_math_v2, derive_portable_prefill_descriptor_v2,
    derive_portable_prefill_descriptor_v2_from_layout, portable_prefill_layout_name_v2,
    portable_prefill_layout_v2, PortablePrefillDescriptorInputV2, PortablePrefillLayoutClassV2,
    PortablePrefillLayoutV2, PreRopeKernelPinV1, WeightsScalarMathV1, PORTABLE_PREFILL_ABI_V2,
    PORTABLE_PREFILL_ABI_V2_PREROPE, PORTABLE_PREFILL_LAYOUTS_V2,
};
// `session` is the sole second consumer that justifies exposing these two
// v2-internal helpers crate-wide.
pub(crate) use v2::{class_state_names, effective_window_tokens};

pub use relocate::{
    relocate_plane_bytes, relocate_session_planes, PositionDelta, RelocateAction,
    SessionRelocateReport,
};

pub use session::{
    muse_session_resume_preconditions, muse_session_tail_shortfalls,
    place_windowed_tail_into_engine_ring, verify_nope_planes_require_no_rotation,
    ArtifactTailCoverage, TailCoverageShortfall,
};
pub use session_artifact::{
    MuseSessionArtifact, MuseSessionArtifactReceipt, MuseSessionPlaneWriter, MuseSessionWriter,
    MUSE_EXACT_LOGITS_LAYER, MUSE_EXACT_LOGITS_STATE,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePrefillDescriptorV1 {
    pub semantic_model: SemanticModelId,
    pub family: RepresentationFamilyId,
    pub states: Vec<ExportStateDeclaration>,
    pub bytes_per_state: u64,
    pub restored_bytes: u64,
}

pub fn portable_prefill_token_ids_sha256(token_ids: &[u32]) -> Id32 {
    let mut digest = Sha256::new();
    digest.update(b"kvpack-live-token-ids-v1\0");
    for token in token_ids {
        digest.update(token.to_le_bytes());
    }
    digest.finalize().into()
}

fn domain_id(domain: &[u8], parts: &[&[u8]]) -> Id32 {
    let mut hash = Sha256::new();
    hash.update(domain);
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}
