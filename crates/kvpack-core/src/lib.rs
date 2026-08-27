//! Authoritative, model-aware production-v1 wire contract.
//!
//! Only the fixed binary encodings in this crate are durable or authenticated.
//! Protobuf, JSON and database rows are control-plane representations and must
//! never be used as content identities.  The pre-production record/commit
//! format deliberately has different magic and is not accepted here.

#![forbid(unsafe_code)]

mod canonical;
pub mod chunk;
pub mod consts;
pub mod enums;
pub mod error;
pub mod half;
pub mod identity;
pub mod ids;
pub mod keys;
pub mod manifest;
pub mod pack;
pub mod quant;
pub mod rotation;
pub mod stats;
pub mod transform;
pub mod validator;

pub use chunk::{
    decode_chunk, decode_chunk_with_stats, decode_codec_frame, encode_chunk,
    encode_chunk_with_content_id, encode_codec_frame, ChunkEncoding, ChunkObject,
};
pub use consts::*;
pub use enums::{CacheKind, Codec, DType, Layout, RepresentationMode, TokenAxisRule};
pub use error::PackError;
pub use identity::{
    AuxiliaryInputId, FamilyState, InputCutId, RepresentationFamilyId, SemanticModelId,
    StaticDimension,
};
pub use ids::{
    auxiliary_input_root, chain_prefix_nodes, chunk_id, derive_input_cut, manifest_id,
    namespace_id, realized_cut_schema_id, representation_family_id, semantic_model_id,
    CutChainDeriver, CutStridePolicy, PrefixNode,
};
pub use keys::KeySchedule;
pub use manifest::{
    AtomicGroup, ChunkRef, ChunkSpan, CutManifest, ManifestDeclaration, ManifestKind,
    RealizedCutSchemaId, RealizedStateSchema, Shape, StateDeclaration, StateKey, StateManifest,
};
pub use pack::{
    decode_authenticated_pack, encode_authenticated_pack, inspect_pack_header,
    verify_authenticated_pack, EncodedPack, PackHeader,
};
pub use quant::{
    quantize_k_per_channel, quantize_v_per_token, QuantAxis, QuantConfig, QuantizedTensor,
    MAX_QUANT_DIMENSION, MAX_QUANT_ELEMENTS, MAX_QUANT_GROUP_SIZE,
};
pub use rotation::{
    bind_representation_family, CoefficientSet, DenormalPolicy, F16CacheCast, F32Rounding,
    PhaseOrigin, PositionConvention, RopePairing, RotationFamilyDescriptorV1, RotationFamilyHook,
    RotationOrder, SincosOrder, FIXED_Q30_D7_D6_COEFFICIENT_SHA256, MAX_ROTARY_DIMENSION,
    QWEN25_FRAC64, QWEN25_FRAC64_TABLE_SHA256, QWEN25_ROTARY_DIMENSION, ROTATION_FAMILY_MAGIC,
    ROTATION_FAMILY_VERSION,
};
pub use stats::{
    ChannelRange, SinkScore, StatsSidecar, MAX_SIDECAR_CHANNELS, MAX_SIDECAR_TOKENS,
    MAX_SINK_SCORES,
};
pub use transform::{
    apply_repack_op, apply_transform, inverse_repack_op, CastDType, KvPlaneShape, RepackOp,
    RopeDirection, TransformDescriptor, MAX_PAD_TARGET_BYTES, MAX_PERMUTATION_WIDTH,
    MAX_ROPE_HEAD_DIM, MAX_TRANSFORM_LABEL_BYTES, MAX_TRANSFORM_OPS, TRANSFORM_MAGIC,
    TRANSFORM_VERSION,
};
pub use validator::{validate_family, validate_manifest, ManifestBounds, ValidationContext};

/// 32-byte identifier (SHA-256 digest, HMAC output, or keyed ID).
pub type Id32 = [u8; 32];
