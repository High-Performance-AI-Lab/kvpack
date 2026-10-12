//! Versioned, engine-neutral framing for live prefill KV handoff.
//!
//! This crate authenticates no peer and authorizes no restore by itself.
//! Callers must place the byte stream inside an authenticated transport and
//! independently match the exact model/tokenizer/template/engine identities.
//! Incomplete sessions are never represented by [`VerifiedBundle`].

pub(crate) fn receiver_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("KVPACK_RECEIVER_TIMING").is_some())
}

macro_rules! receiver_timing {
    ($($arg:tt)*) => {
        if crate::receiver_timing_enabled() {
            eprintln!($($arg)*);
        }
    };
}

mod bundle;
mod canonical;
mod coordinator;
mod error;
mod frame;
mod handoff_v2;
mod mac;
mod manifest;
#[cfg(feature = "receiver")]
mod receiver;
mod verify;

pub use bundle::{BundleStager, MaterializedVerifiedBundle, VerifiedBundle, VerifiedPlane};
pub use canonical::{canonical_json, decode_canonical_json, sha256_hex, token_ids_sha256};
pub use coordinator::{
    LayerPermitPoolV1, LayerPermitV1, LayerReadyV1, StreamingCoordinatorV1, VerifiedLayerFilesV1,
    EXPERIMENT_TWO_LAYER_CANONICAL_BYTES,
};
pub use error::{HandoffError, Result};
pub use frame::{
    read_frame, write_frame, Frame, FrameHeader, FrameKind, FrameLimits, FrameReader,
    FRAME_HEADER_BYTES,
};
pub use handoff_v2::{
    AtomicReceiverV2, BeginManifestV2, CommittedGeneration, ComponentKindV2, ComponentV2,
    HandoffSinkV2, HmacIdentityV2, MultimodalIdentityV2, SealCoreV2, SealManifestV2,
    SegmentDescriptorV2, SegmentRoleV2, ValidatedBeginV2, VerifiedSealV2, VerifiedSegmentV2,
    LIVE_HANDOFF_PROTOCOL_V2,
};
pub use mac::MacKey;
pub use manifest::{
    artifact_hmac_sha256, artifact_mac_stream, artifact_sha256, descriptor_chain_sha256,
    AbortManifestV1, AckManifestV1, BeginManifestV1, CanaryRecord, EndpointIdentityV1,
    ExactIdentityV1, GeometryV1, HandoffStrategyV1, LayerHeaderV1, LayoutClassV2, PrecisionV1,
    SealCoreV1, SealManifestV1, TensorRoleV1, ValidationLimits, FRAME_MAGIC,
    LIVE_HANDOFF_PROTOCOL_V1, LIVE_HANDOFF_SCHEMA_V1, PORTABLE_KV_ABI_V1, PORTABLE_KV_ABI_V2,
    PORTABLE_KV_ABI_V2_PREROPE, WIRE_SCHEDULE_DECODE_PRIORITY, WIRE_SCHEDULE_LAYER_ORDER,
};
#[cfg(feature = "receiver")]
pub use receiver::{
    certificate_leaf_sha256_v1, receive_one_v1, receive_one_v1_cancellable,
    receive_one_v1_cancellable_with_ready, receive_one_v1_with_ready, BundleOnlyReceiverSinkV1,
    ReceiverBeginExpectationsV1, ReceiverConfigV1, ReceiverInterruptV1, ReceiverReceiptV1,
    ReceiverSessionStateV1, ReceiverSinkV1, LIVE_HANDOFF_ALPN_V1,
};
pub use verify::{IncrementalVerifierV1, VerifiedLayerPairV1, VerifiedPlaneV1, VerifiedSealV1};
