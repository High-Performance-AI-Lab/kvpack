use crate::{BeginManifestV1, LayerReadyV1, Result, VerifiedSealV1};

/// Interactive sink boundary. Implementations may move `LayerReadyV1` into a
/// blocking `sync_channel(2)`; its permit then remains held through active
/// Ferrite work as well as queue residence.
pub trait ReceiverSinkV1 {
    fn begin(&mut self, begin: &BeginManifestV1) -> Result<()>;
    fn layer_ready(&mut self, layer: LayerReadyV1) -> Result<()>;
    fn seal_verified(&mut self, seal: &VerifiedSealV1) -> Result<()>;
    fn abort(&mut self);
}

#[derive(Default)]
pub struct BundleOnlyReceiverSinkV1;

impl ReceiverSinkV1 for BundleOnlyReceiverSinkV1 {
    fn begin(&mut self, _begin: &BeginManifestV1) -> Result<()> {
        Ok(())
    }

    fn layer_ready(&mut self, _layer: LayerReadyV1) -> Result<()> {
        Ok(())
    }

    fn seal_verified(&mut self, _seal: &VerifiedSealV1) -> Result<()> {
        Ok(())
    }

    fn abort(&mut self) {}
}
