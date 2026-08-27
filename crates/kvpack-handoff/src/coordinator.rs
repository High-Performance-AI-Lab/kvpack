use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::{
    BundleStager, HandoffError, LayerHeaderV1, Result, SealManifestV1, TensorRoleV1,
    ValidationLimits, VerifiedLayerPairV1, VerifiedPlaneV1, VerifiedSealV1,
};

/// Experiment-2's exact two-pair canonical-memory bound at a 32,704-token
/// prompt (`cached_token_count == 32_703`).
pub const EXPERIMENT_TWO_LAYER_CANONICAL_BYTES: u64 = 33_487_872;

#[derive(Debug)]
struct PermitState {
    available: usize,
    in_use: usize,
}

#[derive(Debug)]
struct PermitPoolInner {
    capacity: usize,
    state: Mutex<PermitState>,
    changed: Condvar,
}

/// Global layer-pair permit pool shared by queued and actively consumed work.
#[derive(Clone, Debug)]
pub struct LayerPermitPoolV1 {
    inner: Arc<PermitPoolInner>,
}

impl LayerPermitPoolV1 {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 64 {
            return Err(HandoffError::Validation(
                "layer permit capacity must be in 1..=64".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(PermitPoolInner {
                capacity,
                state: Mutex::new(PermitState {
                    available: capacity,
                    in_use: 0,
                }),
                changed: Condvar::new(),
            }),
        })
    }

    pub fn experiment_v2() -> Self {
        Self::new(2).expect("the fixed experiment capacity is valid")
    }

    pub fn acquire(&self) -> Result<LayerPermitV1> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| HandoffError::Validation("layer permit mutex was poisoned".into()))?;
        while state.available == 0 {
            state =
                self.inner.changed.wait(state).map_err(|_| {
                    HandoffError::Validation("layer permit mutex was poisoned".into())
                })?;
        }
        state.available -= 1;
        state.in_use += 1;
        Ok(LayerPermitV1 {
            inner: Some(Arc::clone(&self.inner)),
        })
    }

    pub fn try_acquire(&self) -> Result<Option<LayerPermitV1>> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| HandoffError::Validation("layer permit mutex was poisoned".into()))?;
        if state.available == 0 {
            return Ok(None);
        }
        state.available -= 1;
        state.in_use += 1;
        Ok(Some(LayerPermitV1 {
            inner: Some(Arc::clone(&self.inner)),
        }))
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn in_use(&self) -> Result<usize> {
        self.inner
            .state
            .lock()
            .map(|state| state.in_use)
            .map_err(|_| HandoffError::Validation("layer permit mutex was poisoned".into()))
    }
}

/// RAII ownership of one canonical K/V pair's receive memory.
#[derive(Debug)]
pub struct LayerPermitV1 {
    inner: Option<Arc<PermitPoolInner>>,
}

impl LayerPermitV1 {
    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        if let Ok(mut state) = inner.state.lock() {
            state.in_use = state.in_use.saturating_sub(1);
            state.available = state.available.saturating_add(1).min(inner.capacity);
            inner.changed.notify_one();
        };
    }
}

impl Drop for LayerPermitV1 {
    fn drop(&mut self) {
        self.release_inner();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLayerFilesV1 {
    key_header: PathBuf,
    key_payload: PathBuf,
    /// `None` for single-role (mla-latent) layers, which stage one plane.
    value_header: Option<PathBuf>,
    value_payload: Option<PathBuf>,
}

impl VerifiedLayerFilesV1 {
    pub fn key_header(&self) -> &Path {
        &self.key_header
    }

    pub fn key_payload(&self) -> &Path {
        &self.key_payload
    }

    pub fn value_header(&self) -> Option<&Path> {
        self.value_header.as_deref()
    }

    pub fn value_payload(&self) -> Option<&Path> {
        self.value_payload.as_deref()
    }
}

/// Emitted exactly once, after both authenticated plane files are durable
/// (after the single plane file, for mla-latent single-role layers).
#[derive(Debug)]
pub struct LayerReadyV1 {
    pair: VerifiedLayerPairV1,
    files: VerifiedLayerFilesV1,
    permit: LayerPermitV1,
    verification_duration_ns: u64,
}

impl LayerReadyV1 {
    pub fn layer(&self) -> u32 {
        self.pair.layer()
    }

    pub fn pair(&self) -> &VerifiedLayerPairV1 {
        &self.pair
    }

    pub fn files(&self) -> &VerifiedLayerFilesV1 {
        &self.files
    }

    pub fn canonical_bytes(&self) -> u64 {
        self.pair.canonical_bytes()
    }

    /// CPU verification plus request-private staging work for both planes.
    pub const fn verification_duration_ns(&self) -> u64 {
        self.verification_duration_ns
    }

    pub fn into_parts(self) -> (VerifiedLayerPairV1, VerifiedLayerFilesV1, LayerPermitV1) {
        (self.pair, self.files, self.permit)
    }
}

struct PendingKey {
    plane: VerifiedPlaneV1,
    files: (PathBuf, PathBuf),
    permit: LayerPermitV1,
    verification_duration_ns: u64,
}

/// Ordered verifier/stager coordinator used by both the diagnostic receiver
/// and the in-process Ferrite qualification worker.
pub struct StreamingCoordinatorV1 {
    stager: BundleStager,
    pending_key: Option<PendingKey>,
    ready_layers: u32,
    terminal_prepared: bool,
}

impl StreamingCoordinatorV1 {
    pub fn create(
        final_path: impl AsRef<Path>,
        begin: crate::BeginManifestV1,
        limits: ValidationLimits,
    ) -> Result<Self> {
        let stager = BundleStager::create(final_path, begin, limits)?;
        Ok(Self {
            stager,
            pending_key: None,
            ready_layers: 0,
            terminal_prepared: false,
        })
    }

    pub fn staging_path(&self) -> &Path {
        self.stager.staging_path()
    }

    pub const fn ready_layers(&self) -> u32 {
        self.ready_layers
    }

    pub fn validate_next_header(&self, header: &LayerHeaderV1) -> Result<()> {
        self.stager.verifier().validate_next_header(header)
    }

    /// Verify and stage one plane. K must carry a permit acquired before its
    /// payload was read; V completes the pending pair and emits one event.
    /// A K plane from a single-role (mla-latent) layout class completes its
    /// layer immediately and emits one event on its own.
    pub fn ingest_plane(
        &mut self,
        header: LayerHeaderV1,
        payload: Vec<u8>,
        permit: Option<LayerPermitV1>,
    ) -> Result<Option<LayerReadyV1>> {
        let verification_started = Instant::now();
        if self.terminal_prepared {
            return Err(HandoffError::Validation(
                "cannot ingest a plane after terminal seal preparation".into(),
            ));
        }
        match header.role {
            TensorRoleV1::Key if permit.is_none() => {
                return Err(HandoffError::Validation(
                    "a K plane payload requires a pre-acquired layer permit".into(),
                ));
            }
            TensorRoleV1::Value if permit.is_some() => {
                return Err(HandoffError::Validation(
                    "a V plane cannot acquire a second layer permit".into(),
                ));
            }
            _ => {}
        }
        let verify_started = std::time::Instant::now();
        let plane = self.stager.verifier_mut().verify_plane(header, payload)?;
        let verify_ns = verify_started.elapsed();
        let stage_started = std::time::Instant::now();
        self.stager.stage_shared_verified(&plane)?;
        let stage_ns = stage_started.elapsed();
        receiver_timing!(
            "[receiver-timing] plane {} {:?} verify {:?} stage {:?}",
            plane.header.layer,
            plane.header.role,
            verify_ns,
            stage_ns
        );
        let header_path = self.stager.staged_header_path(&plane.header).to_path_buf();
        let payload_path = self.stager.staged_payload_path(&plane.header).to_path_buf();
        match plane.header.role {
            TensorRoleV1::Key => {
                // mla-latent single-role frames complete their layer with
                // this one plane: they never enter the pending pair, so two
                // consecutive mla K planes cannot trip the pair cursor.
                if self
                    .stager
                    .verifier()
                    .is_single_role_frame(plane.header.sequence)
                {
                    let pair = VerifiedLayerPairV1::new_single(plane)?;
                    self.check_ready_cursor(pair.layer(), pair.key().header.sequence)?;
                    return Ok(Some(LayerReadyV1 {
                        pair,
                        files: VerifiedLayerFilesV1 {
                            key_header: header_path,
                            key_payload: payload_path,
                            value_header: None,
                            value_payload: None,
                        },
                        permit: permit.expect("K permit was checked"),
                        verification_duration_ns: duration_ns(verification_started.elapsed()),
                    }));
                }
                if self.pending_key.is_some() {
                    return Err(HandoffError::Validation(
                        "received another K plane before its V plane".into(),
                    ));
                }
                self.pending_key = Some(PendingKey {
                    plane,
                    files: (header_path, payload_path),
                    permit: permit.expect("K permit was checked"),
                    verification_duration_ns: duration_ns(verification_started.elapsed()),
                });
                Ok(None)
            }
            TensorRoleV1::Value => {
                let key = self.pending_key.take().ok_or_else(|| {
                    HandoffError::Validation("received V without a pending K plane".into())
                })?;
                let pair = VerifiedLayerPairV1::new(key.plane, plane)?;
                self.check_ready_cursor(pair.layer(), pair.key().header.sequence)?;
                Ok(Some(LayerReadyV1 {
                    pair,
                    files: VerifiedLayerFilesV1 {
                        key_header: key.files.0,
                        key_payload: key.files.1,
                        value_header: Some(header_path),
                        value_payload: Some(payload_path),
                    },
                    permit: key.permit,
                    verification_duration_ns: key
                        .verification_duration_ns
                        .saturating_add(duration_ns(verification_started.elapsed())),
                }))
            }
        }
    }

    /// v1 keeps the strict ascending cursor (layers 0,1,2,…); v2 checks the
    /// completed layer against the declared layout walk at its first-frame
    /// sequence — the per-frame verifier has already enforced the walk, so
    /// this is defense in depth. In both cases `ready_layers` counts
    /// completed layers (pairs, or single-role mla-latent planes).
    fn check_ready_cursor(&mut self, layer: u32, sequence: u32) -> Result<()> {
        let expected_layer = {
            let verifier = self.stager.verifier();
            if verifier.begin().is_v2() {
                verifier.expected_layer_at(sequence)?
            } else {
                self.ready_layers
            }
        };
        if layer != expected_layer {
            return Err(HandoffError::Validation(
                "layer-ready order does not match the strict stream cursor".into(),
            ));
        }
        self.ready_layers = self
            .ready_layers
            .checked_add(1)
            .ok_or_else(|| HandoffError::Validation("layer-ready count overflow".into()))?;
        Ok(())
    }

    pub fn verify_and_prepare_seal(&mut self, seal: SealManifestV1) -> Result<VerifiedSealV1> {
        if self.pending_key.is_some() {
            return Err(HandoffError::Validation(
                "cannot seal with an incomplete K/V layer pair".into(),
            ));
        }
        let witness = self.stager.verifier_mut().verify_seal(seal)?;
        self.stager.prepare_shared_seal(&witness)?;
        self.terminal_prepared = true;
        Ok(witness)
    }

    /// F1: authenticate the just-verified seal under the armed tenant key.
    /// The receiver calls this after [`Self::verify_and_prepare_seal`] when
    /// its config arms a [`crate::MacKey`]; any forged bundle (no key, wrong
    /// key, stripped tag) fails closed before publication.
    pub fn authenticate_seal_hmac(
        &self,
        verified: &VerifiedSealV1,
        key: &crate::MacKey,
    ) -> Result<()> {
        self.stager.verifier().authenticate_seal_hmac(verified, key)
    }

    pub fn publish(&mut self) -> Result<PathBuf> {
        if !self.terminal_prepared {
            return Err(HandoffError::Validation(
                "cannot publish before terminal seal verification".into(),
            ));
        }
        self.stager.publish()
    }

    pub fn abort(&mut self) -> Result<()> {
        self.pending_key = None;
        self.stager.abort()
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permit_pool_counts_active_and_queued_ownership_together() {
        let pool = LayerPermitPoolV1::experiment_v2();
        let first = pool.try_acquire().unwrap().unwrap();
        let second = pool.try_acquire().unwrap().unwrap();
        assert_eq!(pool.in_use().unwrap(), 2);
        assert!(pool.try_acquire().unwrap().is_none());
        drop(first);
        assert_eq!(pool.in_use().unwrap(), 1);
        let third = pool.try_acquire().unwrap().unwrap();
        assert_eq!(pool.in_use().unwrap(), 2);
        drop((second, third));
        assert_eq!(pool.in_use().unwrap(), 0);
    }

    #[test]
    fn frozen_experiment_memory_bound_matches_geometry() {
        let cached_tokens = 32_703u64;
        let plane = cached_tokens * 2 * 64 * 2;
        assert_eq!(2 * 2 * plane, EXPERIMENT_TWO_LAYER_CANONICAL_BYTES);
    }
}
