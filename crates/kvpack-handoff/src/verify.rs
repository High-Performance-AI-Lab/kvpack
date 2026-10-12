use sha2::{Digest, Sha256};

use crate::{
    artifact_sha256, canonical_json, descriptor_chain_sha256, sha256_hex, BeginManifestV1,
    HandoffError, LayerHeaderV1, Result, SealManifestV1, TensorRoleV1, ValidationLimits,
};

/// One payload whose descriptor, bounds, ordering, and SHA-256 authenticated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPlaneV1 {
    pub header: LayerHeaderV1,
    pub bytes: Vec<u8>,
}

/// The exact K/V pair for one layer, in canonical token-major F16LE form.
///
/// mla-latent layout classes ship a single packed latent plane per layer
/// (roles `[key]`), so the value plane is optional: pairs built with
/// `VerifiedLayerPairV1::new_single` carry only the key-role plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLayerPairV1 {
    layer: u32,
    key: VerifiedPlaneV1,
    value: Option<VerifiedPlaneV1>,
}

impl VerifiedLayerPairV1 {
    pub(crate) fn new(key: VerifiedPlaneV1, value: VerifiedPlaneV1) -> Result<Self> {
        if key.header.layer != value.header.layer
            || key.header.role != TensorRoleV1::Key
            || value.header.role != TensorRoleV1::Value
            || value.header.sequence != key.header.sequence.saturating_add(1)
        {
            return Err(HandoffError::Validation(
                "verified layer pair is not one adjacent K-then-V pair".into(),
            ));
        }
        Ok(Self {
            layer: key.header.layer,
            key,
            value: Some(value),
        })
    }

    /// One complete single-role layer: the sole (key-role) plane of an
    /// mla-latent layout class.
    pub(crate) fn new_single(key: VerifiedPlaneV1) -> Result<Self> {
        if key.header.role != TensorRoleV1::Key {
            return Err(HandoffError::Validation(
                "verified single-role layer is not a key-role plane".into(),
            ));
        }
        Ok(Self {
            layer: key.header.layer,
            key,
            value: None,
        })
    }

    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub fn key(&self) -> &VerifiedPlaneV1 {
        &self.key
    }

    /// The value plane of a K/V pair; `None` for single-role (mla-latent)
    /// layers, which ship one packed latent plane.
    pub fn value(&self) -> Option<&VerifiedPlaneV1> {
        self.value.as_ref()
    }

    pub fn into_planes(self) -> (VerifiedPlaneV1, Option<VerifiedPlaneV1>) {
        (self.key, self.value)
    }

    pub fn canonical_bytes(&self) -> u64 {
        self.value
            .as_ref()
            .map_or(self.key.header.byte_length, |value| {
                self.key
                    .header
                    .byte_length
                    .saturating_add(value.header.byte_length)
            })
    }
}

/// Opaque proof that the terminal seal authenticated the complete stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSealV1 {
    pub(crate) seal: SealManifestV1,
    pub(crate) transfer_id: String,
    pub(crate) descriptor_chain_sha256: String,
    pub(crate) payload_sha256: String,
}

impl VerifiedSealV1 {
    pub fn artifact_sha256(&self) -> &str {
        &self.seal.artifact_sha256
    }

    pub fn transfer_id(&self) -> &str {
        &self.transfer_id
    }

    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.seal.core.prompt_token_ids
    }

    pub(crate) fn manifest(&self) -> &SealManifestV1 {
        &self.seal
    }
}

/// Rollback-safe verifier for one v1 BEGIN → 2L planes → SEAL stream.
///
/// All fallible validation and hash work is performed on temporary state.
/// The cursor and rolling hashes advance only after the complete plane passes.
pub struct IncrementalVerifierV1 {
    begin: BeginManifestV1,
    limits: ValidationLimits,
    /// v2: the flat (class, layer, role) walk precomputed once when the
    /// begin was validated; per-plane validation then indexes it in O(1).
    layout_walk: Option<Vec<(u32, u32, TensorRoleV1)>>,
    headers: Vec<LayerHeaderV1>,
    payload_bytes: u64,
    payload_hash: Sha256,
    descriptor_hash: Sha256,
    artifact_prefix_hash: Sha256,
    sealed: bool,
}

impl IncrementalVerifierV1 {
    pub fn new(begin: BeginManifestV1, limits: ValidationLimits) -> Result<Self> {
        begin.validate(&limits)?;
        let layout_walk = begin.is_v2().then(|| begin.layout_walk_v2());
        let mut descriptor_hash = Sha256::new();
        descriptor_hash.update(b"kvpack-live-descriptor-chain-v1\0");
        let mut artifact_prefix_hash = Sha256::new();
        artifact_prefix_hash.update(b"kvpack-live-artifact-v1\0");
        artifact_prefix_hash.update(canonical_json(&begin)?);
        artifact_prefix_hash.update(b"\n");
        Ok(Self {
            begin,
            limits,
            layout_walk,
            headers: Vec::new(),
            payload_bytes: 0,
            payload_hash: Sha256::new(),
            descriptor_hash,
            artifact_prefix_hash,
            sealed: false,
        })
    }

    pub fn begin(&self) -> &BeginManifestV1 {
        &self.begin
    }

    /// v2: the layer declared at `sequence` in the precomputed walk.
    pub(crate) fn expected_layer_at(&self, sequence: u32) -> Result<u32> {
        let walk = self
            .layout_walk
            .as_ref()
            .ok_or_else(|| HandoffError::Validation("a v1 begin declares no layout walk".into()))?;
        walk.get(sequence as usize)
            .map(|&(_, layer, _)| layer)
            .ok_or_else(|| {
                HandoffError::Validation(format!(
                    "layer frame {sequence} is outside the declared layout table"
                ))
            })
    }

    /// v2: whether the frame at `sequence` belongs to a single-role layout
    /// class (mla-latent), whose layer completes with that one plane.
    pub(crate) fn is_single_role_frame(&self, sequence: u32) -> bool {
        self.layout_walk
            .as_ref()
            .and_then(|walk| walk.get(sequence as usize))
            .is_some_and(|&(class_idx, _, _)| {
                self.begin.layout_table[class_idx as usize].roles.len() == 1
            })
    }

    pub fn next_sequence(&self) -> u32 {
        u32::try_from(self.headers.len()).unwrap_or(u32::MAX)
    }

    pub fn headers(&self) -> &[LayerHeaderV1] {
        &self.headers
    }

    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub const fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// F1: authenticate the verified seal under the armed tenant key. The
    /// verifier already proved the seal matches begin + headers + core, so
    /// this recomputes the keyed tag over those authenticated manifests.
    /// Called by the receiver/engine after [`Self::verify_seal`] when the
    /// deployment arms artifact-level authentication.
    pub fn authenticate_seal_hmac(
        &self,
        verified: &VerifiedSealV1,
        key: &crate::MacKey,
    ) -> Result<()> {
        verified
            .manifest()
            .authenticate_hmac(&self.begin, &self.headers, key)
    }

    /// Validate the next descriptor without advancing any rolling state.
    pub fn validate_next_header(&self, header: &LayerHeaderV1) -> Result<()> {
        if self.sealed {
            return Err(HandoffError::Validation(
                "cannot validate a layer header after the terminal seal".into(),
            ));
        }
        let sequence = self.next_sequence();
        if let Some(walk) = &self.layout_walk {
            let entry = walk.get(sequence as usize).copied().ok_or_else(|| {
                HandoffError::Validation(format!(
                    "layer frame {sequence} is outside the declared layout table"
                ))
            })?;
            let expected = self.begin.expected_from_walk_entry(entry);
            header.validate_for_v2_expected(&self.begin, sequence, &expected)?;
        } else {
            header.validate_for(&self.begin, sequence)?;
        }
        if header.byte_length > self.limits.max_frame_bytes {
            return Err(HandoffError::Validation(
                "layer descriptor exceeds the configured frame bound".into(),
            ));
        }
        Ok(())
    }

    /// Verify the next plane, taking ownership of its payload so the
    /// authenticated bytes move into the returned plane without a copy.
    pub fn verify_plane(
        &mut self,
        header: LayerHeaderV1,
        payload: Vec<u8>,
    ) -> Result<VerifiedPlaneV1> {
        if self.sealed {
            return Err(HandoffError::Validation(
                "cannot verify a layer plane after the terminal seal".into(),
            ));
        }
        let sequence = u32::try_from(self.headers.len())
            .map_err(|_| HandoffError::Validation("layer sequence exceeds u32".into()))?;
        self.validate_next_header(&header)?;
        if header.byte_length > self.limits.max_frame_bytes
            || u64::try_from(payload.len()).ok() != Some(header.byte_length)
            || sha256_hex(&payload) != header.sha256
        {
            return Err(HandoffError::Validation(format!(
                "layer frame {sequence} length or SHA-256 mismatch"
            )));
        }
        // F4: a pre-RoPE Key plane is f32-LE; reject any non-finite
        // element so an authenticated but NaN/Inf-bearing plane never
        // reaches the consumer's f16 cache. SHA authenticates bytes, not
        // values, so this value gate is mandatory and separate.
        if self.begin.is_prerope_v2() && header.role == TensorRoleV1::Key {
            crate::manifest::validate_prerope_key_plane_finite(&payload)?;
        }
        let next_payload_bytes = self
            .payload_bytes
            .checked_add(header.byte_length)
            .ok_or_else(|| HandoffError::Validation("payload byte count overflow".into()))?;
        if next_payload_bytes > self.limits.max_total_bytes
            || next_payload_bytes > self.begin.expected_payload_bytes
        {
            return Err(HandoffError::Validation(
                "received payload exceeds its declared or configured total bound".into(),
            ));
        }

        let encoded_header = canonical_json(&header)?;
        let mut payload_hash = self.payload_hash.clone();
        payload_hash.update(&payload);
        let mut descriptor_hash = self.descriptor_hash.clone();
        descriptor_hash.update(&encoded_header);
        descriptor_hash.update(b"\n");
        let mut artifact_prefix_hash = self.artifact_prefix_hash.clone();
        artifact_prefix_hash.update(&encoded_header);
        artifact_prefix_hash.update(b"\n");

        self.payload_hash = payload_hash;
        self.descriptor_hash = descriptor_hash;
        self.artifact_prefix_hash = artifact_prefix_hash;
        self.payload_bytes = next_payload_bytes;
        self.headers.push(header.clone());
        Ok(VerifiedPlaneV1 {
            header,
            bytes: payload,
        })
    }

    pub fn verify_seal(&mut self, seal: SealManifestV1) -> Result<VerifiedSealV1> {
        if self.sealed {
            return Err(HandoffError::Validation(
                "terminal seal was already verified".into(),
            ));
        }
        let payload_sha256 = hex::encode(self.payload_hash.clone().finalize());
        let descriptor_sha256 = hex::encode(self.descriptor_hash.clone().finalize());
        if descriptor_sha256 != descriptor_chain_sha256(&self.headers)? {
            return Err(HandoffError::Validation(
                "incremental descriptor chain disagrees with compatibility verifier".into(),
            ));
        }
        seal.validate_for(
            &self.begin,
            &self.headers,
            self.payload_bytes,
            &payload_sha256,
        )?;
        let mut artifact_hash = self.artifact_prefix_hash.clone();
        artifact_hash.update(canonical_json(&seal.core)?);
        let incremental_artifact = hex::encode(artifact_hash.finalize());
        if incremental_artifact != seal.artifact_sha256
            || incremental_artifact != artifact_sha256(&self.begin, &self.headers, &seal.core)?
        {
            return Err(HandoffError::Validation(
                "incremental artifact identity disagrees with terminal seal".into(),
            ));
        }
        self.sealed = true;
        Ok(VerifiedSealV1 {
            transfer_id: self.begin.transfer_id.clone(),
            descriptor_chain_sha256: descriptor_sha256,
            payload_sha256,
            seal,
        })
    }
}
