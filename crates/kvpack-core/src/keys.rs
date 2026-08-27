use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{Id32, PackError};

/// HKDF-separated tenant/epoch keys.  It cannot be cloned and all key bytes
/// are zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KeySchedule {
    namespace: [u8; 32],
    prefix: [u8; 32],
    manifest_auth: [u8; 32],
    manifest_encryption: [u8; 32],
    chunk_identity: [u8; 32],
    object_identity: [u8; 32],
    chunk_encryption: [u8; 32],
}

impl KeySchedule {
    pub fn derive(root: &[u8; 32], tenant: &Id32, epoch: u64) -> Result<Self, PackError> {
        Self::derive_with_epoch_root(root, root, tenant, epoch)
    }

    pub fn derive_with_epoch_root(
        stable_root: &[u8; 32],
        epoch_root: &[u8; 32],
        tenant: &Id32,
        epoch: u64,
    ) -> Result<Self, PackError> {
        if epoch == 0 {
            return Err(PackError::Semantics("key epoch must be nonzero"));
        }
        let stable = Hkdf::<Sha256>::new(Some(tenant), stable_root);
        let mut salt = [0u8; 40];
        salt[..32].copy_from_slice(tenant);
        salt[32..].copy_from_slice(&epoch.to_le_bytes());
        let epoch_keys = Hkdf::<Sha256>::new(Some(&salt), epoch_root);
        fn expand(hk: &Hkdf<Sha256>, label: &[u8]) -> Result<[u8; 32], PackError> {
            let mut out = [0u8; 32];
            hk.expand(label, &mut out)
                .map_err(|_| PackError::Authentication("HKDF expansion failed"))?;
            Ok(out)
        }
        Ok(Self {
            namespace: expand(&stable, b"kvpack/v1/stable/namespace")?,
            prefix: expand(&stable, b"kvpack/v1/stable/prefix")?,
            chunk_identity: expand(&stable, b"kvpack/v1/stable/chunk-identity")?,
            manifest_auth: expand(&epoch_keys, b"kvpack/v1/epoch/manifest-auth")?,
            manifest_encryption: expand(&epoch_keys, b"kvpack/v1/epoch/manifest-encryption")?,
            object_identity: expand(&epoch_keys, b"kvpack/v1/epoch/object-identity")?,
            chunk_encryption: expand(&epoch_keys, b"kvpack/v1/epoch/chunk-encryption")?,
        })
    }

    pub fn namespace_key(&self) -> &[u8; 32] {
        &self.namespace
    }
    pub fn prefix_key(&self) -> &[u8; 32] {
        &self.prefix
    }
    pub fn manifest_auth_key(&self) -> &[u8; 32] {
        &self.manifest_auth
    }
    pub fn manifest_encryption_key(&self) -> &[u8; 32] {
        &self.manifest_encryption
    }
    pub fn chunk_identity_key(&self) -> &[u8; 32] {
        &self.chunk_identity
    }
    pub fn object_identity_key(&self) -> &[u8; 32] {
        &self.object_identity
    }
    pub fn chunk_encryption_key(&self) -> &[u8; 32] {
        &self.chunk_encryption
    }
}
