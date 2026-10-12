use kvpack_core::Id32;
use sha2::{Digest, Sha256};

/// Small canonical framing helper for catalog-only publication intents.
/// These digests fence retries; they are deliberately not durable wire IDs.
pub(crate) struct IntentHasher(Sha256);

impl IntentHasher {
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        Self(digest)
    }

    pub(crate) fn byte(&mut self, value: u8) {
        self.0.update([value]);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    pub(crate) fn id(&mut self, value: &Id32) {
        self.0.update(value);
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.0.update(value);
    }

    pub(crate) fn finish(self) -> Id32 {
        self.0.finalize().into()
    }
}
