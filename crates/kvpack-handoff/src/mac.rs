//! F1 (red-team 2026-08-09): keyed artifact authentication.
//!
//! `kvpack-handoff` authenticates no peer by itself (see `lib.rs`): a live
//! handoff rides inside an authenticated transport (mTLS), and the plain
//! SHA-256 in the manifest is integrity-only. That is enough when the only
//! path a bundle ever takes is the authenticated receiver. It is not enough
//! once a sealed bundle can also reach an engine through a local file, a
//! cache/locality hop, or a future network API that skips re-auth: there the
//! whole artifact — label, identity, planes, seal — is forgeable without a
//! key.
//!
//! [`MacKey`] is the optional close. A deployment that needs artifact-level
//! authentication arms a 256-bit tenant key; the producer tags the artifact
//! with [`crate::artifact_hmac_sha256`] and the consumer rejects any sealed
//! bundle whose tag does not verify under the armed key. The tag covers the
//! same begin + headers + core stream as `artifact_sha256`, domain-separated
//! so a plain SHA-256 can never be replayed as a keyed tag. The key
//! zeroizes on drop.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::{HandoffError, Result};

type HmacSha256 = Hmac<Sha256>;

/// A 256-bit symmetric MAC key for artifact authentication. Clone for
/// arming multiple receivers from one provisioned key; the underlying
/// bytes zeroize when every clone drops.
#[derive(Clone)]
pub struct MacKey(Zeroizing<[u8; 32]>);

impl std::fmt::Debug for MacKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacKey").finish_non_exhaustive()
    }
}

impl MacKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// Build a key from a 32-byte slice; any other length fails closed.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let key: [u8; 32] = bytes
            .try_into()
            .map_err(|_| HandoffError::Validation("MAC key must be exactly 32 bytes".into()))?;
        Ok(Self::from_bytes(key))
    }

    /// Decode a 64-character lowercase-hex key.
    pub fn from_hex(hex_key: &str) -> Result<Self> {
        if hex_key.len() != 64
            || !hex_key
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(HandoffError::Validation(
                "MAC key must be 64 lowercase hexadecimal digits".into(),
            ));
        }
        let mut bytes = [0u8; 32];
        hex::decode_to_slice(hex_key, &mut bytes)
            .map_err(|_| HandoffError::Validation("MAC key hex decode failed".into()))?;
        Ok(Self::from_bytes(bytes))
    }

    /// Lowercase-hex HMAC-SHA256 tag over the supplied byte stream. Used by
    /// the producer to stamp `artifact_hmac_sha256`.
    pub(crate) fn tag_hex(&self, stream: &[u8]) -> Result<String> {
        let mut mac = HmacSha256::new_from_slice(self.0.as_ref())
            .map_err(|_| HandoffError::Validation("HMAC key length rejected".into()))?;
        mac.update(stream);
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    /// Constant-time verification of an expected lowercase-hex tag. Used by
    /// the consumer; any mismatch fails closed.
    pub(crate) fn verify_hex(&self, stream: &[u8], expected_hex: &str) -> Result<()> {
        let mut mac = HmacSha256::new_from_slice(self.0.as_ref())
            .map_err(|_| HandoffError::Validation("HMAC key length rejected".into()))?;
        mac.update(stream);
        let expected = hex::decode(expected_hex)
            .map_err(|_| HandoffError::Validation("artifact HMAC tag is not valid hex".into()))?;
        mac.verify_slice(&expected)
            .map_err(|_| HandoffError::Validation("artifact HMAC tag rejected".into()))
    }

    /// Domain-separated HMAC-SHA256 for an authenticated protocol layered on
    /// kvpack's content-addressed objects.
    ///
    /// A content digest is not an authorization proof.  Verifier rounds,
    /// snapshot catalog entries, and other protocol records can use this
    /// method without reusing the artifact-seal MAC domain.  The explicit
    /// domain length makes the transcript prefix unambiguous.
    pub fn tag_domain_hex(&self, domain: &[u8], stream: &[u8]) -> Result<String> {
        if domain.is_empty() || domain.len() > u16::MAX as usize {
            return Err(HandoffError::Validation(
                "MAC domain must contain 1..=65535 bytes".into(),
            ));
        }
        let mut mac = HmacSha256::new_from_slice(self.0.as_ref())
            .map_err(|_| HandoffError::Validation("HMAC key length rejected".into()))?;
        mac.update(b"kvpack-domain-mac-v1\0");
        mac.update(&(domain.len() as u16).to_be_bytes());
        mac.update(domain);
        mac.update(stream);
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    /// Constant-time verification for [`Self::tag_domain_hex`].
    pub fn verify_domain_hex(
        &self,
        domain: &[u8],
        stream: &[u8],
        expected_hex: &str,
    ) -> Result<()> {
        if domain.is_empty() || domain.len() > u16::MAX as usize {
            return Err(HandoffError::Validation(
                "MAC domain must contain 1..=65535 bytes".into(),
            ));
        }
        if expected_hex.len() != 64
            || !expected_hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(HandoffError::Validation(
                "protocol HMAC tag must be 64 lowercase hex characters".into(),
            ));
        }
        let expected = hex::decode(expected_hex)
            .map_err(|_| HandoffError::Validation("protocol HMAC tag is not valid hex".into()))?;
        let mut mac = HmacSha256::new_from_slice(self.0.as_ref())
            .map_err(|_| HandoffError::Validation("HMAC key length rejected".into()))?;
        mac.update(b"kvpack-domain-mac-v1\0");
        mac.update(&(domain.len() as u16).to_be_bytes());
        mac.update(domain);
        mac.update(stream);
        mac.verify_slice(&expected)
            .map_err(|_| HandoffError::Validation("protocol HMAC tag rejected".into()))
    }
}

#[cfg(test)]
mod tests {
    use sha2::Digest;

    use super::*;

    #[test]
    fn wrong_key_rejects_a_tag_and_right_key_accepts_it() {
        let producer = MacKey::from_hex(&"a".repeat(64)).unwrap();
        let other = MacKey::from_hex(&"b".repeat(64)).unwrap();
        let stream = b"authenticated artifact stream";
        let tag = producer.tag_hex(stream).unwrap();
        // A plain SHA-256 of the stream is never accepted as a tag.
        assert!(producer
            .verify_hex(stream, &hex::encode(sha2::Sha256::digest(stream)))
            .is_err());
        // A different key rejects the producer's tag.
        assert!(other.verify_hex(stream, &tag).is_err());
        // The producer's own key accepts it.
        producer.verify_hex(stream, &tag).unwrap();
        // Any tamper of the stream rejects under the producer's key.
        let mut tampered = stream.to_vec();
        tampered[0] ^= 0xff;
        assert!(producer.verify_hex(&tampered, &tag).is_err());
    }

    #[test]
    fn non_32_byte_keys_and_bad_hex_fail_closed() {
        assert!(MacKey::from_slice(&[0u8; 31]).is_err());
        assert!(MacKey::from_slice(&[0u8; 33]).is_err());
        assert!(MacKey::from_hex(&"a".repeat(63)).is_err());
        assert!(MacKey::from_hex("Z".repeat(64).as_str()).is_err());
    }

    #[test]
    fn protocol_tags_are_domain_separated_and_constant_time_verified() {
        let key = MacKey::from_bytes([0x5a; 32]);
        let tag = key
            .tag_domain_hex(b"muser-verifier-round-v1", b"round transcript")
            .unwrap();
        key.verify_domain_hex(b"muser-verifier-round-v1", b"round transcript", &tag)
            .unwrap();
        assert!(key
            .verify_domain_hex(b"muser-verifier-session-v1", b"round transcript", &tag)
            .is_err());
        assert!(key
            .verify_domain_hex(b"muser-verifier-round-v1", b"changed", &tag)
            .is_err());
        assert!(key
            .verify_domain_hex(
                b"muser-verifier-round-v1",
                b"round transcript",
                &tag.to_uppercase(),
            )
            .is_err());
        assert!(key
            .verify_domain_hex(b"muser-verifier-round-v1", b"round transcript", "00")
            .is_err());
        assert!(key.tag_domain_hex(b"", b"round transcript").is_err());
    }
}
