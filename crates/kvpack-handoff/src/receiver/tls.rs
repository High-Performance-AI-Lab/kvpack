use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{RootCertStore, ServerConfig, ServerConnection, StreamOwned};
use sha2::{Digest, Sha256};

use super::config::ReceiverConfigV1;
use crate::{HandoffError, Result};

pub const LIVE_HANDOFF_ALPN_V1: &[u8] = b"kvpack-handoff/1";

/// WS9 transport squeeze: the direct link pairs an M3 Ultra with a GB10;
/// both have hardware AES, so AES-256-GCM is offered first, ahead of the
/// ChaCha20-Poly1305 and AES-128-GCM TLS 1.3 fallbacks. rustls picks the
/// negotiated suite by server preference, so this order is authoritative.
fn handoff_cipher_suites() -> Vec<rustls::SupportedCipherSuite> {
    vec![
        rustls::crypto::ring::cipher_suite::TLS13_AES_256_GCM_SHA384,
        rustls::crypto::ring::cipher_suite::TLS13_CHACHA20_POLY1305_SHA256,
        rustls::crypto::ring::cipher_suite::TLS13_AES_128_GCM_SHA256,
    ]
}

fn handoff_crypto_provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::CryptoProvider {
        cipher_suites: handoff_cipher_suites(),
        ..rustls::crypto::ring::default_provider()
    }
}

pub(super) fn tls_server_config(config: &ReceiverConfigV1) -> Result<Arc<ServerConfig>> {
    let certificates = load_certificates(&config.server_cert)?;
    let private_key = load_private_key(&config.server_key)?;
    let mut roots = RootCertStore::empty();
    for certificate in load_certificates(&config.client_ca)? {
        roots.add(certificate).map_err(|error| {
            HandoffError::Validation(format!("client CA certificate is invalid: {error}"))
        })?;
    }
    let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots))
        .build()
        .map_err(|error| HandoffError::Validation(format!("build mTLS verifier: {error}")))?;
    let mut tls = ServerConfig::builder_with_provider(Arc::new(handoff_crypto_provider()))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| HandoffError::Validation(format!("pin TLS 1.3: {error}")))?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certificates, private_key)
        .map_err(|error| HandoffError::Validation(format!("build TLS identity: {error}")))?;
    tls.alpn_protocols = vec![LIVE_HANDOFF_ALPN_V1.to_vec()];
    Ok(Arc::new(tls))
}

fn load_certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let certificates = rustls_pemfile::certs(&mut reader).collect::<std::io::Result<Vec<_>>>()?;
    if certificates.is_empty() {
        return Err(HandoffError::Validation(
            "certificate file contains no certificates".into(),
        ));
    }
    Ok(certificates)
}

/// SHA-256 fingerprint of the first DER certificate used as a TLS leaf.
///
/// Product controllers can bind readiness evidence to the same normalized
/// bytes rustls presents on the authenticated connection.
pub fn certificate_leaf_sha256_v1(path: &Path) -> Result<String> {
    let certificates = load_certificates(path)?;
    Ok(hex::encode(Sha256::digest(
        certificates
            .first()
            .expect("load_certificates rejects an empty chain")
            .as_ref(),
    )))
}

fn load_private_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    let mut reader = BufReader::new(File::open(path)?);
    rustls_pemfile::private_key(&mut reader)?
        .ok_or_else(|| HandoffError::Validation("private-key file contains no private key".into()))
}

pub(super) fn verify_tls_peer(
    stream: &StreamOwned<ServerConnection, std::net::TcpStream>,
    expected_sha256: &str,
) -> Result<()> {
    if stream.conn.alpn_protocol() != Some(LIVE_HANDOFF_ALPN_V1) {
        return Err(HandoffError::Validation(
            "TLS session did not negotiate kvpack-handoff/1".into(),
        ));
    }
    let certificate = stream
        .conn
        .peer_certificates()
        .and_then(|certificates| certificates.first())
        .ok_or_else(|| {
            HandoffError::Validation("mTLS peer did not present a leaf certificate".into())
        })?;
    let actual = hex::encode(Sha256::digest(certificate.as_ref()));
    if actual != expected_sha256 {
        return Err(HandoffError::Validation(
            "mTLS peer certificate SHA-256 mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::CipherSuite;

    #[test]
    fn cipher_suites_prefer_aes256_gcm_with_tls13_chacha_fallback() {
        let suites = handoff_cipher_suites();
        assert_eq!(
            suites
                .iter()
                .map(rustls::SupportedCipherSuite::suite)
                .collect::<Vec<_>>(),
            vec![
                CipherSuite::TLS13_AES_256_GCM_SHA384,
                CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
                CipherSuite::TLS13_AES_128_GCM_SHA256,
            ]
        );
    }

    #[test]
    fn provider_keeps_tls13_only_and_ring_signing() {
        let provider = handoff_crypto_provider();
        assert!(provider.cipher_suites.iter().all(|suite| matches!(
            suite.suite(),
            CipherSuite::TLS13_AES_256_GCM_SHA384
                | CipherSuite::TLS13_CHACHA20_POLY1305_SHA256
                | CipherSuite::TLS13_AES_128_GCM_SHA256
        )));
        assert!(!provider.signature_verification_algorithms.all.is_empty());
    }
}
