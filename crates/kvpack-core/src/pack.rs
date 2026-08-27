use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::{
    manifest_id, validate_manifest, CutManifest, Id32, KeySchedule, PackError, ValidationContext,
    ALIGNMENT, FLAG_ENCRYPTED, FOOTER_MAGIC, KNOWN_FLAGS, MAX_MANIFEST_BYTES, PACK_FOOTER_BYTES,
    PACK_HEADER_BYTES, PACK_MAGIC, WIRE_VERSION,
};

type HmacSha256 = Hmac<Sha256>;
const HEADER_DIGEST_OFFSET: usize = 148;
const FOOTER_HMAC_OFFSET: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackHeader {
    pub tenant_namespace: Id32,
    pub manifest_id: Id32,
    pub key_epoch: u64,
    pub manifest_bytes: u64,
    pub plaintext_manifest_bytes: u64,
    pub encrypted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPack {
    pub bytes: Vec<u8>,
    pub manifest_id: Id32,
}

fn put_u16(dst: &mut [u8], offset: usize, value: u16) {
    dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(dst: &mut [u8], offset: usize, value: u64) {
    dst[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn get_u16(src: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(src[offset..offset + 2].try_into().unwrap())
}
fn get_u32(src: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(src[offset..offset + 4].try_into().unwrap())
}
fn get_u64(src: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(src[offset..offset + 8].try_into().unwrap())
}

fn manifest_encryption_key(
    keys: &KeySchedule,
    id: &Id32,
    salt: &[u8; 16],
) -> Result<[u8; 32], PackError> {
    let hk = Hkdf::<Sha256>::new(Some(salt), keys.manifest_encryption_key());
    let mut key = [0u8; 32];
    let mut info = Vec::with_capacity(64);
    info.extend_from_slice(b"kvpack/v1/manifest-aead\0");
    info.extend_from_slice(id);
    hk.expand(&info, &mut key)
        .map_err(|_| PackError::Authentication("manifest key derivation failed"))?;
    Ok(key)
}

fn authenticate(
    keys: &KeySchedule,
    header: &[u8],
    manifest: &[u8],
    manifest_len: u64,
    key_epoch: u64,
    file_len: u64,
) -> Id32 {
    let mut mac =
        <HmacSha256 as Mac>::new_from_slice(keys.manifest_auth_key()).expect("fixed HMAC key");
    mac.update(b"kvpack/v1/manifest-auth\0");
    mac.update(header);
    mac.update(manifest);
    mac.update(&manifest_len.to_le_bytes());
    mac.update(&key_epoch.to_le_bytes());
    mac.update(&file_len.to_le_bytes());
    mac.finalize().into_bytes().into()
}

pub fn encode_authenticated_pack(
    manifest: &CutManifest,
    keys: &KeySchedule,
    encrypt: bool,
    context: &ValidationContext,
) -> Result<EncodedPack, PackError> {
    validate_manifest(manifest, context)?;
    let canonical = manifest.encode_canonical()?;
    if canonical.len() > MAX_MANIFEST_BYTES {
        return Err(PackError::Bounds("manifest exceeds production bound"));
    }
    let id = manifest_id(&canonical);
    let stored_len = canonical
        .len()
        .checked_add(if encrypt { 16 } else { 0 })
        .ok_or(PackError::Bounds("manifest length overflow"))?;
    let file_len = PACK_HEADER_BYTES
        .checked_add(stored_len)
        .and_then(|v| v.checked_add(PACK_FOOTER_BYTES))
        .ok_or(PackError::Bounds("pack file length overflow"))?;
    let mut header = vec![0u8; PACK_HEADER_BYTES];
    header[..8].copy_from_slice(PACK_MAGIC);
    put_u16(&mut header, 8, WIRE_VERSION);
    put_u16(&mut header, 10, PACK_HEADER_BYTES as u16);
    put_u32(&mut header, 12, ALIGNMENT as u32);
    put_u32(&mut header, 16, if encrypt { FLAG_ENCRYPTED } else { 0 });
    put_u64(&mut header, 24, stored_len as u64);
    put_u64(&mut header, 32, canonical.len() as u64);
    put_u64(&mut header, 40, manifest.key_epoch);
    header[56..88].copy_from_slice(&manifest.tenant_namespace);
    header[88..120].copy_from_slice(&id);
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 12];
    if encrypt {
        getrandom::fill(&mut salt)
            .map_err(|_| PackError::Authentication("manifest salt generation failed"))?;
        getrandom::fill(&mut nonce)
            .map_err(|_| PackError::Authentication("manifest nonce generation failed"))?;
        header[120..136].copy_from_slice(&salt);
        header[136..148].copy_from_slice(&nonce);
    }
    let digest: Id32 = Sha256::digest(&header).into();
    header[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32].copy_from_slice(&digest);
    let body = if encrypt {
        let key = Zeroizing::new(manifest_encryption_key(keys, &id, &salt)?);
        ChaCha20Poly1305::new(Key::from_slice(key.as_ref()))
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &canonical,
                    aad: &header,
                },
            )
            .map_err(|_| PackError::Authentication("manifest encryption failed"))?
    } else {
        canonical
    };
    let mut footer = vec![0u8; PACK_FOOTER_BYTES];
    footer[..8].copy_from_slice(FOOTER_MAGIC);
    put_u16(&mut footer, 8, WIRE_VERSION);
    put_u16(&mut footer, 10, PACK_FOOTER_BYTES as u16);
    put_u64(&mut footer, 16, body.len() as u64);
    put_u64(&mut footer, 24, manifest.key_epoch);
    put_u64(&mut footer, 40, file_len as u64);
    footer[48..80].copy_from_slice(&id);
    let tag = authenticate(
        keys,
        &header,
        &body,
        body.len() as u64,
        manifest.key_epoch,
        file_len as u64,
    );
    footer[FOOTER_HMAC_OFFSET..FOOTER_HMAC_OFFSET + 32].copy_from_slice(&tag);
    let mut bytes = Vec::with_capacity(file_len);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&body);
    bytes.extend_from_slice(&footer);
    Ok(EncodedPack {
        bytes,
        manifest_id: id,
    })
}

fn parse_pack_header_framing(bytes: &[u8]) -> Result<PackHeader, PackError> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(PackError::Truncated("truncated production pack header"));
    }
    let header = &bytes[..PACK_HEADER_BYTES];
    if &header[..8] != PACK_MAGIC {
        return Err(PackError::BadMagic("invalid production pack magic"));
    }
    if get_u16(header, 8) != WIRE_VERSION
        || get_u16(header, 10) as usize != PACK_HEADER_BYTES
        || get_u32(header, 12) as usize != ALIGNMENT
    {
        return Err(PackError::BadMagic(
            "invalid production pack header contract",
        ));
    }
    let flags = get_u32(header, 16);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(PackError::Reserved("unknown pack flags"));
    }
    if get_u32(header, 20) != 0
        || get_u64(header, 48) != 0
        || header[180..].iter().any(|byte| *byte != 0)
    {
        return Err(PackError::Reserved(
            "pack header reserved bytes are nonzero",
        ));
    }
    let encrypted = flags & FLAG_ENCRYPTED != 0;
    if !encrypted && header[120..148].iter().any(|byte| *byte != 0) {
        return Err(PackError::Reserved(
            "unencrypted manifest salt or nonce is nonzero",
        ));
    }
    Ok(PackHeader {
        tenant_namespace: header[56..88].try_into().unwrap(),
        manifest_id: header[88..120].try_into().unwrap(),
        key_epoch: get_u64(header, 40),
        manifest_bytes: get_u64(header, 24),
        plaintext_manifest_bytes: get_u64(header, 32),
        encrypted,
    })
}

fn verify_pack_header_digest(header: &[u8]) -> Result<(), PackError> {
    let mut copy = header.to_vec();
    let expected: Id32 = copy[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32]
        .try_into()
        .unwrap();
    copy[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32].fill(0);
    let actual: Id32 = Sha256::digest(&copy).into();
    if actual != expected {
        return Err(PackError::Checksum("pack header digest mismatch"));
    }
    Ok(())
}

pub fn inspect_pack_header(bytes: &[u8]) -> Result<PackHeader, PackError> {
    let parsed = parse_pack_header_framing(bytes)?;
    verify_pack_header_digest(&bytes[..PACK_HEADER_BYTES])?;
    Ok(parsed)
}

/// Authenticate a pack's framing, header digest, footer binding, and HMAC
/// without decoding or re-encoding the manifest body.  Callers that cache
/// the decoded form still re-prove the on-disk bytes on every fetch, so
/// on-disk tampering fails closed exactly like the full decode.
pub fn verify_authenticated_pack(
    bytes: &[u8],
    keys: &KeySchedule,
) -> Result<PackHeader, PackError> {
    let minimum = PACK_HEADER_BYTES + PACK_FOOTER_BYTES;
    if bytes.len() < minimum {
        return Err(PackError::Truncated("truncated production pack"));
    }
    let parsed = parse_pack_header_framing(bytes)?;
    if parsed.manifest_bytes as usize > MAX_MANIFEST_BYTES + 16
        || parsed.plaintext_manifest_bytes as usize > MAX_MANIFEST_BYTES
    {
        return Err(PackError::Bounds("manifest exceeds production bound"));
    }
    let footer_offset = bytes.len() - PACK_FOOTER_BYTES;
    let footer = &bytes[footer_offset..];
    if &footer[..8] != FOOTER_MAGIC {
        return Err(PackError::BadMagic(
            "invalid production commit footer magic",
        ));
    }
    if get_u16(footer, 8) != WIRE_VERSION
        || get_u16(footer, 10) as usize != PACK_FOOTER_BYTES
        || get_u32(footer, 12) != 0
        || get_u64(footer, 32) != 0
        || footer[112..].iter().any(|byte| *byte != 0)
    {
        return Err(PackError::Reserved("invalid production commit footer"));
    }
    let body = &bytes[PACK_HEADER_BYTES..footer_offset];
    if body.len() as u64 != parsed.manifest_bytes
        || get_u64(footer, 16) != parsed.manifest_bytes
        || get_u64(footer, 24) != parsed.key_epoch
        || get_u64(footer, 40) != bytes.len() as u64
        || footer[48..80] != parsed.manifest_id
    {
        return Err(PackError::Bounds("pack footer binding mismatch"));
    }
    verify_pack_header_digest(&bytes[..PACK_HEADER_BYTES])?;
    let expected = &footer[FOOTER_HMAC_OFFSET..FOOTER_HMAC_OFFSET + 32];
    let mut verifier =
        <HmacSha256 as Mac>::new_from_slice(keys.manifest_auth_key()).expect("fixed HMAC key");
    verifier.update(b"kvpack/v1/manifest-auth\0");
    verifier.update(&bytes[..PACK_HEADER_BYTES]);
    verifier.update(body);
    verifier.update(&(body.len() as u64).to_le_bytes());
    verifier.update(&parsed.key_epoch.to_le_bytes());
    verifier.update(&(bytes.len() as u64).to_le_bytes());
    verifier
        .verify_slice(expected)
        .map_err(|_| PackError::Authentication("manifest HMAC authentication failed"))?;
    Ok(parsed)
}

pub fn decode_authenticated_pack(
    bytes: &[u8],
    keys: &KeySchedule,
    context: &ValidationContext,
) -> Result<CutManifest, PackError> {
    let parsed = verify_authenticated_pack(bytes, keys)?;
    let footer_offset = bytes.len() - PACK_FOOTER_BYTES;
    let body = &bytes[PACK_HEADER_BYTES..footer_offset];
    let canonical = if parsed.encrypted {
        if body.len() < 16 {
            return Err(PackError::Truncated("truncated encrypted manifest"));
        }
        let salt: [u8; 16] = bytes[120..136].try_into().unwrap();
        let nonce: [u8; 12] = bytes[136..148].try_into().unwrap();
        let key = Zeroizing::new(manifest_encryption_key(keys, &parsed.manifest_id, &salt)?);
        ChaCha20Poly1305::new(Key::from_slice(key.as_ref()))
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: body,
                    aad: &bytes[..PACK_HEADER_BYTES],
                },
            )
            .map_err(|_| PackError::Authentication("manifest AEAD authentication failed"))?
    } else {
        body.to_vec()
    };
    if canonical.len() as u64 != parsed.plaintext_manifest_bytes
        || manifest_id(&canonical) != parsed.manifest_id
    {
        return Err(PackError::Authentication(
            "manifest content identity mismatch",
        ));
    }
    // `decode_canonical` already re-encodes the decoded manifest and
    // byte-compares it against `canonical`, so a second encode here could not
    // reject anything the decode accepted.
    let manifest = CutManifest::decode_canonical(&canonical)?;
    if manifest.tenant_namespace != parsed.tenant_namespace
        || manifest.key_epoch != parsed.key_epoch
    {
        return Err(PackError::Authentication(
            "manifest header identity mismatch",
        ));
    }
    validate_manifest(&manifest, context)?;
    Ok(manifest)
}
