use std::borrow::Cow;

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::{
    chunk_id, representation_family_id, ChunkRef, ChunkSpan, Codec, Id32, KeySchedule, PackError,
    RepresentationFamilyId, StateKey, StatsSidecar, ALIGNMENT, CHUNK_HEADER_BYTES,
    CHUNK_HEADER_SIDECAR_OFFSET, CHUNK_MAGIC, CODEC_FRAME_HEADER_BYTES, FLAG_ENCRYPTED,
    KNOWN_FLAGS, LOSSLESS_FRAME_MAGIC, MAX_CHUNK_OBJECT_BYTES, MAX_CHUNK_PLAINTEXT,
    MAX_CODEC_OVERHEAD, MAX_STATS_SIDECAR_BYTES, RAW_FRAME_MAGIC, WIRE_VERSION,
};

type HmacSha256 = Hmac<Sha256>;
const HEADER_DIGEST_OFFSET: usize = 204;

#[derive(Debug, Clone, Copy)]
pub struct ChunkEncoding<'a> {
    pub tenant_namespace: Id32,
    pub family: &'a RepresentationFamilyId,
    pub state_key: &'a StateKey,
    pub span: ChunkSpan,
    pub key_epoch: u64,
    pub encrypt: bool,
    /// Optional M7 attention-statistics sidecar.  When present it is written
    /// into the chunk header tail and hashed into the object identity; when
    /// absent the encoded object is byte-identical to the pre-sidecar format.
    pub stats_sidecar: Option<&'a StatsSidecar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkObject {
    pub chunk_id: Id32,
    pub object_key: Id32,
    pub object_digest: Id32,
    pub plaintext_bytes: u32,
    pub bytes: Vec<u8>,
}

fn get_u16(source: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        source[offset..offset + 2]
            .try_into()
            .expect("checked header"),
    )
}

fn get_u32(source: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        source[offset..offset + 4]
            .try_into()
            .expect("checked header"),
    )
}

fn get_u64(source: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        source[offset..offset + 8]
            .try_into()
            .expect("checked header"),
    )
}

fn align(value: usize) -> Result<usize, PackError> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|rounded| rounded & !(ALIGNMENT - 1))
        .ok_or(PackError::Bounds("chunk object length overflow"))
}

fn family_state<'a>(
    family: &'a RepresentationFamilyId,
    key: &StateKey,
) -> Result<&'a crate::FamilyState, PackError> {
    family
        .states
        .binary_search_by(|state| state.key.cmp(key))
        .ok()
        .map(|index| &family.states[index])
        .ok_or(PackError::Semantics(
            "chunk state is absent from representation family",
        ))
}

struct ObjectIdentity<'a> {
    tenant: &'a Id32,
    family: &'a Id32,
    content: &'a Id32,
    key_epoch: u64,
    codec: Codec,
    codec_version: u16,
    encrypted: bool,
    salt: &'a [u8; 16],
    nonce: &'a [u8; 12],
    /// SHA-256 of the canonical stats sidecar; `None` (pre-sidecar objects)
    /// contributes nothing, keeping the HMAC input byte-identical.
    stats_digest: Option<&'a Id32>,
}

fn derive_object_key(keys: &KeySchedule, identity: &ObjectIdentity<'_>) -> Id32 {
    let mut mac =
        <HmacSha256 as Mac>::new_from_slice(keys.object_identity_key()).expect("fixed HMAC key");
    mac.update(b"kvpack/v1/chunk-object\0");
    mac.update(identity.tenant);
    mac.update(identity.family);
    mac.update(identity.content);
    mac.update(&identity.key_epoch.to_le_bytes());
    mac.update(&identity.codec.wire().to_le_bytes());
    mac.update(&identity.codec_version.to_le_bytes());
    mac.update(&[u8::from(identity.encrypted)]);
    mac.update(identity.salt);
    mac.update(identity.nonce);
    if let Some(stats_digest) = identity.stats_digest {
        mac.update(b"kvpack/v1/chunk-object-stats\0");
        mac.update(stats_digest);
    }
    mac.finalize().into_bytes().into()
}

fn derive_encryption_key(
    keys: &KeySchedule,
    content: &Id32,
    object: &Id32,
    salt: &[u8; 16],
) -> Result<[u8; 32], PackError> {
    let hk = Hkdf::<Sha256>::new(Some(salt), keys.chunk_encryption_key());
    let mut key = [0u8; 32];
    let mut info = Vec::with_capacity(96);
    info.extend_from_slice(b"kvpack/v1/chunk-aead\0");
    info.extend_from_slice(content);
    info.extend_from_slice(object);
    hk.expand(&info, &mut key)
        .map_err(|_| PackError::Authentication("chunk key derivation failed"))?;
    Ok(key)
}

fn frame_header(magic: &[u8; 8], plaintext_bytes: usize) -> Result<Vec<u8>, PackError> {
    let length = u32::try_from(plaintext_bytes)
        .map_err(|_| PackError::Bounds("codec plaintext length exceeds u32"))?;
    let mut result = Vec::with_capacity(CODEC_FRAME_HEADER_BYTES);
    result.extend_from_slice(magic);
    result.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    result.extend_from_slice(&0u16.to_le_bytes());
    result.extend_from_slice(&length.to_le_bytes());
    Ok(result)
}

fn repeated_run(input: &[u8], start: usize) -> usize {
    let mut length = 1usize;
    while start + length < input.len() && length < 128 && input[start + length] == input[start] {
        length += 1;
    }
    length
}

/// Encode one independently decodable, deterministic stdlib-friendly codec
/// frame.  The lossless lane uses canonical PackBits-style RLE packets.
pub fn encode_codec_frame(codec: Codec, plaintext: &[u8]) -> Result<Vec<u8>, PackError> {
    if plaintext.is_empty() || plaintext.len() > MAX_CHUNK_PLAINTEXT {
        return Err(PackError::Bounds("chunk plaintext must be in 1..=4 MiB"));
    }
    match codec {
        Codec::Raw => {
            let mut result = frame_header(RAW_FRAME_MAGIC, plaintext.len())?;
            result.extend_from_slice(plaintext);
            Ok(result)
        }
        Codec::Lossless => {
            let mut result = frame_header(LOSSLESS_FRAME_MAGIC, plaintext.len())?;
            let mut index = 0usize;
            while index < plaintext.len() {
                let run = repeated_run(plaintext, index);
                if run >= 3 {
                    result.push(0x80 | ((run - 1) as u8));
                    result.push(plaintext[index]);
                    index += run;
                    continue;
                }
                let literal_start = index;
                index += run;
                while index < plaintext.len() && index - literal_start < 128 {
                    let next_run = repeated_run(plaintext, index);
                    if next_run >= 3 {
                        break;
                    }
                    if index - literal_start + next_run > 128 {
                        index = literal_start + 128;
                        break;
                    }
                    index += next_run;
                }
                let length = index - literal_start;
                result.push((length - 1) as u8);
                result.extend_from_slice(&plaintext[literal_start..index]);
            }
            if result.len() > MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD {
                return Err(PackError::Bounds("lossless frame exceeds encoded bound"));
            }
            Ok(result)
        }
    }
}

pub fn decode_codec_frame(
    codec: Codec,
    encoded: &[u8],
    expected_plaintext_bytes: usize,
) -> Result<Vec<u8>, PackError> {
    if encoded.len() < CODEC_FRAME_HEADER_BYTES {
        return Err(PackError::Truncated("truncated codec frame"));
    }
    let expected_magic = match codec {
        Codec::Raw => RAW_FRAME_MAGIC,
        Codec::Lossless => LOSSLESS_FRAME_MAGIC,
    };
    if &encoded[..8] != expected_magic {
        return Err(PackError::BadMagic(
            "codec frame magic does not match codec",
        ));
    }
    if get_u16(encoded, 8) != WIRE_VERSION {
        return Err(PackError::BadMagic("unsupported codec frame version"));
    }
    if get_u16(encoded, 10) != 0 {
        return Err(PackError::Reserved("codec frame reserved field is nonzero"));
    }
    if get_u32(encoded, 12) as usize != expected_plaintext_bytes
        || expected_plaintext_bytes == 0
        || expected_plaintext_bytes > MAX_CHUNK_PLAINTEXT
    {
        return Err(PackError::Bounds("codec frame decoded length mismatch"));
    }
    let payload = &encoded[CODEC_FRAME_HEADER_BYTES..];
    let decoded = match codec {
        Codec::Raw => {
            if payload.len() != expected_plaintext_bytes {
                return Err(PackError::Bounds("raw frame length mismatch"));
            }
            // A Raw frame is canonical by construction once its header and
            // length fields validate (above), so the re-encode canonicity
            // check cannot reject anything the header checks did not; skip it
            // and copy the payload out exactly once.
            return Ok(payload.to_vec());
        }
        Codec::Lossless => {
            let mut result = Vec::with_capacity(expected_plaintext_bytes);
            let mut index = 0usize;
            while index < payload.len() {
                let control = payload[index];
                index += 1;
                let length = (control as usize & 0x7f) + 1;
                if control & 0x80 != 0 {
                    let value = *payload
                        .get(index)
                        .ok_or(PackError::Truncated("truncated lossless repeat packet"))?;
                    index += 1;
                    if result.len().saturating_add(length) > expected_plaintext_bytes {
                        return Err(PackError::Bounds("lossless frame expands past bound"));
                    }
                    result.resize(result.len() + length, value);
                } else {
                    let end = index
                        .checked_add(length)
                        .ok_or(PackError::Bounds("lossless literal offset overflow"))?;
                    let literal = payload
                        .get(index..end)
                        .ok_or(PackError::Truncated("truncated lossless literal packet"))?;
                    if result.len().saturating_add(length) > expected_plaintext_bytes {
                        return Err(PackError::Bounds("lossless frame expands past bound"));
                    }
                    result.extend_from_slice(literal);
                    index = end;
                }
            }
            if result.len() != expected_plaintext_bytes {
                return Err(PackError::Bounds("lossless decoded length mismatch"));
            }
            result
        }
    };
    if encode_codec_frame(codec, &decoded)? != encoded {
        return Err(PackError::Reserved("codec frame is not canonical"));
    }
    Ok(decoded)
}

pub fn encode_chunk(
    plaintext: &[u8],
    encoding: &ChunkEncoding<'_>,
    keys: &KeySchedule,
) -> Result<ChunkObject, PackError> {
    encode_chunk_with_content_id(plaintext, encoding, keys, None)
}

/// `encode_chunk` with an optional caller-precomputed plaintext content id.
/// Callers that already derived `chunk_id` over the plaintext (for dedup
/// lookups) pass it here to avoid a second full-plaintext HMAC.  A stale id
/// cannot produce a readable object: `decode_chunk` re-derives the content id
/// from the plaintext and fails closed on mismatch.
pub fn encode_chunk_with_content_id(
    plaintext: &[u8],
    encoding: &ChunkEncoding<'_>,
    keys: &KeySchedule,
    content_id: Option<&Id32>,
) -> Result<ChunkObject, PackError> {
    if encoding.key_epoch == 0 {
        return Err(PackError::Semantics("chunk key epoch must be nonzero"));
    }
    if encoding.span.plaintext_bytes as usize != plaintext.len() || encoding.span.token_count == 0 {
        return Err(PackError::Semantics(
            "chunk plaintext does not match its declared span",
        ));
    }
    let family_state = family_state(encoding.family, encoding.state_key)?;
    let codec = family_state.codec;
    let codec_version = family_state.codec_version;
    if codec_version != 1 {
        return Err(PackError::Codec("unsupported chunk codec version"));
    }
    let encoded = encode_codec_frame(codec, plaintext)?;
    let family = representation_family_id(encoding.family)?;
    let content = match content_id {
        Some(content) => *content,
        None => chunk_id(
            keys.chunk_identity_key(),
            &encoding.tenant_namespace,
            encoding.family,
            encoding.state_key,
            &encoding.span,
            plaintext,
        )?,
    };
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 12];
    if encoding.encrypt {
        getrandom::fill(&mut salt)
            .map_err(|_| PackError::Authentication("chunk salt generation failed"))?;
        getrandom::fill(&mut nonce)
            .map_err(|_| PackError::Authentication("chunk nonce generation failed"))?;
    }
    let sidecar_bytes = match encoding.stats_sidecar {
        Some(sidecar) => {
            let encoded = sidecar.encode_canonical()?;
            if encoded.is_empty() || encoded.len() > MAX_STATS_SIDECAR_BYTES {
                return Err(PackError::Bounds(
                    "stats sidecar exceeds the chunk header capacity",
                ));
            }
            Some(encoded)
        }
        None => None,
    };
    let stats_digest: Option<Id32> = sidecar_bytes.as_ref().map(|encoded| {
        use sha2::Digest as _;
        Sha256::digest(encoded).into()
    });
    let object_key = derive_object_key(
        keys,
        &ObjectIdentity {
            tenant: &encoding.tenant_namespace,
            family: &family,
            content: &content,
            key_epoch: encoding.key_epoch,
            codec,
            codec_version,
            encrypted: encoding.encrypt,
            salt: &salt,
            nonce: &nonce,
            stats_digest: stats_digest.as_ref(),
        },
    );
    let payload_bytes = encoded
        .len()
        .checked_add(if encoding.encrypt { 16 } else { 0 })
        .ok_or(PackError::Bounds("chunk payload length overflow"))?;
    let object_bytes = align(
        CHUNK_HEADER_BYTES
            .checked_add(payload_bytes)
            .ok_or(PackError::Bounds("chunk object length overflow"))?,
    )?;
    if object_bytes > MAX_CHUNK_OBJECT_BYTES {
        return Err(PackError::Bounds("encoded chunk object exceeds bound"));
    }
    // Build the object with exactly one allocation: header fields are
    // appended in wire order, then the payload, then the zero padding — no
    // full-object zero-fill ahead of the overwrite.
    let flags: u32 = if encoding.encrypt { FLAG_ENCRYPTED } else { 0 };
    let mut bytes = Vec::with_capacity(object_bytes);
    bytes.extend_from_slice(CHUNK_MAGIC);
    bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(CHUNK_HEADER_BYTES as u16).to_le_bytes());
    bytes.extend_from_slice(&(ALIGNMENT as u32).to_le_bytes());
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&codec.wire().to_le_bytes());
    bytes.extend_from_slice(&codec_version.to_le_bytes());
    bytes.extend_from_slice(&(plaintext.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(payload_bytes as u32).to_le_bytes());
    bytes.extend_from_slice(&(object_bytes as u32).to_le_bytes());
    bytes.extend_from_slice(&encoding.key_epoch.to_le_bytes());
    bytes.extend_from_slice(&encoding.tenant_namespace);
    bytes.extend_from_slice(&family);
    bytes.extend_from_slice(&content);
    bytes.extend_from_slice(&object_key);
    bytes.extend_from_slice(&salt);
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&[0u8; 32]); // header digest placeholder
    debug_assert_eq!(bytes.len(), CHUNK_HEADER_SIDECAR_OFFSET);
    if let Some(sidecar) = &sidecar_bytes {
        bytes.extend_from_slice(&(sidecar.len() as u16).to_le_bytes());
        bytes.extend_from_slice(sidecar);
    }
    bytes.resize(CHUNK_HEADER_BYTES, 0); // reserved tail (zero)
    debug_assert_eq!(bytes.len(), CHUNK_HEADER_BYTES);
    let header_digest: Id32 = Sha256::digest(&bytes).into();
    bytes[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32].copy_from_slice(&header_digest);
    let payload = if encoding.encrypt {
        let key = Zeroizing::new(derive_encryption_key(keys, &content, &object_key, &salt)?);
        ChaCha20Poly1305::new(Key::from_slice(key.as_ref()))
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &encoded,
                    aad: &bytes[..CHUNK_HEADER_BYTES],
                },
            )
            .map_err(|_| PackError::Authentication("chunk encryption failed"))?
    } else {
        encoded
    };
    bytes.extend_from_slice(&payload);
    bytes.resize(object_bytes, 0);
    let object_digest: Id32 = Sha256::digest(&bytes).into();
    Ok(ChunkObject {
        chunk_id: content,
        object_key,
        object_digest,
        plaintext_bytes: plaintext.len() as u32,
        bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn decode_chunk(
    bytes: &[u8],
    expected: &ChunkRef,
    expected_span: &ChunkSpan,
    expected_tenant: &Id32,
    expected_family: &RepresentationFamilyId,
    expected_state_key: &StateKey,
    keys: &KeySchedule,
) -> Result<Vec<u8>, PackError> {
    Ok(decode_chunk_with_stats(
        bytes,
        expected,
        expected_span,
        expected_tenant,
        expected_family,
        expected_state_key,
        keys,
    )?
    .0)
}

/// `decode_chunk` that additionally returns the authenticated M7 statistics
/// sidecar when the chunk carries one (`None` for pre-sidecar objects).
#[allow(clippy::too_many_arguments)]
pub fn decode_chunk_with_stats(
    bytes: &[u8],
    expected: &ChunkRef,
    expected_span: &ChunkSpan,
    expected_tenant: &Id32,
    expected_family: &RepresentationFamilyId,
    expected_state_key: &StateKey,
    keys: &KeySchedule,
) -> Result<(Vec<u8>, Option<StatsSidecar>), PackError> {
    if bytes.len() < CHUNK_HEADER_BYTES
        || bytes.len() > MAX_CHUNK_OBJECT_BYTES
        || bytes.len() % ALIGNMENT != 0
    {
        return Err(PackError::Bounds("invalid chunk object length"));
    }
    if &bytes[..8] != CHUNK_MAGIC {
        return Err(PackError::BadMagic("invalid production chunk magic"));
    }
    if get_u16(bytes, 8) != WIRE_VERSION
        || get_u16(bytes, 10) as usize != CHUNK_HEADER_BYTES
        || get_u32(bytes, 12) as usize != ALIGNMENT
    {
        return Err(PackError::BadMagic("invalid chunk header contract"));
    }
    let flags = get_u32(bytes, 16);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(PackError::Reserved("unknown chunk flags"));
    }
    let codec = Codec::from_wire(get_u16(bytes, 20))?;
    let codec_version = get_u16(bytes, 22);
    if codec_version != 1 {
        return Err(PackError::Codec("unsupported chunk codec version"));
    }
    let plaintext_bytes = get_u32(bytes, 24) as usize;
    let encoded_bytes = get_u32(bytes, 28) as usize;
    let payload_bytes = get_u32(bytes, 32) as usize;
    if plaintext_bytes == 0
        || plaintext_bytes > MAX_CHUNK_PLAINTEXT
        || !(CODEC_FRAME_HEADER_BYTES..=MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD)
            .contains(&encoded_bytes)
        || get_u32(bytes, 36) as usize != bytes.len()
    {
        return Err(PackError::Bounds("invalid chunk size fields"));
    }
    let encrypted = flags & FLAG_ENCRYPTED != 0;
    if payload_bytes != encoded_bytes + if encrypted { 16 } else { 0 } {
        return Err(PackError::Bounds(
            "chunk payload length does not match encryption mode",
        ));
    }
    let payload_end = CHUNK_HEADER_BYTES
        .checked_add(payload_bytes)
        .ok_or(PackError::Bounds("chunk payload offset overflow"))?;
    if payload_end > bytes.len() {
        return Err(PackError::Truncated("truncated chunk payload"));
    }
    if bytes[payload_end..].iter().any(|byte| *byte != 0) {
        return Err(PackError::Reserved("chunk padding is nonzero"));
    }
    // Sidecar presence is the nonzero length prefix in the header tail; no
    // flag bit is consumed, preserving the pre-sidecar flag contract.
    let sidecar_len = get_u16(bytes, CHUNK_HEADER_SIDECAR_OFFSET) as usize;
    let stats_sidecar = if sidecar_len != 0 {
        if sidecar_len > MAX_STATS_SIDECAR_BYTES {
            return Err(PackError::Bounds(
                "chunk stats sidecar length is outside bounds",
            ));
        }
        let sidecar_start = CHUNK_HEADER_SIDECAR_OFFSET + 2;
        let sidecar_end = sidecar_start + sidecar_len;
        let sidecar = StatsSidecar::decode_canonical(&bytes[sidecar_start..sidecar_end])?;
        if bytes[sidecar_end..CHUNK_HEADER_BYTES]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(PackError::Reserved(
                "chunk header sidecar padding is nonzero",
            ));
        }
        let digest: Id32 = Sha256::digest(&bytes[sidecar_start..sidecar_end]).into();
        Some((sidecar, digest))
    } else {
        if bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_BYTES]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(PackError::Reserved(
                "chunk header reserved bytes are nonzero",
            ));
        }
        None
    };
    if !encrypted && bytes[176..204].iter().any(|byte| *byte != 0) {
        return Err(PackError::Reserved(
            "unencrypted chunk salt or nonce is nonzero",
        ));
    }
    if bytes.len() != expected.object_bytes as usize
        || plaintext_bytes != expected.plaintext_bytes as usize
        || plaintext_bytes != expected_span.plaintext_bytes as usize
    {
        return Err(PackError::Bounds("chunk reference size mismatch"));
    }

    // Envelope authentication follows all framing/reserved checks.
    let actual_object: Id32 = Sha256::digest(bytes).into();
    if actual_object != expected.object_digest {
        return Err(PackError::Authentication("chunk object digest mismatch"));
    }
    let mut header = bytes[..CHUNK_HEADER_BYTES].to_vec();
    let expected_header: Id32 = header[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32]
        .try_into()
        .expect("fixed digest range");
    header[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32].fill(0);
    let actual_header: Id32 = Sha256::digest(&header).into();
    if actual_header != expected_header {
        return Err(PackError::Authentication("chunk header digest mismatch"));
    }

    let family_state = family_state(expected_family, expected_state_key)?;
    if codec != family_state.codec || codec_version != family_state.codec_version {
        return Err(PackError::Semantics(
            "chunk codec does not match representation family",
        ));
    }
    let tenant: Id32 = bytes[48..80].try_into().expect("fixed tenant range");
    let family: Id32 = bytes[80..112].try_into().expect("fixed family range");
    let expected_family_id = representation_family_id(expected_family)?;
    if &tenant != expected_tenant
        || family != expected_family_id
        || get_u64(bytes, 40) != expected.key_epoch
    {
        return Err(PackError::Authentication(
            "chunk namespace, family, or epoch mismatch",
        ));
    }
    let content: Id32 = bytes[112..144].try_into().expect("fixed content range");
    let object_key: Id32 = bytes[144..176].try_into().expect("fixed object-key range");
    let salt: [u8; 16] = bytes[176..192].try_into().expect("fixed salt range");
    let nonce: [u8; 12] = bytes[192..204].try_into().expect("fixed nonce range");
    let derived_object_key = derive_object_key(
        keys,
        &ObjectIdentity {
            tenant: &tenant,
            family: &family,
            content: &content,
            key_epoch: expected.key_epoch,
            codec,
            codec_version,
            encrypted,
            salt: &salt,
            nonce: &nonce,
            stats_digest: stats_sidecar.as_ref().map(|(_, digest)| digest),
        },
    );
    if content != expected.chunk_id
        || object_key != expected.object_key
        || object_key != derived_object_key
    {
        return Err(PackError::Authentication(
            "chunk identity or object key mismatch",
        ));
    }
    let encoded: Cow<'_, [u8]> = if encrypted {
        let key = Zeroizing::new(derive_encryption_key(keys, &content, &object_key, &salt)?);
        Cow::Owned(
            ChaCha20Poly1305::new(Key::from_slice(key.as_ref()))
                .decrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: &bytes[CHUNK_HEADER_BYTES..payload_end],
                        aad: &bytes[..CHUNK_HEADER_BYTES],
                    },
                )
                .map_err(|_| PackError::Authentication("chunk AEAD authentication failed"))?,
        )
    } else {
        Cow::Borrowed(&bytes[CHUNK_HEADER_BYTES..payload_end])
    };
    if encoded.len() != encoded_bytes {
        return Err(PackError::Bounds("chunk encoded length mismatch"));
    }
    let plaintext = decode_codec_frame(codec, &encoded, plaintext_bytes)?;
    let derived_content = chunk_id(
        keys.chunk_identity_key(),
        &tenant,
        expected_family,
        expected_state_key,
        expected_span,
        &plaintext,
    )?;
    if derived_content != content {
        return Err(PackError::Authentication(
            "chunk plaintext identity mismatch",
        ));
    }
    Ok((plaintext, stats_sidecar.map(|(sidecar, _)| sidecar)))
}
