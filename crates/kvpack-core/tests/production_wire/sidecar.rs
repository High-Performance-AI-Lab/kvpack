//! M7 statistics-sidecar wire tests: authenticated identity linkage,
//! byte-identical behavior when absent, and fail-closed tamper handling.

use crate::common::*;
use kvpack_core::half::f32_to_f16;
use kvpack_core::{
    decode_chunk_with_stats, ChannelRange, SinkScore, StatsSidecar, CHUNK_HEADER_BYTES,
    CHUNK_HEADER_SIDECAR_OFFSET, KNOWN_FLAGS,
};
use sha2::{Digest, Sha256};

fn f16(value: f32) -> u16 {
    f32_to_f16(value)
}

fn f16_family(codec: Codec) -> RepresentationFamilyId {
    let mut family = family(codec);
    family.states[0].dtype = DType::F16;
    family
}

/// Hand-computed fixture: 4 tokens × 4 fp16 channels.
///   token 0:  1.0,  0.5, -1.0, 2.0
///   token 1:  0.0,  1.5, -0.5, 1.0
///   token 2: -2.0,  0.25, 3.0, 0.0
///   token 3:  4.0, -0.75, 0.5, 1.0
fn fixture_values() -> [[f32; 4]; 4] {
    [
        [1.0, 0.5, -1.0, 2.0],
        [0.0, 1.5, -0.5, 1.0],
        [-2.0, 0.25, 3.0, 0.0],
        [4.0, -0.75, 0.5, 1.0],
    ]
}

fn fixture_plaintext() -> Vec<u8> {
    let mut bytes = Vec::new();
    for token in fixture_values() {
        for value in token {
            bytes.extend_from_slice(&f16(value).to_le_bytes());
        }
    }
    bytes
}

/// The hand-computed expectation: channel min/max from the columns, L2 norms
/// sqrt(6.25)=2.5, sqrt(3.5), sqrt(13.0625), sqrt(17.8125); top-2 sinks are
/// tokens 3 and 2.
fn expected_sidecar() -> StatsSidecar {
    StatsSidecar {
        channel_ranges: vec![
            ChannelRange {
                min_bits: f16(-2.0),
                max_bits: f16(4.0),
            },
            ChannelRange {
                min_bits: f16(-0.75),
                max_bits: f16(1.5),
            },
            ChannelRange {
                min_bits: f16(-1.0),
                max_bits: f16(3.0),
            },
            ChannelRange {
                min_bits: f16(0.0),
                max_bits: f16(2.0),
            },
        ],
        key_l2_norms: vec![
            f16(6.25f32.sqrt()),
            f16(3.5f32.sqrt()),
            f16(13.0625f32.sqrt()),
            f16(17.8125f32.sqrt()),
        ],
        sink_scores: vec![
            SinkScore {
                token_index: 3,
                score_bits: f16(17.8125f32.sqrt()),
            },
            SinkScore {
                token_index: 2,
                score_bits: f16(13.0625f32.sqrt()),
            },
        ],
    }
}

fn encode_with_sidecar(
    plaintext: &[u8],
    sidecar: Option<&StatsSidecar>,
    encrypt: bool,
    keys: &KeySchedule,
    family: &RepresentationFamilyId,
    tenant: &Id32,
) -> (kvpack_core::ChunkObject, ChunkRef, ChunkSpan, StateKey) {
    let span = ChunkSpan {
        token_start: 0,
        token_count: 4,
        plaintext_offset: 0,
        plaintext_bytes: plaintext.len() as u32,
    };
    let state_key = family.states[0].key.clone();
    let object = encode_chunk(
        plaintext,
        &ChunkEncoding {
            tenant_namespace: *tenant,
            family,
            state_key: &state_key,
            span,
            key_epoch: 7,
            encrypt,
            stats_sidecar: sidecar,
        },
        keys,
    )
    .unwrap();
    let reference = ChunkRef {
        chunk_id: object.chunk_id,
        object_key: object.object_key,
        object_digest: object.object_digest,
        key_epoch: 7,
        plaintext_bytes: object.plaintext_bytes,
        object_bytes: object.bytes.len() as u32,
    };
    (object, reference, span, state_key)
}

fn sidecar_fixture(encrypt: bool) -> (Vec<u8>, ChunkRef, ChunkSpan, KeySchedule) {
    let (_, _, _, _, _, keys) = fixture(Codec::Raw, encrypt, false);
    let tenant = id(1);
    let family = f16_family(Codec::Raw);
    let plaintext = fixture_plaintext();
    let sidecar = expected_sidecar();
    let (object, reference, span, _) =
        encode_with_sidecar(&plaintext, Some(&sidecar), encrypt, &keys, &family, &tenant);
    (object.bytes, reference, span, keys)
}

#[test]
fn derived_sidecar_matches_hand_computed_statistics() {
    let sidecar = StatsSidecar::derive_f16(4, 4, 2, &fixture_plaintext()).unwrap();
    assert_eq!(sidecar, expected_sidecar());
}

#[test]
fn sidecar_round_trips_through_authenticated_chunk() {
    let tenant = id(1);
    let family = f16_family(Codec::Raw);
    for encrypt in [false, true] {
        let (_, _, _, _, _, keys) = fixture(Codec::Raw, encrypt, false);
        let plaintext = fixture_plaintext();
        let sidecar = expected_sidecar();
        let (object, reference, span, state_key) =
            encode_with_sidecar(&plaintext, Some(&sidecar), encrypt, &keys, &family, &tenant);
        let flags = u32::from_le_bytes(object.bytes[16..20].try_into().unwrap());
        // No flag bit is consumed: presence is the nonzero length prefix.
        assert_eq!(flags & !KNOWN_FLAGS, 0);
        let sidecar_len = u16::from_le_bytes(
            object.bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_SIDECAR_OFFSET + 2]
                .try_into()
                .unwrap(),
        );
        assert!(sidecar_len > 0);
        let (decoded, decoded_sidecar) = decode_chunk_with_stats(
            &object.bytes,
            &reference,
            &span,
            &tenant,
            &family,
            &state_key,
            &keys,
        )
        .unwrap();
        assert_eq!(decoded, plaintext);
        assert_eq!(decoded_sidecar, Some(sidecar.clone()));
    }
}

#[test]
fn absent_sidecar_keeps_pre_sidecar_byte_layout() {
    let tenant = id(1);
    let family = f16_family(Codec::Raw);
    let (_, _, _, _, _, keys) = fixture(Codec::Raw, false, false);
    let plaintext = fixture_plaintext();
    let (object, reference, span, state_key) =
        encode_with_sidecar(&plaintext, None, false, &keys, &family, &tenant);
    let flags = u32::from_le_bytes(object.bytes[16..20].try_into().unwrap());
    assert_eq!(flags, 0, "no flags without encryption or sidecar");
    // The complete header tail stays zero — the exact pre-sidecar layout.
    assert!(
        object.bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_BYTES]
            .iter()
            .all(|byte| *byte == 0)
    );
    let (decoded, decoded_sidecar) = decode_chunk_with_stats(
        &object.bytes,
        &reference,
        &span,
        &tenant,
        &family,
        &state_key,
        &keys,
    )
    .unwrap();
    assert_eq!(decoded, plaintext);
    assert_eq!(decoded_sidecar, None);
}

#[test]
fn bit_flipped_sidecar_fails_authentication() {
    let (mut bytes, reference, span, keys) = sidecar_fixture(false);
    let family = f16_family(Codec::Raw);
    let state_key = family.states[0].key.clone();
    let tenant = id(1);
    // Byte 16 of the sidecar body: first channel-range low byte.
    let sidecar_offset = CHUNK_HEADER_SIDECAR_OFFSET + 2 + 16;
    bytes[sidecar_offset] ^= 0x01;
    let result = decode_chunk_with_stats(
        &bytes, &reference, &span, &tenant, &family, &state_key, &keys,
    );
    assert!(matches!(result, Err(PackError::Authentication(_))));
}

/// A forger who can rewrite the object and repair its plain SHA-256 digests
/// still cannot attach a different sidecar: the keyed object identity is
/// derived over the sidecar digest and no longer matches.
#[test]
fn forged_sidecar_with_repaired_digests_fails_authentication() {
    let (mut bytes, reference, span, keys) = sidecar_fixture(false);
    let family = f16_family(Codec::Raw);
    let state_key = family.states[0].key.clone();
    let tenant = id(1);
    // A different, perfectly valid sidecar: token 0's norm raised to 9.0,
    // which legitimately re-ranks token 0 as the top sink.
    let mut forged = expected_sidecar();
    forged.key_l2_norms[0] = f16(9.0);
    forged.sink_scores = vec![
        SinkScore {
            token_index: 0,
            score_bits: f16(9.0),
        },
        SinkScore {
            token_index: 3,
            score_bits: f16(17.8125f32.sqrt()),
        },
    ];
    let forged_bytes = forged.encode_canonical().unwrap();
    let length = u16::from_le_bytes(
        bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_SIDECAR_OFFSET + 2]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(forged_bytes.len(), length);
    bytes[CHUNK_HEADER_SIDECAR_OFFSET + 2..CHUNK_HEADER_SIDECAR_OFFSET + 2 + length]
        .copy_from_slice(&forged_bytes);
    // Repair the plain header and object digests so only the keyed identity
    // check can catch the forgery.
    let mut header = bytes[..CHUNK_HEADER_BYTES].to_vec();
    header[204..236].fill(0);
    let header_digest: Id32 = Sha256::digest(&header).into();
    bytes[204..236].copy_from_slice(&header_digest);
    let object_digest: Id32 = Sha256::digest(&bytes).into();
    let mut reference = reference;
    reference.object_digest = object_digest;
    let result = decode_chunk_with_stats(
        &bytes, &reference, &span, &tenant, &family, &state_key, &keys,
    );
    assert!(
        matches!(result, Err(PackError::Authentication(_))),
        "{result:?}"
    );
}

#[test]
fn malformed_sidecar_fails_closed_before_authentication() {
    let (mut bytes, reference, span, keys) = sidecar_fixture(false);
    let family = f16_family(Codec::Raw);
    let state_key = family.states[0].key.clone();
    let tenant = id(1);
    // Point the length field past the header tail.
    bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_SIDECAR_OFFSET + 2]
        .copy_from_slice(&4000u16.to_le_bytes());
    let result = decode_chunk_with_stats(
        &bytes, &reference, &span, &tenant, &family, &state_key, &keys,
    );
    assert!(result.is_err());
    // Non-canonical sidecar content (sink reserved field set), digests
    // repaired: rejected by the sidecar decoder itself.
    let (mut bytes, reference, span, keys) = sidecar_fixture(false);
    let length = u16::from_le_bytes(
        bytes[CHUNK_HEADER_SIDECAR_OFFSET..CHUNK_HEADER_SIDECAR_OFFSET + 2]
            .try_into()
            .unwrap(),
    ) as usize;
    let last = CHUNK_HEADER_SIDECAR_OFFSET + 2 + length - 1;
    bytes[last] = 1; // sink trailing reserved byte
    let mut header = bytes[..CHUNK_HEADER_BYTES].to_vec();
    header[204..236].fill(0);
    let header_digest: Id32 = Sha256::digest(&header).into();
    bytes[204..236].copy_from_slice(&header_digest);
    let object_digest: Id32 = Sha256::digest(&bytes).into();
    let mut reference = reference;
    reference.object_digest = object_digest;
    let result = decode_chunk_with_stats(
        &bytes, &reference, &span, &tenant, &family, &state_key, &keys,
    );
    assert!(matches!(
        result,
        Err(PackError::Reserved(_)) | Err(PackError::Semantics(_))
    ));
}

#[test]
fn sidecar_consumes_no_flag_bit() {
    // The pre-sidecar flag contract is unchanged: only the encryption bit is
    // known, so error phase ordering and multi-language parity are preserved.
    assert_eq!(kvpack_core::KNOWN_FLAGS, 1);
}
