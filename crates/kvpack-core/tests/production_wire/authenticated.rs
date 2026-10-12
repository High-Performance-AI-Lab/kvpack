use crate::common::*;

#[test]
fn raw_and_lossless_chunks_verify_plaintext_identity() {
    for codec in [Codec::Raw, Codec::Lossless] {
        for encrypted in [false, true] {
            let (object, _, reference, span, manifest, keys) = fixture(codec, encrypted, true);
            assert_eq!(
                decode_chunk(
                    &object,
                    &reference,
                    &span,
                    &manifest.tenant_namespace,
                    &manifest.family,
                    &manifest.states[0].key,
                    &keys,
                )
                .unwrap(),
                (0u8..16).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn plaintext_and_encrypted_manifests_authenticate() {
    for encrypted in [false, true] {
        let (_, pack, _, _, expected, keys) = fixture(Codec::Raw, false, encrypted);
        assert_eq!(
            decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).unwrap(),
            expected
        );
    }
}

#[test]
fn error_precedence_is_framing_reserved_then_authentication() {
    let (_, mut pack, _, _, _, keys) = fixture(Codec::Raw, false, false);
    pack[0] ^= 1;
    assert_eq!(
        decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).unwrap_err(),
        PackError::BadMagic("invalid production pack magic")
    );
    let (_, mut pack, _, _, _, keys) = fixture(Codec::Raw, false, false);
    pack[200] = 1;
    assert_eq!(
        decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).unwrap_err(),
        PackError::Reserved("pack header reserved bytes are nonzero")
    );
    let (_, mut pack, _, _, _, keys) = fixture(Codec::Raw, false, false);
    pack[kvpack_core::PACK_HEADER_BYTES + 10] ^= 1;
    assert_eq!(
        decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).unwrap_err(),
        PackError::Authentication("manifest HMAC authentication failed")
    );

    let (_, mut pack, _, _, _, keys) = fixture(Codec::Raw, false, false);
    pack[148] ^= 1;
    let footer_reserved = pack.len() - kvpack_core::PACK_FOOTER_BYTES + 112;
    pack[footer_reserved] = 1;
    assert_eq!(
        decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).unwrap_err(),
        PackError::Reserved("invalid production commit footer")
    );
}

#[test]
fn chunk_splice_and_padding_mutation_are_rejected() {
    let (mut object, _, reference, span, manifest, keys) = fixture(Codec::Raw, false, false);
    object[kvpack_core::CHUNK_HEADER_BYTES] ^= 1;
    assert_eq!(
        decode_chunk(
            &object,
            &reference,
            &span,
            &manifest.tenant_namespace,
            &manifest.family,
            &manifest.states[0].key,
            &keys,
        )
        .unwrap_err(),
        PackError::Authentication("chunk object digest mismatch")
    );

    let (mut object, _, mut reference, span, manifest, keys) = fixture(Codec::Raw, false, false);
    *object.last_mut().unwrap() = 1;
    use sha2::{Digest, Sha256};
    reference.object_digest = Sha256::digest(&object).into();
    assert_eq!(
        decode_chunk(
            &object,
            &reference,
            &span,
            &manifest.tenant_namespace,
            &manifest.family,
            &manifest.states[0].key,
            &keys,
        )
        .unwrap_err(),
        PackError::Reserved("chunk padding is nonzero")
    );
}

#[test]
fn every_bit_of_authenticated_chunk_and_pack_is_sensitive() {
    let (mut chunk, mut pack, reference, span, manifest, keys) =
        fixture(Codec::Lossless, true, true);
    for index in 0..pack.len() {
        for bit in 0..8 {
            pack[index] ^= 1 << bit;
            assert!(
                decode_authenticated_pack(&pack, &keys, &ValidationContext::default()).is_err()
            );
            pack[index] ^= 1 << bit;
        }
    }
    for index in 0..chunk.len() {
        for bit in 0..8 {
            chunk[index] ^= 1 << bit;
            assert!(decode_chunk(
                &chunk,
                &reference,
                &span,
                &manifest.tenant_namespace,
                &manifest.family,
                &manifest.states[0].key,
                &keys,
            )
            .is_err());
            chunk[index] ^= 1 << bit;
        }
    }
}
