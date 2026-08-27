use crate::common::*;

#[test]
fn excluded_cache_and_lossy_lanes_have_no_accepted_values() {
    for value in [2, 3, 4, 5, 6] {
        assert!(matches!(
            CacheKind::from_wire(value),
            Err(PackError::UnknownEnum {
                what: "cache kind",
                ..
            })
        ));
    }
    for value in [3, 4, 5] {
        assert!(matches!(
            Codec::from_wire(value),
            Err(PackError::UnknownEnum { what: "codec", .. })
        ));
    }
}

#[test]
fn prefix_chain_binds_semantic_family_auxiliary_and_exact_count() {
    let tenant = id(1);
    let keys = KeySchedule::derive(&id(99), &tenant, 7).unwrap();
    let family = family(Codec::Raw);
    let semantic = semantic();
    let auxiliary = [AuxiliaryInputId {
        type_id: id(20),
        value_id: id(21),
    }];
    let derive = |semantic: &SemanticModelId,
                  family: &RepresentationFamilyId,
                  auxiliary: &[AuxiliaryInputId],
                  tokens: &[u32]| {
        derive_input_cut(
            keys.prefix_key(),
            &tenant,
            semantic,
            family,
            tokens,
            auxiliary,
        )
        .unwrap()
        .0
    };
    let base = derive(&semantic, &family, &auxiliary, &[1, 2, 3, 4]);
    let mut other_semantic = semantic;
    other_semantic.qualified_math = id(15);
    assert_ne!(
        base,
        derive(&other_semantic, &family, &auxiliary, &[1, 2, 3, 4])
    );
    let mut other_family = family.clone();
    other_family.topology = id(5);
    assert_ne!(
        base,
        derive(&semantic, &other_family, &auxiliary, &[1, 2, 3, 4])
    );
    let other_auxiliary = [AuxiliaryInputId {
        type_id: id(20),
        value_id: id(22),
    }];
    assert_ne!(
        base,
        derive(&semantic, &family, &other_auxiliary, &[1, 2, 3, 4])
    );
    assert_ne!(base, derive(&semantic, &family, &auxiliary, &[1, 2, 3]));
    assert_eq!(base.token_count, 4);

    let invalid_auxiliary = [AuxiliaryInputId {
        type_id: [0; 32],
        value_id: id(21),
    }];
    assert_eq!(
        derive_input_cut(
            keys.prefix_key(),
            &tenant,
            &semantic,
            &family,
            &[1],
            &invalid_auxiliary,
        )
        .unwrap_err(),
        PackError::Semantics("auxiliary identity contains a zero component")
    );
}

#[test]
fn key_rotation_preserves_content_identity_but_changes_object_identity() {
    let tenant = id(1);
    let root = id(99);
    let family = family(Codec::Raw);
    let key = family.states[0].key.clone();
    let span = ChunkSpan {
        token_start: 0,
        token_count: 4,
        plaintext_offset: 0,
        plaintext_bytes: 16,
    };
    let plaintext: Vec<_> = (0u8..16).collect();
    let first_keys = KeySchedule::derive(&root, &tenant, 7).unwrap();
    let second_keys = KeySchedule::derive(&root, &tenant, 8).unwrap();
    let first = encode_chunk(
        &plaintext,
        &ChunkEncoding {
            tenant_namespace: tenant,
            family: &family,
            state_key: &key,
            span,
            key_epoch: 7,
            encrypt: false,
            stats_sidecar: None,
        },
        &first_keys,
    )
    .unwrap();
    let second = encode_chunk(
        &plaintext,
        &ChunkEncoding {
            tenant_namespace: tenant,
            family: &family,
            state_key: &key,
            span,
            key_epoch: 8,
            encrypt: false,
            stats_sidecar: None,
        },
        &second_keys,
    )
    .unwrap();
    assert_eq!(first.chunk_id, second.chunk_id);
    assert_ne!(first.object_key, second.object_key);
    assert_ne!(first.object_digest, second.object_digest);
    assert_ne!(representation_family_id(&family).unwrap(), [0; 32]);
}

#[test]
fn key_separation_wrong_context_and_random_nonce_boundaries_hold() {
    use std::collections::BTreeSet;

    fn require_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}
    require_zeroize_on_drop::<KeySchedule>();

    let tenant = id(1);
    let root = id(99);
    let first = KeySchedule::derive(&root, &tenant, 7).unwrap();
    let second = KeySchedule::derive(&root, &tenant, 8).unwrap();
    assert_eq!(first.namespace_key(), second.namespace_key());
    assert_eq!(first.prefix_key(), second.prefix_key());
    assert_eq!(first.chunk_identity_key(), second.chunk_identity_key());
    assert_ne!(first.manifest_auth_key(), second.manifest_auth_key());
    assert_ne!(
        first.manifest_encryption_key(),
        second.manifest_encryption_key()
    );
    assert_ne!(first.object_identity_key(), second.object_identity_key());
    assert_ne!(first.chunk_encryption_key(), second.chunk_encryption_key());
    let separated = [
        *first.namespace_key(),
        *first.prefix_key(),
        *first.chunk_identity_key(),
        *first.manifest_auth_key(),
        *first.manifest_encryption_key(),
        *first.object_identity_key(),
        *first.chunk_encryption_key(),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    assert_eq!(separated.len(), 7);

    let (chunk, pack, reference, span, manifest, keys) = fixture(Codec::Raw, true, true);
    let wrong_epoch = KeySchedule::derive(&root, &tenant, 8).unwrap();
    assert!(decode_authenticated_pack(&pack, &wrong_epoch, &ValidationContext::default()).is_err());
    assert!(decode_chunk(
        &chunk,
        &reference,
        &span,
        &id(2),
        &manifest.family,
        &manifest.states[0].key,
        &keys,
    )
    .is_err());
    assert!(decode_chunk(
        &chunk,
        &reference,
        &span,
        &manifest.tenant_namespace,
        &manifest.family,
        &manifest.states[0].key,
        &wrong_epoch,
    )
    .is_err());

    let mut chunk_randomness = BTreeSet::new();
    let mut pack_randomness = BTreeSet::new();
    for _ in 0..64 {
        let (chunk, pack, _, _, _, _) = fixture(Codec::Raw, true, true);
        assert!(chunk_randomness.insert(chunk[176..204].to_vec()));
        assert!(pack_randomness.insert(pack[120..148].to_vec()));
    }
}
