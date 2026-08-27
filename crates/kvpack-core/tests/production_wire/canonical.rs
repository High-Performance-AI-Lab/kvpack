use crate::common::*;

#[test]
fn family_schema_and_manifest_reencode_identically() {
    let (_, _, _, _, manifest, _) = fixture(Codec::Raw, false, false);
    let family_bytes = manifest.family.encode_canonical().unwrap();
    assert_eq!(
        RepresentationFamilyId::decode_canonical(&family_bytes)
            .unwrap()
            .encode_canonical()
            .unwrap(),
        family_bytes
    );
    let schema_bytes = manifest.realized_schema.encode_canonical().unwrap();
    assert_eq!(
        RealizedCutSchemaId::decode_canonical(&schema_bytes)
            .unwrap()
            .encode_canonical()
            .unwrap(),
        schema_bytes
    );
    let bytes = manifest.encode_canonical().unwrap();
    let decoded = CutManifest::decode_canonical(&bytes).unwrap();
    assert_eq!(decoded, manifest);
    assert_eq!(decoded.encode_canonical().unwrap(), bytes);
}

#[test]
fn delta_schema_covers_the_exact_append_and_caps_depth_at_seven() {
    let (manifest, keys, _) = delta_fixture(7);
    validate_manifest(&manifest, &ValidationContext::default()).unwrap();
    let schema = manifest.realized_schema.encode_canonical().unwrap();
    assert_eq!(
        RealizedCutSchemaId::decode_canonical(&schema).unwrap(),
        manifest.realized_schema
    );
    let pack =
        encode_authenticated_pack(&manifest, &keys, false, &ValidationContext::default()).unwrap();
    assert_eq!(
        decode_authenticated_pack(&pack.bytes, &keys, &ValidationContext::default()).unwrap(),
        manifest
    );

    let (mut too_deep, _, _) = delta_fixture(8);
    assert_eq!(
        validate_manifest(&too_deep, &ValidationContext::default()).unwrap_err(),
        PackError::Graph("delta depth is outside 1..=7")
    );
    if let ManifestKind::Delta {
        parent_cut, depth, ..
    } = &mut too_deep.realized_schema.kind
    {
        *depth = 7;
        parent_cut.auxiliary_input_root = id(55);
    }
    assert_eq!(
        validate_manifest(&too_deep, &ValidationContext::default()).unwrap_err(),
        PackError::Graph("delta parent cut is not a compatible strict prefix")
    );
}

#[test]
fn reference_only_full_compaction_reuses_delta_chunk_bytes() {
    let (mut compacted, keys, delta_object) = delta_fixture(7);
    let state_key = compacted.family.states[0].key.clone();
    let first_span = ChunkSpan {
        token_start: 0,
        token_count: 2,
        plaintext_offset: 0,
        plaintext_bytes: 8,
    };
    let first_plaintext: Vec<u8> = (0..8).collect();
    let first_object = encode_chunk(
        &first_plaintext,
        &ChunkEncoding {
            tenant_namespace: compacted.tenant_namespace,
            family: &compacted.family,
            state_key: &state_key,
            span: first_span,
            key_epoch: 7,
            encrypt: false,
            stats_sidecar: None,
        },
        &keys,
    )
    .unwrap();
    let first_reference = ChunkRef {
        chunk_id: first_object.chunk_id,
        object_key: first_object.object_key,
        object_digest: first_object.object_digest,
        key_epoch: 7,
        plaintext_bytes: 8,
        object_bytes: first_object.bytes.len() as u32,
    };
    let delta_reference = compacted.states[0].chunks[0].clone();
    let delta_span = compacted.realized_schema.states[0].chunk_spans[0];
    compacted.realized_schema.kind = ManifestKind::Full;
    compacted.realized_schema.states[0] = RealizedStateSchema {
        key: state_key.clone(),
        full_shape: Shape::new(&[4, 4]).unwrap(),
        segment_shape: Shape::new(&[4, 4]).unwrap(),
        strides: vec![4, 1],
        logical_start: 0,
        logical_count: 4,
        physical_offset_bytes: 0,
        physical_span_bytes: 16,
        complete_physical_bytes: 16,
        absolute_position: 4,
        window: 0,
        chunk_spans: vec![first_span, delta_span],
    };
    compacted.realized_schema.segment_restored_bytes = 16;
    compacted.states[0].chunks = vec![first_reference.clone(), delta_reference.clone()];
    validate_manifest(&compacted, &ValidationContext::default()).unwrap();
    assert_eq!(
        decode_chunk(
            &first_object.bytes,
            &first_reference,
            &first_span,
            &compacted.tenant_namespace,
            &compacted.family,
            &state_key,
            &keys,
        )
        .unwrap(),
        first_plaintext
    );
    assert_eq!(
        decode_chunk(
            &delta_object,
            &delta_reference,
            &delta_span,
            &compacted.tenant_namespace,
            &compacted.family,
            &state_key,
            &keys,
        )
        .unwrap(),
        (8..16).collect::<Vec<_>>()
    );
}

#[test]
fn realized_range_and_physical_span_are_semantically_bound() {
    let (_, _, _, _, mut manifest, _) = fixture(Codec::Raw, false, false);
    manifest.realized_schema.states[0].absolute_position = 3;
    assert_eq!(
        validate_manifest(&manifest, &ValidationContext::default()).unwrap_err(),
        PackError::Semantics("realized state range does not match the exact manifest cut")
    );

    let (_, _, _, _, mut manifest, _) = fixture(Codec::Raw, false, false);
    manifest.realized_schema.states[0].physical_span_bytes = 15;
    assert_eq!(
        validate_manifest(&manifest, &ValidationContext::default()).unwrap_err(),
        PackError::Semantics("realized physical span is not canonical")
    );
}

#[test]
fn legacy_magic_and_unknown_versions_fail_closed() {
    let (_, _, _, _, manifest, _) = fixture(Codec::Raw, false, false);
    let mut bytes = manifest.encode_canonical().unwrap();
    bytes[..8].copy_from_slice(b"IOKVPK1\0");
    assert!(matches!(
        CutManifest::decode_canonical(&bytes),
        Err(PackError::BadMagic(_))
    ));
    let mut bytes = manifest.encode_canonical().unwrap();
    bytes[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert_eq!(
        CutManifest::decode_canonical(&bytes).unwrap_err(),
        PackError::BadMagic("unsupported manifest version")
    );
}

#[test]
fn every_canonical_and_envelope_parser_rejects_exhaustive_truncation() {
    let (chunk, pack, reference, span, manifest, keys) = fixture(Codec::Lossless, true, true);
    let family = manifest.family.encode_canonical().unwrap();
    let schema = manifest.realized_schema.encode_canonical().unwrap();
    let manifest_bytes = manifest.encode_canonical().unwrap();
    let declaration = StateDeclaration {
        key: StateKey::new(0, "attention.k"),
        full_shape: Shape::new(&[4, 4]).unwrap(),
        segment_shape: Shape::new(&[4, 4]).unwrap(),
        strides: vec![4, 1],
        logical_start: 0,
        logical_count: 4,
        absolute_position: 4,
        window: 0,
        atomic_group: 1,
    }
    .canonical_schema_bytes()
    .unwrap();
    for end in 0..family.len() {
        assert!(RepresentationFamilyId::decode_canonical(&family[..end]).is_err());
    }
    for end in 0..schema.len() {
        assert!(RealizedCutSchemaId::decode_canonical(&schema[..end]).is_err());
    }
    for end in 0..manifest_bytes.len() {
        assert!(CutManifest::decode_canonical(&manifest_bytes[..end]).is_err());
    }
    for end in 0..declaration.len() {
        assert!(StateDeclaration::decode_canonical_schema(&declaration[..end]).is_err());
    }
    for codec in [Codec::Raw, Codec::Lossless] {
        let frame = encode_codec_frame(codec, b"aaabbbbccdefgh").unwrap();
        for end in 0..frame.len() {
            assert!(decode_codec_frame(codec, &frame[..end], 14).is_err());
        }
    }
    for end in 0..pack.len() {
        assert!(
            decode_authenticated_pack(&pack[..end], &keys, &ValidationContext::default()).is_err()
        );
    }
    for end in 0..chunk.len() {
        assert!(decode_chunk(
            &chunk[..end],
            &reference,
            &span,
            &manifest.tenant_namespace,
            &manifest.family,
            &manifest.states[0].key,
            &keys,
        )
        .is_err());
    }
}

#[test]
fn overflow_counts_trailing_bytes_and_all_legacy_magics_fail_closed() {
    let (_, mut pack, _, _, manifest, _) = fixture(Codec::Raw, false, false);
    for legacy in [
        b"IOKVPK1\0",
        b"IOKVREC\0",
        b"IOKVCM1\0",
        b"IOKVFTR\0",
        b"IOKVLZ1\0",
        b"IOKVQ81\0",
        b"IOKVENC\0",
    ] {
        pack[..8].copy_from_slice(legacy);
        assert!(matches!(
            inspect_pack_header(&pack),
            Err(PackError::BadMagic(_))
        ));
    }

    let mut family = manifest.family.encode_canonical().unwrap();
    family[116..120].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        RepresentationFamilyId::decode_canonical(&family).unwrap_err(),
        PackError::Bounds("family state count is outside bounds")
    );
    let mut schema = manifest.realized_schema.encode_canonical().unwrap();
    schema[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        RealizedCutSchemaId::decode_canonical(&schema).unwrap_err(),
        PackError::Bounds("realized state count is outside bounds")
    );
    let mut manifest_bytes = manifest.encode_canonical().unwrap();
    manifest_bytes.push(0);
    assert_eq!(
        CutManifest::decode_canonical(&manifest_bytes).unwrap_err(),
        PackError::Reserved("trailing canonical bytes")
    );
}
