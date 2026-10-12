use super::*;

fn descriptor() -> RotationFamilyDescriptorV1 {
    RotationFamilyDescriptorV1::pinned_qwen25([7; 32]).unwrap()
}

#[test]
fn qwen25_table_hash_and_low_bits_are_pinned() {
    let value = descriptor();
    assert_eq!(value.frac64_sha256, QWEN25_FRAC64_TABLE_SHA256);
    for index in [0usize, 1, 7, 31, 63] {
        let offset = index * 8;
        let increment = u64::from_le_bytes(value.frac64_le[offset..offset + 8].try_into().unwrap());
        assert_ne!(increment as u32, 0);
    }
}

#[test]
fn canonical_round_trip_and_identity_are_stable() {
    let value = descriptor();
    let encoded = value.encode_canonical().unwrap();
    assert_eq!(
        RotationFamilyDescriptorV1::decode_canonical(&encoded).unwrap(),
        value
    );
    assert_eq!(value.identity().unwrap(), value.identity().unwrap());
}

#[test]
fn table_identity_and_order_tampering_fail_closed() {
    let value = descriptor();
    let mut wrong_hash = value.clone();
    wrong_hash.frac64_sha256[0] ^= 1;
    assert!(wrong_hash.encode_canonical().is_err());

    let mut reordered = value.clone();
    reordered.frac64_le[..16].rotate_left(8);
    assert!(reordered.encode_canonical().is_err());

    let mut encoded = value.encode_canonical().unwrap();
    encoded[10] = 1;
    assert!(RotationFamilyDescriptorV1::decode_canonical(&encoded).is_err());
}

#[test]
fn truncated_unknown_and_lifted32_tables_fail_closed() {
    let value = descriptor();
    let encoded = value.encode_canonical().unwrap();
    for cut in [0usize, 7, 12, encoded.len() - 1] {
        assert!(RotationFamilyDescriptorV1::decode_canonical(&encoded[..cut]).is_err());
    }
    let mut unknown = encoded.clone();
    unknown[12..14].copy_from_slice(&99u16.to_le_bytes());
    assert!(RotationFamilyDescriptorV1::decode_canonical(&unknown).is_err());

    let lifted = QWEN25_FRAC64
        .into_iter()
        .flat_map(|value| (value & 0xffff_ffff_0000_0000).to_le_bytes())
        .collect();
    assert!(RotationFamilyDescriptorV1::new(
        QWEN25_ROTARY_DIMENSION,
        [7; 32],
        lifted,
        FIXED_Q30_D7_D6_COEFFICIENT_SHA256,
    )
    .is_err());
}

#[test]
fn absent_hook_is_neutral_and_present_hook_is_identity_bound() {
    let base = [3; 32];
    assert_eq!(
        RotationFamilyHook::default().bind_engine_cache_abi(&base),
        base
    );
    let first = RotationFamilyHook::from_descriptor(&descriptor()).unwrap();
    let mut other = descriptor();
    other.model_representation_id = [8; 32];
    let second = RotationFamilyHook::from_descriptor(&other).unwrap();
    assert_ne!(first.bind_engine_cache_abi(&base), base);
    assert_ne!(
        first.bind_engine_cache_abi(&base),
        second.bind_engine_cache_abi(&base)
    );
}

#[test]
fn hook_changes_the_representation_family_identity_only_when_present() {
    let family = crate::RepresentationFamilyId {
        engine_cache_abi: [3; 32],
        mode: crate::RepresentationMode::Native,
        page_size_tokens: 256,
        topology: [4; 32],
        shard_map: [5; 32],
        states: vec![crate::FamilyState {
            key: crate::StateKey::new(0, "attention.k"),
            cache_kind: crate::CacheKind::OrdinaryKv,
            dtype: crate::DType::U8,
            codec: crate::Codec::Raw,
            codec_version: 1,
            layout: crate::Layout::Contiguous,
            token_axis_rule: crate::TokenAxisRule::Direct,
            token_axis: 0,
            elements_per_token: 4,
            dimensions: vec![
                crate::StaticDimension::Token,
                crate::StaticDimension::Fixed(4),
            ],
            dependencies: vec![],
        }],
    };
    let unbound = bind_representation_family(&family, RotationFamilyHook::default());
    assert_eq!(
        crate::representation_family_id(&unbound).unwrap(),
        crate::representation_family_id(&family).unwrap()
    );

    let hook = RotationFamilyHook::from_descriptor(&descriptor()).unwrap();
    let bound = bind_representation_family(&family, hook);
    assert_ne!(
        crate::representation_family_id(&bound).unwrap(),
        crate::representation_family_id(&family).unwrap()
    );
    assert_eq!(bound.states, family.states);
}
