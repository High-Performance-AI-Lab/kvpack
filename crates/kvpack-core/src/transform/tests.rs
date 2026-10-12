use super::*;

fn descriptor(ops: Vec<RepackOp>) -> TransformDescriptor {
    TransformDescriptor {
        schema_version: 1,
        name: "qwen25-7b-to-canonical".into(),
        source_layout: "qwen2.5-7b".into(),
        ops,
    }
}

fn every_op() -> Vec<RepackOp> {
    vec![
        RepackOp::PermuteHeadPairs {
            order: vec![1, 0, 3, 2],
        },
        RepackOp::ReorderPlanes { order: vec![1, 0] },
        RepackOp::RegroupLayers {
            order: vec![2, 0, 1],
        },
        RepackOp::PadOrTrim {
            target_bytes: 4_096,
        },
        RepackOp::DtypeCast {
            from: CastDType::Fp16,
            to: CastDType::Fp8E4m3,
            scale_id: Some([7; 32]),
        },
        RepackOp::RopePermute {
            direction: RopeDirection::InterleavedToNeox,
            head_dim: 128,
        },
    ]
}

#[test]
fn canonical_round_trip_covers_every_op() {
    let descriptor = descriptor(every_op());
    let bytes = descriptor.encode().unwrap();
    assert_eq!(TransformDescriptor::decode(&bytes).unwrap(), descriptor);
}

#[test]
fn canonical_encoding_is_stable() {
    // Golden bytes: magic, version, labels, and each op's tag/params.
    let descriptor = descriptor(vec![RepackOp::RopePermute {
        direction: RopeDirection::NeoxToInterleaved,
        head_dim: 4,
    }]);
    let bytes = descriptor.encode().unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(b"KVXFM1\0\0");
    expected.extend_from_slice(&1u16.to_le_bytes()); // version
    expected.extend_from_slice(&22u16.to_le_bytes()); // name length
    expected.extend_from_slice(b"qwen25-7b-to-canonical");
    expected.extend_from_slice(&10u16.to_le_bytes()); // source layout length
    expected.extend_from_slice(b"qwen2.5-7b");
    expected.extend_from_slice(&1u16.to_le_bytes()); // op count
    expected.extend_from_slice(&6u16.to_le_bytes()); // rope-permute tag
    expected.extend_from_slice(&1u16.to_le_bytes()); // neox-to-interleaved
    expected.extend_from_slice(&4u32.to_le_bytes()); // head dim
    assert_eq!(bytes, expected);
}

#[test]
fn transform_id_is_sha256_of_canonical_bytes() {
    let descriptor = descriptor(every_op());
    let expected: Id32 = Sha256::digest(descriptor.encode().unwrap()).into();
    assert_eq!(descriptor.transform_id().unwrap(), expected);
}

#[test]
fn identity_depends_on_every_op_and_param() {
    let base = descriptor(every_op()).transform_id().unwrap();
    let mut changed = descriptor(every_op());
    changed.ops[0] = RepackOp::PermuteHeadPairs {
        order: vec![0, 1, 3, 2],
    };
    assert_ne!(changed.transform_id().unwrap(), base);
    let mut changed = descriptor(every_op());
    changed.source_layout = "gpt-oss-120b".into();
    assert_ne!(changed.transform_id().unwrap(), base);
    let mut changed = descriptor(every_op());
    changed.ops.truncate(5);
    assert_ne!(changed.transform_id().unwrap(), base);
}

#[test]
fn decode_fails_closed() {
    let bytes = descriptor(every_op()).encode().unwrap();
    // Bad magic.
    let mut changed = bytes.clone();
    changed[0] = b'X';
    assert!(TransformDescriptor::decode(&changed).is_err());
    // Trailing bytes.
    let mut changed = bytes.clone();
    changed.push(0);
    assert!(TransformDescriptor::decode(&changed).is_err());
    // Truncation.
    assert!(TransformDescriptor::decode(&bytes[..bytes.len() - 1]).is_err());
    // Unknown op tag: patch a single-op descriptor's tag byte.
    let single = descriptor(vec![RepackOp::PadOrTrim { target_bytes: 8 }])
        .encode()
        .unwrap();
    let mut changed = single.clone();
    let tag_at = changed.len() - 10; // u16 tag + u64 target
    changed[tag_at] = 99;
    assert!(matches!(
        TransformDescriptor::decode(&changed),
        Err(PackError::UnknownEnum {
            what: "transform op",
            value: 99
        })
    ));
    // Unknown enum values inside an op.
    let mut changed = descriptor(vec![RepackOp::RopePermute {
        direction: RopeDirection::NeoxToInterleaved,
        head_dim: 4,
    }])
    .encode()
    .unwrap();
    let direction_at = changed.len() - 6; // u16 direction + u32 head dim
    changed[direction_at] = 9;
    assert!(matches!(
        TransformDescriptor::decode(&changed),
        Err(PackError::UnknownEnum {
            what: "rope direction",
            value: 9
        })
    ));
}

#[test]
fn validation_fails_closed_on_bad_params() {
    // Non-permutation orders.
    assert!(
        descriptor(vec![RepackOp::ReorderPlanes { order: vec![1, 1] }])
            .encode()
            .is_err()
    );
    assert!(
        descriptor(vec![RepackOp::ReorderPlanes { order: vec![0, 2] }])
            .encode()
            .is_err()
    );
    assert!(
        descriptor(vec![RepackOp::PermuteHeadPairs { order: vec![] }])
            .encode()
            .is_err()
    );
    // Pad target zero.
    assert!(descriptor(vec![RepackOp::PadOrTrim { target_bytes: 0 }])
        .encode()
        .is_err());
    // Cast endpoints must differ; fp8 endpoints require a scale-id, and
    // non-fp8 endpoints must not carry one.
    assert!(descriptor(vec![RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Fp16,
        scale_id: None,
    }])
    .encode()
    .is_err());
    assert!(descriptor(vec![RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Fp8E4m3,
        scale_id: None,
    }])
    .encode()
    .is_err());
    assert!(descriptor(vec![RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Bf16,
        scale_id: Some([1; 32]),
    }])
    .encode()
    .is_err());
    // fp16→bf16 is nameable for documentation even though execution refuses.
    assert!(descriptor(vec![RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Bf16,
        scale_id: None,
    }])
    .encode()
    .is_ok());
    // Odd / zero / oversized rope head dims.
    for head_dim in [0, 3, MAX_ROPE_HEAD_DIM + 2] {
        assert!(descriptor(vec![RepackOp::RopePermute {
            direction: RopeDirection::NeoxToInterleaved,
            head_dim,
        }])
        .encode()
        .is_err());
    }
    // Labels: empty, non-ASCII, oversized.
    let mut changed = descriptor(vec![]);
    changed.name = String::new();
    assert!(changed.encode().is_err());
    let mut changed = descriptor(vec![]);
    changed.name = "nön-ascii".into();
    assert!(changed.encode().is_err());
    let mut changed = descriptor(vec![]);
    changed.source_layout = "x".repeat(MAX_TRANSFORM_LABEL_BYTES + 1);
    assert!(changed.encode().is_err());
    // Op count bound and schema version.
    let mut changed = descriptor(vec![
        RepackOp::PadOrTrim { target_bytes: 1 };
        MAX_TRANSFORM_OPS + 1
    ]);
    assert!(changed.encode().is_err());
    changed.ops.truncate(1);
    changed.schema_version = 2;
    assert!(changed.encode().is_err());
}

#[test]
fn json_form_parses_and_denies_the_unknown() {
    let text = r#"{
        "schema_version": 1,
        "name": "qwen25-7b-to-canonical",
        "source_layout": "qwen2.5-7b",
        "ops": [
            {"op": "permute-head-pairs", "order": [1, 0]},
            {"op": "reorder-planes", "order": [1, 0]},
            {"op": "regroup-layers", "order": [2, 0, 1]},
            {"op": "pad-or-trim", "target_bytes": 4096},
            {"op": "dtype-cast", "from": "fp16", "to": "fp8e4m3",
             "scale_id": "000000000000000000000000000000000000000000000000000000000000007b"},
            {"op": "rope-permute", "direction": "interleaved-to-neox", "head_dim": 128}
        ]
    }"#;
    let parsed: TransformDescriptor = serde_json::from_str(text).unwrap();
    let mut expected_ops = every_op();
    expected_ops[0] = RepackOp::PermuteHeadPairs { order: vec![1, 0] };
    expected_ops[4] = RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Fp8E4m3,
        scale_id: Some({
            let mut id = [0u8; 32];
            id[31] = 0x7b;
            id
        }),
    };
    assert_eq!(parsed, descriptor(expected_ops));
    // Unknown field at the top level and inside an op.
    let unknown = text.replace("\"source_layout\"", "\"layout\"");
    assert!(serde_json::from_str::<TransformDescriptor>(&unknown).is_err());
    let unknown = text.replace("\"order\": [1, 0]", "\"order\": [1, 0], \"extra\": 1");
    assert!(serde_json::from_str::<TransformDescriptor>(&unknown).is_err());
    // Unknown op name and enum value.
    let unknown = text.replace("permute-head-pairs", "shuffle-heads");
    assert!(serde_json::from_str::<TransformDescriptor>(&unknown).is_err());
    let unknown = text.replace("fp8e4m3", "fp8e5m2");
    assert!(serde_json::from_str::<TransformDescriptor>(&unknown).is_err());
    // Malformed scale-id.
    let unknown = text.replace(
        "000000000000000000000000000000000000000000000000000000000000007b",
        "zz",
    );
    assert!(serde_json::from_str::<TransformDescriptor>(&unknown).is_err());
    // Parsed-but-invalid descriptors still fail validation.
    let invalid = text.replace("[1, 0]", "[1, 1]");
    let parsed: TransformDescriptor = serde_json::from_str(&invalid).unwrap();
    assert!(parsed.validate().is_err());
}

#[test]
fn identity_transform_has_empty_ops_and_encodes() {
    let descriptor = descriptor(vec![]);
    let bytes = descriptor.encode().unwrap();
    assert_eq!(TransformDescriptor::decode(&bytes).unwrap(), descriptor);
}
