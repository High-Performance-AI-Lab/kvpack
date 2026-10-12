use super::super::CastDType;
use super::*;

fn shape(tokens: u32, kv_heads: u32, head_dim: u32, element_bytes: u32) -> KvPlaneShape {
    KvPlaneShape {
        tokens,
        kv_heads,
        head_dim,
        element_bytes,
    }
}

// ── golden: permute-head-pairs ────────────────────────────────────────
// tokens=2, kv_heads=4, head_dim=2, elt=1: rows are [p0 p1 | p2 p3] of
// head-pair bytes; order [1, 0] swaps the two pairs within each row.
#[test]
fn permute_head_pairs_golden() {
    let plane: Vec<u8> = (0..16).collect();
    let op = RepackOp::PermuteHeadPairs { order: vec![1, 0] };
    let output = apply_repack_op(&op, &shape(2, 4, 2, 1), &plane).unwrap();
    let expected: Vec<u8> = vec![
        4, 5, 6, 7, 0, 1, 2, 3, // row 0: pair 1 then pair 0
        12, 13, 14, 15, 8, 9, 10, 11, // row 1
    ];
    assert_eq!(output, expected);
}

#[test]
fn permute_head_pairs_rejects_geometry_mismatch() {
    let plane: Vec<u8> = (0..16).collect();
    let odd_heads = RepackOp::PermuteHeadPairs { order: vec![0] };
    assert!(apply_repack_op(&odd_heads, &shape(2, 2, 2, 1), &plane).is_err());
    let wrong_width = RepackOp::PermuteHeadPairs {
        order: vec![0, 1, 2],
    };
    assert!(apply_repack_op(&wrong_width, &shape(2, 4, 2, 1), &plane).is_err());
}

// ── golden: reorder-planes / regroup-layers ───────────────────────────
#[test]
fn reorder_planes_golden() {
    let op = RepackOp::ReorderPlanes { order: vec![1, 0] };
    let plane = b"aaaabbbb";
    let output = apply_repack_op(&op, &shape(1, 1, 8, 1), plane).unwrap();
    assert_eq!(output, b"bbbbaaaa");
}

#[test]
fn regroup_layers_golden() {
    let op = RepackOp::RegroupLayers {
        order: vec![2, 0, 1],
    };
    let plane = b"aabbcc";
    let output = apply_repack_op(&op, &shape(1, 1, 6, 1), plane).unwrap();
    assert_eq!(output, b"ccaabb");
}

#[test]
fn slab_ops_reject_indivisible_planes() {
    let op = RepackOp::ReorderPlanes {
        order: vec![0, 1, 2],
    };
    assert!(apply_repack_op(&op, &shape(1, 1, 8, 1), b"aaaabbbb").is_err());
}

// ── golden: pad-or-trim ───────────────────────────────────────────────
#[test]
fn pad_or_trim_golden() {
    let pad = RepackOp::PadOrTrim { target_bytes: 5 };
    assert_eq!(
        apply_repack_op(&pad, &shape(1, 1, 3, 1), b"abc").unwrap(),
        b"abc\0\0"
    );
    let trim = RepackOp::PadOrTrim { target_bytes: 2 };
    assert_eq!(
        apply_repack_op(&trim, &shape(1, 1, 3, 1), b"abc").unwrap(),
        b"ab"
    );
}

// ── dtype-cast: named by the descriptor, rejected at execution ────────
#[test]
fn dtype_cast_is_rejected_at_execution() {
    let cast = RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Bf16,
        scale_id: None,
    };
    let error = apply_repack_op(&cast, &shape(1, 1, 2, 2), &[0; 4]).unwrap_err();
    assert!(error.to_string().contains("not bit-exact"));
}

// ── golden: rope-permute ──────────────────────────────────────────────
// head_dim=4, elt=1: vector [a, b, c, d] is NeoX halves (a, b | c, d);
// interleaved pairs are (a, c), (b, d) → [a, c, b, d].
#[test]
fn rope_permute_golden_both_directions() {
    let plane = vec![10, 11, 12, 13];
    let neox = RepackOp::RopePermute {
        direction: RopeDirection::NeoxToInterleaved,
        head_dim: 4,
    };
    let interleaved = apply_repack_op(&neox, &shape(1, 1, 4, 1), &plane).unwrap();
    assert_eq!(interleaved, vec![10, 12, 11, 13]);
    let back = RepackOp::RopePermute {
        direction: RopeDirection::InterleavedToNeox,
        head_dim: 4,
    };
    assert_eq!(
        apply_repack_op(&back, &shape(1, 1, 4, 1), &interleaved).unwrap(),
        plane
    );
}

#[test]
fn rope_permute_golden_multi_element_width() {
    // elt=2: elements are u16 lanes; permutation is over elements, not bytes.
    let plane: Vec<u8> = (0..8).collect(); // elements [01, 23, 45, 67]
    let op = RepackOp::RopePermute {
        direction: RopeDirection::NeoxToInterleaved,
        head_dim: 4,
    };
    let output = apply_repack_op(&op, &shape(1, 1, 4, 2), &plane).unwrap();
    assert_eq!(output, vec![0, 1, 4, 5, 2, 3, 6, 7]);
}

#[test]
fn rope_permute_rejects_head_dim_mismatch() {
    let op = RepackOp::RopePermute {
        direction: RopeDirection::NeoxToInterleaved,
        head_dim: 8,
    };
    assert!(apply_repack_op(&op, &shape(1, 1, 4, 1), &[0; 4]).is_err());
}

// ── golden: composition through apply_transform ───────────────────────
// head-pair swap, then rope neox→interleaved on the swapped rows.
#[test]
fn composition_golden() {
    let descriptor = TransformDescriptor {
        schema_version: 1,
        name: "qwen-composition".into(),
        source_layout: "qwen2.5-7b".into(),
        ops: vec![
            RepackOp::PermuteHeadPairs { order: vec![1, 0] },
            RepackOp::RopePermute {
                direction: RopeDirection::NeoxToInterleaved,
                head_dim: 2,
            },
        ],
    };
    // One row, 4 heads of dim 2: pairs (h0h1)=(0,1,2,3), (h2h3)=(4,5,6,7).
    let plane: Vec<u8> = (0..8).collect();
    let output = apply_transform(&descriptor, &shape(1, 4, 2, 1), &plane).unwrap();
    // After pair swap: [4,5,6,7,0,1,2,3]; rope on head_dim 2 swaps the two
    // lanes of every head: [4,5,6,7,0,1,2,3] → heads (4,5),(6,7),(0,1),(2,3)
    // → (4,5)→[4,5]? head_dim=2 half=1: out[0]=in[0], out[1]=in[1] — identity.
    assert_eq!(output, vec![4, 5, 6, 7, 0, 1, 2, 3]);

    let descriptor_hd4 = TransformDescriptor {
        ops: vec![RepackOp::RopePermute {
            direction: RopeDirection::NeoxToInterleaved,
            head_dim: 4,
        }],
        ..descriptor.clone()
    };
    // heads are dim 4: [0,1,2,3] → [0,2,1,3], [4,5,6,7] → [4,6,5,7].
    let output = apply_transform(&descriptor_hd4, &shape(1, 2, 4, 1), &plane).unwrap();
    assert_eq!(output, vec![0, 2, 1, 3, 4, 6, 5, 7]);
}

// ── identity transform is the byte identity ───────────────────────────
#[test]
fn identity_transform_passes_bytes_through() {
    let descriptor = TransformDescriptor {
        schema_version: 1,
        name: "identity".into(),
        source_layout: "qwen2.5-7b".into(),
        ops: vec![],
    };
    let plane: Vec<u8> = (0..32).collect();
    assert_eq!(
        apply_transform(&descriptor, &shape(2, 4, 2, 2), &plane).unwrap(),
        plane
    );
}

// ── inverse ops round-trip exactly ────────────────────────────────────
#[test]
fn inverse_ops_round_trip() {
    let ops = vec![
        RepackOp::PermuteHeadPairs {
            order: vec![2, 0, 1],
        },
        RepackOp::ReorderPlanes {
            order: vec![2, 0, 1],
        },
        RepackOp::RegroupLayers {
            order: vec![1, 2, 0],
        },
        RepackOp::RopePermute {
            direction: RopeDirection::NeoxToInterleaved,
            head_dim: 2,
        },
    ];
    // shape: tokens=2, kv_heads=6, head_dim=2, elt=1 → 24 bytes; slab ops
    // divide it into thirds.
    let plane: Vec<u8> = (0..24).collect();
    for op in ops {
        let forward = apply_repack_op(&op, &shape(2, 6, 2, 1), &plane).unwrap();
        let inverse = inverse_repack_op(&op).expect("permutation op is invertible");
        assert_eq!(
            apply_repack_op(&inverse, &shape(2, 6, 2, 1), &forward).unwrap(),
            plane,
            "op {op:?} did not round-trip"
        );
    }
    assert!(inverse_repack_op(&RepackOp::PadOrTrim { target_bytes: 4 }).is_none());
    assert!(inverse_repack_op(&RepackOp::DtypeCast {
        from: CastDType::Fp16,
        to: CastDType::Bf16,
        scale_id: None,
    })
    .is_none());
}

// ── fail-closed geometry checks ───────────────────────────────────────
#[test]
fn executor_rejects_wrong_plane_length_and_zero_dimensions() {
    let op = RepackOp::ReorderPlanes { order: vec![1, 0] };
    assert!(apply_repack_op(&op, &shape(1, 1, 8, 1), &[0; 7]).is_err());
    assert!(apply_repack_op(&op, &shape(0, 1, 8, 1), &[0; 8]).is_err());
    assert!(apply_repack_op(&op, &shape(1, 1, 8, 0), &[0; 8]).is_err());
}

#[test]
fn executor_rejects_non_permutations() {
    let duplicate = RepackOp::ReorderPlanes { order: vec![1, 1] };
    assert!(apply_repack_op(&duplicate, &shape(1, 1, 8, 1), &[0; 8]).is_err());
    let out_of_range = RepackOp::ReorderPlanes { order: vec![0, 2] };
    assert!(apply_repack_op(&out_of_range, &shape(1, 1, 8, 1), &[0; 8]).is_err());
}
