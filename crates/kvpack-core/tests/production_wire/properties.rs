use crate::common::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn raw_and_lossless_codec_frames_round_trip_arbitrary_bytes(
        plaintext in prop::collection::vec(any::<u8>(), 1..16_384)
    ) {
        for codec in [Codec::Raw, Codec::Lossless] {
            let encoded = encode_codec_frame(codec, &plaintext).unwrap();
            prop_assert_eq!(
                decode_codec_frame(codec, &encoded, plaintext.len()).unwrap(),
                plaintext.clone()
            );
        }
    }

    #[test]
    fn state_declarations_round_trip_canonically(
        cut in 1u64..1_000_000,
        width in 1u64..4096,
        group in 1u32..u32::MAX,
    ) {
        let declaration = StateDeclaration {
            key: StateKey::new(3, "attention.v"),
            full_shape: Shape::new(&[cut, width]).unwrap(),
            segment_shape: Shape::new(&[cut, width]).unwrap(),
            strides: vec![width, 1],
            logical_start: 0,
            logical_count: cut,
            absolute_position: cut,
            window: 0,
            atomic_group: group,
        };
        let encoded = declaration.canonical_schema_bytes().unwrap();
        prop_assert_eq!(
            StateDeclaration::decode_canonical_schema(&encoded).unwrap(),
            declaration
        );
    }
}
