use super::*;

/// Deterministic fixture: 6 tokens × 4 channels of pseudo-K/V values with
/// varied per-channel ranges (no RNG — fixed fractions of pi/e).
fn fixture_values() -> Vec<f32> {
    let mut values = Vec::with_capacity(24);
    for token in 0..6 {
        for channel in 0..4 {
            let unit = ((token * 4 + channel + 1) * 37 % 97) as f32 / 97.0;
            let base = unit * 2.0 - 1.0;
            let scale = [0.5, 8.0, 0.05, 3.0][channel];
            values.push(base * scale + channel as f32 * 0.25);
        }
    }
    values
}

fn per_group_bounds(tensor: &QuantizedTensor, values: &[f32]) -> Vec<(f32, f32)> {
    // (group range, |zero_f16 - min_f32|) per group, in canonical group order.
    let group_size = tensor.config.group_size as usize;
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut bounds = Vec::new();
    match tensor.axis {
        QuantAxis::PerChannel => {
            for col in 0..cols {
                for group in 0..rows.div_ceil(group_size) {
                    let start = group * group_size;
                    let end = (start + group_size).min(rows);
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    for row in start..end {
                        let value = values[row * cols + col];
                        min = min.min(value);
                        max = max.max(value);
                    }
                    bounds.push((max - min, min));
                }
            }
        }
        QuantAxis::PerToken => {
            for row in 0..rows {
                for group in 0..cols.div_ceil(group_size) {
                    let start = group * group_size;
                    let end = (start + group_size).min(cols);
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    for col in start..end {
                        let value = values[row * cols + col];
                        min = min.min(value);
                        max = max.max(value);
                    }
                    bounds.push((max - min, min));
                }
            }
        }
    }
    bounds
}

fn assert_round_trip_within_bound(tensor: &QuantizedTensor, values: &[f32]) -> f32 {
    let decoded = tensor.decode().unwrap();
    assert_eq!(decoded.len(), values.len());
    let bounds = per_group_bounds(tensor, values);
    let group_size = tensor.config.group_size as usize;
    let mut worst = 0f32;
    for row in 0..tensor.rows {
        for col in 0..cols(tensor) {
            let index = row * cols(tensor) + col;
            let group = match tensor.axis {
                QuantAxis::PerChannel => col * tensor.rows.div_ceil(group_size) + row / group_size,
                QuantAxis::PerToken => row * cols(tensor).div_ceil(group_size) + col / group_size,
            };
            let (range, min) = bounds[group];
            let zero_bits = tensor.scales[group].1;
            let zero_rounding = (f16_to_f32(zero_bits) - min).abs();
            let bound = tensor.config.error_bound(range, zero_rounding);
            let error = (decoded[index] - values[index]).abs();
            assert!(
                error <= bound + 1e-6,
                "element ({row},{col}) error {error} exceeds documented bound {bound}"
            );
            worst = worst.max(error);
        }
    }
    worst
}

fn cols(tensor: &QuantizedTensor) -> usize {
    tensor.cols
}

#[test]
fn four_bit_k_round_trip_stays_within_documented_bound() {
    let values = fixture_values();
    let config = QuantConfig::new(4, 4).unwrap();
    let tensor = quantize_k_per_channel(&values, 6, 4, config).unwrap();
    let worst = assert_round_trip_within_bound(&tensor, &values);
    // Cross-check against the documented headline bound: <= 3.34% of the
    // per-group range at 4 bits (scale/2 plus fp16 scale rounding).
    let widest_group = 17.0f32; // channel 1 spans about [-8.25, 8.75]
    assert!(worst <= 0.0334 * widest_group);
    // Sanity: quantization is actually happening (error is nonzero).
    assert!(worst > 0.0);
}

#[test]
fn two_bit_v_round_trip_stays_within_documented_bound() {
    let values = fixture_values();
    let config = QuantConfig::new(2, 2).unwrap();
    let tensor = quantize_v_per_token(&values, 6, 4, config).unwrap();
    let worst = assert_round_trip_within_bound(&tensor, &values);
    // Headline bound at 2 bits: 1/6 of the per-group range.
    let widest_group = 17.0f32;
    assert!(worst <= widest_group / 6.0 + 1e-3);
}

#[test]
fn scales_round_trip_bit_exact() {
    let values = fixture_values();
    let config = QuantConfig::new(4, 4).unwrap();
    let tensor = quantize_k_per_channel(&values, 6, 4, config).unwrap();
    let bytes = tensor.encode_canonical().unwrap();
    let decoded = QuantizedTensor::decode_canonical(&bytes).unwrap();
    assert_eq!(decoded, tensor);
    assert_eq!(decoded.scales(), tensor.scales());
    // Dequantization with the round-tripped scales is bit-identical.
    assert_eq!(decoded.decode().unwrap(), tensor.decode().unwrap());
}

#[test]
fn wire_round_trip_both_axes_and_bit_widths() {
    let values = fixture_values();
    for bits in [2, 4] {
        for group_size in [1, 3, 4, 1024] {
            let config = QuantConfig::new(bits, group_size).unwrap();
            for tensor in [
                quantize_k_per_channel(&values, 6, 4, config).unwrap(),
                quantize_v_per_token(&values, 6, 4, config).unwrap(),
            ] {
                let bytes = tensor.encode_canonical().unwrap();
                let decoded = QuantizedTensor::decode_canonical(&bytes).unwrap();
                assert_eq!(decoded, tensor);
                assert_round_trip_within_bound(&decoded, &values);
            }
        }
    }
}

#[test]
fn decode_fails_closed_on_truncation_and_trailing_bytes() {
    let values = fixture_values();
    let tensor = quantize_k_per_channel(&values, 6, 4, QuantConfig::new(4, 4).unwrap()).unwrap();
    let bytes = tensor.encode_canonical().unwrap();
    for cut in 0..bytes.len() {
        assert!(
            QuantizedTensor::decode_canonical(&bytes[..cut]).is_err(),
            "truncated at {cut}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(QuantizedTensor::decode_canonical(&trailing).is_err());
}

#[test]
fn decode_fails_closed_on_corruption() {
    let values = fixture_values();
    let tensor = quantize_k_per_channel(&values, 6, 4, QuantConfig::new(4, 4).unwrap()).unwrap();
    let bytes = tensor.encode_canonical().unwrap();
    // Bad magic.
    let mut corrupt = bytes.clone();
    corrupt[0] = b'X';
    assert!(QuantizedTensor::decode_canonical(&corrupt).is_err());
    // Reserved field nonzero.
    let mut corrupt = bytes.clone();
    corrupt[10] = 1;
    assert!(QuantizedTensor::decode_canonical(&corrupt).is_err());
    // Illegal bit width.
    let mut corrupt = bytes.clone();
    corrupt[12] = 3;
    assert!(QuantizedTensor::decode_canonical(&corrupt).is_err());
    // Tampered group count (offset 34..42).
    let mut corrupt = bytes.clone();
    corrupt[34] = 9;
    assert!(QuantizedTensor::decode_canonical(&corrupt).is_err());
    // Negative scale (sign bit of the first fp16 scale) is invalid.
    let header = 8 + 2 + 2 + 2 + 4 + 8 + 8 + 8;
    let mut corrupt = bytes.clone();
    corrupt[header + 1] |= 0x80;
    assert!(QuantizedTensor::decode_canonical(&corrupt).is_err());
    // Nonzero tail padding bits are non-canonical: 3x1 at 4 bits = 12 bits,
    // so the final byte carries 4 padding bits.
    let tail = vec![0.25f32, -0.5, 1.0];
    let tensor = quantize_k_per_channel(&tail, 3, 1, QuantConfig::new(4, 4).unwrap()).unwrap();
    let mut bytes = tensor.encode_canonical().unwrap();
    let last = bytes.len() - 1;
    bytes[last] |= 0xf0;
    assert!(QuantizedTensor::decode_canonical(&bytes).is_err());
}

#[test]
fn constant_groups_round_trip_exactly() {
    // Constant channel -> zero scale; every code must be zero and decode must
    // reproduce the fp16-rounded constant exactly.
    let values = vec![1.5f32; 12];
    let config = QuantConfig::new(4, 4).unwrap();
    let tensor = quantize_k_per_channel(&values, 3, 4, config).unwrap();
    let decoded = tensor.decode().unwrap();
    for value in decoded {
        assert_eq!(value, f16_to_f32(f32_to_f16(1.5)));
    }
    let bytes = tensor.encode_canonical().unwrap();
    assert!(QuantizedTensor::decode_canonical(&bytes).is_ok());
}

#[test]
fn config_and_input_validation_fail_closed() {
    assert!(QuantConfig::new(1, 4).is_err());
    assert!(QuantConfig::new(8, 4).is_err());
    assert!(QuantConfig::new(4, 0).is_err());
    assert!(QuantConfig::new(4, 1025).is_err());
    let values = fixture_values();
    let config = QuantConfig::new(4, 4).unwrap();
    assert!(quantize_k_per_channel(&values[..23], 6, 4, config).is_err());
    assert!(quantize_k_per_channel(&values, 0, 4, config).is_err());
    let mut nan = values.clone();
    nan[3] = f32::NAN;
    assert!(quantize_k_per_channel(&nan, 6, 4, config).is_err());
}
