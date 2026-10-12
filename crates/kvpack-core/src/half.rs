//! Minimal IEEE-754 half-precision conversions shared by the statistics
//! sidecar and the quantization codec.  Pure bit manipulation; no unsafe and
//! no external dependency, matching the crate's fail-closed wire code.

/// Convert one binary16 value to f32 (round-to-nearest is exact here: every
/// finite f16 is exactly representable in f32).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    match exponent {
        0 => {
            if mantissa == 0 {
                return f32::from_bits(sign << 31);
            }
            // Subnormal f16: normalize into the f32 exponent field.
            let mut mantissa = mantissa;
            let mut exponent = -14i32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x3ff;
            let biased = (exponent + 127) as u32;
            f32::from_bits((sign << 31) | (biased << 23) | (mantissa << 13))
        }
        0x1f => {
            if mantissa == 0 {
                f32::from_bits((sign << 31) | 0x7f80_0000)
            } else {
                // Preserve NaN payload; always set the quiet bit.
                f32::from_bits((sign << 31) | 0x7f80_0000 | (mantissa << 13) | 0x0040_0000)
            }
        }
        _ => {
            let biased = (exponent as i32 - 15 + 127) as u32;
            f32::from_bits((sign << 31) | (biased << 23) | (mantissa << 13))
        }
    }
}

/// Convert f32 to the nearest binary16 value (round-to-nearest-even), matching
/// the IEEE default rounding mode used by hardware F16C/NEON converters.
pub fn f32_to_f16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7e00;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;
    if exponent > 15 {
        // Overflow saturates to infinity (matches F16C conversion).
        return sign | 0x7c00;
    }
    if exponent < -24 {
        // Too small even for f16 subnormals: rounds to zero.
        return sign;
    }
    if exponent >= -14 {
        // Normal f16.
        let biased = ((exponent + 15) as u16) << 10;
        let rounded = round_mantissa(mantissa, 13);
        // Rounding can carry into the exponent; renormalize by re-deriving
        // from the rounded f32 bits.
        if rounded == 0x400 {
            return sign | (((exponent + 16) as u16) << 10);
        }
        return sign | biased | rounded;
    }
    // f16 subnormal: shift the mantissa right, keeping the hidden bit.
    let shift = (-14 - exponent) as u32;
    let full = mantissa | 0x0080_0000;
    let rounded = round_mantissa(full, 13 + shift);
    sign | rounded
}

/// Round `mantissa` (with an implicit leading bit possibly set) down by
/// `shift` bits using round-to-nearest-even.
fn round_mantissa(mantissa: u32, shift: u32) -> u16 {
    let dropped = mantissa & ((1u32 << shift) - 1);
    let kept = mantissa >> shift;
    let halfway = 1u32 << (shift - 1);
    let round_up = dropped > halfway || (dropped == halfway && kept & 1 == 1);
    (kept + u32::from(round_up)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_special_values_round_trip() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0x7c01).is_nan());
        // Largest finite f16 and smallest positive subnormal.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), f32::powi(2.0, -24));
        assert_eq!(f16_to_f32(0x03ff), f32::powi(2.0, -14) * (1023.0 / 1024.0));
    }

    #[test]
    fn f32_to_f16_known_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds above max finite -> inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16(f32::NAN), 0x7e00);
        assert_eq!(f32_to_f16(f32::powi(2.0, -24)), 0x0001);
        assert_eq!(f32_to_f16(f32::powi(2.0, -25)), 0x0000); // exactly half of min subnormal -> ties to even (zero)
    }

    #[test]
    fn f16_f32_round_trip_is_exact_for_all_bit_patterns() {
        // Every finite f16 converts to f32 and back to the identical bits;
        // NaNs normalize to the canonical quiet NaN.
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            let value = f16_to_f32(bits);
            let back = f32_to_f16(value);
            if value.is_nan() {
                assert!(f16_to_f32(back).is_nan());
            } else {
                assert_eq!(back, bits, "round trip failed for {bits:#06x}");
            }
        }
    }

    #[test]
    fn f32_to_f16_rounds_to_nearest_even() {
        // Halfway between 1.0 (0x3c00) and the next f16: ties to even.
        let halfway = 1.0 + f32::powi(2.0, -11);
        assert_eq!(f32_to_f16(halfway), 0x3c00);
        let above = 1.0 + f32::powi(2.0, -11) + f32::powi(2.0, -20);
        assert_eq!(f32_to_f16(above), 0x3c01);
        // 65488 is the exact tie between 0x7bfe and 0x7bff: ties to even.
        assert_eq!(f32_to_f16(65504.0 - 16.0), 0x7bfe);
        // Just above the tie rounds up to max finite, not to infinity.
        assert_eq!(f32_to_f16(65504.0 - 15.0), 0x7bff);
    }
}
