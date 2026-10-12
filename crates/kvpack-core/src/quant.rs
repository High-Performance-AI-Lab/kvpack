//! KIVI-style asymmetric KV quantization codec (M6 fidelity rung 1 payload).
//!
//! Pure-Rust, CPU-only reference codec for the rest-quantized fidelity rung:
//! keys are quantized per channel (groups along the token axis), values per
//! token (groups along the channel axis), at 2 or 4 bits with binary16
//! scale + zero-point per group.  The encoded form is canonical and decodes
//! fail-closed on truncation, reserved-field, and packing violations.
//!
//! Error model (asymmetric round-to-nearest, `L = 2^bits` levels): with group
//! range `r = max - min` the quantization step is `s = r / (L - 1)` and the
//! absolute error per element is bounded by `s / 2` plus the binary16
//! zero-point rounding (≤ half an f16 ulp of `min`).  That is ≈ 3.34% of the
//! group range at 4 bits and ≈ 16.7% at 2 bits — this codec deliberately
//! documents the honest bound instead of claiming sub-2% fidelity.

use crate::canonical::{Decoder, Encoder};
use crate::half::{f16_to_f32, f32_to_f16};
use crate::{PackError, QUANT_K_MAGIC, QUANT_V_MAGIC, WIRE_VERSION};

/// Hard bound on either tensor dimension.
pub const MAX_QUANT_DIMENSION: usize = 1 << 20;
/// Hard bound on the total quantized element count.
pub const MAX_QUANT_ELEMENTS: usize = 1 << 26;
/// Largest quantization group along the grouped axis.
pub const MAX_QUANT_GROUP_SIZE: u32 = 1024;

/// Which tensor axis the quantization groups run along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAxis {
    /// KIVI key path: one scale/zero per (channel, token group).
    PerChannel,
    /// KIVI value path: one scale/zero per (token, channel group).
    PerToken,
}

impl QuantAxis {
    fn magic(self) -> &'static [u8; 8] {
        match self {
            QuantAxis::PerChannel => QUANT_K_MAGIC,
            QuantAxis::PerToken => QUANT_V_MAGIC,
        }
    }

    fn from_magic(magic: &[u8; 8]) -> Result<Self, PackError> {
        if magic == QUANT_K_MAGIC {
            Ok(QuantAxis::PerChannel)
        } else if magic == QUANT_V_MAGIC {
            Ok(QuantAxis::PerToken)
        } else {
            Err(PackError::BadMagic("invalid quantized tensor magic"))
        }
    }
}

/// Quantization parameters: 2 or 4 bits per element, fp16 scales per group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantConfig {
    pub bits: u8,
    pub group_size: u32,
}

impl QuantConfig {
    pub fn new(bits: u8, group_size: u32) -> Result<Self, PackError> {
        if bits != 2 && bits != 4 {
            return Err(PackError::Codec("quantization must use 2 or 4 bits"));
        }
        if group_size == 0 || group_size > MAX_QUANT_GROUP_SIZE {
            return Err(PackError::Bounds(
                "quantization group size is outside bounds",
            ));
        }
        Ok(Self { bits, group_size })
    }

    /// Quantization levels (`2^bits`).
    pub fn levels(self) -> u32 {
        1u32 << self.bits
    }

    /// Documented error bound for one group: `|x - dequant(x)| <= bound` for
    /// every element of a group with range `range` and zero-point rounding
    /// `zero_rounding` (`|zero_f16 - min_f32|`).  See the module docs.
    pub fn error_bound(self, range: f32, zero_rounding: f32) -> f32 {
        let scale = range / (self.levels() - 1) as f32;
        // Rounding of the fp16 scale only ever widens the step by a relative
        // 2^-11; account for it instead of pretending scales are exact f32.
        scale * (0.5 + 2.0f32.powi(-11)) + zero_rounding
    }
}

/// One group: fp16 scale and fp16 zero-point bits plus packed codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedTensor {
    axis: QuantAxis,
    config: QuantConfig,
    rows: usize,
    cols: usize,
    /// `(scale_bits, zero_bits)` per group, in canonical group order.
    scales: Vec<(u16, u16)>,
    /// Packed codes, LSB-first, `bits` per element, final byte zero-padded.
    packed: Vec<u8>,
}

fn groups_along(extent: usize, group_size: u32) -> usize {
    extent.div_ceil(group_size as usize)
}

impl QuantizedTensor {
    pub fn axis(&self) -> QuantAxis {
        self.axis
    }

    pub fn config(&self) -> QuantConfig {
        self.config
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Exact `(scale, zero)` binary16 bit pairs per group, in group order.
    pub fn scales(&self) -> &[(u16, u16)] {
        &self.scales
    }

    fn packed_len(&self) -> usize {
        (self.rows * self.cols * self.config.bits as usize).div_ceil(8)
    }

    /// Decode to the full row-major f32 tensor (`rows * cols` elements).
    pub fn decode(&self) -> Result<Vec<f32>, PackError> {
        let levels = self.config.levels();
        let mut result = vec![0f32; self.rows * self.cols];
        let mut reader = BitReader {
            packed: &self.packed,
            bits: self.config.bits,
            cursor: 0,
        };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let code = reader
                    .next()
                    .ok_or(PackError::Truncated("quantized tensor codes are truncated"))?;
                let (scale_bits, zero_bits) = self.scales[self.group_index(row, col)];
                let scale = f16_to_f32(scale_bits);
                let zero = f16_to_f32(zero_bits);
                if code >= levels {
                    return Err(PackError::Codec(
                        "quantized tensor code exceeds the level count",
                    ));
                }
                result[row * self.cols + col] = zero + code as f32 * scale;
            }
        }
        Ok(result)
    }

    fn group_index(&self, row: usize, col: usize) -> usize {
        let group_size = self.config.group_size as usize;
        match self.axis {
            QuantAxis::PerChannel => {
                col * groups_along(self.rows, self.config.group_size) + row / group_size
            }
            QuantAxis::PerToken => {
                row * groups_along(self.cols, self.config.group_size) + col / group_size
            }
        }
    }

    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(self.axis.magic());
        out.u16(WIRE_VERSION);
        out.u16(0);
        out.u8(self.config.bits);
        out.u8(0);
        out.u32(self.config.group_size);
        out.u64(self.rows as u64);
        out.u64(self.cols as u64);
        out.u64(self.scales.len() as u64);
        for (scale, zero) in &self.scales {
            out.u16(*scale);
            out.u16(*zero);
        }
        out.u64(self.packed.len() as u64);
        out.bytes(&self.packed);
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        if bytes.len() < 8 {
            return Err(PackError::Truncated("truncated quantized tensor"));
        }
        let magic: &[u8; 8] = bytes[..8].try_into().expect("checked magic range");
        let axis = QuantAxis::from_magic(magic)?;
        let mut input = Decoder::new(bytes, axis.magic())?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported quantized tensor version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "quantized tensor reserved field is nonzero",
            ));
        }
        let bits = input.u8()?;
        if input.u8()? != 0 {
            return Err(PackError::Reserved(
                "quantized tensor reserved byte is nonzero",
            ));
        }
        let group_size = input.u32()?;
        let config = QuantConfig::new(bits, group_size)?;
        let rows = usize::try_from(input.u64()?)
            .map_err(|_| PackError::Bounds("quantized tensor rows exceed usize"))?;
        let cols = usize::try_from(input.u64()?)
            .map_err(|_| PackError::Bounds("quantized tensor cols exceed usize"))?;
        validate_dimensions(rows, cols)?;
        let group_count = usize::try_from(input.u64()?)
            .map_err(|_| PackError::Bounds("quantized tensor group count exceeds usize"))?;
        let expected_groups = match axis {
            QuantAxis::PerChannel => cols * groups_along(rows, group_size),
            QuantAxis::PerToken => rows * groups_along(cols, group_size),
        };
        if group_count != expected_groups {
            return Err(PackError::Bounds(
                "quantized tensor group count does not match its shape",
            ));
        }
        let mut scales = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let scale = input.u16()?;
            let zero = input.u16()?;
            let scale_value = f16_to_f32(scale);
            if !scale_value.is_finite() || scale_value < 0.0 || !f16_to_f32(zero).is_finite() {
                return Err(PackError::Semantics(
                    "quantized tensor scale or zero-point is invalid",
                ));
            }
            scales.push((scale, zero));
        }
        let packed_len = usize::try_from(input.u64()?)
            .map_err(|_| PackError::Bounds("quantized tensor packed length exceeds usize"))?;
        let elements = rows * cols;
        let expected_packed = (elements * bits as usize).div_ceil(8);
        if packed_len != expected_packed {
            return Err(PackError::Bounds(
                "quantized tensor packed length does not match its shape",
            ));
        }
        let packed = input.take(packed_len)?.to_vec();
        input.finish()?;
        let tensor = Self {
            axis,
            config,
            rows,
            cols,
            scales,
            packed,
        };
        tensor.validate_packing()?;
        if tensor.encode_canonical()? != bytes {
            return Err(PackError::Reserved("quantized tensor is not canonical"));
        }
        Ok(tensor)
    }

    /// Structural corruption check: every code fits the bit width, unused
    /// tail bits are zero, and zero-scale groups carry only zero codes.
    fn validate_packing(&self) -> Result<(), PackError> {
        let levels = self.config.levels();
        let elements = self.rows * self.cols;
        let mut reader = BitReader {
            packed: &self.packed,
            bits: self.config.bits,
            cursor: 0,
        };
        for _ in 0..elements {
            let code = reader
                .next()
                .ok_or(PackError::Truncated("quantized tensor codes are truncated"))?;
            if code >= levels {
                return Err(PackError::Codec(
                    "quantized tensor code exceeds the level count",
                ));
            }
        }
        let used_bits = elements * self.config.bits as usize;
        if used_bits % 8 != 0 {
            let tail = self.packed[self.packed.len() - 1];
            if tail >> (used_bits % 8) != 0 {
                return Err(PackError::Reserved(
                    "quantized tensor tail padding bits are nonzero",
                ));
            }
        }
        // A zero-scale group must encode every element as code 0; any other
        // payload is non-canonical and rejected.
        let mut reader = BitReader {
            packed: &self.packed,
            bits: self.config.bits,
            cursor: 0,
        };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let code = reader.next().expect("validated code count");
                let group = self.group_index(row, col);
                if f16_to_f32(self.scales[group].0) == 0.0 && code != 0 {
                    return Err(PackError::Codec(
                        "zero-scale quantization group carries a nonzero code",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_dimensions(rows: usize, cols: usize) -> Result<(), PackError> {
    if rows == 0
        || cols == 0
        || rows > MAX_QUANT_DIMENSION
        || cols > MAX_QUANT_DIMENSION
        || rows
            .checked_mul(cols)
            .is_none_or(|elements| elements > MAX_QUANT_ELEMENTS)
    {
        return Err(PackError::Bounds(
            "quantized tensor dimensions are outside bounds",
        ));
    }
    Ok(())
}

struct BitReader<'a> {
    packed: &'a [u8],
    bits: u8,
    cursor: usize,
}

impl BitReader<'_> {
    fn next(&mut self) -> Option<u32> {
        let bit_offset = self.cursor * self.bits as usize;
        let byte_offset = bit_offset / 8;
        let shift = bit_offset % 8;
        let mask = (1u32 << self.bits) - 1;
        let first = *self.packed.get(byte_offset)? as u32;
        let mut value = first >> shift;
        if shift + self.bits as usize > 8 {
            let second = *self.packed.get(byte_offset + 1)? as u32;
            value |= second << (8 - shift);
        }
        self.cursor += 1;
        Some(value & mask)
    }
}

struct BitWriter {
    packed: Vec<u8>,
    bits: u8,
    cursor: usize,
}

impl BitWriter {
    fn push(&mut self, code: u32) {
        let bit_offset = self.cursor * self.bits as usize;
        let byte_offset = bit_offset / 8;
        let shift = bit_offset % 8;
        if byte_offset == self.packed.len() {
            self.packed.push(0);
        }
        self.packed[byte_offset] |= (code << shift) as u8;
        if shift + self.bits as usize > 8 {
            self.packed.push((code >> (8 - shift)) as u8);
        }
        self.cursor += 1;
    }
}

/// KIVI key path: quantize `values` (row-major `tokens × channels`) with one
/// fp16 scale/zero per (channel, token group).
pub fn quantize_k_per_channel(
    values: &[f32],
    tokens: usize,
    channels: usize,
    config: QuantConfig,
) -> Result<QuantizedTensor, PackError> {
    quantize(values, tokens, channels, config, QuantAxis::PerChannel)
}

/// KIVI value path: quantize `values` (row-major `tokens × channels`) with
/// one fp16 scale/zero per (token, channel group).
pub fn quantize_v_per_token(
    values: &[f32],
    tokens: usize,
    channels: usize,
    config: QuantConfig,
) -> Result<QuantizedTensor, PackError> {
    quantize(values, tokens, channels, config, QuantAxis::PerToken)
}

fn quantize(
    values: &[f32],
    rows: usize,
    cols: usize,
    config: QuantConfig,
    axis: QuantAxis,
) -> Result<QuantizedTensor, PackError> {
    validate_dimensions(rows, cols)?;
    if values.len() != rows * cols {
        return Err(PackError::Bounds(
            "quantization source does not match the declared shape",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PackError::Semantics(
            "quantization source contains a non-finite element",
        ));
    }
    let group_size = config.group_size as usize;
    let groups = groups_along(
        match axis {
            QuantAxis::PerChannel => rows,
            QuantAxis::PerToken => cols,
        },
        config.group_size,
    );
    let lines = match axis {
        QuantAxis::PerChannel => cols,
        QuantAxis::PerToken => rows,
    };
    let levels = config.levels();
    let element = |row: usize, col: usize| values[row * cols + col];
    // `line_element(line, offset)` walks one grouped line: a channel down the
    // token axis (per-channel) or a token across the channel axis (per-token).
    let line_element = |line: usize, offset: usize| match axis {
        QuantAxis::PerChannel => element(offset, line),
        QuantAxis::PerToken => element(line, offset),
    };
    let mut scales = Vec::with_capacity(lines * groups);
    let mut line_codes: Vec<Vec<u32>> = Vec::with_capacity(lines);
    for line in 0..lines {
        let mut codes = Vec::with_capacity(groups * group_size);
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(match axis {
                QuantAxis::PerChannel => rows,
                QuantAxis::PerToken => cols,
            });
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for offset in start..end {
                let value = line_element(line, offset);
                min = min.min(value);
                max = max.max(value);
            }
            let range = max - min;
            let scale_bits = f32_to_f16(range / (levels - 1) as f32);
            let zero_bits = f32_to_f16(min);
            let scale = f16_to_f32(scale_bits);
            let zero = f16_to_f32(zero_bits);
            scales.push((scale_bits, zero_bits));
            for offset in start..end {
                let code = if scale == 0.0 {
                    0
                } else {
                    ((line_element(line, offset) - zero) / scale).round() as i64
                };
                codes.push(code.clamp(0, (levels - 1) as i64) as u32);
            }
        }
        line_codes.push(codes);
    }
    // Repack codes from line order into canonical row-major element order.
    let mut writer = BitWriter {
        packed: Vec::with_capacity((rows * cols * config.bits as usize).div_ceil(8)),
        bits: config.bits,
        cursor: 0,
    };
    let packed;
    {
        let group_index = |row: usize, col: usize| match axis {
            QuantAxis::PerChannel => (col, row),
            QuantAxis::PerToken => (row, col),
        };
        for row in 0..rows {
            for col in 0..cols {
                let (line, offset) = group_index(row, col);
                let group = offset / group_size;
                let within = offset % group_size;
                writer.push(line_codes[line][group * group_size + within]);
            }
        }
        packed = writer.packed;
    }
    let tensor = QuantizedTensor {
        axis,
        config,
        rows,
        cols,
        scales,
        packed,
    };
    debug_assert_eq!(tensor.packed.len(), tensor.packed_len());
    Ok(tensor)
}

#[cfg(test)]
mod tests;
