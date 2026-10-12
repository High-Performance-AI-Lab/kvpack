//! M7 attention-statistics sidecar: an optional, bounded, canonical record of
//! per-cut attention statistics derived from the persisted state bytes.
//!
//! The sidecar carries per-channel K min/max (the quantization scales a
//! fidelity-rung re-encode would need), per-token key L2 norms, and the top-m
//! attention-sink scores (m bounded by [`MAX_SINK_SCORES`]).  It lives in the
//! chunk header tail, is hashed into the chunk object identity exactly like
//! the other authenticated header fields, and is therefore integrity-protected
//! by the same header/object digests.  Chunks encoded without a sidecar are
//! byte-identical to the pre-sidecar format.

use crate::canonical::{Decoder, Encoder};
use crate::half::{f16_to_f32, f32_to_f16};
use crate::{Id32, PackError, STATS_SIDECAR_MAGIC, WIRE_VERSION};

/// Maximum per-channel min/max entries one sidecar may carry.
pub const MAX_SIDECAR_CHANNELS: usize = 512;
/// Maximum per-token L2 norms one sidecar may carry.
pub const MAX_SIDECAR_TOKENS: usize = 768;
/// Maximum retained attention-sink scores (the bounded top-m).
pub const MAX_SINK_SCORES: usize = 8;

/// Per-channel minimum and maximum K values, stored as binary16 bits (the
/// asymmetric quantization range for that channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelRange {
    pub min_bits: u16,
    pub max_bits: u16,
}

impl ChannelRange {
    pub fn min(&self) -> f32 {
        f16_to_f32(self.min_bits)
    }

    pub fn max(&self) -> f32 {
        f16_to_f32(self.max_bits)
    }
}

/// One attention-sink score: a token index and its key L2 norm, binary16 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkScore {
    pub token_index: u32,
    pub score_bits: u16,
}

impl SinkScore {
    pub fn score(&self) -> f32 {
        f16_to_f32(self.score_bits)
    }
}

/// Authenticated per-cut attention statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsSidecar {
    /// Per-channel K min/max, one entry per channel, in channel order.
    pub channel_ranges: Vec<ChannelRange>,
    /// Per-token key L2 norms (binary16 bits), one per token, in token order.
    pub key_l2_norms: Vec<u16>,
    /// Top-m sink scores, sorted by descending score with ties broken by
    /// ascending token index; every entry must reproduce the top-m of
    /// `key_l2_norms` exactly.
    pub sink_scores: Vec<SinkScore>,
}

impl StatsSidecar {
    /// Derive the sidecar from one fp16 K-state plane laid out as
    /// `tokens × channels` little-endian binary16 elements (token-major, the
    /// canonical contiguous layout).  Fails closed on any non-finite element
    /// or out-of-bounds shape.  `sink_count` is clamped to the token count.
    pub fn derive_f16(
        tokens: usize,
        channels: usize,
        sink_count: usize,
        bytes: &[u8],
    ) -> Result<Self, PackError> {
        if tokens == 0
            || tokens > MAX_SIDECAR_TOKENS
            || channels == 0
            || channels > MAX_SIDECAR_CHANNELS
            || sink_count == 0
            || sink_count > MAX_SINK_SCORES
        {
            return Err(PackError::Bounds(
                "stats sidecar shape is outside bounded limits",
            ));
        }
        let expected = tokens
            .checked_mul(channels)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or(PackError::Bounds("stats sidecar element count overflow"))?;
        if bytes.len() != expected {
            return Err(PackError::Bounds(
                "stats sidecar source bytes do not match the declared shape",
            ));
        }
        let element = |token: usize, channel: usize| -> Result<f32, PackError> {
            let offset = (token * channels + channel) * 2;
            let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            let value = f16_to_f32(bits);
            if !value.is_finite() {
                return Err(PackError::Semantics(
                    "stats sidecar source contains a non-finite element",
                ));
            }
            Ok(value)
        };
        let mut channel_ranges = Vec::with_capacity(channels);
        for channel in 0..channels {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for token in 0..tokens {
                let value = element(token, channel)?;
                min = min.min(value);
                max = max.max(value);
            }
            channel_ranges.push(ChannelRange {
                min_bits: f32_to_f16(min),
                max_bits: f32_to_f16(max),
            });
        }
        let mut key_l2_norms = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let mut sum = 0f64;
            for channel in 0..channels {
                let value = element(token, channel)? as f64;
                sum += value * value;
            }
            key_l2_norms.push(f32_to_f16(sum.sqrt() as f32));
        }
        let sink_scores = top_sinks(&key_l2_norms, sink_count.min(tokens));
        let sidecar = Self {
            channel_ranges,
            key_l2_norms,
            sink_scores,
        };
        // Fail closed at persist: a sidecar that cannot fit the chunk header
        // tail must never be attached.
        let encoded = sidecar.encode_canonical()?;
        if encoded.len() > crate::MAX_STATS_SIDECAR_BYTES {
            return Err(PackError::Bounds(
                "stats sidecar exceeds the chunk header capacity",
            ));
        }
        Ok(sidecar)
    }

    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(STATS_SIDECAR_MAGIC);
        out.u16(WIRE_VERSION);
        out.u16(0);
        out.u32(
            u32::try_from(self.channel_ranges.len())
                .map_err(|_| PackError::Bounds("stats sidecar channel count exceeds u32"))?,
        );
        for range in &self.channel_ranges {
            out.u16(range.min_bits);
            out.u16(range.max_bits);
        }
        out.u32(
            u32::try_from(self.key_l2_norms.len())
                .map_err(|_| PackError::Bounds("stats sidecar token count exceeds u32"))?,
        );
        for norm in &self.key_l2_norms {
            out.u16(*norm);
        }
        out.u16(
            u16::try_from(self.sink_scores.len())
                .map_err(|_| PackError::Bounds("stats sidecar sink count exceeds u16"))?,
        );
        for sink in &self.sink_scores {
            out.u32(sink.token_index);
            out.u16(sink.score_bits);
            out.u16(0);
        }
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        if bytes.len() > crate::MAX_STATS_SIDECAR_BYTES {
            return Err(PackError::Bounds(
                "stats sidecar exceeds the chunk header capacity",
            ));
        }
        let mut input = Decoder::new(bytes, STATS_SIDECAR_MAGIC)?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported stats sidecar version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "stats sidecar reserved field is nonzero",
            ));
        }
        let channel_count = input.u32()? as usize;
        if channel_count == 0 || channel_count > MAX_SIDECAR_CHANNELS {
            return Err(PackError::Bounds(
                "stats sidecar channel count is outside bounds",
            ));
        }
        let mut channel_ranges = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            let range = ChannelRange {
                min_bits: input.u16()?,
                max_bits: input.u16()?,
            };
            // Quantization ranges must be finite and ordered.
            if !range.min().is_finite() || !range.max().is_finite() || range.min() > range.max() {
                return Err(PackError::Semantics(
                    "stats sidecar channel range is not finite and ordered",
                ));
            }
            channel_ranges.push(range);
        }
        let token_count = input.u32()? as usize;
        if token_count == 0 || token_count > MAX_SIDECAR_TOKENS {
            return Err(PackError::Bounds(
                "stats sidecar token count is outside bounds",
            ));
        }
        let mut key_l2_norms = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let bits = input.u16()?;
            let norm = f16_to_f32(bits);
            if !norm.is_finite() || norm < 0.0 {
                return Err(PackError::Semantics(
                    "stats sidecar key norm is not a finite non-negative value",
                ));
            }
            key_l2_norms.push(bits);
        }
        let sink_count = input.u16()? as usize;
        if sink_count == 0 || sink_count > MAX_SINK_SCORES {
            return Err(PackError::Bounds(
                "stats sidecar sink count is outside bounds",
            ));
        }
        let mut sink_scores = Vec::with_capacity(sink_count);
        for _ in 0..sink_count {
            let sink = SinkScore {
                token_index: input.u32()?,
                score_bits: input.u16()?,
            };
            if input.u16()? != 0 {
                return Err(PackError::Reserved(
                    "stats sidecar sink reserved field is nonzero",
                ));
            }
            sink_scores.push(sink);
        }
        input.finish()?;
        let sidecar = Self {
            channel_ranges,
            key_l2_norms,
            sink_scores,
        };
        // The sink list must be exactly the top-m of the carried norms; a
        // forged or inconsistent ranking is malformed, not merely stale.
        let expected = top_sinks(&sidecar.key_l2_norms, sink_count.min(token_count));
        if sink_count > token_count || sidecar.sink_scores != expected {
            return Err(PackError::Semantics(
                "stats sidecar sink scores are not the exact top-m norms",
            ));
        }
        if sidecar.encode_canonical()? != bytes {
            return Err(PackError::Reserved("stats sidecar is not canonical"));
        }
        Ok(sidecar)
    }

    /// SHA-256 over the canonical encoding; mixed into the chunk object
    /// identity exactly like the other authenticated header fields.
    pub fn identity_digest(&self) -> Result<Id32, PackError> {
        use sha2::{Digest, Sha256};
        Ok(Sha256::digest(self.encode_canonical()?).into())
    }
}

/// Indices of the `count` largest norms, ordered by descending norm with ties
/// broken by ascending token index, materialized as sink scores.
fn top_sinks(key_l2_norms: &[u16], count: usize) -> Vec<SinkScore> {
    let mut order: Vec<usize> = (0..key_l2_norms.len()).collect();
    order.sort_by(|left, right| {
        f16_to_f32(key_l2_norms[*right])
            .total_cmp(&f16_to_f32(key_l2_norms[*left]))
            .then_with(|| left.cmp(right))
    });
    order
        .into_iter()
        .take(count)
        .map(|index| SinkScore {
            token_index: index as u32,
            score_bits: key_l2_norms[index],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16(value: f32) -> u16 {
        f32_to_f16(value)
    }

    /// Hand-computed fixture: 4 tokens × 3 channels.
    ///   token 0:  1.0, -2.0,  0.5
    ///   token 1: -1.5,  4.0,  0.0
    ///   token 2:  0.25, 1.0, -0.75
    ///   token 3:  2.0,  0.0,  3.0
    fn fixture() -> (Vec<u8>, StatsSidecar) {
        let values: [[f32; 3]; 4] = [
            [1.0, -2.0, 0.5],
            [-1.5, 4.0, 0.0],
            [0.25, 1.0, -0.75],
            [2.0, 0.0, 3.0],
        ];
        let mut bytes = Vec::new();
        for token in values {
            for value in token {
                bytes.extend_from_slice(&f16(value).to_le_bytes());
            }
        }
        let sidecar = StatsSidecar {
            channel_ranges: vec![
                ChannelRange {
                    min_bits: f16(-1.5),
                    max_bits: f16(2.0),
                },
                ChannelRange {
                    min_bits: f16(-2.0),
                    max_bits: f16(4.0),
                },
                ChannelRange {
                    min_bits: f16(-0.75),
                    max_bits: f16(3.0),
                },
            ],
            // L2 norms: sqrt(5.25), sqrt(18.25), sqrt(1.625), sqrt(13).
            key_l2_norms: vec![
                f16(5.25f32.sqrt()),
                f16(18.25f32.sqrt()),
                f16(1.625f32.sqrt()),
                f16(13.0f32.sqrt()),
            ],
            sink_scores: vec![
                SinkScore {
                    token_index: 1,
                    score_bits: f16(18.25f32.sqrt()),
                },
                SinkScore {
                    token_index: 3,
                    score_bits: f16(13.0f32.sqrt()),
                },
                SinkScore {
                    token_index: 0,
                    score_bits: f16(5.25f32.sqrt()),
                },
            ],
        };
        (bytes, sidecar)
    }

    #[test]
    fn derive_matches_hand_computed_statistics() {
        let (bytes, expected) = fixture();
        let derived = StatsSidecar::derive_f16(4, 3, 3, &bytes).unwrap();
        assert_eq!(derived, expected);
        // Channel ranges expose the exact quantization scales.
        assert_eq!(derived.channel_ranges[1].min(), -2.0);
        assert_eq!(derived.channel_ranges[1].max(), 4.0);
        // Sink order is descending norm: token 1 > token 3 > token 0.
        assert!(derived.sink_scores[0].score() > derived.sink_scores[1].score());
    }

    #[test]
    fn sink_count_is_clamped_to_token_count() {
        let (bytes, _) = fixture();
        let derived = StatsSidecar::derive_f16(4, 3, 8, &bytes).unwrap();
        assert_eq!(derived.sink_scores.len(), 4);
        // The fourth sink is the smallest norm (token 2).
        assert_eq!(derived.sink_scores[3].token_index, 2);
    }

    #[test]
    fn canonical_round_trip() {
        let (_, sidecar) = fixture();
        let encoded = sidecar.encode_canonical().unwrap();
        let decoded = StatsSidecar::decode_canonical(&encoded).unwrap();
        assert_eq!(decoded, sidecar);
        assert_eq!(sidecar.identity_digest().unwrap().len(), 32);
    }

    #[test]
    fn decode_rejects_malformed_sidecars() {
        let (_, sidecar) = fixture();
        let encoded = sidecar.encode_canonical().unwrap();
        // Truncation at every boundary fails closed.
        for cut in 0..encoded.len() {
            assert!(
                StatsSidecar::decode_canonical(&encoded[..cut]).is_err(),
                "truncated at {cut}"
            );
        }
        // Trailing bytes fail closed.
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(StatsSidecar::decode_canonical(&trailing).is_err());
        // Reserved version field.
        let mut reserved = encoded.clone();
        reserved[10] = 1;
        assert!(StatsSidecar::decode_canonical(&reserved).is_err());
    }

    #[test]
    fn decode_rejects_reordered_sinks_and_inverted_ranges() {
        let (_, sidecar) = fixture();
        // Swapped sink order is not the exact top-m ranking.
        let mut forged = sidecar.clone();
        forged.sink_scores.swap(0, 1);
        assert!(StatsSidecar::decode_canonical(&forged.encode_canonical().unwrap()).is_err());
        // A sink score that disagrees with the carried norm is malformed.
        let mut forged = sidecar.clone();
        forged.sink_scores[0].score_bits = f16(99.0);
        assert!(StatsSidecar::decode_canonical(&forged.encode_canonical().unwrap()).is_err());
        // Inverted channel range.
        let mut forged = sidecar;
        forged.channel_ranges[0] = ChannelRange {
            min_bits: f16(5.0),
            max_bits: f16(-5.0),
        };
        assert!(StatsSidecar::decode_canonical(&forged.encode_canonical().unwrap()).is_err());
    }

    #[test]
    fn derive_fails_closed_on_bad_input() {
        let (bytes, _) = fixture();
        assert!(StatsSidecar::derive_f16(0, 3, 8, &bytes).is_err());
        assert!(StatsSidecar::derive_f16(4, 0, 8, &bytes).is_err());
        assert!(StatsSidecar::derive_f16(4, 3, 0, &bytes).is_err());
        assert!(StatsSidecar::derive_f16(4, 3, 9, &bytes).is_err());
        assert!(StatsSidecar::derive_f16(4, 3, 8, &bytes[..bytes.len() - 2]).is_err());
        assert!(StatsSidecar::derive_f16(MAX_SIDECAR_TOKENS + 1, 3, 8, &[]).is_err());
        // Non-finite element.
        let mut nan = bytes.clone();
        nan[0] = 0x01;
        nan[1] = 0x7c; // NaN
        assert!(StatsSidecar::derive_f16(4, 3, 8, &nan).is_err());
    }
}
