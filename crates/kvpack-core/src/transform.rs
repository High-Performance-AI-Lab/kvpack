//! Engine-ABI translation as a first-class authenticated object (M4,
//! docs/KV_IMPROVEMENT_RESEARCH_2026-08-02.md).
//!
//! A [`TransformDescriptor`] is an ordered list of repack ops that maps a
//! named source layout class onto the canonical layout. The canonical binary
//! encoding is the only authenticated representation: the descriptor's
//! identity is SHA-256 over its canonical bytes. The JSON form is a
//! control-plane representation (deny-unknown, fail-closed) and is never an
//! identity. Ops that are not bit-exact (`dtype-cast`) may be named for
//! documentation but are rejected by the executor (`exec` submodule).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::canonical::{Decoder, Encoder};
use crate::{Id32, PackError};

mod exec;

pub use exec::{apply_repack_op, apply_transform, inverse_repack_op, KvPlaneShape};

/// Canonical-object magic for transform descriptors.
pub const TRANSFORM_MAGIC: &[u8; 8] = b"KVXFM1\0\0";
pub const TRANSFORM_VERSION: u16 = 1;
/// Bounded op list: qualification fixtures never need more.
pub const MAX_TRANSFORM_OPS: usize = 64;
/// `name` / `source_layout` bound (ASCII byte count).
pub const MAX_TRANSFORM_LABEL_BYTES: usize = 128;
/// Widest permutation an op may declare.
pub const MAX_PERMUTATION_WIDTH: usize = 16_384;
/// Largest `pad-or-trim` target (1 GiB).
pub const MAX_PAD_TARGET_BYTES: u64 = 1 << 30;
/// Largest `rope-permute` head dim.
pub const MAX_ROPE_HEAD_DIM: u32 = 4_096;

/// Closed set of dtypes a `dtype-cast` op may name. `fp8e4m3` casts carry a
/// scale sidecar identity; the executor rejects every cast as lossy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CastDType {
    Fp16,
    Bf16,
    #[serde(rename = "fp8e4m3")]
    Fp8E4m3,
}

impl CastDType {
    fn from_wire(value: u16) -> Result<Self, PackError> {
        match value {
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Fp8E4m3),
            other => Err(PackError::UnknownEnum {
                what: "cast dtype",
                value: other as u64,
            }),
        }
    }

    fn wire(self) -> u16 {
        match self {
            Self::Fp16 => 1,
            Self::Bf16 => 2,
            Self::Fp8E4m3 => 3,
        }
    }
}

/// RoPE rotation-dim order: GPT-NeoX half-split vs GPT-J interleaved pairs.
/// Both directions are exact index permutations over one head vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RopeDirection {
    NeoxToInterleaved,
    InterleavedToNeox,
}

impl RopeDirection {
    fn from_wire(value: u16) -> Result<Self, PackError> {
        match value {
            1 => Ok(Self::NeoxToInterleaved),
            2 => Ok(Self::InterleavedToNeox),
            other => Err(PackError::UnknownEnum {
                what: "rope direction",
                value: other as u64,
            }),
        }
    }

    fn wire(self) -> u16 {
        match self {
            Self::NeoxToInterleaved => 1,
            Self::InterleavedToNeox => 2,
        }
    }
}

/// One repack op. Permutation ops use the convention `order[i]` = source
/// index placed at output slot `i`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub enum RepackOp {
    /// Permute adjacent head pairs (heads `2i`, `2i+1`) within every token
    /// row. `order` is a permutation of `0..kv_heads/2`.
    PermuteHeadPairs { order: Vec<u32> },
    /// Permute equal-sized plane slabs of the byte blob. `order` is a
    /// permutation of `0..plane_count`.
    ReorderPlanes { order: Vec<u32> },
    /// Permute equal-sized layer slabs of the byte blob. `order` is a
    /// permutation of `0..layer_count`.
    RegroupLayers { order: Vec<u32> },
    /// Zero-pad or trim the byte blob to exactly `target_bytes`.
    PadOrTrim { target_bytes: u64 },
    /// Documentation-only cast between the closed dtype set. Never exact;
    /// the executor rejects it.
    DtypeCast {
        from: CastDType,
        to: CastDType,
        #[serde(default, with = "hex_option_id32")]
        scale_id: Option<Id32>,
    },
    /// Exact RoPE index permutation over each head vector.
    RopePermute {
        direction: RopeDirection,
        head_dim: u32,
    },
}

impl RepackOp {
    fn encode(&self, out: &mut Encoder) {
        match self {
            Self::PermuteHeadPairs { order } => encode_permutation(out, 1, order),
            Self::ReorderPlanes { order } => encode_permutation(out, 2, order),
            Self::RegroupLayers { order } => encode_permutation(out, 3, order),
            Self::PadOrTrim { target_bytes } => {
                out.u16(4);
                out.u64(*target_bytes);
            }
            Self::DtypeCast { from, to, scale_id } => {
                out.u16(5);
                out.u16(from.wire());
                out.u16(to.wire());
                match scale_id {
                    Some(id) => {
                        out.u8(1);
                        out.id(id);
                    }
                    None => {
                        out.u8(0);
                        out.id(&[0; 32]);
                    }
                }
            }
            Self::RopePermute {
                direction,
                head_dim,
            } => {
                out.u16(6);
                out.u16(direction.wire());
                out.u32(*head_dim);
            }
        }
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        match input.u16()? {
            1 => Ok(Self::PermuteHeadPairs {
                order: decode_permutation(input)?,
            }),
            2 => Ok(Self::ReorderPlanes {
                order: decode_permutation(input)?,
            }),
            3 => Ok(Self::RegroupLayers {
                order: decode_permutation(input)?,
            }),
            4 => Ok(Self::PadOrTrim {
                target_bytes: input.u64()?,
            }),
            5 => {
                let from = CastDType::from_wire(input.u16()?)?;
                let to = CastDType::from_wire(input.u16()?)?;
                let present = input.u8()?;
                let id = input.id()?;
                let scale_id = match present {
                    0 if id == [0; 32] => None,
                    1 => Some(id),
                    _ => {
                        return Err(PackError::Semantics(
                            "transform dtype-cast scale-id marker is outside the v1 bounds",
                        ))
                    }
                };
                Ok(Self::DtypeCast { from, to, scale_id })
            }
            6 => Ok(Self::RopePermute {
                direction: RopeDirection::from_wire(input.u16()?)?,
                head_dim: input.u32()?,
            }),
            other => Err(PackError::UnknownEnum {
                what: "transform op",
                value: other as u64,
            }),
        }
    }

    fn validate(&self) -> Result<(), PackError> {
        match self {
            Self::PermuteHeadPairs { order }
            | Self::ReorderPlanes { order }
            | Self::RegroupLayers { order } => validate_permutation(order),
            Self::PadOrTrim { target_bytes } => {
                if *target_bytes == 0 || *target_bytes > MAX_PAD_TARGET_BYTES {
                    return Err(PackError::Bounds(
                        "transform pad-or-trim target is outside the v1 bounds",
                    ));
                }
                Ok(())
            }
            Self::DtypeCast { from, to, scale_id } => {
                if from == to {
                    return Err(PackError::Semantics(
                        "transform dtype-cast endpoints must differ",
                    ));
                }
                let needs_scale = *from == CastDType::Fp8E4m3 || *to == CastDType::Fp8E4m3;
                if needs_scale != scale_id.is_some() {
                    return Err(PackError::Semantics(
                        "transform dtype-cast scale-id presence must match fp8e4m3 endpoints",
                    ));
                }
                Ok(())
            }
            Self::RopePermute { head_dim, .. } => {
                if *head_dim < 2 || *head_dim > MAX_ROPE_HEAD_DIM || head_dim % 2 != 0 {
                    return Err(PackError::Bounds(
                        "transform rope-permute head dim is outside the v1 bounds",
                    ));
                }
                Ok(())
            }
        }
    }
}

fn encode_permutation(out: &mut Encoder, tag: u16, order: &[u32]) {
    out.u16(tag);
    out.u16(order.len() as u16);
    for index in order {
        out.u32(*index);
    }
}

fn decode_permutation(input: &mut Decoder<'_>) -> Result<Vec<u32>, PackError> {
    let width = input.u16()? as usize;
    let mut order = Vec::with_capacity(width);
    for _ in 0..width {
        order.push(input.u32()?);
    }
    Ok(order)
}

fn validate_permutation(order: &[u32]) -> Result<(), PackError> {
    if order.is_empty() || order.len() > MAX_PERMUTATION_WIDTH {
        return Err(PackError::Bounds(
            "transform permutation width is outside the v1 bounds",
        ));
    }
    let mut seen = vec![false; order.len()];
    for index in order {
        let slot = usize::try_from(*index)
            .ok()
            .filter(|slot| *slot < order.len())
            .ok_or(PackError::Semantics(
                "transform permutation index is outside the permutation width",
            ))?;
        if seen[slot] {
            return Err(PackError::Semantics(
                "transform permutation repeats a source index",
            ));
        }
        seen[slot] = true;
    }
    Ok(())
}

fn validate_label(value: &str, what: &'static str) -> Result<(), PackError> {
    if value.is_empty() || value.len() > MAX_TRANSFORM_LABEL_BYTES || !value.is_ascii() {
        return Err(PackError::Bounds(what));
    }
    Ok(())
}

/// An authenticated engine-ABI translation: ordered repack ops mapping the
/// named source layout class onto the canonical layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformDescriptor {
    pub schema_version: u32,
    pub name: String,
    pub source_layout: String,
    #[serde(default)]
    pub ops: Vec<RepackOp>,
}

impl TransformDescriptor {
    /// Fail-closed validation: unknown or malformed ops/params are errors,
    /// never guesses. Called by `encode`/`decode`; callers parsing the JSON
    /// form must call it too.
    pub fn validate(&self) -> Result<(), PackError> {
        if self.schema_version != u32::from(TRANSFORM_VERSION) {
            return Err(PackError::BadMagic(
                "transform descriptor schema version is outside the v1 bounds",
            ));
        }
        validate_label(&self.name, "transform name is outside the v1 bounds")?;
        validate_label(
            &self.source_layout,
            "transform source layout is outside the v1 bounds",
        )?;
        if self.ops.len() > MAX_TRANSFORM_OPS {
            return Err(PackError::Bounds(
                "transform op count is outside the v1 bounds",
            ));
        }
        for op in &self.ops {
            op.validate()?;
        }
        Ok(())
    }

    /// The only authenticated representation.
    pub fn encode(&self) -> Result<Vec<u8>, PackError> {
        self.validate()?;
        let mut out = Encoder::new(TRANSFORM_MAGIC);
        out.u16(TRANSFORM_VERSION);
        out.string(&self.name)?;
        out.string(&self.source_layout)?;
        out.u16(self.ops.len() as u16);
        for op in &self.ops {
            op.encode(&mut out);
        }
        Ok(out.finish())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PackError> {
        let mut input = Decoder::new(bytes, TRANSFORM_MAGIC)?;
        if input.u16()? != TRANSFORM_VERSION {
            return Err(PackError::BadMagic("invalid transform descriptor version"));
        }
        let descriptor = Self {
            schema_version: u32::from(TRANSFORM_VERSION),
            name: input.string()?,
            source_layout: input.string()?,
            ops: {
                let count = input.u16()? as usize;
                let mut ops = Vec::with_capacity(count);
                for _ in 0..count {
                    ops.push(RepackOp::decode(&mut input)?);
                }
                ops
            },
        };
        input.finish()?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Descriptor identity: SHA-256 over the canonical bytes.
    pub fn transform_id(&self) -> Result<Id32, PackError> {
        Ok(Sha256::digest(self.encode()?).into())
    }
}

/// Lowercase-hex JSON form for `Option<Id32>` (fail-closed on malformed
/// hex); never an identity — the canonical bytes are.
mod hex_option_id32 {
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::Id32;

    pub fn serialize<S: Serializer>(
        value: &Option<Id32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(id) => {
                let hex: String = id.iter().map(|byte| format!("{byte:02x}")).collect();
                serializer.serialize_some(&hex)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Id32>, D::Error> {
        let text: Option<String> = Option::deserialize(deserializer)?;
        match text {
            None => Ok(None),
            Some(text) => {
                let bytes = text.as_bytes();
                if bytes.len() != 64
                    || !bytes
                        .iter()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
                {
                    return Err(serde::de::Error::custom(
                        "scale-id must be 64 lowercase hexadecimal digits",
                    ));
                }
                let mut id = [0u8; 32];
                for (index, pair) in bytes.chunks_exact(2).enumerate() {
                    let pair = std::str::from_utf8(pair).expect("validated hex is ASCII");
                    id[index] = u8::from_str_radix(pair, 16)
                        .map_err(|_| serde::de::Error::custom("invalid scale-id hex"))?;
                }
                Ok(Some(id))
            }
        }
    }
}

#[cfg(test)]
mod tests;
