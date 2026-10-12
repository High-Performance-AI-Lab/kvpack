//! Pure executors for the exact repack ops. Every op is a bytewise/index
//! permutation (or zero-pad/trim) over one KV plane; `dtype-cast` is named
//! by the descriptor for documentation but is rejected here because no
//! fp16/bf16/fp8e4m3 cast is bit-exact.

use super::{
    validate_permutation, RepackOp, RopeDirection, TransformDescriptor, MAX_PERMUTATION_WIDTH,
};
use crate::PackError;

/// Geometry of one KV plane: `[tokens][kv_heads][head_dim]` elements of
/// `element_bytes` each, row-major contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPlaneShape {
    pub tokens: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub element_bytes: u32,
}

impl KvPlaneShape {
    pub fn plane_bytes(&self) -> Result<usize, PackError> {
        for (value, what) in [
            (self.tokens, "tokens"),
            (self.kv_heads, "kv heads"),
            (self.head_dim, "head dim"),
            (self.element_bytes, "element bytes"),
        ] {
            if value == 0 {
                return Err(PackError::Bounds(match what {
                    "tokens" => "transform plane has zero tokens",
                    "kv heads" => "transform plane has zero kv heads",
                    "head dim" => "transform plane has zero head dim",
                    _ => "transform plane has zero element bytes",
                }));
            }
        }
        usize::try_from(self.tokens)
            .ok()
            .and_then(|tokens| {
                tokens
                    .checked_mul(self.kv_heads as usize)?
                    .checked_mul(self.head_dim as usize)?
                    .checked_mul(self.element_bytes as usize)
            })
            .ok_or(PackError::Bounds("transform plane byte count overflows"))
    }
}

fn checked_plane(shape: &KvPlaneShape, plane: &[u8]) -> Result<usize, PackError> {
    let expected = shape.plane_bytes()?;
    if plane.len() != expected {
        return Err(PackError::Bounds(
            "transform plane bytes do not match the declared shape",
        ));
    }
    Ok(expected)
}

/// Slab permutation shared by `reorder-planes` and `regroup-layers`:
/// `order[i]` = source slab placed at output slot `i`.
fn permute_slabs(order: &[u32], plane: &[u8]) -> Result<Vec<u8>, PackError> {
    if order.len() > MAX_PERMUTATION_WIDTH || plane.len() % order.len() != 0 {
        return Err(PackError::Bounds(
            "transform slab count does not divide the plane bytes",
        ));
    }
    validate_permutation(order)?;
    let slab = plane.len() / order.len();
    let mut output = vec![0u8; plane.len()];
    for (slot, source) in order.iter().enumerate() {
        let source = *source as usize * slab;
        output[slot * slab..(slot + 1) * slab].copy_from_slice(&plane[source..source + slab]);
    }
    Ok(output)
}

/// `order[i]` = source head pair placed at output pair slot `i`, within
/// every token row.
fn permute_head_pairs(
    order: &[u32],
    shape: &KvPlaneShape,
    plane: &[u8],
) -> Result<Vec<u8>, PackError> {
    if shape.kv_heads % 2 != 0 || order.len() != (shape.kv_heads / 2) as usize {
        return Err(PackError::Bounds(
            "transform head-pair order does not match the declared kv heads",
        ));
    }
    validate_permutation(order)?;
    let row_bytes = (shape.kv_heads as usize)
        .checked_mul(shape.head_dim as usize)
        .and_then(|row| row.checked_mul(shape.element_bytes as usize))
        .ok_or(PackError::Bounds("transform row byte count overflows"))?;
    let pair_bytes = 2 * shape.head_dim as usize * shape.element_bytes as usize;
    let mut output = vec![0u8; plane.len()];
    for row in 0..shape.tokens as usize {
        let row_start = row * row_bytes;
        for (slot, source) in order.iter().enumerate() {
            let source = row_start + *source as usize * pair_bytes;
            let target = row_start + slot * pair_bytes;
            output[target..target + pair_bytes]
                .copy_from_slice(&plane[source..source + pair_bytes]);
        }
    }
    Ok(output)
}

/// Exact RoPE index permutation over each head vector of `head_dim`
/// elements: NeoX keeps the rotation dims as two half-split runs,
/// interleaved keeps them as adjacent pairs.
fn rope_permute(
    direction: RopeDirection,
    head_dim: u32,
    shape: &KvPlaneShape,
    plane: &[u8],
) -> Result<Vec<u8>, PackError> {
    if head_dim != shape.head_dim || head_dim < 2 || head_dim % 2 != 0 {
        return Err(PackError::Bounds(
            "transform rope head dim does not match the declared plane shape",
        ));
    }
    let element_bytes = shape.element_bytes as usize;
    let head_bytes = head_dim as usize * element_bytes;
    let half = (head_dim / 2) as usize;
    let mut output = vec![0u8; plane.len()];
    for head in 0..plane.len() / head_bytes {
        let base = head * head_bytes;
        for index in 0..half {
            // (source, target) element-index pairs for one rotation dim.
            let moves: [(usize, usize); 2] = match direction {
                // out[2i] = in[i], out[2i+1] = in[i + half]
                RopeDirection::NeoxToInterleaved => {
                    [(index, 2 * index), (index + half, 2 * index + 1)]
                }
                // out[i] = in[2i], out[i + half] = in[2i+1]
                RopeDirection::InterleavedToNeox => {
                    [(2 * index, index), (2 * index + 1, index + half)]
                }
            };
            for (source, target) in moves {
                let source = base + source * element_bytes;
                let target = base + target * element_bytes;
                output[target..target + element_bytes]
                    .copy_from_slice(&plane[source..source + element_bytes]);
            }
        }
    }
    Ok(output)
}

/// Apply one op to one plane. Fails closed on any geometry, width, or
/// exactness violation.
pub fn apply_repack_op(
    op: &RepackOp,
    shape: &KvPlaneShape,
    plane: &[u8],
) -> Result<Vec<u8>, PackError> {
    op.validate()?;
    checked_plane(shape, plane)?;
    match op {
        RepackOp::PermuteHeadPairs { order } => permute_head_pairs(order, shape, plane),
        RepackOp::ReorderPlanes { order } | RepackOp::RegroupLayers { order } => {
            permute_slabs(order, plane)
        }
        RepackOp::PadOrTrim { target_bytes } => {
            let target = usize::try_from(*target_bytes)
                .map_err(|_| PackError::Bounds("transform pad target overflows usize"))?;
            let mut output = vec![0u8; target];
            let kept = plane.len().min(target);
            output[..kept].copy_from_slice(&plane[..kept]);
            Ok(output)
        }
        RepackOp::DtypeCast { .. } => Err(PackError::Semantics(
            "transform dtype-cast is not bit-exact; the executor rejects lossy casts",
        )),
        RepackOp::RopePermute {
            direction,
            head_dim,
        } => rope_permute(*direction, *head_dim, shape, plane),
    }
}

/// Apply the descriptor's ops in declared order.
pub fn apply_transform(
    descriptor: &TransformDescriptor,
    shape: &KvPlaneShape,
    plane: &[u8],
) -> Result<Vec<u8>, PackError> {
    descriptor.validate()?;
    let mut current = plane.to_vec();
    for op in &descriptor.ops {
        current = apply_repack_op(op, shape, &current)?;
    }
    Ok(current)
}

/// The exact inverse of one op, when it is a bijection: permutations invert
/// their order, `rope-permute` flips direction. `pad-or-trim` and
/// `dtype-cast` are not invertible and return `None`.
pub fn inverse_repack_op(op: &RepackOp) -> Option<RepackOp> {
    match op {
        RepackOp::PermuteHeadPairs { order } => Some(RepackOp::PermuteHeadPairs {
            order: invert(order),
        }),
        RepackOp::ReorderPlanes { order } => Some(RepackOp::ReorderPlanes {
            order: invert(order),
        }),
        RepackOp::RegroupLayers { order } => Some(RepackOp::RegroupLayers {
            order: invert(order),
        }),
        RepackOp::RopePermute {
            direction,
            head_dim,
        } => Some(RepackOp::RopePermute {
            direction: match direction {
                RopeDirection::NeoxToInterleaved => RopeDirection::InterleavedToNeox,
                RopeDirection::InterleavedToNeox => RopeDirection::NeoxToInterleaved,
            },
            head_dim: *head_dim,
        }),
        RepackOp::PadOrTrim { .. } | RepackOp::DtypeCast { .. } => None,
    }
}

fn invert(order: &[u32]) -> Vec<u32> {
    let mut inverse = vec![0u32; order.len()];
    for (slot, source) in order.iter().enumerate() {
        inverse[*source as usize] = slot as u32;
    }
    inverse
}

#[cfg(test)]
mod tests;
