use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::{
    AuxiliaryInputId, Id32, InputCutId, PackError, RealizedCutSchemaId, RepresentationFamilyId,
    SemanticModelId, StateKey, MAX_CUT_CHAIN_CUTS, PREFIX_BLOCK_TOKENS,
};

type HmacSha256 = Hmac<Sha256>;

fn hmac(key: &[u8], parts: &[&[u8]]) -> Id32 {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts every key length");
    for part in parts {
        mac.update(part);
    }
    mac.finalize().into_bytes().into()
}

fn sha256(parts: &[&[u8]]) -> Id32 {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    hash.finalize().into()
}

pub fn namespace_id(key: &[u8; 32], operator_tenant_id: &[u8]) -> Result<Id32, PackError> {
    let length = u32::try_from(operator_tenant_id.len())
        .map_err(|_| PackError::Bounds("tenant ID is too long"))?;
    Ok(hmac(
        key,
        &[
            b"kvpack/v1/namespace\0",
            &length.to_le_bytes(),
            operator_tenant_id,
        ],
    ))
}

pub fn semantic_model_id(value: &SemanticModelId) -> Id32 {
    sha256(&[
        b"kvpack/v1/semantic-model\0",
        &value.weights_config,
        &value.adapters,
        &value.tokenizer_template,
        &value.position_semantics,
        &value.qualified_math,
    ])
}

pub fn representation_family_id(value: &RepresentationFamilyId) -> Result<Id32, PackError> {
    Ok(sha256(&[
        b"kvpack/v1/representation-family\0",
        &value.encode_canonical()?,
    ]))
}

pub fn realized_cut_schema_id(value: &RealizedCutSchemaId) -> Result<Id32, PackError> {
    Ok(sha256(&[
        b"kvpack/v1/realized-cut-schema\0",
        &value.encode_canonical()?,
    ]))
}

pub fn manifest_id(canonical_manifest: &[u8]) -> Id32 {
    sha256(&[b"kvpack/v1/manifest\0", canonical_manifest])
}

pub fn auxiliary_input_root(
    key: &[u8; 32],
    tenant: &Id32,
    inputs: &[AuxiliaryInputId],
) -> Result<Id32, PackError> {
    let count = u32::try_from(inputs.len())
        .map_err(|_| PackError::Bounds("too many auxiliary identities"))?;
    let mut framed = Vec::with_capacity(inputs.len().saturating_mul(72).saturating_add(4));
    framed.extend_from_slice(&count.to_le_bytes());
    for (ordinal, input) in inputs.iter().enumerate() {
        if input.type_id == [0; 32] || input.value_id == [0; 32] {
            return Err(PackError::Semantics(
                "auxiliary identity contains a zero component",
            ));
        }
        let ordinal = u64::try_from(ordinal)
            .map_err(|_| PackError::Bounds("auxiliary ordinal exceeds u64"))?;
        framed.extend_from_slice(&ordinal.to_le_bytes());
        framed.extend_from_slice(&32u32.to_le_bytes());
        framed.extend_from_slice(&input.type_id);
        framed.extend_from_slice(&32u32.to_le_bytes());
        framed.extend_from_slice(&input.value_id);
    }
    Ok(hmac(
        key,
        &[b"kvpack/v1/auxiliary-input-root\0", tenant, &framed],
    ))
}

pub fn chunk_id(
    key: &[u8; 32],
    tenant: &Id32,
    family: &RepresentationFamilyId,
    state_key: &StateKey,
    span: &crate::ChunkSpan,
    plaintext: &[u8],
) -> Result<Id32, PackError> {
    if span.token_count == 0 || span.plaintext_bytes as usize != plaintext.len() {
        return Err(PackError::Semantics(
            "chunk plaintext does not match its declared span",
        ));
    }
    let family = representation_family_id(family)?;
    let name_length = u16::try_from(state_key.state_name.len())
        .map_err(|_| PackError::Bounds("state name is too long"))?;
    let plaintext_length = u32::try_from(plaintext.len())
        .map_err(|_| PackError::Bounds("chunk plaintext exceeds u32"))?;
    Ok(hmac(
        key,
        &[
            b"kvpack/v1/chunk-content\0",
            tenant,
            &family,
            &state_key.layer.to_le_bytes(),
            &name_length.to_le_bytes(),
            state_key.state_name.as_bytes(),
            &span.token_start.to_le_bytes(),
            &span.token_count.to_le_bytes(),
            &span.plaintext_offset.to_le_bytes(),
            &plaintext_length.to_le_bytes(),
            plaintext,
        ],
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixNode {
    pub token_count: u64,
    pub id: Id32,
    pub reusable: bool,
}

fn prefix_context_root(
    key: &[u8; 32],
    tenant: &Id32,
    semantic_model: &SemanticModelId,
    family: &RepresentationFamilyId,
    auxiliary_root: &Id32,
) -> Result<Id32, PackError> {
    let semantic = semantic_model_id(semantic_model);
    let family = representation_family_id(family)?;
    Ok(hmac(
        key,
        &[
            b"kvpack/v1/prefix-context\0",
            tenant,
            &semantic,
            &family,
            auxiliary_root,
        ],
    ))
}

/// One-pass keyed prefix chain over canonical fixed-width little-endian u32
/// token identifiers.  Only the keyed nodes are returned; token witnesses are
/// never included in a manifest or catalog row.
pub fn chain_prefix_nodes(
    key: &[u8; 32],
    tenant: &Id32,
    semantic_model: &SemanticModelId,
    family: &RepresentationFamilyId,
    auxiliary_root: &Id32,
    tokens: &[u32],
) -> Result<Vec<PrefixNode>, PackError> {
    let mut result = Vec::with_capacity(tokens.len().div_ceil(PREFIX_BLOCK_TOKENS));
    let context = prefix_context_root(key, tenant, semantic_model, family, auxiliary_root)?;
    let mut parent = hmac(key, &[b"kvpack/v1/prefix-root\0", &context]);
    for (index, block) in tokens.chunks(PREFIX_BLOCK_TOKENS).enumerate() {
        let token_start = index
            .checked_mul(PREFIX_BLOCK_TOKENS)
            .ok_or(PackError::Bounds("prefix token offset overflow"))?;
        let byte_length = block
            .len()
            .checked_mul(4)
            .ok_or(PackError::Bounds("prefix block byte length overflow"))?;
        let mut framed = Vec::with_capacity(byte_length + 56);
        framed.extend_from_slice(&context);
        framed.extend_from_slice(&parent);
        let block_index = u64::try_from(index)
            .map_err(|_| PackError::Bounds("prefix block index exceeds u64"))?;
        let token_start_u64 = u64::try_from(token_start)
            .map_err(|_| PackError::Bounds("prefix token offset exceeds u64"))?;
        let block_tokens = u32::try_from(block.len())
            .map_err(|_| PackError::Bounds("prefix block token count exceeds u32"))?;
        let block_bytes = u32::try_from(byte_length)
            .map_err(|_| PackError::Bounds("prefix block byte length exceeds u32"))?;
        framed.extend_from_slice(&block_index.to_le_bytes());
        framed.extend_from_slice(&token_start_u64.to_le_bytes());
        framed.extend_from_slice(&block_tokens.to_le_bytes());
        framed.extend_from_slice(&block_bytes.to_le_bytes());
        for token in block {
            framed.extend_from_slice(&token.to_le_bytes());
        }
        parent = hmac(key, &[b"kvpack/v1/prefix-node\0", &framed]);
        let token_count = token_start
            .checked_add(block.len())
            .ok_or(PackError::Bounds("prefix token count overflow"))?;
        result.push(PrefixNode {
            token_count: u64::try_from(token_count)
                .map_err(|_| PackError::Bounds("prefix token count exceeds u64"))?,
            id: parent,
            reusable: block.len() == PREFIX_BLOCK_TOKENS,
        });
    }
    Ok(result)
}

pub fn derive_input_cut(
    key: &[u8; 32],
    tenant: &Id32,
    semantic_model: &SemanticModelId,
    family: &RepresentationFamilyId,
    tokens: &[u32],
    auxiliary_inputs: &[AuxiliaryInputId],
) -> Result<(InputCutId, Vec<PrefixNode>), PackError> {
    let auxiliary_input_root = auxiliary_input_root(key, tenant, auxiliary_inputs)?;
    let nodes = chain_prefix_nodes(
        key,
        tenant,
        semantic_model,
        family,
        &auxiliary_input_root,
        tokens,
    )?;
    let token_root = match nodes.last() {
        Some(node) => node.id,
        None => {
            let context =
                prefix_context_root(key, tenant, semantic_model, family, &auxiliary_input_root)?;
            hmac(key, &[b"kvpack/v1/prefix-root\0", &context])
        }
    };
    Ok((
        InputCutId {
            token_root,
            auxiliary_input_root,
            token_count: u64::try_from(tokens.len())
                .map_err(|_| PackError::Bounds("input token count exceeds u64"))?,
        },
        nodes,
    ))
}

/// Cut-boundary spacing for the rolling per-cut keyed chain.  `Flat` cuts
/// every stride; `TwoTier` cuts every `head_stride` inside the first
/// `head_tokens` tokens and every `tail_stride` beyond it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutStridePolicy {
    Flat(u32),
    TwoTier {
        head_tokens: u32,
        head_stride: u32,
        tail_stride: u32,
    },
}

impl CutStridePolicy {
    /// Fail-closed validation: strides are nonzero and both strides divide
    /// the head region, so a cut never straddles the head/tail transition.
    pub fn validate(&self) -> Result<(), PackError> {
        match *self {
            CutStridePolicy::Flat(stride) => {
                if stride == 0 {
                    return Err(PackError::Semantics("cut stride is zero"));
                }
            }
            CutStridePolicy::TwoTier {
                head_tokens,
                head_stride,
                tail_stride,
            } => {
                if head_stride == 0 {
                    return Err(PackError::Semantics("cut head stride is zero"));
                }
                if tail_stride == 0 {
                    return Err(PackError::Semantics("cut tail stride is zero"));
                }
                if head_tokens % head_stride != 0 {
                    return Err(PackError::Semantics(
                        "cut head region is not a multiple of the head stride",
                    ));
                }
                if head_tokens % tail_stride != 0 {
                    return Err(PackError::Semantics(
                        "cut head region is not a multiple of the tail stride",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Stride of the cut that starts at `token_offset`, the absolute token
    /// position of the chain tip.
    fn stride_at(&self, token_offset: u64) -> u32 {
        match *self {
            CutStridePolicy::Flat(stride) => stride,
            CutStridePolicy::TwoTier {
                head_tokens,
                head_stride,
                tail_stride,
            } => {
                if token_offset < u64::from(head_tokens) {
                    head_stride
                } else {
                    tail_stride
                }
            }
        }
    }
}

/// Streaming rolling keyed chain over cut boundaries: `h_0` binds the
/// namespace root, then each completed cut folds its canonical little-endian
/// u32 token bytes into the chain tip.  One `Id32` is emitted per cut
/// boundary, in chain order; a trailing partial cut is buffered, never
/// emitted.  Derivation fails closed once the chain would exceed
/// `MAX_CUT_CHAIN_CUTS` cuts, and the deriver stays poisoned afterwards.
pub struct CutChainDeriver {
    key: [u8; 32],
    tip: Id32,
    pending: Vec<u8>,
    pending_front: usize,
    consumed_tokens: u64,
    cuts: usize,
    policy: CutStridePolicy,
    poisoned: bool,
}

impl CutChainDeriver {
    pub fn new(
        store_key: &[u8; 32],
        namespace: &Id32,
        policy: CutStridePolicy,
    ) -> Result<Self, PackError> {
        policy.validate()?;
        Ok(Self {
            key: *store_key,
            tip: hmac(store_key, &[b"kvpack/v1/cut-chain\0", namespace]),
            pending: Vec::new(),
            pending_front: 0,
            consumed_tokens: 0,
            cuts: 0,
            policy,
            poisoned: false,
        })
    }

    /// Push the next token slice; returns the identities of every cut
    /// boundary the slice completes, in chain order.
    pub fn push_tokens(&mut self, tokens: &[u32]) -> Result<Vec<Id32>, PackError> {
        if self.poisoned {
            return Err(PackError::Poisoned("cut chain deriver is poisoned"));
        }
        self.pending.reserve(tokens.len().saturating_mul(4));
        for token in tokens {
            self.pending.extend_from_slice(&token.to_le_bytes());
        }
        let mut emitted = Vec::new();
        loop {
            let stride = self.policy.stride_at(self.consumed_tokens);
            let stride_bytes = stride as usize * 4;
            if self.pending.len() - self.pending_front < stride_bytes {
                break;
            }
            if self.cuts == MAX_CUT_CHAIN_CUTS {
                self.poisoned = true;
                return Err(PackError::Bounds("cut chain exceeds 512 cuts"));
            }
            self.tip = hmac(
                &self.key,
                &[
                    b"kvpack/v1/cut-chain\0",
                    &self.tip,
                    &self.pending[self.pending_front..self.pending_front + stride_bytes],
                ],
            );
            self.pending_front += stride_bytes;
            self.consumed_tokens += u64::from(stride);
            self.cuts += 1;
            emitted.push(self.tip);
        }
        // Compact the consumed prefix once per call instead of draining it
        // per emitted cut.
        if self.pending_front == self.pending.len() {
            self.pending.clear();
        } else if self.pending_front > 0 {
            self.pending.drain(..self.pending_front);
        }
        self.pending_front = 0;
        Ok(emitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn id_from_hex(text: &str) -> Id32 {
        let mut id = [0u8; 32];
        for (index, byte) in id.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("golden hex");
        }
        id
    }

    /// Independent in-test recomputation of the chain straight from the
    /// `hmac` crate, duplicating the framing without touching the deriver.
    fn reference_chain(store_key: &[u8; 32], namespace: &Id32, cuts: &[&[u32]]) -> Vec<Id32> {
        let mut mac = HmacSha256::new_from_slice(store_key).expect("HMAC accepts every key length");
        mac.update(b"kvpack/v1/cut-chain\0");
        mac.update(namespace);
        let mut tip: Id32 = mac.finalize().into_bytes().into();
        let mut result = Vec::with_capacity(cuts.len());
        for cut in cuts {
            let mut mac =
                HmacSha256::new_from_slice(store_key).expect("HMAC accepts every key length");
            mac.update(b"kvpack/v1/cut-chain\0");
            mac.update(&tip);
            for token in *cut {
                mac.update(&token.to_le_bytes());
            }
            tip = mac.finalize().into_bytes().into();
            result.push(tip);
        }
        result
    }

    #[test]
    fn flat_chain_matches_golden_vectors() {
        // Golden hex digests recomputed with an independent Python HMAC
        // implementation over the same framing.
        let golden = [
            "9719e6c4c15299bcd372bbc82fa7bd66da1ce6eff0d9e560f88ee950f67a4bdf",
            "01f3e699e6163a5476de16c269e4b42e9101fb16278fe8db9857b3e8d00a07ca",
            "4fed72eb6d5348b472f4c4e9ea2a046d4a7e3036cab6f4b629a68eaa7deed563",
            "074af1e1afbc34306fed857383ce96b331d042f270d176d791b5c3bd456cf05d",
        ];
        let tokens: Vec<u32> = (0..1024).collect();
        let mut deriver =
            CutChainDeriver::new(&key(7), &key(3), CutStridePolicy::Flat(256)).unwrap();
        let emitted = deriver.push_tokens(&tokens).unwrap();
        let expected: Vec<Id32> = golden.iter().map(|text| id_from_hex(text)).collect();
        assert_eq!(emitted, expected);
        let cuts: Vec<&[u32]> = tokens.chunks(256).collect();
        assert_eq!(emitted, reference_chain(&key(7), &key(3), &cuts));
    }

    #[test]
    fn two_tier_chain_emits_at_head_and_tail_boundaries() {
        let policy = CutStridePolicy::TwoTier {
            head_tokens: 512,
            head_stride: 64,
            tail_stride: 256,
        };
        let boundaries = [
            64usize, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 1280, 1536,
        ];
        let tokens: Vec<u32> = (0..1536).collect();
        // Odd-sized pushes prove the streaming buffer reassembles cuts and
        // emits exactly when a boundary completes.
        let mut deriver = CutChainDeriver::new(&key(9), &key(5), policy).unwrap();
        let mut emitted = Vec::new();
        let mut pushed = 0usize;
        for chunk in tokens.chunks(100) {
            emitted.extend(deriver.push_tokens(chunk).unwrap());
            pushed += chunk.len();
            let expected = boundaries
                .iter()
                .filter(|boundary| **boundary <= pushed)
                .count();
            assert_eq!(emitted.len(), expected);
        }
        assert_eq!(emitted.len(), boundaries.len());
        let mut cuts: Vec<&[u32]> = Vec::new();
        let mut start = 0usize;
        for boundary in boundaries {
            cuts.push(&tokens[start..boundary]);
            start = boundary;
        }
        assert_eq!(emitted, reference_chain(&key(9), &key(5), &cuts));
    }

    #[test]
    fn flat_zero_stride_fails_closed() {
        assert_eq!(
            CutStridePolicy::Flat(0).validate().unwrap_err(),
            PackError::Semantics("cut stride is zero")
        );
        assert_eq!(
            CutChainDeriver::new(&key(7), &key(3), CutStridePolicy::Flat(0)).err(),
            Some(PackError::Semantics("cut stride is zero"))
        );
    }

    #[test]
    fn two_tier_zero_head_stride_fails_closed() {
        let policy = CutStridePolicy::TwoTier {
            head_tokens: 512,
            head_stride: 0,
            tail_stride: 256,
        };
        assert_eq!(
            policy.validate().unwrap_err(),
            PackError::Semantics("cut head stride is zero")
        );
    }

    #[test]
    fn two_tier_zero_tail_stride_fails_closed() {
        let policy = CutStridePolicy::TwoTier {
            head_tokens: 512,
            head_stride: 64,
            tail_stride: 0,
        };
        assert_eq!(
            policy.validate().unwrap_err(),
            PackError::Semantics("cut tail stride is zero")
        );
    }

    #[test]
    fn two_tier_head_region_must_be_a_multiple_of_the_head_stride() {
        let policy = CutStridePolicy::TwoTier {
            head_tokens: 100,
            head_stride: 64,
            tail_stride: 50,
        };
        assert_eq!(
            policy.validate().unwrap_err(),
            PackError::Semantics("cut head region is not a multiple of the head stride")
        );
    }

    #[test]
    fn two_tier_head_region_must_be_a_multiple_of_the_tail_stride() {
        let policy = CutStridePolicy::TwoTier {
            head_tokens: 192,
            head_stride: 64,
            tail_stride: 100,
        };
        assert_eq!(
            policy.validate().unwrap_err(),
            PackError::Semantics("cut head region is not a multiple of the tail stride")
        );
    }

    #[test]
    fn valid_policies_validate() {
        CutStridePolicy::Flat(256).validate().unwrap();
        CutStridePolicy::TwoTier {
            head_tokens: 4096,
            head_stride: 64,
            tail_stride: 256,
        }
        .validate()
        .unwrap();
        // An empty head region degenerates to the flat tail stride.
        CutStridePolicy::TwoTier {
            head_tokens: 0,
            head_stride: 64,
            tail_stride: 256,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn chain_fails_closed_at_513_cuts() {
        let tokens: Vec<u32> = (0..513).collect();
        let mut deriver = CutChainDeriver::new(&key(7), &key(3), CutStridePolicy::Flat(1)).unwrap();
        assert_eq!(deriver.push_tokens(&tokens[..512]).unwrap().len(), 512);
        assert_eq!(
            deriver.push_tokens(&tokens[512..]).unwrap_err(),
            PackError::Bounds("cut chain exceeds 512 cuts")
        );
        assert_eq!(
            deriver.push_tokens(&tokens[..1]).unwrap_err(),
            PackError::Poisoned("cut chain deriver is poisoned")
        );
    }

    #[test]
    fn empty_and_partial_input_emit_nothing() {
        let tokens: Vec<u32> = (0..266).collect();
        let mut deriver =
            CutChainDeriver::new(&key(7), &key(3), CutStridePolicy::Flat(256)).unwrap();
        assert!(deriver.push_tokens(&[]).unwrap().is_empty());
        assert!(deriver.push_tokens(&tokens[..100]).unwrap().is_empty());
        let emitted = deriver.push_tokens(&tokens[100..256]).unwrap();
        assert_eq!(
            emitted,
            reference_chain(&key(7), &key(3), &[&tokens[..256]])
        );
        // The trailing partial cut is buffered, never emitted.
        assert!(deriver.push_tokens(&tokens[256..]).unwrap().is_empty());
    }
}
