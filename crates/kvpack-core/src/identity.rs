//! Typed, canonical identity components for the closed production-v1 model.
//!
//! Static representation compatibility and one realized cut are deliberately
//! separate.  Catalog generations, locations, tombstones, and publication
//! state are not durable identity material.

use crate::canonical::{Decoder, Encoder};
use crate::{
    CacheKind, Codec, DType, Id32, Layout, PackError, RepresentationMode, StateKey, TokenAxisRule,
    FAMILY_MAGIC, MAX_DEPENDENCIES_PER_STATE, MAX_RANK, MAX_STATES, MAX_STATE_NAME_BYTES,
    WIRE_VERSION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticModelId {
    pub weights_config: Id32,
    pub adapters: Id32,
    pub tokenizer_template: Id32,
    pub position_semantics: Id32,
    pub qualified_math: Id32,
}

/// The identity of exactly one input prefix.
///
/// `token_root` is the final node of the keyed fixed-width token chain.  No
/// token witness is retained.  The token count is part of the identity rather
/// than an independently caller-selected logical cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputCutId {
    pub token_root: Id32,
    pub auxiliary_input_root: Id32,
    pub token_count: u64,
}

/// A typed opaque auxiliary identity.  Both fields are already identities;
/// raw prompts, images, embeddings, or other user content are never retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AuxiliaryInputId {
    pub type_id: Id32,
    pub value_id: Id32,
}

/// One dimension in a static family shape.  Exactly one dimension is the
/// logical token extent; all remaining dimensions are immutable and nonzero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StaticDimension {
    Token,
    Fixed(u64),
}

/// Static compatibility for one ordinary-KV state stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FamilyState {
    pub key: StateKey,
    pub cache_kind: CacheKind,
    pub dtype: DType,
    pub codec: Codec,
    pub codec_version: u16,
    pub layout: Layout,
    pub token_axis_rule: TokenAxisRule,
    pub token_axis: u8,
    pub elements_per_token: u64,
    pub dimensions: Vec<StaticDimension>,
    pub dependencies: Vec<StateKey>,
}

/// Static engine/cache compatibility shared by every exact cut in a family.
///
/// Concrete token extents, shapes, strides, ranges, chunks, and restored byte
/// counts are intentionally absent and belong to `RealizedCutSchemaId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepresentationFamilyId {
    pub engine_cache_abi: Id32,
    pub mode: RepresentationMode,
    pub page_size_tokens: u32,
    pub topology: Id32,
    pub shard_map: Id32,
    pub states: Vec<FamilyState>,
}

pub(crate) fn encode_semantic(value: &SemanticModelId, out: &mut Encoder) {
    out.id(&value.weights_config);
    out.id(&value.adapters);
    out.id(&value.tokenizer_template);
    out.id(&value.position_semantics);
    out.id(&value.qualified_math);
}

pub(crate) fn decode_semantic(input: &mut Decoder<'_>) -> Result<SemanticModelId, PackError> {
    Ok(SemanticModelId {
        weights_config: input.id()?,
        adapters: input.id()?,
        tokenizer_template: input.id()?,
        position_semantics: input.id()?,
        qualified_math: input.id()?,
    })
}

impl FamilyState {
    pub(crate) fn encode(&self, out: &mut Encoder) -> Result<(), PackError> {
        self.key.encode(out)?;
        out.u16(self.cache_kind.wire());
        out.u16(self.dtype.wire());
        out.u16(self.codec.wire());
        out.u16(self.codec_version);
        out.u16(self.layout.wire());
        out.u16(self.token_axis_rule.wire());
        out.u8(self.token_axis);
        out.u8(0);
        out.u64(self.elements_per_token);
        out.u8(u8::try_from(self.dimensions.len())
            .map_err(|_| PackError::Bounds("family state rank exceeds u8"))?);
        for dimension in &self.dimensions {
            out.u64(match dimension {
                StaticDimension::Token => 0,
                StaticDimension::Fixed(value) => *value,
            });
        }
        out.u16(
            u16::try_from(self.dependencies.len())
                .map_err(|_| PackError::Bounds("too many state dependencies"))?,
        );
        for dependency in &self.dependencies {
            dependency.encode(out)?;
        }
        Ok(())
    }

    pub(crate) fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let key = StateKey::decode(input)?;
        let cache_kind = CacheKind::from_wire(input.u16()?)?;
        let dtype = DType::from_wire(input.u16()?)?;
        let codec = Codec::from_wire(input.u16()?)?;
        let codec_version = input.u16()?;
        let layout = Layout::from_wire(input.u16()?)?;
        let token_axis_rule = TokenAxisRule::from_wire(input.u16()?)?;
        let token_axis = input.u8()?;
        if input.u8()? != 0 {
            return Err(PackError::Reserved("family state reserved byte is nonzero"));
        }
        let elements_per_token = input.u64()?;
        let rank = input.u8()? as usize;
        if rank == 0 || rank > MAX_RANK {
            return Err(PackError::Bounds("family state rank is outside 1..=8"));
        }
        let mut dimensions = Vec::with_capacity(rank);
        for _ in 0..rank {
            let value = input.u64()?;
            dimensions.push(if value == 0 {
                StaticDimension::Token
            } else {
                StaticDimension::Fixed(value)
            });
        }
        let dependency_count = input.u16()? as usize;
        if dependency_count > MAX_DEPENDENCIES_PER_STATE {
            return Err(PackError::Bounds("too many state dependencies"));
        }
        let mut dependencies = Vec::with_capacity(dependency_count);
        for _ in 0..dependency_count {
            dependencies.push(StateKey::decode(input)?);
        }
        Ok(Self {
            key,
            cache_kind,
            dtype,
            codec,
            codec_version,
            layout,
            token_axis_rule,
            token_axis,
            elements_per_token,
            dimensions,
            dependencies,
        })
    }
}

impl RepresentationFamilyId {
    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(FAMILY_MAGIC);
        out.u16(WIRE_VERSION);
        out.u16(0);
        self.encode_body(&mut out)?;
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        let mut input = Decoder::new(bytes, FAMILY_MAGIC)?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported family version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved("family reserved field is nonzero"));
        }
        let value = Self::decode_body(&mut input)?;
        input.finish()?;
        if value.encode_canonical()? != bytes {
            return Err(PackError::Reserved("family encoding is not canonical"));
        }
        Ok(value)
    }

    pub(crate) fn encode_body(&self, out: &mut Encoder) -> Result<(), PackError> {
        out.id(&self.engine_cache_abi);
        out.u16(self.mode.wire());
        out.u16(0);
        out.u32(self.page_size_tokens);
        out.id(&self.topology);
        out.id(&self.shard_map);
        out.u32(
            u32::try_from(self.states.len())
                .map_err(|_| PackError::Bounds("too many family states"))?,
        );
        for state in &self.states {
            state.encode(out)?;
        }
        Ok(())
    }

    pub(crate) fn decode_body(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let engine_cache_abi = input.id()?;
        let mode = RepresentationMode::from_wire(input.u16()?)?;
        if input.u16()? != 0 {
            return Err(PackError::Reserved("family reserved field is nonzero"));
        }
        let page_size_tokens = input.u32()?;
        let topology = input.id()?;
        let shard_map = input.id()?;
        let state_count = input.u32()? as usize;
        if state_count == 0 || state_count > MAX_STATES {
            return Err(PackError::Bounds("family state count is outside bounds"));
        }
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            states.push(FamilyState::decode(input)?);
        }
        Ok(Self {
            engine_cache_abi,
            mode,
            page_size_tokens,
            topology,
            shard_map,
            states,
        })
    }
}

pub(crate) fn validate_state_key(key: &StateKey) -> Result<(), PackError> {
    if key.state_name.is_empty()
        || key.state_name.len() > MAX_STATE_NAME_BYTES
        || key.state_name.as_bytes().contains(&0)
    {
        return Err(PackError::Semantics("invalid state name"));
    }
    Ok(())
}
