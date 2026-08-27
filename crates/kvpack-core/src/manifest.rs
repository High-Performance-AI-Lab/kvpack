use std::cmp::Ordering;

use crate::canonical::{Decoder, Encoder};
use crate::identity::{decode_semantic, encode_semantic};
use crate::{
    AuxiliaryInputId, Id32, InputCutId, PackError, RepresentationFamilyId, MANIFEST_MAGIC,
    MAX_ATOMIC_GROUPS, MAX_CHUNKS_PER_STATE, MAX_MANIFEST_BYTES, MAX_RANK, MAX_STATES,
    SCHEMA_MAGIC, STATE_SCHEMA_MAGIC, WIRE_VERSION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u64; MAX_RANK],
    rank: u8,
}

impl Shape {
    pub fn new(dims: &[u64]) -> Result<Self, PackError> {
        if dims.is_empty() || dims.len() > MAX_RANK {
            return Err(PackError::Bounds("state rank must be in 1..=8"));
        }
        if dims.contains(&0) {
            return Err(PackError::Bounds("state dimensions must be nonzero"));
        }
        let mut slots = [0; MAX_RANK];
        slots[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            dims: slots,
            rank: dims.len() as u8,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    pub fn dims(&self) -> &[u64] {
        &self.dims[..self.rank()]
    }

    pub fn element_count(&self) -> Result<u64, PackError> {
        self.dims().iter().try_fold(1u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(PackError::Bounds("shape element count overflows u64"))
        })
    }

    pub(crate) fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let rank = input.u8()? as usize;
        if rank == 0 || rank > MAX_RANK {
            return Err(PackError::Bounds("state rank must be in 1..=8"));
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(input.u64()?);
        }
        Self::new(&dims)
    }

    pub(crate) fn encode(&self, out: &mut Encoder) {
        out.u8(self.rank);
        for dimension in self.dims() {
            out.u64(*dimension);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateKey {
    pub layer: u32,
    pub state_name: String,
}

impl StateKey {
    pub fn new(layer: u32, state_name: impl Into<String>) -> Self {
        Self {
            layer,
            state_name: state_name.into(),
        }
    }

    pub(crate) fn encode(&self, out: &mut Encoder) -> Result<(), PackError> {
        out.u32(self.layer);
        out.string(&self.state_name)
    }

    pub(crate) fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        Ok(Self {
            layer: input.u32()?,
            state_name: input.string()?,
        })
    }
}

impl Ord for StateKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.layer
            .cmp(&other.layer)
            .then_with(|| self.state_name.as_bytes().cmp(other.state_name.as_bytes()))
    }
}

impl PartialOrd for StateKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkSpan {
    pub token_start: u64,
    pub token_count: u64,
    pub plaintext_offset: u64,
    pub plaintext_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub chunk_id: Id32,
    pub object_key: Id32,
    pub object_digest: Id32,
    pub key_epoch: u64,
    pub plaintext_bytes: u32,
    pub object_bytes: u32,
}

/// Caller-visible dynamic state declaration.  Byte totals, chunk spans, and
/// schema identities are derived by the writer from the completed stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDeclaration {
    pub key: StateKey,
    pub full_shape: Shape,
    pub segment_shape: Shape,
    pub strides: Vec<u64>,
    pub logical_start: u64,
    pub logical_count: u64,
    pub absolute_position: u64,
    pub window: u64,
    pub atomic_group: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizedStateSchema {
    pub key: StateKey,
    pub full_shape: Shape,
    pub segment_shape: Shape,
    pub strides: Vec<u64>,
    pub logical_start: u64,
    pub logical_count: u64,
    pub physical_offset_bytes: u64,
    pub physical_span_bytes: u64,
    pub complete_physical_bytes: u64,
    pub absolute_position: u64,
    pub window: u64,
    pub chunk_spans: Vec<ChunkSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateManifest {
    pub key: StateKey,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicGroup {
    pub id: u32,
    pub states: Vec<StateKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestKind {
    Full,
    Delta {
        parent: Id32,
        parent_cut: InputCutId,
        depth: u8,
    },
}

impl ManifestKind {
    pub fn depth(&self) -> u8 {
        match self {
            Self::Full => 0,
            Self::Delta { depth, .. } => *depth,
        }
    }

    pub fn parent(&self) -> Option<&Id32> {
        match self {
            Self::Full => None,
            Self::Delta { parent, .. } => Some(parent),
        }
    }

    pub fn parent_cut(&self) -> Option<&InputCutId> {
        match self {
            Self::Full => None,
            Self::Delta { parent_cut, .. } => Some(parent_cut),
        }
    }

    fn encode(&self, out: &mut Encoder) {
        match self {
            Self::Full => {
                out.u8(0);
                out.u8(0);
                out.u16(0);
            }
            Self::Delta {
                parent,
                parent_cut,
                depth,
            } => {
                out.u8(1);
                out.u8(*depth);
                out.u16(0);
                out.id(parent);
                encode_input_cut(parent_cut, out);
            }
        }
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let tag = input.u8()?;
        let depth = input.u8()?;
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "manifest-kind reserved field is nonzero",
            ));
        }
        match tag {
            0 if depth == 0 => Ok(Self::Full),
            0 => Err(PackError::Reserved("full manifest carries a delta depth")),
            1 => Ok(Self::Delta {
                parent: input.id()?,
                parent_cut: decode_input_cut(input)?,
                depth,
            }),
            value => Err(PackError::UnknownEnum {
                what: "manifest kind",
                value: value as u64,
            }),
        }
    }
}

/// Complete concrete schema for one exact cut.  Its canonical digest is the
/// `RealizedCutSchemaId`; the full preimage is embedded in the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizedCutSchemaId {
    pub kind: ManifestKind,
    pub states: Vec<RealizedStateSchema>,
    pub atomic_groups: Vec<AtomicGroup>,
    pub segment_restored_bytes: u64,
    pub complete_restored_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestDeclaration {
    pub semantic_model: crate::SemanticModelId,
    pub input_tokens: Vec<u32>,
    pub auxiliary_inputs: Vec<AuxiliaryInputId>,
    pub family: RepresentationFamilyId,
    pub kind: ManifestKind,
    pub states: Vec<StateDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutManifest {
    pub tenant_namespace: Id32,
    pub key_epoch: u64,
    pub semantic_model: crate::SemanticModelId,
    pub input_cut: InputCutId,
    pub family: RepresentationFamilyId,
    pub realized_schema: RealizedCutSchemaId,
    pub states: Vec<StateManifest>,
}

fn encode_input_cut(value: &InputCutId, out: &mut Encoder) {
    out.id(&value.token_root);
    out.id(&value.auxiliary_input_root);
    out.u64(value.token_count);
}

fn decode_input_cut(input: &mut Decoder<'_>) -> Result<InputCutId, PackError> {
    Ok(InputCutId {
        token_root: input.id()?,
        auxiliary_input_root: input.id()?,
        token_count: input.u64()?,
    })
}

impl StateDeclaration {
    pub fn canonical_schema_bytes(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(STATE_SCHEMA_MAGIC);
        out.u16(WIRE_VERSION);
        out.u16(0);
        self.encode(&mut out)?;
        Ok(out.finish())
    }

    pub fn decode_canonical_schema(bytes: &[u8]) -> Result<Self, PackError> {
        let mut input = Decoder::new(bytes, STATE_SCHEMA_MAGIC)?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported state-schema version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "state-schema reserved field is nonzero",
            ));
        }
        let value = Self::decode(&mut input)?;
        input.finish()?;
        if value.canonical_schema_bytes()? != bytes {
            return Err(PackError::Reserved(
                "state-schema encoding is not canonical",
            ));
        }
        Ok(value)
    }

    fn encode(&self, out: &mut Encoder) -> Result<(), PackError> {
        self.key.encode(out)?;
        self.full_shape.encode(out);
        self.segment_shape.encode(out);
        out.u8(u8::try_from(self.strides.len())
            .map_err(|_| PackError::Bounds("too many state strides"))?);
        for stride in &self.strides {
            out.u64(*stride);
        }
        out.u64(self.logical_start);
        out.u64(self.logical_count);
        out.u64(self.absolute_position);
        out.u64(self.window);
        out.u32(self.atomic_group);
        Ok(())
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let key = StateKey::decode(input)?;
        let full_shape = Shape::decode(input)?;
        let segment_shape = Shape::decode(input)?;
        let stride_count = input.u8()? as usize;
        if stride_count > MAX_RANK {
            return Err(PackError::Bounds("too many state strides"));
        }
        let mut strides = Vec::with_capacity(stride_count);
        for _ in 0..stride_count {
            strides.push(input.u64()?);
        }
        Ok(Self {
            key,
            full_shape,
            segment_shape,
            strides,
            logical_start: input.u64()?,
            logical_count: input.u64()?,
            absolute_position: input.u64()?,
            window: input.u64()?,
            atomic_group: input.u32()?,
        })
    }
}

impl RealizedStateSchema {
    fn encode(&self, out: &mut Encoder) -> Result<(), PackError> {
        self.key.encode(out)?;
        self.full_shape.encode(out);
        self.segment_shape.encode(out);
        out.u8(u8::try_from(self.strides.len())
            .map_err(|_| PackError::Bounds("too many state strides"))?);
        for stride in &self.strides {
            out.u64(*stride);
        }
        out.u64(self.logical_start);
        out.u64(self.logical_count);
        out.u64(self.physical_offset_bytes);
        out.u64(self.physical_span_bytes);
        out.u64(self.complete_physical_bytes);
        out.u64(self.absolute_position);
        out.u64(self.window);
        out.u32(
            u32::try_from(self.chunk_spans.len())
                .map_err(|_| PackError::Bounds("too many chunk spans"))?,
        );
        for span in &self.chunk_spans {
            out.u64(span.token_start);
            out.u64(span.token_count);
            out.u64(span.plaintext_offset);
            out.u32(span.plaintext_bytes);
        }
        Ok(())
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let key = StateKey::decode(input)?;
        let full_shape = Shape::decode(input)?;
        let segment_shape = Shape::decode(input)?;
        let stride_count = input.u8()? as usize;
        if stride_count > MAX_RANK {
            return Err(PackError::Bounds("too many state strides"));
        }
        let mut strides = Vec::with_capacity(stride_count);
        for _ in 0..stride_count {
            strides.push(input.u64()?);
        }
        let logical_start = input.u64()?;
        let logical_count = input.u64()?;
        let physical_offset_bytes = input.u64()?;
        let physical_span_bytes = input.u64()?;
        let complete_physical_bytes = input.u64()?;
        let absolute_position = input.u64()?;
        let window = input.u64()?;
        let chunk_count = input.u32()? as usize;
        if chunk_count == 0 || chunk_count > MAX_CHUNKS_PER_STATE {
            return Err(PackError::Bounds(
                "state chunk-span count is outside bounds",
            ));
        }
        let mut chunk_spans = Vec::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            chunk_spans.push(ChunkSpan {
                token_start: input.u64()?,
                token_count: input.u64()?,
                plaintext_offset: input.u64()?,
                plaintext_bytes: input.u32()?,
            });
        }
        Ok(Self {
            key,
            full_shape,
            segment_shape,
            strides,
            logical_start,
            logical_count,
            physical_offset_bytes,
            physical_span_bytes,
            complete_physical_bytes,
            absolute_position,
            window,
            chunk_spans,
        })
    }
}

impl AtomicGroup {
    fn encode(&self, out: &mut Encoder) -> Result<(), PackError> {
        out.u32(self.id);
        out.u32(
            u32::try_from(self.states.len())
                .map_err(|_| PackError::Bounds("too many atomic-group states"))?,
        );
        for state in &self.states {
            state.encode(out)?;
        }
        Ok(())
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let id = input.u32()?;
        let state_count = input.u32()? as usize;
        if state_count == 0 || state_count > MAX_STATES {
            return Err(PackError::Bounds(
                "atomic-group state count is outside bounds",
            ));
        }
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            states.push(StateKey::decode(input)?);
        }
        Ok(Self { id, states })
    }
}

impl RealizedCutSchemaId {
    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(SCHEMA_MAGIC);
        out.u16(WIRE_VERSION);
        out.u16(0);
        self.encode_body(&mut out)?;
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        let mut input = Decoder::new(bytes, SCHEMA_MAGIC)?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported realized-schema version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "realized-schema reserved field is nonzero",
            ));
        }
        let value = Self::decode_body(&mut input)?;
        input.finish()?;
        if value.encode_canonical()? != bytes {
            return Err(PackError::Reserved(
                "realized-schema encoding is not canonical",
            ));
        }
        Ok(value)
    }

    fn encode_body(&self, out: &mut Encoder) -> Result<(), PackError> {
        self.kind.encode(out);
        out.u32(
            u32::try_from(self.states.len())
                .map_err(|_| PackError::Bounds("too many realized states"))?,
        );
        for state in &self.states {
            state.encode(out)?;
        }
        out.u32(
            u32::try_from(self.atomic_groups.len())
                .map_err(|_| PackError::Bounds("too many atomic groups"))?,
        );
        for group in &self.atomic_groups {
            group.encode(out)?;
        }
        out.u64(self.segment_restored_bytes);
        out.u64(self.complete_restored_bytes);
        Ok(())
    }

    fn decode_body(input: &mut Decoder<'_>) -> Result<Self, PackError> {
        let kind = ManifestKind::decode(input)?;
        let state_count = input.u32()? as usize;
        if state_count == 0 || state_count > MAX_STATES {
            return Err(PackError::Bounds("realized state count is outside bounds"));
        }
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            states.push(RealizedStateSchema::decode(input)?);
        }
        let group_count = input.u32()? as usize;
        if group_count == 0 || group_count > MAX_ATOMIC_GROUPS {
            return Err(PackError::Bounds("atomic-group count is outside bounds"));
        }
        let mut atomic_groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            atomic_groups.push(AtomicGroup::decode(input)?);
        }
        Ok(Self {
            kind,
            states,
            atomic_groups,
            segment_restored_bytes: input.u64()?,
            complete_restored_bytes: input.u64()?,
        })
    }
}

impl CutManifest {
    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        let mut out = Encoder::new(MANIFEST_MAGIC);
        out.u16(WIRE_VERSION);
        out.u16(0);
        out.id(&self.tenant_namespace);
        out.u64(self.key_epoch);
        encode_semantic(&self.semantic_model, &mut out);
        encode_input_cut(&self.input_cut, &mut out);
        self.family.encode_body(&mut out)?;
        self.realized_schema.encode_body(&mut out)?;
        out.u32(
            u32::try_from(self.states.len())
                .map_err(|_| PackError::Bounds("too many manifest states"))?,
        );
        for state in &self.states {
            state.key.encode(&mut out)?;
            out.u32(
                u32::try_from(state.chunks.len())
                    .map_err(|_| PackError::Bounds("too many state chunks"))?,
            );
            for chunk in &state.chunks {
                out.id(&chunk.chunk_id);
                out.id(&chunk.object_key);
                out.id(&chunk.object_digest);
                out.u64(chunk.key_epoch);
                out.u32(chunk.plaintext_bytes);
                out.u32(chunk.object_bytes);
            }
        }
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(PackError::Bounds("manifest exceeds production bound"));
        }
        let mut input = Decoder::new(bytes, MANIFEST_MAGIC)?;
        if input.u16()? != WIRE_VERSION {
            return Err(PackError::BadMagic("unsupported manifest version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved("manifest reserved field is nonzero"));
        }
        let tenant_namespace = input.id()?;
        let key_epoch = input.u64()?;
        let semantic_model = decode_semantic(&mut input)?;
        let input_cut = decode_input_cut(&mut input)?;
        let family = RepresentationFamilyId::decode_body(&mut input)?;
        let realized_schema = RealizedCutSchemaId::decode_body(&mut input)?;
        let state_count = input.u32()? as usize;
        if state_count == 0 || state_count > MAX_STATES {
            return Err(PackError::Bounds("manifest state count is outside bounds"));
        }
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            let key = StateKey::decode(&mut input)?;
            let chunk_count = input.u32()? as usize;
            if chunk_count == 0 || chunk_count > MAX_CHUNKS_PER_STATE {
                return Err(PackError::Bounds("state chunk count is outside bounds"));
            }
            let mut chunks = Vec::with_capacity(chunk_count);
            for _ in 0..chunk_count {
                chunks.push(ChunkRef {
                    chunk_id: input.id()?,
                    object_key: input.id()?,
                    object_digest: input.id()?,
                    key_epoch: input.u64()?,
                    plaintext_bytes: input.u32()?,
                    object_bytes: input.u32()?,
                });
            }
            states.push(StateManifest { key, chunks });
        }
        input.finish()?;
        let value = Self {
            tenant_namespace,
            key_epoch,
            semantic_model,
            input_cut,
            family,
            realized_schema,
            states,
        };
        if value.encode_canonical()? != bytes {
            return Err(PackError::Reserved("manifest encoding is not canonical"));
        }
        Ok(value)
    }
}

pub(crate) fn is_zero_id(id: &Id32) -> bool {
    id.iter().all(|byte| *byte == 0)
}
