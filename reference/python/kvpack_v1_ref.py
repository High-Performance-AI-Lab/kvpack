"""Independent Python 3 standard-library production-v1 reference codec.

This module intentionally shares no generated schema or implementation code
with Rust.  It is a pre-freeze conformance oracle for canonical objects,
identities, codec frames, authenticated chunk objects, and pack envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import secrets
import struct
from typing import Callable, Iterable, Sequence


ALIGNMENT = 4096
CHUNK_HEADER_BYTES = ALIGNMENT
PACK_HEADER_BYTES = ALIGNMENT
PACK_FOOTER_BYTES = ALIGNMENT
MAX_CHUNK_PLAINTEXT = 4 * 1024 * 1024
CODEC_FRAME_HEADER_BYTES = 16
MAX_CODEC_OVERHEAD = CODEC_FRAME_HEADER_BYTES + (MAX_CHUNK_PLAINTEXT + 127) // 128
MAX_CHUNK_OBJECT_BYTES = (
    CHUNK_HEADER_BYTES + MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD + 16 + ALIGNMENT
)
MAX_RANK = 8
MAX_DELTA_DEPTH = 7
PREFIX_BLOCK_TOKENS = 256
MAX_STATE_NAME_BYTES = 255
MAX_STATES = 65_536
MAX_ATOMIC_GROUPS = MAX_STATES
MAX_CHUNKS_PER_STATE = 65_536
MAX_DEPENDENCIES_PER_STATE = 64
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
WIRE_VERSION = 1

PACK_MAGIC = b"KVPKP1\0\0"
CHUNK_MAGIC = b"KVCHK1\0\0"
FOOTER_MAGIC = b"KVCMT1\0\0"
MANIFEST_MAGIC = b"KVMNF1\0\0"
FAMILY_MAGIC = b"KVFAM1\0\0"
SCHEMA_MAGIC = b"KVRCS1\0\0"
STATE_SCHEMA_MAGIC = b"KVSTS1\0\0"
RAW_FRAME_MAGIC = b"KVRAW1\0\0"
LOSSLESS_FRAME_MAGIC = b"KVRLE1\0\0"
STATS_SIDECAR_MAGIC = b"KVSSC1\0\0"

CHUNK_HEADER_SIDECAR_OFFSET = 236
MAX_STATS_SIDECAR_BYTES = CHUNK_HEADER_BYTES - CHUNK_HEADER_SIDECAR_OFFSET - 2
MAX_SIDECAR_CHANNELS = 512
MAX_SIDECAR_TOKENS = 768
MAX_SINK_SCORES = 8

CACHE_ORDINARY_KV = 1
CODEC_RAW = 1
CODEC_LOSSLESS = 2
LAYOUT_CONTIGUOUS = 1
LAYOUT_STRIDED = 2
MODE_NATIVE = 1
MODE_PORTABLE = 2
TOKEN_AXIS_DIRECT = 1
TOKEN_AXIS_GATHER = 2
DTYPE_WIDTHS = {1: 1, 2: 4, 3: 1, 4: 2, 5: 4, 6: 2, 7: 2, 8: 4, 9: 8}


class WireError(Exception):
    """Stable cross-language verdict category and message."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message

    def verdict(self) -> str:
        return f"{self.category}:{self.message}"


def _fail(category: str, message: str) -> None:
    raise WireError(category, message)


def _check_uint(value: int, bits: int, what: str) -> int:
    if not isinstance(value, int) or value < 0 or value >= 1 << bits:
        _fail("bounds", f"{what} is outside u{bits} range")
    return value


def _id32(value: bytes, what: str = "identity") -> bytes:
    value = bytes(value)
    if len(value) != 32:
        _fail("bounds", f"{what} must be 32 bytes")
    return value


def _u8(value: int) -> bytes:
    return struct.pack("<B", _check_uint(value, 8, "integer"))


def _u16(value: int) -> bytes:
    return struct.pack("<H", _check_uint(value, 16, "integer"))


def _u32(value: int) -> bytes:
    return struct.pack("<I", _check_uint(value, 32, "integer"))


def _u64(value: int) -> bytes:
    return struct.pack("<Q", _check_uint(value, 64, "integer"))


class Reader:
    def __init__(self, data: bytes, magic: bytes | None = None):
        self.data = memoryview(bytes(data))
        self.offset = 0
        if magic is not None:
            if len(self.data) < 8:
                _fail("truncated", "truncated canonical object")
            if bytes(self.data[:8]) != magic:
                _fail("bad_magic", "invalid canonical object magic")
            self.offset = 8

    def take(self, length: int) -> bytes:
        if length < 0:
            _fail("bounds", "canonical offset overflow")
        end = self.offset + length
        if end < self.offset or end > len(self.data):
            _fail("truncated", "truncated canonical object")
        value = bytes(self.data[self.offset:end])
        self.offset = end
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def identity(self) -> bytes:
        return self.take(32)

    def text(self) -> str:
        length = self.u16()
        raw = self.take(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            _fail("semantics", "state name is not UTF-8")

    def finish(self) -> None:
        if self.offset != len(self.data):
            _fail("reserved", "trailing canonical bytes")


@dataclass(frozen=True, order=True)
class StateKey:
    layer: int
    name: str


@dataclass(frozen=True)
class SemanticModel:
    weights_config: bytes
    adapters: bytes
    tokenizer_template: bytes
    position_semantics: bytes
    qualified_math: bytes


@dataclass(frozen=True)
class InputCut:
    token_root: bytes
    auxiliary_input_root: bytes
    token_count: int


@dataclass(frozen=True)
class AuxiliaryInput:
    type_id: bytes
    value_id: bytes


@dataclass(frozen=True)
class FamilyState:
    key: StateKey
    cache_kind: int
    dtype: int
    codec: int
    codec_version: int
    layout: int
    token_axis_rule: int
    token_axis: int
    elements_per_token: int
    dimensions: tuple[int | None, ...]
    dependencies: tuple[StateKey, ...] = ()


@dataclass(frozen=True)
class Family:
    engine_cache_abi: bytes
    mode: int
    page_size_tokens: int
    topology: bytes
    shard_map: bytes
    states: tuple[FamilyState, ...]


@dataclass(frozen=True)
class Shape:
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class ChunkSpan:
    token_start: int
    token_count: int
    plaintext_offset: int
    plaintext_bytes: int


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: bytes
    object_key: bytes
    object_digest: bytes
    key_epoch: int
    plaintext_bytes: int
    object_bytes: int


@dataclass(frozen=True)
class ManifestKind:
    parent: bytes | None = None
    parent_cut: InputCut | None = None
    depth: int = 0

    @property
    def is_delta(self) -> bool:
        return self.parent is not None


@dataclass(frozen=True)
class StateDeclaration:
    key: StateKey
    full_shape: Shape
    segment_shape: Shape
    strides: tuple[int, ...]
    logical_start: int
    logical_count: int
    absolute_position: int
    window: int
    atomic_group: int


@dataclass(frozen=True)
class RealizedState:
    key: StateKey
    full_shape: Shape
    segment_shape: Shape
    strides: tuple[int, ...]
    logical_start: int
    logical_count: int
    physical_offset_bytes: int
    physical_span_bytes: int
    complete_physical_bytes: int
    absolute_position: int
    window: int
    chunk_spans: tuple[ChunkSpan, ...]


@dataclass(frozen=True)
class AtomicGroup:
    group_id: int
    states: tuple[StateKey, ...]


@dataclass(frozen=True)
class RealizedSchema:
    kind: ManifestKind
    states: tuple[RealizedState, ...]
    atomic_groups: tuple[AtomicGroup, ...]
    segment_restored_bytes: int
    complete_restored_bytes: int


@dataclass(frozen=True)
class StateManifest:
    key: StateKey
    chunks: tuple[ChunkRef, ...]


@dataclass(frozen=True)
class Manifest:
    tenant_namespace: bytes
    key_epoch: int
    semantic_model: SemanticModel
    input_cut: InputCut
    family: Family
    realized_schema: RealizedSchema
    states: tuple[StateManifest, ...]


@dataclass(frozen=True)
class PrefixNode:
    token_count: int
    node_id: bytes
    reusable: bool


@dataclass(frozen=True)
class KeySchedule:
    namespace: bytes
    prefix: bytes
    manifest_auth: bytes
    manifest_encryption: bytes
    chunk_identity: bytes
    object_identity: bytes
    chunk_encryption: bytes


@dataclass(frozen=True)
class ChunkObject:
    chunk_id: bytes
    object_key: bytes
    object_digest: bytes
    plaintext_bytes: int
    data: bytes


def _encode_state_key(value: StateKey) -> bytes:
    raw = value.name.encode("utf-8")
    if len(raw) >= 1 << 16:
        _fail("bounds", "string is too long")
    return _u32(value.layer) + _u16(len(raw)) + raw


def _decode_state_key(reader: Reader) -> StateKey:
    return StateKey(reader.u32(), reader.text())


def _encode_semantic(value: SemanticModel) -> bytes:
    return b"".join(
        _id32(item)
        for item in (
            value.weights_config,
            value.adapters,
            value.tokenizer_template,
            value.position_semantics,
            value.qualified_math,
        )
    )


def _decode_semantic(reader: Reader) -> SemanticModel:
    return SemanticModel(*(reader.identity() for _ in range(5)))


def _encode_input_cut(value: InputCut) -> bytes:
    return _id32(value.token_root) + _id32(value.auxiliary_input_root) + _u64(value.token_count)


def _decode_input_cut(reader: Reader) -> InputCut:
    return InputCut(reader.identity(), reader.identity(), reader.u64())


def _encode_shape(value: Shape) -> bytes:
    rank = len(value.dimensions)
    if rank >= 1 << 8:
        _fail("bounds", "state rank exceeds u8")
    return _u8(rank) + b"".join(_u64(item) for item in value.dimensions)


def _decode_shape(reader: Reader) -> Shape:
    rank = reader.u8()
    if rank == 0 or rank > MAX_RANK:
        _fail("bounds", "state rank must be in 1..=8")
    dimensions = tuple(reader.u64() for _ in range(rank))
    if any(value == 0 for value in dimensions):
        _fail("bounds", "state dimensions must be nonzero")
    return Shape(dimensions)


def _shape_elements(shape: Shape) -> int:
    result = 1
    for value in shape.dimensions:
        result *= value
        if result >= 1 << 64:
            _fail("bounds", "shape element count overflows u64")
    return result


def _enum(value: int, accepted: set[int], what: str) -> int:
    if value not in accepted:
        _fail("unknown_enum", f"unknown {what} {value}")
    return value


def _encode_family_state(value: FamilyState) -> bytes:
    dimensions = b"".join(_u64(0 if item is None else item) for item in value.dimensions)
    dependencies = b"".join(_encode_state_key(item) for item in value.dependencies)
    return b"".join(
        (
            _encode_state_key(value.key),
            _u16(value.cache_kind),
            _u16(value.dtype),
            _u16(value.codec),
            _u16(value.codec_version),
            _u16(value.layout),
            _u16(value.token_axis_rule),
            _u8(value.token_axis),
            _u8(0),
            _u64(value.elements_per_token),
            _u8(len(value.dimensions)),
            dimensions,
            _u16(len(value.dependencies)),
            dependencies,
        )
    )


def _decode_family_state(reader: Reader) -> FamilyState:
    key = _decode_state_key(reader)
    cache_kind = _enum(reader.u16(), {CACHE_ORDINARY_KV}, "cache kind")
    dtype = _enum(reader.u16(), set(DTYPE_WIDTHS), "dtype")
    codec = _enum(reader.u16(), {CODEC_RAW, CODEC_LOSSLESS}, "codec")
    codec_version = reader.u16()
    layout = _enum(reader.u16(), {LAYOUT_CONTIGUOUS, LAYOUT_STRIDED}, "layout")
    token_rule = _enum(reader.u16(), {TOKEN_AXIS_DIRECT, TOKEN_AXIS_GATHER}, "token axis rule")
    token_axis = reader.u8()
    if reader.u8() != 0:
        _fail("reserved", "family state reserved byte is nonzero")
    elements = reader.u64()
    rank = reader.u8()
    if rank == 0 or rank > MAX_RANK:
        _fail("bounds", "family state rank is outside bounds")
    dimensions = tuple(None if (value := reader.u64()) == 0 else value for _ in range(rank))
    dependency_count = reader.u16()
    if dependency_count > MAX_DEPENDENCIES_PER_STATE:
        _fail("bounds", "too many state dependencies")
    dependencies = tuple(_decode_state_key(reader) for _ in range(dependency_count))
    return FamilyState(
        key,
        cache_kind,
        dtype,
        codec,
        codec_version,
        layout,
        token_rule,
        token_axis,
        elements,
        dimensions,
        dependencies,
    )


def _encode_family_body(value: Family) -> bytes:
    return b"".join(
        (
            _id32(value.engine_cache_abi),
            _u16(value.mode),
            _u16(0),
            _u32(value.page_size_tokens),
            _id32(value.topology),
            _id32(value.shard_map),
            _u32(len(value.states)),
            b"".join(_encode_family_state(item) for item in value.states),
        )
    )


def _decode_family_body(reader: Reader) -> Family:
    engine = reader.identity()
    mode = _enum(reader.u16(), {MODE_NATIVE, MODE_PORTABLE}, "representation mode")
    if reader.u16() != 0:
        _fail("reserved", "family reserved field is nonzero")
    page = reader.u32()
    topology = reader.identity()
    shard = reader.identity()
    count = reader.u32()
    if count == 0 or count > MAX_STATES:
        _fail("bounds", "family state count is outside bounds")
    return Family(engine, mode, page, topology, shard, tuple(_decode_family_state(reader) for _ in range(count)))


def encode_family(value: Family) -> bytes:
    return FAMILY_MAGIC + _u16(WIRE_VERSION) + _u16(0) + _encode_family_body(value)


def decode_family(data: bytes, semantic: bool = False) -> Family:
    reader = Reader(data, FAMILY_MAGIC)
    if reader.u16() != WIRE_VERSION:
        _fail("bad_magic", "unsupported family version")
    if reader.u16() != 0:
        _fail("reserved", "family reserved field is nonzero")
    value = _decode_family_body(reader)
    reader.finish()
    if encode_family(value) != bytes(data):
        _fail("reserved", "family encoding is not canonical")
    if semantic:
        validate_family(value)
    return value


def _encode_kind(value: ManifestKind) -> bytes:
    if not value.is_delta:
        return _u8(0) + _u8(value.depth) + _u16(0)
    if value.parent_cut is None:
        _fail("semantics", "delta kind has no parent cut")
    return (
        _u8(1)
        + _u8(value.depth)
        + _u16(0)
        + _id32(value.parent or b"")
        + _encode_input_cut(value.parent_cut)
    )


def _decode_kind(reader: Reader) -> ManifestKind:
    tag = reader.u8()
    depth = reader.u8()
    if reader.u16() != 0:
        _fail("reserved", "manifest-kind reserved field is nonzero")
    if tag == 0:
        if depth != 0:
            _fail("reserved", "full manifest carries a delta depth")
        return ManifestKind()
    if tag == 1:
        return ManifestKind(reader.identity(), _decode_input_cut(reader), depth)
    _fail("unknown_enum", f"unknown manifest kind {tag}")


def _encode_declaration_body(value: StateDeclaration) -> bytes:
    return b"".join(
        (
            _encode_state_key(value.key),
            _encode_shape(value.full_shape),
            _encode_shape(value.segment_shape),
            _u8(len(value.strides)),
            b"".join(_u64(item) for item in value.strides),
            _u64(value.logical_start),
            _u64(value.logical_count),
            _u64(value.absolute_position),
            _u64(value.window),
            _u32(value.atomic_group),
        )
    )


def _decode_declaration_body(reader: Reader) -> StateDeclaration:
    key = _decode_state_key(reader)
    full = _decode_shape(reader)
    segment = _decode_shape(reader)
    count = reader.u8()
    if count > MAX_RANK:
        _fail("bounds", "too many state strides")
    strides = tuple(reader.u64() for _ in range(count))
    return StateDeclaration(
        key,
        full,
        segment,
        strides,
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.u32(),
    )


def encode_state_declaration(value: StateDeclaration) -> bytes:
    return STATE_SCHEMA_MAGIC + _u16(WIRE_VERSION) + _u16(0) + _encode_declaration_body(value)


def decode_state_declaration(data: bytes) -> StateDeclaration:
    reader = Reader(data, STATE_SCHEMA_MAGIC)
    if reader.u16() != WIRE_VERSION:
        _fail("bad_magic", "unsupported state-schema version")
    if reader.u16() != 0:
        _fail("reserved", "state-schema reserved field is nonzero")
    value = _decode_declaration_body(reader)
    reader.finish()
    if encode_state_declaration(value) != bytes(data):
        _fail("reserved", "state-schema encoding is not canonical")
    return value


def _encode_realized_state(value: RealizedState) -> bytes:
    return b"".join(
        (
            _encode_state_key(value.key),
            _encode_shape(value.full_shape),
            _encode_shape(value.segment_shape),
            _u8(len(value.strides)),
            b"".join(_u64(item) for item in value.strides),
            _u64(value.logical_start),
            _u64(value.logical_count),
            _u64(value.physical_offset_bytes),
            _u64(value.physical_span_bytes),
            _u64(value.complete_physical_bytes),
            _u64(value.absolute_position),
            _u64(value.window),
            _u32(len(value.chunk_spans)),
            b"".join(
                _u64(span.token_start)
                + _u64(span.token_count)
                + _u64(span.plaintext_offset)
                + _u32(span.plaintext_bytes)
                for span in value.chunk_spans
            ),
        )
    )


def _decode_realized_state(reader: Reader) -> RealizedState:
    key = _decode_state_key(reader)
    full = _decode_shape(reader)
    segment = _decode_shape(reader)
    stride_count = reader.u8()
    if stride_count > MAX_RANK:
        _fail("bounds", "too many state strides")
    strides = tuple(reader.u64() for _ in range(stride_count))
    logical_start = reader.u64()
    logical_count = reader.u64()
    physical_offset = reader.u64()
    physical_span = reader.u64()
    complete_physical = reader.u64()
    absolute = reader.u64()
    window = reader.u64()
    chunk_count = reader.u32()
    if chunk_count == 0 or chunk_count > MAX_CHUNKS_PER_STATE:
        _fail("bounds", "state chunk-span count is outside bounds")
    spans = tuple(
        ChunkSpan(reader.u64(), reader.u64(), reader.u64(), reader.u32())
        for _ in range(chunk_count)
    )
    return RealizedState(
        key,
        full,
        segment,
        strides,
        logical_start,
        logical_count,
        physical_offset,
        physical_span,
        complete_physical,
        absolute,
        window,
        spans,
    )


def _encode_group(value: AtomicGroup) -> bytes:
    return _u32(value.group_id) + _u32(len(value.states)) + b"".join(
        _encode_state_key(item) for item in value.states
    )


def _decode_group(reader: Reader) -> AtomicGroup:
    group_id = reader.u32()
    count = reader.u32()
    if count == 0 or count > MAX_STATES:
        _fail("bounds", "atomic-group state count is outside bounds")
    return AtomicGroup(group_id, tuple(_decode_state_key(reader) for _ in range(count)))


def _encode_schema_body(value: RealizedSchema) -> bytes:
    return b"".join(
        (
            _encode_kind(value.kind),
            _u32(len(value.states)),
            b"".join(_encode_realized_state(item) for item in value.states),
            _u32(len(value.atomic_groups)),
            b"".join(_encode_group(item) for item in value.atomic_groups),
            _u64(value.segment_restored_bytes),
            _u64(value.complete_restored_bytes),
        )
    )


def _decode_schema_body(reader: Reader) -> RealizedSchema:
    kind = _decode_kind(reader)
    state_count = reader.u32()
    if state_count == 0 or state_count > MAX_STATES:
        _fail("bounds", "realized state count is outside bounds")
    states = tuple(_decode_realized_state(reader) for _ in range(state_count))
    group_count = reader.u32()
    if group_count == 0 or group_count > MAX_ATOMIC_GROUPS:
        _fail("bounds", "atomic-group count is outside bounds")
    groups = tuple(_decode_group(reader) for _ in range(group_count))
    return RealizedSchema(kind, states, groups, reader.u64(), reader.u64())


def encode_schema(value: RealizedSchema) -> bytes:
    return SCHEMA_MAGIC + _u16(WIRE_VERSION) + _u16(0) + _encode_schema_body(value)


def decode_schema(data: bytes) -> RealizedSchema:
    reader = Reader(data, SCHEMA_MAGIC)
    if reader.u16() != WIRE_VERSION:
        _fail("bad_magic", "unsupported realized-schema version")
    if reader.u16() != 0:
        _fail("reserved", "realized-schema reserved field is nonzero")
    value = _decode_schema_body(reader)
    reader.finish()
    if encode_schema(value) != bytes(data):
        _fail("reserved", "realized-schema encoding is not canonical")
    return value


def encode_manifest(value: Manifest) -> bytes:
    payload = b"".join(
        _encode_state_key(state.key)
        + _u32(len(state.chunks))
        + b"".join(
            _id32(chunk.chunk_id)
            + _id32(chunk.object_key)
            + _id32(chunk.object_digest)
            + _u64(chunk.key_epoch)
            + _u32(chunk.plaintext_bytes)
            + _u32(chunk.object_bytes)
            for chunk in state.chunks
        )
        for state in value.states
    )
    return b"".join(
        (
            MANIFEST_MAGIC,
            _u16(WIRE_VERSION),
            _u16(0),
            _id32(value.tenant_namespace),
            _u64(value.key_epoch),
            _encode_semantic(value.semantic_model),
            _encode_input_cut(value.input_cut),
            _encode_family_body(value.family),
            _encode_schema_body(value.realized_schema),
            _u32(len(value.states)),
            payload,
        )
    )


def decode_manifest(data: bytes, semantic: bool = False) -> Manifest:
    data = bytes(data)
    if len(data) > MAX_MANIFEST_BYTES:
        _fail("bounds", "manifest exceeds production bound")
    reader = Reader(data, MANIFEST_MAGIC)
    if reader.u16() != WIRE_VERSION:
        _fail("bad_magic", "unsupported manifest version")
    if reader.u16() != 0:
        _fail("reserved", "manifest reserved field is nonzero")
    tenant = reader.identity()
    epoch = reader.u64()
    model = _decode_semantic(reader)
    input_cut = _decode_input_cut(reader)
    family = _decode_family_body(reader)
    schema = _decode_schema_body(reader)
    state_count = reader.u32()
    if state_count == 0 or state_count > MAX_STATES:
        _fail("bounds", "manifest state count is outside bounds")
    states: list[StateManifest] = []
    for _ in range(state_count):
        key = _decode_state_key(reader)
        chunk_count = reader.u32()
        if chunk_count == 0 or chunk_count > MAX_CHUNKS_PER_STATE:
            _fail("bounds", "state chunk count is outside bounds")
        chunks = tuple(
            ChunkRef(
                reader.identity(),
                reader.identity(),
                reader.identity(),
                reader.u64(),
                reader.u32(),
                reader.u32(),
            )
            for _ in range(chunk_count)
        )
        states.append(StateManifest(key, chunks))
    reader.finish()
    value = Manifest(tenant, epoch, model, input_cut, family, schema, tuple(states))
    if encode_manifest(value) != data:
        _fail("reserved", "manifest encoding is not canonical")
    if semantic:
        validate_manifest(value)
    return value


def _sha256(*parts: bytes) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def _hmac(key: bytes, *parts: bytes) -> bytes:
    value = hmac.new(key, digestmod=hashlib.sha256)
    for part in parts:
        value.update(part)
    return value.digest()


def semantic_model_id(value: SemanticModel) -> bytes:
    return _sha256(b"kvpack/v1/semantic-model\0", _encode_semantic(value))


def representation_family_id(value: Family) -> bytes:
    return _sha256(b"kvpack/v1/representation-family\0", encode_family(value))


def realized_schema_id(value: RealizedSchema) -> bytes:
    return _sha256(b"kvpack/v1/realized-cut-schema\0", encode_schema(value))


def manifest_id(value: Manifest | bytes) -> bytes:
    canonical = encode_manifest(value) if isinstance(value, Manifest) else bytes(value)
    return _sha256(b"kvpack/v1/manifest\0", canonical)


def namespace_id(key: bytes, operator_tenant_id: bytes) -> bytes:
    if len(operator_tenant_id) >= 1 << 32:
        _fail("bounds", "tenant ID is too long")
    return _hmac(key, b"kvpack/v1/namespace\0", _u32(len(operator_tenant_id)), operator_tenant_id)


def auxiliary_input_root(key: bytes, tenant: bytes, inputs: Sequence[AuxiliaryInput]) -> bytes:
    framed = bytearray(_u32(len(inputs)))
    for ordinal, item in enumerate(inputs):
        type_id = _id32(item.type_id, "auxiliary type identity")
        value_id = _id32(item.value_id, "auxiliary value identity")
        if type_id == bytes(32) or value_id == bytes(32):
            _fail("semantics", "auxiliary identity contains a zero component")
        framed += _u64(ordinal) + _u32(32) + type_id + _u32(32) + value_id
    return _hmac(key, b"kvpack/v1/auxiliary-input-root\0", _id32(tenant), bytes(framed))


def _prefix_context(
    key: bytes,
    tenant: bytes,
    model: SemanticModel,
    family: Family,
    auxiliary_root: bytes,
) -> bytes:
    return _hmac(
        key,
        b"kvpack/v1/prefix-context\0",
        _id32(tenant),
        semantic_model_id(model),
        representation_family_id(family),
        _id32(auxiliary_root),
    )


def chain_prefix_nodes(
    key: bytes,
    tenant: bytes,
    model: SemanticModel,
    family: Family,
    auxiliary_root: bytes,
    tokens: Sequence[int],
) -> tuple[PrefixNode, ...]:
    context = _prefix_context(key, tenant, model, family, auxiliary_root)
    parent = _hmac(key, b"kvpack/v1/prefix-root\0", context)
    result: list[PrefixNode] = []
    for index, start in enumerate(range(0, len(tokens), PREFIX_BLOCK_TOKENS)):
        block = tokens[start : start + PREFIX_BLOCK_TOKENS]
        token_bytes = b"".join(_u32(token) for token in block)
        framed = b"".join(
            (
                context,
                parent,
                _u64(index),
                _u64(start),
                _u32(len(block)),
                _u32(len(token_bytes)),
                token_bytes,
            )
        )
        parent = _hmac(key, b"kvpack/v1/prefix-node\0", framed)
        result.append(PrefixNode(start + len(block), parent, len(block) == PREFIX_BLOCK_TOKENS))
    return tuple(result)


def derive_input_cut(
    key: bytes,
    tenant: bytes,
    model: SemanticModel,
    family: Family,
    tokens: Sequence[int],
    inputs: Sequence[AuxiliaryInput],
) -> tuple[InputCut, tuple[PrefixNode, ...]]:
    auxiliary = auxiliary_input_root(key, tenant, inputs)
    nodes = chain_prefix_nodes(key, tenant, model, family, auxiliary, tokens)
    if nodes:
        root = nodes[-1].node_id
    else:
        context = _prefix_context(key, tenant, model, family, auxiliary)
        root = _hmac(key, b"kvpack/v1/prefix-root\0", context)
    return InputCut(root, auxiliary, len(tokens)), nodes


def chunk_id(
    key: bytes,
    tenant: bytes,
    family: Family,
    state_key: StateKey,
    span: ChunkSpan,
    plaintext: bytes,
) -> bytes:
    plaintext = bytes(plaintext)
    if span.token_count == 0 or span.plaintext_bytes != len(plaintext):
        _fail("semantics", "chunk plaintext does not match its declared span")
    name = state_key.name.encode("utf-8")
    return _hmac(
        key,
        b"kvpack/v1/chunk-content\0",
        _id32(tenant),
        representation_family_id(family),
        _u32(state_key.layer),
        _u16(len(name)),
        name,
        _u64(span.token_start),
        _u64(span.token_count),
        _u64(span.plaintext_offset),
        _u32(len(plaintext)),
        plaintext,
    )


def _hkdf(root: bytes, salt: bytes, label: bytes) -> bytes:
    prk = _hmac(salt, root)
    return _hmac(prk, label, b"\x01")


def derive_key_schedule(root: bytes, tenant: bytes, epoch: int) -> KeySchedule:
    root = _id32(root, "root key")
    tenant = _id32(tenant, "tenant namespace")
    if epoch == 0:
        _fail("semantics", "key epoch must be nonzero")
    stable = lambda label: _hkdf(root, tenant, label)
    epoch_salt = tenant + _u64(epoch)
    rotating = lambda label: _hkdf(root, epoch_salt, label)
    return KeySchedule(
        stable(b"kvpack/v1/stable/namespace"),
        stable(b"kvpack/v1/stable/prefix"),
        rotating(b"kvpack/v1/epoch/manifest-auth"),
        rotating(b"kvpack/v1/epoch/manifest-encryption"),
        stable(b"kvpack/v1/stable/chunk-identity"),
        rotating(b"kvpack/v1/epoch/object-identity"),
        rotating(b"kvpack/v1/epoch/chunk-encryption"),
    )


def encode_codec_frame(codec: int, plaintext: bytes) -> bytes:
    plaintext = bytes(plaintext)
    if not 1 <= len(plaintext) <= MAX_CHUNK_PLAINTEXT:
        _fail("bounds", "chunk plaintext must be in 1..=4 MiB")
    if codec == CODEC_RAW:
        return RAW_FRAME_MAGIC + _u16(WIRE_VERSION) + _u16(0) + _u32(len(plaintext)) + plaintext
    if codec != CODEC_LOSSLESS:
        _fail("unknown_enum", f"unknown codec {codec}")
    output = bytearray(LOSSLESS_FRAME_MAGIC + _u16(WIRE_VERSION) + _u16(0) + _u32(len(plaintext)))

    def run(start: int) -> int:
        length = 1
        while start + length < len(plaintext) and length < 128 and plaintext[start + length] == plaintext[start]:
            length += 1
        return length

    index = 0
    while index < len(plaintext):
        repeated = run(index)
        if repeated >= 3:
            output += bytes((0x80 | (repeated - 1), plaintext[index]))
            index += repeated
            continue
        literal_start = index
        index += repeated
        while index < len(plaintext) and index - literal_start < 128:
            repeated = run(index)
            if repeated >= 3:
                break
            if index - literal_start + repeated > 128:
                index = literal_start + 128
                break
            index += repeated
        length = index - literal_start
        output.append(length - 1)
        output += plaintext[literal_start:index]
    if len(output) > MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD:
        _fail("bounds", "lossless frame exceeds encoded bound")
    return bytes(output)


def decode_codec_frame(codec: int, encoded: bytes, expected_plaintext_bytes: int) -> bytes:
    encoded = bytes(encoded)
    if len(encoded) < CODEC_FRAME_HEADER_BYTES:
        _fail("truncated", "truncated codec frame")
    expected_magic = RAW_FRAME_MAGIC if codec == CODEC_RAW else LOSSLESS_FRAME_MAGIC if codec == CODEC_LOSSLESS else None
    if expected_magic is None:
        _fail("unknown_enum", f"unknown codec {codec}")
    if encoded[:8] != expected_magic:
        _fail("bad_magic", "codec frame magic does not match codec")
    version, reserved, decoded_bytes = struct.unpack_from("<HHI", encoded, 8)
    if version != WIRE_VERSION:
        _fail("bad_magic", "unsupported codec frame version")
    if reserved != 0:
        _fail("reserved", "codec frame reserved field is nonzero")
    if decoded_bytes != expected_plaintext_bytes or not 1 <= expected_plaintext_bytes <= MAX_CHUNK_PLAINTEXT:
        _fail("bounds", "codec frame decoded length mismatch")
    payload = encoded[16:]
    if codec == CODEC_RAW:
        if len(payload) != expected_plaintext_bytes:
            _fail("bounds", "raw frame length mismatch")
        decoded = payload
    else:
        result = bytearray()
        index = 0
        while index < len(payload):
            control = payload[index]
            index += 1
            length = (control & 0x7F) + 1
            if control & 0x80:
                if index >= len(payload):
                    _fail("truncated", "truncated lossless repeat packet")
                value = payload[index]
                index += 1
                if len(result) + length > expected_plaintext_bytes:
                    _fail("bounds", "lossless frame expands past bound")
                result += bytes((value,)) * length
            else:
                end = index + length
                if end > len(payload):
                    _fail("truncated", "truncated lossless literal packet")
                if len(result) + length > expected_plaintext_bytes:
                    _fail("bounds", "lossless frame expands past bound")
                result += payload[index:end]
                index = end
        if len(result) != expected_plaintext_bytes:
            _fail("bounds", "lossless decoded length mismatch")
        decoded = bytes(result)
    if encode_codec_frame(codec, decoded) != encoded:
        _fail("reserved", "codec frame is not canonical")
    return decoded


def _checked_u64(value: int, message: str) -> int:
    if value < 0 or value >= 1 << 64:
        _fail("bounds", message)
    return value


def _validate_state_key(key: StateKey) -> None:
    raw = key.name.encode("utf-8")
    if not 1 <= len(raw) <= MAX_STATE_NAME_BYTES or b"\0" in raw:
        _fail("semantics", "state key name is outside canonical bounds")


def validate_family(family: Family) -> None:
    if any(
        _id32(identity) == bytes(32)
        for identity in (family.engine_cache_abi, family.topology, family.shard_map)
    ):
        _fail("semantics", "representation family contains a zero identity")
    if family.page_size_tokens == 0:
        _fail("semantics", "representation family page size must be nonzero")
    if not 1 <= len(family.states) <= MAX_STATES:
        _fail("bounds", "family state count is outside bounds")
    previous: StateKey | None = None
    keys = {state.key for state in family.states}
    graph: dict[StateKey, tuple[StateKey, ...]] = {}
    for state in family.states:
        _validate_state_key(state.key)
        if previous is not None and previous >= state.key:
            _fail("semantics", "family states are not unique canonical order")
        previous = state.key
        if state.codec_version != 1:
            _fail("codec", "unsupported family codec version")
        rank = len(state.dimensions)
        if rank == 0 or rank > MAX_RANK or state.token_axis >= rank:
            _fail("semantics", "family token axis is outside its rank")
        token_dimensions = tuple(index for index, value in enumerate(state.dimensions) if value is None)
        if token_dimensions != (state.token_axis,):
            _fail("semantics", "family must have exactly one declared token dimension")
        elements = 1
        for dimension in state.dimensions:
            if dimension is not None:
                elements = _checked_u64(elements * dimension, "family elements-per-token overflows u64")
        if elements != state.elements_per_token or elements == 0:
            _fail("semantics", "family elements-per-token is not canonical")
        if state.dtype not in DTYPE_WIDTHS:
            _fail("semantics", "family dtype has no fixed width")
        if state.layout == LAYOUT_CONTIGUOUS and state.token_axis_rule == TOKEN_AXIS_GATHER:
            _fail("semantics", "contiguous family cannot require strided token gathering")
        if family.mode == MODE_PORTABLE and (
            state.layout != LAYOUT_CONTIGUOUS or state.token_axis_rule != TOKEN_AXIS_DIRECT
        ):
            _fail("semantics", "portable family requires contiguous direct token streams")
        graph[state.key] = state.dependencies
        dependency_previous: StateKey | None = None
        for dependency in state.dependencies:
            if dependency == state.key:
                _fail("graph", "state depends on itself")
            if dependency not in keys:
                _fail("graph", "state dependency is missing")
            if dependency_previous is not None and dependency_previous >= dependency:
                _fail("graph", "state dependencies are not unique canonical order")
            dependency_previous = dependency

    visiting: set[StateKey] = set()
    done: set[StateKey] = set()

    def visit(key: StateKey) -> None:
        if key in done:
            return
        if key in visiting:
            _fail("graph", "state dependency cycle")
        visiting.add(key)
        for dependency in graph[key]:
            visit(dependency)
        visiting.remove(key)
        done.add(key)

    for key in sorted(graph):
        visit(key)


def _validate_shape(shape: Shape, family: FamilyState, expected_tokens: int) -> None:
    if len(shape.dimensions) != len(family.dimensions):
        _fail("semantics", "realized shape rank does not match family rank")
    for index, (actual, static) in enumerate(zip(shape.dimensions, family.dimensions)):
        expected = expected_tokens if static is None else static
        if actual != expected or ((index == family.token_axis) != (static is None)):
            _fail("semantics", "realized shape does not satisfy static family dimensions")


def _physical_footprint(shape: Shape, strides: Sequence[int], width: int) -> int:
    last = 0
    for dimension, stride in zip(shape.dimensions, strides):
        last = _checked_u64(
            last + (dimension - 1) * stride,
            "state physical footprint overflows u64",
        )
    return _checked_u64((last + 1) * width, "state physical footprint overflows u64")


def _validate_atomic_groups(manifest: Manifest) -> None:
    expected = {state.key for state in manifest.family.states}
    observed: set[StateKey] = set()
    previous_id = 0
    for group in manifest.realized_schema.atomic_groups:
        if group.group_id == 0 or group.group_id <= previous_id:
            _fail("semantics", "atomic groups are not unique canonical order")
        previous_id = group.group_id
        previous_key: StateKey | None = None
        for state in group.states:
            if state not in expected:
                _fail("graph", "atomic group names an unknown state")
            if previous_key is not None and previous_key >= state:
                _fail("semantics", "atomic-group states are not unique canonical order")
            if state in observed:
                _fail("graph", "state appears in multiple atomic groups")
            observed.add(state)
            previous_key = state
    if observed != expected:
        _fail("graph", "atomic groups do not cover the complete family inventory")


def validate_manifest(manifest: Manifest, maximum_restored_bytes: int = 4 * 1024**4) -> None:
    if _id32(manifest.tenant_namespace) == bytes(32):
        _fail("semantics", "tenant namespace is zero")
    if manifest.key_epoch == 0:
        _fail("semantics", "key epoch must be nonzero")
    if any(
        _id32(identity) == bytes(32)
        for identity in (
            manifest.semantic_model.weights_config,
            manifest.semantic_model.adapters,
            manifest.semantic_model.tokenizer_template,
            manifest.semantic_model.position_semantics,
            manifest.semantic_model.qualified_math,
        )
    ):
        _fail("semantics", "semantic model identity contains a zero component")
    if (
        _id32(manifest.input_cut.token_root) == bytes(32)
        or _id32(manifest.input_cut.auxiliary_input_root) == bytes(32)
        or manifest.input_cut.token_count == 0
    ):
        _fail("semantics", "input cut identity is invalid")
    validate_family(manifest.family)

    kind = manifest.realized_schema.kind
    if not kind.is_delta:
        range_start = 0
        range_count = manifest.input_cut.token_count
    else:
        assert kind.parent is not None and kind.parent_cut is not None
        if (
            _id32(kind.parent) == bytes(32)
            or _id32(kind.parent_cut.token_root) == bytes(32)
            or _id32(kind.parent_cut.auxiliary_input_root) == bytes(32)
        ):
            _fail("graph", "delta parent identity is invalid")
        if kind.depth == 0 or kind.depth > MAX_DELTA_DEPTH:
            _fail("graph", "delta depth is outside 1..=7")
        if (
            kind.parent_cut.auxiliary_input_root != manifest.input_cut.auxiliary_input_root
            or kind.parent_cut.token_count == 0
            or kind.parent_cut.token_count >= manifest.input_cut.token_count
        ):
            _fail("graph", "delta parent cut is not a compatible strict prefix")
        range_start = kind.parent_cut.token_count
        range_count = manifest.input_cut.token_count - range_start

    schema = manifest.realized_schema
    if (
        len(schema.states) != len(manifest.family.states)
        or len(manifest.states) != len(manifest.family.states)
        or len(schema.states) > MAX_STATES
    ):
        _fail("bounds", "manifest, family, and schema state counts differ or exceed bounds")
    _validate_atomic_groups(manifest)

    segment_total = 0
    complete_total = 0
    for family_state, realized, payload in zip(manifest.family.states, schema.states, manifest.states):
        if family_state.key != realized.key or realized.key != payload.key:
            _fail("semantics", "family, realized schema, and payload state order differ")
        _validate_shape(realized.full_shape, family_state, manifest.input_cut.token_count)
        _validate_shape(realized.segment_shape, family_state, range_count)
        if (
            realized.logical_start != range_start
            or realized.logical_count != range_count
            or realized.absolute_position != manifest.input_cut.token_count
            or realized.window != 0
        ):
            _fail("semantics", "realized state range does not match the exact manifest cut")
        if len(realized.strides) != len(realized.full_shape.dimensions) or any(
            stride == 0 for stride in realized.strides
        ):
            _fail("semantics", "realized state strides do not match its rank")
        if family_state.layout == LAYOUT_CONTIGUOUS:
            expected_stride = 1
            for dimension, stride in reversed(tuple(zip(realized.full_shape.dimensions, realized.strides))):
                if stride != expected_stride:
                    _fail("semantics", "contiguous state has noncanonical strides")
                expected_stride = _checked_u64(
                    expected_stride * dimension,
                    "contiguous stride calculation overflows u64",
                )
        width = DTYPE_WIDTHS.get(family_state.dtype)
        if width is None:
            _fail("semantics", "family dtype has no fixed width")
        segment_bytes = _checked_u64(
            _shape_elements(realized.segment_shape) * width,
            "segment byte count overflows u64",
        )
        complete_bytes = _checked_u64(
            _shape_elements(realized.full_shape) * width,
            "complete byte count overflows u64",
        )
        physical_offset = _checked_u64(
            range_start * realized.strides[family_state.token_axis] * width,
            "physical token offset overflows u64",
        )
        physical_span = _physical_footprint(realized.segment_shape, realized.strides, width)
        complete_physical = _physical_footprint(realized.full_shape, realized.strides, width)
        if (
            realized.physical_offset_bytes != physical_offset
            or realized.physical_span_bytes != physical_span
            or realized.complete_physical_bytes != complete_physical
        ):
            _fail("semantics", "realized physical span is not canonical")
        if (
            not realized.chunk_spans
            or len(realized.chunk_spans) > MAX_CHUNKS_PER_STATE
            or len(payload.chunks) != len(realized.chunk_spans)
        ):
            _fail("bounds", "state chunk count is outside bounds")
        bytes_per_token = _checked_u64(
            family_state.elements_per_token * width,
            "bytes per token overflows u64",
        )
        if bytes_per_token == 0 or bytes_per_token > MAX_CHUNK_PLAINTEXT:
            _fail("bounds", "one state token does not fit in a bounded chunk")
        expected_token = range_start
        expected_offset = _checked_u64(
            range_start * bytes_per_token,
            "state plaintext base offset overflows u64",
        )
        for span, chunk in zip(realized.chunk_spans, payload.chunks):
            if (
                span.token_start != expected_token
                or span.token_count == 0
                or span.plaintext_offset != expected_offset
            ):
                _fail("semantics", "chunk spans are not a contiguous token/byte partition")
            expected_bytes = _checked_u64(
                span.token_count * bytes_per_token,
                "chunk span byte count overflows u64",
            )
            if (
                expected_bytes != span.plaintext_bytes
                or expected_bytes == 0
                or expected_bytes > MAX_CHUNK_PLAINTEXT
                or chunk.plaintext_bytes != span.plaintext_bytes
            ):
                _fail("semantics", "chunk span has a noncanonical plaintext size")
            if (
                _id32(chunk.chunk_id) == bytes(32)
                or _id32(chunk.object_key) == bytes(32)
                or _id32(chunk.object_digest) == bytes(32)
                or chunk.key_epoch == 0
            ):
                _fail("semantics", "chunk reference contains a zero identity")
            if (
                chunk.object_bytes % ALIGNMENT != 0
                or chunk.object_bytes <= CHUNK_HEADER_BYTES
                or chunk.object_bytes > MAX_CHUNK_OBJECT_BYTES
            ):
                _fail("semantics", "chunk object size is not aligned")
            expected_token = _checked_u64(
                expected_token + span.token_count,
                "chunk token range overflows u64",
            )
            expected_offset = _checked_u64(
                expected_offset + expected_bytes,
                "chunk byte range overflows u64",
            )
        range_end = _checked_u64(range_start + range_count, "state token range overflows u64")
        end_offset = _checked_u64(
            range_end * bytes_per_token,
            "state plaintext end offset overflows u64",
        )
        if expected_token != range_end or expected_offset != end_offset:
            _fail("semantics", "chunk spans do not cover the complete realized state segment")
        segment_total = _checked_u64(
            segment_total + segment_bytes,
            "segment restored byte total overflows u64",
        )
        complete_total = _checked_u64(
            complete_total + complete_bytes,
            "complete restored byte total overflows u64",
        )
    if (
        schema.segment_restored_bytes != segment_total
        or schema.complete_restored_bytes != complete_total
        or schema.complete_restored_bytes > maximum_restored_bytes
    ):
        _fail("bounds", "manifest restored byte totals are invalid")


def _rotl32(value: int, count: int) -> int:
    return ((value << count) | (value >> (32 - count))) & 0xFFFFFFFF


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] = _rotl32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotl32(state[b] ^ state[c], 7)


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    key = _id32(key, "ChaCha20 key")
    if len(nonce) != 12:
        _fail("bounds", "ChaCha20 nonce must be 12 bytes")
    initial = list(struct.unpack("<4I", b"expand 32-byte k"))
    initial += list(struct.unpack("<8I", key))
    initial += [counter & 0xFFFFFFFF]
    initial += list(struct.unpack("<3I", nonce))
    state = initial.copy()
    for _ in range(10):
        _quarter_round(state, 0, 4, 8, 12)
        _quarter_round(state, 1, 5, 9, 13)
        _quarter_round(state, 2, 6, 10, 14)
        _quarter_round(state, 3, 7, 11, 15)
        _quarter_round(state, 0, 5, 10, 15)
        _quarter_round(state, 1, 6, 11, 12)
        _quarter_round(state, 2, 7, 8, 13)
        _quarter_round(state, 3, 4, 9, 14)
    return struct.pack("<16I", *((value + original) & 0xFFFFFFFF for value, original in zip(state, initial)))


def _chacha20_xor(key: bytes, nonce: bytes, data: bytes, counter: int = 1) -> bytes:
    output = bytearray(len(data))
    for offset in range(0, len(data), 64):
        if counter >= 1 << 32:
            _fail("bounds", "ChaCha20 counter overflow")
        block = _chacha20_block(key, counter, nonce)
        part = data[offset : offset + 64]
        output[offset : offset + len(part)] = bytes(a ^ b for a, b in zip(part, block))
        counter += 1
    return bytes(output)


def _poly1305(key: bytes, message: bytes) -> bytes:
    key = _id32(key, "Poly1305 key")
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:], "little")
    accumulator = 0
    prime = (1 << 130) - 5
    for offset in range(0, len(message), 16):
        block = message[offset : offset + 16]
        accumulator = (accumulator + int.from_bytes(block + b"\x01", "little")) % prime
        accumulator = (accumulator * r) % prime
    return ((accumulator + s) % (1 << 128)).to_bytes(16, "little")


def _pad16(value: bytes) -> bytes:
    return bytes((-len(value)) % 16)


def _aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    one_time = _chacha20_block(key, 0, nonce)[:32]
    ciphertext = _chacha20_xor(key, nonce, plaintext)
    mac_data = aad + _pad16(aad) + ciphertext + _pad16(ciphertext) + _u64(len(aad)) + _u64(len(ciphertext))
    return ciphertext + _poly1305(one_time, mac_data)


def _aead_decrypt(key: bytes, nonce: bytes, stored: bytes, aad: bytes, message: str) -> bytes:
    if len(stored) < 16:
        _fail("truncated", "truncated encrypted payload")
    ciphertext, tag = stored[:-16], stored[-16:]
    one_time = _chacha20_block(key, 0, nonce)[:32]
    mac_data = aad + _pad16(aad) + ciphertext + _pad16(ciphertext) + _u64(len(aad)) + _u64(len(ciphertext))
    if not hmac.compare_digest(tag, _poly1305(one_time, mac_data)):
        _fail("authentication", message)
    return _chacha20_xor(key, nonce, ciphertext)


def _family_state(family: Family, key: StateKey) -> FamilyState:
    for state in family.states:
        if state.key == key:
            return state
    _fail("semantics", "chunk state is absent from representation family")


def _object_key(
    keys: KeySchedule,
    tenant: bytes,
    family_id: bytes,
    content_id: bytes,
    epoch: int,
    codec: int,
    codec_version: int,
    encrypted: bool,
    salt: bytes,
    nonce: bytes,
    stats_digest: bytes | None = None,
) -> bytes:
    parts = [
        b"kvpack/v1/chunk-object\0",
        tenant,
        family_id,
        content_id,
        _u64(epoch),
        _u16(codec),
        _u16(codec_version),
        bytes((int(encrypted),)),
        salt,
        nonce,
    ]
    if stats_digest is not None:
        parts += [b"kvpack/v1/chunk-object-stats\0", stats_digest]
    return _hmac(keys.object_identity, *parts)


def _chunk_encryption_key(keys: KeySchedule, content: bytes, object_key: bytes, salt: bytes) -> bytes:
    return _hkdf(
        keys.chunk_encryption,
        salt,
        b"kvpack/v1/chunk-aead\0" + content + object_key,
    )


def _f16_to_f32(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _f32_to_f16(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        # Rust f32_to_f16 saturates to infinity; struct 'e' raises instead.
        return 0x7C00 if value > 0 else 0xFC00


def _top_sinks(key_l2_norms: Sequence[int], count: int) -> list[tuple[int, int]]:
    order = sorted(range(len(key_l2_norms)), key=lambda i: (-_f16_to_f32(key_l2_norms[i]), i))
    return [(index, key_l2_norms[index]) for index in order[:count]]


def encode_sidecar(
    channel_ranges: Sequence[tuple[int, int]],
    key_l2_norms: Sequence[int],
    sink_scores: Sequence[tuple[int, int]],
) -> bytes:
    """Canonical KVSSC1 encoding of one M7 statistics sidecar."""
    out = bytearray(STATS_SIDECAR_MAGIC)
    out += _u16(WIRE_VERSION)
    out += _u16(0)
    out += _u32(len(channel_ranges))
    for minimum, maximum in channel_ranges:
        out += _u16(minimum)
        out += _u16(maximum)
    out += _u32(len(key_l2_norms))
    for norm in key_l2_norms:
        out += _u16(norm)
    out += _u16(len(sink_scores))
    for token_index, score in sink_scores:
        out += _u32(token_index)
        out += _u16(score)
        out += _u16(0)
    return bytes(out)


def decode_sidecar(data: bytes) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]]:
    """Fail-closed canonical KVSSC1 decode mirroring StatsSidecar::decode_canonical."""
    data = bytes(data)
    if len(data) > MAX_STATS_SIDECAR_BYTES:
        _fail("bounds", "stats sidecar exceeds the chunk header capacity")
    reader = Reader(data, STATS_SIDECAR_MAGIC)
    if reader.u16() != WIRE_VERSION:
        _fail("bad_magic", "unsupported stats sidecar version")
    if reader.u16() != 0:
        _fail("reserved", "stats sidecar reserved field is nonzero")
    channel_count = reader.u32()
    if channel_count == 0 or channel_count > MAX_SIDECAR_CHANNELS:
        _fail("bounds", "stats sidecar channel count is outside bounds")
    channel_ranges = []
    for _ in range(channel_count):
        minimum, maximum = reader.u16(), reader.u16()
        if (
            not math.isfinite(_f16_to_f32(minimum))
            or not math.isfinite(_f16_to_f32(maximum))
            or _f16_to_f32(minimum) > _f16_to_f32(maximum)
        ):
            _fail("semantics", "stats sidecar channel range is not finite and ordered")
        channel_ranges.append((minimum, maximum))
    token_count = reader.u32()
    if token_count == 0 or token_count > MAX_SIDECAR_TOKENS:
        _fail("bounds", "stats sidecar token count is outside bounds")
    key_l2_norms = []
    for _ in range(token_count):
        bits = reader.u16()
        if not math.isfinite(_f16_to_f32(bits)) or _f16_to_f32(bits) < 0.0:
            _fail("semantics", "stats sidecar key norm is not a finite non-negative value")
        key_l2_norms.append(bits)
    sink_count = reader.u16()
    if sink_count == 0 or sink_count > MAX_SINK_SCORES:
        _fail("bounds", "stats sidecar sink count is outside bounds")
    sink_scores = []
    for _ in range(sink_count):
        token_index, score = reader.u32(), reader.u16()
        if reader.u16() != 0:
            _fail("reserved", "stats sidecar sink reserved field is nonzero")
        sink_scores.append((token_index, score))
    reader.finish()
    if sink_count > token_count or sink_scores != _top_sinks(key_l2_norms, min(sink_count, token_count)):
        _fail("semantics", "stats sidecar sink scores are not the exact top-m norms")
    if encode_sidecar(channel_ranges, key_l2_norms, sink_scores) != data:
        _fail("reserved", "stats sidecar is not canonical")
    return channel_ranges, key_l2_norms, sink_scores


def derive_sidecar_f16(
    tokens: int, channels: int, sink_count: int, data: bytes
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]]:
    """Derive the sidecar from one token-major fp16 plane (tokens x channels)."""
    data = bytes(data)
    if (
        tokens == 0
        or tokens > MAX_SIDECAR_TOKENS
        or channels == 0
        or channels > MAX_SIDECAR_CHANNELS
        or sink_count == 0
        or sink_count > MAX_SINK_SCORES
    ):
        _fail("bounds", "stats sidecar shape is outside bounded limits")
    if len(data) != tokens * channels * 2:
        _fail("bounds", "stats sidecar source bytes do not match the declared shape")

    def element(token: int, channel: int) -> float:
        bits = struct.unpack_from("<H", data, (token * channels + channel) * 2)[0]
        value = _f16_to_f32(bits)
        if not math.isfinite(value):
            _fail("semantics", "stats sidecar source contains a non-finite element")
        return value

    channel_ranges = []
    for channel in range(channels):
        values = [element(token, channel) for token in range(tokens)]
        channel_ranges.append((_f32_to_f16(min(values)), _f32_to_f16(max(values))))
    key_l2_norms = []
    for token in range(tokens):
        summed = sum(element(token, channel) ** 2 for channel in range(channels))
        # Match the Rust f64 accumulate -> f32 -> f16 rounding chain.
        root32 = struct.unpack("<f", struct.pack("<f", math.sqrt(summed)))[0]
        key_l2_norms.append(_f32_to_f16(root32))
    sink_scores = _top_sinks(key_l2_norms, min(sink_count, tokens))
    encoded = encode_sidecar(channel_ranges, key_l2_norms, sink_scores)
    if len(encoded) > MAX_STATS_SIDECAR_BYTES:
        _fail("bounds", "stats sidecar exceeds the chunk header capacity")
    return channel_ranges, key_l2_norms, sink_scores


def encode_chunk(
    plaintext: bytes,
    tenant: bytes,
    family: Family,
    state_key: StateKey,
    span: ChunkSpan,
    epoch: int,
    encrypt: bool,
    keys: KeySchedule,
    *,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    stats_sidecar: bytes | None = None,
) -> ChunkObject:
    plaintext = bytes(plaintext)
    if epoch == 0:
        _fail("semantics", "chunk key epoch must be nonzero")
    if span.plaintext_bytes != len(plaintext) or span.token_count == 0:
        _fail("semantics", "chunk plaintext does not match its declared span")
    state = _family_state(family, state_key)
    if state.codec_version != 1:
        _fail("codec", "unsupported chunk codec version")
    encoded = encode_codec_frame(state.codec, plaintext)
    family_id = representation_family_id(family)
    content = chunk_id(keys.chunk_identity, tenant, family, state_key, span, plaintext)
    if encrypt:
        salt = secrets.token_bytes(16) if salt is None else bytes(salt)
        nonce = secrets.token_bytes(12) if nonce is None else bytes(nonce)
        if len(salt) != 16 or len(nonce) != 12:
            _fail("bounds", "chunk salt or nonce length is invalid")
    else:
        if salt not in (None, bytes(16)) or nonce not in (None, bytes(12)):
            _fail("reserved", "unencrypted chunk salt or nonce is nonzero")
        salt, nonce = bytes(16), bytes(12)
    stats_digest = None
    if stats_sidecar is not None:
        stats_sidecar = bytes(stats_sidecar)
        if len(stats_sidecar) == 0 or len(stats_sidecar) > MAX_STATS_SIDECAR_BYTES:
            _fail("bounds", "stats sidecar exceeds the chunk header capacity")
        # Fail closed at persist: only a canonical sidecar may be attached.
        decode_sidecar(stats_sidecar)
        stats_digest = _sha256(stats_sidecar)
    object_key = _object_key(
        keys,
        tenant,
        family_id,
        content,
        epoch,
        state.codec,
        state.codec_version,
        encrypt,
        salt,
        nonce,
        stats_digest,
    )
    payload_bytes = len(encoded) + (16 if encrypt else 0)
    object_bytes = ((CHUNK_HEADER_BYTES + payload_bytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    if object_bytes > MAX_CHUNK_OBJECT_BYTES:
        _fail("bounds", "encoded chunk object exceeds bound")
    header = bytearray(CHUNK_HEADER_BYTES)
    header[:8] = CHUNK_MAGIC
    struct.pack_into(
        "<HHIIHHIIIIQ",
        header,
        8,
        WIRE_VERSION,
        CHUNK_HEADER_BYTES,
        ALIGNMENT,
        int(encrypt),
        state.codec,
        state.codec_version,
        len(plaintext),
        len(encoded),
        payload_bytes,
        object_bytes,
        epoch,
    )
    header[48:80] = _id32(tenant)
    header[80:112] = family_id
    header[112:144] = content
    header[144:176] = object_key
    header[176:192] = salt
    header[192:204] = nonce
    if stats_sidecar is not None:
        header[CHUNK_HEADER_SIDECAR_OFFSET:CHUNK_HEADER_SIDECAR_OFFSET + 2] = _u16(len(stats_sidecar))
        header[CHUNK_HEADER_SIDECAR_OFFSET + 2:CHUNK_HEADER_SIDECAR_OFFSET + 2 + len(stats_sidecar)] = stats_sidecar
    header[204:236] = _sha256(bytes(header))
    payload = (
        _aead_encrypt(_chunk_encryption_key(keys, content, object_key, salt), nonce, encoded, bytes(header))
        if encrypt
        else encoded
    )
    data = bytes(header) + payload + bytes(object_bytes - CHUNK_HEADER_BYTES - len(payload))
    return ChunkObject(content, object_key, _sha256(data), len(plaintext), data)


def decode_chunk_with_stats(
    data: bytes,
    expected: ChunkRef,
    span: ChunkSpan,
    tenant: bytes,
    family: Family,
    state_key: StateKey,
    keys: KeySchedule,
) -> tuple[bytes, tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]] | None]:
    data = bytes(data)
    if len(data) < CHUNK_HEADER_BYTES or len(data) > MAX_CHUNK_OBJECT_BYTES or len(data) % ALIGNMENT:
        _fail("bounds", "invalid chunk object length")
    if data[:8] != CHUNK_MAGIC:
        _fail("bad_magic", "invalid production chunk magic")
    version, header_bytes, alignment = struct.unpack_from("<HHI", data, 8)
    if version != WIRE_VERSION or header_bytes != CHUNK_HEADER_BYTES or alignment != ALIGNMENT:
        _fail("bad_magic", "invalid chunk header contract")
    flags = struct.unpack_from("<I", data, 16)[0]
    if flags & ~1:
        _fail("reserved", "unknown chunk flags")
    codec, codec_version = struct.unpack_from("<HH", data, 20)
    _enum(codec, {CODEC_RAW, CODEC_LOSSLESS}, "codec")
    if codec_version != 1:
        _fail("codec", "unsupported chunk codec version")
    plaintext_bytes, encoded_bytes, payload_bytes, object_bytes = struct.unpack_from("<IIII", data, 24)
    if (
        plaintext_bytes == 0
        or plaintext_bytes > MAX_CHUNK_PLAINTEXT
        or not CODEC_FRAME_HEADER_BYTES <= encoded_bytes <= MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD
        or object_bytes != len(data)
    ):
        _fail("bounds", "invalid chunk size fields")
    encrypted = bool(flags & 1)
    if payload_bytes != encoded_bytes + (16 if encrypted else 0):
        _fail("bounds", "chunk payload length does not match encryption mode")
    payload_end = CHUNK_HEADER_BYTES + payload_bytes
    if payload_end > len(data):
        _fail("truncated", "truncated chunk payload")
    if any(data[payload_end:]):
        _fail("reserved", "chunk padding is nonzero")
    # Sidecar presence is the nonzero length prefix in the header tail; no
    # flag bit is consumed, preserving the pre-sidecar flag contract.
    stats_sidecar = None
    stats_digest = None
    sidecar_len = struct.unpack_from("<H", data, CHUNK_HEADER_SIDECAR_OFFSET)[0]
    if sidecar_len != 0:
        if sidecar_len > MAX_STATS_SIDECAR_BYTES:
            _fail("bounds", "chunk stats sidecar length is outside bounds")
        sidecar_start = CHUNK_HEADER_SIDECAR_OFFSET + 2
        sidecar_end = sidecar_start + sidecar_len
        stats_sidecar = decode_sidecar(data[sidecar_start:sidecar_end])
        if any(data[sidecar_end:CHUNK_HEADER_BYTES]):
            _fail("reserved", "chunk header sidecar padding is nonzero")
        stats_digest = _sha256(data[sidecar_start:sidecar_end])
    else:
        if any(data[CHUNK_HEADER_SIDECAR_OFFSET:CHUNK_HEADER_BYTES]):
            _fail("reserved", "chunk header reserved bytes are nonzero")
    if not encrypted and any(data[176:204]):
        _fail("reserved", "unencrypted chunk salt or nonce is nonzero")
    if len(data) != expected.object_bytes or plaintext_bytes != expected.plaintext_bytes or plaintext_bytes != span.plaintext_bytes:
        _fail("bounds", "chunk reference size mismatch")
    if _sha256(data) != _id32(expected.object_digest):
        _fail("authentication", "chunk object digest mismatch")
    header = bytearray(data[:CHUNK_HEADER_BYTES])
    expected_header = bytes(header[204:236])
    header[204:236] = bytes(32)
    if _sha256(bytes(header)) != expected_header:
        _fail("authentication", "chunk header digest mismatch")
    state = _family_state(family, state_key)
    if codec != state.codec or codec_version != state.codec_version:
        _fail("semantics", "chunk codec does not match representation family")
    actual_tenant = data[48:80]
    family_id = data[80:112]
    epoch = struct.unpack_from("<Q", data, 40)[0]
    if actual_tenant != _id32(tenant) or family_id != representation_family_id(family) or epoch != expected.key_epoch:
        _fail("authentication", "chunk namespace, family, or epoch mismatch")
    content = data[112:144]
    object_key = data[144:176]
    salt = data[176:192]
    nonce = data[192:204]
    derived_object = _object_key(
        keys,
        actual_tenant,
        family_id,
        content,
        expected.key_epoch,
        codec,
        codec_version,
        encrypted,
        salt,
        nonce,
        stats_digest,
    )
    if content != expected.chunk_id or object_key != expected.object_key or object_key != derived_object:
        _fail("authentication", "chunk identity or object key mismatch")
    stored = data[CHUNK_HEADER_BYTES:payload_end]
    if encrypted:
        encoded = _aead_decrypt(
            _chunk_encryption_key(keys, content, object_key, salt),
            nonce,
            stored,
            data[:CHUNK_HEADER_BYTES],
            "chunk AEAD authentication failed",
        )
    else:
        encoded = stored
    if len(encoded) != encoded_bytes:
        _fail("bounds", "chunk encoded length mismatch")
    plaintext = decode_codec_frame(codec, encoded, plaintext_bytes)
    if chunk_id(keys.chunk_identity, actual_tenant, family, state_key, span, plaintext) != content:
        _fail("authentication", "chunk plaintext identity mismatch")
    return plaintext, stats_sidecar


def decode_chunk(
    data: bytes,
    expected: ChunkRef,
    span: ChunkSpan,
    tenant: bytes,
    family: Family,
    state_key: StateKey,
    keys: KeySchedule,
) -> bytes:
    return decode_chunk_with_stats(data, expected, span, tenant, family, state_key, keys)[0]


@dataclass(frozen=True)
class PackHeader:
    tenant_namespace: bytes
    manifest_id: bytes
    key_epoch: int
    manifest_bytes: int
    plaintext_manifest_bytes: int
    encrypted: bool


def _manifest_encryption_key(keys: KeySchedule, identity: bytes, salt: bytes) -> bytes:
    return _hkdf(
        keys.manifest_encryption,
        salt,
        b"kvpack/v1/manifest-aead\0" + identity,
    )


def _pack_authentication(
    keys: KeySchedule,
    header: bytes,
    body: bytes,
    manifest_length: int,
    epoch: int,
    file_length: int,
) -> bytes:
    return _hmac(
        keys.manifest_auth,
        b"kvpack/v1/manifest-auth\0",
        header,
        body,
        _u64(manifest_length),
        _u64(epoch),
        _u64(file_length),
    )


def encode_pack(
    manifest: Manifest,
    keys: KeySchedule,
    encrypt: bool,
    *,
    salt: bytes | None = None,
    nonce: bytes | None = None,
) -> tuple[bytes, bytes]:
    validate_manifest(manifest)
    canonical = encode_manifest(manifest)
    if len(canonical) > MAX_MANIFEST_BYTES:
        _fail("bounds", "manifest exceeds production bound")
    identity = manifest_id(canonical)
    stored_length = len(canonical) + (16 if encrypt else 0)
    file_length = PACK_HEADER_BYTES + stored_length + PACK_FOOTER_BYTES
    if encrypt:
        salt = secrets.token_bytes(16) if salt is None else bytes(salt)
        nonce = secrets.token_bytes(12) if nonce is None else bytes(nonce)
        if len(salt) != 16 or len(nonce) != 12:
            _fail("bounds", "manifest salt or nonce length is invalid")
    else:
        if salt not in (None, bytes(16)) or nonce not in (None, bytes(12)):
            _fail("reserved", "unencrypted manifest salt or nonce is nonzero")
        salt, nonce = bytes(16), bytes(12)
    header = bytearray(PACK_HEADER_BYTES)
    header[:8] = PACK_MAGIC
    struct.pack_into("<HHIIIQQQ", header, 8, WIRE_VERSION, PACK_HEADER_BYTES, ALIGNMENT, int(encrypt), 0, stored_length, len(canonical), manifest.key_epoch)
    header[56:88] = manifest.tenant_namespace
    header[88:120] = identity
    header[120:136] = salt
    header[136:148] = nonce
    header[148:180] = _sha256(bytes(header))
    body = (
        _aead_encrypt(_manifest_encryption_key(keys, identity, salt), nonce, canonical, bytes(header))
        if encrypt
        else canonical
    )
    footer = bytearray(PACK_FOOTER_BYTES)
    footer[:8] = FOOTER_MAGIC
    struct.pack_into("<HHIQQQQ", footer, 8, WIRE_VERSION, PACK_FOOTER_BYTES, 0, len(body), manifest.key_epoch, 0, file_length)
    footer[48:80] = identity
    footer[80:112] = _pack_authentication(keys, bytes(header), body, len(body), manifest.key_epoch, file_length)
    return bytes(header) + body + bytes(footer), identity


def _parse_pack_header(data: bytes) -> PackHeader:
    if len(data) < PACK_HEADER_BYTES:
        _fail("truncated", "truncated production pack header")
    header = data[:PACK_HEADER_BYTES]
    if header[:8] != PACK_MAGIC:
        _fail("bad_magic", "invalid production pack magic")
    version, header_bytes, alignment = struct.unpack_from("<HHI", header, 8)
    if version != WIRE_VERSION or header_bytes != PACK_HEADER_BYTES or alignment != ALIGNMENT:
        _fail("bad_magic", "invalid production pack header contract")
    flags = struct.unpack_from("<I", header, 16)[0]
    if flags & ~1:
        _fail("reserved", "unknown pack flags")
    if struct.unpack_from("<I", header, 20)[0] != 0 or struct.unpack_from("<Q", header, 48)[0] != 0 or any(header[180:]):
        _fail("reserved", "pack header reserved bytes are nonzero")
    encrypted = bool(flags & 1)
    if not encrypted and any(header[120:148]):
        _fail("reserved", "unencrypted manifest salt or nonce is nonzero")
    return PackHeader(
        header[56:88],
        header[88:120],
        struct.unpack_from("<Q", header, 40)[0],
        struct.unpack_from("<Q", header, 24)[0],
        struct.unpack_from("<Q", header, 32)[0],
        encrypted,
    )


def _verify_pack_header_digest(header: bytes) -> None:
    copy = bytearray(header)
    expected = bytes(copy[148:180])
    copy[148:180] = bytes(32)
    if _sha256(bytes(copy)) != expected:
        _fail("checksum", "pack header digest mismatch")


def inspect_pack_header(data: bytes) -> PackHeader:
    data = bytes(data)
    parsed = _parse_pack_header(data)
    _verify_pack_header_digest(data[:PACK_HEADER_BYTES])
    return parsed


def decode_pack(data: bytes, keys: KeySchedule) -> Manifest:
    data = bytes(data)
    if len(data) < PACK_HEADER_BYTES + PACK_FOOTER_BYTES:
        _fail("truncated", "truncated production pack")
    parsed = _parse_pack_header(data)
    if parsed.manifest_bytes > MAX_MANIFEST_BYTES + 16 or parsed.plaintext_manifest_bytes > MAX_MANIFEST_BYTES:
        _fail("bounds", "manifest exceeds production bound")
    footer_offset = len(data) - PACK_FOOTER_BYTES
    footer = data[footer_offset:]
    if footer[:8] != FOOTER_MAGIC:
        _fail("bad_magic", "invalid production commit footer magic")
    version, footer_bytes, reserved = struct.unpack_from("<HHI", footer, 8)
    if (
        version != WIRE_VERSION
        or footer_bytes != PACK_FOOTER_BYTES
        or reserved != 0
        or struct.unpack_from("<Q", footer, 32)[0] != 0
        or any(footer[112:])
    ):
        _fail("reserved", "invalid production commit footer")
    body = data[PACK_HEADER_BYTES:footer_offset]
    if (
        len(body) != parsed.manifest_bytes
        or struct.unpack_from("<Q", footer, 16)[0] != parsed.manifest_bytes
        or struct.unpack_from("<Q", footer, 24)[0] != parsed.key_epoch
        or struct.unpack_from("<Q", footer, 40)[0] != len(data)
        or footer[48:80] != parsed.manifest_id
    ):
        _fail("bounds", "pack footer binding mismatch")
    _verify_pack_header_digest(data[:PACK_HEADER_BYTES])
    expected_tag = _pack_authentication(
        keys,
        data[:PACK_HEADER_BYTES],
        body,
        len(body),
        parsed.key_epoch,
        len(data),
    )
    if not hmac.compare_digest(expected_tag, footer[80:112]):
        _fail("authentication", "manifest HMAC authentication failed")
    if parsed.encrypted:
        if len(body) < 16:
            _fail("truncated", "truncated encrypted manifest")
        canonical = _aead_decrypt(
            _manifest_encryption_key(keys, parsed.manifest_id, data[120:136]),
            data[136:148],
            body,
            data[:PACK_HEADER_BYTES],
            "manifest AEAD authentication failed",
        )
    else:
        canonical = body
    if len(canonical) != parsed.plaintext_manifest_bytes or manifest_id(canonical) != parsed.manifest_id:
        _fail("authentication", "manifest content identity mismatch")
    manifest = decode_manifest(canonical)
    if encode_manifest(manifest) != canonical:
        _fail("reserved", "manifest encoding is not canonical")
    if manifest.tenant_namespace != parsed.tenant_namespace or manifest.key_epoch != parsed.key_epoch:
        _fail("authentication", "manifest header identity mismatch")
    validate_manifest(manifest)
    return manifest


@dataclass(frozen=True)
class ReferenceFixture:
    root_key: bytes
    keys: KeySchedule
    plaintext: bytes
    span: ChunkSpan
    chunk: ChunkObject
    manifest: Manifest
    pack: bytes
    pack_id: bytes
    auxiliary_inputs: tuple[AuxiliaryInput, ...]
    tokens: tuple[int, ...]


def reference_fixture(codec: int, encrypt_chunk: bool, encrypt_pack: bool) -> ReferenceFixture:
    """Deterministic complete fixture shared only by conformance tests."""

    identity = lambda value: bytes((value,)) * 32
    root = identity(99)
    tenant = identity(1)
    keys = derive_key_schedule(root, tenant, 7)
    model = SemanticModel(*(identity(value) for value in range(10, 15)))
    state_key = StateKey(0, "attention.k")
    family = Family(
        identity(2),
        MODE_NATIVE,
        256,
        identity(3),
        identity(4),
        (
            FamilyState(
                state_key,
                CACHE_ORDINARY_KV,
                1,
                codec,
                1,
                LAYOUT_CONTIGUOUS,
                TOKEN_AXIS_DIRECT,
                0,
                4,
                (None, 4),
            ),
        ),
    )
    inputs = (AuxiliaryInput(identity(20), identity(21)),)
    tokens = (1, 2, 3, 4)
    input_cut, _ = derive_input_cut(keys.prefix, tenant, model, family, tokens, inputs)
    plaintext = bytes(range(16))
    span = ChunkSpan(0, 4, 0, 16)
    chunk = encode_chunk(
        plaintext,
        tenant,
        family,
        state_key,
        span,
        7,
        encrypt_chunk,
        keys,
        salt=bytes(range(0xA0, 0xB0)) if encrypt_chunk else None,
        nonce=bytes(range(0xB0, 0xBC)) if encrypt_chunk else None,
    )
    reference = ChunkRef(
        chunk.chunk_id,
        chunk.object_key,
        chunk.object_digest,
        7,
        chunk.plaintext_bytes,
        len(chunk.data),
    )
    schema = RealizedSchema(
        ManifestKind(),
        (
            RealizedState(
                state_key,
                Shape((4, 4)),
                Shape((4, 4)),
                (4, 1),
                0,
                4,
                0,
                16,
                16,
                4,
                0,
                (span,),
            ),
        ),
        (AtomicGroup(1, (state_key,)),),
        16,
        16,
    )
    manifest = Manifest(
        tenant,
        7,
        model,
        input_cut,
        family,
        schema,
        (StateManifest(state_key, (reference,)),),
    )
    pack, pack_id = encode_pack(
        manifest,
        keys,
        encrypt_pack,
        salt=bytes(range(0xC0, 0xD0)) if encrypt_pack else None,
        nonce=bytes(range(0xD0, 0xDC)) if encrypt_pack else None,
    )
    return ReferenceFixture(root, keys, plaintext, span, chunk, manifest, pack, pack_id, inputs, tokens)


@dataclass(frozen=True)
class SidecarFixture:
    keys: KeySchedule
    tenant: bytes
    family: Family
    state_key: StateKey
    span: ChunkSpan
    plaintext: bytes
    sidecar: bytes
    chunk: ChunkObject
    reference: ChunkRef


def reference_sidecar_fixture() -> SidecarFixture:
    """Deterministic M7 sidecar chunk fixture: 4 tokens x 4 fp16 channels,
    unencrypted, top-2 sinks. Shares the reference tenant/keys/epoch so the
    multi-language parity tests can mix it with the base fixtures."""

    identity = lambda value: bytes((value,)) * 32
    tenant = identity(1)
    keys = derive_key_schedule(identity(99), tenant, 7)
    state_key = StateKey(0, "attention.k")
    family = Family(
        identity(2),
        MODE_NATIVE,
        256,
        identity(3),
        identity(4),
        (
            FamilyState(
                state_key,
                CACHE_ORDINARY_KV,
                6,  # DTYPE F16
                CODEC_RAW,
                1,
                LAYOUT_CONTIGUOUS,
                TOKEN_AXIS_DIRECT,
                0,
                4,
                (None, 4),
            ),
        ),
    )
    values = (
        (1.0, 0.5, -1.0, 2.0),
        (0.0, 1.5, -0.5, 1.0),
        (-2.0, 0.25, 3.0, 0.0),
        (4.0, -0.75, 0.5, 1.0),
    )
    plaintext = b"".join(struct.pack("<e", value) for token in values for value in token)
    span = ChunkSpan(0, 4, 0, len(plaintext))
    sidecar = encode_sidecar(*derive_sidecar_f16(4, 4, 2, plaintext))
    chunk = encode_chunk(
        plaintext, tenant, family, state_key, span, 7, False, keys,
        stats_sidecar=sidecar,
    )
    reference = ChunkRef(
        chunk.chunk_id,
        chunk.object_key,
        chunk.object_digest,
        7,
        chunk.plaintext_bytes,
        len(chunk.data),
    )
    return SidecarFixture(keys, tenant, family, state_key, span, plaintext, sidecar, chunk, reference)


def reference_delta_chain(
    codec: int, encrypt_chunk: bool, encrypt_pack: bool
) -> tuple[ReferenceFixture, ...]:
    """Deterministic full base followed by the maximum seven v1 deltas."""

    seed = reference_fixture(codec, False, False)
    identity = lambda value: bytes((value,)) * 32
    root = seed.root_key
    keys = seed.keys
    tenant = identity(1)
    model = seed.manifest.semantic_model
    family = seed.manifest.family
    state_key = family.states[0].key
    inputs = seed.auxiliary_inputs
    all_tokens = tuple(range(1, 9))
    result: list[ReferenceFixture] = []
    parent: ReferenceFixture | None = None

    for stage in range(8):
        token_count = stage + 1
        tokens = all_tokens[:token_count]
        input_cut, _ = derive_input_cut(keys.prefix, tenant, model, family, tokens, inputs)
        plaintext_offset = stage * 4
        plaintext = bytes(range(plaintext_offset, plaintext_offset + 4))
        span = ChunkSpan(stage, 1, plaintext_offset, 4)
        chunk = encode_chunk(
            plaintext,
            tenant,
            family,
            state_key,
            span,
            7,
            encrypt_chunk,
            keys,
            salt=bytes((0x20 + stage + index) & 0xFF for index in range(16))
            if encrypt_chunk
            else None,
            nonce=bytes((0x40 + stage + index) & 0xFF for index in range(12))
            if encrypt_chunk
            else None,
        )
        reference = ChunkRef(
            chunk.chunk_id,
            chunk.object_key,
            chunk.object_digest,
            7,
            chunk.plaintext_bytes,
            len(chunk.data),
        )
        kind = (
            ManifestKind()
            if parent is None
            else ManifestKind(parent.pack_id, parent.manifest.input_cut, stage)
        )
        schema = RealizedSchema(
            kind,
            (
                RealizedState(
                    state_key,
                    Shape((token_count, 4)),
                    Shape((token_count if stage == 0 else 1, 4)),
                    (4, 1),
                    0 if stage == 0 else stage,
                    token_count if stage == 0 else 1,
                    0 if stage == 0 else plaintext_offset,
                    token_count * 4 if stage == 0 else 4,
                    token_count * 4,
                    token_count,
                    0,
                    (span,),
                ),
            ),
            (AtomicGroup(1, (state_key,)),),
            token_count * 4 if stage == 0 else 4,
            token_count * 4,
        )
        manifest = Manifest(
            tenant,
            7,
            model,
            input_cut,
            family,
            schema,
            (StateManifest(state_key, (reference,)),),
        )
        pack, pack_id = encode_pack(
            manifest,
            keys,
            encrypt_pack,
            salt=bytes((0x60 + stage + index) & 0xFF for index in range(16))
            if encrypt_pack
            else None,
            nonce=bytes((0x80 + stage + index) & 0xFF for index in range(12))
            if encrypt_pack
            else None,
        )
        current = ReferenceFixture(
            root,
            keys,
            plaintext,
            span,
            chunk,
            manifest,
            pack,
            pack_id,
            inputs,
            tokens,
        )
        result.append(current)
        parent = current
    return tuple(result)
