pub use kvpack_core::{
    decode_authenticated_pack, decode_chunk, decode_codec_frame, derive_input_cut,
    encode_authenticated_pack, encode_chunk, encode_codec_frame, inspect_pack_header,
    representation_family_id, validate_manifest, AtomicGroup, AuxiliaryInputId, CacheKind,
    ChunkEncoding, ChunkRef, ChunkSpan, Codec, CutManifest, DType, FamilyState, Id32, KeySchedule,
    Layout, ManifestKind, PackError, RealizedCutSchemaId, RealizedStateSchema,
    RepresentationFamilyId, RepresentationMode, SemanticModelId, Shape, StateDeclaration, StateKey,
    StateManifest, StaticDimension, TokenAxisRule, ValidationContext,
};

pub fn id(byte: u8) -> Id32 {
    [byte; 32]
}

pub fn semantic() -> SemanticModelId {
    SemanticModelId {
        weights_config: id(10),
        adapters: id(11),
        tokenizer_template: id(12),
        position_semantics: id(13),
        qualified_math: id(14),
    }
}

pub fn family(codec: Codec) -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(2),
        mode: RepresentationMode::Native,
        page_size_tokens: 256,
        topology: id(3),
        shard_map: id(4),
        states: vec![FamilyState {
            key: StateKey::new(0, "attention.k"),
            cache_kind: CacheKind::OrdinaryKv,
            dtype: DType::U8,
            codec,
            codec_version: 1,
            layout: Layout::Contiguous,
            token_axis_rule: TokenAxisRule::Direct,
            token_axis: 0,
            elements_per_token: 4,
            dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(4)],
            dependencies: vec![],
        }],
    }
}

pub fn fixture(
    codec: Codec,
    encrypt_chunk: bool,
    encrypt_manifest: bool,
) -> (
    Vec<u8>,
    Vec<u8>,
    ChunkRef,
    ChunkSpan,
    CutManifest,
    KeySchedule,
) {
    let root = id(99);
    let tenant = id(1);
    let keys = KeySchedule::derive(&root, &tenant, 7).unwrap();
    let family = family(codec);
    let semantic_model = semantic();
    let auxiliary = [AuxiliaryInputId {
        type_id: id(20),
        value_id: id(21),
    }];
    let (input_cut, _) = derive_input_cut(
        keys.prefix_key(),
        &tenant,
        &semantic_model,
        &family,
        &[1, 2, 3, 4],
        &auxiliary,
    )
    .unwrap();
    let plaintext: Vec<u8> = (0..16).collect();
    let span = ChunkSpan {
        token_start: 0,
        token_count: 4,
        plaintext_offset: 0,
        plaintext_bytes: plaintext.len() as u32,
    };
    let state_key = family.states[0].key.clone();
    let object = encode_chunk(
        &plaintext,
        &ChunkEncoding {
            tenant_namespace: tenant,
            family: &family,
            state_key: &state_key,
            span,
            key_epoch: 7,
            encrypt: encrypt_chunk,
            stats_sidecar: None,
        },
        &keys,
    )
    .unwrap();
    let reference = ChunkRef {
        chunk_id: object.chunk_id,
        object_key: object.object_key,
        object_digest: object.object_digest,
        key_epoch: 7,
        plaintext_bytes: object.plaintext_bytes,
        object_bytes: object.bytes.len() as u32,
    };
    let schema = RealizedCutSchemaId {
        kind: ManifestKind::Full,
        states: vec![RealizedStateSchema {
            key: state_key.clone(),
            full_shape: Shape::new(&[4, 4]).unwrap(),
            segment_shape: Shape::new(&[4, 4]).unwrap(),
            strides: vec![4, 1],
            logical_start: 0,
            logical_count: 4,
            physical_offset_bytes: 0,
            physical_span_bytes: 16,
            complete_physical_bytes: 16,
            absolute_position: 4,
            window: 0,
            chunk_spans: vec![span],
        }],
        atomic_groups: vec![AtomicGroup {
            id: 1,
            states: vec![state_key.clone()],
        }],
        segment_restored_bytes: 16,
        complete_restored_bytes: 16,
    };
    let manifest = CutManifest {
        tenant_namespace: tenant,
        key_epoch: 7,
        semantic_model,
        input_cut,
        family,
        realized_schema: schema,
        states: vec![StateManifest {
            key: state_key,
            chunks: vec![reference.clone()],
        }],
    };
    let pack = encode_authenticated_pack(
        &manifest,
        &keys,
        encrypt_manifest,
        &ValidationContext::default(),
    )
    .unwrap();
    (object.bytes, pack.bytes, reference, span, manifest, keys)
}

pub fn delta_fixture(depth: u8) -> (CutManifest, KeySchedule, Vec<u8>) {
    let (_, _, _, _, mut manifest, keys) = fixture(Codec::Raw, false, false);
    let auxiliary = [AuxiliaryInputId {
        type_id: id(20),
        value_id: id(21),
    }];
    let (parent_cut, _) = derive_input_cut(
        keys.prefix_key(),
        &manifest.tenant_namespace,
        &manifest.semantic_model,
        &manifest.family,
        &[1, 2],
        &auxiliary,
    )
    .unwrap();
    let span = ChunkSpan {
        token_start: 2,
        token_count: 2,
        plaintext_offset: 8,
        plaintext_bytes: 8,
    };
    let plaintext: Vec<u8> = (8..16).collect();
    let state_key = manifest.family.states[0].key.clone();
    let object = encode_chunk(
        &plaintext,
        &ChunkEncoding {
            tenant_namespace: manifest.tenant_namespace,
            family: &manifest.family,
            state_key: &state_key,
            span,
            key_epoch: 7,
            encrypt: false,
            stats_sidecar: None,
        },
        &keys,
    )
    .unwrap();
    manifest.realized_schema = RealizedCutSchemaId {
        kind: ManifestKind::Delta {
            parent: id(50),
            parent_cut,
            depth,
        },
        states: vec![RealizedStateSchema {
            key: state_key.clone(),
            full_shape: Shape::new(&[4, 4]).unwrap(),
            segment_shape: Shape::new(&[2, 4]).unwrap(),
            strides: vec![4, 1],
            logical_start: 2,
            logical_count: 2,
            physical_offset_bytes: 8,
            physical_span_bytes: 8,
            complete_physical_bytes: 16,
            absolute_position: 4,
            window: 0,
            chunk_spans: vec![span],
        }],
        atomic_groups: vec![AtomicGroup {
            id: 1,
            states: vec![state_key.clone()],
        }],
        segment_restored_bytes: 8,
        complete_restored_bytes: 16,
    };
    manifest.states = vec![StateManifest {
        key: state_key,
        chunks: vec![ChunkRef {
            chunk_id: object.chunk_id,
            object_key: object.object_key,
            object_digest: object.object_digest,
            key_epoch: 7,
            plaintext_bytes: object.plaintext_bytes,
            object_bytes: object.bytes.len() as u32,
        }],
    }];
    (manifest, keys, object.bytes)
}
