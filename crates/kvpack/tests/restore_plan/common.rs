pub(crate) use std::collections::BTreeMap;
pub(crate) use std::sync::Arc;

pub(crate) use kvpack::wire::{
    AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, InputCutId, Layout,
    ManifestDeclaration, ManifestKind, RepresentationFamilyId, RepresentationMode, SemanticModelId,
    Shape, StateDeclaration, StateKey, StaticDimension, TokenAxisRule, ValidationContext,
};
pub(crate) use kvpack::{
    ArtifactWriter, AuthenticatedRestorePlan, LocalStore, RestoreAvailableSource,
    RestoreCancellation, RestoreLimits, RestoreRequest, RestoreResourceRequirements,
    RestoreStatePlan, RestoreTier, StoreConfig, StoreError, UtilizationPolicy, VerifiedRestoreSink,
    WritePolicy,
};

pub fn id(value: u8) -> [u8; 32] {
    [value; 32]
}

pub fn semantic() -> SemanticModelId {
    SemanticModelId {
        weights_config: id(1),
        adapters: id(2),
        tokenizer_template: id(3),
        position_semantics: id(4),
        qualified_math: id(5),
    }
}

pub fn auxiliary() -> Vec<AuxiliaryInputId> {
    vec![AuxiliaryInputId {
        type_id: id(30),
        value_id: id(31),
    }]
}

pub fn family() -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Native,
        page_size_tokens: 256,
        topology: id(7),
        shard_map: id(8),
        states: vec![FamilyState {
            key: StateKey::new(0, "k"),
            cache_kind: CacheKind::OrdinaryKv,
            dtype: DType::U8,
            codec: Codec::Raw,
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

pub fn two_state_family() -> RepresentationFamilyId {
    let mut family = family();
    let mut value = family.states[0].clone();
    value.key = StateKey::new(0, "v");
    family.states.push(value);
    family
}

pub struct Fixture {
    pub temp: tempfile::TempDir,
    pub store: Arc<LocalStore>,
}

pub fn fixture(name: &[u8]) -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let key = kvpack::load_store_key(&key_path, temp.path()).unwrap();
    let store = Arc::new(
        LocalStore::open(
            StoreConfig {
                object_root: temp.path().join("objects"),
                catalog_path: temp.path().join("catalog/catalog.sqlite"),
                operator_tenant_id: name.to_vec(),
                key_epoch: 1,
                minimum_readable_key_epoch: 1,
                catalog_epoch: 1,
                quota_bytes: 256 * 1024 * 1024,
                staging_quota_bytes: 256 * 1024 * 1024,
                endurance_bytes_per_five_minutes: 256 * 1024 * 1024,
            },
            key,
        )
        .unwrap(),
    );
    Fixture { temp, store }
}

pub fn declaration(
    tokens: usize,
    kind: ManifestKind,
    logical_start: u64,
    logical_count: u64,
) -> ManifestDeclaration {
    ManifestDeclaration {
        semantic_model: semantic(),
        input_tokens: (0..tokens as u32).collect(),
        auxiliary_inputs: auxiliary(),
        family: family(),
        kind,
        states: vec![StateDeclaration {
            key: StateKey::new(0, "k"),
            full_shape: Shape::new(&[tokens as u64, 4]).unwrap(),
            segment_shape: Shape::new(&[logical_count, 4]).unwrap(),
            strides: vec![4, 1],
            logical_start,
            logical_count,
            absolute_position: tokens as u64,
            window: 0,
            atomic_group: 1,
        }],
    }
}

pub fn two_state_declaration(tokens: usize) -> ManifestDeclaration {
    ManifestDeclaration {
        semantic_model: semantic(),
        input_tokens: (0..tokens as u32).collect(),
        auxiliary_inputs: auxiliary(),
        family: two_state_family(),
        kind: ManifestKind::Full,
        states: ["k", "v"]
            .into_iter()
            .map(|name| StateDeclaration {
                key: StateKey::new(0, name),
                full_shape: Shape::new(&[tokens as u64, 4]).unwrap(),
                segment_shape: Shape::new(&[tokens as u64, 4]).unwrap(),
                strides: vec![4, 1],
                logical_start: 0,
                logical_count: tokens as u64,
                absolute_position: tokens as u64,
                window: 0,
                atomic_group: 7,
            })
            .collect(),
    }
}

pub fn publish_delta_chain(
    store: Arc<LocalStore>,
) -> (kvpack::PublishedArtifact, kvpack::PublishedArtifact) {
    let family = family();
    let mut root = ArtifactWriter::begin(
        Arc::clone(&store),
        declaration(256, ManifestKind::Full, 0, 256),
        WritePolicy::exact_qualified(id(50), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = root.next_state(StateKey::new(0, "k")).unwrap();
    source.write_all(&vec![1; 256 * 4]).unwrap();
    source.finish().unwrap();
    let root = root.commit().unwrap();
    let parent_cut = store
        .derive_input_cut(
            &semantic(),
            &family,
            &(0..256u32).collect::<Vec<_>>(),
            &auxiliary(),
        )
        .unwrap()
        .0;
    let mut child = ArtifactWriter::begin(
        Arc::clone(&store),
        declaration(
            512,
            ManifestKind::Delta {
                parent: root.manifest_id,
                parent_cut,
                depth: 99,
            },
            256,
            256,
        ),
        WritePolicy::exact_qualified(id(51), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = child.next_state(StateKey::new(0, "k")).unwrap();
    source.write_all(&vec![2; 256 * 4]).unwrap();
    source.finish().unwrap();
    let child = child.commit().unwrap();
    (root, child)
}

pub fn request(tokens: usize, minimum_key_epoch: u64) -> RestoreRequest {
    RestoreRequest {
        semantic_model: semantic(),
        family: family(),
        input_tokens: (0..tokens as u32).collect(),
        auxiliary_inputs: auxiliary(),
        minimum_key_epoch,
        maximum_candidates: 8,
    }
}

pub fn input_cut(store: &LocalStore, tokens: usize) -> InputCutId {
    store
        .derive_input_cut(
            &semantic(),
            &family(),
            &(0..tokens as u32).collect::<Vec<_>>(),
            &auxiliary(),
        )
        .unwrap()
        .0
}

#[derive(Default)]
pub struct ShadowSink {
    pub shadow: BTreeMap<StateKey, Vec<u8>>,
    pub installed: BTreeMap<StateKey, Vec<u8>>,
    pub writes: usize,
    pub begun: bool,
    pub fail_begin: bool,
    pub fail_write: Option<usize>,
    pub fail_commit: bool,
    pub cancel_after_write: Option<(usize, RestoreCancellation)>,
    pub aborted: bool,
    pub reset: bool,
}

impl VerifiedRestoreSink for ShadowSink {
    fn begin_restore(
        &mut self,
        _: [u8; 32],
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
        self.begun = true;
        if self.fail_begin {
            return Err(StoreError::State("injected begin failure"));
        }
        for state in states {
            self.shadow.insert(
                state.declaration.key.clone(),
                vec![0; state.plaintext_bytes as usize],
            );
        }
        Ok(())
    }

    fn write_verified_chunk(
        &mut self,
        state: &StateKey,
        offset: u64,
        plaintext: &[u8],
    ) -> Result<(), StoreError> {
        self.writes += 1;
        if self.fail_write == Some(self.writes) {
            return Err(StoreError::State("injected scatter failure"));
        }
        let target = self.shadow.get_mut(state).unwrap();
        let offset = offset as usize;
        target[offset..offset + plaintext.len()].copy_from_slice(plaintext);
        if let Some((write, cancellation)) = &self.cancel_after_write {
            if *write == self.writes {
                cancellation.cancel();
            }
        }
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        self.installed = std::mem::take(&mut self.shadow);
        if self.fail_commit {
            return Err(StoreError::State("injected commit failure"));
        }
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.shadow.clear();
        self.aborted = true;
    }

    fn reset_restore(&mut self) {
        self.shadow.clear();
        self.installed.clear();
        self.reset = true;
    }
}

pub fn error_category(error: &StoreError) -> &'static str {
    match error {
        StoreError::Authentication(_) | StoreError::Pack(_) => "integrity",
        StoreError::Cancelled => "cancelled",
        StoreError::Quota(_) => "quota",
        StoreError::State(_) => "state",
        _ => "other",
    }
}

pub fn first_chunk_path(fixture: &Fixture) -> std::path::PathBuf {
    chunk_paths(fixture).remove(0)
}

pub fn chunk_paths(fixture: &Fixture) -> Vec<std::path::PathBuf> {
    let chunks = fixture.temp.path().join("objects/chunks");
    let mut paths = std::fs::read_dir(chunks)
        .unwrap()
        .flat_map(|shard| std::fs::read_dir(shard.unwrap().path()).unwrap())
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

pub fn chunk_snapshot(fixture: &Fixture) -> BTreeMap<std::path::PathBuf, Vec<u8>> {
    let chunks = fixture.temp.path().join("objects/chunks");
    std::fs::read_dir(chunks)
        .unwrap()
        .flat_map(|shard| std::fs::read_dir(shard.unwrap().path()).unwrap())
        .map(|entry| {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect()
}

pub fn manifest_path(fixture: &Fixture, manifest_id: [u8; 32]) -> std::path::PathBuf {
    let name = manifest_id
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    fixture
        .temp
        .path()
        .join("objects/manifests")
        .join(&name[..2])
        .join(format!("{name}.kvpack"))
}
