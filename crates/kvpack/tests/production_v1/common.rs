pub(crate) use std::collections::BTreeMap;
pub(crate) use std::sync::Arc;

pub(crate) use kvpack::wire::{
    AtomicGroup, AuxiliaryInputId, CacheKind, ChunkSpan, Codec, DType, FamilyState, Layout,
    ManifestDeclaration, ManifestKind, RealizedCutSchemaId, RealizedStateSchema,
    RepresentationFamilyId, RepresentationMode, SemanticModelId, Shape, StateDeclaration, StateKey,
    StaticDimension, TokenAxisRule,
};
pub(crate) use kvpack::{
    ArtifactLocator, ArtifactWriter, AuthenticatedArtifact, CatalogBackupBounds, FileKeyProvider,
    FsckBounds, InMemoryKeyProvider, InventoryObjectKind, InventorySnapshotBounds, KeyEpochWindow,
    KeyProvider, KeyProviderQualification, LinuxOsKeyStoreProvider, LocalStore,
    MacOsKeychainProvider, MutationReplay, OpenExpectations, ReconciliationBounds,
    RemoteImportFence, RemoteMutation, RestoreSelection, RestoreStatePlan, SourceLeaseState,
    StoreConfig, StoreError, StoreKey, VerifiedRestoreSink, WritePolicy,
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

pub fn family(codec: Codec) -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Native,
        page_size_tokens: 256,
        topology: id(7),
        shard_map: id(8),
        states: ["k", "v"]
            .into_iter()
            .map(|name| FamilyState {
                key: StateKey::new(0, name),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token: 8,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(8)],
                dependencies: vec![],
            })
            .collect(),
    }
}

pub fn states() -> Vec<StateDeclaration> {
    ["k", "v"]
        .into_iter()
        .map(|name| StateDeclaration {
            key: StateKey::new(0, name),
            full_shape: Shape::new(&[4, 8]).unwrap(),
            segment_shape: Shape::new(&[4, 8]).unwrap(),
            strides: vec![8, 1],
            logical_start: 0,
            logical_count: 4,
            absolute_position: 4,
            window: 0,
            atomic_group: 1,
        })
        .collect()
}

pub fn realized_schema() -> RealizedCutSchemaId {
    RealizedCutSchemaId {
        kind: ManifestKind::Full,
        states: ["k", "v"]
            .into_iter()
            .map(|name| RealizedStateSchema {
                key: StateKey::new(0, name),
                full_shape: Shape::new(&[4, 8]).unwrap(),
                segment_shape: Shape::new(&[4, 8]).unwrap(),
                strides: vec![8, 1],
                logical_start: 0,
                logical_count: 4,
                physical_offset_bytes: 0,
                physical_span_bytes: 32,
                complete_physical_bytes: 32,
                absolute_position: 4,
                window: 0,
                chunk_spans: vec![ChunkSpan {
                    token_start: 0,
                    token_count: 4,
                    plaintext_offset: 0,
                    plaintext_bytes: 32,
                }],
            })
            .collect(),
        atomic_groups: vec![AtomicGroup {
            id: 1,
            states: vec![StateKey::new(0, "k"), StateKey::new(0, "v")],
        }],
        segment_restored_bytes: 64,
        complete_restored_bytes: 64,
    }
}

pub struct Fixture {
    pub _temp: tempfile::TempDir,
    pub store: Arc<LocalStore>,
}

pub fn fixture(tenant: &[u8]) -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let key = kvpack::load_store_key(&key_path, temp.path()).unwrap();
    let store = Arc::new(
        LocalStore::open(
            StoreConfig {
                object_root: temp.path().join("objects"),
                catalog_path: temp.path().join("catalog/catalog.sqlite"),
                operator_tenant_id: tenant.to_vec(),
                key_epoch: 1,
                minimum_readable_key_epoch: 1,
                catalog_epoch: 1,
                quota_bytes: 1024 * 1024 * 1024,
                staging_quota_bytes: 1024 * 1024 * 1024,
                endurance_bytes_per_five_minutes: 1024 * 1024 * 1024,
            },
            key,
        )
        .unwrap(),
    );
    Fixture { _temp: temp, store }
}

pub fn declaration(codec: Codec) -> ManifestDeclaration {
    ManifestDeclaration {
        semantic_model: semantic(),
        input_tokens: vec![10, 20, 30, 40],
        auxiliary_inputs: auxiliary(),
        family: family(codec),
        kind: ManifestKind::Full,
        states: states(),
    }
}

pub fn publish(
    store: Arc<LocalStore>,
    codec: Codec,
) -> (
    kvpack::PublishedArtifact,
    RepresentationFamilyId,
    RealizedCutSchemaId,
    kvpack::wire::InputCutId,
) {
    publish_with_options(store, codec, [10, 20, 30, 40], 10, true)
}

pub fn publish_with_options(
    store: Arc<LocalStore>,
    codec: Codec,
    input_tokens: [u32; 4],
    idempotency: u8,
    encrypt: bool,
) -> (
    kvpack::PublishedArtifact,
    RepresentationFamilyId,
    RealizedCutSchemaId,
    kvpack::wire::InputCutId,
) {
    let mut declaration = declaration(codec);
    declaration.input_tokens = input_tokens.to_vec();
    let family = declaration.family.clone();
    let policy = WritePolicy::exact_qualified(id(idempotency), semantic(), &family)
        .unwrap()
        .with_encryption(encrypt);
    let mut writer = ArtifactWriter::begin(Arc::clone(&store), declaration, policy).unwrap();
    {
        let mut state = writer.next_state(StateKey::new(0, "k")).unwrap();
        state.write_all(&[11; 32]).unwrap();
        state.finish().unwrap();
    }
    {
        let mut state = writer.next_state(StateKey::new(0, "v")).unwrap();
        state.write_all(&[22; 32]).unwrap();
        state.finish().unwrap();
    }
    let published = writer.commit().unwrap();
    let input = store
        .derive_input_cut(&semantic(), &family, &input_tokens, &auxiliary())
        .unwrap()
        .0;
    (published, family, realized_schema(), input)
}

#[derive(Clone)]
pub struct TestKeyProvider {
    pub provider_name: &'static str,
    pub stable_root: [u8; 32],
    pub epoch_roots: Vec<(u64, [u8; 32])>,
}

impl KeyProvider for TestKeyProvider {
    fn name(&self) -> &'static str {
        self.provider_name
    }

    fn qualification(&self) -> KeyProviderQualification {
        KeyProviderQualification::Production
    }

    fn load_tenant_keys(
        &self,
        _tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError> {
        StoreKey::from_epoch_roots(
            self.stable_root,
            self.epoch_roots
                .iter()
                .copied()
                .filter(|(epoch, _)| (window.minimum_readable..=window.active).contains(epoch)),
        )
    }
}

pub fn provider_key(
    provider: &dyn KeyProvider,
    tenant: &[u8],
    minimum_readable: u64,
    active: u64,
) -> StoreKey {
    kvpack::load_store_key_from_provider(
        provider,
        tenant,
        KeyEpochWindow {
            minimum_readable,
            active,
        },
        true,
    )
    .unwrap()
}

pub fn epoch_config(
    root: &std::path::Path,
    catalog_name: &str,
    tenant: &[u8],
    minimum_readable_key_epoch: u64,
    key_epoch: u64,
) -> StoreConfig {
    StoreConfig {
        object_root: root.join("objects"),
        catalog_path: root.join("catalog").join(catalog_name),
        operator_tenant_id: tenant.to_vec(),
        key_epoch,
        minimum_readable_key_epoch,
        catalog_epoch: 1,
        quota_bytes: 1024 * 1024 * 1024,
        staging_quota_bytes: 1024 * 1024 * 1024,
        endurance_bytes_per_five_minutes: 1024 * 1024 * 1024,
    }
}

pub fn open_expectations(
    published: kvpack::PublishedArtifact,
    family: RepresentationFamilyId,
    schema: RealizedCutSchemaId,
    input_cut: kvpack::wire::InputCutId,
) -> OpenExpectations {
    OpenExpectations {
        locator: ArtifactLocator::Manifest(published.manifest_id),
        semantic_model: semantic(),
        input_cut,
        family,
        realized_schema: schema,
        minimum_key_epoch: 1,
    }
}

#[derive(Default)]
pub struct ShadowSink {
    pub shadow: BTreeMap<StateKey, Vec<u8>>,
    pub installed: BTreeMap<StateKey, Vec<u8>>,
    pub committed: bool,
    pub aborted: bool,
}

impl VerifiedRestoreSink for ShadowSink {
    fn begin_restore(
        &mut self,
        _: [u8; 32],
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
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
        let target = self
            .shadow
            .get_mut(state)
            .ok_or(StoreError::State("missing shadow"))?;
        target[offset as usize..offset as usize + plaintext.len()].copy_from_slice(plaintext);
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        self.installed = std::mem::take(&mut self.shadow);
        self.committed = true;
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.shadow.clear();
        self.aborted = true;
    }
}
