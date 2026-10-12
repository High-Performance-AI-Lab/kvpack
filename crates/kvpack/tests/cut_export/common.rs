pub(crate) use std::collections::BTreeMap;
pub(crate) use std::io::{Cursor, Read};
pub(crate) use std::sync::{Arc, Mutex};

pub(crate) use kvpack::wire::{
    AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, Layout, ManifestKind,
    RepresentationFamilyId, RepresentationMode, SemanticModelId, StateKey, StaticDimension,
    TokenAxisRule, ValidationContext,
};
pub(crate) use kvpack::{
    AuditBatch, AuditEventKind, AuditExporter, AuditObjectKind, AuthenticatedRestorePlan,
    ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateDeclaration, LocalStore,
    PublishedCut, RestoreCancellation, RestoreLimits, RestoreStatePlan, RetentionInputs,
    StoreConfig, StoreError, VerifiedRestoreSink, WritePolicy,
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

pub fn family(
    codec: Codec,
    state_names: &[&str],
    elements_per_token: u64,
) -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Native,
        page_size_tokens: 256,
        topology: id(7),
        shard_map: id(8),
        states: state_names
            .iter()
            .map(|name| FamilyState {
                key: StateKey::new(0, *name),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token,
                dimensions: vec![
                    StaticDimension::Token,
                    StaticDimension::Fixed(elements_per_token),
                ],
                dependencies: vec![],
            })
            .collect(),
    }
}

pub fn declaration(family: RepresentationFamilyId, token_count: usize) -> ExportDeclaration {
    let width = family.states[0].elements_per_token;
    ExportDeclaration {
        semantic_model: semantic(),
        input_tokens: (0..token_count as u32).collect(),
        auxiliary_inputs: auxiliary(),
        states: family
            .states
            .iter()
            .map(|state| ExportStateDeclaration {
                key: state.key.clone(),
                strides: vec![width, 1],
                atomic_group: 1,
            })
            .collect(),
        family,
    }
}

pub struct Fixture {
    pub _temp: tempfile::TempDir,
    pub store: Arc<LocalStore>,
}

pub fn fixture(name: &[u8]) -> Fixture {
    fixture_with_limits(name, 512 * 1024 * 1024, 512 * 1024 * 1024)
}

pub fn fixture_with_limits(name: &[u8], quota_bytes: u64, endurance_bytes: u64) -> Fixture {
    fixture_with_all_limits(name, quota_bytes, quota_bytes, endurance_bytes)
}

pub fn fixture_with_all_limits(
    name: &[u8],
    quota_bytes: u64,
    staging_quota_bytes: u64,
    endurance_bytes: u64,
) -> Fixture {
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
                quota_bytes,
                staging_quota_bytes,
                endurance_bytes_per_five_minutes: endurance_bytes,
            },
            key,
        )
        .unwrap(),
    );
    Fixture { _temp: temp, store }
}

pub struct CountingReader {
    inner: Cursor<Vec<u8>>,
    pub returned: usize,
    maximum_read: usize,
}

impl CountingReader {
    pub fn new(bytes: Vec<u8>, maximum_read: usize) -> Self {
        Self {
            inner: Cursor::new(bytes),
            returned: 0,
            maximum_read,
        }
    }
}

impl Read for CountingReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let limit = output.len().min(self.maximum_read);
        let read = self.inner.read(&mut output[..limit])?;
        self.returned += read;
        Ok(read)
    }
}

pub struct FailingReader {
    inner: Cursor<Vec<u8>>,
    fail_after: usize,
    returned: usize,
}

impl FailingReader {
    pub fn new(bytes: Vec<u8>, fail_after: usize) -> Self {
        Self {
            inner: Cursor::new(bytes),
            fail_after,
            returned: 0,
        }
    }
}

impl Read for FailingReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.returned >= self.fail_after {
            return Err(std::io::Error::other("injected source read failure"));
        }
        let limit = output.len().min(self.fail_after - self.returned);
        let read = self.inner.read(&mut output[..limit])?;
        self.returned += read;
        Ok(read)
    }
}

#[derive(Default)]
pub struct ShadowSink {
    pub shadow: BTreeMap<StateKey, Vec<u8>>,
    pub installed: BTreeMap<StateKey, Vec<u8>>,
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
        let output = self
            .shadow
            .get_mut(state)
            .ok_or(StoreError::State("missing shadow state"))?;
        let offset = usize::try_from(offset).map_err(|_| StoreError::State("offset overflow"))?;
        output[offset..offset + plaintext.len()].copy_from_slice(plaintext);
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        self.installed = std::mem::take(&mut self.shadow);
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.shadow.clear();
        self.aborted = true;
    }
}

pub fn restore_cut(
    store: Arc<LocalStore>,
    family: &RepresentationFamilyId,
    cut: &PublishedCut,
) -> ShadowSink {
    let plan = AuthenticatedRestorePlan::build_exact_manifest(
        store,
        cut.manifest_id,
        1,
        RestoreLimits::default(),
        &ValidationContext::default(),
    )
    .unwrap();
    assert_eq!(plan.family(), family);
    assert_eq!(plan.matched_cut(), cut.input_cut);
    assert_eq!(plan.realized_schema(), &cut.realized_schema);
    let mut sink = ShadowSink::default();
    let installed = plan
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    installed.engine_free().unwrap();
    sink
}

pub fn export_two_states(
    store: Arc<LocalStore>,
    family: RepresentationFamilyId,
    token_count: usize,
    idempotency: [u8; 32],
    k: u8,
    v: u8,
) -> kvpack::PublishedCutSet {
    let declaration = declaration(family.clone(), token_count);
    let mut session = ExportSession::begin(
        store,
        declaration,
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(idempotency, semantic(), &family).unwrap(),
    )
    .unwrap();
    let bytes = token_count * family.states[0].elements_per_token as usize;
    for (name, value) in [("k", k), ("v", v)] {
        let mut source = CountingReader::new(vec![value; bytes], 37);
        session
            .next_state(StateKey::new(0, name))
            .unwrap()
            .write_source(&mut source)
            .unwrap();
        assert_eq!(source.returned, bytes);
    }
    session.commit().unwrap()
}

#[derive(Default)]
pub struct CapturedAudit(pub Mutex<Vec<CapturedAuditRecord>>);

pub type CapturedAuditRecord = (u64, AuditEventKind, AuditObjectKind, [u8; 32]);

impl AuditExporter for CapturedAudit {
    fn export(&self, batch: &AuditBatch) -> Result<(), StoreError> {
        self.0
            .lock()
            .unwrap()
            .extend(batch.records().iter().map(|record| {
                (
                    record.sequence(),
                    record.event(),
                    record.object(),
                    *record.object_id(),
                )
            }));
        Ok(())
    }
}
