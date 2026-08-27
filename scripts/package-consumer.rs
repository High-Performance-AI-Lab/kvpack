use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use kvpack::wire::{
    CacheKind, Codec, DType, FamilyState, Id32, Layout, RepresentationFamilyId, RepresentationMode,
    SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
};
use kvpack::{
    AuthenticatedRestorePlan, ExportCutPolicy, ExportDeclaration, ExportSession,
    ExportStateDeclaration, LocalStore, RestoreCancellation, RestoreLimits, RestoreRequest,
    RestoreStatePlan, RestoreTier, StoreConfig, StoreError, VerifiedRestoreSink, WritePolicy,
};

fn id(value: u8) -> Id32 {
    [value; 32]
}

fn semantic_model() -> SemanticModelId {
    SemanticModelId {
        weights_config: id(1),
        adapters: id(2),
        tokenizer_template: id(3),
        position_semantics: id(4),
        qualified_math: id(5),
    }
}

fn family() -> RepresentationFamilyId {
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

#[derive(Default)]
struct DetachedSink {
    shadow: BTreeMap<StateKey, Vec<u8>>,
    installed: BTreeMap<StateKey, Vec<u8>>,
}

impl VerifiedRestoreSink for DetachedSink {
    fn begin_restore(
        &mut self,
        _artifact: Id32,
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
        self.shadow.clear();
        for state in states {
            let length = usize::try_from(state.plaintext_bytes)
                .map_err(|_| StoreError::State("package smoke state exceeds usize"))?;
            self.shadow
                .insert(state.declaration.key.clone(), vec![0; length]);
        }
        Ok(())
    }

    fn write_verified_chunk(
        &mut self,
        state: &StateKey,
        logical_offset: u64,
        plaintext: &[u8],
    ) -> Result<(), StoreError> {
        let target = self
            .shadow
            .get_mut(state)
            .ok_or(StoreError::State("package smoke received an unknown state"))?;
        let start = usize::try_from(logical_offset)
            .map_err(|_| StoreError::State("package smoke offset exceeds usize"))?;
        let end = start
            .checked_add(plaintext.len())
            .ok_or(StoreError::State("package smoke write range overflow"))?;
        let output = target.get_mut(start..end).ok_or(StoreError::State(
            "package smoke write exceeds shadow state",
        ))?;
        output.copy_from_slice(plaintext);
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        self.installed = std::mem::take(&mut self.shadow);
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.shadow.clear();
    }

    fn reset_restore(&mut self) {
        self.shadow.clear();
        self.installed.clear();
    }
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = std::env::temp_dir().join(format!(
        "kvpack-package-smoke-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    let _cleanup = RemoveOnDrop(root.clone());

    let key_path = root.join("keys/root.key");
    kvpack::create_store_key_random(&key_path, &root)?;
    let key = kvpack::load_store_key(&key_path, &root)?;
    let store = Arc::new(LocalStore::open(
        StoreConfig {
            object_root: root.join("objects"),
            catalog_path: root.join("internal/catalog.sqlite"),
            operator_tenant_id: b"package-smoke-tenant".to_vec(),
            key_epoch: 1,
            minimum_readable_key_epoch: 1,
            catalog_epoch: 1,
            quota_bytes: 64 * 1024 * 1024,
            staging_quota_bytes: 64 * 1024 * 1024,
            endurance_bytes_per_five_minutes: 64 * 1024 * 1024,
        },
        key,
    )?);

    let semantic_model = semantic_model();
    let family = family();
    let tokens = vec![10, 11, 12, 13];
    let mut export = ExportSession::begin(
        Arc::clone(&store),
        ExportDeclaration {
            semantic_model,
            input_tokens: tokens.clone(),
            auxiliary_inputs: vec![],
            family: family.clone(),
            states: vec![ExportStateDeclaration {
                key: StateKey::new(0, "k"),
                strides: vec![4, 1],
                atomic_group: 1,
            }],
        },
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(40), semantic_model, &family)?,
    )?;
    let mut source = std::io::Cursor::new((0u8..16).collect::<Vec<_>>());
    export
        .next_state(StateKey::new(0, "k"))?
        .write_source(&mut source)?;
    let published = export.commit()?;
    assert_eq!(published.exact_final.input_cut.token_count, 4);

    let mut publication = store.authenticated_publication_source(
        &published.exact_final.manifest_id,
        &kvpack::wire::ValidationContext::default(),
    )?;
    assert_eq!(publication.chunk_count(), 1);
    let (publication_chunk, publication_object) = publication.read_chunk_object(0)?;
    assert_eq!(
        publication_chunk.object_bytes() as usize,
        publication_object.len()
    );
    assert!(publication.expected_bytes() >= publication_object.len() as u64);
    publication.release()?;

    let candidates = store.restore_candidates(RestoreRequest {
        semantic_model,
        family,
        input_tokens: tokens,
        auxiliary_inputs: vec![],
        minimum_key_epoch: 1,
        maximum_candidates: 8,
    })?;
    let candidate = candidates
        .iter()
        .find(|candidate| candidate.tier() == RestoreTier::Local)
        .ok_or("package smoke did not discover its exact local cut")?;
    let plan =
        AuthenticatedRestorePlan::build(Arc::clone(&store), candidate, RestoreLimits::default())?;
    let transfer = plan.prepare_scatter_transfer(id(41))?;
    assert!(!transfer.batches().is_empty());
    drop(transfer);

    let mut sink = DetachedSink::default();
    let installed = plan.restore_sequential(&mut sink, &RestoreCancellation::default())?;
    assert_eq!(
        sink.installed.get(&StateKey::new(0, "k")),
        Some(&(0u8..16).collect::<Vec<_>>())
    );
    installed.engine_free()?;

    let _ = std::mem::size_of::<kvpack_core::PackHeader>();
    assert_eq!(kvpack_handoff::LIVE_HANDOFF_SCHEMA_V1, 1);
    println!("packaged Rust safe-path smoke passed");
    Ok(())
}
