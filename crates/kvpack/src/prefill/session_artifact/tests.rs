use std::fs::OpenOptions;
use std::io::{Read as _, Seek, SeekFrom, Write};

use kvpack_core::{FamilyState, RepresentationFamilyId, RepresentationMode, SemanticModelId};

use super::*;
use crate::{
    bind_weights_scalar_math_v2, create_store_key_random, derive_portable_prefill_descriptor_v2,
    load_store_key, LocalStore, PortablePrefillDescriptorInputV2, PreRopeKernelPinV1,
    RestoreCancellation, RestoreStatePlan, StoreConfig, VerifiedRestoreSink, WeightsScalarMathV1,
    PORTABLE_PREFILL_ABI_V2,
};

fn id(value: u8) -> Id32 {
    [value; 32]
}

struct Fixture {
    _temp: tempfile::TempDir,
    config: StoreConfig,
    key_path: std::path::PathBuf,
    store: Arc<LocalStore>,
}

impl Fixture {
    fn new(tenant: &[u8]) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        create_store_key_random(&key_path, temp.path()).unwrap();
        let config = StoreConfig {
            object_root: temp.path().join("objects"),
            catalog_path: temp.path().join("catalog/catalog.sqlite"),
            operator_tenant_id: tenant.to_vec(),
            key_epoch: 1,
            minimum_readable_key_epoch: 1,
            catalog_epoch: 1,
            quota_bytes: 2 * 1024 * 1024 * 1024,
            staging_quota_bytes: 2 * 1024 * 1024 * 1024,
            endurance_bytes_per_five_minutes: 2 * 1024 * 1024 * 1024,
        };
        let key = load_store_key(&key_path, temp.path()).unwrap();
        let store = Arc::new(LocalStore::open(config.clone(), key).unwrap());
        Self {
            _temp: temp,
            config,
            key_path,
            store,
        }
    }
}

fn reopen_store(
    config: StoreConfig,
    key_path: &std::path::Path,
    allowed_root: &std::path::Path,
) -> Arc<LocalStore> {
    let key = load_store_key(key_path, allowed_root).unwrap();
    Arc::new(LocalStore::open(config, key).unwrap())
}

fn muse_descriptor(cached_token_count: u32) -> PortablePrefillDescriptorV1 {
    let input = PortablePrefillDescriptorInputV2 {
        model_sha256: id(0x82),
        adapter_sha256: [0; 32],
        tokenizer_sha256: id(0x11),
        chat_template_sha256: id(0x22),
        context_policy_sha256: id(0x33),
        model_revision: "muse-glimmer-30b@Q4_K_XL".into(),
        tokenizer_revision: "muse-glimmer-30b-tokenizer@1".into(),
        producer_engine_abi: "llama.cpp-pr26841".into(),
        consumer_engine_abi: "ferrite-metal-v1".into(),
        portable_abi: PORTABLE_PREFILL_ABI_V2.into(),
        compute_precision: "float16".into(),
        kv_precision: "float16".into(),
        weight_precision: "q4_k_m".into(),
        cached_token_count,
        max_context_tokens: 131_072,
        layout_name: MUSE_LAYOUT_NAME.into(),
        transform: None,
        prerope_kernel_pin: None::<PreRopeKernelPinV1>,
    };
    let descriptor = derive_portable_prefill_descriptor_v2(&input).unwrap();
    bind_weights_scalar_math_v2(
        descriptor,
        &WeightsScalarMathV1 {
            qk_scale_factor_bits: 3.87f64.to_bits(),
            output_multiplier_bits: 0.196_116_135_138_184_04_f64.to_bits(),
            final_logit_softcapping_bits: 20.0f64.to_bits(),
            post_norm_eps_bits: 1e-8_f64.to_bits(),
        },
    )
    .unwrap()
}

fn prompt(cached: u32) -> Vec<u32> {
    (0..cached).map(|token| token + 100).collect()
}

fn policy(descriptor: &PortablePrefillDescriptorV1, value: u8) -> WritePolicy {
    WritePolicy::exact_qualified(id(value), descriptor.semantic_model, &descriptor.family).unwrap()
}

fn write_complete(
    store: Arc<LocalStore>,
    descriptor: &PortablePrefillDescriptorV1,
    prompt: &[u32],
    policy_id: u8,
) -> MuseSessionArtifactReceipt {
    let mut writer = MuseSessionWriter::begin(
        store,
        descriptor.clone(),
        prompt.to_vec(),
        policy(descriptor, policy_id),
    )
    .unwrap();
    for (ordinal, family) in descriptor.family.states.iter().enumerate() {
        let tokens = if family.token_axis_rule == TokenAxisRule::TailWindow {
            prompt.len().min(2_048)
        } else {
            prompt.len()
        };
        let bytes = tokens
            * family.elements_per_token as usize
            * family.dtype.width_bytes().unwrap() as usize;
        let mut plane = writer.next_plane(family.key.clone()).unwrap();
        plane.write_all(&vec![ordinal as u8; bytes]).unwrap();
        plane.finish().unwrap();
    }
    writer.commit().unwrap()
}

#[derive(Default)]
struct MemorySink {
    planes: BTreeMap<StateKey, Vec<u8>>,
    writes: Vec<(StateKey, u64, usize)>,
}

impl VerifiedRestoreSink for MemorySink {
    fn begin_restore(&mut self, _: Id32, states: &[RestoreStatePlan]) -> Result<(), StoreError> {
        for state in states {
            self.planes.insert(
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
        let plane = self.planes.get_mut(state).ok_or(StoreError::State(
            "restore targeted an undeclared Muse plane",
        ))?;
        let offset = usize::try_from(offset)
            .map_err(|_| StoreError::State("Muse plane offset exceeds usize"))?;
        let end = offset
            .checked_add(plaintext.len())
            .ok_or(StoreError::State("Muse plane write overflows usize"))?;
        let target = plane
            .get_mut(offset..end)
            .ok_or(StoreError::State("Muse plane write exceeds compact bounds"))?;
        target.copy_from_slice(plaintext);
        self.writes
            .push((state.clone(), offset as u64, plaintext.len()));
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.planes.clear();
    }
}

#[test]
fn positive_start_tail_window_restores_into_compact_bounds() {
    let fixture = Fixture::new(b"tail-window-positive-start");
    let state_key = StateKey::new(0, "attn.k");
    let semantic_model = SemanticModelId {
        weights_config: id(1),
        adapters: id(2),
        tokenizer_template: id(3),
        position_semantics: id(4),
        qualified_math: id(5),
    };
    let family = RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Portable,
        page_size_tokens: 1,
        topology: id(7),
        shard_map: id(8),
        states: vec![FamilyState {
            key: state_key.clone(),
            cache_kind: CacheKind::OrdinaryKv,
            dtype: DType::U8,
            codec: Codec::Raw,
            codec_version: 1,
            layout: Layout::Contiguous,
            token_axis_rule: TokenAxisRule::TailWindow,
            token_axis: 0,
            elements_per_token: 4,
            dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(4)],
            dependencies: Vec::new(),
        }],
    };
    let declaration = ManifestDeclaration {
        semantic_model,
        input_tokens: vec![10, 11, 12, 13],
        auxiliary_inputs: Vec::new(),
        family: family.clone(),
        kind: ManifestKind::Full,
        states: vec![StateDeclaration {
            key: state_key.clone(),
            full_shape: Shape::new(&[2, 4]).unwrap(),
            segment_shape: Shape::new(&[2, 4]).unwrap(),
            strides: vec![4, 1],
            logical_start: 2,
            logical_count: 2,
            absolute_position: 4,
            window: 0,
            atomic_group: 1,
        }],
    };
    let payload = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
    let mut writer = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration,
        WritePolicy::exact_qualified(id(0x44), semantic_model, &family).unwrap(),
    )
    .unwrap();
    let mut plane = writer.next_state(state_key.clone()).unwrap();
    plane.write_all(&payload).unwrap();
    plane.finish().unwrap();
    let published = writer.commit().unwrap();
    let plan = AuthenticatedRestorePlan::build_exact_manifest(
        Arc::clone(&fixture.store),
        published.manifest_id,
        1,
        RestoreLimits::default(),
        &ValidationContext::default(),
    )
    .unwrap();
    assert_eq!(plan.states()[0].declaration.logical_start, 2);
    assert_eq!(plan.states()[0].plaintext_bytes, payload.len() as u64);

    let mut sequential = MemorySink::default();
    let installed = plan
        .restore_sequential(&mut sequential, &RestoreCancellation::default())
        .unwrap();
    assert_eq!(sequential.writes, vec![(state_key.clone(), 0, 8)]);
    assert_eq!(sequential.planes[&state_key], payload);
    installed.engine_free().unwrap();

    let mut parallel = MemorySink::default();
    let installed = plan
        .restore_parallel(&mut parallel, &RestoreCancellation::default(), 2)
        .unwrap();
    assert_eq!(parallel.writes, vec![(state_key.clone(), 0, 8)]);
    assert_eq!(parallel.planes[&state_key], payload);
    installed.engine_free().unwrap();
}

#[test]
fn mixed_ranges_use_full_nope_and_trailing_swa_planes() {
    let descriptor = muse_descriptor(4_096);
    let states = muse_state_declarations(&descriptor, muse_layout().unwrap(), 4_096).unwrap();
    assert_eq!(states.len(), 104);
    for (family, state) in descriptor.family.states.iter().zip(&states) {
        if family.token_axis_rule == TokenAxisRule::TailWindow {
            assert_eq!((state.logical_start, state.logical_count), (2_048, 2_048));
            assert_eq!(state.full_shape.dims()[0], 2_048);
        } else {
            assert_eq!((state.logical_start, state.logical_count), (0, 4_096));
            assert_eq!(state.full_shape.dims()[0], 4_096);
        }
        assert_eq!(state.absolute_position, 4_096);
        assert_eq!(state.window, 0);
    }
    let fixture = Fixture::new(b"muse-session-mixed-ranges");
    let writer = MuseSessionWriter::begin(
        Arc::clone(&fixture.store),
        descriptor.clone(),
        prompt(4_096),
        policy(&descriptor, 0x40),
    )
    .unwrap();
    drop(writer);
    assert_eq!(fixture.store.stat().unwrap().manifests, 0);
}

#[test]
fn authenticated_session_reopens_after_store_restart() {
    let fixture = Fixture::new(b"muse-session-restart");
    let descriptor = muse_descriptor(2);
    let tokens = prompt(2);
    let receipt = write_complete(Arc::clone(&fixture.store), &descriptor, &tokens, 0x41);
    assert_eq!(receipt.tail_coverage.len(), 78);
    assert_eq!(fixture.store.stat().unwrap().manifests, 1);

    let config = fixture.config.clone();
    let key_path = fixture.key_path.clone();
    let allowed_root = fixture._temp.path().to_path_buf();
    drop(fixture.store);
    let reopened = reopen_store(config, &key_path, &allowed_root);
    let artifact = MuseSessionArtifact::open(
        Arc::clone(&reopened),
        receipt.manifest_id,
        &descriptor,
        &tokens,
        1,
        RestoreLimits::default(),
    )
    .unwrap();
    assert_eq!(artifact.manifest_id(), receipt.manifest_id);
    assert_eq!(artifact.input_cut(), receipt.input_cut);
    assert_eq!(
        artifact.prompt_token_ids_sha256(),
        receipt.prompt_token_ids_sha256
    );
    assert_eq!(artifact.tail_coverage(), &receipt.tail_coverage);
    assert_eq!(artifact.restore_plan().states().len(), 104);
    let mut sink = MemorySink::default();
    let installed = artifact
        .restore_plan()
        .restore_sequential(&mut sink, &RestoreCancellation::default())
        .unwrap();
    for (ordinal, family) in descriptor.family.states.iter().enumerate() {
        let plane = &sink.planes[&family.key];
        assert!(!plane.is_empty());
        assert!(plane.iter().all(|byte| *byte == ordinal as u8));
    }
    installed.engine_free().unwrap();

    let mut wrong_tokens = tokens.clone();
    wrong_tokens[0] ^= 1;
    let error = MuseSessionArtifact::open(
        reopened,
        receipt.manifest_id,
        &descriptor,
        &wrong_tokens,
        1,
        RestoreLimits::default(),
    )
    .err()
    .expect("wrong prompt identity must fail");
    assert!(matches!(error, StoreError::Authentication(_)));
}

#[test]
fn manifest_byte_tamper_fails_authenticated_open() {
    let fixture = Fixture::new(b"muse-session-tamper");
    let descriptor = muse_descriptor(2);
    let tokens = prompt(2);
    let receipt = write_complete(Arc::clone(&fixture.store), &descriptor, &tokens, 0x42);
    let path = fixture.store.manifest_path(&receipt.manifest_id);
    let config = fixture.config.clone();
    let key_path = fixture.key_path.clone();
    let allowed_root = fixture._temp.path().to_path_buf();
    drop(fixture.store);

    let length = std::fs::metadata(&path).unwrap().len();
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(length / 2)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xA5;
    file.seek(SeekFrom::Start(length / 2)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();

    let reopened = reopen_store(config, &key_path, &allowed_root);
    let error = MuseSessionArtifact::open(
        reopened,
        receipt.manifest_id,
        &descriptor,
        &tokens,
        1,
        RestoreLimits::default(),
    )
    .err()
    .expect("tampered session must fail");
    assert!(matches!(
        error,
        StoreError::Authentication(_) | StoreError::Pack(_)
    ));
}

#[test]
fn dropped_partial_session_never_publishes_a_manifest() {
    let fixture = Fixture::new(b"muse-session-atomicity");
    let descriptor = muse_descriptor(2);
    let tokens = prompt(2);
    let mut writer = MuseSessionWriter::begin(
        Arc::clone(&fixture.store),
        descriptor.clone(),
        tokens.clone(),
        policy(&descriptor, 0x43),
    )
    .unwrap();
    let family = &descriptor.family.states[0];
    let bytes = tokens.len()
        * family.elements_per_token as usize
        * family.dtype.width_bytes().unwrap() as usize;
    let mut plane = writer.next_plane(family.key.clone()).unwrap();
    plane.write_all(&vec![0xCC; bytes]).unwrap();
    plane.finish().unwrap();
    drop(writer);
    assert_eq!(fixture.store.stat().unwrap().manifests, 0);

    let config = fixture.config.clone();
    let key_path = fixture.key_path.clone();
    let allowed_root = fixture._temp.path().to_path_buf();
    drop(fixture.store);
    let reopened = reopen_store(config, &key_path, &allowed_root);
    assert_eq!(reopened.stat().unwrap().manifests, 0);
}
