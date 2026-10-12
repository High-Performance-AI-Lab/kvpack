//! M6/M7 end-to-end tests: authenticated statistics sidecars through the
//! persist and restore paths, fidelity-rung demotion semantics, and
//! guided-recompute restore planning for tombstoned chunks.

use std::sync::Arc;

use kvpack::wire::half::f32_to_f16;
use kvpack::wire::{
    AuxiliaryInputId, CacheKind, ChannelRange, ChunkSpan, Codec, DType, FamilyState, Id32, Layout,
    ManifestDeclaration, ManifestKind, RepresentationFamilyId, RepresentationMode, SemanticModelId,
    Shape, SinkScore, StateDeclaration, StateKey, StaticDimension, StatsSidecar, TokenAxisRule,
};
use kvpack::{
    ArtifactWriter, AuthenticatedRestorePlan, LocalStore, PublishedArtifact, RestoreCancellation,
    RestoreLimits, RestoreRequest, RestoreStatePlan, RestoreTier, StoreConfig, StoreError,
    UtilizationPolicy, VerifiedRestoreSink, WritePolicy,
};

fn id(value: u8) -> Id32 {
    [value; 32]
}

fn f16(value: f32) -> u16 {
    f32_to_f16(value)
}

fn semantic() -> SemanticModelId {
    SemanticModelId {
        weights_config: id(1),
        adapters: id(2),
        tokenizer_template: id(3),
        position_semantics: id(4),
        qualified_math: id(5),
    }
}

fn auxiliary() -> Vec<AuxiliaryInputId> {
    vec![AuxiliaryInputId {
        type_id: id(30),
        value_id: id(31),
    }]
}

fn f16_family() -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Native,
        page_size_tokens: 256,
        topology: id(7),
        shard_map: id(8),
        states: vec![FamilyState {
            key: StateKey::new(0, "k"),
            cache_kind: CacheKind::OrdinaryKv,
            dtype: DType::F16,
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

fn u8_family() -> RepresentationFamilyId {
    let mut family = f16_family();
    family.states[0].dtype = DType::U8;
    family
}

/// Hand-computed fixture: 8 tokens × 4 fp16 channels.
fn fixture_tokens() -> [[f32; 4]; 8] {
    [
        [1.0, 0.5, -1.0, 2.0],
        [0.0, 1.5, -0.5, 1.0],
        [-2.0, 0.25, 3.0, 0.0],
        [4.0, -0.75, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [-1.0, -1.0, 2.0, -2.0],
        [3.0, 0.0, -0.25, 1.5],
        [0.125, 2.0, 1.0, -0.5],
    ]
}

fn fixture_plaintext() -> Vec<u8> {
    let mut bytes = Vec::new();
    for token in fixture_tokens() {
        for value in token {
            bytes.extend_from_slice(&f16(value).to_le_bytes());
        }
    }
    bytes
}

/// The hand-computed sidecar: per-channel min/max from the columns of
/// `fixture_tokens`, per-token L2 norms, top-3 sink tokens by norm.
fn expected_sidecar() -> StatsSidecar {
    let tokens = fixture_tokens();
    let channel = |c: usize, pick: fn(f32, f32) -> f32| {
        tokens
            .iter()
            .map(|token| token[c])
            .reduce(pick)
            .map(f16)
            .unwrap()
    };
    let norms: Vec<u16> = tokens
        .iter()
        .map(|token| {
            let sum: f32 = token.iter().map(|value| value * value).sum();
            f16(sum.sqrt())
        })
        .collect();
    // Norms: t3 = sqrt(17.8125) ≈ 4.22, t2 = sqrt(13.0625) ≈ 3.61,
    // t6 = sqrt(11.8125) ≈ 3.44 are the top three.
    StatsSidecar {
        channel_ranges: vec![
            ChannelRange {
                min_bits: channel(0, f32::min),
                max_bits: channel(0, f32::max),
            },
            ChannelRange {
                min_bits: channel(1, f32::min),
                max_bits: channel(1, f32::max),
            },
            ChannelRange {
                min_bits: channel(2, f32::min),
                max_bits: channel(2, f32::max),
            },
            ChannelRange {
                min_bits: channel(3, f32::min),
                max_bits: channel(3, f32::max),
            },
        ],
        key_l2_norms: norms.clone(),
        sink_scores: vec![
            SinkScore {
                token_index: 3,
                score_bits: norms[3],
            },
            SinkScore {
                token_index: 2,
                score_bits: norms[2],
            },
            SinkScore {
                token_index: 6,
                score_bits: norms[6],
            },
        ],
    }
}

struct Fixture {
    _temp: tempfile::TempDir,
    store: Arc<LocalStore>,
}

fn fixture(name: &[u8]) -> Fixture {
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
    Fixture { _temp: temp, store }
}

fn declaration(tokens: usize, family: &RepresentationFamilyId) -> ManifestDeclaration {
    ManifestDeclaration {
        semantic_model: semantic(),
        input_tokens: (0..tokens as u32).collect(),
        auxiliary_inputs: auxiliary(),
        family: family.clone(),
        kind: ManifestKind::Full,
        states: vec![StateDeclaration {
            key: StateKey::new(0, "k"),
            full_shape: Shape::new(&[tokens as u64, 4]).unwrap(),
            segment_shape: Shape::new(&[tokens as u64, 4]).unwrap(),
            strides: vec![4, 1],
            logical_start: 0,
            logical_count: tokens as u64,
            absolute_position: tokens as u64,
            window: 0,
            atomic_group: 1,
        }],
    }
}

fn publish_f16(store: &Arc<LocalStore>, idempotency: u8, sidecars: bool) -> PublishedArtifact {
    let family = f16_family();
    let policy = WritePolicy::exact_qualified(id(idempotency), semantic(), &family).unwrap();
    let policy = if sidecars {
        policy.with_stats_sidecars(3).unwrap()
    } else {
        policy
    };
    let mut writer =
        ArtifactWriter::begin(Arc::clone(store), declaration(8, &family), policy).unwrap();
    let mut state = writer.next_state(StateKey::new(0, "k")).unwrap();
    state.write_all(&fixture_plaintext()).unwrap();
    state.finish().unwrap();
    writer.commit().unwrap()
}

fn only_chunk_object_key(store: &LocalStore) -> Id32 {
    let connection = store.lock_catalog().unwrap();
    let raw: Vec<u8> = connection
        .query_row("SELECT object_key FROM chunks", [], |row| row.get(0))
        .unwrap();
    raw.try_into().unwrap()
}

fn only_chunk_object_bytes(store: &LocalStore) -> u64 {
    let connection = store.lock_catalog().unwrap();
    connection
        .query_row("SELECT object_bytes FROM chunks", [], |row| row.get(0))
        .unwrap()
}

fn local_restore_plan(
    store: &Arc<LocalStore>,
    tokens: usize,
) -> (AuthenticatedRestorePlan, kvpack::RestoreCandidate) {
    let candidates = store
        .restore_candidates(RestoreRequest {
            semantic_model: semantic(),
            family: f16_family(),
            input_tokens: (0..tokens as u32).collect(),
            auxiliary_inputs: auxiliary(),
            minimum_key_epoch: 1,
            maximum_candidates: 2,
        })
        .unwrap();
    let candidate = candidates
        .iter()
        .find(|candidate| candidate.tier() == RestoreTier::Local)
        .expect("a local candidate exists")
        .clone();
    let plan =
        AuthenticatedRestorePlan::build(Arc::clone(store), &candidate, RestoreLimits::default())
            .unwrap();
    (plan, candidate)
}

struct NullSink;

impl VerifiedRestoreSink for NullSink {
    fn begin_restore(
        &mut self,
        _artifact: Id32,
        _states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
        Ok(())
    }

    fn write_verified_chunk(
        &mut self,
        _state: &StateKey,
        _logical_offset: u64,
        _plaintext: &[u8],
    ) -> Result<(), StoreError> {
        panic!("a guided-recompute plan must never write bytes")
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        Ok(())
    }

    fn abort_restore(&mut self) {}
}

#[test]
fn sidecar_derived_at_persist_matches_hand_computed_statistics_at_restore() {
    let fixture = fixture(b"m7-sidecar-e2e");
    publish_f16(&fixture.store, 40, true);
    let (plan, candidate) = local_restore_plan(&fixture.store, 8);
    assert_eq!(candidate.tombstoned_chunks(), 0);
    assert!(!plan.requires_guided_recompute());
    let sidecars = plan.chunk_stats_sidecars().unwrap();
    assert_eq!(sidecars.len(), 1);
    let expected = expected_sidecar();
    assert_eq!(sidecars[0], Some(expected.clone()));
    // Hand-checked anchor values, not just self-consistency: channel 0 spans
    // [-2, 4], channel 2 spans [-1, 3], and the top sink is token 3.
    let sidecar = sidecars[0].as_ref().unwrap();
    assert_eq!(sidecar.channel_ranges[0].min(), -2.0);
    assert_eq!(sidecar.channel_ranges[0].max(), 4.0);
    assert_eq!(sidecar.channel_ranges[2].min(), -1.0);
    assert_eq!(sidecar.channel_ranges[2].max(), 3.0);
    assert_eq!(sidecar.sink_scores.len(), 3);
    assert_eq!(sidecar.sink_scores[0].token_index, 3);
    assert_eq!(expected.key_l2_norms.len(), 8);
}

#[test]
fn manifests_without_sidecars_are_unaffected() {
    let fixture = fixture(b"m7-no-sidecar-e2e");
    publish_f16(&fixture.store, 41, false);
    let (plan, _) = local_restore_plan(&fixture.store, 8);
    let sidecars = plan.chunk_stats_sidecars().unwrap();
    assert_eq!(sidecars, vec![None]);
}

#[test]
fn sidecar_policy_rejects_non_fp16_states_closed() {
    let fixture = fixture(b"m7-bad-dtype");
    let family = u8_family();
    let policy = WritePolicy::exact_qualified(id(42), semantic(), &family)
        .unwrap()
        .with_stats_sidecars(3)
        .unwrap();
    let mut writer =
        ArtifactWriter::begin(Arc::clone(&fixture.store), declaration(8, &family), policy).unwrap();
    let mut state = writer.next_state(StateKey::new(0, "k")).unwrap();
    state.write_all(&[7u8; 8 * 4]).unwrap();
    assert!(state.finish().is_err());
}

#[test]
fn fidelity_rung_defaults_to_zero_and_default_eviction_is_unchanged() {
    let fixture = fixture(b"m6-default-rung");
    publish_f16(&fixture.store, 43, false);
    let object_key = only_chunk_object_key(&fixture.store);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(0)
    );
    let durable_before = fixture.store.stat().unwrap().durable_bytes;
    // Default watermark eviction deletes coldest objects outright, exactly
    // like before the fidelity ladder existed.
    let report = fixture
        .store
        .maintain_capacity(1, UtilizationPolicy::default(), 16)
        .unwrap();
    assert!(report.chunks_evicted >= 1);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        None
    );
    assert_eq!(fixture.store.stat().unwrap().durable_bytes, 0);
    assert!(durable_before > 0);
}

#[test]
fn rung_transitions_proceed_one_rung_at_a_time() {
    let fixture = fixture(b"m6-rung-order");
    publish_f16(&fixture.store, 44, false);
    let object_key = only_chunk_object_key(&fixture.store);
    let durable_before = fixture.store.stat().unwrap().durable_bytes;
    let object_bytes = only_chunk_object_bytes(&fixture.store);
    let chunk_path = fixture.store.chunk_path(&object_key);
    assert!(chunk_path.exists());
    assert!(object_bytes > 0);

    // Rung 0 -> 1: catalog annotation only, bytes untouched.
    let report = fixture.store.demote_fidelity_one_rung(8).unwrap();
    assert_eq!(report.demoted, 1);
    assert_eq!(report.quantized, 1);
    assert_eq!(report.tombstoned, 0);
    assert_eq!(report.freed_bytes, 0);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(1)
    );
    assert!(chunk_path.exists());
    assert_eq!(fixture.store.stat().unwrap().durable_bytes, durable_before);

    // Rung 1 -> 2: bytes dropped, chained key + catalog row retained.
    let report = fixture.store.demote_fidelity_one_rung(8).unwrap();
    assert_eq!(report.demoted, 1);
    assert_eq!(report.tombstoned, 1);
    assert_eq!(report.freed_bytes, object_bytes);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(2)
    );
    assert!(!chunk_path.exists());
    assert_eq!(
        fixture.store.stat().unwrap().durable_bytes,
        durable_before - object_bytes
    );

    // Demotion never skips ahead: rung 2 is not demotable further.
    let report = fixture.store.demote_fidelity_one_rung(8).unwrap();
    assert_eq!(report.demoted, 0);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(2)
    );

    // Eviction collects the manifest; the tombstoned chunk row — the chained
    // key and its token-cut metadata — is retained for guided recompute
    // until pressure collects it, and the demoted bytes are not
    // double-released (durable hits exactly zero).
    let report = fixture
        .store
        .maintain_capacity_with_fidelity_demotion(1, UtilizationPolicy::default(), 16)
        .unwrap();
    assert!(report.manifests_evicted >= 1);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(2)
    );
    assert_eq!(fixture.store.stat().unwrap().durable_bytes, 0);
    // Final collection of the tombstone-rung row frees nothing more.
    assert!(fixture.store.gc_one().unwrap());
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        None
    );
    assert_eq!(fixture.store.stat().unwrap().durable_bytes, 0);
}

#[test]
fn tombstoned_chunks_plan_guided_recompute_and_never_serve_bytes() {
    let fixture = fixture(b"m6-guided-recompute");
    publish_f16(&fixture.store, 45, true);
    let object_key = only_chunk_object_key(&fixture.store);
    fixture.store.demote_fidelity_one_rung(8).unwrap();
    fixture.store.demote_fidelity_one_rung(8).unwrap();
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(2)
    );

    // The restore planner still authenticates the chained key + token cut
    // row and keeps the candidate, marked as guided recompute.
    let (plan, candidate) = local_restore_plan(&fixture.store, 8);
    assert_eq!(candidate.tombstoned_chunks(), 1);
    assert!(plan.requires_guided_recompute());
    let spans = plan.recompute_spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].0, StateKey::new(0, "k"));
    assert_eq!(
        spans[0].1,
        ChunkSpan {
            token_start: 0,
            token_count: 8,
            plaintext_offset: 0,
            plaintext_bytes: 64,
        }
    );

    // Byte-serving paths fail closed on the tombstoned plan.
    let mut sink = NullSink;
    let result = plan.restore_sequential(&mut sink, &RestoreCancellation::default());
    assert!(matches!(result, Err(StoreError::Expectation(_))));
    let scatter = plan.prepare_scatter_transfer(id(60));
    assert!(matches!(scatter, Err(StoreError::Expectation(_))));
    let stats = plan.chunk_stats_sidecars();
    assert!(matches!(stats, Err(StoreError::Expectation(_))));
}

#[test]
fn watermark_pressure_with_demotion_enabled_demotes_before_evicting() {
    let fixture = fixture(b"m6-watermark-demotion");
    publish_f16(&fixture.store, 46, false);
    let object_key = only_chunk_object_key(&fixture.store);
    let durable_before = fixture.store.stat().unwrap().durable_bytes;
    // One round of watermark pressure demotes 0 -> 1 instead of deleting.
    let report = fixture
        .store
        .maintain_capacity_with_fidelity_demotion(durable_before, UtilizationPolicy::default(), 16)
        .unwrap();
    assert_eq!(report.chunks_evicted, 0);
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(1)
    );
    assert!(fixture.store.chunk_path(&object_key).exists());
    assert_eq!(fixture.store.stat().unwrap().durable_bytes, durable_before);
    // A second round tombstones 1 -> 2 and releases the durable bytes.
    fixture
        .store
        .maintain_capacity_with_fidelity_demotion(durable_before, UtilizationPolicy::default(), 16)
        .unwrap();
    assert_eq!(
        fixture.store.chunk_fidelity_rung(&object_key).unwrap(),
        Some(2)
    );
    assert!(!fixture.store.chunk_path(&object_key).exists());
    assert!(fixture.store.stat().unwrap().durable_bytes < durable_before);
}
