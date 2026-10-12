use std::io::Cursor;
use std::sync::Arc;

use kvpack_core::{
    CacheKind, Codec, DType, FamilyState, Layout, RepresentationFamilyId, RepresentationMode,
    SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
};
use sha2::{Digest, Sha256};

use crate::store::publication::DurabilityFaultPoint;
use crate::{
    portable_prefill_token_ids_sha256, ExportCutPolicy, ExportStateDeclaration,
    ProvisionalExportDeclaration, ProvisionalExportSeal, ProvisionalExportSession, StoreConfig,
    WritePolicy,
};

use super::*;

struct Fixture {
    _temp: tempfile::TempDir,
    config: StoreConfig,
    key_path: PathBuf,
    store: Arc<LocalStore>,
}

fn id(value: u8) -> Id32 {
    [value; 32]
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

fn family() -> RepresentationFamilyId {
    RepresentationFamilyId {
        engine_cache_abi: id(6),
        mode: RepresentationMode::Portable,
        page_size_tokens: 256,
        topology: id(7),
        shard_map: id(8),
        states: ["k", "v"]
            .into_iter()
            .map(|name| FamilyState {
                key: StateKey::new(0, name),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token: 8,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(8)],
                dependencies: Vec::new(),
            })
            .collect(),
    }
}

fn prompt() -> Vec<u32> {
    (1..=301).collect()
}

fn declaration(source: Id32) -> ProvisionalExportDeclaration {
    let family = family();
    ProvisionalExportDeclaration {
        semantic_model: semantic(),
        cached_token_count: 300,
        sealed_prompt_token_ids_sha256: portable_prefill_token_ids_sha256(&prompt()),
        source_declaration_digest: source,
        auxiliary_inputs: Vec::new(),
        states: family
            .states
            .iter()
            .map(|state| ExportStateDeclaration {
                key: state.key.clone(),
                strides: vec![8, 1],
                atomic_group: 1,
            })
            .collect(),
        family,
    }
}

fn fixture(name: &[u8]) -> Fixture {
    fixture_with_limits(name, 1 << 30, 1 << 30, 1 << 30)
}

fn fixture_with_limits(name: &[u8], quota: u64, staging_quota: u64, endurance: u64) -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    crate::create_store_key_random(&key_path, temp.path()).unwrap();
    let config = StoreConfig {
        object_root: temp.path().join("objects"),
        catalog_path: temp.path().join("catalog/catalog.sqlite"),
        operator_tenant_id: name.to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: quota,
        staging_quota_bytes: staging_quota,
        endurance_bytes_per_five_minutes: endurance,
    };
    let store = Arc::new(
        LocalStore::open(
            config.clone(),
            crate::load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap(),
    );
    Fixture {
        _temp: temp,
        config,
        key_path,
        store,
    }
}

fn begin(store: Arc<LocalStore>, source: Id32) -> ProvisionalExportSession {
    let declaration = declaration(source);
    let policy = WritePolicy::exact_qualified(source, semantic(), &declaration.family)
        .unwrap()
        .with_maximum_restored_bytes(4_800)
        .unwrap();
    ProvisionalExportSession::begin(store, declaration, ExportCutPolicy::production_v1(), policy)
        .unwrap()
}

fn state_bytes(value: u8) -> Vec<u8> {
    vec![value; 300 * 8]
}

fn stage(session: &mut ProvisionalExportSession, name: &str, value: u8) {
    let bytes = state_bytes(value);
    session
        .stage_state(
            StateKey::new(0, name),
            Sha256::digest(&bytes).into(),
            &mut Cursor::new(bytes),
        )
        .unwrap();
}

fn seal() -> ProvisionalExportSeal {
    ProvisionalExportSeal {
        prompt_token_ids: prompt(),
        artifact_digest: id(90),
    }
}

fn catalog_count(store: &LocalStore, table: &str) -> u64 {
    let connection = store.lock_catalog().unwrap();
    connection
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
}

fn upload_children(store: &LocalStore) -> usize {
    fs::read_dir(store.config.object_root.join("uploads"))
        .unwrap()
        .count()
}

#[test]
fn staging_is_unreachable_and_cancel_clears_private_state() {
    let fixture = fixture(b"provisional-hidden");
    let source = id(40);
    let mut session = begin(Arc::clone(&fixture.store), source);
    stage(&mut session, "k", 11);
    let stat = fixture.store.stat().unwrap();
    assert_eq!((stat.chunks, stat.manifests), (0, 0));
    assert_eq!(catalog_count(&fixture.store, "locations"), 0);
    assert_eq!(catalog_count(&fixture.store, "prefix_checkpoints"), 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 2);
    assert_eq!(upload_children(&fixture.store), 1);
    session.cancel().unwrap();
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
    assert_eq!(upload_children(&fixture.store), 0);
}

#[test]
fn exact_sources_seal_publish_reopen_and_replay_without_writes() {
    let fixture = fixture(b"provisional-publish-replay");
    let source = id(41);
    let mut first = begin(Arc::clone(&fixture.store), source);
    stage(&mut first, "k", 11);
    stage(&mut first, "v", 22);
    let first = first.seal_and_publish(seal()).unwrap();
    assert_eq!(first.chunk_count, 4);
    assert_eq!(first.staged_chunk_count, 4);
    assert_eq!(first.promoted_chunk_count, 4);
    assert!(first.staged_bytes > 0);
    assert_eq!(first.published.exact_final.input_cut.token_count, 300);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
    assert_eq!(upload_children(&fixture.store), 0);
    let stat = fixture.store.stat().unwrap();
    assert_eq!((stat.chunks, stat.manifests), (4, 2));

    let mut replay = begin(Arc::clone(&fixture.store), source);
    stage(&mut replay, "k", 11);
    stage(&mut replay, "v", 22);
    let replay = replay.seal_and_publish(seal()).unwrap();
    assert_eq!(replay.published, first.published);
    assert_eq!(replay.staged_bytes, 0);
    assert_eq!(replay.promoted_bytes, 0);
    assert_eq!(replay.promoted_chunk_count, 0);
    let replay_stat = fixture.store.stat().unwrap();
    assert_eq!(replay_stat.chunks, stat.chunks);
    assert_eq!(replay_stat.manifests, stat.manifests);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);

    let mut changed_seal = begin(Arc::clone(&fixture.store), source);
    stage(&mut changed_seal, "k", 11);
    stage(&mut changed_seal, "v", 22);
    let mut invalid = seal();
    invalid.artifact_digest = id(91);
    assert!(changed_seal.seal_and_publish(invalid).is_err());
    let unchanged = fixture.store.stat().unwrap();
    assert_eq!(
        (unchanged.chunks, unchanged.manifests),
        (stat.chunks, stat.manifests)
    );
}

#[test]
fn state_order_bounds_digest_and_seal_commitment_fail_closed() {
    for (case, mutate) in [("short", 0u8), ("long", 1u8), ("digest", 2u8)] {
        let fixture = fixture(case.as_bytes());
        let source = id(50 + mutate);
        let mut session = begin(Arc::clone(&fixture.store), source);
        let mut bytes = state_bytes(11);
        let digest: Id32 = Sha256::digest(&bytes).into();
        match mutate {
            0 => {
                bytes.pop();
            }
            1 => bytes.push(0),
            2 => bytes[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(session
            .stage_state(StateKey::new(0, "k"), digest, &mut Cursor::new(bytes))
            .is_err());
        drop(session);
        assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
        assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
        assert_eq!(upload_children(&fixture.store), 0);
    }

    let wrong_order_fixture = fixture(b"wrong-order");
    let mut wrong_order = begin(Arc::clone(&wrong_order_fixture.store), id(54));
    let bytes = state_bytes(22);
    assert!(wrong_order
        .stage_state(
            StateKey::new(0, "v"),
            Sha256::digest(&bytes).into(),
            &mut Cursor::new(bytes),
        )
        .is_err());
    drop(wrong_order);

    let fixture = fixture(b"wrong-seal");
    let mut wrong_seal = begin(Arc::clone(&fixture.store), id(55));
    stage(&mut wrong_seal, "k", 11);
    stage(&mut wrong_seal, "v", 22);
    let mut invalid = seal();
    invalid.prompt_token_ids.pop();
    assert!(wrong_seal.seal_and_publish(invalid).is_err());
    assert_eq!(fixture.store.stat().unwrap().manifests, 0);
    assert_eq!(fixture.store.stat().unwrap().chunks, 0);
    assert_eq!(catalog_count(&fixture.store, "prefix_checkpoints"), 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
}

#[test]
fn quota_staging_and_endurance_reject_before_upload_directory() {
    for (name, quota, staging, endurance) in [
        (b"quota".as_slice(), 1, 1 << 30, 1 << 30),
        (b"staging".as_slice(), 1 << 30, 1, 1 << 30),
        (b"endurance".as_slice(), 1 << 30, 1 << 30, 1),
    ] {
        let fixture = fixture_with_limits(name, quota, staging, endurance);
        let source = id(name[0]);
        let declaration = declaration(source);
        let policy = WritePolicy::exact_qualified(source, semantic(), &declaration.family)
            .unwrap()
            .with_maximum_restored_bytes(4_800)
            .unwrap();
        assert!(ProvisionalExportSession::begin(
            Arc::clone(&fixture.store),
            declaration,
            ExportCutPolicy::production_v1(),
            policy,
        )
        .is_err());
        assert_eq!(upload_children(&fixture.store), 0);
        assert_eq!(catalog_count(&fixture.store, "uploads"), 0);
    }
}

#[test]
fn begin_identity_and_seal_artifact_are_immutable() {
    let fixture = fixture(b"provisional-immutable-intent");
    let source = id(58);
    let first = begin(Arc::clone(&fixture.store), source);
    let mut changed = declaration(source);
    changed.sealed_prompt_token_ids_sha256 = id(59);
    let policy = WritePolicy::exact_qualified(source, semantic(), &changed.family)
        .unwrap()
        .with_maximum_restored_bytes(4_800)
        .unwrap();
    assert!(ProvisionalExportSession::begin(
        Arc::clone(&fixture.store),
        changed,
        ExportCutPolicy::production_v1(),
        policy,
    )
    .is_err());
    first.cancel().unwrap();

    let mut zero_artifact = begin(Arc::clone(&fixture.store), id(59));
    stage(&mut zero_artifact, "k", 11);
    stage(&mut zero_artifact, "v", 22);
    let mut invalid = seal();
    invalid.artifact_digest = [0; 32];
    assert!(zero_artifact.seal_and_publish(invalid).is_err());
    assert_eq!(fixture.store.stat().unwrap().chunks, 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
}

#[test]
fn catalog_begin_fault_removes_only_same_inode_unreferenced_promotions() {
    let fixture = fixture(b"provisional-catalog-fault");
    let mut session = begin(Arc::clone(&fixture.store), id(60));
    stage(&mut session, "k", 11);
    stage(&mut session, "v", 22);
    *fixture.store.durability_fault.lock().unwrap() = Some(DurabilityFaultPoint::CatalogBegin);
    assert!(session.seal_and_publish(seal()).is_err());
    let stat = fixture.store.stat().unwrap();
    assert_eq!(
        (stat.chunks, stat.manifests, stat.reserved_bytes),
        (0, 0, 0)
    );
    assert_eq!(catalog_count(&fixture.store, "locations"), 0);
    assert_eq!(catalog_count(&fixture.store, "prefix_checkpoints"), 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
    assert_eq!(upload_children(&fixture.store), 0);
}

#[test]
fn startup_reconciliation_aborts_exact_provisional_directory() {
    let fixture = fixture(b"provisional-reconcile");
    let mut session = begin(Arc::clone(&fixture.store), id(61));
    stage(&mut session, "k", 11);
    std::mem::forget(session);
    let reopened = LocalStore::open(
        fixture.config.clone(),
        crate::load_store_key(&fixture.key_path, fixture._temp.path()).unwrap(),
    )
    .unwrap();
    assert_eq!(reopened.stat().unwrap().reserved_bytes, 0);
    assert_eq!(catalog_count(&reopened, "upload_chunks"), 0);
    assert_eq!(upload_children(&reopened), 0);
}

#[test]
fn startup_reconciliation_rejects_unknown_or_nested_entries() {
    let fixture = fixture(b"provisional-reconcile-unknown");
    drop(fixture.store);
    let unknown = fixture
        .config
        .object_root
        .join("uploads/not-a-provisional-run");
    fs::create_dir(&unknown).unwrap();
    fs::create_dir(unknown.join("nested")).unwrap();
    assert!(LocalStore::open(
        fixture.config,
        crate::load_store_key(&fixture.key_path, fixture._temp.path()).unwrap(),
    )
    .is_err());
}

#[test]
fn stale_session_token_cannot_kill_the_live_reservation() {
    let fixture = fixture(b"provisional-fencing");
    let source = id(62);
    let mut stale = begin(Arc::clone(&fixture.store), source);
    // A second begin on the same RECEIVING key resumes and mints a fresh
    // fencing token, invalidating the stale session.
    let mut live = begin(Arc::clone(&fixture.store), source);
    let bytes = state_bytes(11);
    assert!(stale
        .stage_state(
            StateKey::new(0, "k"),
            Sha256::digest(&bytes).into(),
            &mut Cursor::new(bytes),
        )
        .is_err());
    // The stale session's fenced cancel (via Drop) must leave the live
    // reservation untouched.
    drop(stale);
    let reserved = fixture.store.stat().unwrap().reserved_bytes;
    assert!(reserved > 0);
    stage(&mut live, "k", 11);
    stage(&mut live, "v", 22);
    let receipt = live.seal_and_publish(seal()).unwrap();
    assert_eq!(receipt.chunk_count, 4);
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
}

#[test]
fn corrupted_staged_chunk_fails_promotion_closed() {
    let fixture = fixture(b"provisional-bitrot");
    let source = id(63);
    let mut session = begin(Arc::clone(&fixture.store), source);
    stage(&mut session, "k", 11);
    stage(&mut session, "v", 22);
    let staged = fixture
        .store
        .config
        .object_root
        .join("uploads")
        .join(hex(&source))
        .join(format!("{:020}.kvchunk", 0));
    let mut bytes = fs::read(&staged).unwrap();
    bytes[0] ^= 1;
    fs::write(&staged, bytes).unwrap();
    assert!(session.seal_and_publish(seal()).is_err());
    let stat = fixture.store.stat().unwrap();
    assert_eq!(
        (stat.chunks, stat.manifests, stat.reserved_bytes),
        (0, 0, 0)
    );
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
    assert_eq!(upload_children(&fixture.store), 0);
}

#[test]
fn expired_lease_refuses_stage_and_seal() {
    let fixture = fixture(b"provisional-lease-expired");
    let source = id(64);
    let mut session = begin(Arc::clone(&fixture.store), source);
    stage(&mut session, "k", 11);
    force_lease_expiry(&fixture.store, &source);
    let bytes = state_bytes(22);
    assert!(session
        .stage_state(
            StateKey::new(0, "v"),
            Sha256::digest(&bytes).into(),
            &mut Cursor::new(bytes),
        )
        .is_err());
    drop(session);

    let mut sealed = begin(Arc::clone(&fixture.store), id(65));
    stage(&mut sealed, "k", 11);
    stage(&mut sealed, "v", 22);
    force_lease_expiry(&fixture.store, &id(65));
    assert!(sealed.seal_and_publish(seal()).is_err());
    let stat = fixture.store.stat().unwrap();
    assert_eq!(
        (stat.chunks, stat.manifests, stat.reserved_bytes),
        (0, 0, 0)
    );
}

#[test]
fn stat_reaps_expired_upload_and_frees_quota() {
    let fixture = fixture(b"provisional-lease-reap");
    let source = id(66);
    let mut session = begin(Arc::clone(&fixture.store), source);
    stage(&mut session, "k", 11);
    std::mem::forget(session);
    assert!(fixture.store.stat().unwrap().reserved_bytes > 0);
    assert_eq!(upload_children(&fixture.store), 1);
    force_lease_expiry(&fixture.store, &source);
    let stat = fixture.store.stat().unwrap();
    assert_eq!(stat.reserved_bytes, 0);
    assert_eq!(stat.active_uploads, 0);
    assert_eq!(upload_children(&fixture.store), 0);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
}

#[test]
fn reconciliation_bound_is_per_upload_directory() {
    let within = fixture(b"provisional-bound-within");
    let source = id(67);
    let session = begin(Arc::clone(&within.store), source);
    std::mem::forget(session);
    let directory = within.config.object_root.join("uploads").join(hex(&source));
    for ordinal in 0..PROVISIONAL_DIRECTORY_ENTRY_BOUND {
        fs::write(
            directory.join(format!("{ordinal:020}.kvchunk")),
            b"x".as_slice(),
        )
        .unwrap();
    }
    drop(within.store);
    let reopened = LocalStore::open(
        within.config.clone(),
        crate::load_store_key(&within.key_path, within._temp.path()).unwrap(),
    )
    .unwrap();
    assert_eq!(upload_children(&reopened), 0);
    drop(reopened);

    let beyond = fixture(b"provisional-bound-beyond");
    let source = id(68);
    let session = begin(Arc::clone(&beyond.store), source);
    std::mem::forget(session);
    let directory = beyond.config.object_root.join("uploads").join(hex(&source));
    for ordinal in 0..=PROVISIONAL_DIRECTORY_ENTRY_BOUND {
        fs::write(
            directory.join(format!("{ordinal:020}.kvchunk")),
            b"x".as_slice(),
        )
        .unwrap();
    }
    drop(beyond.store);
    assert!(LocalStore::open(
        beyond.config.clone(),
        crate::load_store_key(&beyond.key_path, beyond._temp.path()).unwrap(),
    )
    .is_err());
}

#[test]
fn seal_persists_boundary_token_and_provenance() {
    let fixture = fixture(b"provisional-seal-metadata");
    let source = id(69);
    let mut session = begin(Arc::clone(&fixture.store), source);
    session.record_clock_offset_ns(7_500).unwrap();
    let provenance = session.provenance();
    assert!(provenance.source_wall_clock_ns > 0);
    assert_eq!(provenance.clock_offset_ns, Some(7_500));
    assert!(provenance.quiesced);
    stage(&mut session, "k", 11);
    stage(&mut session, "v", 22);
    let receipt = session.seal_and_publish(seal()).unwrap();
    assert_eq!(receipt.boundary_token_id, 301);
    assert_eq!(receipt.provenance, provenance);
    let metadata = fixture
        .store
        .provisional_upload_metadata(&source)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.boundary_token_id, Some(301));
    assert_eq!(metadata.provenance, provenance);

    let mut late = begin(Arc::clone(&fixture.store), id(70));
    stage(&mut late, "k", 11);
    assert!(late.record_clock_offset_ns(1).is_err());
    late.cancel().unwrap();
}

fn force_lease_expiry(store: &LocalStore, source: &Id32) {
    let connection = store.lock_catalog().unwrap();
    connection
        .execute(
            "UPDATE uploads SET lease_expires_ns=1 WHERE idempotency_key=?1",
            [source.as_slice()],
        )
        .unwrap();
}

#[test]
fn real_16384_prompt_inventory_publishes_exactly_3072_objects() {
    const PROMPT_TOKENS: usize = 16_384;
    const CACHED_TOKENS: usize = PROMPT_TOKENS - 1;
    let fixture = fixture_with_limits(b"provisional-real-inventory", 2 << 30, 2 << 30, 2 << 30);
    let mut family_states = Vec::with_capacity(48);
    let mut states = Vec::with_capacity(48);
    for layer in 0..24 {
        for name in ["attn.k", "attn.v"] {
            let key = StateKey::new(layer, name);
            family_states.push(FamilyState {
                key: key.clone(),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token: 1,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(1)],
                dependencies: Vec::new(),
            });
            states.push(ExportStateDeclaration {
                key,
                strides: vec![1, 1],
                atomic_group: layer + 1,
            });
        }
    }
    let family = RepresentationFamilyId {
        engine_cache_abi: id(71),
        mode: RepresentationMode::Portable,
        page_size_tokens: 256,
        topology: id(72),
        shard_map: id(73),
        states: family_states,
    };
    let prompt = (1..=PROMPT_TOKENS as u32).collect::<Vec<_>>();
    let source = id(74);
    let declaration = ProvisionalExportDeclaration {
        semantic_model: semantic(),
        cached_token_count: CACHED_TOKENS as u32,
        sealed_prompt_token_ids_sha256: portable_prefill_token_ids_sha256(&prompt),
        source_declaration_digest: source,
        auxiliary_inputs: Vec::new(),
        family: family.clone(),
        states,
    };
    let policy = WritePolicy::exact_qualified(source, semantic(), &family)
        .unwrap()
        .with_maximum_restored_bytes((CACHED_TOKENS * 48) as u64)
        .unwrap();
    let mut session = ProvisionalExportSession::begin(
        Arc::clone(&fixture.store),
        declaration,
        ExportCutPolicy::production_v1(),
        policy,
    )
    .unwrap();
    for layer in 0..24 {
        for (name, value) in [("attn.k", layer as u8), ("attn.v", layer as u8 + 24)] {
            let bytes = vec![value; CACHED_TOKENS];
            let receipt = session
                .stage_state(
                    StateKey::new(layer, name),
                    Sha256::digest(&bytes).into(),
                    &mut Cursor::new(bytes),
                )
                .unwrap();
            assert_eq!(receipt.chunk_count, 64);
        }
    }
    let receipt = session
        .seal_and_publish(ProvisionalExportSeal {
            prompt_token_ids: prompt,
            artifact_digest: id(75),
        })
        .unwrap();
    assert_eq!(receipt.chunk_count, 3_072);
    assert_eq!(receipt.promoted_chunk_count, 3_072);
    assert_eq!(fixture.store.stat().unwrap().chunks, 3_072);
    assert_eq!(catalog_count(&fixture.store, "upload_chunks"), 0);
    assert_eq!(upload_children(&fixture.store), 0);
}

/// Twin store sharing the reference store's root key, so manifest ids are
/// comparable across the two (RED_TEAM HIGH-4 regression coverage).
fn twin_fixture(reference: &Fixture) -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    fs::create_dir_all(key_path.parent().unwrap()).unwrap();
    fs::copy(&reference.key_path, &key_path).unwrap();
    let config = StoreConfig {
        object_root: temp.path().join("objects"),
        catalog_path: temp.path().join("catalog/catalog.sqlite"),
        operator_tenant_id: reference.config.operator_tenant_id.clone(),
        ..reference.config.clone()
    };
    let store = Arc::new(
        LocalStore::open(
            config.clone(),
            crate::load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap(),
    );
    Fixture {
        _temp: temp,
        config,
        key_path,
        store,
    }
}

/// Deterministic export: encryption carries fresh salt/nonce bytes per
/// attempt, so only an unencrypted policy produces comparable manifest ids
/// across two stores.
fn begin_unencrypted(store: Arc<LocalStore>, source: Id32) -> ProvisionalExportSession {
    let declaration = declaration(source);
    let policy = WritePolicy::exact_qualified(source, semantic(), &declaration.family)
        .unwrap()
        .with_maximum_restored_bytes(4_800)
        .unwrap()
        .with_encryption(false);
    ProvisionalExportSession::begin(store, declaration, ExportCutPolicy::production_v1(), policy)
        .unwrap()
}

#[test]
fn aborted_upload_reinitializes_and_republishes_the_same_manifest() {
    let reference = fixture(b"provisional-reinit");
    let source = id(70);
    // Reference: the same declaration exported without any interruption.
    let mut smooth = begin_unencrypted(Arc::clone(&reference.store), source);
    stage(&mut smooth, "k", 11);
    stage(&mut smooth, "v", 22);
    let smooth = smooth.seal_and_publish(seal()).unwrap();

    // Crash mid-export in the twin store: the reconcile-equivalent abort
    // discards the staged state and leaves a terminal ABORTED row.
    let crashed = twin_fixture(&reference);
    let mut interrupted = begin_unencrypted(Arc::clone(&crashed.store), source);
    stage(&mut interrupted, "k", 11);
    crashed.store.abort_upload(&source).unwrap();
    drop(interrupted);
    assert_eq!(crashed.store.stat().unwrap().reserved_bytes, 0);
    assert_eq!(upload_children(&crashed.store), 0);

    // RE_INIT: the identical declaration re-reserves the burned key and the
    // caller re-streams from scratch.
    let mut recovered = begin_unencrypted(Arc::clone(&crashed.store), source);
    assert!(crashed.store.stat().unwrap().reserved_bytes > 0);
    stage(&mut recovered, "k", 11);
    stage(&mut recovered, "v", 22);
    let recovered = recovered.seal_and_publish(seal()).unwrap();
    assert_eq!(recovered.published, smooth.published);
    assert_eq!(recovered.staged_chunk_count, smooth.staged_chunk_count);
    assert_eq!(crashed.store.stat().unwrap().reserved_bytes, 0);
    assert_eq!(catalog_count(&crashed.store, "upload_chunks"), 0);
}

#[test]
fn reinit_rejects_a_mutated_declaration_under_the_burned_key() {
    let fixture = fixture(b"provisional-reinit-mutated");
    let source = id(71);
    let mut session = begin(Arc::clone(&fixture.store), source);
    stage(&mut session, "k", 11);
    session.cancel().unwrap();

    // Same idempotency key, but the declaration changed: RE_INIT requires a
    // byte-identical declaration digest, so this is a new upload under an
    // already-burned key and fails closed.
    let mut mutated = declaration(source);
    mutated.cached_token_count = 299;
    let policy = WritePolicy::exact_qualified(source, semantic(), &mutated.family)
        .unwrap()
        .with_maximum_restored_bytes(4_800)
        .unwrap();
    let result = ProvisionalExportSession::begin(
        Arc::clone(&fixture.store),
        mutated,
        ExportCutPolicy::production_v1(),
        policy,
    );
    let error = match result {
        Ok(_) => panic!("mutated declaration re-initialized the burned key"),
        Err(error) => error,
    };
    assert!(matches!(error, StoreError::Expectation(_)));
}

#[test]
fn reinit_fences_the_stale_session_and_aborted_receive() {
    let fixture = fixture(b"provisional-reinit-fence");
    let source = id(72);
    let mut stale = begin(Arc::clone(&fixture.store), source);
    stage(&mut stale, "k", 11);

    // Crash-reconcile equivalent: terminal ABORTED with the stale session
    // still holding its pre-abort fencing token.
    fixture.store.abort_upload(&source).unwrap();
    // Nobody can drive the terminal row to receiving.
    assert!(fixture.store.mark_receiving(&source).is_err());

    // RE_INIT mints a fresh fencing token; the stale session is dead.
    let mut live = begin(Arc::clone(&fixture.store), source);
    let bytes = state_bytes(22);
    assert!(stale
        .stage_state(
            StateKey::new(0, "v"),
            Sha256::digest(&bytes).into(),
            &mut Cursor::new(bytes),
        )
        .is_err());
    // The stale session's fenced cancel (via Drop) must leave the
    // re-initialized reservation untouched.
    drop(stale);
    assert!(fixture.store.stat().unwrap().reserved_bytes > 0);

    stage(&mut live, "k", 11);
    stage(&mut live, "v", 22);
    let receipt = live.seal_and_publish(seal()).unwrap();
    assert_eq!(receipt.chunk_count, 4);
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
}
