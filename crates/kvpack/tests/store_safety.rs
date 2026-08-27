use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use kvpack::{FillCancellation, LocalStore, SingleflightFill, StoreConfig, StoreError};

fn config(root: &std::path::Path) -> StoreConfig {
    StoreConfig {
        object_root: root.join("objects"),
        catalog_path: root.join("catalog/catalog.sqlite"),
        operator_tenant_id: b"tenant-a".to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: 1 << 30,
        staging_quota_bytes: 1 << 30,
        endurance_bytes_per_five_minutes: 1 << 30,
    }
}

#[test]
fn one_hundred_concurrent_misses_run_one_fill() {
    let fills = Arc::new(SingleflightFill::default());
    let invocations = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(101));
    let mut threads = Vec::new();
    for _ in 0..100 {
        let fills = Arc::clone(&fills);
        let invocations = Arc::clone(&invocations);
        let start = Arc::clone(&start);
        threads.push(std::thread::spawn(move || {
            start.wait();
            fills
                .get_or_fill([7; 32], || {
                    invocations.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    Ok(vec![9; 4096])
                })
                .unwrap()
        }));
    }
    start.wait();
    for thread in threads {
        assert_eq!(&*thread.join().unwrap(), &[9; 4096]);
    }
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
}

#[test]
fn one_hundred_failed_waiters_share_failure_and_a_fresh_coordinator_refills() {
    let fills = Arc::new(SingleflightFill::default());
    let invocations = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(101));
    let mut threads = Vec::new();
    for _ in 0..100 {
        let fills = Arc::clone(&fills);
        let invocations = Arc::clone(&invocations);
        let start = Arc::clone(&start);
        threads.push(std::thread::spawn(move || {
            start.wait();
            fills.get_or_fill([6; 32], || {
                invocations.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(50));
                Err(StoreError::State("injected shared fill failure"))
            })
        }));
    }
    start.wait();
    for thread in threads {
        assert!(thread.join().unwrap().is_err());
    }
    assert_eq!(invocations.load(Ordering::SeqCst), 1);

    drop(fills);
    let restarted = SingleflightFill::default();
    assert_eq!(
        &*restarted.get_or_fill([6; 32], || Ok(vec![5; 8])).unwrap(),
        &[5; 8]
    );
}

#[test]
fn leader_panic_wakes_waiters_and_allows_reassignment() {
    let fills = Arc::new(SingleflightFill::default());
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let leader_fills = Arc::clone(&fills);
    let leader = std::thread::spawn(move || {
        let _ = leader_fills.get_or_fill([8; 32], || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            panic!("injected leader death");
        });
    });
    entered_rx.recv().unwrap();
    let waiter_fills = Arc::clone(&fills);
    let waiter = std::thread::spawn(move || {
        waiter_fills.get_or_fill([8; 32], || {
            panic!("waiter must not become leader while the first leader is live")
        })
    });
    std::thread::sleep(Duration::from_millis(20));
    release_tx.send(()).unwrap();
    assert!(leader.join().is_err());
    assert!(matches!(waiter.join().unwrap(), Err(StoreError::State(_))));
    assert_eq!(
        &*fills.get_or_fill([8; 32], || Ok(vec![3; 4])).unwrap(),
        &[3; 4]
    );
}

#[test]
fn cancelled_waiter_detaches_without_cancelling_the_shared_fill() {
    let fills = Arc::new(SingleflightFill::default());
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let leader_fills = Arc::clone(&fills);
    let leader = std::thread::spawn(move || {
        leader_fills
            .get_or_fill([9; 32], || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(vec![4; 16])
            })
            .unwrap()
    });
    entered_rx.recv().unwrap();
    let cancellation = FillCancellation::default();
    let waiter_cancel = cancellation.clone();
    let waiter_fills = Arc::clone(&fills);
    let waiter = std::thread::spawn(move || {
        waiter_fills.get_or_fill_cancellable([9; 32], &waiter_cancel, || {
            panic!("cancelled waiter must not fetch")
        })
    });
    cancellation.cancel();
    assert!(matches!(waiter.join().unwrap(), Err(StoreError::Cancelled)));
    release_tx.send(()).unwrap();
    assert_eq!(&*leader.join().unwrap(), &[4; 16]);
}

#[test]
fn normal_restart_never_scans_payload_objects() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    LocalStore::open(
        config(temp.path()),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    let shard = temp.path().join("objects/chunks/aa");
    std::fs::create_dir_all(&shard).unwrap();
    std::fs::write(shard.join("malformed.kvchunk"), b"not a chunk").unwrap();

    LocalStore::open(
        config(temp.path()),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .expect("normal restart must trust DB/WAL, not scan payloads");
}

#[test]
fn corrupt_catalog_fails_readiness_closed() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let cfg = config(temp.path());
    std::fs::create_dir_all(cfg.catalog_path.parent().unwrap()).unwrap();
    std::fs::write(&cfg.catalog_path, b"interior sqlite corruption").unwrap();
    assert!(
        LocalStore::open(cfg, kvpack::load_store_key(&key_path, temp.path()).unwrap()).is_err()
    );
}

#[test]
fn newer_catalog_schema_fails_readiness_closed() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let cfg = config(temp.path());
    LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    let connection = rusqlite::Connection::open(&cfg.catalog_path).unwrap();
    connection
        .execute(
            "INSERT INTO schema_migrations(version,applied_ns) VALUES(?1,0)",
            [kvpack::CATALOG_SCHEMA_VERSION + 1],
        )
        .unwrap();
    drop(connection);
    assert!(matches!(
        LocalStore::open(cfg, kvpack::load_store_key(&key_path, temp.path()).unwrap()),
        Err(kvpack::StoreError::State(
            "catalog contains an unknown or newer schema migration"
        ))
    ));
}

#[test]
fn restart_finishes_cataloged_chunk_eviction_without_payload_scan() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let cfg = config(temp.path());
    let store = LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    let object_key = [61u8; 32];
    let name = object_key
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let source = cfg
        .object_root
        .join("chunks")
        .join(&name[..2])
        .join(format!("{name}.kvchunk"));
    let trash = cfg.object_root.join("trash").join(format!("{name}.trash"));
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, b"staged-object").unwrap();
    let connection = rusqlite::Connection::open(&cfg.catalog_path).unwrap();
    connection.execute("INSERT INTO chunks(tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,refcount,location_state,created_ns,last_access_ns,retention_segment,frequency_estimate) VALUES(?1,?2,?3,?4,1,13,13,0,'EVICTING',1,1,'PROBATIONARY',1)", rusqlite::params![store.tenant_namespace().as_slice(), object_key.as_slice(), [62u8; 32].as_slice(), [63u8; 32].as_slice()]).unwrap();
    connection.execute("INSERT INTO locations(tenant,object_kind,object_id,tier,state,locator) VALUES(?1,'chunk',?2,'local','EVICTING',?3)", rusqlite::params![store.tenant_namespace().as_slice(), object_key.as_slice(), source.to_string_lossy()]).unwrap();
    connection
        .execute(
            "UPDATE tenants SET durable_bytes=13 WHERE namespace=?1",
            [store.tenant_namespace().as_slice()],
        )
        .unwrap();
    std::fs::rename(&source, &trash).unwrap();
    drop(connection);
    drop(store);

    let reopened =
        LocalStore::open(cfg, kvpack::load_store_key(&key_path, temp.path()).unwrap()).unwrap();
    assert_eq!(reopened.stat().unwrap().chunks, 0);
    assert_eq!(reopened.stat().unwrap().durable_bytes, 0);
    assert!(!source.exists());
    assert!(!trash.exists());
}

#[test]
fn restart_quarantine_catalogs_every_recovered_partial_byte() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let cfg = config(temp.path());
    let store = LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    drop(store);
    let partial = cfg.object_root.join("partials/restart.upload.partial");
    std::fs::write(&partial, b"partial-state").unwrap();
    let orphan = cfg.object_root.join("quarantine/orphan.quarantine");
    std::fs::write(&orphan, b"orphan").unwrap();

    let reopened = LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    assert!(!partial.exists());
    assert_eq!(reopened.stat().unwrap().quarantine_bytes, 19);
    let connection = rusqlite::Connection::open(&cfg.catalog_path).unwrap();
    let (path_token, file_bytes, lifetime, reason): (String, u64, u64, String) = connection
        .query_row(
            "SELECT path_token,file_bytes,expires_ns-created_ns,reason FROM quarantine_entries WHERE tenant=?1 AND object_kind='restart_partial'",
            [reopened.tenant_namespace().as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(file_bytes, 13);
    assert_eq!(lifetime, 24 * 60 * 60 * 1_000_000_000);
    assert_eq!(reason, "partial recovered during restart");
    assert!(cfg
        .object_root
        .join("quarantine")
        .join(path_token)
        .is_file());
    let orphan_reason: String = connection
        .query_row(
            "SELECT reason FROM quarantine_entries WHERE tenant=?1 AND object_kind='recovered_quarantine'",
            [reopened.tenant_namespace().as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_reason,
        "uncataloged quarantine file recovered during restart"
    );
    assert!(orphan.is_file());
}

// M2 cross-chain physical dedup hardening: two independent delta chains that
// share a 4096-token prefix must persist exactly one physical copy of every
// shared chunk, must report the saved bytes in stat(), and must never let
// eviction of one chain's tip touch chunks the other chain still references.
mod cross_chain_dedup {
    use std::collections::BTreeMap;
    use std::io::Cursor;
    use std::sync::Arc;

    use kvpack::wire::{
        AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, Layout, RepresentationFamilyId,
        RepresentationMode, SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
    };
    use kvpack::{
        AuthenticatedRestorePlan, ExportCutPolicy, ExportDeclaration, ExportSession,
        ExportStateDeclaration, LocalStore, RestoreCancellation, RestoreLimits, RestoreRequest,
        RestoreStatePlan, StoreError, VerifiedRestoreSink, WritePolicy,
    };

    const PREFIX_TOKENS: usize = 4096;
    const EXTENSION_TOKENS: usize = 256;
    const CHAIN_TOKENS: usize = PREFIX_TOKENS + EXTENSION_TOKENS;
    const BYTES_PER_TOKEN: usize = 4;

    fn id(value: u8) -> [u8; 32] {
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
                elements_per_token: BYTES_PER_TOKEN as u64,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(4)],
                dependencies: vec![],
            }],
        }
    }

    fn auxiliary(chain: u8) -> Vec<AuxiliaryInputId> {
        vec![AuxiliaryInputId {
            type_id: id(30),
            value_id: id(chain),
        }]
    }

    fn declaration(chain: u8) -> ExportDeclaration {
        ExportDeclaration {
            semantic_model: semantic(),
            input_tokens: (0..CHAIN_TOKENS as u32).collect(),
            auxiliary_inputs: auxiliary(chain),
            family: family(),
            states: vec![ExportStateDeclaration {
                key: StateKey::new(0, "k"),
                strides: vec![BYTES_PER_TOKEN as u64, 1],
                atomic_group: 1,
            }],
        }
    }

    fn chain_bytes(extension: u8) -> Vec<u8> {
        let mut bytes = vec![7u8; PREFIX_TOKENS * BYTES_PER_TOKEN];
        bytes.extend(vec![extension; EXTENSION_TOKENS * BYTES_PER_TOKEN]);
        bytes
    }

    fn publish_chain(
        store: &Arc<LocalStore>,
        chain: u8,
        extension: u8,
        idempotency: u8,
    ) -> kvpack::PublishedCutSet {
        let policy = WritePolicy::exact_qualified(id(idempotency), semantic(), &family()).unwrap();
        let mut session = ExportSession::begin(
            Arc::clone(store),
            declaration(chain),
            ExportCutPolicy::production_v1(),
            policy,
        )
        .unwrap();
        session
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut Cursor::new(chain_bytes(extension)))
            .unwrap();
        session.commit().unwrap()
    }

    fn chunk_file_sizes(root: &std::path::Path) -> Vec<u64> {
        let chunks = root.join("objects/chunks");
        let mut sizes = std::fs::read_dir(chunks)
            .unwrap()
            .flat_map(|shard| std::fs::read_dir(shard.unwrap().path()).unwrap())
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .collect::<Vec<_>>();
        sizes.sort();
        sizes
    }

    #[derive(Default)]
    struct ShadowSink {
        shadow: BTreeMap<StateKey, Vec<u8>>,
        installed: BTreeMap<StateKey, Vec<u8>>,
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
            Ok(())
        }

        fn abort_restore(&mut self) {
            self.shadow.clear();
        }
    }

    fn restore_chain(
        store: &Arc<LocalStore>,
        chain: u8,
        expect_tokens: u64,
    ) -> (ShadowSink, kvpack::InstalledRestore) {
        let request = RestoreRequest {
            semantic_model: semantic(),
            family: family(),
            input_tokens: (0..CHAIN_TOKENS as u32).collect(),
            auxiliary_inputs: auxiliary(chain),
            minimum_key_epoch: 1,
            maximum_candidates: 8,
        };
        let candidate = store.restore_candidates(request).unwrap()[0].clone();
        assert_eq!(candidate.matched_cut().token_count, expect_tokens);
        let plan = AuthenticatedRestorePlan::build(
            Arc::clone(store),
            &candidate,
            RestoreLimits::default(),
        )
        .unwrap();
        let mut sink = ShadowSink::default();
        let installed = plan
            .restore_sequential(&mut sink, &RestoreCancellation::default())
            .unwrap();
        (sink, installed)
    }

    #[test]
    fn shared_prefix_persists_one_physical_copy_and_survives_tip_eviction() {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
        let store = Arc::new(
            LocalStore::open(
                super::config(temp.path()),
                kvpack::load_store_key(&key_path, temp.path()).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(store.stat().unwrap().deduplicated_bytes, 0);

        // Chain A alone: 17 blocks of 256 tokens, 17 chunk objects. Within one
        // chain the delta-depth bound (MAX_DELTA_DEPTH=7) forces compaction
        // Full manifests at nodes 9 and 17 that re-reference earlier chunks:
        // blocks 1-8 get refcount 3, block 9 refcount 2, blocks 10-16 refcount
        // 2, block 17 refcount 1. Every reference beyond the first is a
        // physical copy never written, so deduplicated_bytes is exactly
        // (8*2 + 1 + 7) * S = 24*S even before any cross-chain sharing.
        let tip_a = publish_chain(&store, 31, 8, 63).exact_final.manifest_id;
        let stat = store.stat().unwrap();
        assert_eq!(stat.chunks, 17);
        let sizes = chunk_file_sizes(temp.path());
        assert_eq!(sizes.len(), 17);
        assert!(sizes.iter().all(|size| *size == sizes[0]));
        let block_object_bytes = sizes[0];
        assert_eq!(stat.deduplicated_bytes, 24 * block_object_bytes);

        // Chain B shares the 4096-token prefix: 16 of its 17 blocks dedup
        // against chain A's physical chunks, so exactly ONE new chunk object
        // (and one new file) lands in the store. Shared refcounts double:
        // blocks 1-8 reach 6, blocks 9-16 reach 4, and deduplicated_bytes is
        // exactly (8*5 + 1*3 + 7*3) * S = 64*S.
        let tip_b = publish_chain(&store, 32, 9, 64).exact_final.manifest_id;
        let stat = store.stat().unwrap();
        assert_eq!(stat.chunks, 18);
        assert_eq!(stat.deduplicated_bytes, 64 * block_object_bytes);
        assert_eq!(chunk_file_sizes(temp.path()).len(), 18);
        let connection =
            rusqlite::Connection::open(temp.path().join("catalog/catalog.sqlite")).unwrap();
        let refcount_histogram = |connection: &rusqlite::Connection| -> Vec<(u64, u64)> {
            let mut statement = connection
                .prepare("SELECT refcount,COUNT(*) FROM chunks GROUP BY refcount ORDER BY refcount")
                .unwrap();
            statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(
            refcount_histogram(&connection),
            vec![(1, 2), (4, 8), (6, 8)]
        );

        // While a restore of chain B is held, its pins cover the shared
        // chunks, and the eviction candidate predicates (manifest_chunks JOIN
        // pins) must exclude EVERY manifest: chain B's chain is pinned, and
        // chain A's tip is a compaction Full manifest that references the
        // pinned shared chunks. No victim may be selected.
        let (held_sink, held_restore) = restore_chain(&store, 32, CHAIN_TOKENS as u64);
        assert_eq!(held_sink.installed[&StateKey::new(0, "k")], chain_bytes(9));
        assert_eq!(store.stat().unwrap().pins, 17);
        assert!(!store.evict_manifest_one().unwrap());
        assert!(!store.gc_one().unwrap());
        held_restore.engine_free().unwrap();
        assert_eq!(store.stat().unwrap().pins, 0);

        // With the hold released, tombstone chain A's tip (operator delete):
        // the eviction ordering collects tombstoned leaves first, so chain A's
        // tip is the deterministic victim.
        connection
            .execute(
                "INSERT INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'manifest',?2,1,1)",
                rusqlite::params![store.tenant_namespace().as_slice(), tip_a.as_slice()],
            )
            .unwrap();
        assert!(store.evict_manifest_one().unwrap());
        let manifest_present = |connection: &rusqlite::Connection, manifest_id: [u8; 32]| -> u64 {
            connection
                .query_row(
                    "SELECT COUNT(*) FROM manifests WHERE tenant=?1 AND manifest_id=?2",
                    rusqlite::params![store.tenant_namespace().as_slice(), manifest_id.as_slice()],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(manifest_present(&connection, tip_a), 0);
        assert_eq!(manifest_present(&connection, tip_b), 1);
        // The evicted tip released one reference on every shared chunk and
        // dropped its unique chunk to refcount 0, but GC must not collect a
        // chunk until it is unreferenced: only chain A's tip chunk is a
        // victim. deduplicated_bytes drops to (8*4 + 8*2) * S = 48*S.
        let stat = store.stat().unwrap();
        assert_eq!(stat.chunks, 18);
        assert_eq!(stat.deduplicated_bytes, 48 * block_object_bytes);
        assert!(store.gc_one().unwrap());
        let stat = store.stat().unwrap();
        assert_eq!(stat.chunks, 17);
        assert_eq!(stat.deduplicated_bytes, 48 * block_object_bytes);
        assert!(!store.gc_one().unwrap());
        assert_eq!(
            refcount_histogram(&connection),
            vec![(1, 1), (3, 8), (5, 8)]
        );
        drop(connection);
        // Chain B's 17 physical chunks (16 shared + its tip) are all on disk.
        assert_eq!(chunk_file_sizes(temp.path()).len(), 17);

        // Chain A's tip is gone: its restore falls back to the shared prefix.
        let request = RestoreRequest {
            semantic_model: semantic(),
            family: family(),
            input_tokens: (0..CHAIN_TOKENS as u32).collect(),
            auxiliary_inputs: auxiliary(31),
            minimum_key_epoch: 1,
            maximum_candidates: 8,
        };
        let candidate = store.restore_candidates(request).unwrap()[0].clone();
        assert_eq!(candidate.matched_cut().token_count, PREFIX_TOKENS as u64);

        // Chain B restores fully from the surviving physical chunks.
        let (sink, installed) = restore_chain(&store, 32, CHAIN_TOKENS as u64);
        assert_eq!(sink.installed[&StateKey::new(0, "k")], chain_bytes(9));
        installed.engine_free().unwrap();
        let stat = store.stat().unwrap();
        assert_eq!(stat.pins, 0);
        assert_eq!(stat.chunks, 17);
    }
}

#[test]
fn catalog_epoch_cannot_move_backward_on_a_stale_config_restart() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let mut cfg = config(temp.path());
    cfg.catalog_epoch = 7;
    let store = LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .unwrap();
    drop(store);

    // A restart with a stale config that rolls the catalog epoch backward
    // must fail closed: the catalog epoch fences every remote capability
    // (audit N2).
    let mut stale = cfg.clone();
    stale.catalog_epoch = 6;
    assert!(matches!(
        LocalStore::open(
            stale,
            kvpack::load_store_key(&key_path, temp.path()).unwrap()
        ),
        Err(kvpack::StoreError::State(
            "catalog epoch cannot move backward in an existing catalog"
        ))
    ));

    // The same epoch (or a forward one) reopens cleanly.
    assert!(LocalStore::open(
        cfg.clone(),
        kvpack::load_store_key(&key_path, temp.path()).unwrap(),
    )
    .is_ok());
    cfg.catalog_epoch = 8;
    assert!(LocalStore::open(cfg, kvpack::load_store_key(&key_path, temp.path()).unwrap()).is_ok());
}

// Audit N3: the decoded-manifest LRU must not let a lenient caller's
// validation pass stand in for a later, stricter caller's bounds. A cached
// decode is re-validated against the CALLER's ValidationContext on every hit.
mod manifest_cache_bounds {
    use std::io::Cursor;
    use std::sync::Arc;

    use kvpack::wire::{
        AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, Layout, ManifestBounds,
        RepresentationFamilyId, RepresentationMode, SemanticModelId, StateKey, StaticDimension,
        TokenAxisRule, ValidationContext,
    };
    use kvpack::{
        ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateDeclaration, LocalStore,
        StoreError, WritePolicy,
    };

    fn id(value: u8) -> [u8; 32] {
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

    #[test]
    fn lenient_then_strict_caller_revalidates_cached_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
        let store = Arc::new(
            LocalStore::open(
                super::config(temp.path()),
                kvpack::load_store_key(&key_path, temp.path()).unwrap(),
            )
            .unwrap(),
        );
        let declaration = ExportDeclaration {
            semantic_model: semantic(),
            input_tokens: (0..4096u32).collect(),
            auxiliary_inputs: vec![AuxiliaryInputId {
                type_id: id(30),
                value_id: id(31),
            }],
            family: family(),
            states: vec![ExportStateDeclaration {
                key: StateKey::new(0, "k"),
                strides: vec![4, 1],
                atomic_group: 1,
            }],
        };
        let policy = WritePolicy::exact_qualified(id(50), semantic(), &family()).unwrap();
        let mut session = ExportSession::begin(
            Arc::clone(&store),
            declaration,
            ExportCutPolicy::production_v1(),
            policy,
        )
        .unwrap();
        session
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut Cursor::new(vec![7u8; 4096 * 4]))
            .unwrap();
        let manifest_id = session.commit().unwrap().exact_final.manifest_id;

        // Lenient caller: validates and populates the cache.
        let lenient = ValidationContext::default();
        store
            .read_authenticated_manifest_object(&manifest_id, &lenient)
            .unwrap();
        // A second lenient read is a cache hit and still succeeds.
        store
            .read_authenticated_manifest_object(&manifest_id, &lenient)
            .unwrap();

        // Strict caller: the cached decode was validated under the lenient
        // bounds, but the hit must re-validate against THESE bounds and fail.
        let strict = ValidationContext {
            bounds: ManifestBounds {
                max_restored_bytes: 1,
                ..ManifestBounds::default()
            },
        };
        assert!(matches!(
            store.read_authenticated_manifest_object(&manifest_id, &strict),
            Err(StoreError::Pack(_))
        ));
        // The strict failure must not poison the cache for lenient callers.
        store
            .read_authenticated_manifest_object(&manifest_id, &lenient)
            .unwrap();
    }
}
