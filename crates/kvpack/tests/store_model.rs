use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use kvpack::wire::{
    AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, Layout, RepresentationFamilyId,
    RepresentationMode, SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
};
use kvpack::{
    ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateDeclaration, LocalStore,
    StoreConfig, WritePolicy,
};
use rusqlite::{params, Connection};

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
            elements_per_token: 8,
            dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(8)],
            dependencies: vec![],
        }],
    }
}

fn config(root: &Path) -> StoreConfig {
    StoreConfig {
        object_root: root.join("objects"),
        catalog_path: root.join("catalog/catalog.sqlite"),
        operator_tenant_id: b"randomized-store-model".to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes: 1 << 30,
        staging_quota_bytes: 1 << 30,
        endurance_bytes_per_five_minutes: 1 << 30,
    }
}

fn publish(store: Arc<LocalStore>, sequence: u64) {
    let family = family();
    let token_base = u32::try_from(sequence.saturating_mul(8)).unwrap();
    let declaration = ExportDeclaration {
        semantic_model: semantic(),
        input_tokens: (token_base..token_base + 4).collect(),
        auxiliary_inputs: vec![AuxiliaryInputId {
            type_id: id(30),
            value_id: id(31),
        }],
        states: vec![ExportStateDeclaration {
            key: StateKey::new(0, "k"),
            strides: vec![8, 1],
            atomic_group: 1,
        }],
        family: family.clone(),
    };
    let mut idempotency = [0x5au8; 32];
    idempotency[..8].copy_from_slice(&sequence.to_le_bytes());
    let mut session = ExportSession::begin(
        store,
        declaration,
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(idempotency, semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = Cursor::new(vec![(sequence & 0xff) as u8; 32]);
    session
        .next_state(StateKey::new(0, "k"))
        .unwrap()
        .write_source(&mut source)
        .unwrap();
    session.commit().unwrap();
}

#[derive(Default)]
struct ModelSnapshot {
    manifests: BTreeSet<[u8; 32]>,
    chunks: BTreeSet<[u8; 32]>,
    eligible_manifests: BTreeSet<[u8; 32]>,
    eligible_chunks: BTreeSet<[u8; 32]>,
    tombstones: BTreeSet<[u8; 32]>,
}

fn model_snapshot(connection: &Connection, tenant: &[u8]) -> ModelSnapshot {
    let active_upload: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM uploads WHERE tenant=?1 AND state IN ('INIT','RESERVED','RECEIVING','VERIFIED')",
            [tenant],
            |row| row.get(0),
        )
        .unwrap();
    let manifests = ids(
        connection,
        "SELECT manifest_id FROM manifests WHERE tenant=?1",
        tenant,
    );
    let chunks = ids(
        connection,
        "SELECT object_key FROM chunks WHERE tenant=?1",
        tenant,
    );
    let available_manifests = ids(
        connection,
        "SELECT object_id FROM locations WHERE tenant=?1 AND object_kind='manifest' AND tier='local' AND state='AVAILABLE'",
        tenant,
    );
    let available_chunks = ids(
        connection,
        "SELECT object_key FROM chunks WHERE tenant=?1 AND location_state='AVAILABLE'",
        tenant,
    );
    let tombstones = ids(
        connection,
        "SELECT object_id FROM tombstones WHERE tenant=?1 AND object_kind='manifest'",
        tenant,
    );
    let pinned = ids(
        connection,
        "SELECT object_key FROM pins WHERE tenant=?1",
        tenant,
    );
    let leased_chunks = ids(
        connection,
        "SELECT object_id FROM leases WHERE tenant=?1 AND object_kind='chunk' AND state='ACTIVE' AND expires_ns>0",
        tenant,
    );
    let leased_manifests = ids(
        connection,
        "SELECT object_id FROM leases WHERE tenant=?1 AND object_kind='manifest' AND state='ACTIVE' AND expires_ns>0",
        tenant,
    );
    let mut manifest_chunks = BTreeMap::<[u8; 32], BTreeSet<[u8; 32]>>::new();
    let mut statement = connection
        .prepare("SELECT manifest_id,object_key FROM manifest_chunks WHERE tenant=?1")
        .unwrap();
    let rows = statement
        .query_map([tenant], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .unwrap();
    for row in rows {
        let (manifest, chunk) = row.unwrap();
        manifest_chunks
            .entry(as_id(manifest))
            .or_default()
            .insert(as_id(chunk));
    }
    let mut live_children = BTreeSet::new();
    let mut statement = connection
        .prepare("SELECT parent_id FROM manifests WHERE tenant=?1 AND parent_id IS NOT NULL")
        .unwrap();
    let rows = statement
        .query_map([tenant], |row| row.get::<_, Vec<u8>>(0))
        .unwrap();
    for row in rows {
        live_children.insert(as_id(row.unwrap()));
    }
    let eligible_manifests = if active_upload == 0 {
        manifests
            .iter()
            .copied()
            .filter(|manifest| available_manifests.contains(manifest))
            .filter(|manifest| !tombstones.contains(manifest))
            .filter(|manifest| !live_children.contains(manifest))
            .filter(|manifest| !leased_manifests.contains(manifest))
            .filter(|manifest| {
                manifest_chunks.get(manifest).is_none_or(|objects| {
                    objects.is_disjoint(&pinned) && objects.is_disjoint(&leased_chunks)
                })
            })
            .collect()
    } else {
        BTreeSet::new()
    };
    let refcounts = chunk_refcounts(connection, tenant);
    let eligible_chunks = if active_upload == 0 {
        chunks
            .iter()
            .copied()
            .filter(|chunk| available_chunks.contains(chunk))
            .filter(|chunk| refcounts.get(chunk) == Some(&0))
            .filter(|chunk| !pinned.contains(chunk) && !leased_chunks.contains(chunk))
            .collect()
    } else {
        BTreeSet::new()
    };
    ModelSnapshot {
        manifests,
        chunks,
        eligible_manifests,
        eligible_chunks,
        tombstones,
    }
}

fn ids(connection: &Connection, sql: &str, tenant: &[u8]) -> BTreeSet<[u8; 32]> {
    let mut statement = connection.prepare(sql).unwrap();
    statement
        .query_map([tenant], |row| row.get::<_, Vec<u8>>(0))
        .unwrap()
        .map(|row| as_id(row.unwrap()))
        .collect()
}

fn as_id(value: Vec<u8>) -> [u8; 32] {
    value.try_into().unwrap()
}

fn chunk_refcounts(connection: &Connection, tenant: &[u8]) -> BTreeMap<[u8; 32], u64> {
    let mut statement = connection
        .prepare("SELECT object_key,refcount FROM chunks WHERE tenant=?1")
        .unwrap();
    statement
        .query_map([tenant], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, u64>(1)?))
        })
        .unwrap()
        .map(|row| {
            let (key, count) = row.unwrap();
            (as_id(key), count)
        })
        .collect()
}

fn assert_invariants(
    store: &LocalStore,
    connection: &Connection,
    tenant: &[u8],
    evicted: &BTreeSet<[u8; 32]>,
) {
    let refcounts = chunk_refcounts(connection, tenant);
    for (chunk, catalog_count) in &refcounts {
        let reference_count: u64 = connection
            .query_row(
                "SELECT COUNT(DISTINCT manifest_id) FROM manifest_chunks WHERE tenant=?1 AND object_key=?2",
                params![tenant, chunk.as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(*catalog_count, reference_count, "chunk refcount drifted");
    }
    let missing_references: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM manifest_chunks mc LEFT JOIN chunks c ON c.tenant=mc.tenant AND c.object_key=mc.object_key WHERE mc.tenant=?1 AND c.object_key IS NULL",
            [tenant],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(missing_references, 0);
    let stale_prefixes: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM prefix_checkpoints p LEFT JOIN manifests m ON m.tenant=p.tenant AND m.manifest_id=p.manifest_id WHERE p.tenant=?1 AND m.manifest_id IS NULL",
            [tenant],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stale_prefixes, 0);
    let catalog_bytes: u64 = connection
        .query_row(
            "SELECT COALESCE((SELECT SUM(object_bytes) FROM chunks WHERE tenant=?1),0)+COALESCE((SELECT SUM(file_bytes) FROM manifests WHERE tenant=?1),0)",
            [tenant],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(store.stat().unwrap().durable_bytes, catalog_bytes);
    let invalid_locations: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM locations l WHERE l.tenant=?1 AND l.state='AVAILABLE' AND ((l.object_kind='chunk' AND NOT EXISTS(SELECT 1 FROM chunks c WHERE c.tenant=l.tenant AND c.object_key=l.object_id)) OR (l.object_kind='manifest' AND NOT EXISTS(SELECT 1 FROM manifests m WHERE m.tenant=l.tenant AND m.manifest_id=l.object_id)))",
            [tenant],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(invalid_locations, 0);
    let mut statement = connection
        .prepare(
            "SELECT locator FROM locations WHERE tenant=?1 AND tier='local' AND state='AVAILABLE'",
        )
        .unwrap();
    for path in statement
        .query_map([tenant], |row| row.get::<_, String>(0))
        .unwrap()
    {
        assert!(Path::new(&path.unwrap()).is_file());
    }
    let snapshot = model_snapshot(connection, tenant);
    assert!(evicted.is_subset(&snapshot.tombstones));
    assert!(snapshot.manifests.is_disjoint(&snapshot.tombstones));
}

fn toggle_pin(connection: &Connection, tenant: &[u8], object: [u8; 32]) {
    let deleted = connection
        .execute(
            "DELETE FROM pins WHERE tenant=?1 AND object_key=?2",
            params![tenant, object.as_slice()],
        )
        .unwrap();
    if deleted == 0 {
        connection
            .execute(
                "INSERT INTO pins(tenant,pin_id,object_key,owner_pid,owner_start,created_ns) VALUES(?1,?2,?2,1,?3,1)",
                params![tenant, object.as_slice(), id(90).as_slice()],
            )
            .unwrap();
    }
}

fn toggle_lease(
    connection: &Connection,
    tenant: &[u8],
    kind: &str,
    object: [u8; 32],
    sequence: u64,
) {
    let deleted = connection
        .execute(
            "DELETE FROM leases WHERE tenant=?1 AND object_kind=?2 AND object_id=?3",
            params![tenant, kind, object.as_slice()],
        )
        .unwrap();
    if deleted == 0 {
        let mut lease_id = [0xc3u8; 32];
        lease_id[..8].copy_from_slice(&sequence.to_le_bytes());
        lease_id[8] = if kind == "chunk" { 1 } else { 2 };
        connection
            .execute(
                "INSERT INTO leases(tenant,lease_id,object_kind,object_id,owner_id,state,expires_ns,created_ns) VALUES(?1,?2,?3,?4,?5,'ACTIVE',?6,1)",
                params![
                    tenant,
                    lease_id.as_slice(),
                    kind,
                    object.as_slice(),
                    id(91).as_slice(),
                    i64::MAX,
                ],
            )
            .unwrap();
    }
}

fn toggle_upload(connection: &Connection, tenant: &[u8]) {
    let upload = id(240);
    let deleted = connection
        .execute(
            "DELETE FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![tenant, upload.as_slice()],
        )
        .unwrap();
    if deleted == 0 {
        connection
            .execute(
                "INSERT INTO uploads(tenant,idempotency_key,state,reserved_bytes,expected_bytes,created_ns,updated_ns,next_chunk_ordinal,generation,intent_digest) VALUES(?1,?2,'RESERVED',0,1,1,1,0,1,?3)",
                params![tenant, upload.as_slice(), id(241).as_slice()],
            )
            .unwrap();
    }
}

fn choose(values: &BTreeSet<[u8; 32]>, random: u64) -> Option<[u8; 32]> {
    if values.is_empty() {
        None
    } else {
        values.iter().nth(random as usize % values.len()).copied()
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

#[test]
fn randomized_gc_pin_lease_upload_and_restart_history_matches_the_store_model() {
    let temp = tempfile::tempdir().unwrap();
    let key_path = temp.path().join("keys/root.key");
    kvpack::create_store_key_random(&key_path, temp.path()).unwrap();
    let configuration = config(temp.path());
    let mut store = Arc::new(
        LocalStore::open(
            configuration.clone(),
            kvpack::load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap(),
    );
    let tenant = store.tenant_namespace();
    let mut connection = Connection::open(&configuration.catalog_path).unwrap();
    let mut next_publication = 1u64;
    for _ in 0..6 {
        publish(Arc::clone(&store), next_publication);
        next_publication += 1;
    }
    let mut random = 0x4b56_5041_434b_5331u64;
    let mut evicted = BTreeSet::new();
    let mut operation_counts = [0u64; 9];
    let mut restarts = 0u64;
    for step in 0..300u64 {
        let value = next_random(&mut random);
        let before = model_snapshot(&connection, &tenant);
        let operation = (value % 9) as usize;
        operation_counts[operation] += 1;
        match operation {
            0 => {
                if let Some(chunk) = choose(&before.chunks, value >> 8) {
                    toggle_pin(&connection, &tenant, chunk);
                }
            }
            1 => {
                if let Some(chunk) = choose(&before.chunks, value >> 8) {
                    toggle_lease(&connection, &tenant, "chunk", chunk, step);
                }
            }
            2 => {
                if let Some(manifest) = choose(&before.manifests, value >> 8) {
                    toggle_lease(&connection, &tenant, "manifest", manifest, step);
                }
            }
            3 => toggle_upload(&connection, &tenant),
            4 => {
                let changed = store.evict_manifest_one().unwrap();
                let after = model_snapshot(&connection, &tenant);
                let removed: BTreeSet<_> = before
                    .manifests
                    .difference(&after.manifests)
                    .copied()
                    .collect();
                assert_eq!(changed, !before.eligible_manifests.is_empty());
                assert_eq!(removed.len(), usize::from(changed));
                assert!(removed.is_subset(&before.eligible_manifests));
                evicted.extend(removed);
            }
            5 => {
                let changed = store.gc_one().unwrap();
                let after = model_snapshot(&connection, &tenant);
                let removed: BTreeSet<_> =
                    before.chunks.difference(&after.chunks).copied().collect();
                assert_eq!(changed, !before.eligible_chunks.is_empty());
                assert_eq!(removed.len(), usize::from(changed));
                assert!(removed.is_subset(&before.eligible_chunks));
            }
            6 if before.manifests.len() < 12 => {
                publish(Arc::clone(&store), next_publication);
                next_publication += 1;
            }
            7 => {
                if let Some(chunk) = choose(&before.chunks, value >> 8) {
                    store.record_chunk_access(chunk, 128, 1_000_000).unwrap();
                    store.flush_access_epochs().unwrap();
                }
            }
            _ => {}
        }
        if step > 0 && step % 50 == 0 {
            drop(connection);
            drop(store);
            store = Arc::new(
                LocalStore::open(
                    configuration.clone(),
                    kvpack::load_store_key(&key_path, temp.path()).unwrap(),
                )
                .unwrap(),
            );
            connection = Connection::open(&configuration.catalog_path).unwrap();
            restarts += 1;
        }
        assert_invariants(&store, &connection, &tenant, &evicted);
    }
    assert!(operation_counts.into_iter().all(|count| count > 0));
    assert_eq!(restarts, 5);
    assert!(!evicted.is_empty());
}
