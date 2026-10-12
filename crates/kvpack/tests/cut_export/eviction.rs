use super::common::*;

#[test]
fn leaf_delta_eviction_keeps_its_parent_and_releases_only_delta_chunks() {
    let fixture = fixture(b"marginal-physical-eviction");
    let family = family(Codec::Raw, &["k"], 8);
    let mut session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 300),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(63), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = Cursor::new(vec![7; 300 * 8]);
    session
        .next_state(StateKey::new(0, "k"))
        .unwrap()
        .write_source(&mut source)
        .unwrap();
    let cuts = session.commit().unwrap();
    assert_eq!(cuts.checkpoints.len(), 1);
    let checkpoint = cuts.checkpoints[0].manifest_id;
    let exact_final = cuts.exact_final.manifest_id;
    fixture.store.flush_access_epochs().unwrap();

    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT mc.object_key,c.refcount FROM manifest_chunks mc \
             JOIN chunks c ON c.tenant=mc.tenant AND c.object_key=mc.object_key \
             WHERE mc.tenant=?1 AND mc.manifest_id=?2 ORDER BY mc.ordinal",
        )
        .unwrap();
    let chunks = statement
        .query_map(
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                exact_final.as_slice()
            ],
            |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, u64>(1)?)),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(chunks.len(), 1);
    let unique = chunks[0].0.clone();
    assert_eq!(chunks[0].1, 1);
    let parent_chunk: Vec<u8> = connection
        .query_row(
            "SELECT mc.object_key FROM manifest_chunks mc \
             WHERE mc.tenant=?1 AND mc.manifest_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                checkpoint.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute(
            "UPDATE policy_objects SET score=0 WHERE tenant=?1",
            [fixture.store.tenant_namespace().as_slice()],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE policy_objects SET score=1000000 WHERE tenant=?1 AND object_key=?2",
            rusqlite::params![fixture.store.tenant_namespace().as_slice(), parent_chunk],
        )
        .unwrap();
    drop(statement);
    drop(connection);

    assert!(fixture.store.evict_manifest_one().unwrap());
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let present = |manifest_id: &[u8]| -> u64 {
        connection
            .query_row(
                "SELECT COUNT(*) FROM manifests WHERE tenant=?1 AND manifest_id=?2",
                rusqlite::params![fixture.store.tenant_namespace().as_slice(), manifest_id],
                |row| row.get(0),
            )
            .unwrap()
    };
    assert_eq!(present(&checkpoint), 1);
    assert_eq!(present(&exact_final), 0);
    let (parent_refs, unique_refs): (u64, u64) = (
        connection
            .query_row(
                "SELECT refcount FROM chunks WHERE tenant=?1 AND object_key=?2",
                rusqlite::params![fixture.store.tenant_namespace().as_slice(), parent_chunk],
                |row| row.get(0),
            )
            .unwrap(),
        connection
            .query_row(
                "SELECT refcount FROM chunks WHERE tenant=?1 AND object_key=?2",
                rusqlite::params![fixture.store.tenant_namespace().as_slice(), unique],
                |row| row.get(0),
            )
            .unwrap(),
    );
    assert_eq!((parent_refs, unique_refs), (1, 0));
    drop(connection);
    assert!(fixture.store.gc_one().unwrap());
}
