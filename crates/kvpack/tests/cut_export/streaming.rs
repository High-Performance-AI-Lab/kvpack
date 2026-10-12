use super::common::*;

#[test]
fn chunking_honors_four_mib_and_checkpoint_boundaries() {
    let fixture = fixture(b"bounded-export-chunks");
    let width = 32 * 1024;
    let family = family(Codec::Raw, &["k"], width);
    let mut session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 300),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(43), semantic(), &family).unwrap(),
    )
    .unwrap();
    let expected = 300 * width as usize;
    let mut source = CountingReader::new(vec![5; expected], 17 * 1024);
    session
        .next_state(StateKey::new(0, "k"))
        .unwrap()
        .write_source(&mut source)
        .unwrap();
    assert_eq!(source.returned, expected);
    let cuts = session.commit().unwrap();
    assert_eq!(cuts.checkpoints.len(), 1);
    assert_eq!(fixture.store.stat().unwrap().chunks, 3);
    let final_schema = &cuts.exact_final.realized_schema.states[0];
    assert_eq!(
        final_schema
            .chunk_spans
            .iter()
            .map(|span| (span.token_start, span.token_count))
            .collect::<Vec<_>>(),
        [(256, 44)]
    );
    assert_eq!(
        cuts.checkpoints[0].realized_schema.states[0]
            .chunk_spans
            .iter()
            .map(|span| (span.token_start, span.token_count))
            .collect::<Vec<_>>(),
        [(0, 128), (128, 128)]
    );
}

#[test]
fn out_of_order_drop_short_and_extra_sources_poison_the_session() {
    let fixture = fixture(b"poisoned-cut-export");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let policy = WritePolicy::exact_qualified(id(44), semantic(), &family).unwrap();
    let mut session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        policy,
    )
    .unwrap();
    assert!(matches!(
        session.next_state(StateKey::new(0, "v")),
        Err(StoreError::State(_))
    ));
    assert!(matches!(session.commit(), Err(StoreError::Poisoned(_))));

    let mut short = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(45), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = Cursor::new(vec![1; 31]);
    assert!(matches!(
        short
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source),
        Err(StoreError::State(_))
    ));
    assert!(matches!(short.commit(), Err(StoreError::Poisoned(_))));

    let mut extra = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(46), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = Cursor::new(vec![1; 33]);
    assert!(matches!(
        extra
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source),
        Err(StoreError::State(_))
    ));
    assert!(matches!(extra.commit(), Err(StoreError::Poisoned(_))));

    let mut read_error = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(49), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = FailingReader::new(vec![1; 32], 16);
    assert!(matches!(
        read_error
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source),
        Err(StoreError::Io {
            op: "read export state source",
            ..
        })
    ));
    assert!(matches!(read_error.commit(), Err(StoreError::Poisoned(_))));
}

#[test]
fn duplicate_declaration_is_rejected_before_reservation() {
    let fixture = fixture(b"duplicate-export-declaration");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let mut declared = declaration(family.clone(), 4);
    declared.states[1].key = declared.states[0].key.clone();
    assert!(matches!(
        ExportSession::begin(
            Arc::clone(&fixture.store),
            declared,
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(50), semantic(), &family).unwrap(),
        ),
        Err(StoreError::State(
            "export state order, strides, or atomic group is invalid"
        ))
    ));
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let uploads: u64 = connection
        .query_row("SELECT COUNT(*) FROM uploads", [], |row| row.get(0))
        .unwrap();
    assert_eq!(uploads, 0);
}

#[test]
fn failed_catalog_publication_aborts_the_complete_export() {
    let fixture = fixture(b"failed-export-publication");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let mut session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(51), semantic(), &family).unwrap(),
    )
    .unwrap();
    for (name, value) in [("k", 1), ("v", 2)] {
        let mut source = Cursor::new(vec![value; 32]);
        session
            .next_state(StateKey::new(0, name))
            .unwrap()
            .write_source(&mut source)
            .unwrap();
    }
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER inject_prefix_publication_failure BEFORE INSERT ON prefix_checkpoints BEGIN SELECT RAISE(ABORT,'injected publication failure'); END;",
        )
        .unwrap();
    drop(connection);

    assert!(matches!(session.commit(), Err(StoreError::Catalog(_))));
    let stat = fixture.store.stat().unwrap();
    assert_eq!(stat.reserved_bytes, 0);
    assert_eq!(stat.durable_bytes, 0);
    assert_eq!(stat.manifests, 0);

    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let (state, reserved, prefixes): (String, u64, u64) = connection
        .query_row(
            "SELECT u.state,u.reserved_bytes,(SELECT COUNT(*) FROM prefix_checkpoints) FROM uploads u WHERE u.tenant=?1 AND u.idempotency_key=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                id(51).as_slice()
            ],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(state, "ABORTED");
    assert_eq!(reserved, 0);
    assert_eq!(prefixes, 0);
}
