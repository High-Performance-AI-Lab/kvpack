use super::common::*;

#[test]
fn publication_audit_is_atomic_ordered_and_idempotent() {
    let fixture = fixture(b"publication-audit");
    let family = family(Codec::Raw, &["k"], 8);
    let write_once = || {
        let mut session = ExportSession::begin(
            Arc::clone(&fixture.store),
            declaration(family.clone(), 300),
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(66), semantic(), &family).unwrap(),
        )
        .unwrap();
        let mut source = Cursor::new(vec![7; 2_400]);
        session
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source)
            .unwrap();
        session.commit().unwrap()
    };
    let first = write_once();
    let status = fixture.store.audit_status().unwrap();
    assert_eq!(status.pending_records, 5);
    assert_eq!(status.lost_records, 0);

    let replay = write_once();
    assert_eq!(replay, first);
    assert_eq!(fixture.store.audit_status().unwrap().pending_records, 5);

    let exporter = CapturedAudit::default();
    let report = fixture.store.export_audit_batch(&exporter, 16).unwrap();
    assert_eq!(report.exported_records, 5);
    let records = exporter.0.lock().unwrap();
    assert_eq!(first.checkpoints.len(), 1);
    assert_eq!(
        records.as_slice(),
        [
            (1, AuditEventKind::Reserved, AuditObjectKind::Upload, id(66),),
            (
                2,
                AuditEventKind::Receiving,
                AuditObjectKind::Upload,
                id(66),
            ),
            (3, AuditEventKind::Verified, AuditObjectKind::Upload, id(66),),
            (
                4,
                AuditEventKind::Published,
                AuditObjectKind::Manifest,
                first.checkpoints[0].manifest_id,
            ),
            (
                5,
                AuditEventKind::Published,
                AuditObjectKind::Manifest,
                first.exact_final.manifest_id,
            ),
        ]
    );
}

#[test]
fn cancellation_releases_the_reservation_and_reinit_requires_identity() {
    let fixture = fixture(b"cancelled-cut-export");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 300),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(47), semantic(), &family).unwrap(),
    )
    .unwrap();
    assert!(fixture.store.stat().unwrap().reserved_bytes > 0);
    session.cancel().unwrap();
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);

    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let (state, reserved): (String, u64) = connection
        .query_row(
            "SELECT state,reserved_bytes FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                id(47).as_slice()
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(state, "ABORTED");
    assert_eq!(reserved, 0);

    // RE_INIT (docs/UPLOAD_REINIT_DESIGN.md): a byte-identical declaration
    // re-reserves the burned key; a mutated declaration fails closed.
    let reinitialized = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 300),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(47), semantic(), &family).unwrap(),
    )
    .unwrap();
    assert!(fixture.store.stat().unwrap().reserved_bytes > 0);
    reinitialized.cancel().unwrap();

    let mut mutated = declaration(family.clone(), 300);
    mutated.input_tokens.push(301);
    assert!(matches!(
        ExportSession::begin(
            Arc::clone(&fixture.store),
            mutated,
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(47), semantic(), &family).unwrap(),
        ),
        Err(StoreError::Expectation(_))
    ));
}
