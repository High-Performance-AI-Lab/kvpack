use super::common::*;

#[test]
fn capacity_and_endurance_denial_precede_upload_and_source_bytes() {
    let family = family(Codec::Raw, &["k", "v"], 8);
    for (fixture, expected) in [
        (
            fixture_with_all_limits(b"capacity-denial", 1, 512 * 1024 * 1024, 512 * 1024 * 1024),
            "quota",
        ),
        (
            fixture_with_limits(b"endurance-denial", 512 * 1024 * 1024, 1),
            "endurance",
        ),
        (
            fixture_with_all_limits(b"staging-denial", 512 * 1024 * 1024, 1, 512 * 1024 * 1024),
            "staging",
        ),
    ] {
        let result = ExportSession::begin(
            Arc::clone(&fixture.store),
            declaration(family.clone(), 4),
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(52), semantic(), &family).unwrap(),
        );
        match (expected, result) {
            ("quota", Err(StoreError::Quota(message))) => {
                assert_eq!(message, "tenant durable quota reservation refused");
            }
            ("endurance", Err(StoreError::Endurance(message))) => {
                assert_eq!(message, "five-minute endurance budget exhausted");
            }
            ("staging", Err(StoreError::Quota(message))) => {
                assert_eq!(message, "tenant staging quota reservation refused");
            }
            _ => panic!("unexpected admission result"),
        }
        let stat = fixture.store.stat().unwrap();
        assert_eq!(stat.reserved_bytes, 0);
        assert_eq!(stat.chunks, 0);
        assert_eq!(stat.manifests, 0);
        let connection =
            rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite"))
                .unwrap();
        let uploads: u64 = connection
            .query_row("SELECT COUNT(*) FROM uploads", [], |row| row.get(0))
            .unwrap();
        assert_eq!(uploads, 0);
    }
}

#[test]
fn quarantine_backlog_larger_than_the_maintenance_bound_stops_new_writes() {
    let fixture = fixture_with_all_limits(
        b"bounded-quarantine-admission",
        1_000_000,
        1_000_000,
        1_000_000,
    );
    let mut connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let transaction = connection.transaction().unwrap();
    for ordinal in 0..=1024u64 {
        let mut entry_id = [0xabu8; 32];
        entry_id[..8].copy_from_slice(&ordinal.to_le_bytes());
        transaction
            .execute(
                "INSERT INTO quarantine_entries(tenant,entry_id,object_kind,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'test',?3,20000,?4,?5,'test backlog')",
                rusqlite::params![
                    fixture.store.tenant_namespace().as_slice(),
                    entry_id.as_slice(),
                    format!("missing-{ordinal}.quarantine"),
                    ordinal,
                    i64::MAX,
                ],
            )
            .unwrap();
    }
    transaction.commit().unwrap();
    drop(connection);

    let family = family(Codec::Raw, &["k"], 8);
    match ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(64), semantic(), &family).unwrap(),
    ) {
        Err(StoreError::Quota(message)) => {
            assert_eq!(message, "quarantine capacity requires bounded maintenance");
        }
        Err(error) => panic!("unexpected quarantine admission error: {error:?}"),
        Ok(_) => panic!("write unexpectedly bypassed quarantine maintenance bound"),
    }
    let stat = fixture.store.stat().unwrap();
    assert_eq!(stat.quarantine_bytes, 20_000);
    assert_eq!(stat.reserved_bytes, 0);
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let uploads: u64 = connection
        .query_row("SELECT COUNT(*) FROM uploads", [], |row| row.get(0))
        .unwrap();
    assert_eq!(uploads, 0);
}

#[test]
fn high_pressure_writer_admission_uses_tinylfu_value_before_reservation() {
    let fixture = fixture_with_limits(b"automatic-tinylfu-admission", 1_000_000, 1_000_000);
    let family = family(Codec::Raw, &["k", "v"], 8);
    fixture
        .store
        .record_chunk_access(id(60), 1_000, 1_000_000)
        .unwrap();
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    connection
        .execute(
            "UPDATE tenants SET durable_bytes=740000 WHERE namespace=?1",
            [fixture.store.tenant_namespace().as_slice()],
        )
        .unwrap();

    let low_value = RetentionInputs {
        predicted_reuse_millis: 1,
        avoided_prefill_ns: 1,
        ..RetentionInputs::default()
    };
    match ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(61), semantic(), &family)
            .unwrap()
            .with_retention(low_value)
            .unwrap(),
    ) {
        Err(StoreError::Quota(message)) => {
            assert_eq!(message, "TinyLFU rejected lower-value cache admission");
        }
        Err(error) => panic!("unexpected admission error: {error:?}"),
        Ok(_) => panic!("low-value admission unexpectedly succeeded"),
    }
    let low_uploads: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM uploads WHERE idempotency_key=?1",
            [id(61).as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(low_uploads, 0);

    let high_value = RetentionInputs {
        predicted_reuse_millis: 1000,
        avoided_prefill_ns: 1_000_000_000,
        ..RetentionInputs::default()
    };
    let admitted = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 4),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(62), semantic(), &family)
            .unwrap()
            .with_retention(high_value)
            .unwrap(),
    )
    .unwrap();
    assert!(fixture.store.stat().unwrap().reserved_bytes > 0);
    drop(admitted);
    assert_eq!(fixture.store.stat().unwrap().reserved_bytes, 0);
}
