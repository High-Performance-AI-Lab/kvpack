use super::common::*;

#[test]
fn published_replay_requires_identical_source_bytes_and_does_not_rewrite() {
    let fixture = fixture(b"exact-cut-replay");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let first = export_two_states(
        Arc::clone(&fixture.store),
        family.clone(),
        300,
        id(42),
        1,
        2,
    );
    let before = fixture.store.stat().unwrap();
    let replay = export_two_states(
        Arc::clone(&fixture.store),
        family.clone(),
        300,
        id(42),
        1,
        2,
    );
    assert_eq!(replay, first);
    assert_eq!(fixture.store.stat().unwrap(), before);

    let mut changed = ExportSession::begin(
        Arc::clone(&fixture.store),
        declaration(family.clone(), 300),
        ExportCutPolicy::production_v1(),
        WritePolicy::exact_qualified(id(42), semantic(), &family).unwrap(),
    )
    .unwrap();
    let mut source = Cursor::new(vec![3; 300 * 8]);
    assert!(matches!(
        changed
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source),
        Err(StoreError::Authentication(_))
    ));
    assert!(matches!(changed.commit(), Err(StoreError::Poisoned(_))));
    assert_eq!(fixture.store.stat().unwrap(), before);
}

#[test]
fn active_export_retry_fences_generation_bounds_and_declaration() {
    let fixture = fixture(b"export-intent-fence");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let declared = declaration(family.clone(), 300);
    let policy = WritePolicy::exact_qualified(id(48), semantic(), &family)
        .unwrap()
        .with_publication_generation(7)
        .unwrap();
    let mut session = ExportSession::begin(
        Arc::clone(&fixture.store),
        declared.clone(),
        ExportCutPolicy::production_v1(),
        policy,
    )
    .unwrap();

    let changed_generation = WritePolicy::exact_qualified(id(48), semantic(), &family)
        .unwrap()
        .with_publication_generation(8)
        .unwrap();
    assert!(matches!(
        ExportSession::begin(
            Arc::clone(&fixture.store),
            declared.clone(),
            ExportCutPolicy::production_v1(),
            changed_generation,
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different publication generation"
        ))
    ));

    let mut changed_declaration = declared.clone();
    changed_declaration.states[0].atomic_group = 2;
    assert!(matches!(
        ExportSession::begin(
            Arc::clone(&fixture.store),
            changed_declaration,
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(48), semantic(), &family)
                .unwrap()
                .with_publication_generation(7)
                .unwrap(),
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different immutable declaration"
        ))
    ));

    assert!(matches!(
        ExportSession::begin(
            Arc::clone(&fixture.store),
            declaration(family.clone(), 301),
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(id(48), semantic(), &family)
                .unwrap()
                .with_publication_generation(7)
                .unwrap(),
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with different write bounds"
        ))
    ));

    for (key, value) in [("k", 1), ("v", 2)] {
        let mut source = Cursor::new(vec![value; 300 * 8]);
        session
            .next_state(StateKey::new(0, key))
            .unwrap()
            .write_source(&mut source)
            .unwrap();
    }
    let cuts = session.commit().unwrap();
    assert_eq!(cuts.exact_final.publication_generation, 7);
    assert!(cuts
        .checkpoints
        .iter()
        .all(|cut| cut.publication_generation == 7));
}
