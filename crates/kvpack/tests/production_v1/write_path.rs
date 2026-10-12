use super::common::*;

#[test]
fn dropped_state_permanently_poisons_writer() {
    let fixture = fixture(b"tenant-a");
    let declaration = declaration(Codec::Raw);
    let family = declaration.family.clone();
    let mut writer = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration,
        WritePolicy::exact_qualified(id(99), semantic(), &family).unwrap(),
    )
    .unwrap();
    {
        let mut state = writer.next_state(StateKey::new(0, "k")).unwrap();
        state.write_all(&[1; 3]).unwrap();
    }
    assert!(matches!(
        writer.next_state(StateKey::new(0, "k")),
        Err(StoreError::Poisoned(_))
    ));
    assert!(matches!(writer.commit(), Err(StoreError::Poisoned(_))));
}

#[test]
fn malformed_state_declaration_is_rejected_before_streaming() {
    let fixture = fixture(b"tenant-malformed-declaration");
    let mut declaration = declaration(Codec::Raw);
    let family = declaration.family.clone();
    declaration.states[0].strides.clear();
    assert!(matches!(
        ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            declaration,
            WritePolicy::exact_qualified(id(98), semantic(), &family).unwrap(),
        ),
        Err(StoreError::State(
            "state declaration rank, strides, token axis, or atomic group is invalid"
        ))
    ));
}

#[test]
fn exact_delta_parent_is_authenticated_before_streaming() {
    let fixture = fixture(b"tenant-delta-parent");
    let (parent, family, _, parent_cut) = publish(Arc::clone(&fixture.store), Codec::Raw);
    let mut child = declaration(Codec::Raw);
    child.input_tokens.extend([50, 60]);
    child.kind = ManifestKind::Delta {
        parent: parent.manifest_id,
        parent_cut,
        depth: 99,
    };
    for state in &mut child.states {
        state.full_shape = Shape::new(&[6, 8]).unwrap();
        state.segment_shape = Shape::new(&[2, 8]).unwrap();
        state.logical_start = 4;
        state.logical_count = 2;
        state.absolute_position = 6;
    }
    let mut writer = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        child.clone(),
        WritePolicy::exact_qualified(id(11), semantic(), &family).unwrap(),
    )
    .unwrap();
    for (name, byte) in [("k", 33), ("v", 44)] {
        let mut state = writer.next_state(StateKey::new(0, name)).unwrap();
        state.write_all(&[byte; 16]).unwrap();
        state.finish().unwrap();
    }
    writer.commit().unwrap();
    assert_eq!(fixture.store.stat().unwrap().manifests, 2);

    if let ManifestKind::Delta { parent_cut, .. } = &mut child.kind {
        parent_cut.token_root = id(88);
    }
    assert!(matches!(
        ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            child,
            WritePolicy::exact_qualified(id(12), semantic(), &family).unwrap(),
        ),
        Err(StoreError::Expectation(
            "delta parent cut does not identify the exact child prefix"
        ))
    ));
    assert_eq!(fixture.store.stat().unwrap().manifests, 2);
}

#[test]
fn development_magic_is_rejected() {
    let mut bytes = vec![0u8; 8192];
    bytes[..8].copy_from_slice(b"IOKVPK1\0");
    let cursor = std::io::Cursor::new(bytes);
    assert!(matches!(
        kvpack::inspect_untrusted(cursor, kvpack::InspectionBounds::default()),
        Err(kvpack::PackError::BadMagic(_))
    ));
}

#[test]
fn published_idempotency_retry_is_exact_and_does_not_rewrite() {
    let fixture = fixture(b"tenant-idempotent");
    let (published, family, _, _) = publish(Arc::clone(&fixture.store), Codec::Raw);
    assert_eq!(published.publication_generation, 1);
    let declaration = declaration(Codec::Raw);
    let policy = WritePolicy::exact_qualified(id(10), semantic(), &family).unwrap();
    let writer = ArtifactWriter::begin(
        Arc::clone(&fixture.store),
        declaration.clone(),
        policy.clone(),
    )
    .unwrap();
    assert_eq!(writer.published_replay(), Some(published));
    assert_eq!(writer.commit().unwrap(), published);
    assert_eq!(fixture.store.stat().unwrap().manifests, 1);

    let generation_conflict = WritePolicy::exact_qualified(id(10), semantic(), &family)
        .unwrap()
        .with_publication_generation(2)
        .unwrap();
    assert!(matches!(
        ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            declaration.clone(),
            generation_conflict,
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different publication generation"
        ))
    ));

    let bound_conflict = WritePolicy::exact_qualified(id(10), semantic(), &family)
        .unwrap()
        .with_maximum_restored_bytes((4 * 1024 * 1024 * 1024 * 1024) - 1)
        .unwrap();
    assert!(matches!(
        ArtifactWriter::begin(
            Arc::clone(&fixture.store),
            declaration.clone(),
            bound_conflict,
        ),
        Err(StoreError::Expectation(
            "idempotency key was reused with a different immutable declaration"
        ))
    ));

    let mut conflicting = declaration;
    conflicting.input_tokens[3] = 41;
    assert!(matches!(
        ArtifactWriter::begin(Arc::clone(&fixture.store), conflicting, policy),
        Err(StoreError::Expectation(_))
    ));
    let connection =
        rusqlite::Connection::open(fixture._temp.path().join("catalog/catalog.sqlite")).unwrap();
    let generation: u64 = connection
        .query_row(
            "SELECT generation FROM manifests WHERE tenant=?1 AND manifest_id=?2",
            rusqlite::params![
                fixture.store.tenant_namespace().as_slice(),
                published.manifest_id.as_slice()
            ],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(generation, 1);
}
