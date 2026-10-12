use super::common::*;

#[test]
fn one_pass_export_publishes_independent_256_512_and_final_cuts() {
    let fixture = fixture(b"cut-aware-export");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let cuts = export_two_states(
        Arc::clone(&fixture.store),
        family.clone(),
        600,
        id(40),
        11,
        22,
    );
    assert_eq!(
        cuts.checkpoints
            .iter()
            .map(|cut| cut.input_cut.token_count)
            .collect::<Vec<_>>(),
        [256, 512]
    );
    assert_eq!(cuts.exact_final.input_cut.token_count, 600);
    assert_ne!(
        cuts.checkpoints[1].manifest_id,
        cuts.exact_final.manifest_id
    );
    let stat = fixture.store.stat().unwrap();
    assert_eq!(stat.manifests, 3);
    assert_eq!(
        stat.chunks, 6,
        "chunks are reused across full cut manifests"
    );
    assert!(fixture
        .store
        .prometheus_metrics()
        .unwrap()
        .contains("kvpack_bytes_total{kind=\"source_read\"} 9600"));

    for cut in cuts
        .checkpoints
        .iter()
        .chain(std::iter::once(&cuts.exact_final))
    {
        let sink = restore_cut(Arc::clone(&fixture.store), &family, cut);
        assert!(!sink.aborted);
        let state_bytes = cut.input_cut.token_count as usize * 8;
        assert_eq!(
            sink.installed[&StateKey::new(0, "k")],
            vec![11; state_bytes]
        );
        assert_eq!(
            sink.installed[&StateKey::new(0, "v")],
            vec![22; state_bytes]
        );
    }
}

#[test]
fn exact_final_reuses_the_checkpoint_when_aligned() {
    let fixture = fixture(b"aligned-cut-export");
    let family = family(Codec::Lossless, &["k", "v"], 8);
    let cuts = export_two_states(Arc::clone(&fixture.store), family, 512, id(41), 7, 9);
    assert_eq!(cuts.checkpoints.len(), 2);
    assert_eq!(cuts.exact_final, cuts.checkpoints[1]);
    assert_eq!(fixture.store.stat().unwrap().manifests, 2);
}

#[test]
fn checkpoint_export_compacts_after_seven_deltas_without_rewriting_chunks() {
    let fixture = fixture(b"checkpoint-delta-compaction");
    let family = family(Codec::Raw, &["k", "v"], 8);
    let cuts = export_two_states(
        Arc::clone(&fixture.store),
        family.clone(),
        9 * 256,
        id(65),
        3,
        4,
    );

    assert_eq!(cuts.checkpoints.len(), 9);
    assert_eq!(
        cuts.checkpoints
            .iter()
            .map(|cut| cut.realized_schema.kind.depth())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5, 6, 7, 0]
    );
    assert!(matches!(
        cuts.checkpoints[8].realized_schema.kind,
        ManifestKind::Full
    ));
    assert_eq!(fixture.store.stat().unwrap().chunks, 18);

    for cut in [&cuts.checkpoints[7], &cuts.checkpoints[8]] {
        let sink = restore_cut(Arc::clone(&fixture.store), &family, cut);
        assert_eq!(
            sink.installed[&StateKey::new(0, "k")].len(),
            cut.input_cut.token_count as usize * 8
        );
        assert_eq!(
            sink.installed[&StateKey::new(0, "v")].len(),
            cut.input_cut.token_count as usize * 8
        );
    }
}
