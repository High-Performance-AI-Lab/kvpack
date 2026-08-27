use super::common::*;

#[test]
fn only_the_exact_cut_is_published_to_the_prefix_catalog() {
    let fixture = fixture(b"tenant-exact-cut");
    let (_, family, _, input_cut) = publish(Arc::clone(&fixture.store), Codec::Raw);
    let (_, nodes) = fixture
        .store
        .derive_input_cut(&semantic(), &family, &[10, 20, 30, 40], &auxiliary())
        .unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, input_cut.token_root);
    let family_id = kvpack::wire::representation_family_id(&family).unwrap();
    let semantic_id = kvpack::wire::semantic_model_id(&semantic());
    assert_eq!(
        fixture
            .store
            .resolve_prefix(&nodes, &semantic_id, &family_id, 8)
            .unwrap()
            .unwrap()
            .token_count,
        4
    );
}
