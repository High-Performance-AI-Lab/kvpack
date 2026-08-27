use super::common::*;

#[test]
fn exact_manifest_scatter_transfer_is_pinned_read_only_and_fully_bound() {
    let fixture = fixture(b"restore-scatter-transfer");
    let (_, child) = publish_delta_chain(Arc::clone(&fixture.store));
    let plan = AuthenticatedRestorePlan::build_exact_manifest(
        Arc::clone(&fixture.store),
        child.manifest_id,
        1,
        RestoreLimits::default(),
        &ValidationContext::default(),
    )
    .unwrap();
    assert_eq!(plan.semantic_model(), &semantic());
    assert_eq!(plan.family(), &family());
    assert_eq!(plan.matched_cut().token_count, 512);
    assert_eq!(plan.key_epoch(), 1);
    assert_eq!(
        plan.resources().shadow_bytes,
        plan.realized_schema().complete_restored_bytes
    );
    let attempt = id(117);
    let expected_pin_ids = plan.scatter_pin_ids(attempt).unwrap();
    assert_ne!(expected_pin_ids, plan.scatter_pin_ids(id(118)).unwrap());
    let transfer = plan.prepare_scatter_transfer(attempt).unwrap();

    assert_eq!(transfer.manifest_id(), child.manifest_id);
    assert_eq!(transfer.attempt(), attempt);
    assert_eq!(transfer.resources(), plan.resources());
    assert_eq!(transfer.batches().len(), 1);
    let batch = transfer.batch(0).unwrap();
    assert_eq!(batch.batch_number(), 0);
    assert_eq!(batch.total_batches(), 1);
    assert_eq!(batch.descriptors().len(), 2);
    assert_eq!(batch.files().len(), 2);
    assert_eq!(batch.pin_ids().unwrap().len(), 2);
    assert_eq!(transfer.pin_ids().unwrap(), expected_pin_ids);
    assert_eq!(fixture.store.stat().unwrap().pins, 2);

    for (index, (descriptor, file)) in batch.descriptors().iter().zip(batch.files()).enumerate() {
        assert!(descriptor.verify_digest());
        assert_eq!(descriptor.manifest_id, child.manifest_id);
        assert_eq!(descriptor.chunk_ordinal, index as u64);
        assert_eq!(descriptor.fd_index, index as u32);
        assert_eq!(descriptor.fd_offset, 0);
        assert_eq!(descriptor.fd_bytes, descriptor.object_bytes);
        assert_eq!(descriptor.target_bytes, descriptor.plaintext_bytes);
        assert_eq!(descriptor.atomic_group, 1);
        assert_eq!(descriptor.attempt, attempt);
        let status = rustix::fs::fcntl_getfl(file).unwrap();
        assert_eq!(
            status & rustix::fs::OFlags::RWMODE,
            rustix::fs::OFlags::RDONLY
        );
        assert!(rustix::io::fcntl_getfd(file)
            .unwrap()
            .contains(rustix::io::FdFlags::CLOEXEC));
    }

    let original = batch.descriptors()[0].clone();
    let invalid = |descriptor: kvpack::AuthenticatedScatterDescriptor| {
        assert!(!descriptor.verify_digest());
    };
    let mut changed = original.clone();
    changed.manifest_id[0] ^= 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.state_key.layer ^= 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.state_key.state_name.push('x');
    invalid(changed);
    let mut changed = original.clone();
    changed.chunk_ordinal += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.batch_number += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.fd_index += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.fd_offset += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.fd_bytes += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.object_key[0] ^= 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.object_digest[0] ^= 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.object_bytes += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.plaintext_bytes += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.key_epoch += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.target_offset += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.target_bytes += 1;
    invalid(changed);
    let mut changed = original.clone();
    changed.atomic_group += 1;
    invalid(changed);
    let mut changed = original;
    changed.attempt[0] ^= 1;
    invalid(changed);

    drop(transfer);
    assert_eq!(fixture.store.stat().unwrap().pins, 0);
}
