#[cfg(test)]
mod tests {
    use super::super::{RetentionInputs, UploadReservation};
    use super::*;
    use crate::{create_store_key_random, load_store_key, StoreConfig};
    use std::os::unix::fs::PermissionsExt;

    struct Fixture {
        temp: tempfile::TempDir,
        config: StoreConfig,
        key_path: PathBuf,
        store: LocalStore,
    }

    fn fixture() -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        create_store_key_random(&key_path, temp.path()).unwrap();
        let config = StoreConfig {
            object_root: temp.path().join("objects"),
            catalog_path: temp.path().join("catalog/catalog.sqlite"),
            operator_tenant_id: b"audit-tenant-canary".to_vec(),
            key_epoch: 1,
            minimum_readable_key_epoch: 1,
            catalog_epoch: 1,
            quota_bytes: 1 << 30,
            staging_quota_bytes: 1 << 30,
            endurance_bytes_per_five_minutes: 1 << 30,
        };
        let store = LocalStore::open(
            config.clone(),
            load_store_key(&key_path, temp.path()).unwrap(),
        )
        .unwrap();
        Fixture {
            temp,
            config,
            key_path,
            store,
        }
    }

    fn event(sequence: u64) -> AuditEventKey {
        let mut object_id = [0u8; 32];
        object_id[..8].copy_from_slice(&sequence.to_be_bytes());
        object_id[8] = 1;
        AuditEventKey::new(
            AuditEventKind::Published,
            AuditObjectKind::Manifest,
            object_id,
            1,
        )
    }

    fn append(store: &LocalStore, events: &[AuditEventKey]) -> u64 {
        let mut connection = store.lock_catalog().unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        assert_eq!(
            preflight_events(&transaction, &store.tenant_namespace, events).unwrap(),
            AuditCapacity::Ready
        );
        let inserted = append_events(&transaction, &store.tenant_namespace, events, 1).unwrap();
        transaction.commit().unwrap();
        inserted
    }

    struct FailingExporter;

    impl AuditExporter for FailingExporter {
        fn export(&self, _batch: &AuditBatch) -> Result<(), StoreError> {
            Err(StoreError::Busy)
        }
    }

    struct NoopExporter;

    impl AuditExporter for NoopExporter {
        fn export(&self, _batch: &AuditBatch) -> Result<(), StoreError> {
            Ok(())
        }
    }

    #[test]
    fn durable_batches_retry_exactly_and_publish_private_rotated_segments() {
        let Fixture {
            temp,
            config,
            key_path,
            store,
        } = fixture();
        assert_eq!(append(&store, &[event(1), event(2), event(3)]), 3);
        let before = store.audit_status().unwrap();
        assert_eq!(before.pending_records, 3);
        assert_eq!(before.next_sequence, 4);
        assert_eq!(before.lost_records, 0);
        drop(store);

        let store =
            LocalStore::open(config, load_store_key(&key_path, temp.path()).unwrap()).unwrap();
        assert!(matches!(
            store.export_audit_batch(&FailingExporter, 3),
            Err(StoreError::Busy)
        ));
        let failed = store.audit_status().unwrap();
        assert_eq!(failed.pending_records, 3);
        assert_eq!(failed.delivery_failures, 1);

        let directory = temp.path().join("operator-audit");
        let policy =
            AuditDirectoryPolicy::new(MIN_AUDIT_SEGMENT_BYTES, 2, Duration::from_secs(60 * 60))
                .unwrap();
        let exporter = AuditDirectoryExporter::new(directory.clone(), policy).unwrap();
        let batch = {
            let connection = store.lock_catalog().unwrap();
            load_batch(&connection, &store.tenant_namespace, 3).unwrap()
        };
        exporter.export(&batch).unwrap();
        exporter.export(&batch).unwrap();
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);

        let report = store.export_audit_batch(&exporter, 3).unwrap();
        assert_eq!(report.exported_records, 3);
        assert_eq!(report.status.pending_records, 0);
        assert_eq!(report.status.retained_delivered_records, 3);
        assert_eq!(report.status.lost_records, 0);

        for sequence in [4, 5] {
            let record = AuditRecord {
                sequence,
                event: AuditEventKind::Collected,
                object: AuditObjectKind::Chunk,
                object_id: [sequence as u8; 32],
                generation: 1,
                occurred_unix_ns: sequence,
            };
            exporter
                .export(&AuditBatch {
                    records: vec![record],
                })
                .unwrap();
        }
        let entries = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        for entry in entries {
            assert_eq!(
                entry.metadata().unwrap().permissions().mode() & 0o777,
                0o600
            );
            let text = fs::read_to_string(entry.path()).unwrap();
            assert!(!text.contains("audit-tenant-canary"));
            assert!(!text.contains("prompt-canary"));
        }
    }

    #[test]
    fn bounded_delivery_retention_counts_pruned_records_without_loss() {
        let fixture = fixture();
        let events = (1..=MAX_RETAINED_DELIVERED_AUDIT_RECORDS + 4)
            .map(event)
            .collect::<Vec<_>>();
        assert_eq!(append(&fixture.store, &events), events.len() as u64);
        let mut total_exported = 0u64;
        while fixture.store.audit_status().unwrap().pending_records > 0 {
            total_exported += fixture
                .store
                .export_audit_batch(&NoopExporter, MAX_AUDIT_BATCH_RECORDS)
                .unwrap()
                .exported_records;
        }
        assert_eq!(total_exported, events.len() as u64);
        let status = fixture.store.audit_status().unwrap();
        assert_eq!(
            status.retained_delivered_records,
            MAX_RETAINED_DELIVERED_AUDIT_RECORDS
        );
        assert_eq!(status.retention_pruned_records, 4);
        assert_eq!(status.lost_records, 0);
        let metrics = fixture.store.prometheus_metrics().unwrap();
        assert!(metrics.contains("kvpack_audit_total{outcome=\"retention_pruned\"} 4"));
    }

    #[test]
    fn full_pending_queue_commits_backpressure_counter_before_cache_mutation() {
        let fixture = fixture();
        let mut connection = fixture.store.lock_catalog().unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        transaction
            .execute_batch(&format!(
                "WITH RECURSIVE seq(value) AS (VALUES(1) UNION ALL SELECT value+1 FROM seq WHERE value<{}) INSERT INTO audit_outbox(tenant,stream,sequence,event_kind,object_kind,object_id,generation,occurred_ns) SELECT X'{}','publication',value,'reserved','upload',CAST(printf('%032d',value) AS BLOB),0,1 FROM seq; UPDATE audit_state SET next_sequence={} WHERE tenant=X'{}' AND stream='publication';",
                MAX_PENDING_AUDIT_RECORDS,
                hex(&fixture.store.tenant_namespace),
                MAX_PENDING_AUDIT_RECORDS + 1,
                hex(&fixture.store.tenant_namespace),
            ))
            .unwrap();
        transaction.commit().unwrap();
        drop(connection);
        let result = fixture.store.reserve_upload(
            &[0xf1; 32],
            UploadReservation {
                expected_bytes: 1,
                publication_generation: 1,
                intent_digest: [0xf2; 32],
                retention: RetentionInputs::conservative(1, 1),
            },
        );
        assert!(matches!(result, Err(StoreError::Busy)));
        let status = fixture.store.audit_status().unwrap();
        assert_eq!(status.pending_records, MAX_PENDING_AUDIT_RECORDS);
        assert_eq!(status.backpressure_events, 1);
        assert_eq!(status.lost_records, 0);
        let stat = fixture.store.stat().unwrap();
        assert_eq!(stat.reserved_bytes, 0);
        let connection = fixture.store.lock_catalog().unwrap();
        let uploads: u64 = connection
            .query_row("SELECT COUNT(*) FROM uploads", [], |row| row.get(0))
            .unwrap();
        assert_eq!(uploads, 0);
    }

    #[test]
    fn public_debug_output_redacts_object_identities() {
        let record = AuditRecord {
            sequence: 1,
            event: AuditEventKind::Published,
            object: AuditObjectKind::Manifest,
            object_id: [0xab; 32],
            generation: 1,
            occurred_unix_ns: 1,
        };
        let debug = format!("{record:?}");
        assert!(debug.contains("[opaque]"));
        assert!(!debug.contains("abababab"));
    }
}
