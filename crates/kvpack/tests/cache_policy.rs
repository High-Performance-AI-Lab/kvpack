use std::sync::Arc;

use kvpack::{
    access_flush_schedule, replay_cache_policy, replay_tinylfu, retention_value, AdmissionDecision,
    CachePressure, CacheReplayConfig, CacheReplayOutcome, CacheReplayPolicy, CacheReplayTier,
    CacheTraceEvent, LocalStore, PolicyReplayEvent, ReplayTierProfile, RetentionInputs,
    StoreConfig, TinyLfuConfig, UtilizationPolicy, CATALOG_SCHEMA_VERSION,
};

fn id(value: u8) -> [u8; 32] {
    [value; 32]
}

fn config(root: &std::path::Path, quota_bytes: u64) -> StoreConfig {
    StoreConfig {
        object_root: root.join("objects"),
        catalog_path: root.join("catalog/catalog.sqlite"),
        operator_tenant_id: b"cache-policy-tenant".to_vec(),
        key_epoch: 1,
        minimum_readable_key_epoch: 1,
        catalog_epoch: 1,
        quota_bytes,
        staging_quota_bytes: quota_bytes,
        endurance_bytes_per_five_minutes: quota_bytes,
    }
}

fn store(root: &std::path::Path, quota_bytes: u64) -> Arc<LocalStore> {
    let key_path = root.join("keys/root.key");
    kvpack::create_store_key_random(&key_path, root).unwrap();
    Arc::new(
        LocalStore::open(
            config(root, quota_bytes),
            kvpack::load_store_key(&key_path, root).unwrap(),
        )
        .unwrap(),
    )
}

fn replay_retention() -> RetentionInputs {
    RetentionInputs {
        predicted_reuse_millis: 900,
        avoided_prefill_ns: 20_000_000,
        best_restore_ns: 1_000,
        physical_bytes: 100,
        shared_ancestor_bytes: 0,
        write_wear_bytes: 10,
        promotion_bandwidth_bytes_per_second: 1_000_000_000,
        queue_interference_ns: 100,
    }
}

fn replay_config(capacity_bytes: u64, gateway_bytes_per_second: u64) -> CacheReplayConfig {
    CacheReplayConfig {
        capacity_bytes,
        maximum_entries: if capacity_bytes >= 800 { 8 } else { 3 },
        resident: ReplayTierProfile {
            bytes_per_second: 10_000_000_000,
            fixed_latency_ns: 10,
            initially_available: true,
        },
        local: ReplayTierProfile {
            bytes_per_second: 1_000_000_000,
            fixed_latency_ns: 100,
            initially_available: true,
        },
        gateway: ReplayTierProfile {
            bytes_per_second: gateway_bytes_per_second,
            fixed_latency_ns: 1_000,
            initially_available: true,
        },
    }
}

#[test]
fn tinylfu_replay_is_deterministic_and_protects_repeated_accesses() {
    let configuration = TinyLfuConfig {
        capacity_bytes: 1_000,
        sketch_width: 64,
        reset_after_accesses: 1_000,
        protected_fraction_millis: 800,
    };
    let events = [
        PolicyReplayEvent::Access {
            object_key: id(1),
            object_bytes: 100,
            avoided_recompute_ns: 1_000,
            now_ns: 1,
        },
        PolicyReplayEvent::Access {
            object_key: id(1),
            object_bytes: 100,
            avoided_recompute_ns: 1_000,
            now_ns: 2,
        },
        PolicyReplayEvent::Access {
            object_key: id(2),
            object_bytes: 100,
            avoided_recompute_ns: 1_000,
            now_ns: 3,
        },
        PolicyReplayEvent::Admission {
            object_key: id(3),
            object_bytes: 100,
            avoided_recompute_ns: 500,
            utilization_millis: 849,
        },
        PolicyReplayEvent::Admission {
            object_key: id(3),
            object_bytes: 100,
            avoided_recompute_ns: 500,
            utilization_millis: 900,
        },
        PolicyReplayEvent::Admission {
            object_key: id(3),
            object_bytes: 100,
            avoided_recompute_ns: 500,
            utilization_millis: 950,
        },
        PolicyReplayEvent::Evict,
    ];
    let first = replay_tinylfu(configuration, &events).unwrap();
    let second = replay_tinylfu(configuration, &events).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.admissions,
        [
            AdmissionDecision::Admit,
            AdmissionDecision::RejectLowerFrequency,
            AdmissionDecision::RejectPromotionStopped,
        ]
    );
    assert_eq!(first.victims, [Some(id(2))]);
}

#[test]
fn cache_trace_replay_compares_policies_capacity_bandwidth_and_failures() {
    let retention = replay_retention();
    let trace = vec![
        CacheTraceEvent::Access {
            object_key: id(1),
            source_tier: CacheReplayTier::Local,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(1),
            source_tier: CacheReplayTier::Resident,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(1),
            source_tier: CacheReplayTier::Resident,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(2),
            source_tier: CacheReplayTier::Gateway,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(3),
            source_tier: CacheReplayTier::Local,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(4),
            source_tier: CacheReplayTier::Local,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(4),
            source_tier: CacheReplayTier::Local,
            retention,
        },
        CacheTraceEvent::SourceFailed(CacheReplayTier::Gateway),
        CacheTraceEvent::Access {
            object_key: id(5),
            source_tier: CacheReplayTier::Gateway,
            retention,
        },
        CacheTraceEvent::SourceRecovered(CacheReplayTier::Gateway),
        CacheTraceEvent::Access {
            object_key: id(5),
            source_tier: CacheReplayTier::Gateway,
            retention,
        },
        CacheTraceEvent::Access {
            object_key: id(5),
            source_tier: CacheReplayTier::Gateway,
            retention,
        },
    ];
    let policies = [
        CacheReplayPolicy::Lru,
        CacheReplayPolicy::Lfu,
        CacheReplayPolicy::TinyLfu,
        CacheReplayPolicy::Arc,
        CacheReplayPolicy::ClockPro,
    ];
    let constrained = replay_config(450, 100_000_000);
    let roomy = replay_config(800, 100_000_000);
    let mut constrained_evictions = 0;
    let mut roomy_evictions = 0;
    for policy in policies {
        let first = replay_cache_policy(policy, constrained, &trace).unwrap();
        let second = replay_cache_policy(policy, constrained, &trace).unwrap();
        assert_eq!(first, second, "{policy:?} replay changed across runs");
        assert!(first.resident_bytes <= constrained.capacity_bytes);
        assert_eq!(first.source_failures, 1);
        assert!(first
            .outcomes
            .contains(&CacheReplayOutcome::SourceUnavailable));
        constrained_evictions += first.evictions.len();

        let roomy_result = replay_cache_policy(policy, roomy, &trace).unwrap();
        assert!(roomy_result.resident_bytes <= roomy.capacity_bytes);
        assert!(roomy_result.admitted >= first.admitted);
        roomy_evictions += roomy_result.evictions.len();
    }
    assert!(constrained_evictions > roomy_evictions);

    let slow = replay_cache_policy(CacheReplayPolicy::Lru, roomy, &trace).unwrap();
    let fast = replay_cache_policy(
        CacheReplayPolicy::Lru,
        replay_config(800, 1_000_000_000),
        &trace,
    )
    .unwrap();
    assert_eq!(slow.outcomes, fast.outcomes);
    assert!(fast.total_service_ns < slow.total_service_ns);

    let without_failure: Vec<_> = trace
        .iter()
        .copied()
        .filter(|event| {
            !matches!(
                event,
                CacheTraceEvent::SourceFailed(_) | CacheTraceEvent::SourceRecovered(_)
            )
        })
        .collect();
    let available = replay_cache_policy(CacheReplayPolicy::Lru, roomy, &without_failure).unwrap();
    assert_eq!(available.source_failures, 0);
    assert!(!available
        .outcomes
        .contains(&CacheReplayOutcome::SourceUnavailable));

    let invalid = CacheReplayConfig {
        gateway: ReplayTierProfile {
            bytes_per_second: 0,
            initially_available: false,
            ..roomy.gateway
        },
        ..roomy
    };
    assert!(matches!(
        replay_cache_policy(CacheReplayPolicy::Lru, invalid, &trace),
        Err(kvpack::StoreError::State(
            "cache replay configuration is invalid"
        ))
    ));
}

#[test]
fn physical_retention_value_accounts_for_every_cost_once() {
    let baseline = RetentionInputs {
        predicted_reuse_millis: 800,
        avoided_prefill_ns: 10_000_000,
        best_restore_ns: 1_000_000,
        physical_bytes: 10_000,
        shared_ancestor_bytes: 2_000,
        write_wear_bytes: 1_000,
        promotion_bandwidth_bytes_per_second: 1_000_000_000,
        queue_interference_ns: 500_000,
    };
    let score = retention_value(4, baseline).unwrap();
    assert!(score > 0);
    assert!(
        retention_value(
            4,
            RetentionInputs {
                predicted_reuse_millis: 400,
                ..baseline
            }
        )
        .unwrap()
            < score
    );
    assert!(
        retention_value(
            4,
            RetentionInputs {
                shared_ancestor_bytes: 6_000,
                ..baseline
            }
        )
        .unwrap()
            > score
    );
    assert!(
        retention_value(
            4,
            RetentionInputs {
                write_wear_bytes: 20_000,
                ..baseline
            }
        )
        .unwrap()
            < score
    );
    assert_eq!(
        retention_value(
            4,
            RetentionInputs {
                best_restore_ns: baseline.avoided_prefill_ns,
                ..baseline
            }
        )
        .unwrap(),
        0
    );
    assert!(matches!(
        retention_value(
            1,
            RetentionInputs {
                shared_ancestor_bytes: baseline.physical_bytes + 1,
                ..baseline
            }
        ),
        Err(kvpack::StoreError::State("retention inputs are invalid"))
    ));
}

#[test]
fn access_flush_retry_schedule_attempts_at_thirty_and_sixty_seconds() {
    let second = 1_000_000_000;
    let first_pending = 10 * second;
    let initial = access_flush_schedule(first_pending, None, 11 * second);
    assert_eq!(initial.next_attempt_ns, 40 * second);
    assert_eq!(initial.hard_deadline_ns, 70 * second);
    assert!(!initial.overdue);
    let retry = access_flush_schedule(first_pending, Some(40 * second), 41 * second);
    assert_eq!(retry.next_attempt_ns, 70 * second);
    assert_eq!(retry.hard_deadline_ns, 70 * second);
    assert!(!retry.overdue);
    let overdue = access_flush_schedule(first_pending, Some(70 * second), 71 * second);
    assert_eq!(overdue.next_attempt_ns, 101 * second);
    assert!(overdue.overdue);
}

#[test]
fn utilization_thresholds_are_exact_and_include_reservations() {
    let temp = tempfile::tempdir().unwrap();
    let store = store(temp.path(), 100);
    let connection =
        rusqlite::Connection::open(temp.path().join("catalog/catalog.sqlite")).unwrap();
    let set_usage = |durable: u64, reserved: u64| {
        connection
            .execute(
                "UPDATE tenants SET durable_bytes=?2,reserved_bytes=?3 WHERE namespace=?1",
                rusqlite::params![store.tenant_namespace().as_slice(), durable, reserved],
            )
            .unwrap();
    };
    let policy = UtilizationPolicy::default();
    set_usage(74, 10);
    assert_eq!(
        store.cache_pressure(100, policy).unwrap(),
        CachePressure::Normal
    );
    set_usage(75, 10);
    assert_eq!(
        store.cache_pressure(100, policy).unwrap(),
        CachePressure::Evicting
    );
    set_usage(82, 10);
    assert_eq!(
        store.cache_pressure(100, policy).unwrap(),
        CachePressure::Emergency
    );
    set_usage(85, 10);
    assert_eq!(
        store.cache_pressure(100, policy).unwrap(),
        CachePressure::PromotionStopped
    );
}

#[test]
fn fresh_catalog_records_every_ordered_migration() {
    let temp = tempfile::tempdir().unwrap();
    let _store = store(temp.path(), 1 << 20);
    let connection =
        rusqlite::Connection::open(temp.path().join("catalog/catalog.sqlite")).unwrap();
    let mut statement = connection
        .prepare("SELECT version FROM schema_migrations ORDER BY version")
        .unwrap();
    let versions = statement
        .query_map([], |row| row.get::<_, i64>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(versions, (1..=CATALOG_SCHEMA_VERSION).collect::<Vec<_>>());
}

#[test]
fn quarantine_pruning_is_catalog_driven_capacity_bounded_and_expiry_aware() {
    let temp = tempfile::tempdir().unwrap();
    let store = store(temp.path(), 1_000);
    let directory = temp.path().join("objects/quarantine");
    std::fs::write(directory.join("old.quarantine"), [1u8; 8]).unwrap();
    std::fs::write(directory.join("new.quarantine"), [2u8; 8]).unwrap();
    let connection =
        rusqlite::Connection::open(temp.path().join("catalog/catalog.sqlite")).unwrap();
    for (entry, token, created) in [
        (id(70), "old.quarantine", 1u64),
        (id(71), "new.quarantine", 2u64),
    ] {
        connection
            .execute(
                "INSERT INTO quarantine_entries(tenant,entry_id,object_kind,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'test',?3,8,?4,?5,'test quarantine')",
                rusqlite::params![
                    store.tenant_namespace().as_slice(),
                    entry.as_slice(),
                    token,
                    created,
                    i64::MAX,
                ],
            )
            .unwrap();
    }

    assert_eq!(
        store
            .prune_quarantine(1_000, UtilizationPolicy::default())
            .unwrap(),
        8
    );
    assert!(!directory.join("old.quarantine").exists());
    assert!(directory.join("new.quarantine").exists());
    assert_eq!(store.stat().unwrap().quarantine_bytes, 8);
    let rows: u64 = connection
        .query_row("SELECT COUNT(*) FROM quarantine_entries", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(rows, 1);

    connection
        .execute("UPDATE quarantine_entries SET expires_ns=0", [])
        .unwrap();
    assert_eq!(
        store
            .prune_quarantine(1_000, UtilizationPolicy::default())
            .unwrap(),
        8
    );
    assert!(!directory.join("new.quarantine").exists());
    assert_eq!(store.stat().unwrap().quarantine_bytes, 0);
}
