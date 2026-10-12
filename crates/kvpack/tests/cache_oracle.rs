use kvpack::{
    replay_cache_policy, replay_offline_oracle, CacheReplayConfig, CacheReplayPolicy,
    CacheReplayTier, CacheTraceEvent, OfflineOracleBounds, ReplayTierProfile, RetentionInputs,
    StoreError,
};

fn id(value: u8) -> [u8; 32] {
    [value; 32]
}

fn retention(bytes: u64) -> RetentionInputs {
    RetentionInputs {
        predicted_reuse_millis: 1_000,
        avoided_prefill_ns: 100,
        best_restore_ns: 0,
        physical_bytes: bytes,
        shared_ancestor_bytes: 0,
        write_wear_bytes: 0,
        promotion_bandwidth_bytes_per_second: 0,
        queue_interference_ns: 0,
    }
}

fn config(capacity_bytes: u64, maximum_entries: usize) -> CacheReplayConfig {
    let profile = ReplayTierProfile {
        bytes_per_second: 1_000_000_000,
        fixed_latency_ns: 0,
        initially_available: true,
    };
    CacheReplayConfig {
        capacity_bytes,
        maximum_entries,
        resident: profile,
        local: profile,
        gateway: profile,
    }
}

fn accesses(keys: &[u8]) -> Vec<CacheTraceEvent> {
    keys.iter()
        .map(|key| CacheTraceEvent::Access {
            object_key: id(*key),
            source_tier: CacheReplayTier::Local,
            retention: retention(1),
        })
        .collect()
}

#[test]
fn future_knowledge_is_an_exact_lower_bound_for_online_policies() {
    let trace = accesses(&[1, 2, 3, 1, 2, 3]);
    let configuration = config(2, 2);
    let oracle =
        replay_offline_oracle(configuration, &trace, OfflineOracleBounds::default()).unwrap();
    assert_eq!(oracle.accesses, 6);
    assert_eq!(oracle.hits, 2);
    assert_eq!(oracle.misses, 4);
    assert_eq!(oracle.total_service_ns, 4);
    assert!(oracle.peak_states > 1);
    assert!(oracle.evaluated_transitions > 0);

    for policy in [
        CacheReplayPolicy::Lru,
        CacheReplayPolicy::Lfu,
        CacheReplayPolicy::TinyLfu,
        CacheReplayPolicy::Arc,
        CacheReplayPolicy::ClockPro,
    ] {
        let online = replay_cache_policy(policy, configuration, &trace).unwrap();
        assert!(
            oracle.total_service_ns <= online.total_service_ns,
            "offline oracle exceeded {policy:?}"
        );
    }
    assert_eq!(
        replay_cache_policy(CacheReplayPolicy::Lru, configuration, &trace)
            .unwrap()
            .total_service_ns,
        6
    );
}

#[test]
fn oracle_retains_an_object_across_a_known_source_outage() {
    let object = CacheTraceEvent::Access {
        object_key: id(7),
        source_tier: CacheReplayTier::Gateway,
        retention: retention(1),
    };
    let trace = [
        object,
        CacheTraceEvent::SourceFailed(CacheReplayTier::Gateway),
        object,
        CacheTraceEvent::SourceRecovered(CacheReplayTier::Gateway),
    ];
    let oracle =
        replay_offline_oracle(config(1, 1), &trace, OfflineOracleBounds::default()).unwrap();
    assert_eq!(oracle.hits, 1);
    assert_eq!(oracle.misses, 1);
    assert_eq!(oracle.source_failures, 0);
    assert_eq!(oracle.total_service_ns, 1);
}

#[test]
fn oracle_is_deterministic_and_enforces_every_resource_bound() {
    let trace = accesses(&[1, 2, 3, 1]);
    let configuration = config(2, 2);
    let first =
        replay_offline_oracle(configuration, &trace, OfflineOracleBounds::default()).unwrap();
    let second =
        replay_offline_oracle(configuration, &trace, OfflineOracleBounds::default()).unwrap();
    assert_eq!(first, second);

    let too_few_transitions = OfflineOracleBounds {
        maximum_transitions: 1,
        ..OfflineOracleBounds::default()
    };
    assert!(matches!(
        replay_offline_oracle(configuration, &trace, too_few_transitions),
        Err(StoreError::State(
            "offline oracle transition bound exceeded"
        ))
    ));

    let too_few_objects = OfflineOracleBounds {
        maximum_distinct_objects: 2,
        ..OfflineOracleBounds::default()
    };
    assert!(matches!(
        replay_offline_oracle(configuration, &trace, too_few_objects),
        Err(StoreError::State(
            "offline oracle distinct-object bound exceeded"
        ))
    ));
}

#[test]
fn immutable_object_sizes_cannot_change_inside_an_oracle_trace() {
    let trace = [
        CacheTraceEvent::Access {
            object_key: id(9),
            source_tier: CacheReplayTier::Local,
            retention: retention(1),
        },
        CacheTraceEvent::Access {
            object_key: id(9),
            source_tier: CacheReplayTier::Local,
            retention: retention(2),
        },
    ];
    assert!(matches!(
        replay_offline_oracle(config(2, 2), &trace, OfflineOracleBounds::default()),
        Err(StoreError::State(
            "offline oracle object size changed within the trace"
        ))
    ));
}
