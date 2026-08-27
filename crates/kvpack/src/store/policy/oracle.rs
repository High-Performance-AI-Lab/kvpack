use std::collections::BTreeMap;

use kvpack_core::Id32;

use super::replay::{
    profile, replay_service_ns, validate_replay_config, CacheReplayConfig, CacheReplayTier,
    CacheTraceEvent,
};
use crate::StoreError;

pub const MAX_OFFLINE_ORACLE_EVENTS: usize = 4_096;
pub const MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS: usize = 16;
pub const MAX_OFFLINE_ORACLE_STATES: usize = 65_536;
pub const MAX_OFFLINE_ORACLE_TRANSITIONS: u64 = 50_000_000;

/// Explicit resource limits for the future-knowledge cache oracle. The exact
/// dynamic program is deliberately restricted to small fixed traces; it is a
/// qualification baseline and never a production admission policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineOracleBounds {
    pub maximum_events: usize,
    pub maximum_distinct_objects: usize,
    pub maximum_states: usize,
    pub maximum_transitions: u64,
}

impl Default for OfflineOracleBounds {
    fn default() -> Self {
        Self {
            maximum_events: MAX_OFFLINE_ORACLE_EVENTS,
            maximum_distinct_objects: MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS,
            maximum_states: MAX_OFFLINE_ORACLE_STATES,
            maximum_transitions: MAX_OFFLINE_ORACLE_TRANSITIONS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineOracleResult {
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub admitted: u64,
    pub rejected: u64,
    pub source_failures: u64,
    pub evictions: u64,
    pub resident_bytes: u64,
    pub total_service_ns: u64,
    pub peak_states: usize,
    pub evaluated_transitions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct OracleCost {
    hits: u64,
    misses: u64,
    admitted: u64,
    rejected: u64,
    source_failures: u64,
    evictions: u64,
    total_service_ns: u64,
}

/// Find the exact minimum-service cache history for a bounded trace. At each
/// successful miss the oracle may reject the object or retain it after evicting
/// any subset of the current resident set. It knows the entire future trace,
/// including source outages, and therefore provides a lower bound for online
/// admission and eviction policies.
pub fn replay_offline_oracle(
    config: CacheReplayConfig,
    events: &[CacheTraceEvent],
    bounds: OfflineOracleBounds,
) -> Result<OfflineOracleResult, StoreError> {
    validate_replay_config(config, events.len())?;
    validate_bounds(bounds, events.len())?;
    let (object_indices, object_bytes) = inventory(events, bounds.maximum_distinct_objects)?;
    let mut availability = BTreeMap::from([
        (
            CacheReplayTier::Resident,
            config.resident.initially_available,
        ),
        (CacheReplayTier::Local, config.local.initially_available),
        (CacheReplayTier::Gateway, config.gateway.initially_available),
    ]);
    let mut states = BTreeMap::from([(0u64, OracleCost::default())]);
    let mut peak_states = 1usize;
    let mut evaluated_transitions = 0u64;
    let mut accesses = 0u64;

    for event in events {
        let CacheTraceEvent::Access {
            object_key,
            source_tier,
            retention,
        } = *event
        else {
            match *event {
                CacheTraceEvent::SourceFailed(tier) => {
                    availability.insert(tier, false);
                }
                CacheTraceEvent::SourceRecovered(tier) => {
                    availability.insert(tier, true);
                }
                CacheTraceEvent::Access { .. } => unreachable!(),
            }
            continue;
        };
        accesses = accesses.saturating_add(1);
        let retention = retention.validate()?;
        let index = object_indices[&object_key];
        let object_bit = 1u64 << index;
        let source_available = availability[&source_tier];
        let mut next = BTreeMap::new();

        for (&resident, &cost) in &states {
            if resident & object_bit != 0 {
                let mut hit = cost;
                hit.hits = hit.hits.saturating_add(1);
                record_transition(&mut next, resident, hit, bounds, &mut evaluated_transitions)?;
                continue;
            }

            let mut miss = cost;
            miss.misses = miss.misses.saturating_add(1);
            if !source_available {
                miss.source_failures = miss.source_failures.saturating_add(1);
                miss.total_service_ns = miss
                    .total_service_ns
                    .saturating_add(retention.avoided_prefill_ns);
                record_transition(
                    &mut next,
                    resident,
                    miss,
                    bounds,
                    &mut evaluated_transitions,
                )?;
                continue;
            }

            miss.total_service_ns = miss
                .total_service_ns
                .saturating_add(replay_service_ns(profile(config, source_tier), retention));
            let mut rejected = miss;
            rejected.rejected = rejected.rejected.saturating_add(1);
            record_transition(
                &mut next,
                resident,
                rejected,
                bounds,
                &mut evaluated_transitions,
            )?;

            if retention.physical_bytes <= config.capacity_bytes {
                let mut retained = resident;
                loop {
                    let admitted_state = retained | object_bit;
                    if state_fits(
                        admitted_state,
                        &object_bytes,
                        config.capacity_bytes,
                        config.maximum_entries,
                    ) {
                        let mut admitted = miss;
                        admitted.admitted = admitted.admitted.saturating_add(1);
                        admitted.evictions = admitted.evictions.saturating_add(u64::from(
                            resident.count_ones().saturating_sub(retained.count_ones()),
                        ));
                        record_transition(
                            &mut next,
                            admitted_state,
                            admitted,
                            bounds,
                            &mut evaluated_transitions,
                        )?;
                    }
                    if retained == 0 {
                        break;
                    }
                    retained = (retained - 1) & resident;
                }
            }
        }
        states = next;
        peak_states = peak_states.max(states.len());
    }

    let (&resident, &cost) = states
        .iter()
        .min_by(|(left_state, left), (right_state, right)| {
            compare_final(**left_state, **left, **right_state, **right, &object_bytes)
        })
        .ok_or(StoreError::State("offline oracle produced no state"))?;
    Ok(OfflineOracleResult {
        accesses,
        hits: cost.hits,
        misses: cost.misses,
        admitted: cost.admitted,
        rejected: cost.rejected,
        source_failures: cost.source_failures,
        evictions: cost.evictions,
        resident_bytes: state_bytes(resident, &object_bytes).unwrap_or(u64::MAX),
        total_service_ns: cost.total_service_ns,
        peak_states,
        evaluated_transitions,
    })
}

fn validate_bounds(bounds: OfflineOracleBounds, event_count: usize) -> Result<(), StoreError> {
    if bounds.maximum_events == 0
        || bounds.maximum_events > MAX_OFFLINE_ORACLE_EVENTS
        || event_count > bounds.maximum_events
        || bounds.maximum_distinct_objects == 0
        || bounds.maximum_distinct_objects > MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS
        || bounds.maximum_states == 0
        || bounds.maximum_states > MAX_OFFLINE_ORACLE_STATES
        || bounds.maximum_transitions == 0
        || bounds.maximum_transitions > MAX_OFFLINE_ORACLE_TRANSITIONS
    {
        return Err(StoreError::State("offline oracle bounds are invalid"));
    }
    Ok(())
}

fn inventory(
    events: &[CacheTraceEvent],
    maximum_distinct_objects: usize,
) -> Result<(BTreeMap<Id32, usize>, Vec<u64>), StoreError> {
    let mut indices = BTreeMap::new();
    let mut bytes = Vec::new();
    for event in events {
        let CacheTraceEvent::Access {
            object_key,
            retention,
            ..
        } = *event
        else {
            continue;
        };
        let retention = retention.validate()?;
        if let Some(index) = indices.get(&object_key).copied() {
            if bytes[index] != retention.physical_bytes {
                return Err(StoreError::State(
                    "offline oracle object size changed within the trace",
                ));
            }
            continue;
        }
        if indices.len() >= maximum_distinct_objects {
            return Err(StoreError::State(
                "offline oracle distinct-object bound exceeded",
            ));
        }
        let index = bytes.len();
        indices.insert(object_key, index);
        bytes.push(retention.physical_bytes);
    }
    Ok((indices, bytes))
}

fn record_transition(
    states: &mut BTreeMap<u64, OracleCost>,
    resident: u64,
    candidate: OracleCost,
    bounds: OfflineOracleBounds,
    evaluated_transitions: &mut u64,
) -> Result<(), StoreError> {
    *evaluated_transitions = evaluated_transitions
        .checked_add(1)
        .ok_or(StoreError::State(
            "offline oracle transition bound exceeded",
        ))?;
    if *evaluated_transitions > bounds.maximum_transitions {
        return Err(StoreError::State(
            "offline oracle transition bound exceeded",
        ));
    }
    match states.get_mut(&resident) {
        Some(current) if better(candidate, *current) => *current = candidate,
        Some(_) => {}
        None => {
            states.insert(resident, candidate);
            if states.len() > bounds.maximum_states {
                return Err(StoreError::State("offline oracle state bound exceeded"));
            }
        }
    }
    Ok(())
}

fn better(candidate: OracleCost, current: OracleCost) -> bool {
    candidate.total_service_ns < current.total_service_ns
        || (candidate.total_service_ns == current.total_service_ns
            && (candidate.hits > current.hits
                || (candidate.hits == current.hits
                    && (candidate.source_failures < current.source_failures
                        || (candidate.source_failures == current.source_failures
                            && (candidate.evictions < current.evictions
                                || (candidate.evictions == current.evictions
                                    && candidate.admitted < current.admitted)))))))
}

fn compare_final(
    left_state: u64,
    left: OracleCost,
    right_state: u64,
    right: OracleCost,
    object_bytes: &[u64],
) -> std::cmp::Ordering {
    left.total_service_ns
        .cmp(&right.total_service_ns)
        .then_with(|| right.hits.cmp(&left.hits))
        .then_with(|| left.source_failures.cmp(&right.source_failures))
        .then_with(|| left.evictions.cmp(&right.evictions))
        .then_with(|| left.admitted.cmp(&right.admitted))
        .then_with(|| {
            state_bytes(left_state, object_bytes)
                .unwrap_or(u64::MAX)
                .cmp(&state_bytes(right_state, object_bytes).unwrap_or(u64::MAX))
        })
        .then_with(|| left_state.cmp(&right_state))
}

fn state_fits(state: u64, object_bytes: &[u64], capacity: u64, maximum_entries: usize) -> bool {
    state.count_ones() as usize <= maximum_entries
        && state_bytes(state, object_bytes).is_some_and(|bytes| bytes <= capacity)
}

fn state_bytes(state: u64, object_bytes: &[u64]) -> Option<u64> {
    object_bytes
        .iter()
        .enumerate()
        .filter(|(index, _)| state & (1u64 << index) != 0)
        .try_fold(0u64, |total, (_, bytes)| total.checked_add(*bytes))
}
