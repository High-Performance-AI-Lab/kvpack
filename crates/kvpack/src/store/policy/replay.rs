use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyReplayEvent {
    Access {
        object_key: Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        now_ns: u64,
    },
    Admission {
        object_key: Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        utilization_millis: u16,
    },
    Evict,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PolicyReplayResult {
    pub admissions: Vec<AdmissionDecision>,
    pub victims: Vec<Option<Id32>>,
}

pub fn replay_tinylfu(
    config: TinyLfuConfig,
    events: &[PolicyReplayEvent],
) -> Result<PolicyReplayResult, StoreError> {
    let mut policy = TinyLfuPolicy::new(config)?;
    let mut result = PolicyReplayResult::default();
    for event in events {
        match *event {
            PolicyReplayEvent::Access {
                object_key,
                object_bytes,
                avoided_recompute_ns,
                now_ns,
            } => policy.record_access(object_key, object_bytes, avoided_recompute_ns, now_ns),
            PolicyReplayEvent::Admission {
                object_key,
                object_bytes,
                avoided_recompute_ns,
                utilization_millis,
            } => result.admissions.push(policy.admission_decision(
                &object_key,
                object_bytes,
                avoided_recompute_ns,
                utilization_millis,
            )),
            PolicyReplayEvent::Evict => {
                let victim = policy.victim();
                if let Some(victim) = victim {
                    policy.remove(&victim);
                }
                result.victims.push(victim);
            }
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReplayPolicy {
    Lru,
    Lfu,
    TinyLfu,
    Arc,
    ClockPro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CacheReplayTier {
    Resident,
    Local,
    Gateway,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayTierProfile {
    pub bytes_per_second: u64,
    pub fixed_latency_ns: u64,
    pub initially_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReplayConfig {
    pub capacity_bytes: u64,
    pub maximum_entries: usize,
    pub resident: ReplayTierProfile,
    pub local: ReplayTierProfile,
    pub gateway: ReplayTierProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTraceEvent {
    Access {
        object_key: Id32,
        source_tier: CacheReplayTier,
        retention: RetentionInputs,
    },
    SourceFailed(CacheReplayTier),
    SourceRecovered(CacheReplayTier),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReplayOutcome {
    Hit,
    MissAdmitted,
    MissRejected,
    SourceUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheReplayResult {
    pub outcomes: Vec<CacheReplayOutcome>,
    pub evictions: Vec<Id32>,
    pub hits: u64,
    pub misses: u64,
    pub admitted: u64,
    pub rejected: u64,
    pub source_failures: u64,
    pub resident_bytes: u64,
    pub total_service_ns: u64,
}

#[derive(Debug, Clone, Copy)]
struct ReplayEntry {
    retention: RetentionInputs,
    frequency: u64,
    last_access: u64,
    referenced: bool,
    hot: bool,
}

struct ReplayState {
    policy: CacheReplayPolicy,
    config: CacheReplayConfig,
    entries: BTreeMap<Id32, ReplayEntry>,
    resident_bytes: u64,
    tick: u64,
    tiny_lfu: TinyLfuPolicy,
    arc_recent: VecDeque<Id32>,
    arc_frequent: VecDeque<Id32>,
    arc_recent_ghost: VecDeque<Id32>,
    arc_frequent_ghost: VecDeque<Id32>,
    arc_target_bytes: u64,
    clock: VecDeque<Id32>,
}

pub fn replay_cache_policy(
    policy: CacheReplayPolicy,
    config: CacheReplayConfig,
    events: &[CacheTraceEvent],
) -> Result<CacheReplayResult, StoreError> {
    validate_replay_config(config, events.len())?;
    let mut state = ReplayState {
        policy,
        config,
        entries: BTreeMap::new(),
        resident_bytes: 0,
        tick: 0,
        tiny_lfu: TinyLfuPolicy::new(TinyLfuConfig::production(config.capacity_bytes)?)?,
        arc_recent: VecDeque::new(),
        arc_frequent: VecDeque::new(),
        arc_recent_ghost: VecDeque::new(),
        arc_frequent_ghost: VecDeque::new(),
        arc_target_bytes: config.capacity_bytes / 2,
        clock: VecDeque::new(),
    };
    let mut availability = BTreeMap::from([
        (
            CacheReplayTier::Resident,
            config.resident.initially_available,
        ),
        (CacheReplayTier::Local, config.local.initially_available),
        (CacheReplayTier::Gateway, config.gateway.initially_available),
    ]);
    let mut result = CacheReplayResult::default();
    for event in events {
        match *event {
            CacheTraceEvent::SourceFailed(tier) => {
                availability.insert(tier, false);
            }
            CacheTraceEvent::SourceRecovered(tier) => {
                availability.insert(tier, true);
            }
            CacheTraceEvent::Access {
                object_key,
                source_tier,
                retention,
            } => {
                let retention = retention.validate()?;
                state.tick = state.tick.saturating_add(1);
                if state.entries.contains_key(&object_key) {
                    state.hit(object_key, retention);
                    result.hits = result.hits.saturating_add(1);
                    result.outcomes.push(CacheReplayOutcome::Hit);
                    continue;
                }
                result.misses = result.misses.saturating_add(1);
                if !availability[&source_tier] {
                    result.source_failures = result.source_failures.saturating_add(1);
                    result.total_service_ns = result
                        .total_service_ns
                        .saturating_add(retention.avoided_prefill_ns);
                    result.outcomes.push(CacheReplayOutcome::SourceUnavailable);
                    continue;
                }
                result.total_service_ns = result
                    .total_service_ns
                    .saturating_add(replay_service_ns(profile(config, source_tier), retention));
                let preferred_victim = if policy == CacheReplayPolicy::TinyLfu {
                    let projected = state
                        .resident_bytes
                        .saturating_add(retention.physical_bytes)
                        .saturating_mul(1000)
                        .checked_div(config.capacity_bytes)
                        .unwrap_or(1000)
                        .min(1000) as u16;
                    match state.tiny_lfu.automatic_admission_decision(
                        &object_key,
                        retention,
                        projected,
                    )? {
                        AdmissionDecision::Admit => None,
                        AdmissionDecision::AdmitOverVictim(victim) => Some(victim),
                        AdmissionDecision::RejectLowerFrequency
                        | AdmissionDecision::RejectPromotionStopped => {
                            result.rejected = result.rejected.saturating_add(1);
                            result.outcomes.push(CacheReplayOutcome::MissRejected);
                            continue;
                        }
                    }
                } else {
                    None
                };
                if retention.physical_bytes > config.capacity_bytes
                    || !state.make_room(retention.physical_bytes, preferred_victim, &mut result)
                {
                    result.rejected = result.rejected.saturating_add(1);
                    result.outcomes.push(CacheReplayOutcome::MissRejected);
                    continue;
                }
                state.insert(object_key, retention);
                result.admitted = result.admitted.saturating_add(1);
                result.outcomes.push(CacheReplayOutcome::MissAdmitted);
            }
        }
    }
    result.resident_bytes = state.resident_bytes;
    Ok(result)
}

impl ReplayState {
    fn hit(&mut self, key: Id32, retention: RetentionInputs) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.frequency = entry.frequency.saturating_add(1);
            entry.last_access = self.tick;
            entry.referenced = true;
            entry.hot |= entry.frequency >= 2;
            entry.retention = retention;
        }
        match self.policy {
            CacheReplayPolicy::TinyLfu => {
                self.tiny_lfu
                    .record_access_with_retention(key, retention, self.tick);
            }
            CacheReplayPolicy::Arc => {
                if remove_key(&mut self.arc_recent, &key) {
                    self.arc_frequent.push_back(key);
                } else {
                    touch(&mut self.arc_frequent, key);
                }
            }
            CacheReplayPolicy::ClockPro => {}
            CacheReplayPolicy::Lru | CacheReplayPolicy::Lfu => {}
        }
    }

    fn insert(&mut self, key: Id32, retention: RetentionInputs) {
        self.resident_bytes = self.resident_bytes.saturating_add(retention.physical_bytes);
        self.entries.insert(
            key,
            ReplayEntry {
                retention,
                frequency: 1,
                last_access: self.tick,
                referenced: true,
                hot: false,
            },
        );
        match self.policy {
            CacheReplayPolicy::TinyLfu => {
                self.tiny_lfu
                    .register_with_retention(key, retention, self.tick);
            }
            CacheReplayPolicy::Arc => {
                if remove_key(&mut self.arc_recent_ghost, &key) {
                    self.arc_target_bytes = self
                        .arc_target_bytes
                        .saturating_add(retention.physical_bytes)
                        .min(self.config.capacity_bytes);
                    self.arc_frequent.push_back(key);
                } else if remove_key(&mut self.arc_frequent_ghost, &key) {
                    self.arc_target_bytes = self
                        .arc_target_bytes
                        .saturating_sub(retention.physical_bytes);
                    self.arc_frequent.push_back(key);
                } else {
                    self.arc_recent.push_back(key);
                }
            }
            CacheReplayPolicy::ClockPro => self.clock.push_back(key),
            CacheReplayPolicy::Lru | CacheReplayPolicy::Lfu => {}
        }
    }

    fn make_room(
        &mut self,
        incoming: u64,
        mut preferred: Option<Id32>,
        result: &mut CacheReplayResult,
    ) -> bool {
        while self.resident_bytes.saturating_add(incoming) > self.config.capacity_bytes
            || self.entries.len() >= self.config.maximum_entries
        {
            let victim = preferred
                .take()
                .filter(|key| self.entries.contains_key(key))
                .or_else(|| self.victim());
            let Some(victim) = victim else {
                return false;
            };
            self.remove(victim);
            result.evictions.push(victim);
        }
        true
    }

    fn victim(&mut self) -> Option<Id32> {
        match self.policy {
            CacheReplayPolicy::Lru => self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.last_access, **key))
                .map(|(key, _)| *key),
            CacheReplayPolicy::Lfu => self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.frequency, entry.last_access, **key))
                .map(|(key, _)| *key),
            CacheReplayPolicy::TinyLfu => self.tiny_lfu.victim(),
            CacheReplayPolicy::Arc => {
                let recent_bytes = self.arc_recent.iter().fold(0u64, |sum, key| {
                    sum.saturating_add(
                        self.entries
                            .get(key)
                            .map(|entry| entry.retention.physical_bytes)
                            .unwrap_or(0),
                    )
                });
                if recent_bytes > self.arc_target_bytes || self.arc_frequent.is_empty() {
                    self.arc_recent.front().copied()
                } else {
                    self.arc_frequent.front().copied()
                }
            }
            CacheReplayPolicy::ClockPro => {
                let bound = self.clock.len().saturating_mul(3).max(1);
                for _ in 0..bound {
                    let key = self.clock.pop_front()?;
                    let Some(entry) = self.entries.get_mut(&key) else {
                        continue;
                    };
                    self.clock.push_back(key);
                    if entry.referenced {
                        entry.referenced = false;
                        entry.hot |= entry.frequency >= 2;
                        continue;
                    }
                    if entry.hot {
                        entry.hot = false;
                        continue;
                    }
                    return Some(key);
                }
                self.clock.front().copied()
            }
        }
    }

    fn remove(&mut self, key: Id32) {
        let Some(entry) = self.entries.remove(&key) else {
            return;
        };
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(entry.retention.physical_bytes);
        match self.policy {
            CacheReplayPolicy::TinyLfu => self.tiny_lfu.remove(&key),
            CacheReplayPolicy::Arc => {
                if remove_key(&mut self.arc_recent, &key) {
                    self.arc_recent_ghost.push_back(key);
                } else if remove_key(&mut self.arc_frequent, &key) {
                    self.arc_frequent_ghost.push_back(key);
                }
                trim_ghosts(
                    &mut self.arc_recent_ghost,
                    &mut self.arc_frequent_ghost,
                    self.config.maximum_entries,
                );
            }
            CacheReplayPolicy::ClockPro => {
                remove_key(&mut self.clock, &key);
            }
            CacheReplayPolicy::Lru | CacheReplayPolicy::Lfu => {}
        }
    }
}

pub(super) fn validate_replay_config(
    config: CacheReplayConfig,
    event_count: usize,
) -> Result<(), StoreError> {
    if config.capacity_bytes == 0
        || config.maximum_entries == 0
        || event_count > 10_000_000
        || [config.resident, config.local, config.gateway]
            .iter()
            .any(|profile| profile.bytes_per_second == 0)
    {
        return Err(StoreError::State("cache replay configuration is invalid"));
    }
    Ok(())
}

pub(super) fn profile(config: CacheReplayConfig, tier: CacheReplayTier) -> ReplayTierProfile {
    match tier {
        CacheReplayTier::Resident => config.resident,
        CacheReplayTier::Local => config.local,
        CacheReplayTier::Gateway => config.gateway,
    }
}

pub(super) fn replay_service_ns(profile: ReplayTierProfile, retention: RetentionInputs) -> u64 {
    let transfer = ((retention.physical_bytes as u128)
        .saturating_mul(1_000_000_000)
        .saturating_add(profile.bytes_per_second as u128 - 1)
        / profile.bytes_per_second as u128)
        .min(u64::MAX as u128) as u64;
    profile
        .fixed_latency_ns
        .saturating_add(transfer)
        .saturating_add(retention.best_restore_ns)
        .saturating_add(retention.queue_interference_ns)
}

fn trim_ghosts(recent: &mut VecDeque<Id32>, frequent: &mut VecDeque<Id32>, maximum_entries: usize) {
    let mut seen = BTreeSet::new();
    recent.retain(|key| seen.insert(*key));
    frequent.retain(|key| seen.insert(*key));
    while recent.len().saturating_add(frequent.len()) > maximum_entries {
        if recent.len() >= frequent.len() {
            recent.pop_front();
        } else {
            frequent.pop_front();
        }
    }
}

pub(super) fn score(frequency: u64, inputs: RetentionInputs) -> u64 {
    if inputs.physical_bytes == 0 || inputs.predicted_reuse_millis == 0 {
        return 0;
    }
    let benefit_ns = inputs
        .avoided_prefill_ns
        .saturating_sub(inputs.best_restore_ns)
        .saturating_sub(inputs.promotion_ns())
        .saturating_sub(inputs.queue_interference_ns);
    if benefit_ns == 0 {
        return 0;
    }
    let effective_bytes = inputs
        .marginal_physical_bytes()
        .saturating_add(inputs.write_wear_bytes)
        .max(1);
    let numerator = (frequency as u128)
        .saturating_mul(inputs.predicted_reuse_millis as u128)
        .saturating_mul(benefit_ns as u128);
    (numerator / 1000 / effective_bytes as u128)
        .max(1)
        .min(i64::MAX as u128) as u64
}

pub(super) fn sketch_index(object_key: &Id32, row: usize, width: usize) -> usize {
    let mut digest = Sha256::new();
    digest.update(b"kvpack/v1/tinylfu\0");
    digest.update((row as u64).to_le_bytes());
    digest.update(object_key);
    let hash: [u8; 32] = digest.finalize().into();
    let value = u64::from_le_bytes(hash[..8].try_into().unwrap()) as usize;
    row * width + (value & (width - 1))
}

pub(super) fn remove_key(queue: &mut VecDeque<Id32>, key: &Id32) -> bool {
    if let Some(index) = queue.iter().position(|candidate| candidate == key) {
        queue.remove(index);
        true
    } else {
        false
    }
}

pub(super) fn touch(queue: &mut VecDeque<Id32>, key: Id32) {
    remove_key(queue, &key);
    queue.push_back(key);
}
