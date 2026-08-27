use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use kvpack_core::Id32;
use sha2::{Digest, Sha256};

use crate::LocalStore;
use crate::StoreError;
use rusqlite::{params, TransactionBehavior};

const SKETCH_DEPTH: usize = 4;
const ACCESS_FLUSH_INTERVAL_NS: u64 = 30_000_000_000;
const ACCESS_FLUSH_HARD_BOUND_NS: u64 = 60_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionInputs {
    pub predicted_reuse_millis: u16,
    pub avoided_prefill_ns: u64,
    pub best_restore_ns: u64,
    pub physical_bytes: u64,
    pub shared_ancestor_bytes: u64,
    pub write_wear_bytes: u64,
    pub promotion_bandwidth_bytes_per_second: u64,
    pub queue_interference_ns: u64,
}

impl RetentionInputs {
    pub const fn conservative(physical_bytes: u64, avoided_prefill_ns: u64) -> Self {
        Self {
            predicted_reuse_millis: 1000,
            avoided_prefill_ns,
            best_restore_ns: 0,
            physical_bytes,
            shared_ancestor_bytes: 0,
            write_wear_bytes: 0,
            promotion_bandwidth_bytes_per_second: 0,
            queue_interference_ns: 0,
        }
    }

    pub fn with_physical_bytes(mut self, physical_bytes: u64) -> Result<Self, StoreError> {
        self.physical_bytes = physical_bytes;
        self.shared_ancestor_bytes = self.shared_ancestor_bytes.min(physical_bytes);
        self.validate()
    }

    pub fn validate(self) -> Result<Self, StoreError> {
        if self.predicted_reuse_millis > 1000
            || self.physical_bytes == 0
            || self.shared_ancestor_bytes > self.physical_bytes
        {
            return Err(StoreError::State("retention inputs are invalid"));
        }
        Ok(self)
    }

    pub fn marginal_physical_bytes(self) -> u64 {
        self.physical_bytes
            .saturating_sub(self.shared_ancestor_bytes)
            .max(1)
    }

    fn promotion_ns(self) -> u64 {
        if self.promotion_bandwidth_bytes_per_second == 0 {
            return 0;
        }
        ((self.physical_bytes as u128)
            .saturating_mul(1_000_000_000)
            .saturating_add(self.promotion_bandwidth_bytes_per_second as u128 - 1)
            / self.promotion_bandwidth_bytes_per_second as u128)
            .min(u64::MAX as u128) as u64
    }
}

impl Default for RetentionInputs {
    fn default() -> Self {
        Self::conservative(1, 1)
    }
}

pub fn retention_value(frequency: u64, inputs: RetentionInputs) -> Result<u64, StoreError> {
    let inputs = inputs.validate()?;
    Ok(score(frequency, inputs))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessFlushSchedule {
    pub next_attempt_ns: u64,
    pub hard_deadline_ns: u64,
    pub overdue: bool,
}

pub fn access_flush_schedule(
    first_pending_ns: u64,
    previous_attempt_ns: Option<u64>,
    now_ns: u64,
) -> AccessFlushSchedule {
    let hard_deadline_ns = first_pending_ns.saturating_add(ACCESS_FLUSH_HARD_BOUND_NS);
    let normal = first_pending_ns.saturating_add(ACCESS_FLUSH_INTERVAL_NS);
    let retry = previous_attempt_ns
        .filter(|attempt| *attempt >= first_pending_ns)
        .map(|attempt| attempt.saturating_add(ACCESS_FLUSH_INTERVAL_NS));
    let mut next_attempt_ns = retry.unwrap_or(normal);
    if now_ns < hard_deadline_ns {
        next_attempt_ns = next_attempt_ns.min(hard_deadline_ns);
    } else {
        next_attempt_ns = next_attempt_ns.max(now_ns.saturating_add(ACCESS_FLUSH_INTERVAL_NS));
    }
    AccessFlushSchedule {
        next_attempt_ns,
        hard_deadline_ns,
        overdue: now_ns > hard_deadline_ns,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionSegment {
    Probationary,
    Protected,
}

impl RetentionSegment {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Probationary => "PROBATIONARY",
            Self::Protected => "PROTECTED",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, StoreError> {
        match value {
            "PROBATIONARY" => Ok(Self::Probationary),
            "PROTECTED" => Ok(Self::Protected),
            _ => Err(StoreError::State(
                "catalog contains an unknown retention segment",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyLfuConfig {
    pub capacity_bytes: u64,
    pub sketch_width: usize,
    pub reset_after_accesses: u64,
    pub protected_fraction_millis: u16,
}

impl TinyLfuConfig {
    pub fn production(capacity_bytes: u64) -> Result<Self, StoreError> {
        if capacity_bytes == 0 {
            return Err(StoreError::State("TinyLFU capacity must be nonzero"));
        }
        Ok(Self {
            capacity_bytes,
            sketch_width: 16_384,
            reset_after_accesses: 1_000_000,
            protected_fraction_millis: 800,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    AdmitOverVictim(Id32),
    RejectLowerFrequency,
    RejectPromotionStopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PolicyEntry {
    object_bytes: u64,
    retention: RetentionInputs,
    frequency: u64,
    score: u64,
    segment: RetentionSegment,
    last_access_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PendingPolicyAccess {
    pub object_key: Id32,
    pub frequency: u64,
    pub score: u64,
    pub segment: RetentionSegment,
    pub last_access_ns: u64,
    pub first_pending_ns: u64,
}

#[derive(Debug)]
pub struct TinyLfuPolicy {
    config: TinyLfuConfig,
    sketch: Vec<u16>,
    accesses: u64,
    entries: BTreeMap<Id32, PolicyEntry>,
    probationary: VecDeque<Id32>,
    protected: VecDeque<Id32>,
    protected_bytes: u64,
    pending: BTreeMap<Id32, PendingPolicyAccess>,
}

impl TinyLfuPolicy {
    pub fn new(config: TinyLfuConfig) -> Result<Self, StoreError> {
        if config.sketch_width < 64
            || !config.sketch_width.is_power_of_two()
            || config.reset_after_accesses == 0
            || config.protected_fraction_millis > 1000
        {
            return Err(StoreError::State("TinyLFU configuration is invalid"));
        }
        Ok(Self {
            config,
            sketch: vec![0; config.sketch_width * SKETCH_DEPTH],
            accesses: 0,
            entries: BTreeMap::new(),
            probationary: VecDeque::new(),
            protected: VecDeque::new(),
            protected_bytes: 0,
            pending: BTreeMap::new(),
        })
    }

    pub fn register(
        &mut self,
        object_key: Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        now_ns: u64,
    ) {
        self.register_with_retention(
            object_key,
            RetentionInputs::conservative(object_bytes, avoided_recompute_ns),
            now_ns,
        );
    }

    pub fn register_with_retention(
        &mut self,
        object_key: Id32,
        retention: RetentionInputs,
        now_ns: u64,
    ) {
        if self.entries.contains_key(&object_key) {
            return;
        }
        let Ok(retention) = retention.validate() else {
            return;
        };
        let object_bytes = retention.physical_bytes;
        let frequency = self.estimate(&object_key).max(1);
        let score = score(frequency, retention);
        self.entries.insert(
            object_key,
            PolicyEntry {
                object_bytes,
                retention,
                frequency,
                score,
                segment: RetentionSegment::Probationary,
                last_access_ns: now_ns,
            },
        );
        self.probationary.push_back(object_key);
        self.mark_pending(object_key);
    }

    pub fn record_access(
        &mut self,
        object_key: Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        now_ns: u64,
    ) {
        self.record_access_with_retention(
            object_key,
            RetentionInputs::conservative(object_bytes, avoided_recompute_ns),
            now_ns,
        );
    }

    pub fn record_access_with_retention(
        &mut self,
        object_key: Id32,
        retention: RetentionInputs,
        now_ns: u64,
    ) {
        let Ok(mut retention) = retention.validate() else {
            return;
        };
        let object_bytes = self
            .entries
            .get(&object_key)
            .map(|entry| entry.object_bytes)
            .unwrap_or(retention.physical_bytes);
        retention.physical_bytes = object_bytes;
        retention.shared_ancestor_bytes = retention.shared_ancestor_bytes.min(object_bytes);
        self.increment(&object_key);
        let estimate = self.estimate(&object_key).max(1);
        self.entries
            .entry(object_key)
            .or_insert_with(|| PolicyEntry {
                object_bytes,
                retention,
                frequency: estimate,
                score: score(estimate, retention),
                segment: RetentionSegment::Probationary,
                last_access_ns: now_ns,
            });
        let was_probationary = self.entries[&object_key].segment == RetentionSegment::Probationary;
        if was_probationary && estimate >= 2 {
            remove_key(&mut self.probationary, &object_key);
            self.protected.push_back(object_key);
            if let Some(entry) = self.entries.get_mut(&object_key) {
                entry.segment = RetentionSegment::Protected;
                self.protected_bytes = self.protected_bytes.saturating_add(entry.object_bytes);
            }
        } else if was_probationary {
            touch(&mut self.probationary, object_key);
        } else {
            touch(&mut self.protected, object_key);
        }
        if let Some(entry) = self.entries.get_mut(&object_key) {
            entry.frequency = estimate;
            entry.object_bytes = object_bytes;
            entry.retention = retention;
            entry.score = score(estimate, retention);
            entry.last_access_ns = now_ns;
        }
        self.demote_protected_overflow();
        self.mark_pending(object_key);
        self.accesses = self.accesses.saturating_add(1);
        if self.accesses >= self.config.reset_after_accesses {
            self.age();
        }
    }

    pub fn admission_decision(
        &self,
        object_key: &Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        utilization_millis: u16,
    ) -> AdmissionDecision {
        self.admission_decision_with_retention(
            object_key,
            RetentionInputs::conservative(object_bytes, avoided_recompute_ns),
            utilization_millis,
        )
    }

    pub fn admission_decision_with_retention(
        &self,
        object_key: &Id32,
        retention: RetentionInputs,
        utilization_millis: u16,
    ) -> AdmissionDecision {
        if utilization_millis >= 950 {
            return AdmissionDecision::RejectPromotionStopped;
        }
        if utilization_millis < 850 {
            return AdmissionDecision::Admit;
        }
        let candidate_frequency = self.estimate(object_key).max(1);
        let candidate_score = score(candidate_frequency, retention);
        let Some((victim, entry)) = self
            .probationary
            .iter()
            .filter_map(|key| self.entries.get(key).map(|entry| (*key, entry)))
            .min_by_key(|(key, entry)| (entry.score, entry.last_access_ns, *key))
        else {
            return AdmissionDecision::Admit;
        };
        if candidate_score > entry.score {
            AdmissionDecision::AdmitOverVictim(victim)
        } else {
            AdmissionDecision::RejectLowerFrequency
        }
    }

    pub fn automatic_admission_decision(
        &mut self,
        object_key: &Id32,
        retention: RetentionInputs,
        utilization_millis: u16,
    ) -> Result<AdmissionDecision, StoreError> {
        let retention = retention.validate()?;
        self.increment(object_key);
        self.accesses = self.accesses.saturating_add(1);
        let decision =
            self.admission_decision_with_retention(object_key, retention, utilization_millis);
        if self.accesses >= self.config.reset_after_accesses {
            self.age();
        }
        Ok(decision)
    }

    pub fn victim(&self) -> Option<Id32> {
        self.probationary
            .iter()
            .chain(&self.protected)
            .filter_map(|key| self.entries.get(key).map(|entry| (*key, entry)))
            .min_by_key(|(key, entry)| {
                (
                    entry.segment == RetentionSegment::Protected,
                    entry.score,
                    entry.last_access_ns,
                    *key,
                )
            })
            .map(|(key, _)| key)
    }

    pub fn remove(&mut self, object_key: &Id32) {
        if let Some(entry) = self.entries.remove(object_key) {
            if entry.segment == RetentionSegment::Protected {
                self.protected_bytes = self.protected_bytes.saturating_sub(entry.object_bytes);
            }
        }
        remove_key(&mut self.probationary, object_key);
        remove_key(&mut self.protected, object_key);
        self.pending.remove(object_key);
    }

    pub(crate) fn drain_pending(&mut self) -> Vec<PendingPolicyAccess> {
        std::mem::take(&mut self.pending).into_values().collect()
    }

    pub(crate) fn restore_pending(&mut self, values: Vec<PendingPolicyAccess>) {
        for value in values {
            self.pending.insert(value.object_key, value);
        }
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn oldest_pending_ns(&self) -> Option<u64> {
        self.pending
            .values()
            .map(|access| access.first_pending_ns)
            .min()
    }

    pub(crate) fn restore_entry(
        &mut self,
        object_key: Id32,
        object_bytes: u64,
        frequency: u64,
        score: u64,
        segment: RetentionSegment,
        last_access_ns: u64,
    ) {
        if self.entries.contains_key(&object_key) {
            return;
        }
        for row in 0..SKETCH_DEPTH {
            let index = sketch_index(&object_key, row, self.config.sketch_width);
            self.sketch[index] = self.sketch[index].max(frequency.min(u16::MAX as u64) as u16);
        }
        let entry = PolicyEntry {
            object_bytes,
            retention: RetentionInputs::conservative(
                object_bytes,
                score.saturating_mul(object_bytes) / frequency.max(1),
            ),
            frequency: frequency.max(1),
            score,
            segment,
            last_access_ns,
        };
        self.entries.insert(object_key, entry);
        match segment {
            RetentionSegment::Probationary => self.probationary.push_back(object_key),
            RetentionSegment::Protected => {
                self.protected.push_back(object_key);
                self.protected_bytes = self.protected_bytes.saturating_add(object_bytes);
            }
        }
    }

    fn mark_pending(&mut self, object_key: Id32) {
        let Some(entry) = self.entries.get(&object_key) else {
            return;
        };
        let first_pending_ns = self
            .pending
            .get(&object_key)
            .map(|pending| pending.first_pending_ns)
            .unwrap_or(entry.last_access_ns);
        self.pending.insert(
            object_key,
            PendingPolicyAccess {
                object_key,
                frequency: entry.frequency,
                score: entry.score,
                segment: entry.segment,
                last_access_ns: entry.last_access_ns,
                first_pending_ns,
            },
        );
    }

    fn increment(&mut self, object_key: &Id32) {
        for row in 0..SKETCH_DEPTH {
            let index = sketch_index(object_key, row, self.config.sketch_width);
            self.sketch[index] = self.sketch[index].saturating_add(1);
        }
    }

    fn estimate(&self, object_key: &Id32) -> u64 {
        (0..SKETCH_DEPTH)
            .map(|row| self.sketch[sketch_index(object_key, row, self.config.sketch_width)] as u64)
            .min()
            .unwrap_or(0)
    }

    fn demote_protected_overflow(&mut self) {
        let bound = self
            .config
            .capacity_bytes
            .saturating_mul(self.config.protected_fraction_millis as u64)
            / 1000;
        while self.protected_bytes > bound {
            let Some(key) = self.protected.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.segment = RetentionSegment::Probationary;
                self.protected_bytes = self.protected_bytes.saturating_sub(entry.object_bytes);
                self.probationary.push_back(key);
            }
        }
    }

    fn age(&mut self) {
        for counter in &mut self.sketch {
            *counter /= 2;
        }
        let keys: Vec<_> = self.entries.keys().copied().collect();
        for key in keys {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.frequency = (entry.frequency / 2).max(1);
                entry.score = score(entry.frequency, entry.retention);
            }
            self.mark_pending(key);
        }
        self.accesses = 0;
    }
}

impl LocalStore {
    pub fn record_chunk_access(
        self: &Arc<Self>,
        object_key: Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
    ) -> Result<(), StoreError> {
        self.policy
            .lock()
            .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
            .record_access(object_key, object_bytes, avoided_recompute_ns, now_ns());
        self.schedule_access_flush();
        Ok(())
    }

    pub fn flush_access_epochs(&self) -> Result<usize, StoreError> {
        let pending = self
            .policy
            .lock()
            .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
            .drain_pending();
        if pending.is_empty() {
            return Ok(0);
        }
        let result = (|| {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let epoch = now_ns() / 30_000_000_000;
            for access in &pending {
                transaction.execute("UPDATE chunks SET frequency_estimate=?3,retention_segment=?4,last_access_ns=?5,last_access_epoch=?6 WHERE tenant=?1 AND object_key=?2", params![self.tenant_namespace.as_slice(), access.object_key.as_slice(), access.frequency, access.segment.as_str(), access.last_access_ns, epoch])?;
                transaction.execute("INSERT INTO policy_objects(tenant,object_key,frequency,segment,score,last_access_ns,last_persisted_epoch) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(tenant,object_key) DO UPDATE SET frequency=excluded.frequency,segment=excluded.segment,score=excluded.score,last_access_ns=excluded.last_access_ns,last_persisted_epoch=excluded.last_persisted_epoch", params![self.tenant_namespace.as_slice(), access.object_key.as_slice(), access.frequency, access.segment.as_str(), access.score, access.last_access_ns, epoch])?;
            }
            transaction.commit()?;
            Ok(pending.len())
        })();
        if result.is_err() {
            if let Ok(mut policy) = self.policy.lock() {
                policy.restore_pending(pending);
            }
        }
        result
    }

    fn schedule_access_flush(self: &Arc<Self>) {
        if self
            .access_flush_scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let first_pending_ns = match self.policy.lock().map(|policy| policy.oldest_pending_ns()) {
            Ok(Some(value)) => value,
            _ => {
                self.access_flush_scheduled.store(false, Ordering::Release);
                return;
            }
        };
        let now = now_ns();
        let previous = match self.access_flush_last_attempt_ns.load(Ordering::Acquire) {
            0 => None,
            value => Some(value),
        };
        let schedule = access_flush_schedule(first_pending_ns, previous, now);
        let delay = std::time::Duration::from_nanos(
            schedule
                .next_attempt_ns
                .saturating_sub(now)
                .min(ACCESS_FLUSH_INTERVAL_NS),
        );
        let store = Arc::downgrade(self);
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            let Some(store) = store.upgrade() else {
                return;
            };
            store
                .access_flush_last_attempt_ns
                .store(now_ns(), Ordering::Release);
            let flushed = store.flush_access_epochs().is_ok();
            store.access_flush_scheduled.store(false, Ordering::Release);
            let pending = store
                .policy
                .lock()
                .map(|policy| policy.has_pending())
                .unwrap_or(false);
            if flushed && !pending {
                store
                    .access_flush_last_attempt_ns
                    .store(0, Ordering::Release);
            }
            if pending {
                store.schedule_access_flush();
            }
        });
    }
}

mod oracle;
mod replay;
pub use oracle::{
    replay_offline_oracle, OfflineOracleBounds, OfflineOracleResult,
    MAX_OFFLINE_ORACLE_DISTINCT_OBJECTS, MAX_OFFLINE_ORACLE_EVENTS, MAX_OFFLINE_ORACLE_STATES,
    MAX_OFFLINE_ORACLE_TRANSITIONS,
};
use replay::{remove_key, score, sketch_index, touch};
pub use replay::{
    replay_cache_policy, replay_tinylfu, CacheReplayConfig, CacheReplayOutcome, CacheReplayPolicy,
    CacheReplayResult, CacheReplayTier, CacheTraceEvent, PolicyReplayEvent, PolicyReplayResult,
    ReplayTierProfile,
};
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
