use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtilizationPolicy {
    pub begin_eviction: f64,
    pub stop_eviction: f64,
    pub emergency_eviction: f64,
    pub stop_promotion: f64,
    pub quarantine_fraction: f64,
}

impl Default for UtilizationPolicy {
    fn default() -> Self {
        Self {
            begin_eviction: 0.85,
            stop_eviction: 0.75,
            emergency_eviction: 0.92,
            stop_promotion: 0.95,
            quarantine_fraction: 0.01,
        }
    }
}

impl UtilizationPolicy {
    pub(super) fn validate(self) -> Result<Self, StoreError> {
        if !(0.0..=1.0).contains(&self.stop_eviction)
            || self.stop_eviction >= self.begin_eviction
            || self.begin_eviction >= self.emergency_eviction
            || self.emergency_eviction >= self.stop_promotion
            || self.stop_promotion > 1.0
            || !(0.0..=0.1).contains(&self.quarantine_fraction)
        {
            return Err(StoreError::State("utilization policy is invalid"));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePressure {
    Normal,
    Evicting,
    Emergency,
    PromotionStopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictionReport {
    pub before_bytes: u64,
    pub after_bytes: u64,
    pub manifests_evicted: u64,
    pub chunks_evicted: u64,
    pub emergency: bool,
    pub blocked: bool,
}

impl LocalStore {
    pub fn cache_pressure(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
    ) -> Result<CachePressure, StoreError> {
        let policy = policy.validate()?;
        if tier_capacity_bytes == 0 {
            return Err(StoreError::State("tier capacity must be nonzero"));
        }
        let stat = self.stat()?;
        let utilization = stat.durable_bytes.saturating_add(stat.reserved_bytes) as f64
            / tier_capacity_bytes as f64;
        Ok(if utilization >= policy.stop_promotion {
            CachePressure::PromotionStopped
        } else if utilization >= policy.emergency_eviction {
            CachePressure::Emergency
        } else if utilization >= policy.begin_eviction {
            CachePressure::Evicting
        } else {
            CachePressure::Normal
        })
    }

    pub fn admission_decision(
        &self,
        object_key: &kvpack_core::Id32,
        object_bytes: u64,
        avoided_recompute_ns: u64,
        tier_capacity_bytes: u64,
    ) -> Result<AdmissionDecision, StoreError> {
        if tier_capacity_bytes == 0 {
            return Err(StoreError::State("tier capacity must be nonzero"));
        }
        let stat = self.stat()?;
        let utilization_millis = stat
            .durable_bytes
            .saturating_add(stat.reserved_bytes)
            .saturating_mul(1000)
            .checked_div(tier_capacity_bytes)
            .unwrap_or(1000)
            .min(1000) as u16;
        Ok(self
            .policy
            .lock()
            .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
            .admission_decision(
                object_key,
                object_bytes,
                avoided_recompute_ns,
                utilization_millis,
            ))
    }

    pub fn maintain_capacity(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
        maximum_operations: usize,
    ) -> Result<EvictionReport, StoreError> {
        self.maintain_capacity_with_headroom(tier_capacity_bytes, policy, maximum_operations, 0)
    }

    /// `maintain_capacity` with the M6 fidelity ladder enabled: watermark
    /// pressure first demotes the coldest chunk objects exactly one rung
    /// (0 resident-fp16 → 1 rest-quantized → 2 tombstone) instead of
    /// deleting them, and only objects already on the tombstone rung are
    /// collected.  Eviction never skips ahead.
    pub fn maintain_capacity_with_fidelity_demotion(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
        maximum_operations: usize,
    ) -> Result<EvictionReport, StoreError> {
        self.maintain_capacity_impl(tier_capacity_bytes, policy, maximum_operations, 0, true)
    }

    pub(crate) fn maintain_capacity_with_headroom(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
        maximum_operations: usize,
        incoming_reserved_bytes: u64,
    ) -> Result<EvictionReport, StoreError> {
        self.maintain_capacity_impl(
            tier_capacity_bytes,
            policy,
            maximum_operations,
            incoming_reserved_bytes,
            false,
        )
    }

    fn maintain_capacity_impl(
        &self,
        tier_capacity_bytes: u64,
        policy: UtilizationPolicy,
        maximum_operations: usize,
        incoming_reserved_bytes: u64,
        fidelity_demotion: bool,
    ) -> Result<EvictionReport, StoreError> {
        let policy = policy.validate()?;
        if tier_capacity_bytes == 0 || maximum_operations == 0 {
            return Err(StoreError::State(
                "capacity maintenance bounds must be nonzero",
            ));
        }
        self.flush_access_epochs()?;
        let stat = self.stat()?;
        let non_durable = stat.reserved_bytes.saturating_add(incoming_reserved_bytes);
        let before = stat.durable_bytes.saturating_add(non_durable);
        let utilization = before as f64 / (tier_capacity_bytes as f64);
        let emergency = utilization >= policy.emergency_eviction;
        if utilization < policy.begin_eviction {
            return Ok(EvictionReport {
                before_bytes: before,
                after_bytes: before,
                manifests_evicted: 0,
                chunks_evicted: 0,
                emergency,
                blocked: false,
            });
        }
        let target = (tier_capacity_bytes as f64 * policy.stop_eviction) as u64;
        let target_durable = target.saturating_sub(non_durable);
        // Track durable bytes locally (decrementing per evicted batch)
        // instead of re-running the full stat() scan per eviction;
        // `after_bytes` below still comes from a fresh stat().
        let mut durable = stat.durable_bytes;
        let mut manifests_evicted = 0u64;
        let mut chunks_evicted = 0u64;
        let mut operations = 0usize;
        let mut blocked = false;
        if fidelity_demotion && durable > target_durable {
            // Watermark pressure demotes the coldest objects exactly one
            // rung per maintenance call instead of deleting them; the
            // standard eviction loop below runs only when nothing remains
            // demotable, so eviction never skips ahead of the ladder.
            let demotion =
                self.demote_fidelity_one_rung(maximum_operations.min(EVICTION_VICTIM_BATCH))?;
            if demotion.demoted > 0 {
                let after = self.stat()?.durable_bytes.saturating_add(non_durable);
                return Ok(EvictionReport {
                    before_bytes: before,
                    after_bytes: after,
                    manifests_evicted: 0,
                    chunks_evicted: 0,
                    emergency,
                    blocked: after > target,
                });
            }
        }
        while durable > target_durable && operations < maximum_operations {
            let remaining = maximum_operations - operations;
            let (chunk_freed, evicted) =
                self.gc_chunk_batch(remaining.min(EVICTION_VICTIM_BATCH))?;
            if evicted > 0 {
                chunks_evicted += evicted;
                operations += evicted as usize;
                durable = durable.saturating_sub(chunk_freed);
                continue;
            }
            let Some(manifest_freed) = self.evict_manifest_one_inner()? else {
                blocked = true;
                break;
            };
            manifests_evicted += 1;
            operations += 1;
            durable = durable.saturating_sub(manifest_freed);
        }
        let after = self.stat()?.durable_bytes.saturating_add(non_durable);
        if operations == maximum_operations && after > target {
            blocked = true;
        }
        Ok(EvictionReport {
            before_bytes: before,
            after_bytes: after,
            manifests_evicted,
            chunks_evicted,
            emergency,
            blocked,
        })
    }
}
