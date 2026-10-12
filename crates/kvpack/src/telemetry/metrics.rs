use super::*;

pub const MAX_PROMETHEUS_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_TRACE_BUFFER_SPANS: usize = 16_384;
const LATENCY_BUCKET_NS: [u64; 12] = [
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    5_000_000_000,
    u64::MAX,
];

pub(super) struct TelemetryState {
    admissions: [u64; AdmissionResult::COUNT],
    queue_depth: [u64; WorkClass::COUNT],
    lookups: [u64; LookupResult::COUNT],
    bytes: [u64; ByteCounter::COUNT],
    latency_bucket: [[u64; LATENCY_BUCKET_NS.len()]; TracePhase::COUNT],
    latency_count: [u64; TracePhase::COUNT],
    latency_sum_ns: [u64; TracePhase::COUNT],
    resources: [u64; ResourceGauge::COUNT],
    lifecycle: [u64; CacheLifecycle::COUNT],
    health: [u64; HealthComponent::COUNT],
    fallbacks: [u64; FallbackReason::COUNT],
    audit: [u64; AuditOutcome::COUNT],
    pub(super) dropped_spans: u64,
    pub(super) spans: VecDeque<OtlpSpanRecord>,
}

impl TelemetryState {
    fn new() -> Self {
        Self {
            admissions: [0; AdmissionResult::COUNT],
            queue_depth: [0; WorkClass::COUNT],
            lookups: [0; LookupResult::COUNT],
            bytes: [0; ByteCounter::COUNT],
            latency_bucket: [[0; LATENCY_BUCKET_NS.len()]; TracePhase::COUNT],
            latency_count: [0; TracePhase::COUNT],
            latency_sum_ns: [0; TracePhase::COUNT],
            resources: [0; ResourceGauge::COUNT],
            lifecycle: [0; CacheLifecycle::COUNT],
            health: [0; HealthComponent::COUNT],
            fallbacks: [0; FallbackReason::COUNT],
            audit: [0; AuditOutcome::COUNT],
            dropped_spans: 0,
            spans: VecDeque::new(),
        }
    }
}

pub struct OperationalTelemetry {
    pub(super) maximum_spans: usize,
    pub(super) state: Mutex<TelemetryState>,
    pub(super) export_serial: Mutex<()>,
}

impl fmt::Debug for OperationalTelemetry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OperationalTelemetry")
            .field("maximum_spans", &self.maximum_spans)
            .field("state", &"[bounded]")
            .finish_non_exhaustive()
    }
}

impl OperationalTelemetry {
    pub fn new(maximum_spans: usize) -> Result<Self, StoreError> {
        if maximum_spans == 0 || maximum_spans > MAX_TRACE_BUFFER_SPANS {
            return Err(StoreError::State("telemetry trace-buffer bound is invalid"));
        }
        Ok(Self {
            maximum_spans,
            state: Mutex::new(TelemetryState::new()),
            export_serial: Mutex::new(()),
        })
    }

    pub fn record_admission(&self, result: AdmissionResult) -> Result<(), StoreError> {
        increment(&mut self.lock()?.admissions[result as usize], 1);
        Ok(())
    }

    pub fn set_queue_depth(&self, class: WorkClass, depth: u64) -> Result<(), StoreError> {
        self.lock()?.queue_depth[class as usize] = depth;
        Ok(())
    }

    pub fn record_lookup(&self, result: LookupResult) -> Result<(), StoreError> {
        increment(&mut self.lock()?.lookups[result as usize], 1);
        Ok(())
    }

    pub fn add_bytes(&self, counter: ByteCounter, bytes: u64) -> Result<(), StoreError> {
        increment(&mut self.lock()?.bytes[counter as usize], bytes);
        Ok(())
    }

    pub fn observe_latency(&self, phase: TracePhase, duration: Duration) -> Result<(), StoreError> {
        let elapsed = duration.as_nanos().min(u64::MAX as u128) as u64;
        let mut state = self.lock()?;
        let phase_index = phase as usize;
        let bucket = LATENCY_BUCKET_NS
            .iter()
            .position(|limit| elapsed <= *limit)
            .unwrap_or(LATENCY_BUCKET_NS.len() - 1);
        increment(&mut state.latency_bucket[phase_index][bucket], 1);
        increment(&mut state.latency_count[phase_index], 1);
        increment(&mut state.latency_sum_ns[phase_index], elapsed);
        Ok(())
    }

    pub fn set_resource(&self, resource: ResourceGauge, value: u64) -> Result<(), StoreError> {
        self.lock()?.resources[resource as usize] = value;
        Ok(())
    }

    pub fn record_lifecycle(&self, lifecycle: CacheLifecycle) -> Result<(), StoreError> {
        self.record_lifecycle_count(lifecycle, 1)
    }

    pub fn record_lifecycle_count(
        &self,
        lifecycle: CacheLifecycle,
        count: u64,
    ) -> Result<(), StoreError> {
        increment(&mut self.lock()?.lifecycle[lifecycle as usize], count);
        Ok(())
    }

    pub fn set_health(&self, component: HealthComponent, healthy: bool) -> Result<(), StoreError> {
        self.lock()?.health[component as usize] = u64::from(healthy);
        Ok(())
    }

    pub fn record_fallback(&self, reason: FallbackReason) -> Result<(), StoreError> {
        increment(&mut self.lock()?.fallbacks[reason as usize], 1);
        Ok(())
    }

    pub fn record_audit(&self, outcome: AuditOutcome) -> Result<(), StoreError> {
        self.record_audit_count(outcome, 1)
    }

    pub fn record_audit_count(&self, outcome: AuditOutcome, count: u64) -> Result<(), StoreError> {
        increment(&mut self.lock()?.audit[outcome as usize], count);
        Ok(())
    }

    pub fn prometheus_text(&self) -> Result<String, StoreError> {
        let state = self.lock()?;
        let mut output = String::with_capacity(16 * 1024);
        metric_header(
            &mut output,
            "kvpack_admission_total",
            "counter",
            "Cache work admission outcomes.",
        )?;
        for label in AdmissionResult::ALL {
            writeln!(
                output,
                "kvpack_admission_total{{result=\"{}\"}} {}",
                label.label(),
                state.admissions[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_queue_depth",
            "gauge",
            "Current bounded scheduler queue depth.",
        )?;
        for label in WorkClass::ALL {
            writeln!(
                output,
                "kvpack_queue_depth{{class=\"{}\"}} {}",
                label.label(),
                state.queue_depth[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_lookup_total",
            "counter",
            "Exact cache lookup outcomes.",
        )?;
        for label in LookupResult::ALL {
            writeln!(
                output,
                "kvpack_lookup_total{{result=\"{}\"}} {}",
                label.label(),
                state.lookups[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_bytes_total",
            "counter",
            "Cache data bytes by closed accounting class.",
        )?;
        for label in ByteCounter::ALL {
            writeln!(
                output,
                "kvpack_bytes_total{{kind=\"{}\"}} {}",
                label.label(),
                state.bytes[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_phase_latency_seconds",
            "histogram",
            "Operation phase latency with fixed buckets.",
        )?;
        for phase in TracePhase::ALL {
            let phase_index = *phase as usize;
            let mut cumulative = 0u64;
            for (index, limit) in LATENCY_BUCKET_NS.iter().enumerate() {
                cumulative = cumulative.saturating_add(state.latency_bucket[phase_index][index]);
                let le = if *limit == u64::MAX {
                    "+Inf".to_owned()
                } else {
                    format_decimal_seconds(*limit)
                };
                writeln!(
                    output,
                    "kvpack_phase_latency_seconds_bucket{{phase=\"{}\",le=\"{}\"}} {}",
                    phase.label(),
                    le,
                    cumulative
                )?;
            }
            writeln!(
                output,
                "kvpack_phase_latency_seconds_sum{{phase=\"{}\"}} {}",
                phase.label(),
                format_decimal_seconds(state.latency_sum_ns[phase_index])
            )?;
            writeln!(
                output,
                "kvpack_phase_latency_seconds_count{{phase=\"{}\"}} {}",
                phase.label(),
                state.latency_count[phase_index]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_resource_current",
            "gauge",
            "Current cache resource accounting.",
        )?;
        for label in ResourceGauge::ALL {
            writeln!(
                output,
                "kvpack_resource_current{{kind=\"{}\"}} {}",
                label.label(),
                state.resources[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_lifecycle_total",
            "counter",
            "Cache publication and retention lifecycle transitions.",
        )?;
        for label in CacheLifecycle::ALL {
            writeln!(
                output,
                "kvpack_lifecycle_total{{transition=\"{}\"}} {}",
                label.label(),
                state.lifecycle[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_health",
            "gauge",
            "Component readiness where one is healthy.",
        )?;
        for label in HealthComponent::ALL {
            writeln!(
                output,
                "kvpack_health{{component=\"{}\"}} {}",
                label.label(),
                state.health[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_fallback_total",
            "counter",
            "Clean cache fallback reasons.",
        )?;
        for label in FallbackReason::ALL {
            writeln!(
                output,
                "kvpack_fallback_total{{reason=\"{}\"}} {}",
                label.label(),
                state.fallbacks[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_audit_total",
            "counter",
            "Durable publication audit outbox outcomes.",
        )?;
        for label in AuditOutcome::ALL {
            writeln!(
                output,
                "kvpack_audit_total{{outcome=\"{}\"}} {}",
                label.label(),
                state.audit[*label as usize]
            )?;
        }
        metric_header(
            &mut output,
            "kvpack_trace_dropped_total",
            "counter",
            "Spans dropped when the bounded OTLP queue is full.",
        )?;
        writeln!(output, "kvpack_trace_dropped_total {}", state.dropped_spans)?;
        if output.len() > MAX_PROMETHEUS_TEXT_BYTES {
            return Err(StoreError::State(
                "Prometheus exposition exceeded its fixed bound",
            ));
        }
        Ok(output)
    }

    pub(super) fn lock(&self) -> Result<MutexGuard<'_, TelemetryState>, StoreError> {
        self.state
            .lock()
            .map_err(|_| StoreError::State("telemetry state mutex poisoned"))
    }
}

impl Default for OperationalTelemetry {
    fn default() -> Self {
        Self::new(4_096).expect("default telemetry bounds are valid")
    }
}

fn metric_header(
    output: &mut String,
    name: &str,
    metric_type: &str,
    help: &str,
) -> Result<(), fmt::Error> {
    writeln!(output, "# HELP {name} {help}")?;
    writeln!(output, "# TYPE {name} {metric_type}")
}

fn format_decimal_seconds(nanoseconds: u64) -> String {
    format!(
        "{}.{:09}",
        nanoseconds / 1_000_000_000,
        nanoseconds % 1_000_000_000
    )
}

pub(super) fn increment(value: &mut u64, amount: u64) {
    *value = value.saturating_add(amount);
}

impl From<fmt::Error> for StoreError {
    fn from(_: fmt::Error) -> Self {
        StoreError::State("formatting bounded telemetry failed")
    }
}
