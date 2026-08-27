use super::metrics::increment;
use super::*;

pub const MAX_OTLP_BATCH_SPANS: usize = 512;
pub const TRACE_CARRIER_VERSION: u32 = 1;
pub const TRACE_CARRIER_BYTES: usize = 32;
pub const TRACE_CARRIER_HEX_BYTES: usize = TRACE_CARRIER_BYTES * 2;
const MAX_SPAN_DURATION_NS: u64 = 24 * 60 * 60 * 1_000_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OpaqueSpanId([u8; 8]);

impl fmt::Debug for OpaqueSpanId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaqueSpanId([opaque])")
    }
}

impl OpaqueSpanId {
    /// Raw OTLP span identity for exporters. The value is random per trace and
    /// is never derived from tenant, model, cache, request, or payload data.
    pub const fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    fn random() -> Result<Self, StoreError> {
        let mut raw = [0; 8];
        getrandom::fill(&mut raw)
            .map_err(|_| StoreError::State("telemetry span entropy unavailable"))?;
        if raw == [0; 8] {
            return Err(StoreError::State("telemetry span identity is zero"));
        }
        Ok(Self(raw))
    }
}

/// Fixed-width opaque context propagated between cache services.
///
/// The carrier contains only random correlation identities. It has no tenant,
/// model, token, request, capability, path, or payload field and grants no
/// authority. Binary and lowercase-hex encodings are canonical and bounded.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TraceCarrier {
    trace_id: [u8; 16],
    parent_span_id: OpaqueSpanId,
}

impl TraceCarrier {
    pub fn decode(encoded: &[u8]) -> Result<Self, StoreError> {
        if encoded.len() != TRACE_CARRIER_BYTES {
            return Err(StoreError::Expectation(
                "trace carrier length is not canonical",
            ));
        }
        let version = u32::from_le_bytes(
            encoded[0..4]
                .try_into()
                .map_err(|_| StoreError::State("trace carrier version slice failed"))?,
        );
        if version != TRACE_CARRIER_VERSION {
            return Err(StoreError::Expectation(
                "trace carrier version is unsupported",
            ));
        }
        if encoded[28..32] != [0; 4] {
            return Err(StoreError::Expectation(
                "trace carrier reserved bytes are nonzero",
            ));
        }
        let trace_id: [u8; 16] = encoded[4..20]
            .try_into()
            .map_err(|_| StoreError::State("trace carrier identity slice failed"))?;
        if trace_id == [0; 16] {
            return Err(StoreError::Expectation("trace carrier identity is zero"));
        }
        let parent_span_id: [u8; 8] = encoded[20..28]
            .try_into()
            .map_err(|_| StoreError::State("trace carrier parent slice failed"))?;
        if parent_span_id == [0; 8] {
            return Err(StoreError::Expectation(
                "trace carrier parent identity is zero",
            ));
        }
        Ok(Self {
            trace_id,
            parent_span_id: OpaqueSpanId(parent_span_id),
        })
    }

    pub fn decode_lower_hex(encoded: &[u8]) -> Result<Self, StoreError> {
        if encoded.len() != TRACE_CARRIER_HEX_BYTES {
            return Err(StoreError::Expectation(
                "trace carrier hex length is not canonical",
            ));
        }
        let mut binary = [0; TRACE_CARRIER_BYTES];
        for (output, pair) in binary.iter_mut().zip(encoded.chunks_exact(2)) {
            *output = lower_hex_nibble(pair[0])?
                .checked_mul(16)
                .and_then(|high| high.checked_add(lower_hex_nibble(pair[1]).ok()?))
                .ok_or(StoreError::Expectation(
                    "trace carrier hex is not canonical lowercase",
                ))?;
        }
        Self::decode(&binary)
    }

    pub fn encode(self) -> [u8; TRACE_CARRIER_BYTES] {
        let mut encoded = [0; TRACE_CARRIER_BYTES];
        encoded[0..4].copy_from_slice(&TRACE_CARRIER_VERSION.to_le_bytes());
        encoded[4..20].copy_from_slice(&self.trace_id);
        encoded[20..28].copy_from_slice(self.parent_span_id.as_bytes());
        encoded
    }

    pub fn encode_lower_hex(self) -> [u8; TRACE_CARRIER_HEX_BYTES] {
        const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
        let binary = self.encode();
        let mut encoded = [0; TRACE_CARRIER_HEX_BYTES];
        for (index, byte) in binary.iter().copied().enumerate() {
            encoded[index * 2] = LOWER_HEX[usize::from(byte >> 4)];
            encoded[index * 2 + 1] = LOWER_HEX[usize::from(byte & 0x0f)];
        }
        encoded
    }

    pub const fn parent_span_id(self) -> OpaqueSpanId {
        self.parent_span_id
    }
}

impl fmt::Debug for TraceCarrier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TraceCarrier([opaque])")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct TraceContext {
    service: ServiceComponent,
    trace_id: [u8; 16],
}

impl TraceContext {
    pub fn new(service: ServiceComponent) -> Result<Self, StoreError> {
        let mut trace_id = [0; 16];
        getrandom::fill(&mut trace_id)
            .map_err(|_| StoreError::State("telemetry trace entropy unavailable"))?;
        if trace_id == [0; 16] {
            return Err(StoreError::State("telemetry trace identity is zero"));
        }
        Ok(Self { service, trace_id })
    }

    pub const fn service(&self) -> ServiceComponent {
        self.service
    }

    pub const fn from_carrier(service: ServiceComponent, carrier: TraceCarrier) -> Self {
        Self {
            service,
            trace_id: carrier.trace_id,
        }
    }

    pub const fn carrier(&self, parent_span_id: OpaqueSpanId) -> TraceCarrier {
        TraceCarrier {
            trace_id: self.trace_id,
            parent_span_id,
        }
    }
}

impl fmt::Debug for TraceContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TraceContext")
            .field("service", &self.service)
            .field("trace_id", &"[opaque]")
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct OtlpSpanRecord {
    schema_version: u32,
    service: ServiceComponent,
    trace_id: [u8; 16],
    span_id: OpaqueSpanId,
    parent_span_id: Option<OpaqueSpanId>,
    phase: TracePhase,
    outcome: SpanOutcome,
    start_unix_ns: u64,
    end_unix_ns: u64,
}

impl OtlpSpanRecord {
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub const fn service(&self) -> ServiceComponent {
        self.service
    }

    pub const fn trace_id(&self) -> &[u8; 16] {
        &self.trace_id
    }

    pub const fn span_id(&self) -> OpaqueSpanId {
        self.span_id
    }

    pub const fn parent_span_id(&self) -> Option<OpaqueSpanId> {
        self.parent_span_id
    }

    pub const fn phase(&self) -> TracePhase {
        self.phase
    }

    pub const fn outcome(&self) -> SpanOutcome {
        self.outcome
    }

    pub const fn start_unix_ns(&self) -> u64 {
        self.start_unix_ns
    }

    pub const fn end_unix_ns(&self) -> u64 {
        self.end_unix_ns
    }
}

impl fmt::Debug for OtlpSpanRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OtlpSpanRecord")
            .field("schema_version", &self.schema_version)
            .field("service", &self.service)
            .field("trace_id", &"[opaque]")
            .field("span_id", &self.span_id)
            .field("parent_span_id", &self.parent_span_id)
            .field("phase", &self.phase)
            .field("outcome", &self.outcome)
            .field("start_unix_ns", &self.start_unix_ns)
            .field("end_unix_ns", &self.end_unix_ns)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OtlpExportOutcome {
    accepted_spans: usize,
    rejected_spans: usize,
}

impl OtlpExportOutcome {
    pub const fn new(accepted_spans: usize, rejected_spans: usize) -> Self {
        Self {
            accepted_spans,
            rejected_spans,
        }
    }

    pub const fn accepted_spans(self) -> usize {
        self.accepted_spans
    }

    pub const fn rejected_spans(self) -> usize {
        self.rejected_spans
    }

    pub const fn dequeued_spans(self) -> usize {
        self.accepted_spans.saturating_add(self.rejected_spans)
    }
}

pub trait OtlpSpanExporter {
    /// Export one already-bounded batch.
    ///
    /// Retryable errors retain the complete batch. A successful result must
    /// account for every input span as accepted or permanently rejected;
    /// rejected spans are removed and included in `kvpack_trace_dropped_total`.
    fn export(&self, spans: &[OtlpSpanRecord]) -> Result<OtlpExportOutcome, StoreError>;
}

impl OperationalTelemetry {
    pub fn record_span(
        &self,
        context: &TraceContext,
        parent_span_id: Option<OpaqueSpanId>,
        phase: TracePhase,
        outcome: SpanOutcome,
        start_unix_ns: u64,
        end_unix_ns: u64,
    ) -> Result<Option<OpaqueSpanId>, StoreError> {
        if start_unix_ns == 0
            || end_unix_ns < start_unix_ns
            || end_unix_ns - start_unix_ns > MAX_SPAN_DURATION_NS
        {
            return Err(StoreError::State("telemetry span time bounds are invalid"));
        }
        let span_id = OpaqueSpanId::random()?;
        let mut state = self.lock()?;
        if state.spans.len() == self.maximum_spans {
            increment(&mut state.dropped_spans, 1);
            return Ok(None);
        }
        state.spans.push_back(OtlpSpanRecord {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            service: context.service,
            trace_id: context.trace_id,
            span_id,
            parent_span_id,
            phase,
            outcome,
            start_unix_ns,
            end_unix_ns,
        });
        Ok(Some(span_id))
    }

    /// Record the engine-owned suffix work for an ancestor-cache restore.
    ///
    /// The engine supplies only a previously issued opaque carrier, a closed
    /// outcome, and bounded timestamps. It cannot add labels or exact-cache
    /// provenance, and this API does not require an engine-specific adapter.
    pub fn record_suffix_work(
        &self,
        carrier: TraceCarrier,
        outcome: SpanOutcome,
        start_unix_ns: u64,
        end_unix_ns: u64,
    ) -> Result<Option<OpaqueSpanId>, StoreError> {
        let context = TraceContext::from_carrier(ServiceComponent::Engine, carrier);
        self.record_span(
            &context,
            Some(carrier.parent_span_id()),
            TracePhase::SuffixWork,
            outcome,
            start_unix_ns,
            end_unix_ns,
        )
    }

    pub fn pending_spans(&self) -> Result<usize, StoreError> {
        Ok(self.lock()?.spans.len())
    }

    pub fn export_otlp_batch(
        &self,
        exporter: &dyn OtlpSpanExporter,
        maximum_spans: usize,
    ) -> Result<usize, StoreError> {
        Ok(self
            .export_otlp_batch_report(exporter, maximum_spans)?
            .dequeued_spans())
    }

    pub fn export_otlp_batch_report(
        &self,
        exporter: &dyn OtlpSpanExporter,
        maximum_spans: usize,
    ) -> Result<OtlpExportOutcome, StoreError> {
        if maximum_spans == 0 || maximum_spans > MAX_OTLP_BATCH_SPANS {
            return Err(StoreError::State("OTLP batch bound is invalid"));
        }
        let _serial = self
            .export_serial
            .lock()
            .map_err(|_| StoreError::State("telemetry export mutex poisoned"))?;
        let batch = {
            let state = self.lock()?;
            state
                .spans
                .iter()
                .take(maximum_spans)
                .cloned()
                .collect::<Vec<_>>()
        };
        if batch.is_empty() {
            return Ok(OtlpExportOutcome::new(0, 0));
        }
        let outcome = exporter.export(&batch)?;
        if outcome.accepted_spans.checked_add(outcome.rejected_spans) != Some(batch.len()) {
            return Err(StoreError::State(
                "OTLP exporter outcome does not account for its batch",
            ));
        }
        let mut state = self.lock()?;
        for expected in &batch {
            let front = state
                .spans
                .front()
                .ok_or(StoreError::State("telemetry export queue changed"))?;
            if front.trace_id != expected.trace_id || front.span_id != expected.span_id {
                return Err(StoreError::State("telemetry export queue changed"));
            }
            state.spans.pop_front();
        }
        increment(
            &mut state.dropped_spans,
            u64::try_from(outcome.rejected_spans).unwrap_or(u64::MAX),
        );
        Ok(outcome)
    }
}

fn lower_hex_nibble(byte: u8) -> Result<u8, StoreError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(StoreError::Expectation(
            "trace carrier hex is not canonical lowercase",
        )),
    }
}
