//! Bounded operational metrics and opaque OpenTelemetry span records.
//!
//! Every label value is a closed enum. Workload strings, tenant/model/cache
//! identities, tokens, capabilities, paths, and payload bytes have no input
//! position in this API.

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use crate::{StoreError, TELEMETRY_SCHEMA_VERSION};

mod labels;
mod metrics;
mod otlp_http;
mod trace;

pub use labels::{
    AdmissionResult, AuditOutcome, ByteCounter, CacheLifecycle, FallbackReason, HealthComponent,
    LookupResult, ResourceGauge, ServiceComponent, SpanOutcome, TracePhase, WorkClass,
};
pub use metrics::{OperationalTelemetry, MAX_PROMETHEUS_TEXT_BYTES, MAX_TRACE_BUFFER_SPANS};
pub use otlp_http::{
    OtlpHttpExporter, MAX_OTLP_HTTP_REQUEST_BYTES, MAX_OTLP_HTTP_RESPONSE_BYTES,
    MAX_OTLP_HTTP_TIMEOUT,
};
pub use trace::{
    OpaqueSpanId, OtlpExportOutcome, OtlpSpanExporter, OtlpSpanRecord, TraceCarrier, TraceContext,
    MAX_OTLP_BATCH_SPANS, TRACE_CARRIER_BYTES, TRACE_CARRIER_HEX_BYTES, TRACE_CARRIER_VERSION,
};

#[cfg(test)]
include!("telemetry/tests/inline.rs");
