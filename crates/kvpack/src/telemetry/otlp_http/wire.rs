use super::*;

pub(super) fn otlp_span(record: &OtlpSpanRecord) -> Span {
    let status_code = match record.outcome() {
        SpanOutcome::Ok => 1,
        SpanOutcome::Rejected | SpanOutcome::Unavailable | SpanOutcome::IntegrityError => 2,
        SpanOutcome::Miss | SpanOutcome::Cancelled => 0,
    };
    Span {
        trace_id: record.trace_id().to_vec(),
        span_id: record.span_id().as_bytes().to_vec(),
        parent_span_id: record
            .parent_span_id()
            .map(|parent| parent.as_bytes().to_vec())
            .unwrap_or_default(),
        flags: 1,
        name: format!("kvpack.{}", record.phase().label()),
        kind: 1,
        start_time_unix_nano: record.start_unix_ns(),
        end_time_unix_nano: record.end_unix_ns(),
        attributes: vec![
            string_attribute("kvpack.phase", record.phase().label()),
            string_attribute("kvpack.outcome", record.outcome().label()),
        ],
        status: Some(Status { code: status_code }),
    }
}

pub(super) fn string_attribute(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.to_owned())),
        }),
    }
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct ExportTraceServiceRequest {
    #[prost(message, repeated, tag = "1")]
    pub(super) resource_spans: Vec<ResourceSpans>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct ExportTraceServiceResponse {
    #[prost(message, optional, tag = "1")]
    pub(super) partial_success: Option<ExportTracePartialSuccess>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct ExportTracePartialSuccess {
    #[prost(int64, tag = "1")]
    pub(super) rejected_spans: i64,
    #[prost(string, tag = "2")]
    pub(super) error_message: String,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct ResourceSpans {
    #[prost(message, optional, tag = "1")]
    pub(super) resource: Option<Resource>,
    #[prost(message, repeated, tag = "2")]
    pub(super) scope_spans: Vec<ScopeSpans>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct Resource {
    #[prost(message, repeated, tag = "1")]
    pub(super) attributes: Vec<KeyValue>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct ScopeSpans {
    #[prost(message, optional, tag = "1")]
    pub(super) scope: Option<InstrumentationScope>,
    #[prost(message, repeated, tag = "2")]
    pub(super) spans: Vec<Span>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct InstrumentationScope {
    #[prost(string, tag = "1")]
    pub(super) name: String,
    #[prost(string, tag = "2")]
    pub(super) version: String,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct Span {
    #[prost(bytes = "vec", tag = "1")]
    pub(super) trace_id: Vec<u8>,
    #[prost(bytes = "vec", tag = "2")]
    pub(super) span_id: Vec<u8>,
    #[prost(bytes = "vec", tag = "4")]
    pub(super) parent_span_id: Vec<u8>,
    #[prost(fixed32, tag = "16")]
    pub(super) flags: u32,
    #[prost(string, tag = "5")]
    pub(super) name: String,
    #[prost(enumeration = "SpanKind", tag = "6")]
    pub(super) kind: i32,
    #[prost(fixed64, tag = "7")]
    pub(super) start_time_unix_nano: u64,
    #[prost(fixed64, tag = "8")]
    pub(super) end_time_unix_nano: u64,
    #[prost(message, repeated, tag = "9")]
    pub(super) attributes: Vec<KeyValue>,
    #[prost(message, optional, tag = "15")]
    pub(super) status: Option<Status>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub(super) enum SpanKind {
    Internal = 1,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct Status {
    #[prost(enumeration = "StatusCode", tag = "3")]
    pub(super) code: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub(super) enum StatusCode {
    Unset = 0,
    Ok = 1,
    Error = 2,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct KeyValue {
    #[prost(string, tag = "1")]
    pub(super) key: String,
    #[prost(message, optional, tag = "2")]
    pub(super) value: Option<AnyValue>,
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct AnyValue {
    #[prost(oneof = "any_value::Value", tags = "1")]
    pub(super) value: Option<any_value::Value>,
}

pub(super) mod any_value {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub(in crate::telemetry::otlp_http) enum Value {
        #[prost(string, tag = "1")]
        StringValue(String),
    }
}
