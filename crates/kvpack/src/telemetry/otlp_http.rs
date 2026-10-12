use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::time::Duration;

use prost::Message;

use super::{OtlpExportOutcome, OtlpSpanExporter, OtlpSpanRecord, ServiceComponent, SpanOutcome};
use crate::error::io_error;
use crate::StoreError;

mod http;
mod wire;

use http::{read_http_response, HttpResponse};
use wire::{
    otlp_span, string_attribute, ExportTraceServiceRequest, ExportTraceServiceResponse,
    InstrumentationScope, Resource, ResourceSpans, ScopeSpans,
};

pub const MAX_OTLP_HTTP_REQUEST_BYTES: usize = 512 * 1024;
pub const MAX_OTLP_HTTP_RESPONSE_BYTES: usize = 64 * 1024;
pub const MAX_OTLP_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HTTP_HEADER_BYTES: usize = 16 * 1024;
const OTLP_TRACE_PATH: &str = "/v1/traces";
const OTLP_CONTENT_TYPE: &str = "application/x-protobuf";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtlpHttpExporter {
    address: SocketAddr,
    authority: String,
    request_timeout: Duration,
}

impl OtlpHttpExporter {
    /// Construct a binary-protobuf OTLP/HTTP exporter to a local collector.
    ///
    /// Production services deliberately accept only the canonical loopback
    /// endpoint `http://IP:PORT/v1/traces`. The local collector owns remote
    /// authentication, buffering, and backend-specific export policy.
    pub fn new(endpoint: &str, request_timeout: Duration) -> Result<Self, StoreError> {
        if request_timeout.is_zero() || request_timeout > MAX_OTLP_HTTP_TIMEOUT {
            return Err(StoreError::Expectation(
                "OTLP HTTP timeout is outside its fixed bound",
            ));
        }
        let authority_and_path =
            endpoint
                .strip_prefix("http://")
                .ok_or(StoreError::Expectation(
                    "OTLP HTTP endpoint must use canonical loopback HTTP",
                ))?;
        let split = authority_and_path.find('/').ok_or(StoreError::Expectation(
            "OTLP HTTP endpoint must include the trace path",
        ))?;
        let (authority, path) = authority_and_path.split_at(split);
        let address: SocketAddr = authority.parse().map_err(|_| {
            StoreError::Expectation("OTLP HTTP endpoint authority is not a socket address")
        })?;
        if !address.ip().is_loopback()
            || address.port() == 0
            || path != OTLP_TRACE_PATH
            || authority != address.to_string()
            || endpoint != format!("http://{address}{OTLP_TRACE_PATH}")
        {
            return Err(StoreError::Expectation(
                "OTLP HTTP endpoint is not canonical loopback /v1/traces",
            ));
        }
        Ok(Self {
            address,
            authority: authority.to_owned(),
            request_timeout,
        })
    }

    pub const fn address(&self) -> SocketAddr {
        self.address
    }

    pub const fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn encode_request(spans: &[OtlpSpanRecord]) -> Result<Vec<u8>, StoreError> {
        let mut resource_spans = Vec::new();
        for service in ServiceComponent::ALL {
            let encoded_spans = spans
                .iter()
                .filter(|span| span.service() == *service)
                .map(otlp_span)
                .collect::<Vec<_>>();
            if encoded_spans.is_empty() {
                continue;
            }
            resource_spans.push(ResourceSpans {
                resource: Some(Resource {
                    attributes: vec![
                        string_attribute("service.name", service.label()),
                        string_attribute(
                            "kvpack.telemetry.schema_version",
                            &crate::TELEMETRY_SCHEMA_VERSION.to_string(),
                        ),
                    ],
                }),
                scope_spans: vec![ScopeSpans {
                    scope: Some(InstrumentationScope {
                        name: "kvpack".to_owned(),
                        version: env!("CARGO_PKG_VERSION").to_owned(),
                    }),
                    spans: encoded_spans,
                }],
            });
        }
        let request = ExportTraceServiceRequest { resource_spans };
        let encoded_len = request.encoded_len();
        if encoded_len == 0 || encoded_len > MAX_OTLP_HTTP_REQUEST_BYTES {
            return Err(StoreError::State(
                "OTLP HTTP request is outside its fixed byte bound",
            ));
        }
        let mut encoded = Vec::with_capacity(encoded_len);
        request
            .encode(&mut encoded)
            .map_err(|_| StoreError::State("OTLP protobuf request encoding failed"))?;
        Ok(encoded)
    }

    fn send(&self, body: &[u8]) -> Result<HttpResponse, StoreError> {
        let mut stream = TcpStream::connect_timeout(&self.address, self.request_timeout)
            .map_err(io_error("connect OTLP HTTP collector"))?;
        stream
            .set_read_timeout(Some(self.request_timeout))
            .map_err(io_error("set OTLP HTTP read timeout"))?;
        stream
            .set_write_timeout(Some(self.request_timeout))
            .map_err(io_error("set OTLP HTTP write timeout"))?;
        let header = format!(
            "POST {OTLP_TRACE_PATH} HTTP/1.1\r\nHost: {}\r\nContent-Type: {OTLP_CONTENT_TYPE}\r\nAccept: {OTLP_CONTENT_TYPE}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.authority,
            body.len()
        );
        if header.len() > MAX_HTTP_HEADER_BYTES {
            return Err(StoreError::State("OTLP HTTP request header is too large"));
        }
        stream
            .write_all(header.as_bytes())
            .map_err(io_error("write OTLP HTTP request header"))?;
        stream
            .write_all(body)
            .map_err(io_error("write OTLP HTTP request body"))?;
        stream
            .flush()
            .map_err(io_error("flush OTLP HTTP request"))?;
        let response = read_http_response(&mut stream)?;
        let _ = stream.shutdown(Shutdown::Both);
        Ok(response)
    }
}

impl OtlpSpanExporter for OtlpHttpExporter {
    fn export(&self, spans: &[OtlpSpanRecord]) -> Result<OtlpExportOutcome, StoreError> {
        if spans.is_empty() || spans.len() > super::MAX_OTLP_BATCH_SPANS {
            return Err(StoreError::State(
                "OTLP HTTP span batch is outside its bound",
            ));
        }
        let body = Self::encode_request(spans)?;
        let response = self.send(&body)?;
        if response.status == 200 {
            if response.content_type.as_deref() != Some(OTLP_CONTENT_TYPE) {
                return Err(StoreError::State(
                    "OTLP HTTP success response has the wrong content type",
                ));
            }
            let decoded = ExportTraceServiceResponse::decode(response.body.as_slice())
                .map_err(|_| StoreError::State("OTLP HTTP success response is not protobuf"))?;
            let rejected = match decoded.partial_success {
                Some(partial) => usize::try_from(partial.rejected_spans).map_err(|_| {
                    StoreError::State("OTLP HTTP partial rejection count is invalid")
                })?,
                None => 0,
            };
            if rejected > spans.len() {
                return Err(StoreError::State(
                    "OTLP HTTP partial rejection exceeds the request batch",
                ));
            }
            return Ok(OtlpExportOutcome::new(spans.len() - rejected, rejected));
        }
        if matches!(response.status, 429 | 502 | 503 | 504) {
            return Err(StoreError::State("OTLP HTTP collector requested retry"));
        }
        Ok(OtlpExportOutcome::new(0, spans.len()))
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    use super::http::find_bytes;
    use super::wire::{any_value, ExportTracePartialSuccess, KeyValue, SpanKind, StatusCode};
    use super::*;
    use crate::{OperationalTelemetry, ServiceComponent, SpanOutcome, TraceContext, TracePhase};

    #[derive(Clone, Copy)]
    enum TestResponse {
        Full,
        Partial,
        Retryable,
        Permanent,
    }

    fn collector(
        response: TestResponse,
    ) -> (
        OtlpHttpExporter,
        mpsc::Receiver<ExportTraceServiceRequest>,
        thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let exporter = OtlpHttpExporter::new(
            &format!("http://{address}/v1/traces"),
            Duration::from_secs(2),
        )
        .unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let header_end = loop {
                if let Some(position) = find_bytes(&request, b"\r\n\r\n") {
                    break position + 4;
                }
                let mut buffer = [0u8; 1024];
                let read = stream.read(&mut buffer).unwrap();
                assert_ne!(read, 0);
                request.extend_from_slice(&buffer[..read]);
            };
            let header = std::str::from_utf8(&request[..header_end]).unwrap();
            assert!(header.starts_with("POST /v1/traces HTTP/1.1\r\n"));
            assert!(header.contains("Content-Type: application/x-protobuf\r\n"));
            let content_length = header
                .split("\r\n")
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            while request.len() - header_end < content_length {
                let mut buffer = [0u8; 1024];
                let read = stream.read(&mut buffer).unwrap();
                assert_ne!(read, 0);
                request.extend_from_slice(&buffer[..read]);
            }
            assert_eq!(request.len() - header_end, content_length);
            sender
                .send(ExportTraceServiceRequest::decode(&request[header_end..]).unwrap())
                .unwrap();
            let partial_body = ExportTraceServiceResponse {
                partial_success: Some(ExportTracePartialSuccess {
                    rejected_spans: 1,
                    error_message: "one rejected".to_owned(),
                }),
            }
            .encode_to_vec();
            match response {
                TestResponse::Full => stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .unwrap(),
                TestResponse::Partial => {
                    let header = b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
                    stream.write_all(header).unwrap();
                    stream
                        .write_all(format!("{:x}\r\n", partial_body.len()).as_bytes())
                        .unwrap();
                    stream.write_all(&partial_body).unwrap();
                    stream.write_all(b"\r\n0\r\n\r\n").unwrap();
                }
                TestResponse::Retryable => stream
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/x-protobuf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .unwrap(),
                TestResponse::Permanent => stream
                    .write_all(
                        b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/x-protobuf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .unwrap(),
            }
        });
        (exporter, receiver, worker)
    }

    fn record(telemetry: &OperationalTelemetry, outcome: SpanOutcome, offset: u64) {
        let context = TraceContext::new(ServiceComponent::Agent).unwrap();
        telemetry
            .record_span(
                &context,
                None,
                TracePhase::Transfer,
                outcome,
                1_000 + offset,
                2_000 + offset,
            )
            .unwrap();
    }

    fn attribute<'a>(attributes: &'a [KeyValue], key: &str) -> &'a str {
        attributes
            .iter()
            .find(|attribute| attribute.key == key)
            .and_then(|attribute| attribute.value.as_ref())
            .and_then(|value| value.value.as_ref())
            .map(|value| match value {
                any_value::Value::StringValue(value) => value.as_str(),
            })
            .unwrap()
    }

    #[test]
    fn exporter_sends_canonical_bounded_otlp_protobuf() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        record(&telemetry, SpanOutcome::Ok, 0);
        let (exporter, request, worker) = collector(TestResponse::Full);
        let outcome = telemetry.export_otlp_batch_report(&exporter, 8).unwrap();
        assert_eq!(outcome, OtlpExportOutcome::new(1, 0));
        assert_eq!(telemetry.pending_spans().unwrap(), 0);
        let request = request.recv().unwrap();
        assert_eq!(request.resource_spans.len(), 1);
        let resource = &request.resource_spans[0];
        let attributes = &resource.resource.as_ref().unwrap().attributes;
        assert_eq!(attribute(attributes, "service.name"), "kvpack-agent");
        assert_eq!(
            attribute(attributes, "kvpack.telemetry.schema_version"),
            "1"
        );
        let scope = &resource.scope_spans[0];
        assert_eq!(scope.scope.as_ref().unwrap().name, "kvpack");
        let span = &scope.spans[0];
        assert_eq!(span.trace_id.len(), 16);
        assert_eq!(span.span_id.len(), 8);
        assert!(span.parent_span_id.is_empty());
        assert_eq!(span.name, "kvpack.transfer");
        assert_eq!(span.kind, SpanKind::Internal as i32);
        assert_eq!(attribute(&span.attributes, "kvpack.phase"), "transfer");
        assert_eq!(attribute(&span.attributes, "kvpack.outcome"), "ok");
        assert_eq!(span.status.as_ref().unwrap().code, StatusCode::Ok as i32);
        worker.join().unwrap();
    }

    #[test]
    fn partial_success_dequeues_and_counts_only_rejected_spans() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        record(&telemetry, SpanOutcome::Ok, 0);
        record(&telemetry, SpanOutcome::Unavailable, 1);
        let (exporter, request, worker) = collector(TestResponse::Partial);
        let outcome = telemetry.export_otlp_batch_report(&exporter, 8).unwrap();
        assert_eq!(outcome, OtlpExportOutcome::new(1, 1));
        assert_eq!(
            request.recv().unwrap().resource_spans[0].scope_spans[0]
                .spans
                .len(),
            2
        );
        assert_eq!(telemetry.pending_spans().unwrap(), 0);
        assert!(telemetry
            .prometheus_text()
            .unwrap()
            .contains("kvpack_trace_dropped_total 1"));
        worker.join().unwrap();
    }

    #[test]
    fn retryable_failure_retains_but_permanent_failure_drops_the_batch() {
        let retry = OperationalTelemetry::new(8).unwrap();
        record(&retry, SpanOutcome::Unavailable, 0);
        let (exporter, request, worker) = collector(TestResponse::Retryable);
        assert!(retry.export_otlp_batch_report(&exporter, 8).is_err());
        assert_eq!(request.recv().unwrap().resource_spans.len(), 1);
        assert_eq!(retry.pending_spans().unwrap(), 1);
        worker.join().unwrap();

        let permanent = OperationalTelemetry::new(8).unwrap();
        record(&permanent, SpanOutcome::Rejected, 0);
        let (exporter, request, worker) = collector(TestResponse::Permanent);
        let outcome = permanent.export_otlp_batch_report(&exporter, 8).unwrap();
        assert_eq!(outcome, OtlpExportOutcome::new(0, 1));
        assert_eq!(request.recv().unwrap().resource_spans.len(), 1);
        assert_eq!(permanent.pending_spans().unwrap(), 0);
        assert!(permanent
            .prometheus_text()
            .unwrap()
            .contains("kvpack_trace_dropped_total 1"));
        worker.join().unwrap();
    }

    #[test]
    fn endpoint_and_timeout_surface_is_closed() {
        assert!(
            OtlpHttpExporter::new("http://127.0.0.1:4318/v1/traces", Duration::from_secs(1))
                .is_ok()
        );
        assert!(
            OtlpHttpExporter::new("http://[::1]:4318/v1/traces", Duration::from_secs(1)).is_ok()
        );
        for endpoint in [
            "https://127.0.0.1:4318/v1/traces",
            "http://localhost:4318/v1/traces",
            "http://127.0.0.1:0/v1/traces",
            "http://127.0.0.1:4318/other",
            "http://192.0.2.1:4318/v1/traces",
            "http://127.0.0.1:4318/v1/traces?x=1",
        ] {
            assert!(OtlpHttpExporter::new(endpoint, Duration::from_secs(1)).is_err());
        }
        assert!(OtlpHttpExporter::new("http://127.0.0.1:4318/v1/traces", Duration::ZERO).is_err());
        assert!(OtlpHttpExporter::new(
            "http://127.0.0.1:4318/v1/traces",
            MAX_OTLP_HTTP_TIMEOUT + Duration::from_secs(1)
        )
        .is_err());
    }
}
