#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use super::*;

    #[test]
    fn prometheus_surface_has_only_closed_labels_and_fixed_cardinality() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        telemetry
            .record_admission(AdmissionResult::Admitted)
            .unwrap();
        telemetry.set_queue_depth(WorkClass::Demand, 3).unwrap();
        telemetry.record_lookup(LookupResult::Ancestor).unwrap();
        telemetry.add_bytes(ByteCounter::Restored, 4096).unwrap();
        telemetry
            .observe_latency(TracePhase::Verification, Duration::from_millis(3))
            .unwrap();
        telemetry
            .set_resource(ResourceGauge::RestorePins, 2)
            .unwrap();
        telemetry
            .record_lifecycle(CacheLifecycle::Published)
            .unwrap();
        telemetry.set_health(HealthComponent::Store, true).unwrap();
        telemetry
            .record_fallback(FallbackReason::NoExactCut)
            .unwrap();
        telemetry.record_audit(AuditOutcome::Enqueued).unwrap();

        let text = telemetry.prometheus_text().unwrap();
        assert!(text.len() <= MAX_PROMETHEUS_TEXT_BYTES);
        assert!(text.contains("kvpack_admission_total{result=\"admitted\"} 1"));
        assert!(text.contains("kvpack_queue_depth{class=\"demand\"} 3"));
        assert!(text.contains("kvpack_lookup_total{result=\"ancestor\"} 1"));
        assert!(text.contains("kvpack_bytes_total{kind=\"restored\"} 4096"));
        assert!(text.contains("phase=\"verification\",le=\"0.005000000\"} 1"));
        assert!(text.contains("kvpack_resource_current{kind=\"restore_pins\"} 2"));
        assert!(text.contains("kvpack_health{component=\"store\"} 1"));
        assert!(text.contains("kvpack_audit_total{outcome=\"enqueued\"} 1"));
        for forbidden in [
            "tenant-canary",
            "prompt-canary",
            "token-canary",
            "capability-canary",
            "path-canary",
        ] {
            assert!(!text.contains(forbidden));
        }
        assert_eq!(text.matches("kvpack_admission_total{result=").count(), 5);
        assert_eq!(text.matches("kvpack_lookup_total{result=").count(), 6);
        assert_eq!(text.matches("kvpack_resource_current{kind=").count(), 10);
        assert_eq!(text.matches("kvpack_health{component=").count(), 6);
        assert_eq!(text.matches("kvpack_audit_total{outcome=").count(), 6);
        assert!(!text.contains("kvpack_resource_current{kind=\"reserved_bytes\"}"));
    }

    #[test]
    fn trace_records_share_random_context_without_debug_disclosure() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        let context = TraceContext::new(ServiceComponent::Store).unwrap();
        let root = telemetry
            .record_span(
                &context,
                None,
                TracePhase::Request,
                SpanOutcome::Ok,
                1_000,
                2_000,
            )
            .unwrap()
            .unwrap();
        assert_ne!(*root.as_bytes(), [0; 8]);
        telemetry
            .record_span(
                &context,
                Some(root),
                TracePhase::ExactLookup,
                SpanOutcome::Miss,
                2_000,
                3_000,
            )
            .unwrap()
            .unwrap();
        let exporter = RecordingExporter::default();
        assert_eq!(telemetry.export_otlp_batch(&exporter, 8).unwrap(), 2);
        let spans = exporter.spans.lock().unwrap();
        assert_eq!(spans[0].trace_id(), spans[1].trace_id());
        assert_ne!(spans[0].span_id(), spans[1].span_id());
        assert_eq!(spans[1].parent_span_id(), Some(root));
        assert!(format!("{context:?}").contains("[opaque]"));
        assert!(format!("{:?}", spans[0]).contains("[opaque]"));
    }

    #[test]
    fn trace_carrier_has_one_canonical_bounded_encoding() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        let context = TraceContext::new(ServiceComponent::CacheControlPlane).unwrap();
        let root = telemetry
            .record_span(&context, None, TracePhase::Request, SpanOutcome::Ok, 1, 2)
            .unwrap()
            .unwrap();
        let carrier = context.carrier(root);
        let binary = carrier.encode();
        let lower_hex = carrier.encode_lower_hex();

        assert_eq!(binary.len(), TRACE_CARRIER_BYTES);
        assert_eq!(lower_hex.len(), TRACE_CARRIER_HEX_BYTES);
        assert_eq!(TraceCarrier::decode(&binary).unwrap(), carrier);
        assert_eq!(TraceCarrier::decode_lower_hex(&lower_hex).unwrap(), carrier);
        assert_eq!(carrier.parent_span_id(), root);
        assert_eq!(format!("{carrier:?}"), "TraceCarrier([opaque])");

        assert!(TraceCarrier::decode(&binary[..binary.len() - 1]).is_err());
        let mut invalid = binary;
        invalid[0..4].copy_from_slice(&(TRACE_CARRIER_VERSION + 1).to_le_bytes());
        assert!(TraceCarrier::decode(&invalid).is_err());
        let mut invalid = binary;
        invalid[28] = 1;
        assert!(TraceCarrier::decode(&invalid).is_err());
        let mut invalid = binary;
        invalid[4..20].fill(0);
        assert!(TraceCarrier::decode(&invalid).is_err());
        let mut invalid = binary;
        invalid[20..28].fill(0);
        assert!(TraceCarrier::decode(&invalid).is_err());
        let mut invalid_hex = lower_hex;
        invalid_hex[0] = b'A';
        assert!(TraceCarrier::decode_lower_hex(&invalid_hex).is_err());
        assert!(TraceCarrier::decode_lower_hex(&lower_hex[..lower_hex.len() - 1]).is_err());
    }

    #[test]
    fn propagated_services_and_engine_suffix_share_only_opaque_parentage() {
        let telemetry = OperationalTelemetry::new(8).unwrap();
        let cache_context = TraceContext::new(ServiceComponent::CacheControlPlane).unwrap();
        let cache_root = telemetry
            .record_span(
                &cache_context,
                None,
                TracePhase::Request,
                SpanOutcome::Ok,
                10,
                20,
            )
            .unwrap()
            .unwrap();
        let gateway_carrier = cache_context.carrier(cache_root);
        let gateway_context =
            TraceContext::from_carrier(ServiceComponent::Gateway, gateway_carrier);
        let gateway_request = telemetry
            .record_span(
                &gateway_context,
                Some(gateway_carrier.parent_span_id()),
                TracePhase::Request,
                SpanOutcome::Ok,
                20,
                30,
            )
            .unwrap()
            .unwrap();
        let suffix_carrier = gateway_context.carrier(gateway_request);
        let suffix = telemetry
            .record_suffix_work(suffix_carrier, SpanOutcome::Ok, 30, 40)
            .unwrap()
            .unwrap();

        let exporter = RecordingExporter::default();
        assert_eq!(telemetry.export_otlp_batch(&exporter, 8).unwrap(), 3);
        let spans = exporter.spans.lock().unwrap();
        assert_eq!(spans[0].trace_id(), spans[1].trace_id());
        assert_eq!(spans[1].trace_id(), spans[2].trace_id());
        assert_eq!(spans[1].parent_span_id(), Some(cache_root));
        assert_eq!(spans[2].service(), ServiceComponent::Engine);
        assert_eq!(spans[2].phase(), TracePhase::SuffixWork);
        assert_eq!(spans[2].parent_span_id(), Some(gateway_request));
        assert_eq!(spans[2].span_id(), suffix);
    }

    #[test]
    fn failed_export_retains_batch_and_success_removes_it() {
        let telemetry = OperationalTelemetry::new(2).unwrap();
        let context = TraceContext::new(ServiceComponent::Gateway).unwrap();
        telemetry
            .record_span(
                &context,
                None,
                TracePhase::Transfer,
                SpanOutcome::Unavailable,
                1,
                2,
            )
            .unwrap();
        assert!(telemetry
            .export_otlp_batch(&FailingExporter, MAX_OTLP_BATCH_SPANS)
            .is_err());
        assert_eq!(telemetry.pending_spans().unwrap(), 1);
        assert!(telemetry
            .export_otlp_batch(&MiscountingExporter, MAX_OTLP_BATCH_SPANS)
            .is_err());
        assert_eq!(telemetry.pending_spans().unwrap(), 1);
        assert_eq!(
            telemetry
                .export_otlp_batch(&RecordingExporter::default(), 1)
                .unwrap(),
            1
        );
        assert_eq!(telemetry.pending_spans().unwrap(), 0);
    }

    #[test]
    fn full_trace_buffer_is_bounded_and_counted() {
        let telemetry = OperationalTelemetry::new(1).unwrap();
        let context = TraceContext::new(ServiceComponent::Agent).unwrap();
        assert!(telemetry
            .record_span(&context, None, TracePhase::Plan, SpanOutcome::Ok, 1, 2,)
            .unwrap()
            .is_some());
        assert!(telemetry
            .record_span(&context, None, TracePhase::Install, SpanOutcome::Ok, 2, 3,)
            .unwrap()
            .is_none());
        assert!(telemetry
            .prometheus_text()
            .unwrap()
            .contains("kvpack_trace_dropped_total 1"));
    }

    #[test]
    fn invalid_bounds_fail_without_mutating_queues() {
        assert!(OperationalTelemetry::new(0).is_err());
        assert!(OperationalTelemetry::new(MAX_TRACE_BUFFER_SPANS + 1).is_err());
        let telemetry = OperationalTelemetry::new(1).unwrap();
        let context = TraceContext::new(ServiceComponent::Store).unwrap();
        assert!(telemetry
            .record_span(&context, None, TracePhase::Request, SpanOutcome::Ok, 2, 1,)
            .is_err());
        assert!(telemetry
            .export_otlp_batch(&RecordingExporter::default(), 0)
            .is_err());
        assert_eq!(telemetry.pending_spans().unwrap(), 0);
    }

    #[derive(Default)]
    struct RecordingExporter {
        spans: StdMutex<Vec<OtlpSpanRecord>>,
    }

    impl OtlpSpanExporter for RecordingExporter {
        fn export(&self, spans: &[OtlpSpanRecord]) -> Result<OtlpExportOutcome, StoreError> {
            self.spans.lock().unwrap().extend_from_slice(spans);
            Ok(OtlpExportOutcome::new(spans.len(), 0))
        }
    }

    struct FailingExporter;

    impl OtlpSpanExporter for FailingExporter {
        fn export(&self, _: &[OtlpSpanRecord]) -> Result<OtlpExportOutcome, StoreError> {
            Err(StoreError::Busy)
        }
    }

    struct MiscountingExporter;

    impl OtlpSpanExporter for MiscountingExporter {
        fn export(&self, _: &[OtlpSpanRecord]) -> Result<OtlpExportOutcome, StoreError> {
            Ok(OtlpExportOutcome::new(0, 0))
        }
    }
}
