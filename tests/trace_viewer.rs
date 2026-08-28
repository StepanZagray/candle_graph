//! Standalone evidence/4 viewer/5 integration tests.

#![cfg(feature = "visualizer")]

use candle_graph::capability::{CaptureContract, CoverageLevel};
use candle_graph::evidence::EvidencePacket;
use candle_graph::nsight::NsightEvidence;
use candle_graph::trace::{
    DeviceIntervalEvent, RunOutcome, SpanKind, SpanRecord, TerminalEvent, TimingMode,
    TraceDocument, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
};
use candle_graph::viewer::trace_view::{project, SCHEMA as VIEWER_SCHEMA};

fn document(outcome: RunOutcome) -> TraceDocument {
    let complete = outcome == RunOutcome::Complete;
    TraceDocument {
        schema: TRACE_SCHEMA.into(),
        run: TraceRunMeta {
            run_id: "viewer-run".into(),
            correlation_id: "viewer/inference-1".into(),
            entrypoint: "demo::infer".into(),
            phase: candle_graph::ExecutionPhase::Infer,
            timestamp: "2026-08-19T00:00:00Z".into(),
            capture_step: 1,
            warmup_steps: 0,
            device: "cpu".into(),
            measured_region_device_synchronized: false,
            timing_mode: TimingMode::Host,
            capture_contract: CaptureContract {
                operations: CoverageLevel::None,
                ..CaptureContract::default()
            },
            comparison_identity: None,
            tags: Default::default(),
            candle_version: None,
        },
        spans: vec![SpanRecord {
            id: "root".into(),
            parent_id: None,
            name: "demo::infer".into(),
            kind: SpanKind::Function,
            measured: complete,
            start_ns: 0,
            closed: complete,
            duration_ns: if complete { 2_000_000 } else { 0 },
            step: None,
        }],
        ops: vec![],
        tensors: vec![],
        tensor_stats: vec![],
        memory: vec![],
        device_memory: vec![],
        device_intervals: vec![],
        gradients: vec![],
        edges: vec![],
        terminal: TerminalEvent {
            outcome,
            timestamp_ns: 2_000_000,
            reason: (!complete).then(|| "model error".into()),
        },
    }
}

#[test]
fn projection_exposes_capabilities_and_host_timing() {
    let evidence = EvidencePacket::from_document(
        document(RunOutcome::Complete),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();
    let payload = project(&evidence);
    assert_eq!(payload["schema"], VIEWER_SCHEMA);
    assert_eq!(payload["summary"]["outer_wall_time_ns"], 2_000_000);
    assert!(payload["views"]["evidence"]["capabilities"].is_object());
    assert!(payload["views"]["trace"]["nodes"][0]["host_total_time_ns"].is_number());
    assert!(payload["views"]["trace"]["nodes"][0]
        .get("total_time_ns")
        .is_none());
}

#[test]
fn failed_capture_remains_viewable_without_a_graph() {
    let evidence = EvidencePacket::from_document(
        document(RunOutcome::Failed),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();
    assert!(evidence.graph.is_none());
    let payload = project(&evidence);
    assert_eq!(payload["summary"]["capture_complete"], false);
    assert_eq!(
        payload["views"]["trace"]["nodes"].as_array().unwrap().len(),
        0
    );
}

#[test]
fn projection_keeps_device_timing_and_gpu_qualification_visible() {
    let mut document = document(RunOutcome::Complete);
    document.run.capture_contract.device_timing = CoverageLevel::Complete;
    document.device_intervals.push(DeviceIntervalEvent {
        span_id: "root".into(),
        device: "cuda:0".into(),
        stream_id: "7".into(),
        clock_id: "cuda-event-0".into(),
        backend: "cuda-event".into(),
        start_ns: 10,
        duration_ns: 100,
    });
    let evidence =
        EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
            .unwrap();
    let payload = project(&evidence);
    assert_eq!(
        payload["views"]["span_costs"]["items"][0]["device_timings"][0]["busy_ns"],
        100
    );
    assert_eq!(
        payload["views"]["gpu"]["correlation_capability"]["level"],
        "unavailable"
    );
    assert_eq!(
        payload["views"]["gpu"]["provenance_capability"]["level"],
        "unavailable"
    );
}
