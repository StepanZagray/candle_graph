//! Unified evidence visualizer tests (`candle-graph/viewer/4`).

use candle_graph::evidence::{EvidencePacket, EVIDENCE_SCHEMA};
use candle_graph::graph::{
    build_from_trace, ExecutionGraph, GradientRecord, GradientRecordState, GraphEdge,
    GraphEdgeKind, GraphNode, GraphNodeKind, GraphSummary, SlowSpan, SCHEMA,
};
use candle_graph::nsight::NsightEvidence;
use candle_graph::trace::analyze_health;
use candle_graph::trace::memory::MemorySummary;
use candle_graph::trace::schema::{SpanKind, SpanRecord, TraceRunMeta, SCHEMA as TRACE_SCHEMA};
use candle_graph::trace::TraceDocument;
use candle_graph::viewer::trace_view::{project, TRACE_VIEWER_SCHEMA};
use candle_graph::viewer::{embed_json, escape_for_script, render_evidence_html};

fn minimal_trace() -> TraceDocument {
    TraceDocument {
        schema: TRACE_SCHEMA.to_string(),
        run: TraceRunMeta {
            run_id: "run-1".into(),
            correlation_id: "demo/update-1".into(),
            entrypoint: "demo::train::loss".into(),
            phase: candle_graph::ExecutionPhase::Train,
            timestamp: "2026-08-04T18:00:00Z".into(),
            capture_step: 1,
            warmup_steps: 0,
            device: "cpu".into(),
            measured_region_device_synchronized: false,
            timing_mode: candle_graph::TimingMode::Host,
            tags: Default::default(),
            candle_version: None,
        },
        spans: vec![
            SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "loss".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns: 5_000_000,
                step: None,
            },
            SpanRecord {
                id: "fwd".into(),
                parent_id: Some("root".into()),
                name: "forward".into(),
                kind: SpanKind::Function,
                measured: false,
                start_ns: 1_000_000,
                closed: true,
                duration_ns: 2_000_000,
                step: None,
            },
        ],
        ops: vec![],
        tensors: vec![],
        memory: vec![],
        device_memory: vec![],
        gradients: vec![],
        edges: vec![],
    }
}

fn minimal_graph() -> ExecutionGraph {
    build_from_trace(&minimal_trace()).unwrap()
}

fn packet(graph: ExecutionGraph) -> EvidencePacket {
    let trace = minimal_trace();
    EvidencePacket {
        schema: EVIDENCE_SCHEMA.into(),
        provenance: trace.run.clone(),
        health: analyze_health(&trace),
        findings: vec!["forward dominates self time".into()],
        facts: vec![],
        gaps: vec!["GPU evidence is unavailable".into()],
        graph,
        gpu: NsightEvidence::unavailable("not captured"),
        comparison: None,
    }
}

#[test]
fn project_trace_viewer_schema() {
    let payload = project(&packet(minimal_graph()));
    assert_eq!(payload["schema"], TRACE_VIEWER_SCHEMA);
    assert_eq!(payload["default_view"], "evidence");
    assert_eq!(payload["summary"]["entrypoint"], "demo::train::loss");
    assert!(payload["summary"]["total_ms"].as_f64().unwrap() > 0.0);
    assert!(payload["views"]["memory"].is_object());
}

#[test]
fn trace_edges_carry_ms_labels() {
    let payload = project(&packet(minimal_graph()));
    let edges = payload["views"]["trace"]["edges"].as_array().unwrap();
    assert!(!edges.is_empty());
    for edge in edges {
        assert!(edge["duration_ms"].is_number());
        let label = edge["label"].as_str().unwrap();
        assert!(
            label.contains("ms"),
            "expected ms on edge label, got {label}"
        );
    }
}

#[test]
fn render_evidence_html_is_standalone() {
    let html = render_evidence_html(&packet(minimal_graph()));
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("data-viewer=\"candle-graph-evidence\""));
    assert!(html.contains("view-panel-evidence"));
    assert!(html.contains("view-panel-gpu"));
    assert!(html.contains("span-tree"));
    assert!(html.contains("peak-breakdown"));
    assert!(html.contains("heat-mode"));
    assert!(!html.contains("https://"));
    assert!(!html.contains("<script src="));
    assert!(html.contains("cg-payload"));
}

#[test]
fn embed_json_escapes_script_breakout() {
    let raw = embed_json(&serde_json::json!({ "x": "</script><script>alert(1)" }));
    assert!(!raw.contains("</script>"));
    assert!(raw.contains("\\u003c"));
}

#[test]
fn escape_for_script_handles_unicode_separators() {
    assert!(escape_for_script("\u{2028}").contains("\\u2028"));
}

#[test]
fn manual_fixture_has_span_tree() {
    let graph = ExecutionGraph {
        schema: SCHEMA.into(),
        spans: vec![GraphNode {
            id: "a".into(),
            parent_id: None,
            name: "root".into(),
            kind: GraphNodeKind::Root,
            start_ns: 0,
            self_time_ns: 500_000,
            total_time_ns: 2_000_000,
            shape: None,
            dtype: None,
            device: None,
            bytes: 0,
            peak_bytes: 0,
            residual_bytes: 0,
            storage_bytes: None,
        }],
        edges: vec![GraphEdge {
            from: "a".into(),
            to: "a".into(),
            kind: GraphEdgeKind::Call,
            duration_ns: 100_000,
            label: None,
        }],
        tensors: vec![],
        gradients: vec![GradientRecord {
            root: "vb".into(),
            key: "w".into(),
            state: GradientRecordState::Present,
            norm: None,
        }],
        summary: GraphSummary {
            entrypoint: "root".into(),
            total_ms: 2.0,
            slowest_spans: vec![SlowSpan {
                id: "a".into(),
                name: "root".into(),
                self_time_ns: 500_000,
            }],
            heaviest_spans: vec![],
            memory: MemorySummary::default(),
        },
        memory: candle_graph::trace::memory::MemoryProfile {
            summary: MemorySummary::default(),
            timeline: vec![],
            peak_breakdown: vec![],
            by_device: vec![],
        },
    };
    let payload = project(&packet(graph));
    assert_eq!(
        payload["views"]["trace"]["nodes"].as_array().unwrap().len(),
        1
    );
    assert_eq!(payload["span_tree"].as_array().unwrap().len(), 1);
    assert_eq!(payload["gradients"].as_array().unwrap().len(), 1);
}
