//! Project [`EvidencePacket`] into `candle-graph/viewer/4` for the unified visualizer.

use serde_json::{json, Value};

use crate::evidence::EvidencePacket;
use crate::graph::{
    ExecutionGraph, GraphEdge, GraphEdgeKind, GraphNode, GraphNodeKind, GraphSummary,
};

pub const TRACE_VIEWER_SCHEMA: &str = "candle-graph/viewer/4";

/// Build the unified evidence payload consumed by [`crate::viewer::render_evidence_html`].
pub fn project(evidence: &EvidencePacket) -> Value {
    let graph = &evidence.graph;
    json!({
        "schema": TRACE_VIEWER_SCHEMA,
        "default_view": "evidence",
        "summary": summary_view(&graph.summary, graph, evidence),
        "views": {
            "trace": trace_view(graph),
            "span_costs": span_costs_view(graph),
            "memory": memory_view(graph),
            "evidence": {
                "provenance": evidence.provenance,
                "health": evidence.health,
                "findings": evidence.findings,
                "facts": evidence.facts,
                "gaps": evidence.gaps,
                "comparison": evidence.comparison,
                "tensors": graph.tensors,
                "gradients": gradients_view(graph),
            },
            "gpu": evidence.gpu,
        },
        "span_tree": span_tree_view(graph),
        "gradients": gradients_view(graph),
    })
}

fn summary_view(
    summary: &GraphSummary,
    graph: &ExecutionGraph,
    evidence: &EvidencePacket,
) -> Value {
    json!({
        "entrypoint": summary.entrypoint,
        "total_ms": summary.total_ms,
        "phase": evidence.provenance.phase,
        "slowest_spans": summary.slowest_spans,
        "heaviest_spans": summary.heaviest_spans,
        "memory": summary.memory,
        "peak_breakdown": graph.memory.peak_breakdown,
    })
}

fn trace_view(graph: &ExecutionGraph) -> Value {
    json!({
        "nodes": graph.spans.iter().map(trace_node).collect::<Vec<_>>(),
        "edges": graph.edges.iter().map(trace_edge).collect::<Vec<_>>(),
    })
}

fn span_costs_view(graph: &ExecutionGraph) -> Value {
    let mut items: Vec<Value> = graph
        .spans
        .iter()
        .map(|node| {
            json!({
                "id": node.id,
                "name": node.name,
                "kind": node_kind_str(&node.kind),
                "parent_id": node.parent_id,
                "start_ns": node.start_ns,
                "self_ms": ns_to_ms(node.self_time_ns),
                "total_ms": ns_to_ms(node.total_time_ns),
                "self_time_ns": node.self_time_ns,
                "total_time_ns": node.total_time_ns,
                "bytes": node.bytes,
                "peak_bytes": node.peak_bytes,
                "storage_bytes": node.storage_bytes,
            })
        })
        .collect();
    items.sort_by(|a, b| {
        b["total_time_ns"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["total_time_ns"].as_u64().unwrap_or(0))
    });
    json!({ "items": items })
}

fn memory_view(graph: &ExecutionGraph) -> Value {
    json!({
        "summary": graph.memory.summary,
        "timeline": graph.memory.timeline,
        "peak_breakdown": graph.memory.peak_breakdown,
        "by_device": graph.memory.by_device,
    })
}

fn span_tree_view(graph: &ExecutionGraph) -> Value {
    Value::Array(
        graph
            .spans
            .iter()
            .filter(|node| node.kind != GraphNodeKind::Op)
            .map(|node| {
                json!({
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "name": node.name,
                    "kind": node_kind_str(&node.kind),
                    "self_ms": ns_to_ms(node.self_time_ns),
                    "total_ms": ns_to_ms(node.total_time_ns),
                    "self_ratio": self_ratio(node),
                    "bytes": node.bytes,
                    "peak_bytes": node.peak_bytes,
                })
            })
            .collect(),
    )
}

fn gradients_view(graph: &ExecutionGraph) -> Value {
    Value::Array(
        graph
            .gradients
            .iter()
            .map(|g| {
                json!({
                    "root": g.root,
                    "key": g.key,
                    "state": format!("{:?}", g.state).to_lowercase(),
                    "norm": g.norm,
                })
            })
            .collect(),
    )
}

fn trace_node(node: &GraphNode) -> Value {
    json!({
        "id": node.id,
        "label": node.name,
        "short_label": short_label(&node.name),
        "kind": node_kind_str(&node.kind),
        "parent_id": node.parent_id,
        "start_ns": node.start_ns,
        "self_time_ns": node.self_time_ns,
        "total_time_ns": node.total_time_ns,
        "self_time_ms": ns_to_ms(node.self_time_ns),
        "total_time_ms": ns_to_ms(node.total_time_ns),
        "self_ratio": self_ratio(node),
        "shape": node.shape,
        "dtype": node.dtype,
        "device": node.device,
        "bytes": node.bytes,
        "peak_bytes": node.peak_bytes,
        "residual_bytes": node.residual_bytes,
        "storage_bytes": node.storage_bytes,
        "memory_ratio": memory_ratio(node, node.peak_bytes),
    })
}

fn trace_edge(edge: &GraphEdge) -> Value {
    let duration_ms = ns_to_ms(edge.duration_ns);
    let label = edge
        .label
        .as_ref()
        .filter(|s| s.contains("ms"))
        .cloned()
        .unwrap_or_else(|| format!("{duration_ms:.2} ms"));
    json!({
        "id": format!("{}->{}", edge.from, edge.to),
        "from": edge.from,
        "to": edge.to,
        "kind": match edge.kind {
            GraphEdgeKind::Call => "call",
            GraphEdgeKind::Data => "data",
        },
        "duration_ns": edge.duration_ns,
        "duration_ms": duration_ms,
        "label": label,
    })
}

fn node_kind_str(kind: &GraphNodeKind) -> &'static str {
    match kind {
        GraphNodeKind::Root => "root",
        GraphNodeKind::Function => "function",
        GraphNodeKind::Module => "module",
        GraphNodeKind::Op => "op",
        GraphNodeKind::Other => "other",
    }
}

fn short_label(name: &str) -> String {
    name.rsplit("::").next().unwrap_or(name).to_string()
}

fn ns_to_ms(ns: u64) -> f64 {
    (ns as f64) / 1_000_000.0
}

fn self_ratio(node: &GraphNode) -> f64 {
    if node.total_time_ns == 0 {
        0.0
    } else {
        node.self_time_ns as f64 / node.total_time_ns as f64
    }
}

fn memory_ratio(node: &GraphNode, peak: u64) -> f64 {
    if peak == 0 {
        0.0
    } else {
        node.peak_bytes as f64 / peak as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidencePacket, EVIDENCE_SCHEMA};
    use crate::graph::{
        GradientRecord, GradientRecordState, GraphEdge, GraphNode, GraphSummary, HeavySpan,
        SlowSpan, SCHEMA,
    };
    use crate::nsight::NsightEvidence;
    use crate::trace::memory::MemorySummary;
    use crate::trace::{EvidenceCoverage, TraceHealth, TraceRunMeta};
    use std::collections::BTreeMap;

    fn sample_graph() -> ExecutionGraph {
        ExecutionGraph {
            schema: SCHEMA.into(),
            spans: vec![
                GraphNode {
                    id: "root".into(),
                    parent_id: None,
                    name: "demo::loss".into(),
                    kind: GraphNodeKind::Root,
                    start_ns: 0,
                    self_time_ns: 1_000_000,
                    total_time_ns: 5_000_000,
                    shape: None,
                    dtype: None,
                    device: None,
                    bytes: 4096,
                    peak_bytes: 4096,
                    residual_bytes: 0,
                    storage_bytes: None,
                },
                GraphNode {
                    id: "child".into(),
                    parent_id: Some("root".into()),
                    name: "forward".into(),
                    kind: GraphNodeKind::Function,
                    start_ns: 1_000_000,
                    self_time_ns: 2_000_000,
                    total_time_ns: 4_000_000,
                    shape: None,
                    dtype: None,
                    device: None,
                    bytes: 4096,
                    peak_bytes: 4096,
                    residual_bytes: 0,
                    storage_bytes: None,
                },
            ],
            edges: vec![GraphEdge {
                from: "root".into(),
                to: "child".into(),
                kind: GraphEdgeKind::Call,
                duration_ns: 4_000_000,
                label: None,
            }],
            tensors: vec![],
            gradients: vec![GradientRecord {
                root: "vb".into(),
                key: "w".into(),
                state: GradientRecordState::Present,
                norm: Some(1.0),
            }],
            summary: GraphSummary {
                entrypoint: "demo::loss".into(),
                total_ms: 5.0,
                slowest_spans: vec![SlowSpan {
                    id: "root".into(),
                    name: "demo::loss".into(),
                    self_time_ns: 1_000_000,
                }],
                heaviest_spans: vec![HeavySpan {
                    id: "root".into(),
                    name: "demo::loss".into(),
                    bytes: 4096,
                    peak_bytes: 4096,
                }],
                memory: MemorySummary {
                    peak_bytes: 4096,
                    ..MemorySummary::default()
                },
            },
            memory: crate::trace::memory::MemoryProfile {
                summary: MemorySummary {
                    peak_bytes: 4096,
                    ..MemorySummary::default()
                },
                timeline: vec![],
                peak_breakdown: vec![],
                by_device: vec![],
            },
        }
    }

    fn sample_evidence() -> EvidencePacket {
        EvidencePacket {
            schema: EVIDENCE_SCHEMA.into(),
            provenance: TraceRunMeta {
                run_id: "run-1".into(),
                correlation_id: "demo/update-1".into(),
                entrypoint: "demo::loss".into(),
                phase: crate::phase::ExecutionPhase::Train,
                timestamp: "2026-08-08T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                timing_mode: crate::trace::TimingMode::Host,
                tags: BTreeMap::new(),
                candle_version: None,
            },
            health: TraceHealth {
                trusted: true,
                issues: vec![],
                coverage: EvidenceCoverage::default(),
            },
            findings: vec![],
            facts: vec![],
            gaps: vec![],
            graph: sample_graph(),
            gpu: NsightEvidence::unavailable("not captured"),
            comparison: None,
        }
    }

    #[test]
    fn project_emits_viewer4_schema() {
        let payload = project(&sample_evidence());
        assert_eq!(payload["schema"], TRACE_VIEWER_SCHEMA);
        assert_eq!(payload["default_view"], "evidence");
        assert!(payload["views"]["memory"]["summary"]["peak_bytes"].as_u64() == Some(4096));
    }

    #[test]
    fn every_edge_has_duration_ms_label() {
        let payload = project(&sample_evidence());
        for edge in payload["views"]["trace"]["edges"].as_array().unwrap() {
            assert!(edge["duration_ms"].is_number());
            let label = edge["label"].as_str().unwrap();
            assert!(label.contains("ms"), "edge label must include ms: {label}");
        }
    }
}
