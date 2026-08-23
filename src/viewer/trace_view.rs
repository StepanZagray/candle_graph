//! Project evidence/4 into the standalone viewer/5 payload.

use serde_json::{json, Value};

use crate::evidence::EvidencePacket;
use crate::graph::{ExecutionGraph, GraphEdge, GraphNode, GraphNodeKind};

pub const SCHEMA: &str = "candle-graph/viewer/5";

pub fn project(evidence: &EvidencePacket) -> Value {
    let graph = evidence.graph.as_ref();
    let tensors = graph
        .map(|graph| graph.tensors.as_slice())
        .unwrap_or_default();
    let gradients = graph.map(gradients_view).unwrap_or_else(|| json!([]));
    let mut gpu = serde_json::to_value(&evidence.gpu).expect("Nsight evidence is serializable");
    if let Some(gpu) = gpu.as_object_mut() {
        gpu.insert(
            "correlation_capability".into(),
            serde_json::to_value(&evidence.capabilities.gpu_correlation)
                .expect("capability is serializable"),
        );
        gpu.insert(
            "provenance_capability".into(),
            serde_json::to_value(&evidence.capabilities.provenance_binding)
                .expect("capability is serializable"),
        );
    }
    json!({
        "schema": SCHEMA,
        "default_view": "evidence",
        "summary": {
            "entrypoint": evidence.provenance.entrypoint,
            "phase": evidence.provenance.phase,
            "outer_wall_time_ns": graph.map(|graph| graph.summary.outer_wall_time_ns),
            "logical_peak_live_bytes": evidence.memory.logical.as_ref().and_then(|logical| logical.peak.as_ref().map(|peak| peak.live_bytes)),
            "capture_complete": evidence.health.capture_complete,
            "structurally_valid": evidence.health.structurally_valid,
        },
        "views": {
            "trace": graph.map(trace_view).unwrap_or_else(|| json!({"nodes": [], "edges": []})),
            "span_costs": graph.map(span_costs_view).unwrap_or_else(|| json!({"items": []})),
            "memory": evidence.memory,
            "evidence": {
                "provenance": evidence.provenance,
                "health": evidence.health,
                "capabilities": evidence.capabilities,
                "findings": evidence.findings,
                "facts": evidence.facts,
                "gaps": evidence.gaps,
                "tensors": tensors,
                "gradients": gradients,
            },
            "gpu": gpu,
        },
        "span_tree": graph.map(span_tree_view).unwrap_or_else(|| json!([])),
        "gradients": gradients,
    })
}

fn trace_view(graph: &ExecutionGraph) -> Value {
    json!({
        "nodes": graph.spans.iter().map(trace_node).collect::<Vec<_>>(),
        "edges": graph.edges.iter().map(trace_edge).collect::<Vec<_>>(),
    })
}

fn span_costs_view(graph: &ExecutionGraph) -> Value {
    let mut items = graph
        .spans
        .iter()
        .map(|node| {
            json!({
                "id": node.id,
                "name": node.name,
                "kind": node_kind_str(&node.kind),
                "parent_id": node.parent_id,
                "start_ns": node.start_ns,
                "host_self_time_ns": node.host_self_time_ns,
                "host_total_time_ns": node.host_total_time_ns,
                "device_timings": node.device_timings,
                "allocated_bytes": node.allocated_bytes,
                "peak_live_bytes": node.peak_live_bytes,
                "dense_bytes": node.dense_bytes,
            })
        })
        .collect::<Vec<_>>();
    items.sort_by_key(|item| std::cmp::Reverse(item["host_total_time_ns"].as_u64().unwrap_or(0)));
    json!({ "items": items })
}

fn span_tree_view(graph: &ExecutionGraph) -> Value {
    Value::Array(
        graph
            .spans
            .iter()
            .filter(|node| !matches!(node.kind, GraphNodeKind::Op | GraphNodeKind::Tensor))
            .map(|node| {
                json!({
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "name": node.name,
                    "kind": node_kind_str(&node.kind),
                    "host_self_time_ns": node.host_self_time_ns,
                    "host_total_time_ns": node.host_total_time_ns,
                    "host_self_ratio": host_self_ratio(node),
                    "allocated_bytes": node.allocated_bytes,
                    "peak_live_bytes": node.peak_live_bytes,
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
            .map(|gradient| {
                json!({
                    "root": gradient.root,
                    "key": gradient.key,
                    "state": format!("{:?}", gradient.state).to_lowercase(),
                    "norm": gradient.norm,
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
        "host_self_time_ns": node.host_self_time_ns,
        "host_total_time_ns": node.host_total_time_ns,
        "host_self_ratio": host_self_ratio(node),
        "device_timings": node.device_timings,
        "shape": node.shape,
        "dtype": node.dtype,
        "device": node.device,
        "allocated_bytes": node.allocated_bytes,
        "peak_live_bytes": node.peak_live_bytes,
        "residual_bytes": node.residual_bytes,
        "dense_bytes": node.dense_bytes,
    })
}

fn trace_edge(edge: &GraphEdge) -> Value {
    match edge {
        GraphEdge::Call {
            from_span,
            to_span,
            host_duration_ns,
            label,
        } => json!({
            "id": format!("{from_span}->{to_span}"),
            "from": from_span,
            "to": to_span,
            "kind": "call",
            "host_duration_ns": host_duration_ns,
            "label": label,
        }),
        GraphEdge::Data {
            from_tensor,
            to_tensor,
            label,
        } => json!({
            "id": format!("{from_tensor}->{to_tensor}"),
            "from": from_tensor,
            "to": to_tensor,
            "kind": "data",
            "label": label,
        }),
    }
}

fn node_kind_str(kind: &GraphNodeKind) -> &'static str {
    match kind {
        GraphNodeKind::Root => "root",
        GraphNodeKind::Function => "function",
        GraphNodeKind::Module => "module",
        GraphNodeKind::Op => "op",
        GraphNodeKind::Tensor => "tensor",
        GraphNodeKind::Other => "other",
    }
}

fn short_label(name: &str) -> String {
    name.rsplit("::").next().unwrap_or(name).to_string()
}

fn host_self_ratio(node: &GraphNode) -> f64 {
    if node.host_total_time_ns == 0 {
        0.0
    } else {
        node.host_self_time_ns as f64 / node.host_total_time_ns as f64
    }
}
