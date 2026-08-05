//! Trace-file CLI engine (`import`, `view`, `summary`, `query`).

use anyhow::{Context, Result};
use std::path::Path;

use crate::graph::{build_from_trace, ExecutionGraph, GraphNode};
use crate::trace::parse_trace;

/// Bounded query kinds for trace-derived execution graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceQueryKind {
    Slowest,
    Heaviest,
    Memory,
    Efficiency,
    Spans,
    Gradients,
}

/// Parse a JSONL trace and build an [`ExecutionGraph`].
pub fn load_graph(trace_path: &Path) -> Result<ExecutionGraph> {
    let doc = parse_trace(trace_path)
        .with_context(|| format!("parse trace {}", trace_path.display()))?;
    Ok(build_from_trace(&doc))
}

/// `import` — emit full execution graph JSON.
pub fn run_import(trace_path: &Path, output: Option<&Path>) -> Result<()> {
    let graph = load_graph(trace_path)?;
    let rendered = serde_json::to_string_pretty(&graph)? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// `summary` — emit graph summary JSON.
pub fn run_summary(trace_path: &Path, output: Option<&Path>) -> Result<()> {
    let graph = load_graph(trace_path)?;
    let rendered = serde_json::to_string_pretty(&graph.summary)? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// `query` — emit a bounded slice of graph facts.
pub fn run_query(trace_path: &Path, kind: TraceQueryKind, output: Option<&Path>) -> Result<()> {
    let graph = load_graph(trace_path)?;
    let payload = match kind {
        TraceQueryKind::Slowest => query_slowest(&graph),
        TraceQueryKind::Heaviest => query_heaviest(&graph),
        TraceQueryKind::Memory => query_memory(&graph),
        TraceQueryKind::Efficiency => query_efficiency(&graph),
        TraceQueryKind::Spans => query_spans(&graph),
        TraceQueryKind::Gradients => query_gradients(&graph),
    };
    let rendered = serde_json::to_string_pretty(&payload)? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// `view` — render standalone HTML from a trace (requires `visualizer` feature).
#[cfg(feature = "visualizer")]
pub fn run_view(trace_path: &Path, output: &Path) -> Result<()> {
    let graph = load_graph(trace_path)?;
    let html = crate::viewer::render_trace_html(&graph);
    super::write_output(Some(output), html.as_bytes())
}

fn query_slowest(graph: &ExecutionGraph) -> serde_json::Value {
    let mut ops: Vec<&GraphNode> = graph
        .spans
        .iter()
        .filter(|node| matches!(node.kind, crate::graph::GraphNodeKind::Op))
        .collect();
    ops.sort_by(|left, right| {
        right
            .self_time_ns
            .cmp(&left.self_time_ns)
            .then_with(|| left.id.cmp(&right.id))
    });
    ops.truncate(50);

    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "slowest",
        "entrypoint": graph.summary.entrypoint,
        "total_ms": graph.summary.total_ms,
        "slowest_spans": graph.summary.slowest_spans,
        "slowest_ops": ops,
    })
}

fn query_heaviest(graph: &ExecutionGraph) -> serde_json::Value {
    let mut ops: Vec<&GraphNode> = graph
        .spans
        .iter()
        .filter(|node| matches!(node.kind, crate::graph::GraphNodeKind::Op))
        .collect();
    ops.sort_by(|left, right| {
        right
            .bytes
            .cmp(&left.bytes)
            .then_with(|| left.id.cmp(&right.id))
    });
    ops.truncate(50);

    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "heaviest",
        "entrypoint": graph.summary.entrypoint,
        "peak_bytes": graph.summary.memory.peak_bytes,
        "heaviest_spans": graph.summary.heaviest_spans,
        "heaviest_ops": ops,
    })
}

fn query_efficiency(graph: &ExecutionGraph) -> serde_json::Value {
    let mut ops: Vec<&GraphNode> = graph
        .spans
        .iter()
        .filter(|node| matches!(node.kind, crate::graph::GraphNodeKind::Op))
        .filter(|node| node.self_time_ns > 0 && node.bytes > 0)
        .collect();
    ops.sort_by(|left, right| {
        let left_score = left.bytes as f64 / left.self_time_ns as f64;
        let right_score = right.bytes as f64 / right.self_time_ns as f64;
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    ops.truncate(50);

    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "efficiency",
        "entrypoint": graph.summary.entrypoint,
        "note": "bytes per nanosecond of self time — higher means more memory traffic per unit compute",
        "ops": ops,
    })
}

fn query_memory(graph: &ExecutionGraph) -> serde_json::Value {
    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "memory",
        "entrypoint": graph.summary.entrypoint,
        "summary": graph.summary.memory,
        "timeline": graph.memory.timeline,
        "peak_breakdown": graph.memory.peak_breakdown,
        "by_device": graph.memory.by_device,
    })
}

fn query_spans(graph: &ExecutionGraph) -> serde_json::Value {
    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "spans",
        "entrypoint": graph.summary.entrypoint,
        "spans": graph.spans,
        "edges": graph.edges,
    })
}

fn query_gradients(graph: &ExecutionGraph) -> serde_json::Value {
    serde_json::json!({
        "schema": "candle-graph/trace-query/1",
        "kind": "gradients",
        "entrypoint": graph.summary.entrypoint,
        "gradients": graph.gradients,
    })
}
