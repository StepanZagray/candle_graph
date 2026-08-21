//! Evidence CLI engine for trace/9, evidence/3, comparison/4, and atomic bundles.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::artifact::{publish_bundle, verify_bundle};
use crate::comparison::{compare_unverified_traces, compare_verified_bundles};
use crate::evidence::{build_evidence, EvidencePacket};
use crate::graph::{ExecutionGraph, GraphNode, GraphNodeKind};
use crate::trace::parse_trace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceQueryKind {
    SlowestHost,
    SlowestDevice,
    Heaviest,
    Memory,
    Spans,
    Tensors,
    Gradients,
    Capabilities,
}

pub fn load_evidence(trace_path: &Path) -> Result<EvidencePacket> {
    build_evidence(trace_path, None)
}

pub fn run_import(trace_path: &Path, output: Option<&Path>) -> Result<()> {
    let evidence = load_evidence(trace_path)?;
    super::write_output(
        output,
        (serde_json::to_string_pretty(&evidence)? + "\n").as_bytes(),
    )
}

pub fn run_summary(trace_path: &Path, output: Option<&Path>) -> Result<()> {
    let evidence = load_evidence(trace_path)?;
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": "candle-graph/summary/2",
        "provenance": evidence.provenance,
        "health": evidence.health,
        "capabilities": evidence.capabilities,
        "findings": evidence.findings,
        "gaps": evidence.gaps,
        "summary": evidence.graph.as_ref().map(|graph| &graph.summary),
        "timing": evidence.timing,
        "memory": evidence.memory,
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

pub fn run_query(trace_path: &Path, kind: TraceQueryKind, output: Option<&Path>) -> Result<()> {
    let evidence = load_evidence(trace_path)?;
    let result = match kind {
        TraceQueryKind::Memory => serde_json::to_value(&evidence.memory)?,
        TraceQueryKind::Capabilities => serde_json::to_value(&evidence.capabilities)?,
        other => {
            let graph = evidence
                .graph
                .as_ref()
                .context("query requires a complete, structurally valid capture")?;
            query_graph(graph, other)
        }
    };
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": "candle-graph/trace-query/2",
        "kind": format!("{kind:?}").to_ascii_lowercase(),
        "capabilities": evidence.capabilities,
        "result": result,
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

#[cfg(feature = "visualizer")]
pub fn run_view(trace_path: &Path, output: &Path, nsight_dir: Option<&Path>) -> Result<()> {
    let evidence = build_evidence(trace_path, nsight_dir)?;
    super::write_output(
        Some(output),
        crate::viewer::render_evidence_html(&evidence).as_bytes(),
    )
}

pub fn run_compare(
    baseline: &[PathBuf],
    candidate: &[PathBuf],
    unverified_traces: bool,
    output: Option<&Path>,
) -> Result<()> {
    let comparison = if unverified_traces {
        let parse_all = |paths: &[PathBuf], cohort: &str| -> Result<Vec<_>> {
            paths
                .iter()
                .map(|path| {
                    parse_trace(path)
                        .with_context(|| format!("parse unverified {cohort} {}", path.display()))
                })
                .collect()
        };
        compare_unverified_traces(
            &parse_all(baseline, "baseline")?,
            &parse_all(candidate, "candidate")?,
        )
    } else {
        compare_verified_bundles(baseline, candidate)?
    };
    super::write_output(
        output,
        (serde_json::to_string_pretty(&comparison)? + "\n").as_bytes(),
    )
}

pub fn run_report(trace: &Path, nsight_dir: Option<&Path>, bundle: &Path) -> Result<()> {
    publish_bundle(bundle, trace, nsight_dir)?;
    Ok(())
}

pub fn run_verify(bundle: &Path, output: Option<&Path>) -> Result<()> {
    let receipt = verify_bundle(bundle)?;
    super::write_output(
        output,
        (serde_json::to_string_pretty(&receipt)? + "\n").as_bytes(),
    )
}

fn query_graph(graph: &ExecutionGraph, kind: TraceQueryKind) -> serde_json::Value {
    match kind {
        TraceQueryKind::SlowestHost => serde_json::json!({
            "entrypoint": graph.summary.entrypoint,
            "outer_wall_time_ns": graph.summary.outer_wall_time_ns,
            "slowest_host_spans": graph.summary.slowest_host_spans,
            "slowest_host_ops": sorted_nodes(graph, |node| matches!(node.kind, GraphNodeKind::Op), |node| node.host_self_time_ns),
        }),
        TraceQueryKind::SlowestDevice => serde_json::json!({
            "entrypoint": graph.summary.entrypoint,
            "slowest_device_spans": graph.summary.slowest_device_spans,
        }),
        TraceQueryKind::Heaviest => serde_json::json!({
            "entrypoint": graph.summary.entrypoint,
            "heaviest_spans": graph.summary.heaviest_spans,
            "heaviest_ops": sorted_nodes(graph, |node| matches!(node.kind, GraphNodeKind::Op) && node.allocated_bytes.is_some(), |node| node.allocated_bytes.unwrap_or(0)),
        }),
        TraceQueryKind::Spans => serde_json::json!({ "spans": graph.spans, "edges": graph.edges }),
        TraceQueryKind::Tensors => serde_json::to_value(&graph.tensors).unwrap_or_default(),
        TraceQueryKind::Gradients => serde_json::to_value(&graph.gradients).unwrap_or_default(),
        TraceQueryKind::Memory | TraceQueryKind::Capabilities => {
            unreachable!("handled without graph")
        }
    }
}

fn sorted_nodes(
    graph: &ExecutionGraph,
    include: impl Fn(&GraphNode) -> bool,
    value: impl Fn(&GraphNode) -> u64,
) -> Vec<&GraphNode> {
    let mut nodes = graph
        .spans
        .iter()
        .filter(|node| include(node))
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| {
        value(right)
            .cmp(&value(left))
            .then_with(|| left.id.cmp(&right.id))
    });
    nodes.truncate(50);
    nodes
}
