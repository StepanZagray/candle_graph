//! Build an execution graph from one complete, structurally valid trace.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{ensure, Result};

use crate::timing::analyze_timing;
use crate::trace::memory::node_memory_metrics;
use crate::trace::{analyze_health, EdgeEvent, GradientState, TraceDocument};

use super::model::*;

pub fn build_from_trace(doc: &TraceDocument) -> Result<ExecutionGraph> {
    let health = analyze_health(doc);
    ensure!(
        health.capture_complete,
        "cannot build a graph from a failed capture"
    );
    ensure!(
        health.structurally_valid,
        "trace is structurally invalid: {}",
        health
            .issues
            .iter()
            .filter(|issue| issue.severity == crate::trace::HealthSeverity::Error)
            .map(|issue| issue.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    );

    let timing = analyze_timing(doc);
    let timing_by_span = timing.device_spans.iter().fold(
        HashMap::<&str, Vec<DeviceNodeTiming>>::new(),
        |mut by_span, item| {
            by_span
                .entry(&item.span_id)
                .or_default()
                .push(DeviceNodeTiming {
                    device: item.device.clone(),
                    clock_id: item.clock_id.clone(),
                    busy_ns: item.busy_ns,
                });
            by_span
        },
    );
    let mut nodes = Vec::new();
    for span in &doc.spans {
        let host_total_time_ns = effective_span_duration(span, doc);
        nodes.push(GraphNode {
            id: span.id.clone(),
            parent_id: span.parent_id.clone(),
            name: span.name.clone(),
            kind: span_node_kind(span.kind, span.parent_id.is_none()),
            start_ns: span.start_ns,
            host_self_time_ns: host_total_time_ns,
            host_total_time_ns,
            device_timings: timing_by_span
                .get(span.id.as_str())
                .cloned()
                .unwrap_or_default(),
            shape: None,
            dtype: None,
            device: None,
            allocated_bytes: None,
            peak_live_bytes: None,
            residual_bytes: None,
            dense_bytes: None,
        });
    }

    let mut op_counter = HashMap::<String, usize>::new();
    let mut op_node_ids = HashMap::new();
    for op in &doc.ops {
        let index = op_counter.entry(op.span_id.clone()).or_default();
        let id = format!("{}/op/{}", op.span_id, *index);
        op_node_ids.insert((op.span_id.clone(), *index), id.clone());
        *index += 1;
        nodes.push(GraphNode {
            id,
            parent_id: Some(op.span_id.clone()),
            name: op.op_name.clone(),
            kind: GraphNodeKind::Op,
            start_ns: op.timestamp_ns,
            host_self_time_ns: op.duration_ns,
            host_total_time_ns: op.duration_ns,
            device_timings: Vec::new(),
            shape: (!op.shape.is_empty()).then(|| op.shape.clone()),
            dtype: Some(op.dtype.clone()),
            device: Some(op.device.clone()),
            allocated_bytes: None,
            peak_live_bytes: None,
            residual_bytes: None,
            dense_bytes: op.output_dense_bytes,
        });
    }

    let tensor_ids = doc
        .tensors
        .iter()
        .map(|tensor| tensor.tensor_id.clone())
        .chain(doc.ops.iter().flat_map(|op| op.inputs.iter().cloned()))
        .chain(doc.ops.iter().filter_map(|op| op.output.clone()))
        .collect::<BTreeSet<_>>();
    for tensor_id in tensor_ids {
        let evidence = doc
            .tensors
            .iter()
            .find(|tensor| tensor.tensor_id == tensor_id);
        nodes.push(GraphNode {
            id: tensor_node_id(&tensor_id),
            parent_id: evidence.map(|tensor| tensor.span_id.clone()),
            name: evidence
                .and_then(|tensor| tensor.label.clone())
                .unwrap_or_else(|| tensor_id.clone()),
            kind: GraphNodeKind::Tensor,
            start_ns: 0,
            host_self_time_ns: 0,
            host_total_time_ns: 0,
            device_timings: Vec::new(),
            shape: evidence
                .map(|tensor| tensor.shape.clone())
                .filter(|shape| !shape.is_empty()),
            dtype: evidence.map(|tensor| tensor.dtype.clone()),
            device: evidence.map(|tensor| tensor.device.clone()),
            allocated_bytes: None,
            peak_live_bytes: None,
            residual_bytes: None,
            dense_bytes: evidence.and_then(|tensor| tensor.dense_bytes),
        });
    }

    for (node_id, metrics) in node_memory_metrics(doc, &op_node_ids) {
        if let Some(node) = nodes.iter_mut().find(|node| node.id == node_id) {
            node.allocated_bytes = metrics.direct_allocated_bytes;
            node.peak_live_bytes = metrics.subtree_peak_live_bytes;
            node.residual_bytes = metrics.subtree_residual_bytes;
            if node.dense_bytes.is_none() {
                node.dense_bytes = metrics.output_dense_bytes;
            }
        }
    }
    compute_host_self_times(&mut nodes);

    let mut edges = doc
        .edges
        .iter()
        .map(|edge| match edge {
            EdgeEvent::Call {
                from_span,
                to_span,
                host_duration_ns,
            } => GraphEdge::Call {
                from_span: from_span.clone(),
                to_span: to_span.clone(),
                host_duration_ns: *host_duration_ns,
                label: None,
            },
            EdgeEvent::Data {
                from_tensor,
                to_tensor,
            } => GraphEdge::Data {
                from_tensor: tensor_node_id(from_tensor),
                to_tensor: tensor_node_id(to_tensor),
                label: None,
            },
        })
        .collect::<Vec<_>>();
    let mut existing = edges.iter().map(edge_key).collect::<HashSet<_>>();
    for node in nodes
        .iter()
        .filter(|node| !matches!(node.kind, GraphNodeKind::Op | GraphNodeKind::Tensor))
    {
        if let Some(parent) = &node.parent_id {
            if existing.insert((parent.clone(), node.id.clone(), "call")) {
                edges.push(GraphEdge::Call {
                    from_span: parent.clone(),
                    to_span: node.id.clone(),
                    host_duration_ns: node.host_total_time_ns,
                    label: None,
                });
            }
        }
    }
    op_counter.clear();
    for op in &doc.ops {
        let index = op_counter.entry(op.span_id.clone()).or_default();
        let op_id = format!("{}/op/{}", op.span_id, *index);
        *index += 1;
        if existing.insert((op.span_id.clone(), op_id.clone(), "call")) {
            edges.push(GraphEdge::Call {
                from_span: op.span_id.clone(),
                to_span: op_id.clone(),
                host_duration_ns: op.duration_ns,
                label: Some(op.op_name.clone()),
            });
        }
        for input in &op.inputs {
            if existing.insert((tensor_node_id(input), op_id.clone(), "data")) {
                edges.push(GraphEdge::Data {
                    from_tensor: tensor_node_id(input),
                    to_tensor: op_id.clone(),
                    label: Some("input".into()),
                });
            }
        }
        if let Some(output) = &op.output {
            if existing.insert((op_id.clone(), tensor_node_id(output), "data")) {
                edges.push(GraphEdge::Data {
                    from_tensor: op_id.clone(),
                    to_tensor: tensor_node_id(output),
                    label: Some("output".into()),
                });
            }
        }
    }
    edges.sort_by_key(edge_key);

    let measured_ids = measured_subtree_ids(doc);
    let mut slowest_host_spans = nodes
        .iter()
        .filter(|node| {
            measured_ids.contains(&node.id)
                && !matches!(node.kind, GraphNodeKind::Op | GraphNodeKind::Tensor)
        })
        .map(|node| HostSpanCost {
            id: node.id.clone(),
            name: node.name.clone(),
            host_self_time_ns: node.host_self_time_ns,
        })
        .collect::<Vec<_>>();
    slowest_host_spans.sort_by_key(|span| std::cmp::Reverse(span.host_self_time_ns));
    slowest_host_spans.truncate(8);
    let mut slowest_device_spans = nodes
        .iter()
        .filter(|node| measured_ids.contains(&node.id))
        .flat_map(|node| {
            node.device_timings.iter().map(|timing| DeviceSpanCost {
                id: node.id.clone(),
                name: node.name.clone(),
                device: timing.device.clone(),
                clock_id: timing.clock_id.clone(),
                device_busy_ns: timing.busy_ns,
            })
        })
        .collect::<Vec<_>>();
    slowest_device_spans.sort_by_key(|span| std::cmp::Reverse(span.device_busy_ns));
    slowest_device_spans.truncate(8);
    let mut heaviest_spans = nodes
        .iter()
        .filter(|node| {
            measured_ids.contains(&node.id)
                && !matches!(node.kind, GraphNodeKind::Op | GraphNodeKind::Tensor)
        })
        .filter_map(|node| {
            node.allocated_bytes.map(|allocated_bytes| HeavySpan {
                id: node.id.clone(),
                name: node.name.clone(),
                allocated_bytes,
                peak_live_bytes: node.peak_live_bytes,
            })
        })
        .collect::<Vec<_>>();
    heaviest_spans.sort_by_key(|span| std::cmp::Reverse(span.allocated_bytes));
    heaviest_spans.truncate(8);
    let outer_wall_time_ns = doc
        .spans
        .iter()
        .find(|span| span.measured)
        .map(|span| span.duration_ns)
        .unwrap_or(0);

    Ok(ExecutionGraph {
        schema: SCHEMA.into(),
        spans: nodes,
        edges,
        tensors: doc
            .tensors
            .iter()
            .map(|tensor| TensorRecord {
                span_id: tensor.span_id.clone(),
                tensor_id: tensor.tensor_id.clone(),
                label: tensor.label.clone(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype.clone(),
                device: tensor.device.clone(),
                requires_grad: tensor.requires_grad,
                dense_bytes: tensor.dense_bytes,
                category: tensor.category,
            })
            .collect(),
        gradients: doc
            .gradients
            .iter()
            .map(|gradient| GradientRecord {
                root: gradient.root.clone(),
                key: gradient.key.clone(),
                state: match gradient.state {
                    GradientState::Present => GradientRecordState::Present,
                    GradientState::Missing => GradientRecordState::Missing,
                    GradientState::Zero => GradientRecordState::Zero,
                    GradientState::NonFinite => GradientRecordState::NonFinite,
                },
                norm: gradient.norm,
            })
            .collect(),
        summary: GraphSummary {
            entrypoint: doc.run.entrypoint.clone(),
            outer_wall_time_ns,
            slowest_host_spans,
            slowest_device_spans,
            heaviest_spans,
        },
    })
}

fn tensor_node_id(id: &str) -> String {
    format!("tensor/{id}")
}

fn edge_key(edge: &GraphEdge) -> (String, String, &'static str) {
    match edge {
        GraphEdge::Call {
            from_span, to_span, ..
        } => (from_span.clone(), to_span.clone(), "call"),
        GraphEdge::Data {
            from_tensor,
            to_tensor,
            ..
        } => (from_tensor.clone(), to_tensor.clone(), "data"),
    }
}

fn measured_subtree_ids(doc: &TraceDocument) -> HashSet<String> {
    let mut ids = doc
        .spans
        .iter()
        .filter(|span| span.measured)
        .map(|span| span.id.clone())
        .collect::<HashSet<_>>();
    loop {
        let before = ids.len();
        for span in &doc.spans {
            if span
                .parent_id
                .as_ref()
                .is_some_and(|parent| ids.contains(parent))
            {
                ids.insert(span.id.clone());
            }
        }
        if ids.len() == before {
            return ids;
        }
    }
}

fn effective_span_duration(span: &crate::trace::SpanRecord, doc: &TraceDocument) -> u64 {
    if span.duration_ns > 0 {
        return span.duration_ns;
    }
    doc.ops
        .iter()
        .filter(|op| op.span_id == span.id)
        .map(|op| op.duration_ns)
        .sum::<u64>()
        .saturating_add(
            doc.spans
                .iter()
                .filter(|child| child.parent_id.as_deref() == Some(&span.id))
                .map(|child| effective_span_duration(child, doc))
                .sum(),
        )
}

fn span_node_kind(kind: crate::trace::SpanKind, root: bool) -> GraphNodeKind {
    if root {
        return GraphNodeKind::Root;
    }
    match kind {
        crate::trace::SpanKind::Function => GraphNodeKind::Function,
        crate::trace::SpanKind::Module => GraphNodeKind::Module,
        crate::trace::SpanKind::Op => GraphNodeKind::Op,
    }
}

fn compute_host_self_times(nodes: &mut [GraphNode]) {
    let child_intervals = nodes
        .iter()
        .filter(|node| !matches!(node.kind, GraphNodeKind::Tensor))
        .fold(
            HashMap::<String, Vec<(u64, u64)>>::new(),
            |mut intervals, node| {
                if let Some(parent) = &node.parent_id {
                    intervals.entry(parent.clone()).or_default().push((
                        node.start_ns,
                        node.start_ns.saturating_add(node.host_total_time_ns),
                    ));
                }
                intervals
            },
        );
    for node in nodes {
        let covered = child_intervals
            .get(&node.id)
            .map(|intervals| host_interval_union(intervals))
            .unwrap_or(0);
        node.host_self_time_ns = node.host_total_time_ns.saturating_sub(covered);
    }
}

fn host_interval_union(intervals: &[(u64, u64)]) -> u64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_unstable();
    let mut total = 0u64;
    let mut current: Option<(u64, u64)> = None;
    for (start, end) in intervals {
        match current {
            None => current = Some((start, end)),
            Some((left, right)) if start <= right => current = Some((left, right.max(end))),
            Some((left, right)) => {
                total = total.saturating_add(right.saturating_sub(left));
                current = Some((start, end));
            }
        }
    }
    if let Some((start, end)) = current {
        total = total.saturating_add(end.saturating_sub(start));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CaptureContract;
    use crate::trace::{
        MemoryCategory, RunOutcome, SpanKind, SpanRecord, TensorEvent, TerminalEvent, TimingMode,
        TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };

    #[test]
    fn semantic_label_collisions_preserve_distinct_backend_tensor_nodes() {
        let doc = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "tensor-label-collision".into(),
                correlation_id: "tensor/label/collision".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-20T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "demo".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns: 1,
                step: None,
            }],
            ops: vec![],
            tensors: ["backend:1", "backend:2"]
                .into_iter()
                .map(|tensor_id| TensorEvent {
                    span_id: "root".into(),
                    tensor_id: tensor_id.into(),
                    label: Some("prediction".into()),
                    shape: vec![1],
                    dtype: "f32".into(),
                    device: "cpu".into(),
                    requires_grad: false,
                    dense_bytes: Some(4),
                    category: MemoryCategory::Activation,
                })
                .collect(),
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 1,
                reason: None,
            },
        };

        let graph = build_from_trace(&doc).unwrap();
        let nodes = graph
            .spans
            .iter()
            .filter(|node| node.kind == GraphNodeKind::Tensor)
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().all(|node| node.name == "prediction"));
        assert_ne!(nodes[0].id, nodes[1].id);
        assert_eq!(
            graph
                .tensors
                .iter()
                .map(|tensor| tensor.tensor_id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["backend:1", "backend:2"])
        );
    }
}
