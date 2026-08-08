//! Build [`ExecutionGraph`] from a parsed trace document.

use std::collections::{HashMap, HashSet};

use crate::trace::memory::{analyze_memory, node_memory_metrics};
use crate::trace::schema::GradientState as TraceGradientState;
use crate::trace::TraceDocument;

use super::model::{
    ExecutionGraph, GradientRecord, GradientRecordState, GraphEdge, GraphEdgeKind, GraphNode,
    GraphNodeKind, GraphSummary, HeavySpan, SlowSpan, SCHEMA,
};

/// Reconstruct span hierarchy, attach ops, infer call/data edges, and compute self times.
pub fn build_from_trace(doc: &TraceDocument) -> ExecutionGraph {
    let memory = analyze_memory(doc);
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut op_counter: HashMap<String, usize> = HashMap::new();
    let mut op_node_ids: HashMap<(String, usize), String> = HashMap::new();

    for span in &doc.spans {
        let is_root = span.parent_id.is_none();
        let total_time_ns = effective_span_duration(span, doc);
        nodes.push(GraphNode {
            id: span.id.clone(),
            parent_id: span.parent_id.clone(),
            name: span.name.clone(),
            kind: span_node_kind(span.kind, is_root),
            self_time_ns: total_time_ns,
            total_time_ns,
            shape: None,
            dtype: None,
            device: None,
            bytes: 0,
            peak_bytes: 0,
            residual_bytes: 0,
            storage_bytes: None,
        });
    }

    for op in &doc.ops {
        let index = op_counter.entry(op.span_id.clone()).or_insert(0);
        let op_index = *index;
        *index += 1;
        let node_id = format!("{}/op/{}", op.span_id, op_index);
        op_node_ids.insert((op.span_id.clone(), op_index), node_id.clone());
        nodes.push(GraphNode {
            id: node_id,
            parent_id: Some(op.span_id.clone()),
            name: op.op_name.clone(),
            kind: GraphNodeKind::Op,
            self_time_ns: op.duration_ns,
            total_time_ns: op.duration_ns,
            shape: if op.shape.is_empty() {
                None
            } else {
                Some(op.shape.clone())
            },
            dtype: Some(op.dtype.clone()),
            device: Some(op.device.clone()),
            bytes: op.storage_bytes.unwrap_or(0),
            peak_bytes: op.storage_bytes.unwrap_or(0),
            residual_bytes: 0,
            storage_bytes: op.storage_bytes,
        });
    }

    let mem_by_node = node_memory_metrics(doc, &op_node_ids);
    for node in &mut nodes {
        if let Some(metrics) = mem_by_node.get(&node.id) {
            if metrics.bytes > 0 {
                node.bytes = metrics.bytes;
            }
            if metrics.peak_bytes > 0 {
                node.peak_bytes = metrics.peak_bytes;
            }
            if metrics.residual_bytes > 0 {
                node.residual_bytes = metrics.residual_bytes;
            }
            if metrics.storage_bytes > 0 && node.storage_bytes.is_none() {
                node.storage_bytes = Some(metrics.storage_bytes);
            }
        }
    }

    compute_self_times(&mut nodes);

    let mut edges: Vec<GraphEdge> = doc
        .edges
        .iter()
        .map(|edge| GraphEdge {
            from: edge.from_span.clone(),
            to: edge.to_span.clone(),
            kind: GraphEdgeKind::Call,
            duration_ns: edge.duration_ns,
            label: None,
        })
        .collect();

    let mut inferred_call: HashSet<(String, String)> = HashSet::new();
    for node in &nodes {
        if node.kind == GraphNodeKind::Op {
            continue;
        }
        if let Some(parent_id) = &node.parent_id {
            let key = (parent_id.clone(), node.id.clone());
            if inferred_call.insert(key.clone())
                && !edges.iter().any(|edge| {
                    edge.from == key.0
                        && edge.to == key.1
                        && matches!(edge.kind, GraphEdgeKind::Call)
                })
            {
                edges.push(GraphEdge {
                    from: parent_id.clone(),
                    to: node.id.clone(),
                    kind: GraphEdgeKind::Call,
                    duration_ns: node.total_time_ns,
                    label: None,
                });
            }
        }
    }

    op_counter.clear();
    for op in &doc.ops {
        let op_index = *op_counter.entry(op.span_id.clone()).or_insert(0);
        *op_counter.get_mut(&op.span_id).expect("just inserted") += 1;
        let op_id = format!("{}/op/{}", op.span_id, op_index);
        if let Some(output) = &op.output {
            for input in &op.inputs {
                if !edges.iter().any(|edge| {
                    edge.from == *input
                        && edge.to == *output
                        && matches!(edge.kind, GraphEdgeKind::Data)
                }) {
                    edges.push(GraphEdge {
                        from: input.clone(),
                        to: output.clone(),
                        kind: GraphEdgeKind::Data,
                        duration_ns: op.duration_ns,
                        label: Some(op.op_name.clone()),
                    });
                }
            }
            let call_key = (op.span_id.clone(), op_id.clone());
            if inferred_call.insert(call_key) {
                edges.push(GraphEdge {
                    from: op.span_id.clone(),
                    to: op_id,
                    kind: GraphEdgeKind::Call,
                    duration_ns: op.duration_ns,
                    label: Some(op.op_name.clone()),
                });
            }
        }
    }

    edges.sort_by(|left, right| {
        left.from
            .cmp(&right.from)
            .then_with(|| left.to.cmp(&right.to))
            .then_with(|| format!("{:?}", left.kind).cmp(&format!("{:?}", right.kind)))
    });

    let gradients = doc
        .gradients
        .iter()
        .map(|gradient| GradientRecord {
            root: gradient.root.clone(),
            key: gradient.key.clone(),
            state: match gradient.state {
                TraceGradientState::Present => GradientRecordState::Present,
                TraceGradientState::Missing => GradientRecordState::Missing,
                TraceGradientState::Zero => GradientRecordState::Zero,
                TraceGradientState::NonFinite => GradientRecordState::NonFinite,
            },
            norm: gradient.norm,
        })
        .collect();

    let root = nodes
        .iter()
        .find(|node| node.parent_id.is_none())
        .or_else(|| nodes.first());
    let total_ms = root
        .map(|node| node.total_time_ns as f64 / 1_000_000.0)
        .unwrap_or(0.0);

    let mut slowest: Vec<SlowSpan> = nodes
        .iter()
        .filter(|node| node.kind != GraphNodeKind::Op)
        .map(|node| SlowSpan {
            id: node.id.clone(),
            name: node.name.clone(),
            self_time_ns: node.self_time_ns,
        })
        .collect();
    slowest.sort_by_key(|span| std::cmp::Reverse(span.self_time_ns));
    slowest.truncate(8);

    let mut heaviest: Vec<HeavySpan> = nodes
        .iter()
        .filter(|node| node.kind != GraphNodeKind::Op)
        .map(|node| HeavySpan {
            id: node.id.clone(),
            name: node.name.clone(),
            bytes: node.bytes,
            peak_bytes: node.peak_bytes,
        })
        .collect();
    heaviest.sort_by_key(|span| std::cmp::Reverse(span.bytes));
    heaviest.truncate(8);

    ExecutionGraph {
        schema: SCHEMA,
        spans: nodes,
        edges,
        gradients,
        summary: GraphSummary {
            entrypoint: doc.run.entrypoint.clone(),
            total_ms,
            slowest_spans: slowest,
            heaviest_spans: heaviest,
            memory: memory.summary.clone(),
        },
        memory,
    }
}

fn effective_span_duration(span: &crate::trace::schema::SpanRecord, doc: &TraceDocument) -> u64 {
    if span.duration_ns > 0 {
        return span.duration_ns;
    }

    let op_total: u64 = doc
        .ops
        .iter()
        .filter(|op| op.span_id == span.id)
        .map(|op| op.duration_ns)
        .sum();

    let child_total: u64 = doc
        .spans
        .iter()
        .filter(|child| child.parent_id.as_deref() == Some(span.id.as_str()))
        .map(|child| effective_span_duration(child, doc))
        .sum();

    op_total.saturating_add(child_total)
}

fn span_node_kind(kind: crate::trace::schema::SpanKind, is_root: bool) -> GraphNodeKind {
    if is_root {
        return GraphNodeKind::Root;
    }
    match kind {
        crate::trace::schema::SpanKind::Function => GraphNodeKind::Function,
        crate::trace::schema::SpanKind::Module => GraphNodeKind::Module,
        crate::trace::schema::SpanKind::Op => GraphNodeKind::Op,
    }
}

fn compute_self_times(nodes: &mut [GraphNode]) {
    let child_totals: HashMap<String, u64> = nodes.iter().fold(HashMap::new(), |mut acc, node| {
        if let Some(parent) = &node.parent_id {
            *acc.entry(parent.clone()).or_insert(0) += node.total_time_ns;
        }
        acc
    });

    for node in nodes.iter_mut() {
        let children = child_totals.get(&node.id).copied().unwrap_or(0);
        node.self_time_ns = node.total_time_ns.saturating_sub(children);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::events::{EdgeEvent, GradientEvent, MemoryEvent, OpEvent};
    use crate::trace::memory::{MemoryAction, MemoryCategory};
    use crate::trace::schema::{
        GradientState, SpanKind, SpanRecord, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };

    fn synthetic_trace() -> TraceDocument {
        TraceDocument {
            schema: TRACE_SCHEMA.to_string(),
            run: TraceRunMeta {
                run_id: "run-1".into(),
                entrypoint: "demo::train::loss".into(),
                phase: "train".into(),
                timestamp: "2026-08-04T18:00:00Z".into(),
                candle_version: None,
            },
            spans: vec![
                SpanRecord {
                    id: "root".into(),
                    parent_id: None,
                    name: "loss".into(),
                    kind: SpanKind::Function,
                    closed: true,
                    duration_ns: 10_000_000,
                    step: None,
                },
                SpanRecord {
                    id: "encoder".into(),
                    parent_id: Some("root".into()),
                    name: "Encoder::forward".into(),
                    kind: SpanKind::Function,
                    closed: true,
                    duration_ns: 5_000_000,
                    step: Some(crate::phase::ExecutionStep::Forward),
                },
                SpanRecord {
                    id: "head".into(),
                    parent_id: Some("root".into()),
                    name: "Head::forward".into(),
                    kind: SpanKind::Function,
                    closed: true,
                    duration_ns: 1_000_000,
                    step: None,
                },
            ],
            ops: vec![
                OpEvent {
                    span_id: "encoder".into(),
                    op_name: "matmul".into(),
                    inputs: vec!["x".into(), "w".into()],
                    output: Some("h0".into()),
                    shape: vec![8, 64],
                    dtype: "f32".into(),
                    device: "cpu".into(),
                    duration_ns: 2_000_000,
                    timestamp_ns: 2_000_000,
                    storage_bytes: Some(8 * 64 * 4),
                    input_storage_bytes: 0,
                },
                OpEvent {
                    span_id: "head".into(),
                    op_name: "linear".into(),
                    inputs: vec!["h0".into()],
                    output: Some("logits".into()),
                    shape: vec![8, 10],
                    dtype: "f32".into(),
                    device: "cpu".into(),
                    duration_ns: 800_000,
                    timestamp_ns: 800_000,
                    storage_bytes: Some(8 * 10 * 4),
                    input_storage_bytes: 0,
                },
            ],
            tensors: Vec::new(),
            memory: vec![
                MemoryEvent {
                    timestamp_ns: 2_000_000,
                    tensor_id: "h0".into(),
                    span_id: "encoder".into(),
                    op_name: Some("matmul".into()),
                    device: "cpu".into(),
                    bytes: 8 * 64 * 4,
                    action: MemoryAction::Alloc,
                    shape: vec![8, 64],
                    dtype: "f32".into(),
                    category: MemoryCategory::Activation,
                },
                MemoryEvent {
                    timestamp_ns: 2_800_000,
                    tensor_id: "logits".into(),
                    span_id: "head".into(),
                    op_name: Some("linear".into()),
                    device: "cpu".into(),
                    bytes: 8 * 10 * 4,
                    action: MemoryAction::Alloc,
                    shape: vec![8, 10],
                    dtype: "f32".into(),
                    category: MemoryCategory::Activation,
                },
            ],
            device_memory: Vec::new(),
            gradients: vec![GradientEvent {
                event_id: "g1".into(),
                root: "vb".into(),
                key: "encoder.weight".into(),
                state: GradientState::Present,
                step: None,
                norm: Some(0.5),
            }],
            edges: vec![EdgeEvent {
                from_span: "h0".into(),
                to_span: "logits".into(),
                duration_ns: 800_000,
            }],
        }
    }

    #[test]
    fn reconstructs_span_tree_and_self_times() {
        let graph = build_from_trace(&synthetic_trace());

        assert_eq!(graph.schema, SCHEMA);
        assert_eq!(graph.summary.entrypoint, "demo::train::loss");
        assert!((graph.summary.total_ms - 10.0).abs() < f64::EPSILON);

        let root = graph.node("root").expect("root span");
        assert_eq!(root.parent_id, None);
        assert_eq!(root.total_time_ns, 10_000_000);
        assert_eq!(root.self_time_ns, 4_000_000);

        let encoder = graph.node("encoder").expect("encoder span");
        assert_eq!(encoder.parent_id.as_deref(), Some("root"));
        assert_eq!(encoder.bytes, 8 * 64 * 4);

        let matmul = graph.node("encoder/op/0").expect("matmul op");
        assert_eq!(matmul.kind, GraphNodeKind::Op);
        assert_eq!(matmul.storage_bytes, Some(8 * 64 * 4));
    }

    #[test]
    fn memory_summary_and_heaviest_spans() {
        let graph = build_from_trace(&synthetic_trace());
        assert!(graph.summary.memory.peak_bytes >= 8 * 64 * 4);
        assert_eq!(graph.summary.memory.alloc_count, 2);
        assert!(!graph.summary.heaviest_spans.is_empty());
        assert_eq!(graph.summary.heaviest_spans[0].id, "encoder");
    }

    #[test]
    fn builds_call_and_data_edges() {
        let graph = build_from_trace(&synthetic_trace());

        assert!(
            graph.edges.iter().any(|edge| {
                edge.from == "root"
                    && edge.to == "encoder"
                    && matches!(edge.kind, GraphEdgeKind::Call)
            }),
            "expected root -> encoder call edge"
        );
    }

    #[test]
    fn records_gradients_and_slowest_spans() {
        let graph = build_from_trace(&synthetic_trace());

        assert_eq!(graph.gradients.len(), 1);
        assert_eq!(graph.summary.slowest_spans[0].id, "root");
    }

    #[test]
    fn attaches_ops_to_declared_span() {
        let doc = TraceDocument {
            schema: TRACE_SCHEMA.to_string(),
            run: TraceRunMeta {
                run_id: "run-2".into(),
                entrypoint: "outer".into(),
                phase: "train".into(),
                timestamp: "2026-08-04T18:00:00Z".into(),
                candle_version: None,
            },
            spans: vec![
                SpanRecord {
                    id: "a".into(),
                    parent_id: None,
                    name: "a".into(),
                    kind: SpanKind::Function,
                    closed: true,
                    duration_ns: 200,
                    step: None,
                },
                SpanRecord {
                    id: "b".into(),
                    parent_id: Some("a".into()),
                    name: "b".into(),
                    kind: SpanKind::Function,
                    closed: true,
                    duration_ns: 100,
                    step: None,
                },
            ],
            ops: vec![OpEvent {
                span_id: "b".into(),
                op_name: "add".into(),
                inputs: vec![],
                output: None,
                shape: vec![],
                dtype: "f32".into(),
                device: "cpu".into(),
                duration_ns: 50,
                timestamp_ns: 50,
                storage_bytes: None,
                input_storage_bytes: 0,
            }],
            tensors: Vec::new(),
            memory: Vec::new(),
            device_memory: Vec::new(),
            gradients: Vec::new(),
            edges: Vec::new(),
        };

        let graph = build_from_trace(&doc);
        let op = graph.node("b/op/0").expect("op under span b");
        assert_eq!(op.parent_id.as_deref(), Some("b"));
    }
}
