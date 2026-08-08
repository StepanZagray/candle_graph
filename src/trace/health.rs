//! Structural trust and evidence-coverage checks for a parsed trace.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::phase::{ExecutionPhase, ExecutionStep};

use super::TraceDocument;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthIssue {
    pub severity: HealthSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCoverage {
    pub spans: usize,
    pub closed_spans: usize,
    pub root_spans: usize,
    pub measured_spans: usize,
    pub operations: usize,
    pub tensors: usize,
    pub memory_events: usize,
    pub device_memory_samples: usize,
    pub gradients: usize,
    pub edges: usize,
    pub forward_spans: usize,
    pub backward_spans: usize,
    pub optimizer_spans: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceHealth {
    pub trusted: bool,
    pub issues: Vec<HealthIssue>,
    pub coverage: EvidenceCoverage,
}

impl TraceHealth {
    pub fn gaps(&self) -> impl Iterator<Item = &HealthIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == HealthSeverity::Warning)
    }
}

pub fn analyze_health(doc: &TraceDocument) -> TraceHealth {
    let ids: HashSet<&str> = doc.spans.iter().map(|span| span.id.as_str()).collect();
    let by_id: HashMap<&str, _> = doc
        .spans
        .iter()
        .map(|span| (span.id.as_str(), span))
        .collect();
    let mut issues = Vec::new();
    if ids.len() != doc.spans.len() {
        error(
            &mut issues,
            "duplicate_span_id",
            format!(
                "span IDs must be unique; found {} records but {} unique IDs",
                doc.spans.len(),
                ids.len()
            ),
        );
    }
    let root_spans = doc
        .spans
        .iter()
        .filter(|span| span.parent_id.is_none())
        .count();
    if root_spans != 1 {
        error(
            &mut issues,
            "root_count",
            format!("expected exactly one root span, found {root_spans}"),
        );
    }
    let measured_spans = doc.spans.iter().filter(|span| span.measured).count();
    if measured_spans != 1 {
        error(
            &mut issues,
            "measurement_count",
            format!("expected exactly one measured region, found {measured_spans}"),
        );
    }
    for span in &doc.spans {
        if !span.closed {
            error(
                &mut issues,
                "open_span",
                format!("span `{}` was not closed", span.id),
            );
        }
        if let Some(parent) = span.parent_id.as_deref() {
            match by_id.get(parent) {
                None => error(
                    &mut issues,
                    "unknown_parent",
                    format!("span `{}` refers to missing parent `{parent}`", span.id),
                ),
                Some(parent_span) => {
                    let child_end = span.start_ns.saturating_add(span.duration_ns);
                    let parent_end = parent_span.start_ns.saturating_add(parent_span.duration_ns);
                    if span.start_ns < parent_span.start_ns || child_end > parent_end {
                        error(
                            &mut issues,
                            "child_outside_parent",
                            format!(
                                "span `{}` interval {}..{} is outside parent `{parent}` interval {}..{}",
                                span.id,
                                span.start_ns,
                                child_end,
                                parent_span.start_ns,
                                parent_end
                            ),
                        );
                    }
                }
            }
        }
        let mut current = Some(span.id.as_str());
        let mut seen = HashSet::new();
        while let Some(id) = current {
            if !seen.insert(id) {
                error(
                    &mut issues,
                    "span_cycle",
                    format!("span `{}` participates in a parent cycle", span.id),
                );
                break;
            }
            current = by_id.get(id).and_then(|item| item.parent_id.as_deref());
        }
    }
    for parent in &doc.spans {
        let child_total: u64 = doc
            .spans
            .iter()
            .filter(|span| span.parent_id.as_deref() == Some(parent.id.as_str()))
            .map(|span| span.duration_ns)
            .sum();
        if child_total > parent.duration_ns {
            error(
                &mut issues,
                "children_exceed_parent",
                format!(
                    "children of span `{}` total {} ns, exceeding its {} ns duration",
                    parent.id, child_total, parent.duration_ns
                ),
            );
        }
    }
    for (kind, span_id) in doc
        .ops
        .iter()
        .map(|x| ("operation", x.span_id.as_str()))
        .chain(doc.tensors.iter().map(|x| ("tensor", x.span_id.as_str())))
        .chain(doc.memory.iter().map(|x| ("memory", x.span_id.as_str())))
    {
        if !ids.contains(span_id) {
            error(
                &mut issues,
                "unknown_span",
                format!("{kind} evidence refers to missing span `{span_id}`"),
            );
        }
    }
    for edge in &doc.edges {
        if !ids.contains(edge.from_span.as_str()) || !ids.contains(edge.to_span.as_str()) {
            error(
                &mut issues,
                "unknown_edge_span",
                format!(
                    "edge `{}` -> `{}` refers to a missing span",
                    edge.from_span, edge.to_span
                ),
            );
        }
    }
    let mut live_memory = HashSet::new();
    let mut memory = doc.memory.iter().collect::<Vec<_>>();
    memory.sort_by_key(|event| event.timestamp_ns);
    for event in memory {
        let key = (event.device.as_str(), event.tensor_id.as_str());
        match event.action {
            super::MemoryAction::Alloc if !live_memory.insert(key) => error(
                &mut issues,
                "duplicate_allocation",
                format!(
                    "tensor `{}` was allocated twice without a free",
                    event.tensor_id
                ),
            ),
            super::MemoryAction::Free if !live_memory.remove(&key) => error(
                &mut issues,
                "unpaired_free",
                format!(
                    "tensor `{}` was freed without a live allocation",
                    event.tensor_id
                ),
            ),
            _ => {}
        }
    }
    if !live_memory.is_empty() {
        warning(
            &mut issues,
            "retained_allocations",
            format!(
                "{} explicit allocations remained live at measurement end",
                live_memory.len()
            ),
        );
    }

    let coverage = EvidenceCoverage {
        spans: doc.spans.len(),
        closed_spans: doc.spans.iter().filter(|span| span.closed).count(),
        root_spans,
        measured_spans,
        operations: doc.ops.len(),
        tensors: doc.tensors.len(),
        memory_events: doc.memory.len(),
        device_memory_samples: doc.device_memory.len(),
        gradients: doc.gradients.len(),
        edges: doc.edges.len(),
        forward_spans: step_count(doc, ExecutionStep::Forward),
        backward_spans: step_count(doc, ExecutionStep::Backward),
        optimizer_spans: step_count(doc, ExecutionStep::Optimizer),
    };

    for (empty, code, message) in [
        (
            coverage.operations == 0,
            "operations_absent",
            "no timed operation evidence was captured",
        ),
        (
            coverage.tensors == 0,
            "tensors_absent",
            "no tensor checkpoints were captured",
        ),
        (
            coverage.memory_events == 0,
            "memory_absent",
            "no tensor memory events were captured",
        ),
        (
            coverage.device_memory_samples == 0,
            "device_memory_absent",
            "no device-memory checkpoints were captured",
        ),
        (
            coverage.gradients == 0 && doc.run.phase == ExecutionPhase::Train,
            "gradients_absent",
            "no gradient facts were captured for this training run",
        ),
        (
            coverage.forward_spans == 0 && doc.run.phase == ExecutionPhase::Train,
            "forward_absent",
            "no forward span was tagged",
        ),
        (
            coverage.backward_spans == 0 && doc.run.phase == ExecutionPhase::Train,
            "backward_absent",
            "no backward span was tagged",
        ),
        (
            coverage.optimizer_spans == 0 && doc.run.phase == ExecutionPhase::Train,
            "optimizer_absent",
            "no optimizer span was tagged",
        ),
    ] {
        if empty {
            warning(&mut issues, code, message);
        }
    }

    let trusted = !issues
        .iter()
        .any(|issue| issue.severity == HealthSeverity::Error);
    TraceHealth {
        trusted,
        issues,
        coverage,
    }
}

fn step_count(doc: &TraceDocument, step: ExecutionStep) -> usize {
    doc.spans
        .iter()
        .filter(|span| span.step == Some(step))
        .count()
}

fn error(issues: &mut Vec<HealthIssue>, code: &str, message: impl Into<String>) {
    issues.push(HealthIssue {
        severity: HealthSeverity::Error,
        code: code.into(),
        message: message.into(),
    });
}

fn warning(issues: &mut Vec<HealthIssue>, code: &str, message: impl Into<String>) {
    issues.push(HealthIssue {
        severity: HealthSeverity::Warning,
        code: code.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{SpanKind, SpanRecord, TimingMode, TraceRunMeta, SCHEMA};

    fn document(spans: Vec<SpanRecord>) -> TraceDocument {
        TraceDocument {
            schema: SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "health".into(),
                correlation_id: "health/update-1".into(),
                entrypoint: "health".into(),
                phase: ExecutionPhase::Infer,
                timestamp: "2026-08-08T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                timing_mode: TimingMode::Host,
                tags: Default::default(),
                candle_version: None,
            },
            spans,
            ops: vec![],
            tensors: vec![],
            memory: vec![],
            device_memory: vec![],
            gradients: vec![],
            edges: vec![],
        }
    }

    fn span(id: &str, parent: Option<&str>, measured: bool, duration_ns: u64) -> SpanRecord {
        SpanRecord {
            id: id.into(),
            parent_id: parent.map(str::to_string),
            name: id.into(),
            kind: SpanKind::Function,
            measured,
            start_ns: 0,
            closed: true,
            duration_ns,
            step: None,
        }
    }

    #[test]
    fn rejects_multiple_roots_before_graph_building() {
        let health = analyze_health(&document(vec![
            span("a", None, true, 10),
            span("b", None, false, 10),
        ]));
        assert!(!health.trusted);
        assert!(health.issues.iter().any(|issue| issue.code == "root_count"));
    }

    #[test]
    fn rejects_disconnected_parent_cycle() {
        let health = analyze_health(&document(vec![
            span("root", None, true, 100),
            span("a", Some("b"), false, 0),
            span("b", Some("a"), false, 0),
        ]));
        assert!(!health.trusted);
        assert!(health.issues.iter().any(|issue| issue.code == "span_cycle"));
    }

    #[test]
    fn rejects_aggregate_child_time_larger_than_parent() {
        let health = analyze_health(&document(vec![
            span("root", None, true, 100),
            span("a", Some("root"), false, 70),
            span("b", Some("root"), false, 70),
        ]));
        assert!(!health.trusted);
        assert!(health
            .issues
            .iter()
            .any(|issue| issue.code == "children_exceed_parent"));
    }

    #[test]
    fn rejects_duplicate_ids_and_child_outside_parent_interval() {
        let mut spans = vec![
            span("root", None, true, 100),
            span("child", Some("root"), false, 10),
            span("child", Some("root"), false, 10),
        ];
        spans[1].start_ns = 101;
        spans[2].start_ns = 101;
        let health = analyze_health(&document(spans));
        assert!(!health.trusted);
        assert!(health
            .issues
            .iter()
            .any(|issue| issue.code == "duplicate_span_id"));
        assert!(health
            .issues
            .iter()
            .any(|issue| issue.code == "child_outside_parent"));
    }
}
