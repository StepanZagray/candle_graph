//! Structural validation and observed evidence coverage for a parsed trace.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::capability::{CoverageLevel, GradientFamilyExpectation};
use crate::phase::{ExecutionPhase, ExecutionStep};

use super::{EdgeEvent, GradientState, MemoryAction, RunOutcome, TraceDocument};

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
    pub device_intervals: usize,
    pub gradients: usize,
    pub call_edges: usize,
    pub data_edges: usize,
    pub forward_spans: usize,
    pub backward_spans: usize,
    pub optimizer_spans: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceHealth {
    /// The event stream is internally consistent enough for derived analysis.
    pub structurally_valid: bool,
    /// The producer emitted a successful terminal event.
    pub capture_complete: bool,
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
    let failed = doc.terminal.outcome == RunOutcome::Failed;
    let ids: HashSet<&str> = doc.spans.iter().map(|span| span.id.as_str()).collect();
    let by_id: HashMap<&str, _> = doc
        .spans
        .iter()
        .map(|span| (span.id.as_str(), span))
        .collect();
    let mut issues = Vec::new();

    match doc.terminal.outcome {
        RunOutcome::Complete if doc.terminal.reason.is_some() => error(
            &mut issues,
            "complete_with_failure_reason",
            "complete terminal outcome cannot contain a failure reason",
        ),
        RunOutcome::Failed
            if doc
                .terminal
                .reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty()) =>
        {
            error(
                &mut issues,
                "failed_without_reason",
                "failed terminal outcome requires a non-empty reason",
            )
        }
        _ => {}
    }
    let latest_host_timestamp_ns = doc
        .spans
        .iter()
        .map(|span| {
            span.start_ns
                .saturating_add(if span.closed { span.duration_ns } else { 0 })
        })
        .chain(
            doc.ops
                .iter()
                .map(|op| op.timestamp_ns.saturating_add(op.duration_ns)),
        )
        .chain(doc.memory.iter().map(|event| event.timestamp_ns))
        .chain(doc.device_memory.iter().map(|event| event.timestamp_ns))
        .max()
        .unwrap_or(0);
    if doc.terminal.timestamp_ns < latest_host_timestamp_ns {
        error(
            &mut issues,
            "terminal_precedes_evidence",
            format!(
                "terminal timestamp {} precedes host evidence ending at {latest_host_timestamp_ns}",
                doc.terminal.timestamp_ns
            ),
        );
    }

    if failed {
        warning(
            &mut issues,
            "capture_failed",
            doc.terminal
                .reason
                .as_deref()
                .unwrap_or("capture ended with a failed outcome"),
        );
    }
    if ids.len() != doc.spans.len() {
        error(&mut issues, "duplicate_span_id", "span IDs must be unique");
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
    if measured_spans != 1 && !failed {
        error(
            &mut issues,
            "measurement_count",
            format!("expected exactly one measured region, found {measured_spans}"),
        );
    }

    for span in &doc.spans {
        if !span.closed {
            if failed {
                warning(
                    &mut issues,
                    "open_span",
                    format!("span `{}` was interrupted", span.id),
                );
            } else {
                error(
                    &mut issues,
                    "open_span",
                    format!("span `{}` was not closed", span.id),
                );
            }
        }
        if let Some(parent) = span.parent_id.as_deref() {
            match by_id.get(parent) {
                None => error(
                    &mut issues,
                    "unknown_parent",
                    format!("span `{}` refers to missing parent `{parent}`", span.id),
                ),
                Some(parent_span) if span.closed && parent_span.closed => {
                    let child_end = span.start_ns.saturating_add(span.duration_ns);
                    let parent_end = parent_span.start_ns.saturating_add(parent_span.duration_ns);
                    if span.start_ns < parent_span.start_ns || child_end > parent_end {
                        error(
                            &mut issues,
                            "child_outside_parent",
                            format!("span `{}` lies outside parent `{parent}`", span.id),
                        );
                    }
                }
                Some(_) => {}
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

    for (kind, span_id) in doc
        .ops
        .iter()
        .map(|x| ("operation", x.span_id.as_str()))
        .chain(doc.tensors.iter().map(|x| ("tensor", x.span_id.as_str())))
        .chain(doc.memory.iter().map(|x| ("memory", x.span_id.as_str())))
        .chain(
            doc.device_intervals
                .iter()
                .map(|x| ("device interval", x.span_id.as_str())),
        )
    {
        if !ids.contains(span_id) {
            error(
                &mut issues,
                "unknown_span",
                format!("{kind} evidence refers to missing span `{span_id}`"),
            );
        }
    }
    for interval in &doc.device_intervals {
        if interval.duration_ns == 0 {
            error(
                &mut issues,
                "empty_device_interval",
                format!(
                    "device interval for `{}` has zero duration",
                    interval.span_id
                ),
            );
        }
    }
    for op in &doc.ops {
        if let Some(span) = by_id.get(op.span_id.as_str()).filter(|span| span.closed) {
            let span_end = span.start_ns.saturating_add(span.duration_ns);
            let op_end = op.timestamp_ns.saturating_add(op.duration_ns);
            if op.timestamp_ns < span.start_ns || op_end > span_end {
                error(
                    &mut issues,
                    "operation_outside_span",
                    format!(
                        "operation `{}` lies outside span `{}`",
                        op.op_name, op.span_id
                    ),
                );
            }
        }
    }
    for sample in &doc.device_memory {
        if sample.used_bytes.is_none()
            && sample.free_bytes.is_none()
            && sample.reserved_bytes.is_none()
            && sample.capacity_bytes.is_none()
        {
            error(
                &mut issues,
                "empty_device_memory_sample",
                format!(
                    "device-memory sample for `{}` contains no measurements",
                    sample.device
                ),
            );
        }
    }
    let known_tensors = doc
        .tensors
        .iter()
        .map(|tensor| tensor.tensor_id.as_str())
        .chain(
            doc.ops
                .iter()
                .flat_map(|op| op.inputs.iter().map(String::as_str)),
        )
        .chain(doc.ops.iter().filter_map(|op| op.output.as_deref()))
        .collect::<HashSet<_>>();
    for edge in &doc.edges {
        match edge {
            EdgeEvent::Call {
                from_span,
                to_span,
                ..
            } if !ids.contains(from_span.as_str()) || !ids.contains(to_span.as_str()) => error(
                &mut issues,
                "unknown_call_edge_span",
                format!("call edge `{from_span}` -> `{to_span}` refers to a missing span"),
            ),
            EdgeEvent::Call {
                from_span,
                to_span,
                host_duration_ns,
            } => {
                let target = by_id[to_span.as_str()];
                if target.parent_id.as_deref() != Some(from_span.as_str()) {
                    error(
                        &mut issues,
                        "call_edge_hierarchy_mismatch",
                        format!(
                            "call edge `{from_span}` -> `{to_span}` does not match the span hierarchy"
                        ),
                    );
                }
                if target.closed && *host_duration_ns != target.duration_ns {
                    error(
                        &mut issues,
                        "call_edge_duration_mismatch",
                        format!(
                            "call edge `{from_span}` -> `{to_span}` reports {host_duration_ns} ns but the span reports {} ns",
                            target.duration_ns
                        ),
                    );
                }
            }
            EdgeEvent::Data {
                from_tensor,
                to_tensor,
            } if from_tensor.is_empty() || to_tensor.is_empty() => error(
                &mut issues,
                "empty_data_edge_endpoint",
                "data-edge tensor IDs cannot be empty",
            ),
            EdgeEvent::Data {
                from_tensor,
                to_tensor,
            } if !known_tensors.contains(from_tensor.as_str())
                || !known_tensors.contains(to_tensor.as_str()) =>
            {
                error(
                    &mut issues,
                    "unknown_data_edge_tensor",
                    format!(
                        "data edge `{from_tensor}` -> `{to_tensor}` refers to unknown tensor evidence"
                    ),
                )
            }
            _ => {}
        }
    }

    let mut live_memory: HashMap<(&str, &str), (u64, HashSet<&str>)> = HashMap::new();
    let mut memory = doc.memory.iter().collect::<Vec<_>>();
    memory.sort_by_key(|event| event.timestamp_ns);
    for event in memory {
        let key = (event.device.as_str(), event.storage_id.as_str());
        match event.action {
            MemoryAction::Alloc => match live_memory.get_mut(&key) {
                Some((bytes, _)) if *bytes != event.bytes => error(
                    &mut issues,
                    "allocation_size_mismatch",
                    format!(
                        "storage `{}` on `{}` has conflicting allocation sizes",
                        event.storage_id, event.device
                    ),
                ),
                Some((_, tensor_ids)) => {
                    if !tensor_ids.insert(event.tensor_id.as_str()) {
                        error(
                            &mut issues,
                            "duplicate_allocation",
                            format!(
                                "tensor `{}` repeated an allocation for storage `{}` on `{}`",
                                event.tensor_id, event.storage_id, event.device
                            ),
                        );
                    }
                }
                None => {
                    live_memory.insert(
                        key,
                        (event.bytes, HashSet::from([event.tensor_id.as_str()])),
                    );
                }
            },
            MemoryAction::Free => match live_memory.remove(&key) {
                None => error(
                    &mut issues,
                    "unpaired_free",
                    format!(
                        "storage `{}` on `{}` was freed while not live",
                        event.storage_id, event.device
                    ),
                ),
                Some((bytes, _)) if bytes != event.bytes => error(
                    &mut issues,
                    "allocation_size_mismatch",
                    format!(
                        "storage `{}` allocated {bytes} bytes but freed {}",
                        event.storage_id, event.bytes
                    ),
                ),
                Some(_) => {}
            },
        }
    }
    if !live_memory.is_empty() {
        warning(
            &mut issues,
            "retained_allocations",
            format!(
                "{} storages remained live at capture end",
                live_memory.len()
            ),
        );
    }

    validate_gradient_manifest(doc, failed, &mut issues);

    let coverage = EvidenceCoverage {
        spans: doc.spans.len(),
        closed_spans: doc.spans.iter().filter(|span| span.closed).count(),
        root_spans,
        measured_spans,
        operations: doc.ops.len(),
        tensors: doc.tensors.len(),
        memory_events: doc.memory.len(),
        device_memory_samples: doc.device_memory.len(),
        device_intervals: doc.device_intervals.len(),
        gradients: doc.gradients.len(),
        call_edges: doc
            .edges
            .iter()
            .filter(|edge| matches!(edge, EdgeEvent::Call { .. }))
            .count(),
        data_edges: doc
            .edges
            .iter()
            .filter(|edge| matches!(edge, EdgeEvent::Data { .. }))
            .count(),
        forward_spans: step_count(doc, ExecutionStep::Forward),
        backward_spans: step_count(doc, ExecutionStep::Backward),
        optimizer_spans: step_count(doc, ExecutionStep::Optimizer),
    };

    for (empty, code, message) in [
        (
            coverage.operations == 0,
            "operations_absent",
            "no operation evidence was captured",
        ),
        (
            coverage.tensors == 0,
            "tensors_absent",
            "no tensor checkpoints were captured",
        ),
        (
            coverage.memory_events == 0,
            "logical_memory_absent",
            "no logical storage events were captured",
        ),
        (
            coverage.device_memory_samples == 0,
            "physical_memory_absent",
            "no physical device-memory samples were captured",
        ),
        (
            coverage.device_intervals == 0,
            "device_timing_absent",
            "no device timing intervals were captured",
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
    let mut required_labels = HashSet::new();
    for required in &doc.run.capture_contract.required_semantic_labels {
        if !required_labels.insert(required.as_str()) {
            warning(
                &mut issues,
                "duplicate_required_semantic_label",
                format!("required semantic label `{required}` is declared more than once"),
            );
        }
        let count = doc
            .spans
            .iter()
            .filter(|span| span.name == *required)
            .count();
        if count != 1 {
            warning(
                &mut issues,
                "required_semantic_label_cardinality",
                format!(
                    "required semantic label `{required}` must occur exactly once; observed {count}"
                ),
            );
        }
    }

    TraceHealth {
        structurally_valid: !issues
            .iter()
            .any(|issue| issue.severity == HealthSeverity::Error),
        capture_complete: !failed,
        issues,
        coverage,
    }
}

fn validate_gradient_manifest(doc: &TraceDocument, failed: bool, issues: &mut Vec<HealthIssue>) {
    let mut event_ids = HashSet::new();
    let mut observed = BTreeMap::<(&str, &str), usize>::new();
    let mut events_by_key = BTreeMap::new();
    for gradient in &doc.gradients {
        if gradient.event_id.trim().is_empty() {
            error(
                issues,
                "empty_gradient_event_id",
                "gradient event IDs must not be empty",
            );
        }
        if gradient.root.trim().is_empty() || gradient.key.trim().is_empty() {
            error(
                issues,
                "empty_gradient_parameter_key",
                "gradient roots and parameter keys must not be empty",
            );
        }
        if !event_ids.insert(gradient.event_id.as_str()) {
            error(
                issues,
                "duplicate_gradient_event_id",
                format!(
                    "gradient event ID {:?} occurs more than once",
                    gradient.event_id
                ),
            );
        }
        *observed
            .entry((gradient.root.as_str(), gradient.key.as_str()))
            .or_default() += 1;
        events_by_key
            .entry((gradient.root.as_str(), gradient.key.as_str()))
            .or_insert(gradient);
        if !gradient.state.norm_is_valid(gradient.norm) {
            error(
                issues,
                "gradient_state_norm_inconsistent",
                format!(
                    "gradient ({:?}, {:?}) state `{}` is inconsistent with norm {:?}",
                    gradient.root, gradient.key, gradient.state, gradient.norm
                ),
            );
        }
    }

    let declared = doc.run.capture_contract.gradients;
    let contract = doc.run.capture_contract.gradient_contract.as_ref();
    match (declared, contract) {
        (CoverageLevel::Complete, None) => {
            error(
                issues,
                "gradient_contract_missing",
                "complete gradient coverage requires an exact gradient contract",
            );
            return;
        }
        (CoverageLevel::Complete, Some(_)) | (_, None) => {}
        (_, Some(_)) => {
            error(
                issues,
                "gradient_contract_without_complete_coverage",
                "an exact gradient contract requires complete declared gradient coverage",
            );
            return;
        }
    }
    let Some(contract) = contract else {
        return;
    };
    if let Err(contract_error) = contract.validate() {
        error(
            issues,
            "gradient_contract_invalid",
            contract_error.to_string(),
        );
        return;
    }

    let expected = contract
        .expected
        .iter()
        .map(|gradient| (gradient.root.as_str(), gradient.key.as_str()))
        .collect::<BTreeSet<_>>();
    for (&(root, key), &count) in &observed {
        if count != 1 {
            error(
                issues,
                "gradient_manifest_duplicate_key",
                format!("gradient ({root:?}, {key:?}) occurs {count} times; expected exactly once"),
            );
        }
        if !expected.contains(&(root, key)) {
            error(
                issues,
                "gradient_manifest_undeclared_key",
                format!("gradient ({root:?}, {key:?}) is absent from the manifest"),
            );
        }
    }
    for parameter in &contract.expected {
        if !observed.contains_key(&(parameter.root.as_str(), parameter.key.as_str())) {
            let root = &parameter.root;
            let key = &parameter.key;
            let message = format!("manifest gradient ({root:?}, {key:?}) was not captured");
            if failed {
                warning(issues, "gradient_manifest_missing_key", message);
            } else {
                error(issues, "gradient_manifest_missing_key", message);
            }
        }
    }

    for family in &contract.families {
        let expected_members = contract
            .expected
            .iter()
            .filter(|parameter| parameter.family == family.family)
            .count();
        let members = contract
            .expected
            .iter()
            .filter(|parameter| parameter.family == family.family)
            .filter_map(|parameter| {
                events_by_key
                    .get(&(parameter.root.as_str(), parameter.key.as_str()))
                    .copied()
            })
            .collect::<Vec<_>>();
        let family_capture_complete = members.len() == expected_members;
        let present = members
            .iter()
            .filter(|gradient| gradient.state == GradientState::Present)
            .count();
        let attached = members
            .iter()
            .filter(|gradient| gradient.state != GradientState::Missing)
            .count();
        let non_finite = members
            .iter()
            .filter(|gradient| gradient.state == GradientState::NonFinite)
            .count();
        if non_finite > 0 {
            error(
                issues,
                "gradient_family_non_finite",
                format!(
                    "gradient family {:?} contains {non_finite} non-finite gradients",
                    family.family
                ),
            );
        }
        match family.expectation {
            GradientFamilyExpectation::Active
                if (!failed || family_capture_complete) && present < family.min_present => error(
                issues,
                "gradient_active_family_below_minimum",
                format!(
                    "active gradient family {:?} has {present} present gradients; requires at least {}",
                    family.family, family.min_present
                ),
            ),
            GradientFamilyExpectation::Inactive if attached > 0 => error(
                issues,
                "gradient_inactive_family_leakage",
                format!(
                    "inactive gradient family {:?} has {attached} attached gradients",
                    family.family
                ),
            ),
            GradientFamilyExpectation::DataConditional
                if (!failed || family_capture_complete)
                    && attached > 0
                    && present < family.min_present =>
            {
                error(
                    issues,
                    "gradient_conditional_family_below_minimum",
                    format!(
                        "data-conditional gradient family {:?} was attached but has {present} present gradients; requires at least {}",
                        family.family, family.min_present
                    ),
                )
            }
            _ => {}
        }
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
    use crate::capability::CaptureContract;
    use crate::trace::{
        MemoryCategory, MemoryEvent, RunOutcome, SpanKind, SpanRecord, TerminalEvent, TimingMode,
        TraceRunMeta, SCHEMA,
    };

    fn failed_document() -> TraceDocument {
        TraceDocument {
            schema: SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "failed".into(),
                correlation_id: "failed/run".into(),
                entrypoint: "demo".into(),
                phase: ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
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
                measured: false,
                start_ns: 0,
                closed: false,
                duration_ns: 0,
                step: None,
            }],
            ops: vec![],
            tensors: vec![],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Failed,
                timestamp_ns: 10,
                reason: Some("boom".into()),
            },
        }
    }

    #[test]
    fn failed_capture_is_diagnosable_without_becoming_complete() {
        let health = analyze_health(&failed_document());
        assert!(health.structurally_valid);
        assert!(!health.capture_complete);
        assert!(health
            .issues
            .iter()
            .any(|issue| issue.code == "capture_failed"));
        assert!(health
            .issues
            .iter()
            .any(|issue| issue.code == "open_span" && issue.severity == HealthSeverity::Warning));
    }

    #[test]
    fn distinct_tensor_aliases_share_one_live_storage() {
        let mut document = failed_document();
        let memory = |timestamp_ns, tensor_id: &str, action| MemoryEvent {
            timestamp_ns,
            storage_id: "shared".into(),
            tensor_id: tensor_id.into(),
            span_id: "root".into(),
            op_name: None,
            device: "cpu".into(),
            bytes: 64,
            action,
            shape: vec![16],
            dtype: "f32".into(),
            category: MemoryCategory::Activation,
        };
        document.memory = vec![
            memory(1, "base", MemoryAction::Alloc),
            memory(2, "view", MemoryAction::Alloc),
            memory(3, "view", MemoryAction::Free),
        ];
        let health = analyze_health(&document);
        assert!(health.structurally_valid);
        assert!(!health
            .issues
            .iter()
            .any(|issue| issue.code == "duplicate_allocation"));
    }
}
