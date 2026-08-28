//! Capability-qualified evidence packet shared by the CLI, bundles, and viewer.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::capability::{
    CapabilityKind, CapabilityLevel, CapabilityState, CoverageLevel, EvidenceCapabilities,
};
use crate::graph::{build_from_trace, ExecutionGraph};
use crate::nsight::{GpuEvidenceStatus, NsightEvidence, ProvenanceBindingState};
use crate::timing::{analyze_timing, TimingProfile};
use crate::trace::memory::{analyze_memory, MemoryProfile};
use crate::trace::{
    analyze_health, parse_trace, HealthSeverity, TensorStatsEvent, TraceDocument, TraceHealth,
    TraceRunMeta,
};

pub const SCHEMA: &str = "candle-graph/evidence/4";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencePacket {
    #[serde(deserialize_with = "deserialize_schema")]
    pub schema: String,
    pub provenance: TraceRunMeta,
    pub health: TraceHealth,
    pub capabilities: EvidenceCapabilities,
    pub findings: Vec<EvidenceFinding>,
    pub facts: Vec<EvidenceFact>,
    pub gaps: Vec<String>,
    /// Failed or structurally invalid captures remain diagnosable without a derived graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<ExecutionGraph>,
    /// Ordered caller-labeled numerical summaries from the trace.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_stats: Vec<TensorStatsEvent>,
    pub timing: TimingProfile,
    pub memory: MemoryProfile,
    pub gpu: NsightEvidence,
}

fn deserialize_schema<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let schema = String::deserialize(deserializer)?;
    if schema != SCHEMA {
        return Err(serde::de::Error::custom(format_args!(
            "unsupported evidence schema {schema:?}; expected {SCHEMA:?}"
        )));
    }
    Ok(schema)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceFinding {
    pub code: String,
    pub summary: String,
    pub source: String,
    pub requires: Vec<CapabilityKind>,
    pub qualification: CapabilityLevel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceFact {
    pub code: String,
    pub label: String,
    pub value: FactValue,
    pub source: String,
    pub capability: CapabilityKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FactValue {
    DurationNs(u64),
    Bytes(u64),
    Count(u64),
    Text(String),
}

pub fn build_evidence(trace: &Path, nsight_dir: Option<&Path>) -> Result<EvidencePacket> {
    let document =
        parse_trace(trace).with_context(|| format!("parse trace {}", trace.display()))?;
    let contract = &document.run.capture_contract;
    let required_application_labels = contract.required_semantic_labels.clone();
    let gpu_expected_semantic_labels = contract.resolved_gpu_expected_semantic_labels();
    let cpu_only_semantic_labels = contract.resolved_cpu_only_semantic_labels();
    let gpu = NsightEvidence::load_optional_with_semantic_contract(
        nsight_dir,
        &required_application_labels,
        &gpu_expected_semantic_labels,
        &cpu_only_semantic_labels,
    );
    EvidencePacket::from_document(document, gpu)
}

impl EvidencePacket {
    /// Reject packets from older semantic contracts, including nested graphs that only happen to
    /// deserialize into the current Rust representation.
    pub fn validate_schema(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA,
            "unsupported evidence schema {:?}; expected {:?}",
            self.schema,
            SCHEMA
        );
        if let Some(graph) = &self.graph {
            ensure!(
                graph.schema == crate::graph::SCHEMA,
                "unsupported graph schema {:?}; expected {:?}",
                graph.schema,
                crate::graph::SCHEMA
            );
        }
        Ok(())
    }

    pub fn from_document(document: TraceDocument, mut gpu: NsightEvidence) -> Result<Self> {
        gpu.bind_to_trace(&document.run.run_id, &document.run.correlation_id);
        let health = analyze_health(&document);
        let timing = analyze_timing(&document);
        let memory = analyze_memory(&document);
        let capabilities = assess_capabilities(&document, &health, &timing, &memory, &gpu);
        let graph = (health.capture_complete && health.structurally_valid)
            .then(|| build_from_trace(&document))
            .transpose()?;
        let tensor_stats = document.tensor_stats.clone();
        let mut findings = Vec::new();
        let mut facts = Vec::new();

        if capabilities.outer_wall_time.is_available() {
            if let Some(duration_ns) = document
                .spans
                .iter()
                .find(|span| span.measured && span.closed)
                .map(|span| span.duration_ns)
            {
                facts.push(EvidenceFact {
                    code: "outer_wall_time".into(),
                    label: "Measured region wall time".into(),
                    value: FactValue::DurationNs(duration_ns),
                    source: "measured_span".into(),
                    capability: CapabilityKind::OuterWallTime,
                });
            }
        }
        if capabilities.gradient_coverage.is_available() {
            if let Some(contract) = document.run.capture_contract.gradient_contract.as_ref() {
                facts.extend([
                    EvidenceFact {
                        code: "gradient_manifest_sha256".into(),
                        label: "Gradient manifest SHA-256".into(),
                        value: FactValue::Text(contract.manifest_sha256.clone()),
                        source: "capture_contract.gradient_contract".into(),
                        capability: CapabilityKind::Gradients,
                    },
                    EvidenceFact {
                        code: "gradient_manifest_entries".into(),
                        label: "Expected gradient parameters".into(),
                        value: FactValue::Count(contract.expected.len() as u64),
                        source: "capture_contract.gradient_contract".into(),
                        capability: CapabilityKind::Gradients,
                    },
                    EvidenceFact {
                        code: "gradient_family_expectations".into(),
                        label: "Gradient family expectations".into(),
                        value: FactValue::Count(contract.families.len() as u64),
                        source: "capture_contract.gradient_contract".into(),
                        capability: CapabilityKind::Gradients,
                    },
                ]);
            }
        }
        if let Some(graph) = &graph {
            if let Some(span) = graph.summary.slowest_host_spans.first() {
                findings.push(EvidenceFinding {
                    code: "largest_observed_host_self_time".into(),
                    summary: format!(
                        "`{}` has the largest observed measured-scope host self-time ({:.2} ms overlap-clipped, {:.2} ms full self-time; `{}` span duration {:.2} ms of {:.2} ms full)",
                        span.name,
                        span.measured_overlap_self_time_ns as f64 / 1_000_000.0,
                        span.host_self_time_ns as f64 / 1_000_000.0,
                        span.scope.as_str(),
                        span.measured_overlap_duration_ns as f64 / 1_000_000.0,
                        span.full_duration_ns as f64 / 1_000_000.0,
                    ),
                    source: span.id.clone(),
                    requires: vec![CapabilityKind::NestedHostTime],
                    qualification: capabilities.nested_host_time.level,
                });
            }
            if let Some(span) = graph.summary.slowest_device_spans.first() {
                findings.push(EvidenceFinding {
                    code: "largest_observed_device_busy_time".into(),
                    summary: format!("`{}` has the largest observed device-busy interval union on `{}` ({:.2} ms)", span.name, span.device, span.device_busy_ns as f64 / 1_000_000.0),
                    source: span.id.clone(),
                    requires: vec![CapabilityKind::NestedDeviceTime],
                    qualification: capabilities.nested_device_time.level,
                });
            }
            let non_present = graph
                .gradients
                .iter()
                .filter(|gradient| {
                    !matches!(gradient.state, crate::graph::GradientRecordState::Present)
                })
                .count();
            if !graph.gradients.is_empty() {
                facts.push(EvidenceFact {
                    code: "gradient_observations".into(),
                    label: "Non-present gradient observations".into(),
                    value: FactValue::Count(non_present as u64),
                    source: "trace.gradient".into(),
                    capability: CapabilityKind::Gradients,
                });
            }
        }
        if capabilities.logical_memory_coverage.is_available() {
            if let Some(logical) = &memory.logical {
                if let Some(peak) = &logical.peak {
                    facts.push(EvidenceFact {
                        code: "logical_peak_live_bytes".into(),
                        label: "Peak live logical storage".into(),
                        value: FactValue::Bytes(peak.live_bytes),
                        source: "logical_storage_lifetimes".into(),
                        capability: CapabilityKind::LogicalMemory,
                    });
                }
            }
        }
        if health.capture_complete
            && health.structurally_valid
            && capabilities.gpu_correlation.is_available()
            && capabilities.provenance_binding.is_available()
        {
            if let Some(kernel) = gpu.kernels.first() {
                findings.push(EvidenceFinding {
                    code: "largest_nsight_kernel_total".into(),
                    summary: format!(
                        "`{}` has the largest normalized Nsight kernel total ({:.2} ms)",
                        kernel.name,
                        kernel.total_ns as f64 / 1_000_000.0
                    ),
                    source: "nsight.cuda_gpu_kern_sum".into(),
                    requires: vec![
                        CapabilityKind::GpuCorrelation,
                        CapabilityKind::ProvenanceBinding,
                    ],
                    qualification: weakest_level(
                        capabilities.gpu_correlation.level,
                        capabilities.provenance_binding.level,
                    ),
                });
            }
        }

        let mut gaps = health
            .gaps()
            .map(|issue| issue.message.clone())
            .collect::<Vec<_>>();
        for state in [
            &capabilities.structural_trace,
            &capabilities.outer_wall_time,
            &capabilities.nested_host_time,
            &capabilities.nested_device_time,
            &capabilities.operation_coverage,
            &capabilities.tensor_coverage,
            &capabilities.gradient_coverage,
            &capabilities.logical_memory_coverage,
            &capabilities.physical_memory_coverage,
            &capabilities.gpu_correlation,
            &capabilities.provenance_binding,
        ] {
            if !state.is_complete() {
                gaps.push(state.reason.clone());
            }
        }
        gaps.sort();
        gaps.dedup();

        Ok(Self {
            schema: SCHEMA.into(),
            provenance: document.run,
            health,
            capabilities,
            findings,
            facts,
            gaps,
            graph,
            tensor_stats,
            timing,
            memory,
            gpu,
        })
    }

    pub fn markdown(&self) -> String {
        let status = if !self.health.structurally_valid {
            "STRUCTURALLY INVALID"
        } else if !self.health.capture_complete {
            "FAILED CAPTURE"
        } else {
            "COMPLETE CAPTURE"
        };
        let mut output = format!(
            "# candle-graph evidence\n\n- Status: **{status}**\n- Entrypoint: `{}`\n- Run: `{}`\n- Phase: `{}`\n- Device: `{}`\n\n## Capability matrix\n\n| Capability | Level | Source | Reason |\n| --- | --- | --- | --- |\n",
            self.provenance.entrypoint, self.provenance.run_id, self.provenance.phase.as_str(), self.provenance.device,
        );
        for (name, state) in capability_rows(&self.capabilities) {
            output.push_str(&format!(
                "| {name} | `{:?}` | `{}` | {} |\n",
                state.level,
                state.source,
                state.reason.replace('|', "\\|")
            ));
        }
        output.push_str("\n## Qualified findings\n\n");
        if self.findings.is_empty() {
            output.push_str("No findings met their evidence prerequisites.\n");
        } else {
            for finding in &self.findings {
                let requirements = finding
                    .requires
                    .iter()
                    .map(|requirement| format!("`{requirement:?}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                output.push_str(&format!(
                    "- [`{:?}`] {} (requires {requirements})\n",
                    finding.qualification, finding.summary
                ));
            }
        }
        output.push_str("\n## Evidence gaps\n\n");
        if self.gaps.is_empty() {
            output.push_str("No declared capability gaps.\n");
        } else {
            for gap in &self.gaps {
                output.push_str(&format!("- {gap}\n"));
            }
        }
        output
    }
}

fn assess_capabilities(
    document: &TraceDocument,
    health: &TraceHealth,
    timing: &TimingProfile,
    memory: &MemoryProfile,
    gpu: &NsightEvidence,
) -> EvidenceCapabilities {
    let trace_validation_source = || format!("{} validation", crate::trace::SCHEMA);
    let structural_trace = if health.structurally_valid {
        CapabilityState::from_coverage(
            CoverageLevel::Complete,
            trace_validation_source(),
            "span and event invariants passed",
        )
    } else {
        CapabilityState::invalid(
            trace_validation_source(),
            "one or more structural invariants failed",
        )
    };
    let measured = document
        .spans
        .iter()
        .filter(|span| span.measured && span.closed)
        .count();
    let outer_wall_time = if !health.structurally_valid {
        CapabilityState::invalid(
            trace_validation_source(),
            "outer wall time is not qualified for a structurally invalid trace",
        )
    } else if !health.capture_complete {
        CapabilityState::unavailable("capture did not complete")
    } else if measured == 1 {
        CapabilityState::from_coverage(
            CoverageLevel::Complete,
            "measured span",
            "one closed measured region was observed",
        )
    } else {
        CapabilityState::invalid(
            "measured span",
            "exactly one closed measured region is required",
        )
    };
    let nested_host_time = if !health.structurally_valid {
        CapabilityState::invalid(
            trace_validation_source(),
            "nested host attribution requires a structurally valid trace",
        )
    } else if health.capture_complete {
        CapabilityState::from_coverage(
            CoverageLevel::Complete,
            "span wall intervals",
            "measured-subtree and concurrent-overlap host intervals are structurally valid",
        )
    } else {
        CapabilityState::unavailable(
            "measured-scope host attribution requires a complete, valid capture",
        )
    };
    let nested_device_time = observed_coverage(
        timing.device_coverage,
        document.device_intervals.len(),
        "device intervals",
        "device timing",
    );
    let operation_coverage = observed_coverage(
        document.run.capture_contract.operations,
        document.ops.len(),
        "trace op events",
        "operation coverage",
    );
    let tensor_coverage = observed_coverage(
        document.run.capture_contract.tensors,
        document.tensors.len(),
        "trace tensor events",
        "tensor coverage",
    );
    let gradient_coverage = assess_gradient_coverage(document, health);
    let logical_memory_coverage = observed_coverage(
        document.run.capture_contract.logical_memory,
        document.memory.len(),
        "storage lifetime events",
        "logical memory coverage",
    );
    let physical_memory_coverage = observed_coverage(
        document.run.capture_contract.physical_memory,
        document.device_memory.len(),
        "device memory samples",
        "physical memory coverage",
    );
    let provenance_binding = match (gpu.status, gpu.provenance.binding) {
        (GpuEvidenceStatus::Unavailable, _) => {
            CapabilityState::unavailable("no Nsight artifacts were supplied")
        }
        (GpuEvidenceStatus::Failed, _) => {
            CapabilityState::invalid("Nsight artifact manifest", "artifact loading failed")
        }
        (_, ProvenanceBindingState::Bound) => CapabilityState::from_coverage(
            CoverageLevel::Complete,
            "capture-manifest.json",
            "manifest IDs and artifact hashes match this trace",
        ),
        (_, ProvenanceBindingState::Partial) => CapabilityState::from_coverage(
            CoverageLevel::Partial,
            "Nsight artifact hashes",
            "artifacts are hashed, but trace binding is incomplete",
        ),
        (_, ProvenanceBindingState::Mismatch) => CapabilityState::invalid(
            "capture-manifest.json",
            "manifest IDs or artifact hashes do not match",
        ),
    };
    let required_labels = &document.run.capture_contract.required_semantic_labels;
    let required_application_labels_present = required_labels
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        == required_labels.len()
        && required_labels.iter().all(|required| {
            document
                .spans
                .iter()
                .filter(|span| span.name == *required)
                .count()
                == 1
        });
    let gpu_correlation = if !required_application_labels_present {
        CapabilityState::invalid(
            "trace semantic labels",
            "one or more labels required for GPU correlation are absent from the application trace",
        )
    } else {
        match gpu.status {
            GpuEvidenceStatus::Available if gpu.correlation.complete => {
                CapabilityState::from_coverage(
                    CoverageLevel::Complete,
                    "Nsight NVTX projection",
                    "application and projected semantic labels matched",
                )
            }
            GpuEvidenceStatus::Available => CapabilityState::from_coverage(
                CoverageLevel::Partial,
                "Nsight NVTX projection",
                gpu.correlation
                    .reason
                    .clone()
                    .unwrap_or_else(|| "correlation is incomplete".into()),
            ),
            GpuEvidenceStatus::Unavailable => CapabilityState::unavailable(
                gpu.reason
                    .clone()
                    .unwrap_or_else(|| "Nsight evidence was not supplied".into()),
            ),
            GpuEvidenceStatus::Failed => CapabilityState::invalid(
                "Nsight normalization",
                gpu.reason
                    .clone()
                    .unwrap_or_else(|| "Nsight normalization failed".into()),
            ),
        }
    };
    debug_assert_eq!(memory.logical.is_some(), !document.memory.is_empty());
    debug_assert_eq!(
        memory.physical.is_some(),
        !document.device_memory.is_empty()
    );
    let mut capabilities = EvidenceCapabilities {
        structural_trace,
        outer_wall_time,
        nested_host_time,
        nested_device_time,
        operation_coverage,
        tensor_coverage,
        gradient_coverage,
        logical_memory_coverage,
        physical_memory_coverage,
        gpu_correlation,
        provenance_binding,
    };
    if !health.structurally_valid {
        for state in [
            &mut capabilities.nested_device_time,
            &mut capabilities.operation_coverage,
            &mut capabilities.tensor_coverage,
            &mut capabilities.gradient_coverage,
            &mut capabilities.logical_memory_coverage,
            &mut capabilities.gpu_correlation,
        ] {
            if state.is_available() {
                *state = CapabilityState::invalid(
                    trace_validation_source(),
                    "structurally invalid trace cannot qualify trace-linked evidence",
                );
            }
        }
    }
    if !health.capture_complete {
        for state in [
            &mut capabilities.nested_device_time,
            &mut capabilities.operation_coverage,
            &mut capabilities.tensor_coverage,
            &mut capabilities.gradient_coverage,
            &mut capabilities.logical_memory_coverage,
            &mut capabilities.physical_memory_coverage,
            &mut capabilities.gpu_correlation,
            &mut capabilities.provenance_binding,
        ] {
            if state.level == CapabilityLevel::Complete {
                state.level = CapabilityLevel::Partial;
                state.reason = format!(
                    "{}; failed capture means observations are diagnostic only",
                    state.reason
                );
            }
        }
    }
    capabilities
}

fn assess_gradient_coverage(document: &TraceDocument, health: &TraceHealth) -> CapabilityState {
    let declared = document.run.capture_contract.gradients;
    if declared != CoverageLevel::Complete {
        return observed_coverage(
            declared,
            document.gradients.len(),
            "trace gradient events",
            "gradient coverage",
        );
    }

    let Some(contract) = document.run.capture_contract.gradient_contract.as_ref() else {
        return CapabilityState::invalid(
            "exact gradient contract",
            "complete gradient coverage requires a digest-bound parameter manifest",
        );
    };
    if let Err(error) = contract.validate() {
        return CapabilityState::invalid(
            "exact gradient contract",
            format!("gradient contract validation failed: {error}"),
        );
    }

    let gradient_issues = health
        .issues
        .iter()
        .filter(|issue| {
            issue.code.starts_with("gradient_")
                || matches!(
                    issue.code.as_str(),
                    "duplicate_gradient_event_id" | "empty_gradient_event_id"
                )
        })
        .collect::<Vec<_>>();
    if gradient_issues
        .iter()
        .any(|issue| issue.severity == HealthSeverity::Error)
    {
        return CapabilityState::invalid(
            "exact gradient contract",
            "gradient events did not satisfy the exact manifest and family contract",
        );
    }
    if !gradient_issues.is_empty() {
        return CapabilityState::from_coverage(
            CoverageLevel::Partial,
            "exact gradient contract",
            "capture ended before every manifest and family expectation could be validated",
        );
    }

    CapabilityState::from_coverage(
        CoverageLevel::Complete,
        "exact gradient contract",
        format!(
            "{} manifest entries and {} family expectations validated against {}",
            contract.expected.len(),
            contract.families.len(),
            contract.manifest_sha256
        ),
    )
}

fn observed_coverage(
    declared: CoverageLevel,
    observations: usize,
    source: &str,
    label: &str,
) -> CapabilityState {
    match (declared, observations) {
        (CoverageLevel::Complete, 0) => CapabilityState::invalid(
            source,
            format!("producer declared complete {label}, but emitted no observations"),
        ),
        (CoverageLevel::Complete, _) => CapabilityState::from_coverage(
            CoverageLevel::Complete,
            source,
            format!("producer declared complete {label}"),
        ),
        (CoverageLevel::Partial, 0) => CapabilityState::unavailable(format!(
            "producer declared partial {label}, but this run emitted no observations"
        )),
        (CoverageLevel::Partial, _) => CapabilityState::from_coverage(
            CoverageLevel::Partial,
            source,
            format!("producer declared partial {label}"),
        ),
        (CoverageLevel::None, 0) => {
            CapabilityState::unavailable(format!("producer did not declare or emit {label}"))
        }
        (CoverageLevel::None, _) => CapabilityState::from_coverage(
            CoverageLevel::Partial,
            source,
            format!("observations exist, but producer did not declare complete {label}"),
        ),
    }
}

fn capability_rows(capabilities: &EvidenceCapabilities) -> [(&'static str, &CapabilityState); 11] {
    [
        ("Structural trace", &capabilities.structural_trace),
        ("Outer wall time", &capabilities.outer_wall_time),
        ("Measured-scope host time", &capabilities.nested_host_time),
        ("Nested device time", &capabilities.nested_device_time),
        ("Operations", &capabilities.operation_coverage),
        ("Tensors", &capabilities.tensor_coverage),
        ("Gradients", &capabilities.gradient_coverage),
        ("Logical memory", &capabilities.logical_memory_coverage),
        ("Physical memory", &capabilities.physical_memory_coverage),
        ("GPU correlation", &capabilities.gpu_correlation),
        ("Provenance binding", &capabilities.provenance_binding),
    ]
}

fn weakest_level(left: CapabilityLevel, right: CapabilityLevel) -> CapabilityLevel {
    use CapabilityLevel::{Complete, Invalid, Partial, Unavailable};
    match (left, right) {
        (Invalid, _) | (_, Invalid) => Invalid,
        (Unavailable, _) | (_, Unavailable) => Unavailable,
        (Partial, _) | (_, Partial) => Partial,
        (Complete, Complete) => Complete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{CaptureContract, MeasurementScope};
    use crate::nsight::NsightEvidence;
    use crate::trace::{
        GradientEvent, GradientState, OpEvent, RunOutcome, SpanKind, SpanRecord, TerminalEvent,
        TimingMode, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };

    #[test]
    fn failed_capture_downgrades_observed_capabilities_and_emits_no_findings() {
        let document = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "failed".into(),
                correlation_id: "failed/run".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract {
                    measurement_scope: MeasurementScope::ProfiledWork,
                    operations: CoverageLevel::Complete,
                    ..CaptureContract::default()
                },
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
            ops: vec![OpEvent {
                span_id: "root".into(),
                op_name: "add".into(),
                inputs: vec![],
                output: None,
                shape: vec![1],
                dtype: "f32".into(),
                device: "cpu".into(),
                duration_ns: 1,
                timestamp_ns: 1,
                output_dense_bytes: Some(4),
                input_dense_bytes: 0,
            }],
            tensors: vec![],
            tensor_stats: vec![],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Failed,
                timestamp_ns: 2,
                reason: Some("interrupted".into()),
            },
        };
        let packet =
            EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
                .unwrap();
        assert_eq!(
            packet.capabilities.operation_coverage.level,
            CapabilityLevel::Partial
        );
        assert_eq!(
            packet.capabilities.structural_trace.source,
            format!("{TRACE_SCHEMA} validation")
        );
        assert!(packet.findings.is_empty());
        assert!(packet.graph.is_none());
    }

    #[test]
    fn tensor_and_gradient_capabilities_enforce_declared_coverage() {
        let document = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "typed-coverage".into(),
                correlation_id: "typed/coverage".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Train,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract {
                    measurement_scope: MeasurementScope::ProfiledWork,
                    tensors: CoverageLevel::Complete,
                    gradients: CoverageLevel::None,
                    ..CaptureContract::default()
                },
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
            tensors: vec![],
            tensor_stats: vec![],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![GradientEvent {
                event_id: "gradient-1".into(),
                root: "parameters".into(),
                key: "weight".into(),
                state: GradientState::Present,
                norm: Some(1.0),
            }],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 1,
                reason: None,
            },
        };
        let packet =
            EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
                .unwrap();
        assert_eq!(
            packet.capabilities.tensor_coverage.level,
            CapabilityLevel::Invalid
        );
        assert_eq!(
            packet.capabilities.gradient_coverage.level,
            CapabilityLevel::Partial
        );
    }

    #[test]
    fn older_capability_packets_default_new_typed_fields() {
        let json = serde_json::json!({
            "structural_trace": CapabilityState::default(),
            "outer_wall_time": CapabilityState::default(),
            "nested_host_time": CapabilityState::default(),
            "nested_device_time": CapabilityState::default(),
            "operation_coverage": CapabilityState::default(),
            "logical_memory_coverage": CapabilityState::default(),
            "physical_memory_coverage": CapabilityState::default(),
            "gpu_correlation": CapabilityState::default(),
            "provenance_binding": CapabilityState::default()
        });
        let capabilities: EvidenceCapabilities = serde_json::from_value(json).unwrap();
        assert_eq!(
            capabilities.tensor_coverage.level,
            CapabilityLevel::Unavailable
        );
        assert_eq!(
            capabilities.gradient_coverage.level,
            CapabilityLevel::Unavailable
        );
    }
}
