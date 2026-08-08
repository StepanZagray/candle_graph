//! One bounded evidence packet for agents, reports, comparisons, and the HTML viewer.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::graph::{build_from_trace, ExecutionGraph};
use crate::nsight::{GpuEvidenceStatus, NsightEvidence};
use crate::trace::{analyze_health, parse_trace, TraceDocument, TraceHealth, TraceRunMeta};

pub const EVIDENCE_SCHEMA: &str = "candle-graph/evidence/1";
pub const COMPARISON_SCHEMA: &str = "candle-graph/comparison/1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencePacket {
    pub schema: String,
    pub provenance: TraceRunMeta,
    pub health: TraceHealth,
    pub findings: Vec<String>,
    pub facts: Vec<EvidenceFact>,
    pub gaps: Vec<String>,
    pub graph: ExecutionGraph,
    pub gpu: NsightEvidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<Comparison>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceFact {
    pub code: String,
    pub label: String,
    pub value: f64,
    pub unit: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub schema: String,
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub comparable: bool,
    pub warnings: Vec<String>,
    pub total_delta_ns: i128,
    pub total_delta_percent: Option<f64>,
    pub baseline_peak_bytes: u64,
    pub candidate_peak_bytes: u64,
    pub peak_delta_bytes: i128,
    pub spans: Vec<SpanComparison>,
    pub gradients: Vec<GradientComparison>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanComparison {
    pub name: String,
    pub baseline_count: usize,
    pub candidate_count: usize,
    pub baseline_total_ns: u64,
    pub candidate_total_ns: u64,
    pub baseline_mean_ns: u64,
    pub candidate_mean_ns: u64,
    pub delta_ns: i128,
    pub delta_percent: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientComparison {
    pub parameter: String,
    pub baseline_state: Option<String>,
    pub candidate_state: Option<String>,
    pub baseline_norm: Option<f64>,
    pub candidate_norm: Option<f64>,
    pub norm_delta: Option<f64>,
}

pub fn build_evidence(
    trace: &Path,
    baseline: Option<&Path>,
    nsight_dir: Option<&Path>,
) -> Result<EvidencePacket> {
    let doc = parse_trace(trace).with_context(|| format!("parse trace {}", trace.display()))?;
    let comparison = baseline
        .map(|path| {
            let baseline =
                parse_trace(path).with_context(|| format!("parse baseline {}", path.display()))?;
            Ok::<_, anyhow::Error>(compare_documents(&baseline, &doc))
        })
        .transpose()?;
    let measured = measured_span_ids(&doc);
    let expected_semantic_keys = doc
        .spans
        .iter()
        .filter(|span| measured.contains(span.id.as_str()))
        .map(|span| span.name.clone())
        .collect::<Vec<_>>();
    EvidencePacket::from_document(
        doc,
        NsightEvidence::load_optional(nsight_dir, &expected_semantic_keys),
        comparison,
    )
}

impl EvidencePacket {
    pub fn from_document(
        doc: TraceDocument,
        gpu: NsightEvidence,
        comparison: Option<Comparison>,
    ) -> Result<Self> {
        let health = analyze_health(&doc);
        let graph = build_from_trace(&doc)?;
        let mut findings = graph
            .summary
            .slowest_spans
            .iter()
            .take(5)
            .map(|span| {
                format!(
                    "{}: {:.2} ms self time",
                    span.name,
                    span.self_time_ns as f64 / 1_000_000.0
                )
            })
            .collect::<Vec<_>>();
        let mut facts = vec![EvidenceFact {
            code: "measured_total".into(),
            label: "Measured update total".into(),
            value: graph.summary.total_ms,
            unit: "ms".into(),
            source: "measured_span".into(),
        }];
        facts.extend(
            graph
                .summary
                .slowest_spans
                .iter()
                .take(8)
                .map(|span| EvidenceFact {
                    code: "span_self_time".into(),
                    label: span.name.clone(),
                    value: span.self_time_ns as f64,
                    unit: "ns".into(),
                    source: span.id.clone(),
                }),
        );
        if !graph.gradients.is_empty() {
            let concerning = graph
                .gradients
                .iter()
                .filter(|gradient| {
                    !matches!(gradient.state, crate::graph::GradientRecordState::Present)
                })
                .count();
            findings.push(format!(
                "{} gradient facts captured; {} require attention",
                graph.gradients.len(),
                concerning
            ));
        }
        if gpu.status == GpuEvidenceStatus::Available {
            if let Some(kernel) = gpu.kernels.first() {
                findings.push(format!(
                    "top GPU kernel `{}`: {:.2} ms total",
                    kernel.name,
                    kernel.total_ns as f64 / 1_000_000.0
                ));
            }
        }
        let mut gaps = health
            .gaps()
            .map(|issue| issue.message.clone())
            .collect::<Vec<_>>();
        if gpu.status != GpuEvidenceStatus::Available {
            gaps.push(
                gpu.reason
                    .clone()
                    .unwrap_or_else(|| "GPU evidence is unavailable".into()),
            );
        }
        Ok(Self {
            schema: EVIDENCE_SCHEMA.into(),
            provenance: doc.run,
            health,
            findings,
            facts,
            gaps,
            graph,
            gpu,
            comparison,
        })
    }

    pub fn markdown(&self) -> String {
        let trust = if self.health.trusted {
            "TRUSTED"
        } else {
            "UNTRUSTED"
        };
        let mut out = format!(
            "# candle-graph evidence\n\n- Status: **{trust}**\n- Entrypoint: `{}`\n- Capture update: {} ({} warmup update{})\n- Device: `{}`\n- Total: {:.2} ms\n\n## Findings\n\n",
            self.provenance.entrypoint,
            self.provenance.capture_step,
            self.provenance.warmup_steps,
            if self.provenance.warmup_steps == 1 { "" } else { "s" },
            self.provenance.device,
            self.graph.summary.total_ms,
        );
        push_list(
            &mut out,
            &self.findings,
            "No trusted findings were derived.",
        );
        out.push_str("\n## Evidence gaps\n\n");
        push_list(&mut out, &self.gaps, "No known gaps.");
        out.push_str("\n## Coverage\n\n```json\n");
        out.push_str(&serde_json::to_string_pretty(&self.health.coverage).unwrap_or_default());
        out.push_str("\n```\n");
        out.push_str("\n## Tensor checkpoints\n\n");
        if self.graph.tensors.is_empty() {
            out.push_str("No tensor checkpoints were captured.\n");
        } else {
            out.push_str("| Tensor | Shape | Dtype | Device | Storage | Requires grad |\n| --- | --- | --- | --- | ---: | --- |\n");
            for tensor in &self.graph.tensors {
                out.push_str(&format!(
                    "| `{}` | `{:?}` | `{}` | `{}` | {} B | {} |\n",
                    tensor.tensor_id,
                    tensor.shape,
                    tensor.dtype,
                    tensor.device,
                    tensor.storage_bytes,
                    tensor.requires_grad
                ));
            }
        }
        out.push_str("\n## Gradient evidence\n\n");
        out.push_str(&format!(
            "{} parameter gradients captured; {} require attention.\n",
            self.graph.gradients.len(),
            self.graph
                .gradients
                .iter()
                .filter(|gradient| !matches!(
                    gradient.state,
                    crate::graph::GradientRecordState::Present
                ))
                .count()
        ));
        if let Some(comparison) = &self.comparison {
            out.push_str("\n## Baseline comparison\n\n");
            match comparison.total_delta_percent {
                Some(percent) => out.push_str(&format!("Total time changed by {percent:+.2}%.\n")),
                None => out.push_str(
                    "Total-time percentage is unavailable because the baseline was zero.\n",
                ),
            }
            for warning in &comparison.warnings {
                out.push_str(&format!("- Warning: {warning}\n"));
            }
        }
        out
    }
}

pub fn compare_documents(baseline: &TraceDocument, candidate: &TraceDocument) -> Comparison {
    let mut warnings = Vec::new();
    let mut comparable = analyze_health(baseline).trusted && analyze_health(candidate).trusted;
    for (name, left, right) in [
        (
            "entrypoint",
            baseline.run.entrypoint.as_str(),
            candidate.run.entrypoint.as_str(),
        ),
        (
            "phase",
            baseline.run.phase.as_str(),
            candidate.run.phase.as_str(),
        ),
        (
            "device",
            baseline.run.device.as_str(),
            candidate.run.device.as_str(),
        ),
    ] {
        if left != right {
            comparable = false;
            warnings.push(format!("{name} differs: `{left}` vs `{right}`"));
        }
    }
    if baseline.run.timing_mode != candidate.run.timing_mode {
        comparable = false;
        warnings.push("timing mode differs".into());
    }
    if baseline.run.warmup_steps != candidate.run.warmup_steps {
        comparable = false;
        warnings.push(format!(
            "warmup differs: {} vs {}",
            baseline.run.warmup_steps, candidate.run.warmup_steps
        ));
    }
    let descriptive = ["source_revision", "source_commit", "build_id"];
    let baseline_conditions = baseline
        .run
        .tags
        .iter()
        .filter(|(key, _)| !descriptive.contains(&key.as_str()))
        .collect::<BTreeMap<_, _>>();
    let candidate_conditions = candidate
        .run
        .tags
        .iter()
        .filter(|(key, _)| !descriptive.contains(&key.as_str()))
        .collect::<BTreeMap<_, _>>();
    if baseline_conditions != candidate_conditions {
        comparable = false;
        warnings.push("workload tags differ; batch/model/precision conditions must match".into());
    }
    for key in descriptive {
        if baseline.run.tags.get(key) != candidate.run.tags.get(key) {
            warnings.push(format!(
                "descriptive `{key}` differs, as expected for a code-change comparison"
            ));
        }
    }
    let baseline_spans = aggregate_spans(baseline);
    let candidate_spans = aggregate_spans(candidate);
    let mut names = baseline_spans
        .keys()
        .chain(candidate_spans.keys())
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    let mut spans = names
        .into_iter()
        .map(|name| {
            let (baseline_count, baseline_total_ns) =
                baseline_spans.get(&name).copied().unwrap_or_default();
            let (candidate_count, candidate_total_ns) =
                candidate_spans.get(&name).copied().unwrap_or_default();
            let delta_ns = candidate_total_ns as i128 - baseline_total_ns as i128;
            SpanComparison {
                name,
                baseline_count,
                candidate_count,
                baseline_total_ns,
                candidate_total_ns,
                baseline_mean_ns: mean(baseline_total_ns, baseline_count),
                candidate_mean_ns: mean(candidate_total_ns, candidate_count),
                delta_ns,
                delta_percent: percent(baseline_total_ns, candidate_total_ns),
            }
        })
        .collect::<Vec<_>>();
    spans.sort_by_key(|span| std::cmp::Reverse(span.delta_ns.unsigned_abs()));
    spans.truncate(50);
    let baseline_total = root_total(baseline);
    let candidate_total = root_total(candidate);
    let baseline_peak = crate::trace::memory::analyze_memory(baseline)
        .summary
        .peak_bytes;
    let candidate_peak = crate::trace::memory::analyze_memory(candidate)
        .summary
        .peak_bytes;
    Comparison {
        schema: COMPARISON_SCHEMA.into(),
        baseline_run_id: baseline.run.run_id.clone(),
        candidate_run_id: candidate.run.run_id.clone(),
        comparable,
        warnings,
        total_delta_ns: candidate_total as i128 - baseline_total as i128,
        total_delta_percent: percent(baseline_total, candidate_total),
        baseline_peak_bytes: baseline_peak,
        candidate_peak_bytes: candidate_peak,
        peak_delta_bytes: candidate_peak as i128 - baseline_peak as i128,
        spans,
        gradients: compare_gradients(baseline, candidate),
    }
}

fn aggregate_spans(doc: &TraceDocument) -> BTreeMap<String, (usize, u64)> {
    let mut result = BTreeMap::new();
    let measured = measured_span_ids(doc);
    for span in doc
        .spans
        .iter()
        .filter(|span| measured.contains(span.id.as_str()))
    {
        let mut path = vec![span.name.clone()];
        let mut parent = span.parent_id.as_deref();
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = parent {
            if !seen.insert(id) {
                break;
            }
            let Some(parent_span) = doc.spans.iter().find(|candidate| candidate.id == id) else {
                break;
            };
            path.push(parent_span.name.clone());
            parent = parent_span.parent_id.as_deref();
        }
        path.reverse();
        let step = span
            .step
            .map(|step| format!("/{step:?}"))
            .unwrap_or_default();
        let key = format!("{} [{}]{step}", path.join("/"), span.kind);
        let entry = result.entry(key).or_insert((0usize, 0u64));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(span.duration_ns);
    }
    result
}

fn measured_span_ids(doc: &TraceDocument) -> std::collections::HashSet<&str> {
    let mut ids = doc
        .spans
        .iter()
        .filter(|span| span.measured)
        .map(|span| span.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    loop {
        let before = ids.len();
        for span in &doc.spans {
            if span
                .parent_id
                .as_deref()
                .is_some_and(|parent| ids.contains(parent))
            {
                ids.insert(span.id.as_str());
            }
        }
        if ids.len() == before {
            return ids;
        }
    }
}

fn compare_gradients(
    baseline: &TraceDocument,
    candidate: &TraceDocument,
) -> Vec<GradientComparison> {
    let baseline = baseline
        .gradients
        .iter()
        .map(|item| (format!("{}/{}", item.root, item.key), item))
        .collect::<BTreeMap<_, _>>();
    let candidate = candidate
        .gradients
        .iter()
        .map(|item| (format!("{}/{}", item.root, item.key), item))
        .collect::<BTreeMap<_, _>>();
    let mut keys = baseline
        .keys()
        .chain(candidate.keys())
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|parameter| {
            let left = baseline.get(&parameter).copied();
            let right = candidate.get(&parameter).copied();
            let changed = left.map(|x| (x.state, x.norm)) != right.map(|x| (x.state, x.norm));
            changed.then(|| GradientComparison {
                parameter,
                baseline_state: left.map(|x| x.state.to_string()),
                candidate_state: right.map(|x| x.state.to_string()),
                baseline_norm: left.and_then(|x| x.norm),
                candidate_norm: right.and_then(|x| x.norm),
                norm_delta: left
                    .and_then(|x| x.norm)
                    .zip(right.and_then(|x| x.norm))
                    .map(|(a, b)| b - a),
            })
        })
        .take(100)
        .collect()
}

fn mean(total: u64, count: usize) -> u64 {
    if count == 0 {
        0
    } else {
        total / count as u64
    }
}

fn root_total(doc: &TraceDocument) -> u64 {
    doc.spans
        .iter()
        .filter(|span| span.measured)
        .map(|span| span.duration_ns)
        .sum()
}

fn percent(baseline: u64, candidate: u64) -> Option<f64> {
    (baseline > 0).then(|| (candidate as f64 - baseline as f64) * 100.0 / baseline as f64)
}

fn push_list(out: &mut String, values: &[String], empty: &str) {
    if values.is_empty() {
        out.push_str(&format!("- {empty}\n"));
    } else {
        for value in values {
            out.push_str(&format!("- {value}\n"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase::{ExecutionPhase, ExecutionStep};
    use crate::trace::{
        GradientEvent, GradientState, SpanKind, SpanRecord, TimingMode, TraceRunMeta, SCHEMA,
    };

    fn document(run_id: &str, measured_ns: u64, forward_ns: u64) -> TraceDocument {
        TraceDocument {
            schema: SCHEMA.into(),
            run: TraceRunMeta {
                run_id: run_id.into(),
                correlation_id: format!("demo/{run_id}"),
                entrypoint: "demo::update".into(),
                phase: ExecutionPhase::Train,
                timestamp: "2026-08-08T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                timing_mode: TimingMode::Host,
                tags: [("physical_batch".into(), "2".into())].into(),
                candle_version: None,
            },
            spans: vec![
                span("session", None, false, measured_ns + 100, None),
                span("update", Some("session"), true, measured_ns, None),
                span(
                    "forward",
                    Some("update"),
                    false,
                    forward_ns,
                    Some(ExecutionStep::Forward),
                ),
                span(
                    "backward",
                    Some("update"),
                    false,
                    measured_ns.saturating_sub(forward_ns + 10),
                    Some(ExecutionStep::Backward),
                ),
                span(
                    "optimizer",
                    Some("update"),
                    false,
                    10,
                    Some(ExecutionStep::Optimizer),
                ),
            ],
            ops: vec![],
            tensors: vec![],
            memory: vec![],
            device_memory: vec![],
            gradients: vec![GradientEvent {
                event_id: "g".into(),
                root: "vb".into(),
                key: "weight".into(),
                state: GradientState::Present,
                norm: Some(forward_ns as f64),
            }],
            edges: vec![],
        }
    }

    fn span(
        id: &str,
        parent: Option<&str>,
        measured: bool,
        duration_ns: u64,
        step: Option<ExecutionStep>,
    ) -> SpanRecord {
        SpanRecord {
            id: id.into(),
            parent_id: parent.map(str::to_string),
            name: id.into(),
            kind: SpanKind::Function,
            measured,
            start_ns: 0,
            closed: true,
            duration_ns,
            step,
        }
    }

    #[test]
    fn compares_measured_region_and_semantic_paths() {
        let comparison =
            compare_documents(&document("base", 100, 60), &document("candidate", 80, 40));
        assert!(comparison.comparable);
        assert_eq!(comparison.total_delta_ns, -20);
        assert_eq!(comparison.total_delta_percent, Some(-20.0));
        assert!(comparison
            .spans
            .iter()
            .any(|span| span.name.contains("session/update/forward")));
        assert_eq!(comparison.gradients.len(), 1);
    }

    #[test]
    fn packet_markdown_exposes_trust_gaps_and_coverage() {
        let packet = EvidencePacket::from_document(
            document("candidate", 80, 40),
            NsightEvidence::unavailable("nsys not installed"),
            None,
        )
        .unwrap();
        let markdown = packet.markdown();
        assert!(markdown.contains("Status: **TRUSTED**"));
        assert!(markdown.contains("nsys not installed"));
        assert!(markdown.contains("optimizer_spans"));
        assert!(!packet.facts.is_empty());
    }
}
