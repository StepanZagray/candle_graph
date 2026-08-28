//! Evidence CLI engine for trace/10, evidence/4, comparison/6, and atomic bundles.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;

use crate::artifact::{
    publish_bundle, verify_bundle, verify_consumed_bundle_files, BundleVerificationReceipt,
};
use crate::campaign::CaptureState;
use crate::comparison::{compare_unverified_traces, compare_verified_bundles, ComparisonVerdict};
use crate::evidence::{build_evidence, EvidencePacket};
use crate::graph::{ExecutionGraph, GraphNode, GraphNodeKind};
use crate::nsight::{GpuEvidenceStatus, ProvenanceBindingState};
use crate::publication::{PublicationReceipt, PublicationStatus, PUBLICATION_SCHEMA};
use crate::trace::{parse_trace, HealthSeverity};

const QUERY_ROW_LIMIT: usize = 50;
const QUERY_LABEL_LIMIT: usize = 100;
const QUERY_DIAGNOSTIC_LIMIT: usize = 50;
const OVERVIEW_LIST_LIMIT: usize = 20;
const OVERVIEW_TIMING_SPAN_LIMIT: usize = 5;
const SUMMARY_SCHEMA: &str = "candle-graph/summary/5";
const QUERY_SCHEMA: &str = "candle-graph/trace-query/5";
const OVERVIEW_SCHEMA: &str = "candle-graph/overview/1";
const VERIFY_SCHEMA: &str = "candle-graph/verify/1";
const PROTOCOL_SCHEMA: &str = "candle-graph/protocol/1";

/// Emitting-tool identity attached to every CLI envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolIdentity {
    pub package: &'static str,
    pub version: &'static str,
}

impl ToolIdentity {
    pub fn current() -> Self {
        Self {
            package: "candle-graph",
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// Label filter for query kinds that carry semantic labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryLabelFilter {
    Exact(String),
    Prefix(String),
}

impl QueryLabelFilter {
    fn matches(&self, value: &str) -> bool {
        match self {
            Self::Exact(label) => value == label,
            Self::Prefix(prefix) => value.starts_with(prefix.as_str()),
        }
    }

    fn envelope(filter: Option<&Self>) -> serde_json::Value {
        serde_json::json!({
            "label": match filter {
                Some(Self::Exact(label)) => Some(label.as_str()),
                _ => None,
            },
            "label_prefix": match filter {
                Some(Self::Prefix(prefix)) => Some(prefix.as_str()),
                _ => None,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QueryAvailability {
    status: GpuEvidenceStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

impl QueryAvailability {
    fn is_available(&self) -> bool {
        self.status == GpuEvidenceStatus::Available
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceQueryKind {
    SlowestHost,
    SlowestDevice,
    Heaviest,
    Memory,
    Spans,
    Tensors,
    TensorStats,
    Gradients,
    Capabilities,
    GpuStatus,
    GpuCorrelation,
    GpuPhases,
    GpuKernels,
    GpuAttributionGaps,
}

impl TraceQueryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SlowestHost => "slowest-host",
            Self::SlowestDevice => "slowest-device",
            Self::Heaviest => "heaviest",
            Self::Memory => "memory",
            Self::Spans => "spans",
            Self::Tensors => "tensors",
            Self::TensorStats => "tensor-stats",
            Self::Gradients => "gradients",
            Self::Capabilities => "capabilities",
            Self::GpuStatus => "gpu-status",
            Self::GpuCorrelation => "gpu-correlation",
            Self::GpuPhases => "gpu-phases",
            Self::GpuKernels => "gpu-kernels",
            Self::GpuAttributionGaps => "gpu-attribution-gaps",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceInputKind {
    RawTrace,
    VerifiedBundle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct EvidenceInput {
    kind: EvidenceInputKind,
    requested_path: PathBuf,
    trace_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_root: Option<PathBuf>,
    evidence_source: &'static str,
    gpu_identity_bound: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    verification: Option<BundleVerificationReceipt>,
}

struct LoadedEvidence {
    packet: EvidencePacket,
    input: EvidenceInput,
}

/// Load a raw trace or the verified evidence packet for a finalized bundle/profile directory.
/// A `trace.jsonl` inside a bundle resolves to that bundle so normalized GPU evidence is retained.
pub fn load_evidence(input: &Path) -> Result<EvidencePacket> {
    Ok(load_evidence_input(input)?.packet)
}

fn load_evidence_input(input: &Path) -> Result<LoadedEvidence> {
    if !input.exists() {
        bail!("evidence input does not exist: {}", input.display());
    }

    if input.is_dir() {
        if input.join("bundle.json").exists() {
            return load_verified_bundle(input, input);
        }
        let trace = input.join("trace.jsonl");
        ensure!(
            trace.is_file(),
            "input directory is neither a finalized bundle nor a raw-trace directory: {}",
            input.display()
        );
        reject_unverified_augmented_parent(input)?;
        return load_raw_trace(input, &trace);
    }

    ensure!(
        input.is_file(),
        "evidence input is not a regular file: {}",
        input.display()
    );
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    if parent.join("bundle.json").exists() {
        return load_verified_bundle(input, parent);
    }
    reject_unverified_augmented_parent(parent)?;
    load_raw_trace(input, input)
}

fn load_raw_trace(requested_path: &Path, trace_path: &Path) -> Result<LoadedEvidence> {
    let packet = build_evidence(trace_path, None)?;
    Ok(LoadedEvidence {
        input: EvidenceInput {
            kind: EvidenceInputKind::RawTrace,
            requested_path: requested_path.to_path_buf(),
            trace_path: trace_path.to_path_buf(),
            bundle_root: None,
            evidence_source: "trace_reconstruction",
            gpu_identity_bound: false,
            verification: None,
        },
        packet,
    })
}

fn load_verified_bundle(requested_path: &Path, root: &Path) -> Result<LoadedEvidence> {
    let verification = verify_bundle(root)
        .with_context(|| format!("verify evidence bundle {}", root.display()))?;
    let trace_path = root.join("trace.jsonl");
    let document = parse_trace(&trace_path)
        .with_context(|| format!("parse verified bundle trace {}", trace_path.display()))?;
    ensure!(
        verification.run_id == document.run.run_id,
        "verified bundle manifest run ID {:?} does not match trace run ID {:?}",
        verification.run_id,
        document.run.run_id
    );

    let evidence_path = root.join("evidence.json");
    let packet: EvidencePacket =
        serde_json::from_slice(&fs::read(&evidence_path).with_context(|| {
            format!("read verified evidence packet {}", evidence_path.display())
        })?)
        .with_context(|| format!("parse verified evidence packet {}", evidence_path.display()))?;
    packet.validate_schema()?;
    ensure!(
        packet.provenance == document.run,
        "verified evidence packet provenance does not match its trace metadata"
    );
    verify_consumed_bundle_files(root, &verification, &["trace.jsonl", "evidence.json"])
        .with_context(|| {
            format!(
                "post-read verify consumed files in evidence bundle {}",
                root.display()
            )
        })?;

    let gpu_identity_bound = packet.gpu.provenance.binding == ProvenanceBindingState::Bound;
    Ok(LoadedEvidence {
        packet,
        input: EvidenceInput {
            kind: EvidenceInputKind::VerifiedBundle,
            requested_path: requested_path.to_path_buf(),
            trace_path,
            bundle_root: Some(root.to_path_buf()),
            evidence_source: "verified_evidence_json",
            gpu_identity_bound,
            verification: Some(verification),
        },
    })
}

fn reject_unverified_augmented_parent(root: &Path) -> Result<()> {
    let evidence = root.join("evidence.json");
    let nsight = root.join("nsight");
    if evidence.exists() || nsight.exists() {
        bail!(
            "input parent {} contains augmented evidence but no bundle.json; refusing to discard unverified GPU evidence and rebuild as trace-only",
            root.display()
        );
    }
    Ok(())
}

fn reject_output_inside_verified_bundle(
    input: &EvidenceInput,
    output: Option<&Path>,
) -> Result<()> {
    let (Some(root), Some(output)) = (input.bundle_root.as_deref(), output) else {
        return Ok(());
    };
    reject_output_inside_bundle_root(root, output)
}

fn reject_output_inside_bundle_root(root: &Path, output: &Path) -> Result<()> {
    let resolved_root = fs::canonicalize(root)
        .with_context(|| format!("resolve verified bundle root {}", root.display()))?;
    let (resolved_output, traversed_bundle) = resolve_write_path(output, &resolved_root)?;
    if traversed_bundle || resolved_output.starts_with(&resolved_root) {
        bail!(
            "refusing to write command output {} inside verified bundle {}",
            output.display(),
            root.display()
        );
    }
    Ok(())
}

/// Bundle root that would be invalidated by writing next to `input`, if `input` is a finalized
/// bundle directory or a file directly inside one.
fn containing_bundle_root(input: &Path) -> Option<&Path> {
    if input.is_dir() && input.join("bundle.json").is_file() {
        return Some(input);
    }
    input
        .parent()
        .filter(|parent| parent.join("bundle.json").is_file())
}

/// Resolve every existing path component (including symbolic links), while retaining a normalized
/// suffix for a not-yet-created output. This mirrors the path that `create_dir_all`/`write` would
/// reach closely enough to reject lexical `..` and pre-existing symlink aliases into a bundle.
fn resolve_write_path(path: &Path, forbidden_root: &Path) -> Result<(PathBuf, bool)> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolve current directory for output path")?
            .join(path)
    };
    let mut resolved = PathBuf::new();
    let mut traversed_forbidden_root = false;
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                match fs::symlink_metadata(&resolved) {
                    Ok(_) => {
                        resolved = fs::canonicalize(&resolved).with_context(|| {
                            format!("resolve output path component {}", resolved.display())
                        })?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("inspect output path component {}", resolved.display())
                        });
                    }
                }
            }
        }
        traversed_forbidden_root |= resolved.starts_with(forbidden_root);
    }
    Ok((resolved, traversed_forbidden_root))
}

pub fn run_import(trace_path: &Path, output: Option<&Path>) -> Result<()> {
    let loaded = load_evidence_input(trace_path)?;
    reject_output_inside_verified_bundle(&loaded.input, output)?;
    super::write_output(
        output,
        (serde_json::to_string_pretty(&loaded.packet)? + "\n").as_bytes(),
    )
}

pub fn run_summary(input_path: &Path, output: Option<&Path>, require_valid: bool) -> Result<()> {
    let loaded = load_evidence_input(input_path)?;
    reject_output_inside_verified_bundle(&loaded.input, output)?;
    let evidence = &loaded.packet;
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": SUMMARY_SCHEMA,
        "tool": ToolIdentity::current(),
        "input": &loaded.input,
        "provenance": &evidence.provenance,
        "health": &evidence.health,
        "capabilities": &evidence.capabilities,
        "findings": &evidence.findings,
        "gaps": &evidence.gaps,
        "summary": evidence.graph.as_ref().map(|graph| &graph.summary),
        "tensor_stats": {
            "events": evidence.tensor_stats.len(),
            "non_finite_events": evidence.tensor_stats.iter().filter(|event| event.non_finite > 0).count(),
        },
        "timing": &evidence.timing,
        "memory": &evidence.memory,
        "gpu": gpu_summary(evidence),
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())?;
    if require_valid {
        let mut failures = Vec::new();
        if !evidence.health.structurally_valid {
            failures.push("structurally_valid is false");
        }
        if !evidence.health.capture_complete {
            failures.push("capture_complete is false");
        }
        if !failures.is_empty() {
            bail!("summary --require-valid: {}", failures.join(", "));
        }
    }
    Ok(())
}

pub fn run_query(
    input_path: &Path,
    kind: TraceQueryKind,
    filter: Option<&QueryLabelFilter>,
    output: Option<&Path>,
) -> Result<()> {
    let loaded = load_evidence_input(input_path)?;
    reject_output_inside_verified_bundle(&loaded.input, output)?;
    let evidence = &loaded.packet;
    let result = match (kind, filter) {
        (TraceQueryKind::TensorStats, Some(filter)) => {
            let matched = evidence
                .tensor_stats
                .iter()
                .filter(|event| filter.matches(&event.label))
                .collect::<Vec<_>>();
            filtered_rows(evidence.tensor_stats.len(), matched)
        }
        (
            TraceQueryKind::Spans | TraceQueryKind::Tensors | TraceQueryKind::Gradients,
            Some(filter),
        ) => {
            let graph = evidence
                .graph
                .as_ref()
                .context("query requires a complete, structurally valid capture")?;
            query_graph_filtered(graph, kind, filter)
        }
        (other, Some(_)) => {
            bail!(
                "query kind {} does not support label filtering",
                other.as_str()
            );
        }
        (TraceQueryKind::Memory, None) => serde_json::to_value(&evidence.memory)?,
        (TraceQueryKind::TensorStats, None) => serde_json::to_value(&evidence.tensor_stats)?,
        (TraceQueryKind::Capabilities, None) => serde_json::to_value(&evidence.capabilities)?,
        (TraceQueryKind::GpuStatus, None) => query_gpu_status(evidence),
        (TraceQueryKind::GpuCorrelation, None) => query_gpu_correlation(evidence),
        (TraceQueryKind::GpuPhases, None) => query_gpu_phases(evidence),
        (TraceQueryKind::GpuKernels, None) => query_gpu_kernels(evidence),
        (TraceQueryKind::GpuAttributionGaps, None) => query_gpu_attribution_gaps(evidence),
        (other, None) => {
            let graph = evidence
                .graph
                .as_ref()
                .context("query requires a complete, structurally valid capture")?;
            query_graph(graph, other)
        }
    };
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": QUERY_SCHEMA,
        "tool": ToolIdentity::current(),
        "kind": kind.as_str(),
        "filter": QueryLabelFilter::envelope(filter),
        "input": &loaded.input,
        "capabilities": &evidence.capabilities,
        "result": result,
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// Complete (untruncated) label-filtered rows, so exact export stays possible.
fn filtered_rows<T: Serialize>(total: usize, matched: Vec<T>) -> serde_json::Value {
    serde_json::json!({
        "total": total,
        "matched": matched.len(),
        "rows": matched,
    })
}

fn query_graph_filtered(
    graph: &ExecutionGraph,
    kind: TraceQueryKind,
    filter: &QueryLabelFilter,
) -> serde_json::Value {
    match kind {
        TraceQueryKind::Spans => {
            let matched = graph
                .spans
                .iter()
                .filter(|node| filter.matches(&node.name))
                .collect::<Vec<_>>();
            let mut result = filtered_rows(graph.spans.len(), matched);
            result["edges"] = serde_json::Value::Null;
            result["edges_reason"] = serde_json::Value::String(
                "edges are omitted for filtered span queries; run without a filter for the full span graph".into(),
            );
            result
        }
        TraceQueryKind::Tensors => {
            let matched = graph
                .tensors
                .iter()
                .filter(|tensor| {
                    filter.matches(tensor.label.as_deref().unwrap_or(&tensor.tensor_id))
                })
                .collect::<Vec<_>>();
            filtered_rows(graph.tensors.len(), matched)
        }
        TraceQueryKind::Gradients => {
            let matched = graph
                .gradients
                .iter()
                .filter(|gradient| {
                    filter.matches(&gradient.key)
                        || filter.matches(&format!("{}/{}", gradient.root, gradient.key))
                })
                .collect::<Vec<_>>();
            filtered_rows(graph.gradients.len(), matched)
        }
        _ => unreachable!("label filtering is dispatched only for spans, tensors, and gradients"),
    }
}

#[cfg(feature = "visualizer")]
pub fn run_view(trace_path: &Path, output: &Path, nsight_dir: Option<&Path>) -> Result<()> {
    if let Some(root) = containing_bundle_root(trace_path) {
        reject_output_inside_bundle_root(root, output)?;
        ensure!(
            nsight_dir.is_none(),
            "bundle inputs already bind their Nsight evidence; drop --nsight-dir for {}",
            trace_path.display()
        );
        let loaded = load_evidence_input(trace_path)?;
        return super::write_output(
            Some(output),
            crate::viewer::render_evidence_html(&loaded.packet).as_bytes(),
        );
    }
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
    require_eligible: bool,
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
        if let Some(output) = output {
            for root in baseline.iter().chain(candidate) {
                reject_output_inside_bundle_root(root, output)?;
            }
        }
        compare_verified_bundles(baseline, candidate)?
    };
    super::write_output(
        output,
        (serde_json::to_string_pretty(&comparison)? + "\n").as_bytes(),
    )?;
    if require_eligible && comparison.verdict == ComparisonVerdict::Ineligible {
        let codes = comparison
            .reasons
            .iter()
            .map(|reason| {
                serde_json::to_value(reason.code)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .expect("comparison reason codes serialize to snake_case strings")
            })
            .collect::<BTreeSet<_>>();
        bail!(
            "comparison ineligible: {}",
            codes.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

pub fn run_report(
    trace: &Path,
    nsight_dir: Option<&Path>,
    bundle: &Path,
    output: Option<&Path>,
) -> Result<()> {
    for ancestor in bundle.ancestors().skip(1) {
        ensure!(
            !ancestor.join("bundle.json").is_file(),
            "refusing to publish bundle {} inside existing bundle {}",
            bundle.display(),
            ancestor.display()
        );
    }
    publish_bundle(bundle, trace, nsight_dir)?;
    let verification = verify_bundle(bundle)
        .with_context(|| format!("deep-verify published bundle {}", bundle.display()))?;
    if let Some(output) = output {
        reject_output_inside_bundle_root(bundle, output)?;
    }
    let receipt = PublicationReceipt {
        schema: PUBLICATION_SCHEMA.into(),
        status: PublicationStatus::Published,
        bundle_path: bundle.to_path_buf(),
        run_id: verification.run_id.clone(),
        verification,
    };
    super::write_output(
        output,
        (serde_json::to_string_pretty(&receipt)? + "\n").as_bytes(),
    )
}

pub fn run_verify(bundle: &Path, semantic: bool, output: Option<&Path>) -> Result<()> {
    let receipt = verify_bundle(bundle)?;
    if let Some(output) = output {
        reject_output_inside_bundle_root(bundle, output)?;
    }
    let semantic_result = if semantic {
        Some(verify_semantic(bundle)?)
    } else {
        None
    };
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": VERIFY_SCHEMA,
        "tool": ToolIdentity::current(),
        "receipt": receipt,
        "semantic": semantic_result,
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// Rederive the evidence packet from the bundle's retained inputs and require
/// an exact match against the published `evidence.json`.
///
/// No path normalization is needed: the only path fields inside an evidence
/// packet (`gpu.raw_report.path` and `gpu.source_csv[].path`) are recorded
/// relative to the Nsight directory root, and publication itself derives
/// `evidence.json` from the staged in-bundle `trace.jsonl` and `nsight/`
/// directory — the same paths this rederivation consumes.
fn verify_semantic(bundle: &Path) -> Result<serde_json::Value> {
    let nsight = bundle.join("nsight");
    let nsight_dir = nsight.is_dir().then_some(nsight.as_path());
    let rederived = build_evidence(&bundle.join("trace.jsonl"), nsight_dir)
        .with_context(|| format!("rederive evidence for bundle {}", bundle.display()))?;
    let rederived = serde_json::to_value(&rederived)?;
    let evidence_path = bundle.join("evidence.json");
    let published: serde_json::Value = serde_json::from_slice(
        &fs::read(&evidence_path)
            .with_context(|| format!("read published evidence {}", evidence_path.display()))?,
    )
    .with_context(|| format!("parse published evidence {}", evidence_path.display()))?;
    if rederived != published {
        bail!(
            "semantic verification failed: rederived evidence differs from published evidence.json for {}",
            bundle.display()
        );
    }
    Ok(serde_json::json!({ "status": "rederived_match" }))
}

/// Bounded first look at a raw trace or verified bundle for agents: health and
/// evidence counts, truncated finding/gap lists, and headline scalars only —
/// never an unbounded collection.
pub fn run_overview(input_path: &Path, output: Option<&Path>) -> Result<()> {
    let loaded = load_evidence_input(input_path)?;
    reject_output_inside_verified_bundle(&loaded.input, output)?;
    let evidence = &loaded.packet;
    let health = &evidence.health;

    let graph_summary = evidence.graph.as_ref().map(|graph| &graph.summary);
    let timing = graph_summary.map(|summary| {
        serde_json::json!({
            "outer_wall_time_ns": summary.outer_wall_time_ns,
            "entrypoint": &summary.entrypoint,
            "slowest_host_spans": bounded_values(&summary.slowest_host_spans, OVERVIEW_TIMING_SPAN_LIMIT),
        })
    });
    let timing_unavailable_reason = graph_summary.is_none().then_some(
        "no derived execution graph: the capture is incomplete or structurally invalid, so no timing headline exists",
    );

    // Only scalar aggregates from the memory profiles; never per-storage lists.
    let logical = evidence.memory.logical.as_ref();
    let memory = serde_json::json!({
        "logical": logical.is_some(),
        "physical": evidence.memory.physical.is_some(),
        "logical_totals": logical.map(|profile| serde_json::json!({
            "storage_allocation_count": profile.storage_allocation_count,
            "matched_storage_free_count": profile.matched_storage_free_count,
            "total_allocated_bytes": profile.total_allocated_bytes,
            "peak_live_bytes": profile.peak.as_ref().map(|peak| peak.live_bytes),
            "peak_timestamp_ns": profile.peak.as_ref().map(|peak| peak.timestamp_ns),
        })),
    });

    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": OVERVIEW_SCHEMA,
        "tool": ToolIdentity::current(),
        "input": &loaded.input,
        "provenance": &evidence.provenance,
        "health": {
            "structurally_valid": health.structurally_valid,
            "capture_complete": health.capture_complete,
            "error_count": health.issues.iter().filter(|issue| issue.severity == HealthSeverity::Error).count(),
            "warning_count": health.issues.iter().filter(|issue| issue.severity == HealthSeverity::Warning).count(),
        },
        "capabilities": &evidence.capabilities,
        "findings": bounded_values(&evidence.findings, OVERVIEW_LIST_LIMIT),
        "gaps": bounded_values(&evidence.gaps, OVERVIEW_LIST_LIMIT),
        "counts": {
            "tensor_stat_events": evidence.tensor_stats.len(),
            "tensor_stat_non_finite_events": evidence.tensor_stats.iter().filter(|event| event.non_finite > 0).count(),
            "graph_spans": evidence.graph.as_ref().map(|graph| graph.spans.len()),
        },
        "timing": timing,
        "timing_unavailable_reason": timing_unavailable_reason,
        "memory": memory,
        "gpu": gpu_summary(evidence),
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

/// Emit the tool's complete versioned protocol: every schema it writes or
/// reads, plus the subcommand catalog, sourced from the crate's exported
/// constants.
pub fn run_protocol(output: Option<&Path>) -> Result<()> {
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "schema": PROTOCOL_SCHEMA,
        "tool": ToolIdentity::current(),
        "schemas": {
            "trace_write": crate::trace::SCHEMA,
            "trace_read": [crate::trace::schema::PREVIOUS_SCHEMA, crate::trace::SCHEMA],
            "graph": crate::graph::SCHEMA,
            "evidence": crate::evidence::SCHEMA,
            "comparison": crate::comparison::SCHEMA,
            "bundle": crate::artifact::SCHEMA,
            "bundle_verification": crate::artifact::VERIFICATION_SCHEMA,
            "publication": PUBLICATION_SCHEMA,
            "campaign": crate::campaign::CAMPAIGN_SCHEMA,
            "campaign_status": crate::campaign::CAMPAIGN_STATUS_SCHEMA,
            "series": crate::campaign::SERIES_SCHEMA,
            "summary": SUMMARY_SCHEMA,
            "query": QUERY_SCHEMA,
            "overview": OVERVIEW_SCHEMA,
            "verify": VERIFY_SCHEMA,
            "gradient_manifest": crate::capability::GRADIENT_MANIFEST_SCHEMA,
            "nsight_capture": crate::nsight::CAPTURE_MANIFEST_SCHEMA,
            "viewer": viewer_schema(),
        },
        "commands": crate::cli::args::command_catalog(),
    }))? + "\n";
    super::write_output(output, rendered.as_bytes())
}

fn viewer_schema() -> serde_json::Value {
    #[cfg(feature = "visualizer")]
    {
        serde_json::Value::String(crate::viewer::trace_view::SCHEMA.into())
    }
    #[cfg(not(feature = "visualizer"))]
    {
        serde_json::Value::Null
    }
}

/// Reconcile a campaign manifest against the bundles on disk.
pub fn run_campaign_status(manifest: &Path, output: Option<&Path>) -> Result<()> {
    let status = crate::campaign::campaign_status(manifest)?;
    super::write_output(
        output,
        (serde_json::to_string_pretty(&status)? + "\n").as_bytes(),
    )
}

/// Build a cross-run series report from a fully published campaign manifest or
/// an explicit ordered list of verified bundle directories.
pub fn run_series(
    manifest: Option<&Path>,
    bundles: &[PathBuf],
    label_prefix: Option<&str>,
    output: Option<&Path>,
) -> Result<()> {
    let roots = match manifest {
        Some(manifest_path) => {
            ensure!(
                bundles.is_empty(),
                "series accepts exactly one of --manifest or --bundle"
            );
            let status = crate::campaign::campaign_status(manifest_path)?;
            let unpublished = status
                .captures
                .iter()
                .filter(|capture| !matches!(capture.state, CaptureState::Published { .. }))
                .map(|capture| {
                    format!(
                        "step {} ({}): {}",
                        capture.capture_step,
                        capture.bundle,
                        capture_state_name(&capture.state)
                    )
                })
                .collect::<Vec<_>>();
            if !unpublished.is_empty() {
                bail!(
                    "series requires every planned capture to be published; not published: {}; run `campaign-status` for the full reconciliation",
                    unpublished.join(", ")
                );
            }
            let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
            status
                .captures
                .iter()
                .map(|capture| base.join(&capture.bundle))
                .collect::<Vec<_>>()
        }
        None => {
            ensure!(
                !bundles.is_empty(),
                "series requires exactly one of --manifest or --bundle"
            );
            bundles.to_vec()
        }
    };
    let report = crate::campaign::build_series(&roots, label_prefix)?;
    super::write_output(
        output,
        (serde_json::to_string_pretty(&report)? + "\n").as_bytes(),
    )
}

fn capture_state_name(state: &CaptureState) -> &'static str {
    match state {
        CaptureState::Missing => "missing",
        CaptureState::Published { .. } => "published",
        CaptureState::FailedRun { .. } => "failed_run",
        CaptureState::VerificationFailed { .. } => "verification_failed",
        CaptureState::IdentityMismatch { .. } => "identity_mismatch",
    }
}

fn query_graph(graph: &ExecutionGraph, kind: TraceQueryKind) -> serde_json::Value {
    match kind {
        TraceQueryKind::SlowestHost => serde_json::json!({
            "entrypoint": graph.summary.entrypoint,
            "outer_wall_time_ns": graph.summary.outer_wall_time_ns,
            "slowest_host_spans": graph.summary.slowest_host_spans,
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
        TraceQueryKind::Memory
        | TraceQueryKind::TensorStats
        | TraceQueryKind::Capabilities
        | TraceQueryKind::GpuStatus
        | TraceQueryKind::GpuCorrelation
        | TraceQueryKind::GpuPhases
        | TraceQueryKind::GpuKernels
        | TraceQueryKind::GpuAttributionGaps => {
            unreachable!("handled without graph")
        }
    }
}

fn gpu_summary(evidence: &EvidencePacket) -> serde_json::Value {
    let correlation = report_availability(
        evidence,
        "nvtx_gpu_proj_trace",
        evidence.gpu.coverage.nvtx_projection,
    );
    let phase_attribution = combined_report_availability(
        evidence,
        &[
            ("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection),
            ("cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline),
        ],
    );
    serde_json::json!({
        "status": evidence.gpu.status,
        "reason": &evidence.gpu.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "correlation": {
            "status": correlation.status,
            "reason": &correlation.reason,
            "complete": correlation.is_available().then_some(evidence.gpu.correlation.complete),
        },
        "coverage": &evidence.gpu.coverage,
        "normalized_rows": gpu_row_counts(evidence),
        "attributed_phases": {
            "status": phase_attribution.status,
            "reason": &phase_attribution.reason,
            "total": phase_attribution.is_available().then_some(evidence.gpu.phase_attribution.len()),
        },
        "diagnostic_count": evidence.gpu.diagnostics.len().saturating_add(evidence.gpu.provenance.diagnostics.len()),
    })
}

fn query_gpu_status(evidence: &EvidencePacket) -> serde_json::Value {
    let diagnostics = bounded_diagnostics(evidence);
    let correlation = report_availability(
        evidence,
        "nvtx_gpu_proj_trace",
        evidence.gpu.coverage.nvtx_projection,
    );
    serde_json::json!({
        "status": evidence.gpu.status,
        "reason": &evidence.gpu.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "capabilities": {
            "gpu_correlation": &evidence.capabilities.gpu_correlation,
            "provenance_binding": &evidence.capabilities.provenance_binding,
        },
        "coverage": &evidence.gpu.coverage,
        "correlation": {
            "status": correlation.status,
            "reason": &correlation.reason,
            "complete": correlation.is_available().then_some(evidence.gpu.correlation.complete),
        },
        "normalized_rows": gpu_row_counts(evidence),
        "source_artifacts": {
            "raw_report": evidence.gpu.raw_report.is_some(),
            "csv_files": evidence.gpu.source_csv.len(),
        },
        "diagnostics": bounded_values(&diagnostics, QUERY_DIAGNOSTIC_LIMIT),
    })
}

fn query_gpu_correlation(evidence: &EvidencePacket) -> serde_json::Value {
    let availability = report_availability(
        evidence,
        "nvtx_gpu_proj_trace",
        evidence.gpu.coverage.nvtx_projection,
    );
    let required_reports = required_report_states(
        evidence,
        &[("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection)],
    );
    if !availability.is_available() {
        return serde_json::json!({
            "status": availability.status,
            "reason": &availability.reason,
            "provenance_binding": evidence.gpu.provenance.binding,
            "required_reports": required_reports,
            "capabilities": {
                "gpu_correlation": &evidence.capabilities.gpu_correlation,
                "provenance_binding": &evidence.capabilities.provenance_binding,
            },
            "mode": null,
            "clock_aligned": null,
            "complete": null,
            "correlation_reason": null,
            "ledger": null,
        });
    }
    let ledger = &evidence.gpu.correlation.ledger;
    let duplicates = ledger
        .duplicates
        .iter()
        .take(QUERY_LABEL_LIMIT)
        .collect::<Vec<_>>();
    serde_json::json!({
        "status": availability.status,
        "reason": &availability.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "required_reports": required_reports,
        "capabilities": {
            "gpu_correlation": &evidence.capabilities.gpu_correlation,
            "provenance_binding": &evidence.capabilities.provenance_binding,
        },
        "mode": &evidence.gpu.correlation.mode,
        "clock_aligned": evidence.gpu.correlation.clock_aligned,
        "complete": evidence.gpu.correlation.complete,
        "correlation_reason": &evidence.gpu.correlation.reason,
        "ledger": {
            "expected": bounded_values(&ledger.expected, QUERY_LABEL_LIMIT),
            "cpu_only": bounded_values(&ledger.cpu_only, QUERY_LABEL_LIMIT),
            "observed": bounded_values(&ledger.observed, QUERY_LABEL_LIMIT),
            "matched": bounded_values(&ledger.matched, QUERY_LABEL_LIMIT),
            "missing_expected": bounded_values(&ledger.missing_expected, QUERY_LABEL_LIMIT),
            "unexpected_observed": bounded_values(&ledger.unexpected_observed, QUERY_LABEL_LIMIT),
            "unexpected_cpu_only": bounded_values(&ledger.unexpected_cpu_only, QUERY_LABEL_LIMIT),
            "duplicates": {
                "total": ledger.duplicates.len(),
                "displayed": duplicates.len(),
                "truncated": duplicates.len() < ledger.duplicates.len(),
                "rows": duplicates,
            },
        },
    })
}

fn query_gpu_phases(evidence: &EvidencePacket) -> serde_json::Value {
    let projected_availability = report_availability(
        evidence,
        "nvtx_gpu_proj_trace",
        evidence.gpu.coverage.nvtx_projection,
    );
    let attributed_availability = combined_report_availability(
        evidence,
        &[
            ("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection),
            ("cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline),
        ],
    );
    let required_reports = required_report_states(
        evidence,
        &[
            ("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection),
            ("cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline),
        ],
    );

    let mut projected = if projected_availability.is_available() {
        evidence.gpu.nvtx_ranges.iter().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    projected.sort_by(|left, right| {
        right
            .projected_duration_ns
            .unwrap_or_default()
            .cmp(&left.projected_duration_ns.unwrap_or_default())
            .then_with(|| left.name.cmp(&right.name))
    });
    let projected_sample_total = projected.len();
    projected.truncate(QUERY_ROW_LIMIT);
    let projected_rows = projected
        .into_iter()
        .map(|row| {
            let join_keys = [
                row.correlation_id.as_ref().map(|_| "correlation_id"),
                row.device.as_ref().map(|_| "device"),
                row.context.as_ref().map(|_| "context"),
                row.stream.as_ref().map(|_| "stream"),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            serde_json::json!({
                "name": &row.name,
                "semantic_key": &row.semantic_key,
                "projected_start_ns": row.projected_start_ns,
                "projected_duration_ns": row.projected_duration_ns,
                "declared_gpu_operations": row.gpu_operations,
                "join_keys": join_keys,
            })
        })
        .collect::<Vec<_>>();
    let projected_population_total = projected_availability
        .is_available()
        .then(|| total_report_rows(evidence, "nvtx_gpu_proj_trace", projected_sample_total));

    let mut attributed = if attributed_availability.is_available() {
        evidence.gpu.phase_attribution.iter().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    attributed.sort_by(|left, right| {
        right
            .gpu_busy_ns
            .cmp(&left.gpu_busy_ns)
            .then_with(|| left.semantic_key.cmp(&right.semantic_key))
    });
    let attributed_total = attributed_availability
        .is_available()
        .then_some(attributed.len());
    attributed.truncate(QUERY_ROW_LIMIT);

    serde_json::json!({
        "status": attributed_availability.status,
        "reason": &attributed_availability.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "required_reports": required_reports,
        "clock_plane": "nsight_projected_not_host_aligned",
        "projected_ranges": {
            "status": projected_availability.status,
            "reason": &projected_availability.reason,
            "population_total": projected_population_total,
            "retained_sample_total": projected_availability.is_available().then_some(projected_sample_total),
            "displayed": projected_rows.len(),
            "population_truncated_before_ranking": projected_population_total.map(|total| projected_sample_total < total),
            "display_truncated": projected_availability.is_available().then_some(projected_rows.len() < projected_sample_total),
            "sample_selection": "earliest_original_start_rows",
            "ordering": "projected_duration_ns_desc_within_retained_sample",
            "global_duration_ranking": false,
            "rows": projected_rows,
        },
        "attributed_phases": {
            "status": attributed_availability.status,
            "reason": &attributed_availability.reason,
            "population_total": attributed_total,
            "displayed": attributed.len(),
            "display_truncated": attributed_total.map(|total| attributed.len() < total),
            "ordering": "gpu_busy_ns_desc_across_normalized_population",
            "rows": attributed,
        },
    })
}

fn query_gpu_kernels(evidence: &EvidencePacket) -> serde_json::Value {
    let availability = report_availability(
        evidence,
        "cuda_gpu_kern_sum",
        evidence.gpu.coverage.kernel_summary,
    );
    let mut kernels = if availability.is_available() {
        evidence.gpu.kernels.iter().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    kernels.sort_by(|left, right| {
        right
            .total_ns
            .cmp(&left.total_ns)
            .then_with(|| left.name.cmp(&right.name))
    });
    let normalized_display_total = kernels.len();
    kernels.truncate(QUERY_ROW_LIMIT);
    let total = availability
        .is_available()
        .then(|| total_report_rows(evidence, "cuda_gpu_kern_sum", normalized_display_total));
    serde_json::json!({
        "status": availability.status,
        "reason": &availability.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "required_reports": required_report_states(
            evidence,
            &[("cuda_gpu_kern_sum", evidence.gpu.coverage.kernel_summary)],
        ),
        "clock_plane": "nsight_gpu",
        "population_total": total,
        "displayed": kernels.len(),
        "display_truncated": total.map(|total| kernels.len() < total),
        "rows": kernels,
    })
}

fn query_gpu_attribution_gaps(evidence: &EvidencePacket) -> serde_json::Value {
    let availability = combined_report_availability(
        evidence,
        &[
            ("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection),
            ("cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline),
        ],
    );
    let required_reports = required_report_states(
        evidence,
        &[
            ("nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection),
            ("cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline),
        ],
    );
    if !availability.is_available() {
        return serde_json::json!({
            "status": availability.status,
            "reason": &availability.reason,
            "provenance_binding": evidence.gpu.provenance.binding,
            "required_reports": required_reports,
            "correlation_complete": null,
            "correlation_reason": null,
            "missing_expected": null,
            "unexpected_observed": null,
            "unexpected_cpu_only": null,
            "duplicate_labels": null,
            "matched_without_exact_gpu_busy_attribution": null,
            "attributed_unexpected": null,
            "projected_range_rows": null,
            "exactly_attributed_phase_rows": null,
            "truncated_reports": null,
            "diagnostics": bounded_values(&bounded_diagnostics(evidence), QUERY_DIAGNOSTIC_LIMIT),
        });
    }
    let ledger = &evidence.gpu.correlation.ledger;
    let attributed = evidence
        .gpu
        .phase_attribution
        .iter()
        .map(|phase| phase.semantic_key.as_str())
        .collect::<BTreeSet<_>>();
    let expected = ledger
        .expected
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let matched_without_attribution = ledger
        .matched
        .iter()
        .filter(|label| !attributed.contains(label.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let attributed_unexpected = attributed
        .difference(&expected)
        .map(|label| (*label).to_string())
        .collect::<Vec<_>>();
    let truncated_reports = evidence
        .gpu
        .limits
        .iter()
        .filter(|(_, limit)| limit.truncated)
        .map(|(report, limit)| {
            serde_json::json!({
                "report": report,
                "total_rows": limit.total_rows,
                "displayed_rows": limit.displayed_rows,
            })
        })
        .take(QUERY_ROW_LIMIT)
        .collect::<Vec<_>>();
    let diagnostics = bounded_diagnostics(evidence);

    serde_json::json!({
        "status": availability.status,
        "reason": &availability.reason,
        "provenance_binding": evidence.gpu.provenance.binding,
        "required_reports": required_reports,
        "correlation_complete": evidence.gpu.correlation.complete,
        "correlation_reason": &evidence.gpu.correlation.reason,
        "missing_expected": bounded_values(&ledger.missing_expected, QUERY_LABEL_LIMIT),
        "unexpected_observed": bounded_values(&ledger.unexpected_observed, QUERY_LABEL_LIMIT),
        "unexpected_cpu_only": bounded_values(&ledger.unexpected_cpu_only, QUERY_LABEL_LIMIT),
        "duplicate_labels": bounded_values(&ledger.duplicates, QUERY_LABEL_LIMIT),
        "matched_without_exact_gpu_busy_attribution": bounded_values(&matched_without_attribution, QUERY_LABEL_LIMIT),
        "attributed_unexpected": bounded_values(&attributed_unexpected, QUERY_LABEL_LIMIT),
        "projected_range_rows": total_report_rows(evidence, "nvtx_gpu_proj_trace", evidence.gpu.nvtx_ranges.len()),
        "exactly_attributed_phase_rows": evidence.gpu.phase_attribution.len(),
        "truncated_reports": {
            "total": evidence.gpu.limits.values().filter(|limit| limit.truncated).count(),
            "displayed": truncated_reports.len(),
            "truncated": truncated_reports.len() < evidence.gpu.limits.values().filter(|limit| limit.truncated).count(),
            "rows": truncated_reports,
        },
        "diagnostics": bounded_values(&diagnostics, QUERY_DIAGNOSTIC_LIMIT),
    })
}

fn gpu_row_counts(evidence: &EvidencePacket) -> serde_json::Value {
    serde_json::json!({
        "kernels": report_row_count(evidence, "cuda_gpu_kern_sum", evidence.gpu.coverage.kernel_summary, evidence.gpu.kernels.len()),
        "runtime_calls": report_row_count(evidence, "cuda_api_sum", evidence.gpu.coverage.runtime_summary, evidence.gpu.runtime_calls.len()),
        "memory_operations": report_row_count(evidence, "cuda_gpu_mem_time_sum", evidence.gpu.coverage.memory_summary, evidence.gpu.memory_operations.len()),
        "projected_ranges": report_row_count(evidence, "nvtx_gpu_proj_trace", evidence.gpu.coverage.nvtx_projection, evidence.gpu.nvtx_ranges.len()),
        "gpu_timeline": report_row_count(evidence, "cuda_gpu_trace", evidence.gpu.coverage.gpu_timeline, evidence.gpu.gpu_timeline.len()),
    })
}

fn report_row_count(
    evidence: &EvidencePacket,
    report_kind: &str,
    covered: bool,
    fallback: usize,
) -> serde_json::Value {
    let availability = report_availability(evidence, report_kind, covered);
    serde_json::json!({
        "status": availability.status,
        "reason": &availability.reason,
        "total": availability.is_available().then(|| total_report_rows(evidence, report_kind, fallback)),
    })
}

fn report_availability(
    evidence: &EvidencePacket,
    report_kind: &str,
    covered: bool,
) -> QueryAvailability {
    let parse_failed = evidence
        .gpu
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.contains(report_kind));
    if covered && !parse_failed {
        return QueryAvailability {
            status: GpuEvidenceStatus::Available,
            reason: None,
        };
    }
    let failed = parse_failed || evidence.gpu.status == GpuEvidenceStatus::Failed;
    QueryAvailability {
        status: if failed {
            GpuEvidenceStatus::Failed
        } else {
            GpuEvidenceStatus::Unavailable
        },
        reason: Some(if failed {
            format!(
                "Nsight report `{report_kind}` failed to normalize; its row population is unknown, not zero"
            )
        } else {
            format!(
                "Nsight report `{report_kind}` was not normalized; its row population is unknown, not zero"
            )
        }),
    }
}

fn combined_report_availability(
    evidence: &EvidencePacket,
    reports: &[(&str, bool)],
) -> QueryAvailability {
    let unavailable = reports
        .iter()
        .filter_map(|(report, covered)| {
            let availability = report_availability(evidence, report, *covered);
            (!availability.is_available()).then_some((*report, availability.status))
        })
        .collect::<Vec<_>>();
    if unavailable.is_empty() {
        return QueryAvailability {
            status: GpuEvidenceStatus::Available,
            reason: None,
        };
    }
    let failed = unavailable
        .iter()
        .any(|(_, status)| *status == GpuEvidenceStatus::Failed);
    let names = unavailable
        .iter()
        .map(|(report, _)| *report)
        .collect::<Vec<_>>();
    QueryAvailability {
        status: if failed {
            GpuEvidenceStatus::Failed
        } else {
            GpuEvidenceStatus::Unavailable
        },
        reason: Some(format!(
            "required normalized Nsight report(s) unavailable: {}; the derived result is unknown, not zero",
            names.join(", ")
        )),
    }
}

fn required_report_states(
    evidence: &EvidencePacket,
    reports: &[(&str, bool)],
) -> serde_json::Value {
    let states = reports
        .iter()
        .map(|(report, covered)| {
            (
                (*report).to_string(),
                serde_json::to_value(report_availability(evidence, report, *covered))
                    .expect("query availability is serializable"),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    serde_json::to_value(states).expect("required report states are serializable")
}

fn total_report_rows(evidence: &EvidencePacket, report_kind: &str, fallback: usize) -> usize {
    let total = evidence
        .gpu
        .limits
        .iter()
        .filter(|(name, _)| name.contains(report_kind))
        .fold(0_usize, |sum, (_, limit)| {
            sum.saturating_add(limit.total_rows)
        });
    total.max(fallback)
}

fn bounded_diagnostics(evidence: &EvidencePacket) -> Vec<String> {
    let mut diagnostics = evidence
        .gpu
        .provenance
        .diagnostics
        .iter()
        .chain(&evidence.gpu.diagnostics)
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.sort();
    diagnostics.dedup();
    diagnostics
}

fn bounded_values<T: Serialize>(values: &[T], limit: usize) -> serde_json::Value {
    let displayed = values.len().min(limit);
    serde_json::json!({
        "total": values.len(),
        "displayed": displayed,
        "truncated": displayed < values.len(),
        "rows": &values[..displayed],
    })
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
