//! Fail-closed replicated performance comparisons.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifact::{verify_bundle, verify_consumed_bundle_files};
use crate::capability::MeasurementScope;
use crate::trace::{analyze_health, parse_trace, ComparisonIdentity, TraceDocument};

pub const SCHEMA: &str = "candle-graph/comparison/6";
pub const MINIMUM_RUNS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonVerdict {
    Ineligible,
    Inconclusive,
    CandidateFaster,
    CandidateSlower,
}

/// Stable, machine-readable cause of comparison ineligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonReasonCode {
    UnverifiedInputs,
    ReceiptCountMismatch,
    ReceiptRunIdMismatch,
    ReceiptDigestInvalid,
    InsufficientRuns,
    DuplicateRunIds,
    NoRuns,
    CaptureSemanticsMismatch,
    IncompleteCapture,
    NotProductionEquivalent,
    UnsynchronizedDeviceRegion,
    MeasuredRegionCountInvalid,
    IdentityMissing,
    IdentityInvalid,
    IdentityConditionsDiffer,
    ImplementationIdMissing,
    ImplementationIdEmpty,
    ImplementationIdInconsistent,
    PairingIncomplete,
    PairIdsDuplicated,
    PairSetsMismatch,
}

/// One ineligibility cause: a stable code plus its human-readable message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReason {
    pub code: ComparisonReasonCode,
    pub message: String,
}

fn reason(code: ComparisonReasonCode, message: impl Into<String>) -> ComparisonReason {
    ComparisonReason {
        code,
        message: message.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleStatistics {
    pub samples_ns: Vec<u64>,
    pub median_ns: f64,
    pub p95_ns: f64,
    pub mad_ns: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub level: f64,
    pub lower_delta_ns: f64,
    pub upper_delta_ns: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStatsComparisonRow {
    pub label: String,
    pub rms_a: f64,
    pub rms_b: f64,
    pub rms_ratio: Option<f64>,
    pub abs_max_ratio: Option<f64>,
    pub non_finite_a: u64,
    pub non_finite_b: u64,
    /// Events averaged into `rms_a`/`abs_max` for this label (duplicates included).
    pub samples_a: usize,
    pub samples_b: usize,
    /// Cohort runs that contained this label; compare against the cohort run totals to see
    /// whether an average covers the whole cohort or only part of it.
    pub runs_a: usize,
    pub runs_b: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TensorStatsComparison {
    pub baseline_runs: usize,
    pub candidate_runs: usize,
    pub matched: Vec<TensorStatsComparisonRow>,
    pub unmatched_a: Vec<String>,
    pub unmatched_b: Vec<String>,
}

/// Trust state of the artifacts supplied to a comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonInputVerification {
    VerifiedBundles,
    UnverifiedTraces,
}

/// One content-addressed bundle input verified immediately before comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedBundleInput {
    pub run_id: String,
    pub manifest_sha256: String,
}

/// Cohort provenance that determines whether a comparison may be eligible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonInputs {
    pub verification: ComparisonInputVerification,
    pub baseline: Vec<VerifiedBundleInput>,
    pub candidate: Vec<VerifiedBundleInput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicatedComparison {
    pub schema: String,
    pub metric: String,
    pub inputs: ComparisonInputs,
    pub comparable: bool,
    pub paired: bool,
    pub verdict: ComparisonVerdict,
    pub reasons: Vec<ComparisonReason>,
    pub baseline_implementation_id: Option<String>,
    pub candidate_implementation_id: Option<String>,
    pub identity: Option<ComparisonIdentity>,
    pub baseline: SampleStatistics,
    pub candidate: SampleStatistics,
    pub median_delta_ns: f64,
    pub median_delta_percent: Option<f64>,
    pub confidence_interval: Option<ConfidenceInterval>,
    /// Numerical mechanism comparison, independent of timing eligibility.
    #[serde(default)]
    pub tensor_stats: TensorStatsComparison,
}

/// Verify finalized evidence bundles and compare their bound trace documents.
pub fn compare_verified_bundles<B: AsRef<Path>, C: AsRef<Path>>(
    baseline: &[B],
    candidate: &[C],
) -> Result<ReplicatedComparison> {
    let (baseline_documents, baseline_inputs) = load_verified_cohort(baseline, "baseline")?;
    let (candidate_documents, candidate_inputs) = load_verified_cohort(candidate, "candidate")?;
    Ok(compare_documents(
        &baseline_documents,
        &candidate_documents,
        ComparisonInputs {
            verification: ComparisonInputVerification::VerifiedBundles,
            baseline: baseline_inputs,
            candidate: candidate_inputs,
        },
    ))
}

/// Compare raw trace documents for diagnostics only. This path is always ineligible.
pub fn compare_unverified_traces(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
) -> ReplicatedComparison {
    compare_documents(
        baseline,
        candidate,
        ComparisonInputs {
            verification: ComparisonInputVerification::UnverifiedTraces,
            baseline: Vec::new(),
            candidate: Vec::new(),
        },
    )
}

fn load_verified_cohort<P: AsRef<Path>>(
    roots: &[P],
    cohort: &str,
) -> Result<(Vec<TraceDocument>, Vec<VerifiedBundleInput>)> {
    let mut documents = Vec::with_capacity(roots.len());
    let mut inputs = Vec::with_capacity(roots.len());
    for (index, root) in roots.iter().enumerate() {
        let root = root.as_ref();
        let receipt = verify_bundle(root).with_context(|| {
            format!("verify {cohort} bundle {} at {}", index + 1, root.display())
        })?;
        let document = parse_trace(root.join("trace.jsonl")).with_context(|| {
            format!(
                "parse verified {cohort} bundle {} trace at {}",
                index + 1,
                root.display()
            )
        })?;
        ensure!(
            document.run.run_id == receipt.run_id,
            "verified {cohort} bundle {} manifest run ID {:?} does not match trace run ID {:?}",
            index + 1,
            receipt.run_id,
            document.run.run_id
        );
        verify_consumed_bundle_files(root, &receipt, &["trace.jsonl"]).with_context(|| {
            format!(
                "post-read verify {cohort} bundle {} trace at {}",
                index + 1,
                root.display()
            )
        })?;
        inputs.push(VerifiedBundleInput {
            run_id: receipt.run_id,
            manifest_sha256: receipt.manifest_sha256,
        });
        documents.push(document);
    }
    Ok((documents, inputs))
}

fn compare_documents(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    inputs: ComparisonInputs,
) -> ReplicatedComparison {
    let mut reasons = Vec::new();
    validate_input_provenance(&inputs, baseline, candidate, &mut reasons);
    let baseline_samples = measured_samples(baseline, "baseline", &mut reasons);
    let candidate_samples = measured_samples(candidate, "candidate", &mut reasons);
    if baseline.len() < MINIMUM_RUNS || candidate.len() < MINIMUM_RUNS {
        reasons.push(reason(
            ComparisonReasonCode::InsufficientRuns,
            format!("at least {MINIMUM_RUNS} independent baseline and candidate runs are required"),
        ));
    }
    require_independent_run_ids(baseline, candidate, &mut reasons);
    require_consistent_capture_semantics(baseline, candidate, &mut reasons);

    let identity = common_identity(baseline, candidate, &mut reasons);
    let baseline_implementation_id = cohort_implementation_id(baseline, "baseline", &mut reasons);
    let candidate_implementation_id =
        cohort_implementation_id(candidate, "candidate", &mut reasons);
    let paired_samples = pair_samples(baseline, candidate, &mut reasons);
    let paired = paired_samples.is_some();
    let comparable = reasons.is_empty();
    let baseline_stats = statistics(baseline_samples);
    let candidate_stats = statistics(candidate_samples);
    let median_delta_ns = paired_samples.as_ref().map_or_else(
        || candidate_stats.median_ns - baseline_stats.median_ns,
        |pairs| {
            let mut deltas = pairs
                .iter()
                .map(|(baseline, candidate)| *candidate as i128 - *baseline as i128)
                .collect::<Vec<_>>();
            deltas.sort_unstable();
            median_i128(&deltas)
        },
    );
    let median_delta_percent = (baseline_stats.median_ns != 0.0)
        .then_some(median_delta_ns / baseline_stats.median_ns * 100.0);
    let confidence_interval = comparable.then(|| {
        let (lower_delta_ns, upper_delta_ns) = bootstrap_delta_ci(
            &baseline_stats.samples_ns,
            &candidate_stats.samples_ns,
            paired_samples.as_deref(),
        );
        ConfidenceInterval {
            level: 0.95,
            lower_delta_ns,
            upper_delta_ns,
        }
    });
    let verdict = match &confidence_interval {
        None => ComparisonVerdict::Ineligible,
        Some(ci) if ci.upper_delta_ns < 0.0 => ComparisonVerdict::CandidateFaster,
        Some(ci) if ci.lower_delta_ns > 0.0 => ComparisonVerdict::CandidateSlower,
        Some(_) => ComparisonVerdict::Inconclusive,
    };
    let tensor_stats = compare_tensor_stats(baseline, candidate);

    ReplicatedComparison {
        schema: SCHEMA.into(),
        metric: "outer_wall_time_ns".into(),
        inputs,
        comparable,
        paired,
        verdict,
        reasons,
        baseline_implementation_id,
        candidate_implementation_id,
        identity,
        baseline: baseline_stats,
        candidate: candidate_stats,
        median_delta_ns,
        median_delta_percent,
        confidence_interval,
        tensor_stats,
    }
}

fn compare_tensor_stats(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
) -> TensorStatsComparison {
    #[derive(Default)]
    struct Aggregate {
        rms: f64,
        abs_max: f64,
        non_finite: u64,
        samples: usize,
        runs: usize,
    }

    fn aggregate(documents: &[TraceDocument]) -> BTreeMap<String, Aggregate> {
        let mut by_label = BTreeMap::<String, Aggregate>::new();
        for document in documents {
            let mut seen = BTreeSet::new();
            for event in &document.tensor_stats {
                let entry = by_label.entry(event.label.clone()).or_default();
                entry.rms += event.rms;
                entry.abs_max += event.abs_max;
                entry.non_finite = entry.non_finite.saturating_add(event.non_finite);
                entry.samples += 1;
                if seen.insert(event.label.as_str()) {
                    entry.runs += 1;
                }
            }
        }
        by_label
    }

    fn mean(value: f64, samples: usize) -> f64 {
        if samples == 0 {
            0.0
        } else {
            value / samples as f64
        }
    }

    fn ratio(a: f64, b: f64) -> Option<f64> {
        if a == 0.0 {
            (b == 0.0).then_some(1.0)
        } else {
            Some(b / a)
        }
    }

    fn ratio_distance(ratio: Option<f64>) -> f64 {
        match ratio {
            Some(value) if value > 0.0 => value.ln().abs(),
            _ => f64::INFINITY,
        }
    }

    let baseline_runs = baseline.len();
    let candidate_runs = candidate.len();
    let baseline = aggregate(baseline);
    let candidate = aggregate(candidate);
    let mut matched = baseline
        .iter()
        .filter_map(|(label, a)| {
            let b = candidate.get(label)?;
            let rms_a = mean(a.rms, a.samples);
            let rms_b = mean(b.rms, b.samples);
            let abs_max_a = mean(a.abs_max, a.samples);
            let abs_max_b = mean(b.abs_max, b.samples);
            Some(TensorStatsComparisonRow {
                label: label.clone(),
                rms_a,
                rms_b,
                rms_ratio: ratio(rms_a, rms_b),
                abs_max_ratio: ratio(abs_max_a, abs_max_b),
                non_finite_a: a.non_finite,
                non_finite_b: b.non_finite,
                samples_a: a.samples,
                samples_b: b.samples,
                runs_a: a.runs,
                runs_b: b.runs,
            })
        })
        .collect::<Vec<_>>();
    matched.sort_by(|a, b| {
        ratio_distance(b.rms_ratio)
            .total_cmp(&ratio_distance(a.rms_ratio))
            .then_with(|| a.label.cmp(&b.label))
    });
    TensorStatsComparison {
        baseline_runs,
        candidate_runs,
        matched,
        unmatched_a: baseline
            .keys()
            .filter(|label| !candidate.contains_key(*label))
            .cloned()
            .collect(),
        unmatched_b: candidate
            .keys()
            .filter(|label| !baseline.contains_key(*label))
            .cloned()
            .collect(),
    }
}

fn validate_input_provenance(
    inputs: &ComparisonInputs,
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<ComparisonReason>,
) {
    match inputs.verification {
        ComparisonInputVerification::UnverifiedTraces => reasons.push(reason(
            ComparisonReasonCode::UnverifiedInputs,
            "unverified raw trace inputs are diagnostic only; finalized verified bundles are required for an eligible comparison",
        )),
        ComparisonInputVerification::VerifiedBundles => {
            validate_verified_cohort(&inputs.baseline, baseline, "baseline", reasons);
            validate_verified_cohort(&inputs.candidate, candidate, "candidate", reasons);
        }
    }
}

fn validate_verified_cohort(
    inputs: &[VerifiedBundleInput],
    documents: &[TraceDocument],
    cohort: &str,
    reasons: &mut Vec<ComparisonReason>,
) {
    if inputs.len() != documents.len() {
        reasons.push(reason(
            ComparisonReasonCode::ReceiptCountMismatch,
            format!("{cohort} bundle receipts must correspond one-to-one with trace documents"),
        ));
        return;
    }
    for (index, (input, document)) in inputs.iter().zip(documents).enumerate() {
        if input.run_id != document.run.run_id {
            reasons.push(reason(
                ComparisonReasonCode::ReceiptRunIdMismatch,
                format!(
                    "{cohort} bundle {} receipt run ID does not match its trace",
                    index + 1
                ),
            ));
        }
        if input.manifest_sha256.len() != 64
            || !input
                .manifest_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            reasons.push(reason(
                ComparisonReasonCode::ReceiptDigestInvalid,
                format!(
                    "{cohort} bundle {} receipt has an invalid manifest SHA-256",
                    index + 1
                ),
            ));
        }
    }
}

fn require_independent_run_ids(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<ComparisonReason>,
) {
    let ids = baseline
        .iter()
        .chain(candidate)
        .map(|document| document.run.run_id.as_str())
        .collect::<Vec<_>>();
    if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
        reasons.push(reason(
            ComparisonReasonCode::DuplicateRunIds,
            "run IDs must be unique across all replicates",
        ));
    }
}

fn require_consistent_capture_semantics(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<ComparisonReason>,
) {
    let Some(first) = baseline.first().or_else(|| candidate.first()) else {
        reasons.push(reason(
            ComparisonReasonCode::NoRuns,
            "comparison contains no runs",
        ));
        return;
    };
    if baseline.iter().chain(candidate).any(|document| {
        document.run.entrypoint != first.run.entrypoint
            || document.run.phase != first.run.phase
            || document.run.device != first.run.device
            || document.run.timing_mode != first.run.timing_mode
            || document.run.warmup_steps != first.run.warmup_steps
            || document.run.capture_step != first.run.capture_step
            || document.run.capture_contract != first.run.capture_contract
            || document.run.measured_region_device_synchronized
                != first.run.measured_region_device_synchronized
    }) {
        reasons.push(reason(
            ComparisonReasonCode::CaptureSemanticsMismatch,
            "entrypoint, phase, device, timing mode, synchronization, warmup, capture step, and capture contract must match",
        ));
    }
}

fn measured_samples(
    docs: &[TraceDocument],
    cohort: &str,
    reasons: &mut Vec<ComparisonReason>,
) -> Vec<u64> {
    docs.iter()
        .enumerate()
        .map(|(index, doc)| {
            let health = analyze_health(doc);
            if !health.structurally_valid || !health.capture_complete {
                reasons.push(reason(
                    ComparisonReasonCode::IncompleteCapture,
                    format!(
                        "{cohort} run {} is not a complete, structurally valid capture",
                        index + 1
                    ),
                ));
            }
            if doc.run.capture_contract.measurement_scope != MeasurementScope::ProductionEquivalent
            {
                reasons.push(reason(
                    ComparisonReasonCode::NotProductionEquivalent,
                    format!(
                        "{cohort} run {} is not declared production-equivalent",
                        index + 1
                    ),
                ));
            }
            if !doc.run.device.starts_with("cpu") && !doc.run.measured_region_device_synchronized {
                reasons.push(reason(
                    ComparisonReasonCode::UnsynchronizedDeviceRegion,
                    format!(
                        "{cohort} run {} does not synchronize its measured device region",
                        index + 1
                    ),
                ));
            }
            let values = doc
                .spans
                .iter()
                .filter(|span| span.measured && span.closed)
                .map(|span| span.duration_ns)
                .collect::<Vec<_>>();
            if values.len() != 1 {
                reasons.push(reason(
                    ComparisonReasonCode::MeasuredRegionCountInvalid,
                    format!(
                        "{cohort} run {} does not contain exactly one closed measured region",
                        index + 1
                    ),
                ));
            }
            values.into_iter().next().unwrap_or(0)
        })
        .collect()
}

fn common_identity(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<ComparisonReason>,
) -> Option<ComparisonIdentity> {
    let identities = baseline
        .iter()
        .chain(candidate)
        .map(|doc| doc.run.comparison_identity.as_ref())
        .collect::<Vec<_>>();
    let Some(first) = identities.first().copied().flatten() else {
        reasons.push(reason(
            ComparisonReasonCode::IdentityMissing,
            "comparison identity is missing",
        ));
        return None;
    };
    if identities.iter().any(|identity| identity.is_none()) {
        reasons.push(reason(
            ComparisonReasonCode::IdentityMissing,
            "comparison identity is missing from one or more runs",
        ));
        return None;
    }
    if let Err(error) = first.validate() {
        reasons.push(reason(
            ComparisonReasonCode::IdentityInvalid,
            format!("comparison identity is invalid: {error}"),
        ));
        return None;
    }
    if identities
        .iter()
        .flatten()
        .any(|identity| !same_conditions(first, identity))
    {
        reasons.push(reason(
            ComparisonReasonCode::IdentityConditionsDiffer,
            "workload, model, configuration, data, seed, batch, precision, or device state differs",
        ));
        return None;
    }
    let mut result = first.clone();
    result.implementation_id = None;
    result.pair_id = None;
    Some(result)
}

fn cohort_implementation_id(
    documents: &[TraceDocument],
    cohort: &str,
    reasons: &mut Vec<ComparisonReason>,
) -> Option<String> {
    let implementation_ids = documents
        .iter()
        .map(|document| {
            document
                .run
                .comparison_identity
                .as_ref()
                .and_then(|identity| identity.implementation_id.as_deref())
        })
        .collect::<Vec<_>>();
    let Some(first) = implementation_ids.first().copied().flatten() else {
        reasons.push(reason(
            ComparisonReasonCode::ImplementationIdMissing,
            format!("{cohort} implementation ID is missing"),
        ));
        return None;
    };
    if implementation_ids.iter().any(|identity| identity.is_none()) {
        reasons.push(reason(
            ComparisonReasonCode::ImplementationIdMissing,
            format!("{cohort} implementation ID is missing from one or more runs"),
        ));
        return None;
    }
    if implementation_ids
        .iter()
        .flatten()
        .any(|identity| identity.trim().is_empty())
    {
        reasons.push(reason(
            ComparisonReasonCode::ImplementationIdEmpty,
            format!("{cohort} implementation ID must not be empty"),
        ));
        return None;
    }
    if implementation_ids
        .iter()
        .flatten()
        .any(|identity| *identity != first)
    {
        reasons.push(reason(
            ComparisonReasonCode::ImplementationIdInconsistent,
            format!("{cohort} implementation ID differs within the cohort"),
        ));
        return None;
    }
    Some(first.to_owned())
}

fn same_conditions(left: &ComparisonIdentity, right: &ComparisonIdentity) -> bool {
    left.workload_id == right.workload_id
        && left.model_id == right.model_id
        && left.config_id == right.config_id
        && left.data_id == right.data_id
        && left.seed_policy == right.seed_policy
        && left.physical_batch == right.physical_batch
        && left.accumulation_steps == right.accumulation_steps
        && left.precision == right.precision
        && left.device_state == right.device_state
}

fn pair_samples(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<ComparisonReason>,
) -> Option<Vec<(u64, u64)>> {
    let any_pair_id = baseline.iter().chain(candidate).any(|doc| {
        doc.run
            .comparison_identity
            .as_ref()
            .and_then(|identity| identity.pair_id.as_ref())
            .is_some()
    });
    if !any_pair_id {
        return None;
    }
    let collect = |docs: &[TraceDocument]| -> Result<BTreeMap<String, u64>, ComparisonReason> {
        let mut values = BTreeMap::new();
        for doc in docs {
            let pair = doc
                .run
                .comparison_identity
                .as_ref()
                .and_then(|identity| identity.pair_id.clone())
                .ok_or_else(|| {
                    reason(
                        ComparisonReasonCode::PairingIncomplete,
                        "pair IDs must be present on every run when pairing is requested",
                    )
                })?;
            let value = doc
                .spans
                .iter()
                .find(|span| span.measured && span.closed)
                .ok_or_else(|| {
                    reason(
                        ComparisonReasonCode::PairingIncomplete,
                        "paired runs require one closed measured region",
                    )
                })?
                .duration_ns;
            if values.insert(pair, value).is_some() {
                return Err(reason(
                    ComparisonReasonCode::PairIdsDuplicated,
                    "pair IDs must be unique within each cohort",
                ));
            }
        }
        Ok(values)
    };
    let left = match collect(baseline) {
        Ok(values) => values,
        Err(cause) => {
            reasons.push(cause);
            return None;
        }
    };
    let right = match collect(candidate) {
        Ok(values) => values,
        Err(cause) => {
            reasons.push(cause);
            return None;
        }
    };
    if left.keys().collect::<BTreeSet<_>>() != right.keys().collect::<BTreeSet<_>>() {
        reasons.push(reason(
            ComparisonReasonCode::PairSetsMismatch,
            "baseline and candidate pair-ID sets must match exactly",
        ));
        return None;
    }
    Some(
        left.into_iter()
            .map(|(key, value)| (value, right[&key]))
            .collect(),
    )
}

fn statistics(samples_ns: Vec<u64>) -> SampleStatistics {
    let median_ns = percentile(&samples_ns, 0.5);
    let p95_ns = percentile(&samples_ns, 0.95);
    let deviations = samples_ns
        .iter()
        .map(|value| value.abs_diff(median_ns.round() as u64))
        .collect::<Vec<_>>();
    SampleStatistics {
        samples_ns,
        median_ns,
        p95_ns,
        mad_ns: percentile(&deviations, 0.5),
    }
}

fn percentile(values: &[u64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let position = quantile * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    values[lower] as f64 * (1.0 - weight) + values[upper] as f64 * weight
}

fn bootstrap_delta_ci(
    baseline: &[u64],
    candidate: &[u64],
    pairs: Option<&[(u64, u64)]>,
) -> (f64, f64) {
    const ITERATIONS: usize = 10_000;
    let mut state = 0x4d595df4d0f33173u64;
    let mut deltas = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        if let Some(pairs) = pairs {
            let mut sample = Vec::with_capacity(pairs.len());
            for _ in 0..pairs.len() {
                let index = random_index(&mut state, pairs.len());
                sample.push(pairs[index].1 as i128 - pairs[index].0 as i128);
            }
            sample.sort_unstable();
            deltas.push(median_i128(&sample));
        } else {
            let baseline_sample = resample(baseline, &mut state);
            let candidate_sample = resample(candidate, &mut state);
            deltas.push(percentile(&candidate_sample, 0.5) - percentile(&baseline_sample, 0.5));
        }
    }
    deltas.sort_by(f64::total_cmp);
    (deltas[249], deltas[9749])
}

fn resample(values: &[u64], state: &mut u64) -> Vec<u64> {
    (0..values.len())
        .map(|_| values[random_index(state, values.len())])
        .collect()
}

fn random_index(state: &mut u64, length: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as usize) % length
}

fn median_i128(values: &[i128]) -> f64 {
    if values.is_empty() {
        0.0
    } else if values.len().is_multiple_of(2) {
        let upper = values.len() / 2;
        (values[upper - 1] as f64 + values[upper] as f64) / 2.0
    } else {
        values[values.len() / 2] as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CaptureContract;
    use crate::trace::{
        RunOutcome, SpanKind, SpanRecord, TensorStatsEvent, TerminalEvent, TimingMode,
        TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };

    fn run(cohort: &str, index: usize, duration_ns: u64, pair_id: Option<String>) -> TraceDocument {
        TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: format!("{cohort}-{index}"),
                correlation_id: format!("{cohort}-{index}"),
                entrypoint: "demo::infer".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 6,
                warmup_steps: 5,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract {
                    measurement_scope: MeasurementScope::ProductionEquivalent,
                    ..CaptureContract::default()
                },
                comparison_identity: Some(ComparisonIdentity {
                    implementation_id: Some(cohort.into()),
                    workload_id: "infer".into(),
                    model_id: "m1".into(),
                    config_id: "c1".into(),
                    data_id: "d1".into(),
                    seed_policy: "fixed".into(),
                    physical_batch: 1,
                    accumulation_steps: 1,
                    precision: "f32".into(),
                    device_state: "exclusive".into(),
                    pair_id,
                }),
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "infer".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns,
                step: None,
            }],
            ops: vec![],
            tensors: vec![],
            tensor_stats: vec![],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: duration_ns,
                reason: None,
            },
        }
    }

    fn compare_test_replicates(
        baseline: &[TraceDocument],
        candidate: &[TraceDocument],
    ) -> ReplicatedComparison {
        let receipts = |documents: &[TraceDocument]| {
            documents
                .iter()
                .map(|document| VerifiedBundleInput {
                    run_id: document.run.run_id.clone(),
                    manifest_sha256: "0".repeat(64),
                })
                .collect()
        };
        compare_documents(
            baseline,
            candidate,
            ComparisonInputs {
                verification: ComparisonInputVerification::VerifiedBundles,
                baseline: receipts(baseline),
                candidate: receipts(candidate),
            },
        )
    }

    #[test]
    fn statistics_expose_raw_median_p95_and_mad() {
        let stats = statistics(vec![10, 11, 12, 13, 100]);
        assert_eq!(stats.samples_ns, vec![10, 11, 12, 13, 100]);
        assert_eq!(stats.median_ns, 12.0);
        assert_eq!(stats.p95_ns, 82.6);
        assert_eq!(stats.mad_ns, 1.0);
    }

    #[test]
    fn tensor_stats_average_every_event_and_expose_sample_and_run_coverage() {
        let stats = |label: &str, rms: f64, abs_max: f64| TensorStatsEvent {
            span_id: "s1".into(),
            label: label.into(),
            shape: vec![1],
            dtype: "f32".into(),
            elements: 1,
            non_finite: if rms == 100.0 { 1 } else { 0 },
            rms,
            abs_max,
            mean: rms,
        };
        let mut baseline = run("base", 0, 100, None);
        baseline.tensor_stats = vec![
            stats("stable", 2.0, 4.0),
            stats("drift", 1.0, 2.0),
            stats("drift", 100.0, 200.0),
            stats("only_a", 3.0, 3.0),
        ];
        let mut candidate = run("next", 0, 90, None);
        candidate.tensor_stats = vec![
            stats("stable", 2.2, 4.4),
            stats("drift", 4.0, 8.0),
            stats("only_b", 5.0, 5.0),
        ];

        let comparison = compare_unverified_traces(&[baseline], &[candidate]);
        let drift = &comparison.tensor_stats.matched[0];
        assert_eq!(drift.label, "drift");
        // Duplicate baseline events are averaged, not discarded after the first occurrence.
        assert_eq!(drift.rms_a, 50.5);
        assert_eq!(drift.rms_ratio, Some(4.0 / 50.5));
        // A non-finite count in a duplicate event is retained.
        assert_eq!(drift.non_finite_a, 1);
        assert_eq!(drift.samples_a, 2);
        assert_eq!(drift.samples_b, 1);
        assert_eq!(drift.runs_a, 1);
        assert_eq!(drift.runs_b, 1);
        assert_eq!(comparison.tensor_stats.baseline_runs, 1);
        assert_eq!(comparison.tensor_stats.candidate_runs, 1);
        assert_eq!(comparison.tensor_stats.unmatched_a, vec!["only_a"]);
        assert_eq!(comparison.tensor_stats.unmatched_b, vec!["only_b"]);
    }

    #[test]
    fn out_of_domain_provenance_fails_closed() {
        let baseline = (0..5)
            .map(|i| run("base", i, 100 + i as u64, None))
            .collect::<Vec<_>>();
        let candidate = (0..5)
            .map(|i| {
                let mut document = run("next", i, 90 + i as u64, None);
                document.run.capture_step = 0;
                document.run.warmup_steps = 0;
                document
            })
            .collect::<Vec<_>>();
        let result = compare_test_replicates(&baseline, &candidate);
        assert!(!result.comparable);
        assert_eq!(result.verdict, ComparisonVerdict::Ineligible);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == ComparisonReasonCode::IncompleteCapture
                && reason
                    .message
                    .contains("not a complete, structurally valid capture")
        }));

        let mut zero_batch = baseline.clone();
        for document in &mut zero_batch {
            document
                .run
                .comparison_identity
                .as_mut()
                .unwrap()
                .physical_batch = 0;
        }
        let result = compare_test_replicates(&zero_batch, &zero_batch.clone());
        assert!(!result.comparable);
        assert_eq!(result.verdict, ComparisonVerdict::Ineligible);
    }

    #[test]
    fn confirms_only_when_replicated_interval_excludes_zero() {
        let baseline = [100, 102, 99, 101, 103]
            .into_iter()
            .enumerate()
            .map(|(i, value)| run("base", i, value, None))
            .collect::<Vec<_>>();
        let candidate = [75, 80, 78, 79, 77]
            .into_iter()
            .enumerate()
            .map(|(i, value)| run("next", i, value, None))
            .collect::<Vec<_>>();
        let result = compare_test_replicates(&baseline, &candidate);
        assert!(result.comparable);
        assert_eq!(result.baseline_implementation_id.as_deref(), Some("base"));
        assert_eq!(result.candidate_implementation_id.as_deref(), Some("next"));
        assert_eq!(
            result
                .identity
                .as_ref()
                .and_then(|identity| identity.implementation_id.as_deref()),
            None
        );
        assert_eq!(result.verdict, ComparisonVerdict::CandidateFaster);
        assert!(result.confidence_interval.unwrap().upper_delta_ns < 0.0);
    }

    #[test]
    fn fewer_than_five_or_duplicate_runs_fail_closed() {
        let baseline = (0..4)
            .map(|i| run("base", i, 100, None))
            .collect::<Vec<_>>();
        let mut candidate = (0..5).map(|i| run("next", i, 90, None)).collect::<Vec<_>>();
        candidate[4].run.run_id = candidate[3].run.run_id.clone();
        let result = compare_test_replicates(&baseline, &candidate);
        assert!(!result.comparable);
        assert_eq!(result.verdict, ComparisonVerdict::Ineligible);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == ComparisonReasonCode::InsufficientRuns
                && reason.message.contains("at least 5")
        }));
        assert!(result.reasons.iter().any(|reason| {
            reason.code == ComparisonReasonCode::DuplicateRunIds
                && reason.message.contains("unique")
        }));
    }

    #[test]
    fn partial_pair_metadata_is_ineligible_instead_of_falling_back() {
        let baseline = (0..5)
            .map(|i| run("base", i, 100 + i as u64, Some(format!("pair-{i}"))))
            .collect::<Vec<_>>();
        let mut candidate = (0..5)
            .map(|i| run("next", i, 90 + i as u64, Some(format!("pair-{i}"))))
            .collect::<Vec<_>>();
        candidate[4]
            .run
            .comparison_identity
            .as_mut()
            .unwrap()
            .pair_id = None;
        let result = compare_test_replicates(&baseline, &candidate);
        assert!(!result.comparable);
        assert!(!result.paired);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == ComparisonReasonCode::PairingIncomplete
                && reason.message.contains("every run")
        }));
    }

    #[test]
    fn paired_even_median_averages_the_middle_deltas() {
        assert_eq!(median_i128(&[-5, -1, 3, 9]), 1.0);
    }

    #[test]
    fn missing_empty_or_inconsistent_implementation_ids_fail_closed() {
        let baseline = (0..5)
            .map(|i| run("base", i, 100 + i as u64, None))
            .collect::<Vec<_>>();
        let candidate = (0..5)
            .map(|i| run("next", i, 90 + i as u64, None))
            .collect::<Vec<_>>();

        for (implementation_id, expected_code, expected_reason) in [
            (
                None,
                ComparisonReasonCode::ImplementationIdMissing,
                "missing",
            ),
            (
                Some("   ".to_string()),
                ComparisonReasonCode::ImplementationIdEmpty,
                "must not be empty",
            ),
        ] {
            let mut invalid = baseline.clone();
            invalid[0]
                .run
                .comparison_identity
                .as_mut()
                .unwrap()
                .implementation_id = implementation_id;
            let result = compare_test_replicates(&invalid, &candidate);
            assert!(!result.comparable);
            assert_eq!(result.verdict, ComparisonVerdict::Ineligible);
            assert!(result.reasons.iter().any(|reason| {
                reason.code == expected_code && reason.message.contains(expected_reason)
            }));
        }

        let mut inconsistent = candidate.clone();
        inconsistent[4]
            .run
            .comparison_identity
            .as_mut()
            .unwrap()
            .implementation_id = Some("another-build".into());
        let result = compare_test_replicates(&baseline, &inconsistent);
        assert!(!result.comparable);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == ComparisonReasonCode::ImplementationIdInconsistent
                && reason.message.contains("differs within")
        }));
    }
}
