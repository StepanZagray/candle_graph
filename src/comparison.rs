//! Fail-closed replicated performance comparisons.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::capability::MeasurementScope;
use crate::trace::{analyze_health, ComparisonIdentity, TraceDocument};

pub const SCHEMA: &str = "candle-graph/comparison/2";
pub const MINIMUM_RUNS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonVerdict {
    Ineligible,
    Inconclusive,
    CandidateFaster,
    CandidateSlower,
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
pub struct ReplicatedComparison {
    pub schema: String,
    pub metric: String,
    pub comparable: bool,
    pub paired: bool,
    pub verdict: ComparisonVerdict,
    pub reasons: Vec<String>,
    pub identity: Option<ComparisonIdentity>,
    pub baseline: SampleStatistics,
    pub candidate: SampleStatistics,
    pub median_delta_ns: f64,
    pub median_delta_percent: Option<f64>,
    pub confidence_interval: Option<ConfidenceInterval>,
}

pub fn compare_replicates(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
) -> ReplicatedComparison {
    let mut reasons = Vec::new();
    let baseline_samples = measured_samples(baseline, "baseline", &mut reasons);
    let candidate_samples = measured_samples(candidate, "candidate", &mut reasons);
    if baseline.len() < MINIMUM_RUNS || candidate.len() < MINIMUM_RUNS {
        reasons.push(format!(
            "at least {MINIMUM_RUNS} independent baseline and candidate runs are required"
        ));
    }
    require_independent_run_ids(baseline, candidate, &mut reasons);
    require_consistent_capture_semantics(baseline, candidate, &mut reasons);

    let identity = common_identity(baseline, candidate, &mut reasons);
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

    ReplicatedComparison {
        schema: SCHEMA.into(),
        metric: "outer_wall_time_ns".into(),
        comparable,
        paired,
        verdict,
        reasons,
        identity,
        baseline: baseline_stats,
        candidate: candidate_stats,
        median_delta_ns,
        median_delta_percent,
        confidence_interval,
    }
}

fn require_independent_run_ids(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<String>,
) {
    let ids = baseline
        .iter()
        .chain(candidate)
        .map(|document| document.run.run_id.as_str())
        .collect::<Vec<_>>();
    if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
        reasons.push("run IDs must be unique across all replicates".into());
    }
}

fn require_consistent_capture_semantics(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<String>,
) {
    let Some(first) = baseline.first().or_else(|| candidate.first()) else {
        reasons.push("comparison contains no runs".into());
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
        reasons.push(
            "entrypoint, phase, device, timing mode, synchronization, warmup, capture step, and capture contract must match"
                .into(),
        );
    }
}

fn measured_samples(docs: &[TraceDocument], cohort: &str, reasons: &mut Vec<String>) -> Vec<u64> {
    docs.iter()
        .enumerate()
        .map(|(index, doc)| {
            let health = analyze_health(doc);
            if !health.structurally_valid || !health.capture_complete {
                reasons.push(format!(
                    "{cohort} run {} is not a complete, structurally valid capture",
                    index + 1
                ));
            }
            if doc.run.capture_contract.measurement_scope != MeasurementScope::ProductionEquivalent
            {
                reasons.push(format!(
                    "{cohort} run {} is not declared production-equivalent",
                    index + 1
                ));
            }
            if !doc.run.device.starts_with("cpu") && !doc.run.measured_region_device_synchronized {
                reasons.push(format!(
                    "{cohort} run {} does not synchronize its measured device region",
                    index + 1
                ));
            }
            let values = doc
                .spans
                .iter()
                .filter(|span| span.measured && span.closed)
                .map(|span| span.duration_ns)
                .collect::<Vec<_>>();
            if values.len() != 1 {
                reasons.push(format!(
                    "{cohort} run {} does not contain exactly one closed measured region",
                    index + 1
                ));
            }
            values.into_iter().next().unwrap_or(0)
        })
        .collect()
}

fn common_identity(
    baseline: &[TraceDocument],
    candidate: &[TraceDocument],
    reasons: &mut Vec<String>,
) -> Option<ComparisonIdentity> {
    let identities = baseline
        .iter()
        .chain(candidate)
        .map(|doc| doc.run.comparison_identity.as_ref())
        .collect::<Vec<_>>();
    let Some(first) = identities.first().copied().flatten() else {
        reasons.push("comparison identity is missing".into());
        return None;
    };
    if identities.iter().any(|identity| identity.is_none()) {
        reasons.push("comparison identity is missing from one or more runs".into());
        return None;
    }
    if identities
        .iter()
        .flatten()
        .any(|identity| !same_conditions(first, identity))
    {
        reasons.push(
            "workload, model, configuration, data, seed, batch, precision, or device state differs"
                .into(),
        );
        return None;
    }
    let mut result = first.clone();
    result.pair_id = None;
    Some(result)
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
    reasons: &mut Vec<String>,
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
    let collect = |docs: &[TraceDocument]| -> Result<BTreeMap<String, u64>, &'static str> {
        let mut values = BTreeMap::new();
        for doc in docs {
            let pair = doc
                .run
                .comparison_identity
                .as_ref()
                .and_then(|identity| identity.pair_id.clone())
                .ok_or("pair IDs must be present on every run when pairing is requested")?;
            let value = doc
                .spans
                .iter()
                .find(|span| span.measured && span.closed)
                .ok_or("paired runs require one closed measured region")?
                .duration_ns;
            if values.insert(pair, value).is_some() {
                return Err("pair IDs must be unique within each cohort");
            }
        }
        Ok(values)
    };
    let left = match collect(baseline) {
        Ok(values) => values,
        Err(reason) => {
            reasons.push(reason.into());
            return None;
        }
    };
    let right = match collect(candidate) {
        Ok(values) => values,
        Err(reason) => {
            reasons.push(reason.into());
            return None;
        }
    };
    if left.keys().collect::<BTreeSet<_>>() != right.keys().collect::<BTreeSet<_>>() {
        reasons.push("baseline and candidate pair-ID sets must match exactly".into());
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
        RunOutcome, SpanKind, SpanRecord, TerminalEvent, TimingMode, TraceRunMeta,
        SCHEMA as TRACE_SCHEMA,
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
                capture_step: 1,
                warmup_steps: 5,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract {
                    measurement_scope: MeasurementScope::ProductionEquivalent,
                    ..CaptureContract::default()
                },
                comparison_identity: Some(ComparisonIdentity {
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

    #[test]
    fn statistics_expose_raw_median_p95_and_mad() {
        let stats = statistics(vec![10, 11, 12, 13, 100]);
        assert_eq!(stats.samples_ns, vec![10, 11, 12, 13, 100]);
        assert_eq!(stats.median_ns, 12.0);
        assert_eq!(stats.p95_ns, 82.6);
        assert_eq!(stats.mad_ns, 1.0);
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
        let result = compare_replicates(&baseline, &candidate);
        assert!(result.comparable);
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
        let result = compare_replicates(&baseline, &candidate);
        assert!(!result.comparable);
        assert_eq!(result.verdict, ComparisonVerdict::Ineligible);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("at least 5")));
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("unique")));
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
        let result = compare_replicates(&baseline, &candidate);
        assert!(!result.comparable);
        assert!(!result.paired);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("every run")));
    }

    #[test]
    fn paired_even_median_averages_the_middle_deltas() {
        assert_eq!(median_i128(&[-5, -1, 3, 9]), 1.0);
    }
}
