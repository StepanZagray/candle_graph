use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_graph::trace::{
    write_jsonl, RunOutcome, SpanRecord, TerminalEvent, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
};
use candle_graph::{
    compare_unverified_traces, compare_verified_bundles, publish_bundle, BundleManifest,
    CaptureContract, ComparisonIdentity, ComparisonInputVerification, ComparisonVerdict,
    ExecutionPhase, MeasurementScope, SpanKind, TimingMode, TraceDocument,
};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "candle-graph-comparison-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(cohort: &str, index: usize, duration_ns: u64) -> TraceDocument {
    TraceDocument {
        schema: TRACE_SCHEMA.into(),
        run: TraceRunMeta {
            run_id: format!("{cohort}-{index}"),
            correlation_id: format!("comparison/{cohort}/{index}"),
            entrypoint: "demo::infer".into(),
            phase: ExecutionPhase::Infer,
            timestamp: "2026-08-21T00:00:00Z".into(),
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
                data_id: "held-out-a".into(),
                seed_policy: "fixed".into(),
                physical_batch: 1,
                accumulation_steps: 1,
                precision: "f32".into(),
                device_state: "exclusive".into(),
                pair_id: None,
            }),
            tags: Default::default(),
            candle_version: None,
        },
        spans: vec![SpanRecord {
            id: "root".into(),
            parent_id: None,
            name: "demo::infer".into(),
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

fn publish(root: &Path, document: &TraceDocument) -> PathBuf {
    let trace = root.join(format!("{}.jsonl", document.run.run_id));
    let bundle = root.join(format!("{}-bundle", document.run.run_id));
    write_jsonl(&trace, &document.to_events()).unwrap();
    publish_bundle(&bundle, &trace, None).unwrap();
    bundle
}

#[test]
fn verified_bundle_cohorts_are_the_only_eligible_public_comparison_path() {
    let root = TempRoot::new();
    let baseline = [100, 102, 99, 101, 103]
        .into_iter()
        .enumerate()
        .map(|(index, duration)| run("base", index, duration))
        .collect::<Vec<_>>();
    let candidate = [75, 80, 78, 79, 77]
        .into_iter()
        .enumerate()
        .map(|(index, duration)| run("next", index, duration))
        .collect::<Vec<_>>();
    let baseline_bundles = baseline
        .iter()
        .map(|document| publish(&root.0, document))
        .collect::<Vec<_>>();
    let candidate_bundles = candidate
        .iter()
        .map(|document| publish(&root.0, document))
        .collect::<Vec<_>>();

    let verified = compare_verified_bundles(&baseline_bundles, &candidate_bundles).unwrap();
    assert!(verified.comparable);
    assert_eq!(verified.verdict, ComparisonVerdict::CandidateFaster);
    assert_eq!(
        verified.inputs.verification,
        ComparisonInputVerification::VerifiedBundles
    );
    assert_eq!(verified.inputs.baseline.len(), 5);
    assert_eq!(verified.inputs.candidate.len(), 5);
    assert!(verified
        .inputs
        .baseline
        .iter()
        .chain(&verified.inputs.candidate)
        .all(|input| input.manifest_sha256.len() == 64));

    let unverified = compare_unverified_traces(&baseline, &candidate);
    assert!(!unverified.comparable);
    assert_eq!(unverified.verdict, ComparisonVerdict::Ineligible);
    assert_eq!(
        unverified.inputs.verification,
        ComparisonInputVerification::UnverifiedTraces
    );
    assert!(unverified.confidence_interval.is_none());
    assert!(unverified
        .reasons
        .iter()
        .any(|reason| reason.contains("unverified raw trace")));
}

#[test]
fn comparison_rejects_a_bundle_modified_after_publication() {
    let root = TempRoot::new();
    let baseline = vec![publish(&root.0, &run("base", 0, 100))];
    let candidate = vec![publish(&root.0, &run("next", 0, 90))];
    fs::write(candidate[0].join("trace.jsonl"), b"tampered\n").unwrap();

    let error = compare_verified_bundles(&baseline, &candidate).unwrap_err();
    assert!(error.to_string().contains("verify candidate bundle"));
}

#[test]
fn comparison_binds_each_verified_manifest_to_its_trace_run_id() {
    let root = TempRoot::new();
    let baseline = vec![publish(&root.0, &run("base", 0, 100))];
    let candidate = vec![publish(&root.0, &run("next", 0, 90))];
    let manifest_path = candidate[0].join("bundle.json");
    let mut manifest: BundleManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest.run_id = "different-run".into();
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = compare_verified_bundles(&baseline, &candidate).unwrap_err();
    assert!(error.to_string().contains("manifest run ID"));
}

#[test]
fn compare_help_describes_verified_bundles_and_explicit_raw_trace_mode() {
    let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
        .args(["compare", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--baseline <BUNDLE>..."));
    assert!(help.contains("--candidate <BUNDLE>..."));
    assert!(help.contains("--unverified-traces"));
    assert!(help.contains("diagnostic/ineligible"));
}
