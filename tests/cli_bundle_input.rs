use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_graph::cli::trace_cli::{load_evidence, run_query, run_summary, TraceQueryKind};
use candle_graph::nsight::{
    CaptureCorrelation, CaptureHardware, CaptureManifest, CaptureRun, CaptureTool,
    GpuEvidenceStatus, ManifestArtifact, ProvenanceBindingState, CAPTURE_MANIFEST_SCHEMA,
};
use candle_graph::trace::{
    write_jsonl, RunOutcome, SpanRecord, TerminalEvent, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
};
use candle_graph::{
    publish_bundle, CaptureContract, CoverageLevel, ExecutionPhase, MeasurementScope, SpanKind,
    TimingMode, TraceDocument,
};
use sha2::{Digest, Sha256};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "candle-graph-cli-{label}-{}-{nonce}",
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

fn trace_document() -> TraceDocument {
    TraceDocument {
        schema: TRACE_SCHEMA.into(),
        run: TraceRunMeta {
            run_id: "cli-bundle-run".into(),
            correlation_id: "cli/bundle/run".into(),
            entrypoint: "demo::update".into(),
            phase: ExecutionPhase::Infer,
            timestamp: "2026-08-23T00:00:00Z".into(),
            capture_step: 1,
            warmup_steps: 0,
            device: "cuda:0".into(),
            measured_region_device_synchronized: true,
            timing_mode: TimingMode::Host,
            capture_contract: CaptureContract {
                measurement_scope: MeasurementScope::ProductionEquivalent,
                device_timing: CoverageLevel::None,
                required_semantic_labels: vec!["phase/gpu".into()],
                gpu_expected_semantic_labels: vec!["phase/gpu".into()],
                ..CaptureContract::default()
            },
            comparison_identity: None,
            tags: BTreeMap::new(),
            candle_version: None,
        },
        spans: vec![
            SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "demo::update".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns: 200,
                step: None,
            },
            SpanRecord {
                id: "gpu".into(),
                parent_id: Some("root".into()),
                name: "phase/gpu".into(),
                kind: SpanKind::Module,
                measured: false,
                start_ns: 10,
                closed: true,
                duration_ns: 100,
                step: None,
            },
        ],
        ops: Vec::new(),
        tensors: Vec::new(),
        memory: Vec::new(),
        device_memory: Vec::new(),
        device_intervals: Vec::new(),
        gradients: Vec::new(),
        edges: Vec::new(),
        terminal: TerminalEvent {
            outcome: RunOutcome::Complete,
            timestamp_ns: 200,
            reason: None,
        },
    }
}

fn manifest_artifact(root: &Path, path: &Path) -> ManifestArtifact {
    let bytes = fs::read(path).unwrap();
    ManifestArtifact {
        path: path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/"),
        size_bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    }
}

fn publish_augmented_bundle(root: &Path) -> (PathBuf, PathBuf) {
    let trace = root.join("raw.jsonl");
    write_jsonl(&trace, &trace_document().to_events()).unwrap();

    let nsight = root.join("nsight-input");
    fs::create_dir(&nsight).unwrap();
    let raw = nsight.join("capture.nsys-rep");
    let kernels = nsight.join("sample_cuda_gpu_kern_sum.csv");
    let projection = nsight.join("sample_nvtx_gpu_proj_trace.csv");
    let gpu_timeline = nsight.join("sample_cuda_gpu_trace.csv");
    fs::write(&raw, b"retained raw report").unwrap();
    fs::write(
        &kernels,
        "Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),Name\n1200,2,600,500,700,gemm\n",
    )
    .unwrap();
    fs::write(
        &projection,
        "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops,CorrId\nphase/gpu,10,100,20,80,2,1\n",
    )
    .unwrap();
    fs::write(
        &gpu_timeline,
        "Name,Start (ns),Duration (ns),CorrId\nkernel/a,20,40,1\nkernel/b,50,20,1\n",
    )
    .unwrap();
    let artifacts = [&raw, &kernels, &projection, &gpu_timeline]
        .into_iter()
        .map(|path| manifest_artifact(&nsight, path))
        .collect();
    let manifest = CaptureManifest {
        schema: CAPTURE_MANIFEST_SCHEMA.into(),
        run: CaptureRun {
            id: "cli-bundle-run".into(),
            started_at: None,
        },
        correlation: CaptureCorrelation {
            id: "cli/bundle/run".into(),
        },
        tool: CaptureTool {
            name: "nsys".into(),
            version: "test".into(),
        },
        commands: vec!["nsys profile demo".into()],
        hardware: CaptureHardware {
            host: Some("test-host".into()),
            devices: vec!["test-gpu".into()],
        },
        source_revisions: BTreeMap::new(),
        required_semantic_labels: vec!["phase/gpu".into()],
        gpu_expected_semantic_labels: vec!["phase/gpu".into()],
        cpu_only_semantic_labels: Vec::new(),
        artifacts,
    };
    fs::write(
        nsight.join("capture-manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let bundle = root.join("profile");
    publish_bundle(&bundle, &trace, Some(&nsight)).unwrap();
    (trace, bundle)
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn bundle_root_and_bundled_trace_retain_verified_gpu_evidence() {
    let root = TempRoot::new("bundle-input");
    let (_, bundle) = publish_augmented_bundle(&root.0);
    let bundled_trace = bundle.join("trace.jsonl");

    for input in [&bundle, &bundled_trace] {
        let evidence = load_evidence(input).unwrap();
        assert_eq!(evidence.gpu.status, GpuEvidenceStatus::Available);
        assert_eq!(
            evidence.gpu.provenance.binding,
            ProvenanceBindingState::Bound
        );
        assert!(evidence.gpu.correlation.complete);
        assert_eq!(evidence.gpu.kernels[0].name, "gemm");
        assert_eq!(evidence.gpu.phase_attribution[0].semantic_key, "phase/gpu");
    }
}

#[test]
fn raw_trace_gpu_status_is_explicitly_unavailable() {
    let root = TempRoot::new("raw-trace");
    let (trace, _) = publish_augmented_bundle(&root.0);
    let output = root.0.join("raw-status.json");

    run_query(&trace, TraceQueryKind::GpuStatus, Some(&output)).unwrap();
    let query = read_json(&output);
    assert_eq!(query["input"]["kind"], "raw_trace");
    assert_eq!(query["result"]["status"], "unavailable");
    assert_eq!(
        query["result"]["reason"],
        "Nsight capture was not requested"
    );
    assert_eq!(query["result"]["normalized_rows"]["kernels"], 0);
}

#[test]
fn gpu_summary_and_queries_use_verified_bounded_bundle_evidence() {
    let root = TempRoot::new("gpu-queries");
    let (_, bundle) = publish_augmented_bundle(&root.0);
    let summary_path = root.0.join("summary.json");
    run_summary(&bundle, Some(&summary_path)).unwrap();
    let summary = read_json(&summary_path);
    assert_eq!(summary["schema"], "candle-graph/summary/3");
    assert_eq!(summary["input"]["kind"], "verified_bundle");
    assert!(summary["input"]["gpu_identity_bound"].as_bool().unwrap());
    assert_eq!(summary["gpu"]["status"], "available");
    assert_eq!(summary["gpu"]["normalized_rows"]["kernels"], 1);

    let cases = [
        (TraceQueryKind::GpuStatus, "gpu-status"),
        (TraceQueryKind::GpuCorrelation, "gpu-correlation"),
        (TraceQueryKind::GpuPhases, "gpu-phases"),
        (TraceQueryKind::GpuKernels, "gpu-kernels"),
        (
            TraceQueryKind::GpuAttributionGaps,
            "gpu-attribution-gaps",
        ),
    ];
    for (kind, name) in cases {
        let output = root.0.join(format!("{name}.json"));
        run_query(&bundle, kind, Some(&output)).unwrap();
        let query = read_json(&output);
        assert_eq!(query["schema"], "candle-graph/trace-query/3");
        assert_eq!(query["kind"], name);
        assert_eq!(query["input"]["kind"], "verified_bundle");
        assert_eq!(query["result"]["status"], "available");
    }

    let correlation = read_json(&root.0.join("gpu-correlation.json"));
    assert!(correlation["result"]["complete"].as_bool().unwrap());
    assert_eq!(correlation["result"]["ledger"]["matched"]["total"], 1);
    let phases = read_json(&root.0.join("gpu-phases.json"));
    assert_eq!(phases["result"]["projected_ranges"]["total"], 1);
    assert_eq!(phases["result"]["attributed_phases"]["total"], 1);
    let kernels = read_json(&root.0.join("gpu-kernels.json"));
    assert_eq!(kernels["result"]["rows"][0]["name"], "gemm");
    let gaps = read_json(&root.0.join("gpu-attribution-gaps.json"));
    assert_eq!(
        gaps["result"]["matched_without_exact_gpu_busy_attribution"]["total"],
        0
    );
}

#[test]
fn augmented_inputs_fail_closed_instead_of_falling_back_to_trace_only() {
    let root = TempRoot::new("fail-closed");
    let (trace, bundle) = publish_augmented_bundle(&root.0);
    let evidence_path = bundle.join("evidence.json");
    let mut evidence_bytes = fs::read(&evidence_path).unwrap();
    let trailing_newline = evidence_bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .unwrap();
    evidence_bytes[trailing_newline] = b' ';
    fs::write(&evidence_path, evidence_bytes).unwrap();
    let error = load_evidence(&bundle.join("trace.jsonl")).unwrap_err();
    assert!(format!("{error:#}").contains("mismatch"));

    let unverified = root.0.join("unverified-profile");
    fs::create_dir(&unverified).unwrap();
    fs::copy(trace, unverified.join("trace.jsonl")).unwrap();
    fs::write(unverified.join("evidence.json"), b"{}").unwrap();
    let error = load_evidence(&unverified.join("trace.jsonl")).unwrap_err();
    assert!(error.to_string().contains("refusing to discard"));
}

#[test]
fn arbitrary_trace_filename_cannot_bypass_augmented_parent_detection() {
    let root = TempRoot::new("arbitrary-name");
    let (trace, _) = publish_augmented_bundle(&root.0);
    let unverified = root.0.join("unverified-custom-profile");
    fs::create_dir(&unverified).unwrap();
    let custom_trace = unverified.join("captured-update.data");
    fs::copy(trace, &custom_trace).unwrap();
    fs::create_dir(unverified.join("nsight")).unwrap();

    let error = load_evidence(&custom_trace).unwrap_err();
    assert!(error.to_string().contains("refusing to discard"));
}
