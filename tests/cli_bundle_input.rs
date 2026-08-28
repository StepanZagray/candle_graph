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
    write_jsonl, RunOutcome, SpanRecord, TensorStatsEvent, TerminalEvent, TraceRunMeta,
    SCHEMA as TRACE_SCHEMA,
};
use candle_graph::{
    publish_bundle, verify_bundle, BundleManifest, CaptureContract, CoverageLevel, ExecutionPhase,
    MeasurementScope, SpanKind, TimingMode, TraceDocument,
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
        tensor_stats: Vec::new(),
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
    publish_bundle_with_reports(root, true, true, true)
}

fn publish_bundle_with_reports(
    root: &Path,
    include_kernels: bool,
    include_projection: bool,
    include_gpu_timeline: bool,
) -> (PathBuf, PathBuf) {
    publish_bundle_with_report_options(
        root,
        include_kernels,
        include_projection,
        include_gpu_timeline,
        false,
    )
}

fn publish_bundle_with_report_options(
    root: &Path,
    include_kernels: bool,
    include_projection: bool,
    include_gpu_timeline: bool,
    include_broken_kernel: bool,
) -> (PathBuf, PathBuf) {
    let trace = root.join("raw.jsonl");
    write_jsonl(&trace, &trace_document().to_events()).unwrap();

    let nsight = root.join("nsight-input");
    fs::create_dir(&nsight).unwrap();
    let raw = nsight.join("capture.nsys-rep");
    let kernels = nsight.join("sample_cuda_gpu_kern_sum.csv");
    let broken_kernel = nsight.join("broken_cuda_gpu_kern_sum.csv");
    let projection = nsight.join("sample_nvtx_gpu_proj_trace.csv");
    let gpu_timeline = nsight.join("sample_cuda_gpu_trace.csv");
    fs::write(&raw, b"retained raw report").unwrap();
    if include_kernels {
        fs::write(
            &kernels,
            "Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),Name\n1200,2,600,500,700,gemm\n",
        )
        .unwrap();
    }
    if include_broken_kernel {
        fs::write(&broken_kernel, "unsupported,columns\n1,2\n").unwrap();
    }
    if include_projection {
        fs::write(
            &projection,
            "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops,CorrId\nphase/gpu,10,100,20,80,2,1\n",
        )
        .unwrap();
    }
    if include_gpu_timeline {
        fs::write(
            &gpu_timeline,
            "Name,Start (ns),Duration (ns),CorrId\nkernel/a,20,40,1\nkernel/b,50,20,1\n",
        )
        .unwrap();
    }
    let mut artifact_paths = vec![raw];
    if include_kernels {
        artifact_paths.push(kernels);
    }
    if include_broken_kernel {
        artifact_paths.push(broken_kernel);
    }
    if include_projection {
        artifact_paths.push(projection);
    }
    if include_gpu_timeline {
        artifact_paths.push(gpu_timeline);
    }
    let artifacts = artifact_paths
        .iter()
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

fn write_packet_and_rebind_manifest(bundle: &Path, packet: &serde_json::Value) {
    let evidence_path = bundle.join("evidence.json");
    let packet_bytes = serde_json::to_vec_pretty(packet).unwrap();
    fs::write(&evidence_path, &packet_bytes).unwrap();

    let manifest_path = bundle.join("bundle.json");
    let mut manifest: BundleManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let evidence = manifest
        .files
        .iter_mut()
        .find(|file| file.path == "evidence.json")
        .unwrap();
    evidence.bytes = packet_bytes.len() as u64;
    evidence.sha256 = format!("{:x}", Sha256::digest(&packet_bytes));
    fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
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
    assert_eq!(
        query["result"]["normalized_rows"]["kernels"]["status"],
        "unavailable"
    );
    assert!(query["result"]["normalized_rows"]["kernels"]["total"].is_null());
}

#[test]
fn gpu_summary_and_queries_use_verified_bounded_bundle_evidence() {
    let root = TempRoot::new("gpu-queries");
    let (_, bundle) = publish_augmented_bundle(&root.0);
    let summary_path = root.0.join("summary.json");
    run_summary(&bundle, Some(&summary_path)).unwrap();
    let summary = read_json(&summary_path);
    assert_eq!(summary["schema"], "candle-graph/summary/4");
    assert_eq!(summary["input"]["kind"], "verified_bundle");
    assert!(summary["input"]["gpu_identity_bound"].as_bool().unwrap());
    assert_eq!(summary["gpu"]["status"], "available");
    assert_eq!(summary["gpu"]["normalized_rows"]["kernels"]["total"], 1);

    let cases = [
        (TraceQueryKind::GpuStatus, "gpu-status"),
        (TraceQueryKind::GpuCorrelation, "gpu-correlation"),
        (TraceQueryKind::GpuPhases, "gpu-phases"),
        (TraceQueryKind::GpuKernels, "gpu-kernels"),
        (TraceQueryKind::GpuAttributionGaps, "gpu-attribution-gaps"),
    ];
    for (kind, name) in cases {
        let output = root.0.join(format!("{name}.json"));
        run_query(&bundle, kind, Some(&output)).unwrap();
        let query = read_json(&output);
        assert_eq!(query["schema"], "candle-graph/trace-query/4");
        assert_eq!(query["kind"], name);
        assert_eq!(query["input"]["kind"], "verified_bundle");
        assert_eq!(query["result"]["status"], "available");
    }

    let correlation = read_json(&root.0.join("gpu-correlation.json"));
    assert!(correlation["result"]["complete"].as_bool().unwrap());
    assert_eq!(correlation["result"]["ledger"]["matched"]["total"], 1);
    let phases = read_json(&root.0.join("gpu-phases.json"));
    assert_eq!(phases["result"]["projected_ranges"]["population_total"], 1);
    assert_eq!(phases["result"]["attributed_phases"]["population_total"], 1);
    assert_eq!(
        phases["result"]["projected_ranges"]["sample_selection"],
        "earliest_original_start_rows"
    );
    assert_eq!(
        phases["result"]["projected_ranges"]["global_duration_ranking"],
        false
    );
    let kernels = read_json(&root.0.join("gpu-kernels.json"));
    assert_eq!(kernels["result"]["rows"][0]["name"], "gemm");
    let gaps = read_json(&root.0.join("gpu-attribution-gaps.json"));
    assert_eq!(
        gaps["result"]["matched_without_exact_gpu_busy_attribution"]["total"],
        0
    );
}

#[test]
fn partial_gpu_reports_remain_unknown_in_report_specific_queries() {
    let kernel_root = TempRoot::new("partial-kernel");
    let (_, kernel_bundle) = publish_bundle_with_reports(&kernel_root.0, true, false, false);

    let status_path = kernel_root.0.join("status.json");
    run_query(
        &kernel_bundle,
        TraceQueryKind::GpuStatus,
        Some(&status_path),
    )
    .unwrap();
    let status = read_json(&status_path);
    assert_eq!(
        status["result"]["normalized_rows"]["kernels"]["status"],
        "available"
    );
    assert_eq!(status["result"]["normalized_rows"]["kernels"]["total"], 1);
    assert_eq!(
        status["result"]["normalized_rows"]["projected_ranges"]["status"],
        "unavailable"
    );
    assert!(status["result"]["normalized_rows"]["projected_ranges"]["total"].is_null());

    for (kind, name) in [
        (TraceQueryKind::GpuCorrelation, "correlation"),
        (TraceQueryKind::GpuPhases, "phases"),
        (TraceQueryKind::GpuAttributionGaps, "gaps"),
    ] {
        let output = kernel_root.0.join(format!("{name}.json"));
        run_query(&kernel_bundle, kind, Some(&output)).unwrap();
        let query = read_json(&output);
        assert_eq!(query["result"]["status"], "unavailable");
        assert!(query["result"]["reason"]
            .as_str()
            .unwrap()
            .contains("unknown, not zero"));
    }
    let correlation = read_json(&kernel_root.0.join("correlation.json"));
    assert!(correlation["result"]["complete"].is_null());
    assert!(correlation["result"]["ledger"].is_null());
    let gaps = read_json(&kernel_root.0.join("gaps.json"));
    assert!(gaps["result"]["missing_expected"].is_null());

    let projection_root = TempRoot::new("partial-projection");
    let (_, projection_bundle) =
        publish_bundle_with_reports(&projection_root.0, false, true, false);
    let correlation_path = projection_root.0.join("correlation.json");
    run_query(
        &projection_bundle,
        TraceQueryKind::GpuCorrelation,
        Some(&correlation_path),
    )
    .unwrap();
    let correlation = read_json(&correlation_path);
    assert_eq!(correlation["result"]["status"], "available");
    assert_eq!(correlation["result"]["complete"], true);

    let phases_path = projection_root.0.join("phases.json");
    run_query(
        &projection_bundle,
        TraceQueryKind::GpuPhases,
        Some(&phases_path),
    )
    .unwrap();
    let phases = read_json(&phases_path);
    assert_eq!(phases["result"]["status"], "unavailable");
    assert_eq!(phases["result"]["projected_ranges"]["status"], "available");
    assert_eq!(phases["result"]["projected_ranges"]["population_total"], 1);
    assert_eq!(
        phases["result"]["attributed_phases"]["status"],
        "unavailable"
    );
    assert!(phases["result"]["attributed_phases"]["population_total"].is_null());

    let kernels_path = projection_root.0.join("kernels.json");
    run_query(
        &projection_bundle,
        TraceQueryKind::GpuKernels,
        Some(&kernels_path),
    )
    .unwrap();
    let kernels = read_json(&kernels_path);
    assert_eq!(kernels["result"]["status"], "unavailable");
    assert!(kernels["result"]["population_total"].is_null());
    assert_eq!(kernels["result"]["rows"].as_array().unwrap().len(), 0);

    let broken_root = TempRoot::new("partial-broken-kernel");
    let (_, broken_bundle) =
        publish_bundle_with_report_options(&broken_root.0, true, true, true, true);
    let broken_path = broken_root.0.join("kernels.json");
    run_query(
        &broken_bundle,
        TraceQueryKind::GpuKernels,
        Some(&broken_path),
    )
    .unwrap();
    let broken = read_json(&broken_path);
    assert_eq!(broken["result"]["status"], "failed");
    assert!(broken["result"]["reason"]
        .as_str()
        .unwrap()
        .contains("failed to normalize"));
    assert!(broken["result"]["population_total"].is_null());
}

#[test]
fn summary_and_query_outputs_cannot_modify_a_verified_bundle() {
    let root = TempRoot::new("protected-output");
    let (_, bundle) = publish_augmented_bundle(&root.0);
    let initial_receipt = verify_bundle(&bundle).unwrap();
    let initial_evidence = fs::read(bundle.join("evidence.json")).unwrap();

    let direct_overwrite = bundle.join("evidence.json");
    let direct_error = run_summary(&bundle, Some(&direct_overwrite)).unwrap_err();
    assert!(direct_error.to_string().contains("inside verified bundle"));

    let injected = bundle.join("injected.json");
    let injection_error =
        run_query(&bundle, TraceQueryKind::GpuStatus, Some(&injected)).unwrap_err();
    assert!(injection_error
        .to_string()
        .contains("inside verified bundle"));
    assert!(!injected.exists());

    let dotdot = bundle.join("nsight/../dotdot.json");
    let dotdot_error = run_summary(&bundle, Some(&dotdot)).unwrap_err();
    assert!(dotdot_error.to_string().contains("inside verified bundle"));
    assert!(!bundle.join("dotdot.json").exists());

    let traversing = bundle.join("new-directory/../../outside.json");
    let traversing_error = run_summary(&bundle, Some(&traversing)).unwrap_err();
    assert!(traversing_error
        .to_string()
        .contains("inside verified bundle"));
    assert!(!bundle.join("new-directory").exists());

    let root_error = run_summary(&bundle, Some(&bundle)).unwrap_err();
    assert!(root_error.to_string().contains("inside verified bundle"));
    assert_eq!(verify_bundle(&bundle).unwrap(), initial_receipt);
    assert_eq!(
        fs::read(bundle.join("evidence.json")).unwrap(),
        initial_evidence
    );
}

#[cfg(unix)]
#[test]
fn output_symlink_alias_into_verified_bundle_is_rejected() {
    let root = TempRoot::new("protected-output-symlink");
    let (_, bundle) = publish_augmented_bundle(&root.0);
    let initial_receipt = verify_bundle(&bundle).unwrap();
    let alias = root.0.join("bundle-alias");
    std::os::unix::fs::symlink(&bundle, &alias).unwrap();
    let output = alias.join("injected.json");

    let error = run_query(&bundle, TraceQueryKind::GpuStatus, Some(&output)).unwrap_err();
    assert!(error.to_string().contains("inside verified bundle"));
    assert!(!bundle.join("injected.json").exists());
    assert_eq!(verify_bundle(&bundle).unwrap(), initial_receipt);
}

#[test]
fn verified_bundle_consumer_rejects_old_evidence_and_graph_schemas() {
    let evidence_root = TempRoot::new("old-evidence-schema");
    let (_, evidence_bundle) = publish_augmented_bundle(&evidence_root.0);
    let mut packet = read_json(&evidence_bundle.join("evidence.json"));
    packet["schema"] = serde_json::json!("candle-graph/evidence/3");
    write_packet_and_rebind_manifest(&evidence_bundle, &packet);
    verify_bundle(&evidence_bundle).unwrap();
    let error = load_evidence(&evidence_bundle).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported evidence schema"));

    let graph_root = TempRoot::new("old-graph-schema");
    let (_, graph_bundle) = publish_augmented_bundle(&graph_root.0);
    let mut packet = read_json(&graph_bundle.join("evidence.json"));
    packet["graph"]["schema"] = serde_json::json!("candle-graph/graph/4");
    write_packet_and_rebind_manifest(&graph_bundle, &packet);
    verify_bundle(&graph_bundle).unwrap();
    let error = load_evidence(&graph_bundle).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported graph schema"));
}

#[test]
fn slowest_host_query_only_reports_measured_scope_headlines() {
    let root = TempRoot::new("slowest-host-scope");
    let (trace, _) = publish_augmented_bundle(&root.0);
    let output = root.0.join("slowest-host.json");
    run_query(&trace, TraceQueryKind::SlowestHost, Some(&output)).unwrap();
    let query = read_json(&output);

    assert!(query["result"].get("slowest_host_spans").is_some());
    assert!(query["result"].get("slowest_host_ops").is_none());
}

#[test]
fn summary_and_query_surface_ordered_tensor_stats() {
    let root = TempRoot::new("tensor-stats");
    let trace = root.0.join("trace.jsonl");
    let mut document = trace_document();
    document.tensor_stats = vec![
        TensorStatsEvent {
            span_id: "gpu".into(),
            label: "seam/out_y".into(),
            shape: vec![2, 3],
            dtype: "f32".into(),
            elements: 6,
            non_finite: 0,
            rms: 1.0,
            abs_max: 2.0,
            mean: 0.25,
        },
        TensorStatsEvent {
            span_id: "gpu".into(),
            label: "seam/gate_logits".into(),
            shape: vec![2],
            dtype: "f32".into(),
            elements: 2,
            non_finite: 1,
            rms: 0.0,
            abs_max: 0.0,
            mean: 0.0,
        },
    ];
    write_jsonl(&trace, &document.to_events()).unwrap();

    let summary_path = root.0.join("summary.json");
    run_summary(&trace, Some(&summary_path)).unwrap();
    let summary = read_json(&summary_path);
    assert_eq!(summary["tensor_stats"]["events"], 2);
    assert_eq!(summary["tensor_stats"]["non_finite_events"], 1);

    let query_path = root.0.join("query.json");
    run_query(&trace, TraceQueryKind::TensorStats, Some(&query_path)).unwrap();
    let query = read_json(&query_path);
    assert_eq!(query["result"][0]["label"], "seam/out_y");
    assert_eq!(query["result"][1]["label"], "seam/gate_logits");
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
