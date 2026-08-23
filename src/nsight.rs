//! Stable normalization seam for official `nsys stats --format csv` reports.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SUMMARY_DISPLAY_LIMIT: usize = 100;
const TIMELINE_DISPLAY_LIMIT: usize = 500;
const CAPTURE_MANIFEST: &str = "capture-manifest.json";
pub const CAPTURE_MANIFEST_SCHEMA: &str = "candle-graph/nsight-capture/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuEvidenceStatus {
    Available,
    Unavailable,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceBindingState {
    Bound,
    Partial,
    Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsightArtifact {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub schema: String,
    pub run: CaptureRun,
    pub correlation: CaptureCorrelation,
    pub tool: CaptureTool,
    pub commands: Vec<String>,
    pub hardware: CaptureHardware,
    pub source_revisions: BTreeMap<String, String>,
    /// All application labels whose exact cardinality is required by the trace contract.
    pub required_semantic_labels: Vec<String>,
    /// The required-label subset expected in `nvtx_gpu_proj_trace`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_expected_semantic_labels: Vec<String>,
    /// The required-label subset that must be absent from `nvtx_gpu_proj_trace`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_only_semantic_labels: Vec<String>,
    pub artifacts: Vec<ManifestArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureRun {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCorrelation {
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureTool {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureHardware {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default)]
    pub devices: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsightProvenance {
    pub binding: ProvenanceBindingState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<CaptureManifest>,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NsightSummaryRow {
    pub name: String,
    pub total_ns: u64,
    pub count: u64,
    pub average_ns: u64,
    pub minimum_ns: u64,
    pub maximum_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NsightTimelineRow {
    pub name: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_key: Option<String>,
    pub start_ns: u64,
    pub duration_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_duration_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_operations: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsightCoverage {
    pub kernel_summary: bool,
    pub runtime_summary: bool,
    pub memory_summary: bool,
    pub nvtx_projection: bool,
    pub gpu_timeline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrelationDuplicate {
    pub semantic_key: String,
    pub occurrences: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsightCorrelationLedger {
    pub expected: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_only: Vec<String>,
    pub observed: Vec<String>,
    pub matched: Vec<String>,
    pub missing_expected: Vec<String>,
    pub unexpected_observed: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unexpected_cpu_only: Vec<String>,
    pub duplicates: Vec<CorrelationDuplicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsightCorrelation {
    pub mode: String,
    pub clock_aligned: bool,
    pub complete: bool,
    pub ledger: NsightCorrelationLedger,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportLimit {
    pub total_rows: usize,
    pub displayed_rows: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseGpuAttribution {
    pub semantic_key: String,
    /// Nsight projected and GPU intervals share an Nsight time plane, but are not aligned to the
    /// host trace clock.
    pub clock_aligned: bool,
    pub projected_start_ns: u64,
    pub projected_duration_ns: u64,
    pub gpu_busy_ns: u64,
    pub gpu_operation_count: usize,
    pub join_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NsightEvidence {
    pub status: GpuEvidenceStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_report: Option<NsightArtifact>,
    #[serde(default)]
    pub source_csv: Vec<NsightArtifact>,
    pub provenance: NsightProvenance,
    pub coverage: NsightCoverage,
    pub correlation: NsightCorrelation,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub limits: BTreeMap<String, ReportLimit>,
    #[serde(default)]
    pub kernels: Vec<NsightSummaryRow>,
    #[serde(default)]
    pub runtime_calls: Vec<NsightSummaryRow>,
    #[serde(default)]
    pub memory_operations: Vec<NsightSummaryRow>,
    #[serde(default)]
    pub nvtx_ranges: Vec<NsightTimelineRow>,
    #[serde(default)]
    pub gpu_timeline: Vec<NsightTimelineRow>,
    #[serde(default)]
    pub phase_attribution: Vec<PhaseGpuAttribution>,
}

impl NsightEvidence {
    /// Bind normalized artifacts to the application trace that is consuming them.
    pub fn bind_to_trace(&mut self, run_id: &str, correlation_id: &str) {
        let Some(manifest) = self.provenance.manifest.as_ref() else {
            self.provenance.binding = ProvenanceBindingState::Partial;
            self.provenance
                .diagnostics
                .push("Trace binding cannot be verified without capture-manifest.json".into());
            return;
        };
        let mut mismatches = Vec::new();
        if manifest.run.id != run_id {
            mismatches.push(format!(
                "manifest run `{}` does not match trace run `{run_id}`",
                manifest.run.id
            ));
        }
        if manifest.correlation.id != correlation_id {
            mismatches.push(format!(
                "manifest correlation `{}` does not match trace correlation `{correlation_id}`",
                manifest.correlation.id
            ));
        }
        if mismatches.is_empty() && self.provenance.binding != ProvenanceBindingState::Mismatch {
            self.provenance.binding = ProvenanceBindingState::Bound;
        } else if !mismatches.is_empty() {
            self.provenance.binding = ProvenanceBindingState::Mismatch;
            self.provenance.diagnostics.extend(mismatches);
        }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: GpuEvidenceStatus::Unavailable,
            reason: Some(reason.into()),
            raw_report: None,
            source_csv: Vec::new(),
            provenance: NsightProvenance {
                binding: ProvenanceBindingState::Partial,
                manifest: None,
                diagnostics: vec!["No capture manifest was loaded".into()],
            },
            coverage: NsightCoverage::default(),
            correlation: empty_correlation(),
            diagnostics: Vec::new(),
            limits: BTreeMap::new(),
            kernels: Vec::new(),
            runtime_calls: Vec::new(),
            memory_operations: Vec::new(),
            nvtx_ranges: Vec::new(),
            gpu_timeline: Vec::new(),
            phase_attribution: Vec::new(),
        }
    }

    /// Load official CSV reports. A missing or invalid capture manifest lowers provenance
    /// confidence without suppressing independently parseable GPU evidence.
    pub fn load_optional(dir: Option<&Path>, expected_semantic_keys: &[String]) -> Self {
        Self::load_optional_with_semantic_contract(
            dir,
            expected_semantic_keys,
            expected_semantic_keys,
            &[],
        )
    }

    pub fn load_optional_with_semantic_contract(
        dir: Option<&Path>,
        required_application_labels: &[String],
        gpu_expected_semantic_keys: &[String],
        cpu_only_semantic_keys: &[String],
    ) -> Self {
        let Some(dir) = dir else {
            return Self::unavailable("Nsight capture was not requested");
        };
        match Self::load_with_semantic_contract(
            dir,
            required_application_labels,
            gpu_expected_semantic_keys,
            cpu_only_semantic_keys,
        ) {
            Ok(evidence) => evidence,
            Err(error) => Self {
                status: GpuEvidenceStatus::Failed,
                reason: Some(error.to_string()),
                ..Self::unavailable("Nsight report normalization failed")
            },
        }
    }

    pub fn load(dir: &Path, expected_semantic_keys: &[String]) -> anyhow::Result<Self> {
        Self::load_with_semantic_contract(dir, expected_semantic_keys, expected_semantic_keys, &[])
    }

    pub fn load_with_semantic_contract(
        dir: &Path,
        required_application_labels: &[String],
        gpu_expected_semantic_keys: &[String],
        cpu_only_semantic_keys: &[String],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            dir.is_dir(),
            "Nsight report directory does not exist: {}",
            dir.display()
        );
        let mut files: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();
        files.sort();

        let raw_report_path = files.iter().find(|path| extension(path) == "nsys-rep");
        let raw_report = raw_report_path
            .map(|path| artifact_metadata(dir, path))
            .transpose()?;
        let mut result = Self {
            status: GpuEvidenceStatus::Unavailable,
            reason: None,
            raw_report,
            source_csv: Vec::new(),
            provenance: load_provenance(dir),
            coverage: NsightCoverage::default(),
            correlation: empty_correlation(),
            diagnostics: Vec::new(),
            limits: BTreeMap::new(),
            kernels: Vec::new(),
            runtime_calls: Vec::new(),
            memory_operations: Vec::new(),
            nvtx_ranges: Vec::new(),
            gpu_timeline: Vec::new(),
            phase_attribution: Vec::new(),
        };

        let mut all_nvtx_ranges = Vec::new();
        let mut all_gpu_timeline = Vec::new();
        for path in files.iter().filter(|path| extension(path) == "csv") {
            let name = path
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or_default();
            let parsed: anyhow::Result<Option<ReportLimit>> = if name.contains("cuda_gpu_kern_sum")
            {
                parse_summary(path).map(|rows| {
                    let (rows, limit) = truncate_rows(rows, SUMMARY_DISPLAY_LIMIT);
                    result.kernels.extend(rows);
                    result.coverage.kernel_summary = true;
                    Some(limit)
                })
            } else if name.contains("cuda_api_sum") {
                parse_summary(path).map(|rows| {
                    let (rows, limit) = truncate_rows(rows, SUMMARY_DISPLAY_LIMIT);
                    result.runtime_calls.extend(rows);
                    result.coverage.runtime_summary = true;
                    Some(limit)
                })
            } else if name.contains("cuda_gpu_mem_time_sum") {
                parse_summary(path).map(|rows| {
                    let (rows, limit) = truncate_rows(rows, SUMMARY_DISPLAY_LIMIT);
                    result.memory_operations.extend(rows);
                    result.coverage.memory_summary = true;
                    Some(limit)
                })
            } else if name.contains("nvtx_gpu_proj_trace") {
                parse_timeline(path, "nvtx_range", true).map(|rows| {
                    let limit = report_limit(rows.len(), TIMELINE_DISPLAY_LIMIT);
                    all_nvtx_ranges.extend(rows);
                    result.coverage.nvtx_projection = true;
                    Some(limit)
                })
            } else if name.contains("cuda_gpu_trace") {
                parse_timeline(path, "gpu_operation", false).map(|rows| {
                    let limit = report_limit(rows.len(), TIMELINE_DISPLAY_LIMIT);
                    all_gpu_timeline.extend(rows);
                    result.coverage.gpu_timeline = true;
                    Some(limit)
                })
            } else {
                Ok(None)
            };
            match parsed {
                Ok(Some(limit)) => {
                    result.source_csv.push(artifact_metadata(dir, path)?);
                    result.limits.insert(name.to_string(), limit);
                }
                Ok(None) => {}
                Err(error) => result
                    .diagnostics
                    .push(format!("{}: {error}", path.display())),
            }
        }

        result.correlation = build_correlation(
            gpu_expected_semantic_keys,
            cpu_only_semantic_keys,
            &all_nvtx_ranges,
            &all_gpu_timeline,
        );
        result.phase_attribution = attribute_gpu_phases(&all_nvtx_ranges, &all_gpu_timeline);
        all_nvtx_ranges.sort_by(|left, right| {
            left.start_ns
                .cmp(&right.start_ns)
                .then_with(|| left.name.cmp(&right.name))
        });
        all_gpu_timeline.sort_by(|left, right| {
            left.start_ns
                .cmp(&right.start_ns)
                .then_with(|| left.name.cmp(&right.name))
        });
        all_nvtx_ranges.truncate(TIMELINE_DISPLAY_LIMIT);
        all_gpu_timeline.truncate(TIMELINE_DISPLAY_LIMIT);
        result.nvtx_ranges = all_nvtx_ranges;
        result.gpu_timeline = all_gpu_timeline;

        validate_provenance(
            dir,
            required_application_labels,
            gpu_expected_semantic_keys,
            cpu_only_semantic_keys,
            &result.source_csv,
            result.raw_report.as_ref(),
            &mut result.provenance,
        );
        let useful_rows = result.kernels.len()
            + result.runtime_calls.len()
            + result.memory_operations.len()
            + result.nvtx_ranges.len()
            + result.gpu_timeline.len();
        if useful_rows == 0 {
            result.reason = status_reason(dir)
                .or_else(|| Some("No supported nsys stats CSV reports were found".into()));
        } else {
            result.status = GpuEvidenceStatus::Available;
        }
        Ok(result)
    }
}

fn empty_correlation() -> NsightCorrelation {
    NsightCorrelation {
        mode: "none".into(),
        clock_aligned: false,
        complete: false,
        ledger: NsightCorrelationLedger::default(),
        reason: Some("No projected NVTX ranges were normalized".into()),
    }
}

fn load_provenance(dir: &Path) -> NsightProvenance {
    let path = dir.join(CAPTURE_MANIFEST);
    match fs::read_to_string(&path) {
        Ok(json) => match serde_json::from_str::<CaptureManifest>(&json) {
            Ok(manifest) => NsightProvenance {
                binding: ProvenanceBindingState::Partial,
                manifest: Some(manifest),
                diagnostics: Vec::new(),
            },
            Err(error) => NsightProvenance {
                binding: ProvenanceBindingState::Partial,
                manifest: None,
                diagnostics: vec![format!("Invalid {}: {error}", path.display())],
            },
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => NsightProvenance {
            binding: ProvenanceBindingState::Partial,
            manifest: None,
            diagnostics: vec![format!("{} is absent", path.display())],
        },
        Err(error) => NsightProvenance {
            binding: ProvenanceBindingState::Partial,
            manifest: None,
            diagnostics: vec![format!("Could not read {}: {error}", path.display())],
        },
    }
}

fn validate_provenance(
    dir: &Path,
    required_application_labels: &[String],
    gpu_expected_semantic_keys: &[String],
    cpu_only_semantic_keys: &[String],
    csv: &[NsightArtifact],
    raw_report: Option<&NsightArtifact>,
    provenance: &mut NsightProvenance,
) {
    let Some(manifest) = provenance.manifest.as_ref() else {
        return;
    };
    let mut mismatches = Vec::new();
    if manifest.schema != CAPTURE_MANIFEST_SCHEMA {
        mismatches.push(format!(
            "manifest schema `{}` is unsupported; expected `{CAPTURE_MANIFEST_SCHEMA}`",
            manifest.schema
        ));
    }
    if raw_report.is_none() {
        mismatches.push("a bound capture requires a retained raw .nsys-rep artifact".into());
    }
    let mut retained = csv
        .iter()
        .chain(raw_report)
        .cloned()
        .map(|artifact| (artifact.path.clone(), artifact))
        .collect::<HashMap<_, _>>();
    let mut declared = BTreeSet::new();

    let manifest_labels = sorted_unique(manifest.required_semantic_labels.iter().cloned());
    if manifest_labels.len() != manifest.required_semantic_labels.len() {
        mismatches.push("manifest required application labels contain duplicates".into());
    }
    let expected_labels = sorted_unique(required_application_labels.iter().cloned());
    if manifest_labels != expected_labels {
        mismatches.push(format!(
            "manifest required application labels do not match the application expectation: expected {expected_labels:?}, declared {manifest_labels:?}"
        ));
    }
    let manifest_uses_legacy_semantics = manifest.gpu_expected_semantic_labels.is_empty()
        && manifest.cpu_only_semantic_labels.is_empty();
    let manifest_gpu_labels = if manifest_uses_legacy_semantics {
        manifest_labels.clone()
    } else {
        sorted_unique(manifest.gpu_expected_semantic_labels.iter().cloned())
    };
    let manifest_cpu_labels = sorted_unique(manifest.cpu_only_semantic_labels.iter().cloned());
    if !manifest_uses_legacy_semantics
        && manifest_gpu_labels.len() != manifest.gpu_expected_semantic_labels.len()
    {
        mismatches.push("manifest GPU-expected labels contain duplicates".into());
    }
    if manifest_cpu_labels.len() != manifest.cpu_only_semantic_labels.len() {
        mismatches.push("manifest CPU-only labels contain duplicates".into());
    }
    let expected_gpu_labels = sorted_unique(gpu_expected_semantic_keys.iter().cloned());
    let expected_cpu_labels = sorted_unique(cpu_only_semantic_keys.iter().cloned());
    if manifest_gpu_labels != expected_gpu_labels {
        mismatches.push(format!(
            "manifest GPU-expected labels do not match the application expectation: expected {expected_gpu_labels:?}, declared {manifest_gpu_labels:?}"
        ));
    }
    if manifest_cpu_labels != expected_cpu_labels {
        mismatches.push(format!(
            "manifest CPU-only labels do not match the application expectation: expected {expected_cpu_labels:?}, declared {manifest_cpu_labels:?}"
        ));
    }

    for artifact in &manifest.artifacts {
        if !declared.insert(artifact.path.as_str()) {
            mismatches.push(format!(
                "manifest declares `{}` more than once",
                artifact.path
            ));
            continue;
        }
        if !is_safe_relative_path(&artifact.path) {
            mismatches.push(format!(
                "manifest artifact path `{}` is not a safe relative path",
                artifact.path
            ));
            continue;
        }
        if !retained.contains_key(&artifact.path) {
            let path = dir.join(&artifact.path);
            if path.is_file() {
                match artifact_metadata(dir, &path) {
                    Ok(metadata) => {
                        retained.insert(artifact.path.clone(), metadata);
                    }
                    Err(error) => mismatches.push(format!(
                        "manifest artifact `{}` could not be hashed: {error}",
                        artifact.path
                    )),
                }
            }
        }
        let actual = retained.get(&artifact.path);
        match actual {
            None => mismatches.push(format!("manifest artifact `{}` is missing", artifact.path)),
            Some(actual) => {
                if actual.size_bytes != artifact.size_bytes {
                    mismatches.push(format!(
                        "manifest artifact `{}` size mismatch: expected {}, observed {}",
                        artifact.path, artifact.size_bytes, actual.size_bytes
                    ));
                }
                if !actual.sha256.eq_ignore_ascii_case(&artifact.sha256) {
                    mismatches.push(format!(
                        "manifest artifact `{}` SHA-256 mismatch",
                        artifact.path
                    ));
                }
            }
        }
    }
    for artifact in csv.iter().chain(raw_report) {
        if !declared.contains(artifact.path.as_str()) {
            mismatches.push(format!(
                "retained artifact `{}` is absent from the manifest",
                artifact.path
            ));
        }
    }
    if !mismatches.is_empty() {
        provenance.binding = ProvenanceBindingState::Mismatch;
        provenance.diagnostics.extend(mismatches);
    }
}

fn is_safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn artifact_metadata(root: &Path, path: &Path) -> anyhow::Result<NsightArtifact> {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let mut file = fs::File::open(path)?;
    let size_bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(NsightArtifact {
        path: relative.to_string_lossy().replace('\\', "/"),
        size_bytes,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn build_correlation(
    expected: &[String],
    cpu_only: &[String],
    ranges: &[NsightTimelineRow],
    gpu_timeline: &[NsightTimelineRow],
) -> NsightCorrelation {
    if ranges.is_empty() {
        let mut correlation = empty_correlation();
        correlation.ledger.expected = sorted_unique(expected.iter().cloned());
        correlation.ledger.cpu_only = sorted_unique(cpu_only.iter().cloned());
        correlation.ledger.missing_expected = correlation.ledger.expected.clone();
        return correlation;
    }

    let expected = sorted_unique(expected.iter().cloned());
    let cpu_only = sorted_unique(cpu_only.iter().cloned());
    let mut counts = BTreeMap::<String, usize>::new();
    for key in ranges.iter().filter_map(|row| row.semantic_key.as_ref()) {
        *counts.entry(key.clone()).or_default() += 1;
    }
    let observed = counts.keys().cloned().collect::<Vec<_>>();
    let expected_set = expected.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let cpu_only_set = cpu_only.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let observed_set = observed.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let matched = expected_set
        .intersection(&observed_set)
        .map(|key| (*key).to_string())
        .collect();
    let missing_expected = expected_set
        .difference(&observed_set)
        .map(|key| (*key).to_string())
        .collect();
    let known_application_set = expected_set
        .union(&cpu_only_set)
        .copied()
        .collect::<BTreeSet<_>>();
    let unexpected_observed = observed_set
        .difference(&known_application_set)
        .map(|key| (*key).to_string())
        .collect();
    let unexpected_cpu_only = observed_set
        .intersection(&cpu_only_set)
        .map(|key| (*key).to_string())
        .collect();
    let duplicates = counts
        .into_iter()
        .filter(|(_, occurrences)| *occurrences > 1)
        .map(|(semantic_key, occurrences)| CorrelationDuplicate {
            semantic_key,
            occurrences,
        })
        .collect::<Vec<_>>();
    let mut joinable_count_checks = 0_usize;
    let mut unjoined_projection_rows = 0_usize;
    let all_ranges_qualified = ranges
        .iter()
        .filter(|row| {
            row.semantic_key
                .as_ref()
                .is_some_and(|key| expected_set.contains(key.as_str()))
        })
        .all(|row| match row.gpu_operations {
            Some(0) => false,
            Some(expected_count)
                if !gpu_timeline.is_empty() && !phase_join_keys(row).is_empty() =>
            {
                joinable_count_checks += 1;
                matching_gpu_operations(row, gpu_timeline)
                    .is_some_and(|operations| operations.len() as u64 == expected_count)
            }
            _ => {
                unjoined_projection_rows += 1;
                true
            }
        });
    let ledger = NsightCorrelationLedger {
        expected,
        cpu_only,
        observed,
        matched,
        missing_expected,
        unexpected_observed,
        unexpected_cpu_only,
        duplicates,
    };
    let complete = !ledger.expected.is_empty()
        && ledger.missing_expected.is_empty()
        && ledger.unexpected_observed.is_empty()
        && ledger.unexpected_cpu_only.is_empty()
        && ledger.duplicates.is_empty()
        && all_ranges_qualified;
    let reason = if complete {
        format!(
            "GPU-expected and observed semantic labels match exactly; {joinable_count_checks} joinable GPU operation count checks passed and {unjoined_projection_rows} projected rows rely on the official projection report; Candle and Nsight clocks remain separate"
        )
    } else {
        format!(
            "Correlation is incomplete: {} missing GPU-expected, {} unexpected observed, {} CPU-only projected, {} duplicate labels, joinable GPU operation counts complete={all_ranges_qualified}; clocks remain separate",
            ledger.missing_expected.len(),
            ledger.unexpected_observed.len(),
            ledger.unexpected_cpu_only.len(),
            ledger.duplicates.len()
        )
    };
    NsightCorrelation {
        mode: "nvtx_projected_range".into(),
        clock_aligned: false,
        complete,
        ledger,
        reason: Some(reason),
    }
}

fn sorted_unique(values: impl Iterator<Item = String>) -> Vec<String> {
    values.collect::<BTreeSet<_>>().into_iter().collect()
}

fn attribute_gpu_phases(
    ranges: &[NsightTimelineRow],
    gpu_timeline: &[NsightTimelineRow],
) -> Vec<PhaseGpuAttribution> {
    ranges
        .iter()
        .filter_map(|range| {
            let semantic_key = range.semantic_key.clone()?;
            let start = range.projected_start_ns?;
            let duration = range.projected_duration_ns?;
            let end = start.saturating_add(duration);
            let operations = matching_gpu_operations(range, gpu_timeline)?;
            if range.gpu_operations? != operations.len() as u64 {
                return None;
            }
            let mut intersections = Vec::new();
            for operation in &operations {
                let operation_end = operation.start_ns.saturating_add(operation.duration_ns);
                let overlap_start = start.max(operation.start_ns);
                let overlap_end = end.min(operation_end);
                if overlap_start < overlap_end {
                    intersections.push((overlap_start, overlap_end));
                }
            }
            Some(PhaseGpuAttribution {
                semantic_key,
                clock_aligned: false,
                projected_start_ns: start,
                projected_duration_ns: duration,
                gpu_busy_ns: interval_union_duration(&mut intersections),
                gpu_operation_count: operations.len(),
                join_keys: phase_join_keys(range),
            })
        })
        .collect()
}

fn matching_gpu_operations<'a>(
    range: &NsightTimelineRow,
    gpu_timeline: &'a [NsightTimelineRow],
) -> Option<Vec<&'a NsightTimelineRow>> {
    let join_keys = phase_join_keys(range);
    if join_keys.is_empty() {
        return None;
    }
    let start = range.projected_start_ns?;
    let end = start.saturating_add(range.projected_duration_ns?);
    Some(
        gpu_timeline
            .iter()
            .filter(|operation| {
                range
                    .correlation_id
                    .as_ref()
                    .is_none_or(|value| operation.correlation_id.as_ref() == Some(value))
                    && range
                        .device
                        .as_ref()
                        .is_none_or(|value| operation.device.as_ref() == Some(value))
                    && range
                        .context
                        .as_ref()
                        .is_none_or(|value| operation.context.as_ref() == Some(value))
                    && range
                        .stream
                        .as_ref()
                        .is_none_or(|value| operation.stream.as_ref() == Some(value))
                    && operation.start_ns < end
                    && operation.start_ns.saturating_add(operation.duration_ns) > start
            })
            .collect(),
    )
}

fn phase_join_keys(range: &NsightTimelineRow) -> Vec<String> {
    [
        ("correlation_id", range.correlation_id.as_ref()),
        ("device", range.device.as_ref()),
        ("context", range.context.as_ref()),
        ("stream", range.stream.as_ref()),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|_| name.to_string()))
    .collect()
}

fn interval_union_duration(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let mut total = 0_u64;
    let mut current: Option<(u64, u64)> = None;
    for &(start, end) in intervals.iter() {
        current = match current {
            None => Some((start, end)),
            Some((current_start, current_end)) if start <= current_end => {
                Some((current_start, current_end.max(end)))
            }
            Some((current_start, current_end)) => {
                total = total.saturating_add(current_end.saturating_sub(current_start));
                Some((start, end))
            }
        };
    }
    if let Some((start, end)) = current {
        total = total.saturating_add(end.saturating_sub(start));
    }
    total
}

fn parse_summary(path: &Path) -> anyhow::Result<Vec<NsightSummaryRow>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let headers = normalized_headers(reader.headers()?);
    anyhow::ensure!(
        has_header(&headers, &["name", "operation", "range", "kernel_name"]),
        "missing operation/name column"
    );
    anyhow::ensure!(
        has_header(&headers, &["total_time_ns", "total_ns"]),
        "missing total-time nanoseconds column"
    );
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        let name = field(
            &headers,
            &record,
            &["name", "operation", "range", "kernel_name"],
        )
        .unwrap_or_default()
        .trim()
        .to_string();
        if name.is_empty() {
            continue;
        }
        rows.push(NsightSummaryRow {
            name,
            total_ns: required_number(
                field(&headers, &record, &["total_time_ns", "total_ns"]),
                "total_ns",
            )?,
            count: number(
                field(
                    &headers,
                    &record,
                    &["instances", "num_calls", "operations", "count"],
                ),
                "count",
            )?,
            average_ns: number(
                field(&headers, &record, &["avg_ns", "average_ns"]),
                "average_ns",
            )?,
            minimum_ns: number(
                field(&headers, &record, &["min_ns", "minimum_ns"]),
                "minimum_ns",
            )?,
            maximum_ns: number(
                field(&headers, &record, &["max_ns", "maximum_ns"]),
                "maximum_ns",
            )?,
            category: field(&headers, &record, &["category"]).map(str::to_string),
        });
    }
    rows.sort_by_key(|row| std::cmp::Reverse(row.total_ns));
    Ok(rows)
}

fn parse_timeline(
    path: &Path,
    kind: &str,
    projected: bool,
) -> anyhow::Result<Vec<NsightTimelineRow>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let headers = normalized_headers(reader.headers()?);
    anyhow::ensure!(
        has_header(&headers, &["name", "operation", "range", "kernel_name"]),
        "missing operation/name column"
    );
    let start_headers: &[&str] = if projected {
        &["orig_start_ns", "orig_start", "start_ns", "start"]
    } else {
        &["start_ns", "start"]
    };
    let duration_headers: &[&str] = if projected {
        &[
            "orig_duration_ns",
            "orig_duration",
            "duration_ns",
            "duration",
            "dur_ns",
        ]
    } else {
        &["duration_ns", "duration", "dur_ns"]
    };
    anyhow::ensure!(
        has_header(&headers, start_headers) && has_header(&headers, duration_headers),
        "missing start/duration nanoseconds columns"
    );
    if projected {
        anyhow::ensure!(
            has_header(
                &headers,
                &["projected_start_ns", "projected_start", "proj_start_ns"]
            ) && has_header(
                &headers,
                &["projected_duration_ns", "projected_duration", "proj_dur_ns"]
            ),
            "missing projected start/duration nanoseconds columns"
        );
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        let name = field(
            &headers,
            &record,
            &["name", "operation", "range", "kernel_name"],
        )
        .unwrap_or_default()
        .trim()
        .to_string();
        if name.is_empty() {
            continue;
        }
        let start_ns = required_number(field(&headers, &record, start_headers), "start_ns")?;
        let duration_ns =
            required_number(field(&headers, &record, duration_headers), "duration_ns")?;
        anyhow::ensure!(duration_ns > 0, "timeline row `{name}` has zero duration");
        let projected_start_ns = optional_number(
            field(
                &headers,
                &record,
                &["projected_start_ns", "projected_start", "proj_start_ns"],
            ),
            "projected_start_ns",
        )?;
        let projected_duration_ns = optional_number(
            field(
                &headers,
                &record,
                &["projected_duration_ns", "projected_duration", "proj_dur_ns"],
            ),
            "projected_duration_ns",
        )?;
        if projected {
            anyhow::ensure!(
                projected_start_ns.is_some()
                    && projected_duration_ns.is_some_and(|duration| duration > 0),
                "projected timeline row `{name}` has invalid projected timing"
            );
        }
        rows.push(NsightTimelineRow {
            semantic_key: (kind == "nvtx_range").then(|| name.clone()),
            name,
            kind: kind.into(),
            device: field(&headers, &record, &["device", "device_id"]).map(str::to_string),
            context: field(&headers, &record, &["context", "context_id", "ctx"])
                .map(str::to_string),
            stream: field(&headers, &record, &["stream", "stream_id", "strm"]).map(str::to_string),
            correlation_id: field(&headers, &record, &["correlation_id", "corrid", "corr_id"])
                .map(str::to_string),
            start_ns,
            duration_ns,
            projected_start_ns,
            projected_duration_ns,
            gpu_operations: optional_number(
                field(
                    &headers,
                    &record,
                    &["num_gpu_ops", "numgpuops", "gpu_operations"],
                ),
                "gpu_operations",
            )?,
        });
    }
    rows.sort_by_key(|row| row.start_ns);
    Ok(rows)
}

fn truncate_rows<T>(mut rows: Vec<T>, limit: usize) -> (Vec<T>, ReportLimit) {
    let report_limit = report_limit(rows.len(), limit);
    rows.truncate(limit);
    (rows, report_limit)
}

fn report_limit(total_rows: usize, limit: usize) -> ReportLimit {
    ReportLimit {
        total_rows,
        displayed_rows: total_rows.min(limit),
        truncated: total_rows > limit,
    }
}

fn has_header(headers: &[String], names: &[&str]) -> bool {
    names
        .iter()
        .any(|name| headers.iter().any(|header| header == name))
}

fn normalized_headers(headers: &csv::StringRecord) -> Vec<String> {
    headers
        .iter()
        .map(|header| {
            header
                .trim()
                .to_ascii_lowercase()
                .replace(['(', ')', '%'], "")
                .replace([' ', '-', '/'], "_")
                .trim_matches('_')
                .to_string()
        })
        .collect()
}

fn field<'a>(headers: &[String], record: &'a csv::StringRecord, names: &[&str]) -> Option<&'a str> {
    names.iter().find_map(|name| {
        headers
            .iter()
            .position(|header| header == name)
            .and_then(|index| record.get(index))
    })
}

fn number(value: Option<&str>, label: &str) -> anyhow::Result<u64> {
    Ok(optional_number(value, label)?.unwrap_or(0))
}

fn required_number(value: Option<&str>, label: &str) -> anyhow::Result<u64> {
    optional_number(value, label)?.ok_or_else(|| anyhow::anyhow!("missing required {label} value"))
}

fn optional_number(value: Option<&str>, label: &str) -> anyhow::Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let cleaned = value.trim().replace(',', "");
    if cleaned.is_empty() {
        return Ok(None);
    }
    if let Ok(value) = cleaned.parse::<u64>() {
        return Ok(Some(value));
    }
    let value = cleaned
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid {label} value {cleaned:?}"))?;
    anyhow::ensure!(value.is_finite(), "non-finite {label} value {cleaned:?}");
    anyhow::ensure!(
        !value.is_sign_negative(),
        "negative {label} value {cleaned:?}"
    );
    // 2^64 is exactly representable as f64; every valid u64 converts to a smaller f64 value.
    anyhow::ensure!(
        value < 18_446_744_073_709_551_616.0,
        "overflowing {label} value {cleaned:?}"
    );
    Ok(Some(value as u64))
}

fn extension(path: &Path) -> &str {
    path.extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
}

fn status_reason(dir: &Path) -> Option<String> {
    let content = fs::read_to_string(dir.join("status.txt")).ok()?;
    content
        .lines()
        .find_map(|line| line.strip_prefix("reason=").map(str::to_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "candle-graph-nsys-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn normalizes_official_summary_headers_and_hashes_source() {
        let dir = temp_dir("summary");
        let path = dir.join("sample_cuda_gpu_kern_sum.csv");
        fs::write(&path, "Time (%),Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),Name\n50.0,1200,3,400,200,600,gemm\n").unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.status, GpuEvidenceStatus::Available);
        assert_eq!(evidence.kernels[0].name, "gemm");
        assert_eq!(evidence.kernels[0].total_ns, 1200);
        assert_eq!(evidence.source_csv[0].sha256.len(), 64);
        assert_eq!(
            evidence.source_csv[0].size_bytes,
            fs::metadata(path).unwrap().len()
        );
        assert_eq!(evidence.provenance.binding, ProvenanceBindingState::Partial);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn malformed_report_is_diagnostic_not_available() {
        let dir = temp_dir("bad");
        fs::write(dir.join("bad_cuda_api_sum.csv"), "Unknown,Value\nx,1\n").unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.status, GpuEvidenceStatus::Unavailable);
        assert!(!evidence.diagnostics.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn required_summary_numbers_reject_malformed_negative_non_finite_and_overflow() {
        let dir = temp_dir("strict-summary-numbers");
        let path = dir.join("sample_cuda_gpu_kern_sum.csv");
        for value in ["garbage", "-1", "NaN", "inf", "18446744073709551616"] {
            fs::write(
                &path,
                format!("Total Time (ns),Instances,Name\n{value},1,gemm\n"),
            )
            .unwrap();
            assert!(
                parse_summary(&path).is_err(),
                "summary value {value:?} must fail"
            );
        }
        fs::write(&path, "Total Time (ns),Instances,Name\n1,,gemm\n").unwrap();
        assert_eq!(parse_summary(&path).unwrap()[0].count, 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn optional_timeline_numbers_are_none_only_when_missing_and_fail_when_malformed() {
        let dir = temp_dir("strict-optional-numbers");
        let path = dir.join("sample_cuda_gpu_trace.csv");
        fs::write(&path, "Name,Start (ns),Duration (ns)\ngemm,1,2\n").unwrap();
        let row = parse_timeline(&path, "gpu_operation", false)
            .unwrap()
            .remove(0);
        assert_eq!(row.gpu_operations, None);
        assert_eq!(row.projected_start_ns, None);

        fs::write(
            &path,
            "Name,Start (ns),Duration (ns),Num GPU Ops\ngemm,1,2,NaN\n",
        )
        .unwrap();
        assert!(parse_timeline(&path, "gpu_operation", false).is_err());
        fs::write(
            &path,
            "Name,Start (ns),Duration (ns),Num GPU Ops\ngemm,1,2,-1\n",
        )
        .unwrap();
        assert!(parse_timeline(&path, "gpu_operation", false).is_err());
        fs::write(
            &path,
            "Name,Start (ns),Duration (ns),Num GPU Ops\ngemm,1,2,1e100\n",
        )
        .unwrap();
        assert!(parse_timeline(&path, "gpu_operation", false).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_official_nvtx_projection_columns_without_gpu_join_identifiers() {
        let dir = temp_dir("official-nvtx-projection");
        fs::write(
            dir.join("sample_nvtx_gpu_proj_trace.csv"),
            include_str!("../tests/fixtures/nsight/nvtx_gpu_proj_trace.csv"),
        )
        .unwrap();
        let required = vec!["pipeline/gpu".into(), "pipeline/prepare".into()];
        let gpu_expected = vec!["pipeline/gpu".into()];
        let cpu_only = vec!["pipeline/prepare".into()];
        let evidence =
            NsightEvidence::load_with_semantic_contract(&dir, &required, &gpu_expected, &cpu_only)
                .unwrap();

        assert_eq!(evidence.status, GpuEvidenceStatus::Available);
        assert!(evidence.correlation.complete);
        assert_eq!(evidence.correlation.ledger.expected, gpu_expected);
        assert_eq!(evidence.correlation.ledger.cpu_only, cpu_only);
        assert!(evidence.correlation.ledger.missing_expected.is_empty());
        assert!(evidence.correlation.ledger.unexpected_cpu_only.is_empty());
        assert_eq!(evidence.nvtx_ranges[0].start_ns, 700);
        assert_eq!(evidence.nvtx_ranges[0].duration_ns, 600);
        assert_eq!(evidence.nvtx_ranges[0].projected_start_ns, Some(1_000));
        assert_eq!(evidence.nvtx_ranges[0].projected_duration_ns, Some(250));
        assert_eq!(evidence.nvtx_ranges[0].gpu_operations, Some(3));
        assert_eq!(evidence.nvtx_ranges[0].correlation_id, None);
        assert_eq!(evidence.nvtx_ranges[0].device, None);
        assert!(evidence.phase_attribution.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn recognizes_official_cuda_context_and_stream_aliases() {
        let dir = temp_dir("official-cuda-aliases");
        let path = dir.join("sample_cuda_gpu_trace.csv");
        fs::write(
            &path,
            include_str!("../tests/fixtures/nsight/cuda_gpu_trace.csv"),
        )
        .unwrap();

        let row = parse_timeline(&path, "gpu_operation", false)
            .unwrap()
            .remove(0);
        assert_eq!(row.context.as_deref(), Some("7"));
        assert_eq!(row.stream.as_deref(), Some("13"));
        assert_eq!(row.correlation_id.as_deref(), Some("41"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn legacy_capture_manifest_deserializes_with_all_required_labels_gpu_expected() {
        let json = serde_json::json!({
            "schema": CAPTURE_MANIFEST_SCHEMA,
            "run": { "id": "run-1" },
            "correlation": { "id": "update-1" },
            "tool": { "name": "nsys", "version": "test" },
            "commands": ["nsys profile app"],
            "hardware": {},
            "source_revisions": {},
            "required_semantic_labels": ["pipeline/gpu"],
            "artifacts": []
        });
        let manifest: CaptureManifest = serde_json::from_value(json).unwrap();
        assert!(manifest.gpu_expected_semantic_labels.is_empty());
        assert!(manifest.cpu_only_semantic_labels.is_empty());
        assert_eq!(
            manifest.required_semantic_labels,
            vec!["pipeline/gpu".to_string()]
        );
    }

    #[test]
    fn gpu_projection_of_explicit_cpu_only_span_is_reported_and_incomplete() {
        let dir = temp_dir("cpu-only-projected");
        let mut csv = include_str!("../tests/fixtures/nsight/nvtx_gpu_proj_trace.csv").to_string();
        csv.push_str("pipeline/prepare,1300,40,1260,90,Push/Pop,4242,17,1,0,0,2,0,2\n");
        fs::write(dir.join("sample_nvtx_gpu_proj_trace.csv"), csv).unwrap();
        let required = vec!["pipeline/gpu".into(), "pipeline/prepare".into()];
        let evidence = NsightEvidence::load_with_semantic_contract(
            &dir,
            &required,
            &["pipeline/gpu".into()],
            &["pipeline/prepare".into()],
        )
        .unwrap();

        assert!(!evidence.correlation.complete);
        assert_eq!(
            evidence.correlation.ledger.unexpected_cpu_only,
            vec!["pipeline/prepare".to_string()]
        );
        assert!(evidence.correlation.ledger.unexpected_observed.is_empty());
        assert!(evidence
            .correlation
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("1 CPU-only projected")));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn correlation_reports_missing_expected_labels() {
        let dir = temp_dir("missing-label");
        fs::write(
            dir.join("sample_nvtx_gpu_proj_trace.csv"),
            "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops\nrun/forward,1,10,2,8,3\n",
        )
        .unwrap();
        let evidence =
            NsightEvidence::load(&dir, &["run/forward".into(), "run/backward".into()]).unwrap();
        assert!(!evidence.correlation.complete);
        assert_eq!(
            evidence.correlation.ledger.missing_expected,
            vec!["run/backward"]
        );
        assert_eq!(evidence.correlation.ledger.matched, vec!["run/forward"]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn correlation_completeness_is_computed_before_display_truncation() {
        let dir = temp_dir("pre-truncation");
        let mut csv = String::from(
            "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops,CorrId\n",
        );
        let mut gpu_csv = String::from("Name,Start (ns),Duration (ns),CorrId\n");
        let mut expected = Vec::new();
        for index in 0..=TIMELINE_DISPLAY_LIMIT {
            let label = format!("phase/{index}");
            expected.push(label.clone());
            csv.push_str(&format!("{label},{index},1,{index},1,1,{index}\n"));
            gpu_csv.push_str(&format!("kernel/{index},{index},1,{index}\n"));
        }
        fs::write(dir.join("many_nvtx_gpu_proj_trace.csv"), csv).unwrap();
        fs::write(dir.join("many_cuda_gpu_trace.csv"), gpu_csv).unwrap();
        let evidence = NsightEvidence::load(&dir, &expected).unwrap();
        assert_eq!(evidence.nvtx_ranges.len(), TIMELINE_DISPLAY_LIMIT);
        assert_eq!(evidence.nvtx_ranges.first().unwrap().name, "phase/0");
        assert_eq!(
            evidence.nvtx_ranges.last().unwrap().name,
            format!("phase/{}", TIMELINE_DISPLAY_LIMIT - 1)
        );
        let omitted_label = format!("phase/{TIMELINE_DISPLAY_LIMIT}");
        assert!(!evidence
            .nvtx_ranges
            .iter()
            .any(|row| row.name == omitted_label));
        assert_eq!(
            evidence.correlation.ledger.observed.len(),
            TIMELINE_DISPLAY_LIMIT + 1
        );
        assert!(evidence.correlation.complete);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn manifest_hash_mismatch_does_not_discard_gpu_evidence() {
        let dir = temp_dir("manifest-mismatch");
        let csv_path = dir.join("sample_cuda_gpu_kern_sum.csv");
        fs::write(&csv_path, "Total Time (ns),Instances,Name\n1200,3,gemm\n").unwrap();
        let manifest = CaptureManifest {
            schema: CAPTURE_MANIFEST_SCHEMA.into(),
            run: CaptureRun {
                id: "run-1".into(),
                started_at: None,
            },
            correlation: CaptureCorrelation {
                id: "update-1".into(),
            },
            tool: CaptureTool {
                name: "nsys".into(),
                version: "test".into(),
            },
            commands: vec!["nsys profile app".into()],
            hardware: CaptureHardware::default(),
            source_revisions: BTreeMap::from([("app".into(), "abc123".into())]),
            required_semantic_labels: Vec::new(),
            gpu_expected_semantic_labels: Vec::new(),
            cpu_only_semantic_labels: Vec::new(),
            artifacts: vec![ManifestArtifact {
                path: "sample_cuda_gpu_kern_sum.csv".into(),
                size_bytes: fs::metadata(&csv_path).unwrap().len(),
                sha256: "0".repeat(64),
            }],
        };
        fs::write(
            dir.join(CAPTURE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.status, GpuEvidenceStatus::Available);
        assert_eq!(
            evidence.provenance.binding,
            ProvenanceBindingState::Mismatch
        );
        assert!(evidence
            .provenance
            .diagnostics
            .iter()
            .any(|message| message.contains("SHA-256 mismatch")));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn valid_manifest_becomes_bound_only_after_trace_ids_match() {
        let dir = temp_dir("manifest-bound");
        let raw_path = dir.join("capture.nsys-rep");
        let csv_path = dir.join("sample_cuda_gpu_kern_sum.csv");
        fs::write(&raw_path, b"raw report").unwrap();
        fs::write(&csv_path, "Total Time (ns),Instances,Name\n1200,3,gemm\n").unwrap();
        let artifacts = [&raw_path, &csv_path]
            .into_iter()
            .map(|path| {
                let artifact = artifact_metadata(&dir, path).unwrap();
                ManifestArtifact {
                    path: artifact.path,
                    size_bytes: artifact.size_bytes,
                    sha256: artifact.sha256,
                }
            })
            .collect();
        let manifest = CaptureManifest {
            schema: CAPTURE_MANIFEST_SCHEMA.into(),
            run: CaptureRun {
                id: "run-1".into(),
                started_at: None,
            },
            correlation: CaptureCorrelation {
                id: "update-1".into(),
            },
            tool: CaptureTool {
                name: "nsys".into(),
                version: "test".into(),
            },
            commands: vec!["nsys profile app".into()],
            hardware: CaptureHardware::default(),
            source_revisions: BTreeMap::new(),
            required_semantic_labels: vec![],
            gpu_expected_semantic_labels: Vec::new(),
            cpu_only_semantic_labels: Vec::new(),
            artifacts,
        };
        fs::write(
            dir.join(CAPTURE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let mut evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.provenance.binding, ProvenanceBindingState::Partial);
        evidence.bind_to_trace("run-1", "update-1");
        assert_eq!(evidence.provenance.binding, ProvenanceBindingState::Bound);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn overlapping_gpu_operations_are_union_attributed_to_each_phase() {
        let dir = temp_dir("phase-overlap");
        fs::write(
            dir.join("sample_nvtx_gpu_proj_trace.csv"),
            "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops,CorrId\nphase/a,1,100,0,100,2,1\nphase/b,2,50,25,50,2,2\n",
        )
        .unwrap();
        fs::write(
            dir.join("sample_cuda_gpu_trace.csv"),
            "Name,Start (ns),Duration (ns),Device,Stream,CorrId\nkernel/a1,0,70,0,1,1\nkernel/a2,40,60,0,2,1\nkernel/b1,0,70,0,1,2\nkernel/b2,40,60,0,2,2\nunrelated,30,20,0,3,3\n",
        )
        .unwrap();
        let evidence = NsightEvidence::load(&dir, &["phase/a".into(), "phase/b".into()]).unwrap();
        assert!(!evidence.correlation.clock_aligned);
        assert!(!evidence.phase_attribution[0].clock_aligned);
        assert_eq!(evidence.phase_attribution[0].gpu_busy_ns, 100);
        assert_eq!(evidence.phase_attribution[0].gpu_operation_count, 2);
        assert_eq!(evidence.phase_attribution[1].gpu_busy_ns, 50);
        assert_eq!(evidence.phase_attribution[1].gpu_operation_count, 2);
        assert!(evidence
            .phase_attribution
            .iter()
            .all(|phase| { phase.join_keys == vec!["correlation_id"] }));
        let _ = fs::remove_dir_all(dir);
    }
}
