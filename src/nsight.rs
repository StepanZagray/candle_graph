//! Stable normalization seam for official `nsys stats --format csv` reports.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuEvidenceStatus {
    Available,
    Unavailable,
    Failed,
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
pub struct NsightCorrelation {
    pub mode: String,
    pub clock_aligned: bool,
    pub complete: bool,
    pub matched_ranges: usize,
    pub total_ranges: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportLimit {
    pub total_rows: usize,
    pub displayed_rows: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NsightEvidence {
    pub status: GpuEvidenceStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_report: Option<String>,
    #[serde(default)]
    pub source_csv: Vec<String>,
    pub coverage: NsightCoverage,
    pub correlation: NsightCorrelation,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub limits: std::collections::BTreeMap<String, ReportLimit>,
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
}

impl NsightEvidence {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: GpuEvidenceStatus::Unavailable,
            reason: Some(reason.into()),
            raw_report: None,
            source_csv: Vec::new(),
            coverage: NsightCoverage::default(),
            correlation: NsightCorrelation {
                mode: "none".into(),
                clock_aligned: false,
                complete: false,
                matched_ranges: 0,
                total_ranges: 0,
                reason: Some("No projected NVTX ranges were normalized".into()),
            },
            diagnostics: Vec::new(),
            limits: Default::default(),
            kernels: Vec::new(),
            runtime_calls: Vec::new(),
            memory_operations: Vec::new(),
            nvtx_ranges: Vec::new(),
            gpu_timeline: Vec::new(),
        }
    }

    /// Load a directory of official CSV reports without interpreting the unstable SQLite export.
    /// Parse failures become explicit GPU-evidence status and do not invalidate application trace.
    pub fn load_optional(dir: Option<&Path>, expected_semantic_keys: &[String]) -> Self {
        let Some(dir) = dir else {
            return Self::unavailable("Nsight capture was not requested");
        };
        match Self::load(dir, expected_semantic_keys) {
            Ok(evidence) => evidence,
            Err(error) => Self {
                status: GpuEvidenceStatus::Failed,
                reason: Some(error.to_string()),
                ..Self::unavailable("Nsight report normalization failed")
            },
        }
    }

    pub fn load(dir: &Path, expected_semantic_keys: &[String]) -> anyhow::Result<Self> {
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

        let raw_report = files
            .iter()
            .find(|path| extension(path) == "nsys-rep")
            .map(|path| path.display().to_string());
        let mut result = Self {
            status: GpuEvidenceStatus::Unavailable,
            reason: None,
            raw_report,
            source_csv: Vec::new(),
            coverage: NsightCoverage::default(),
            correlation: NsightCorrelation {
                mode: "none".into(),
                clock_aligned: false,
                complete: false,
                matched_ranges: 0,
                total_ranges: 0,
                reason: Some("No projected NVTX ranges were normalized".into()),
            },
            diagnostics: Vec::new(),
            limits: Default::default(),
            kernels: Vec::new(),
            runtime_calls: Vec::new(),
            memory_operations: Vec::new(),
            nvtx_ranges: Vec::new(),
            gpu_timeline: Vec::new(),
        };

        for path in files.iter().filter(|path| extension(path) == "csv") {
            let name = path
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or_default();
            let parsed: anyhow::Result<Option<ReportLimit>> = if name.contains("cuda_gpu_kern_sum")
            {
                parse_summary(path).map(|(rows, limit)| {
                    result.kernels.extend(rows);
                    result.coverage.kernel_summary = true;
                    Some(limit)
                })
            } else if name.contains("cuda_api_sum") {
                parse_summary(path).map(|(rows, limit)| {
                    result.runtime_calls.extend(rows);
                    result.coverage.runtime_summary = true;
                    Some(limit)
                })
            } else if name.contains("cuda_gpu_mem_time_sum") {
                parse_summary(path).map(|(rows, limit)| {
                    result.memory_operations.extend(rows);
                    result.coverage.memory_summary = true;
                    Some(limit)
                })
            } else if name.contains("nvtx_gpu_proj_trace") {
                parse_timeline(path, "nvtx_range", true).map(|(rows, limit)| {
                    result.nvtx_ranges.extend(rows);
                    result.coverage.nvtx_projection = true;
                    Some(limit)
                })
            } else if name.contains("cuda_gpu_trace") {
                parse_timeline(path, "gpu_operation", false).map(|(rows, limit)| {
                    result.gpu_timeline.extend(rows);
                    result.coverage.gpu_timeline = true;
                    Some(limit)
                })
            } else {
                Ok(None)
            };
            match parsed {
                Ok(Some(limit)) => {
                    result.source_csv.push(path.display().to_string());
                    result.limits.insert(name.to_string(), limit);
                }
                Ok(None) => {}
                Err(error) => result
                    .diagnostics
                    .push(format!("{}: {error}", path.display())),
            }
        }

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
        if result.coverage.nvtx_projection {
            let expected = expected_semantic_keys
                .iter()
                .collect::<std::collections::HashSet<_>>();
            let matched_ranges = result
                .nvtx_ranges
                .iter()
                .filter(|row| {
                    row.semantic_key
                        .as_ref()
                        .is_some_and(|key| expected.contains(key))
                })
                .count();
            let total_ranges = result.nvtx_ranges.len();
            let complete = total_ranges > 0
                && matched_ranges == total_ranges
                && result
                    .nvtx_ranges
                    .iter()
                    .all(|row| row.gpu_operations.is_some());
            result.correlation = NsightCorrelation {
                mode: "nvtx_projected_range".into(),
                clock_aligned: false,
                complete,
                matched_ranges,
                total_ranges,
                reason: Some(if complete {
                    "Every projected NVTX range matched an exact application semantic label; Candle and Nsight clocks remain separate".into()
                } else {
                    format!(
                        "{matched_ranges}/{total_ranges} projected NVTX ranges matched exact application semantic labels; clocks remain separate"
                    )
                }),
            };
        }
        Ok(result)
    }
}

fn parse_summary(path: &Path) -> anyhow::Result<(Vec<NsightSummaryRow>, ReportLimit)> {
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
            total_ns: number(field(&headers, &record, &["total_time_ns", "total_ns"])),
            count: number(field(
                &headers,
                &record,
                &["instances", "num_calls", "operations", "count"],
            )),
            average_ns: number(field(&headers, &record, &["avg_ns", "average_ns"])),
            minimum_ns: number(field(&headers, &record, &["min_ns", "minimum_ns"])),
            maximum_ns: number(field(&headers, &record, &["max_ns", "maximum_ns"])),
            category: field(&headers, &record, &["category"]).map(str::to_string),
        });
    }
    rows.sort_by_key(|row| std::cmp::Reverse(row.total_ns));
    let total_rows = rows.len();
    rows.truncate(100);
    Ok((
        rows,
        ReportLimit {
            total_rows,
            displayed_rows: total_rows.min(100),
            truncated: total_rows > 100,
        },
    ))
}

fn parse_timeline(
    path: &Path,
    kind: &str,
    projected: bool,
) -> anyhow::Result<(Vec<NsightTimelineRow>, ReportLimit)> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let headers = normalized_headers(reader.headers()?);
    anyhow::ensure!(
        has_header(&headers, &["name", "operation", "range", "kernel_name"]),
        "missing operation/name column"
    );
    anyhow::ensure!(
        has_header(&headers, &["start_ns", "start"])
            && has_header(&headers, &["duration_ns", "duration", "dur_ns"]),
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
        let start_ns =
            required_number(field(&headers, &record, &["start_ns", "start"]), "start_ns")?;
        let duration_ns = required_number(
            field(&headers, &record, &["duration_ns", "duration", "dur_ns"]),
            "duration_ns",
        )?;
        anyhow::ensure!(duration_ns > 0, "timeline row `{name}` has zero duration");
        let projected_start_ns = optional_number(field(
            &headers,
            &record,
            &["projected_start_ns", "projected_start", "proj_start_ns"],
        ));
        let projected_duration_ns = optional_number(field(
            &headers,
            &record,
            &["projected_duration_ns", "projected_duration", "proj_dur_ns"],
        ));
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
            context: field(&headers, &record, &["context", "context_id"]).map(str::to_string),
            stream: field(&headers, &record, &["stream", "stream_id"]).map(str::to_string),
            correlation_id: field(&headers, &record, &["correlation_id", "corrid", "corr_id"])
                .map(str::to_string),
            start_ns,
            duration_ns,
            projected_start_ns,
            projected_duration_ns,
            gpu_operations: optional_number(field(
                &headers,
                &record,
                &["num_gpu_ops", "numgpuops", "gpu_operations"],
            )),
        });
    }
    rows.sort_by_key(|row| row.start_ns);
    let total_rows = rows.len();
    rows.truncate(500);
    Ok((
        rows,
        ReportLimit {
            total_rows,
            displayed_rows: total_rows.min(500),
            truncated: total_rows > 500,
        },
    ))
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

fn number(value: Option<&str>) -> u64 {
    optional_number(value).unwrap_or(0)
}

fn required_number(value: Option<&str>, label: &str) -> anyhow::Result<u64> {
    optional_number(value).ok_or_else(|| anyhow::anyhow!("invalid or missing {label} value"))
}

fn optional_number(value: Option<&str>) -> Option<u64> {
    let cleaned = value?.trim().replace(',', "");
    cleaned
        .parse::<u64>()
        .ok()
        .or_else(|| cleaned.parse::<f64>().ok().map(|x| x.max(0.0) as u64))
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

    #[test]
    fn normalizes_official_summary_headers() {
        let dir = std::env::temp_dir().join(format!("candle-graph-nsys-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample_cuda_gpu_kern_sum.csv");
        fs::write(&path, "Time (%),Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),Name\n50.0,1200,3,400,200,600,gemm\n").unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.status, GpuEvidenceStatus::Available);
        assert_eq!(evidence.kernels[0].name, "gemm");
        assert_eq!(evidence.kernels[0].total_ns, 1200);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn malformed_report_is_diagnostic_not_available() {
        let dir =
            std::env::temp_dir().join(format!("candle-graph-nsys-bad-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("bad_cuda_api_sum.csv"), "Unknown,Value\nx,1\n").unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        assert_eq!(evidence.status, GpuEvidenceStatus::Unavailable);
        assert!(!evidence.diagnostics.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn reports_when_rows_are_truncated() {
        let dir =
            std::env::temp_dir().join(format!("candle-graph-nsys-limit-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut csv = String::from("Total Time (ns),Instances,Name\n");
        for index in 0..101 {
            csv.push_str(&format!("{},1,kernel-{index}\n", index + 1));
        }
        fs::write(dir.join("many_cuda_gpu_kern_sum.csv"), csv).unwrap();
        let evidence = NsightEvidence::load(&dir, &[]).unwrap();
        let limit = evidence.limits.values().next().unwrap();
        assert_eq!(limit.total_rows, 101);
        assert_eq!(limit.displayed_rows, 100);
        assert!(limit.truncated);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn projected_ranges_require_timing_and_exact_semantic_join() {
        let dir =
            std::env::temp_dir().join(format!("candle-graph-nsys-join-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("sample_nvtx_gpu_proj_trace.csv"),
            "Name,Start (ns),Duration (ns),Projected Start (ns),Projected Duration (ns),Num GPU Ops\nrun/forward,1,10,2,8,3\n",
        )
        .unwrap();
        let evidence = NsightEvidence::load(&dir, &["run/forward".into()]).unwrap();
        assert!(evidence.correlation.complete);
        assert_eq!(evidence.correlation.matched_ranges, 1);

        fs::write(
            dir.join("sample_nvtx_gpu_proj_trace.csv"),
            "Name,Num GPU Ops\nrun/forward,3\n",
        )
        .unwrap();
        let invalid = NsightEvidence::load(&dir, &["run/forward".into()]).unwrap();
        assert_eq!(invalid.status, GpuEvidenceStatus::Unavailable);
        assert!(!invalid.diagnostics.is_empty());
        let _ = fs::remove_dir_all(dir);
    }
}
