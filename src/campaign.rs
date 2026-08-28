//! Application-neutral campaign/series layer over published evidence bundles.
//!
//! A *campaign* is a producer-declared plan of capture steps for one training
//! entrypoint. This module reconciles that plan against published,
//! content-verified evidence bundles ([`campaign_status`]) and assembles
//! long-form metric trajectories across verified bundles ([`build_series`]).
//! Everything here is stateless and read-only: commands read artifacts;
//! nothing supervises training.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifact::verify_bundle;
use crate::trace::schema::GradientState;
use crate::trace::{parse_trace, TraceDocument};

pub const CAMPAIGN_SCHEMA: &str = "candle-graph/campaign/1";
pub const CAMPAIGN_STATUS_SCHEMA: &str = "candle-graph/campaign-status/1";
pub const SERIES_SCHEMA: &str = "candle-graph/series/1";

/// One planned capture: a step coordinate and the bundle path expected to hold
/// its published evidence, relative to the campaign manifest's directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedCapture {
    pub capture_step: u64,
    pub bundle: String,
}

/// Producer-declared plan for one training campaign.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignManifest {
    pub schema: String,
    pub campaign_id: String,
    pub entrypoint: String,
    pub planned: Vec<PlannedCapture>,
}

impl CampaignManifest {
    /// Parse a campaign manifest from JSON on disk and validate it.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("read campaign manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse campaign manifest {}", path.display()))?;
        manifest
            .validate()
            .with_context(|| format!("validate campaign manifest {}", path.display()))?;
        Ok(manifest)
    }

    /// Reject manifests that could not be reconciled deterministically.
    pub fn validate(&self) -> Result<()> {
        if self.schema != CAMPAIGN_SCHEMA {
            bail!(
                "unsupported campaign schema {:?}; expected {CAMPAIGN_SCHEMA:?}",
                self.schema
            );
        }
        if self.campaign_id.trim().is_empty() {
            bail!("campaign manifest requires a non-empty campaign_id");
        }
        if self.entrypoint.trim().is_empty() {
            bail!("campaign manifest requires a non-empty entrypoint");
        }
        if self.planned.is_empty() {
            bail!("campaign manifest requires at least one planned capture");
        }
        let mut steps = BTreeSet::new();
        let mut bundles = BTreeSet::new();
        for capture in &self.planned {
            if !steps.insert(capture.capture_step) {
                bail!(
                    "campaign manifest plans duplicate capture_step {}",
                    capture.capture_step
                );
            }
            if !is_safe_relative_path(&capture.bundle) {
                bail!(
                    "campaign manifest has an unsafe bundle path {:?}; \
                     bundle paths must be non-empty relative paths without `..` or backslashes",
                    capture.bundle
                );
            }
            if !bundles.insert(capture.bundle.as_str()) {
                bail!(
                    "campaign manifest plans duplicate bundle path {:?}",
                    capture.bundle
                );
            }
        }
        Ok(())
    }
}

/// Reconciled state of one planned capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CaptureState {
    Missing,
    Published {
        run_id: String,
        manifest_sha256: String,
    },
    FailedRun {
        run_id: String,
        reason: Option<String>,
    },
    VerificationFailed {
        message: String,
    },
    IdentityMismatch {
        message: String,
    },
}

/// One planned capture joined with the state observed on the filesystem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureStatus {
    pub capture_step: u64,
    pub bundle: String,
    pub state: CaptureState,
}

/// Deterministic reconciliation of a campaign plan against published bundles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignStatus {
    pub schema: String,
    pub campaign_id: String,
    pub entrypoint: String,
    pub planned: usize,
    pub published: usize,
    pub missing: usize,
    pub failed: usize,
    pub captures: Vec<CaptureStatus>,
}

/// Reconcile every planned capture in the manifest at `manifest_path` against
/// the filesystem. Bundle paths resolve relative to the manifest's parent
/// directory; every planned capture appears exactly once in the result.
pub fn campaign_status(manifest_path: &Path) -> Result<CampaignStatus> {
    let manifest = CampaignManifest::load(manifest_path)?;
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));

    let mut captures = Vec::with_capacity(manifest.planned.len());
    for planned in &manifest.planned {
        let root = base.join(&planned.bundle);
        let state = reconcile_capture(&root, &manifest.entrypoint, planned.capture_step);
        captures.push(CaptureStatus {
            capture_step: planned.capture_step,
            bundle: planned.bundle.clone(),
            state,
        });
    }
    captures.sort_by_key(|capture| capture.capture_step);

    let published = captures
        .iter()
        .filter(|capture| matches!(capture.state, CaptureState::Published { .. }))
        .count();
    let missing = captures
        .iter()
        .filter(|capture| matches!(capture.state, CaptureState::Missing))
        .count();
    let failed = captures.len() - published - missing;

    Ok(CampaignStatus {
        schema: CAMPAIGN_STATUS_SCHEMA.into(),
        campaign_id: manifest.campaign_id,
        entrypoint: manifest.entrypoint,
        planned: captures.len(),
        published,
        missing,
        failed,
        captures,
    })
}

fn reconcile_capture(root: &Path, entrypoint: &str, capture_step: u64) -> CaptureState {
    if !root.exists() {
        return CaptureState::Missing;
    }
    let receipt = match verify_bundle(root) {
        Ok(receipt) => receipt,
        Err(error) => {
            return CaptureState::VerificationFailed {
                message: format!("{error:#}"),
            }
        }
    };
    let document = match parse_trace(root.join("trace.jsonl")) {
        Ok(document) => document,
        Err(error) => {
            return CaptureState::VerificationFailed {
                message: format!("{error:#}"),
            }
        }
    };
    if document.run.entrypoint != entrypoint {
        return CaptureState::IdentityMismatch {
            message: format!(
                "entrypoint mismatch: manifest declares {:?}, bundle trace observes {:?}",
                entrypoint, document.run.entrypoint
            ),
        };
    }
    if document.run.capture_step != capture_step {
        return CaptureState::IdentityMismatch {
            message: format!(
                "capture_step mismatch: manifest plans {}, bundle trace observes {}",
                capture_step, document.run.capture_step
            ),
        };
    }
    match document.terminal.outcome {
        crate::trace::RunOutcome::Failed => CaptureState::FailedRun {
            run_id: document.run.run_id,
            reason: document.terminal.reason,
        },
        crate::trace::RunOutcome::Complete => CaptureState::Published {
            run_id: document.run.run_id,
            manifest_sha256: receipt.manifest_sha256,
        },
    }
}

/// One verified bundle admitted into a series, keyed by its capture step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeriesInput {
    pub capture_step: u64,
    pub run_id: String,
    pub bundle: String,
    pub manifest_sha256: String,
}

/// Outer wall time of one run. `Some` only when the trace holds exactly one
/// measured, closed span; otherwise the coordinate is ambiguous and stays `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingSeriesPoint {
    pub capture_step: u64,
    pub run_id: String,
    pub outer_wall_time_ns: Option<u64>,
}

/// One tensor-statistics observation on the series coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarSeriesPoint {
    pub capture_step: u64,
    pub run_id: String,
    pub rms: f64,
    pub abs_max: f64,
    pub mean: f64,
    pub non_finite: u64,
    pub elements: u64,
}

/// One gradient observation on the series coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientSeriesPoint {
    pub capture_step: u64,
    pub run_id: String,
    pub state: GradientState,
    pub norm: Option<f64>,
}

/// Long-form metric trajectories across verified bundles of one entrypoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesReport {
    pub schema: String,
    pub entrypoint: String,
    pub coordinate: String,
    pub label_prefix: Option<String>,
    pub inputs: Vec<SeriesInput>,
    pub outer_wall_time: Vec<TimingSeriesPoint>,
    pub tensor_stats: BTreeMap<String, Vec<ScalarSeriesPoint>>,
    pub gradients: BTreeMap<String, Vec<GradientSeriesPoint>>,
}

/// Build metric trajectories across `bundle_roots`, sorted by capture step.
///
/// Every bundle must pass deep content verification and all traces must share
/// one entrypoint and phase; duplicate capture steps are rejected because the
/// series coordinate would be ambiguous. `label_prefix` filters tensor-stat
/// labels and gradient `{root}/{key}` composites; `None` keeps all.
pub fn build_series(bundle_roots: &[PathBuf], label_prefix: Option<&str>) -> Result<SeriesReport> {
    if bundle_roots.is_empty() {
        bail!("series requires at least one bundle");
    }

    struct SeriesEntry {
        bundle: String,
        manifest_sha256: String,
        document: TraceDocument,
    }

    let mut entries = Vec::with_capacity(bundle_roots.len());
    for root in bundle_roots {
        let receipt = verify_bundle(root)
            .with_context(|| format!("verify series bundle {}", root.display()))?;
        let document = parse_trace(root.join("trace.jsonl"))
            .with_context(|| format!("parse trace of series bundle {}", root.display()))?;
        entries.push(SeriesEntry {
            bundle: root.display().to_string(),
            manifest_sha256: receipt.manifest_sha256,
            document,
        });
    }

    let entrypoint = entries[0].document.run.entrypoint.clone();
    let phase = entries[0].document.run.phase;
    let mut seen_steps: BTreeMap<u64, &str> = BTreeMap::new();
    for entry in &entries {
        let run = &entry.document.run;
        if run.entrypoint != entrypoint {
            bail!(
                "series bundles mix entrypoints: {} observes {:?}, expected {:?} from {}",
                entry.bundle,
                run.entrypoint,
                entrypoint,
                entries[0].bundle
            );
        }
        if run.phase != phase {
            bail!(
                "series bundles mix phases: {} observes {:?}, expected {:?} from {}",
                entry.bundle,
                run.phase.as_str(),
                phase.as_str(),
                entries[0].bundle
            );
        }
        if let Some(existing) = seen_steps.insert(run.capture_step, entry.bundle.as_str()) {
            bail!(
                "ambiguous series coordinate: capture_step {} is provided by both {} and {}",
                run.capture_step,
                existing,
                entry.bundle
            );
        }
    }
    entries.sort_by_key(|entry| entry.document.run.capture_step);

    let matches_prefix = |value: &str| label_prefix.is_none_or(|prefix| value.starts_with(prefix));

    let mut inputs = Vec::with_capacity(entries.len());
    let mut outer_wall_time = Vec::with_capacity(entries.len());
    let mut tensor_stats: BTreeMap<String, Vec<ScalarSeriesPoint>> = BTreeMap::new();
    let mut gradients: BTreeMap<String, Vec<GradientSeriesPoint>> = BTreeMap::new();
    for entry in &entries {
        let run = &entry.document.run;
        inputs.push(SeriesInput {
            capture_step: run.capture_step,
            run_id: run.run_id.clone(),
            bundle: entry.bundle.clone(),
            manifest_sha256: entry.manifest_sha256.clone(),
        });

        let mut measured = entry
            .document
            .spans
            .iter()
            .filter(|span| span.measured && span.closed);
        let outer_wall_time_ns = match (measured.next(), measured.next()) {
            (Some(span), None) => Some(span.duration_ns),
            _ => None,
        };
        outer_wall_time.push(TimingSeriesPoint {
            capture_step: run.capture_step,
            run_id: run.run_id.clone(),
            outer_wall_time_ns,
        });

        for stats in &entry.document.tensor_stats {
            if !matches_prefix(&stats.label) {
                continue;
            }
            tensor_stats
                .entry(stats.label.clone())
                .or_default()
                .push(ScalarSeriesPoint {
                    capture_step: run.capture_step,
                    run_id: run.run_id.clone(),
                    rms: stats.rms,
                    abs_max: stats.abs_max,
                    mean: stats.mean,
                    non_finite: stats.non_finite,
                    elements: stats.elements,
                });
        }

        for gradient in &entry.document.gradients {
            let key = format!("{}/{}", gradient.root, gradient.key);
            if !matches_prefix(&key) {
                continue;
            }
            gradients.entry(key).or_default().push(GradientSeriesPoint {
                capture_step: run.capture_step,
                run_id: run.run_id.clone(),
                state: gradient.state,
                norm: gradient.norm,
            });
        }
    }

    Ok(SeriesReport {
        schema: SERIES_SCHEMA.into(),
        entrypoint,
        coordinate: "capture_step".into(),
        label_prefix: label_prefix.map(str::to_owned),
        inputs,
        outer_wall_time,
        tensor_stats,
        gradients,
    })
}

fn is_safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !value.contains('\\')
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::publish_bundle;
    use crate::capability::CaptureContract;
    use crate::trace::{
        write_jsonl, GradientEvent, RunOutcome, SpanKind, SpanRecord, TensorStatsEvent,
        TerminalEvent, TimingMode, TraceDocument, TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "candle-graph-campaign-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn fixture_document(run_id: &str, entrypoint: &str, capture_step: u64) -> TraceDocument {
        TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: run_id.into(),
                correlation_id: format!("campaign/{run_id}"),
                entrypoint: entrypoint.into(),
                phase: crate::ExecutionPhase::Train,
                timestamp: "2026-08-28T00:00:00Z".into(),
                capture_step,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: entrypoint.into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns: 10 + capture_step,
                step: None,
            }],
            ops: vec![],
            tensors: vec![],
            tensor_stats: vec![
                TensorStatsEvent {
                    span_id: "root".into(),
                    label: "loss/total".into(),
                    shape: vec![1],
                    dtype: "f32".into(),
                    elements: 1,
                    non_finite: 0,
                    rms: 2.0 + capture_step as f64,
                    abs_max: 2.0 + capture_step as f64,
                    mean: -(capture_step as f64),
                },
                TensorStatsEvent {
                    span_id: "root".into(),
                    label: "act/mlp".into(),
                    shape: vec![2, 3],
                    dtype: "f32".into(),
                    elements: 6,
                    non_finite: 0,
                    rms: 1.0,
                    abs_max: 3.0,
                    mean: 0.5,
                },
            ],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![
                GradientEvent {
                    event_id: format!("{run_id}-g1"),
                    root: "vb".into(),
                    key: "encoder.weight".into(),
                    state: GradientState::Present,
                    norm: Some(0.5),
                },
                GradientEvent {
                    event_id: format!("{run_id}-g2"),
                    root: "vb".into(),
                    key: "decoder.bias".into(),
                    state: GradientState::Zero,
                    norm: Some(0.0),
                },
            ],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 10 + capture_step,
                reason: None,
            },
        }
    }

    fn publish_fixture(root: &Path, name: &str, document: &TraceDocument) -> PathBuf {
        let trace = root.join(format!("{name}.input.jsonl"));
        write_jsonl(&trace, &document.to_events()).unwrap();
        let destination = root.join(name);
        publish_bundle(&destination, &trace, None).unwrap();
        fs::remove_file(trace).unwrap();
        destination
    }

    fn write_manifest(root: &Path, manifest: &CampaignManifest) -> PathBuf {
        let path = root.join("campaign.json");
        fs::write(&path, serde_json::to_string_pretty(manifest).unwrap()).unwrap();
        path
    }

    #[test]
    fn campaign_status_reconciles_every_planned_capture() {
        let root = temp_root("status");
        publish_fixture(
            &root,
            "b100",
            &fixture_document("run-100", "demo::train", 100),
        );
        publish_fixture(
            &root,
            "b200",
            &fixture_document("run-200", "demo::train", 200),
        );
        // Published under the planned path but captured at the wrong step.
        publish_fixture(
            &root,
            "b300",
            &fixture_document("run-999", "demo::train", 999),
        );
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.into(),
            campaign_id: "demo-campaign".into(),
            entrypoint: "demo::train".into(),
            planned: vec![
                PlannedCapture {
                    capture_step: 300,
                    bundle: "b300".into(),
                },
                PlannedCapture {
                    capture_step: 100,
                    bundle: "b100".into(),
                },
                PlannedCapture {
                    capture_step: 400,
                    bundle: "b400".into(),
                },
                PlannedCapture {
                    capture_step: 200,
                    bundle: "b200".into(),
                },
            ],
        };
        let manifest_path = write_manifest(&root, &manifest);

        let status = campaign_status(&manifest_path).unwrap();
        assert_eq!(status.schema, CAMPAIGN_STATUS_SCHEMA);
        assert_eq!(status.campaign_id, "demo-campaign");
        assert_eq!(status.entrypoint, "demo::train");
        assert_eq!(status.planned, 4);
        assert_eq!(status.published, 2);
        assert_eq!(status.missing, 1);
        assert_eq!(status.failed, 1);
        let steps: Vec<u64> = status
            .captures
            .iter()
            .map(|capture| capture.capture_step)
            .collect();
        assert_eq!(steps, vec![100, 200, 300, 400]);
        match &status.captures[0].state {
            CaptureState::Published {
                run_id,
                manifest_sha256,
            } => {
                assert_eq!(run_id, "run-100");
                assert_eq!(manifest_sha256.len(), 64);
            }
            other => panic!("expected published capture, got {other:?}"),
        }
        assert!(matches!(
            status.captures[1].state,
            CaptureState::Published { .. }
        ));
        match &status.captures[2].state {
            CaptureState::IdentityMismatch { message } => {
                assert!(message.contains("capture_step mismatch"));
                assert!(message.contains("300"));
                assert!(message.contains("999"));
            }
            other => panic!("expected identity mismatch, got {other:?}"),
        }
        assert_eq!(status.captures[3].state, CaptureState::Missing);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_validation_rejects_bad_plans() {
        let good = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.into(),
            campaign_id: "c".into(),
            entrypoint: "demo::train".into(),
            planned: vec![PlannedCapture {
                capture_step: 1,
                bundle: "bundles/b1".into(),
            }],
        };
        good.validate().unwrap();

        let mut wrong_schema = good.clone();
        wrong_schema.schema = "candle-graph/campaign/0".into();
        assert!(wrong_schema
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema"));

        let mut duplicate_steps = good.clone();
        duplicate_steps.planned = vec![
            PlannedCapture {
                capture_step: 1,
                bundle: "a".into(),
            },
            PlannedCapture {
                capture_step: 1,
                bundle: "b".into(),
            },
        ];
        assert!(duplicate_steps
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate capture_step"));

        let mut duplicate_bundles = good.clone();
        duplicate_bundles.planned = vec![
            PlannedCapture {
                capture_step: 1,
                bundle: "same".into(),
            },
            PlannedCapture {
                capture_step: 2,
                bundle: "same".into(),
            },
        ];
        assert!(duplicate_bundles
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate bundle path"));

        for unsafe_path in ["", "/abs/b1", "../escape", "a\\b", "a/../b"] {
            let mut manifest = good.clone();
            manifest.planned[0].bundle = unsafe_path.into();
            assert!(
                manifest
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("unsafe bundle path"),
                "path {unsafe_path:?} must be rejected"
            );
        }

        let mut empty_plan = good.clone();
        empty_plan.planned.clear();
        assert!(empty_plan
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one planned capture"));

        let mut empty_id = good;
        empty_id.campaign_id = "  ".into();
        assert!(empty_id
            .validate()
            .unwrap_err()
            .to_string()
            .contains("campaign_id"));
    }

    #[test]
    fn series_sorts_filters_and_rejects_ambiguity() {
        let root = temp_root("series");
        let b100 = publish_fixture(
            &root,
            "b100",
            &fixture_document("run-100", "demo::train", 100),
        );
        let b200 = publish_fixture(
            &root,
            "b200",
            &fixture_document("run-200", "demo::train", 200),
        );
        // Second measured+closed span makes the outer wall coordinate ambiguous.
        let mut two_measured = fixture_document("run-300", "demo::train", 300);
        two_measured.spans.push(SpanRecord {
            id: "root-b".into(),
            parent_id: None,
            name: "demo::train#b".into(),
            kind: SpanKind::Function,
            measured: true,
            start_ns: 0,
            closed: true,
            duration_ns: 5,
            step: None,
        });
        let b300 = publish_fixture(&root, "b300", &two_measured);

        // Bundles passed out of order; the report must sort by capture_step.
        let report = build_series(&[b300.clone(), b100.clone(), b200.clone()], None).unwrap();
        assert_eq!(report.schema, SERIES_SCHEMA);
        assert_eq!(report.entrypoint, "demo::train");
        assert_eq!(report.coordinate, "capture_step");
        assert_eq!(report.label_prefix, None);
        let input_steps: Vec<u64> = report
            .inputs
            .iter()
            .map(|input| input.capture_step)
            .collect();
        assert_eq!(input_steps, vec![100, 200, 300]);
        assert!(report
            .inputs
            .iter()
            .all(|input| input.manifest_sha256.len() == 64));
        assert_eq!(report.inputs[0].run_id, "run-100");
        assert_eq!(
            report
                .outer_wall_time
                .iter()
                .map(|point| point.outer_wall_time_ns)
                .collect::<Vec<_>>(),
            vec![Some(110), Some(210), None]
        );
        assert_eq!(
            report.tensor_stats.keys().collect::<Vec<_>>(),
            vec!["act/mlp", "loss/total"]
        );
        let loss = &report.tensor_stats["loss/total"];
        assert_eq!(
            loss.iter()
                .map(|point| point.capture_step)
                .collect::<Vec<_>>(),
            vec![100, 200, 300]
        );
        assert_eq!(loss[0].rms, 102.0);
        assert_eq!(
            report.gradients.keys().collect::<Vec<_>>(),
            vec!["vb/decoder.bias", "vb/encoder.weight"]
        );
        let encoder = &report.gradients["vb/encoder.weight"];
        assert_eq!(encoder.len(), 3);
        assert_eq!(encoder[0].state, GradientState::Present);
        assert_eq!(encoder[0].norm, Some(0.5));

        let filtered = build_series(&[b100.clone(), b200.clone()], Some("loss")).unwrap();
        assert_eq!(filtered.label_prefix.as_deref(), Some("loss"));
        assert_eq!(
            filtered.tensor_stats.keys().collect::<Vec<_>>(),
            vec!["loss/total"]
        );
        assert!(filtered.gradients.is_empty());

        let gradient_only = build_series(std::slice::from_ref(&b100), Some("vb/enc")).unwrap();
        assert!(gradient_only.tensor_stats.is_empty());
        assert_eq!(
            gradient_only.gradients.keys().collect::<Vec<_>>(),
            vec!["vb/encoder.weight"]
        );

        let duplicate = publish_fixture(
            &root,
            "b100-dup",
            &fixture_document("run-100-dup", "demo::train", 100),
        );
        let error = build_series(&[b100.clone(), duplicate], None).unwrap_err();
        assert!(error.to_string().contains("ambiguous series coordinate"));

        let other = publish_fixture(
            &root,
            "other",
            &fixture_document("run-other", "other::train", 400),
        );
        let error = build_series(&[b100.clone(), other], None).unwrap_err();
        assert!(error.to_string().contains("mix entrypoints"));

        assert!(build_series(&[], None)
            .unwrap_err()
            .to_string()
            .contains("at least one bundle"));

        let absent = root.join("never-published");
        let error = build_series(&[absent], None).unwrap_err();
        assert!(format!("{error:#}").contains("verify series bundle"));

        fs::remove_dir_all(root).unwrap();
    }
}
