//! Schema types for the current candle-graph execution-evidence stream.

use std::collections::BTreeMap;
use std::fmt;

use crate::phase::ExecutionPhase;
use crate::phase::ExecutionStep;

use serde::{Deserialize, Serialize};

use crate::capability::CaptureContract;

/// Schema identifier written into every trace document and expected on parse.
pub const SCHEMA: &str = "candle-graph/trace/10";
pub const PREVIOUS_SCHEMA: &str = "candle-graph/trace/9";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonIdentity {
    /// Stable caller-owned identity for the implementation or build under test.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_id: Option<String>,
    pub workload_id: String,
    pub model_id: String,
    pub config_id: String,
    pub data_id: String,
    pub seed_policy: String,
    pub physical_batch: u64,
    pub accumulation_steps: u64,
    pub precision: String,
    pub device_state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair_id: Option<String>,
}

/// Run metadata carried in the first JSONL `meta` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRunMeta {
    /// Stable identifier for this probe run (UUID, build id, or caller-defined key).
    pub run_id: String,
    /// Stable caller-owned key shared with NVTX range labels and external artifacts.
    pub correlation_id: String,
    /// Analyzed entrypoint, e.g. `train::leworld_loss` or `Model::forward`.
    pub entrypoint: String,
    /// Execution phase: `train` or `infer`.
    pub phase: ExecutionPhase,
    /// ISO-8601 timestamp when the trace was captured.
    pub timestamp: String,
    /// One-based optimizer update or inference invocation selected for capture.
    pub capture_step: u64,
    /// Completed invocations before the selected capture.
    pub warmup_steps: u64,
    /// Device requested by the caller (`cpu`, `cuda:0`, ...).
    pub device: String,
    /// Whether the single measured region is bounded by device synchronizations.
    /// Nested span durations remain governed by `timing_mode`.
    #[serde(default)]
    pub measured_region_device_synchronized: bool,
    pub timing_mode: TimingMode,
    pub capture_contract: CaptureContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison_identity: Option<ComparisonIdentity>,
    /// Workload-specific provenance such as lesson, batch size, and source revision.
    pub tags: BTreeMap<String, String>,
    /// Optional candle crate version observed at probe time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle_version: Option<String>,
}

impl TraceRunMeta {
    /// Validate provenance domain invariants before trusting a parsed or constructed run.
    /// Producers enforce these at capture time; consumers re-check them because traces can
    /// come from external tools or hand-edited files.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (label, value) in [
            ("run_id", &self.run_id),
            ("correlation_id", &self.correlation_id),
            ("entrypoint", &self.entrypoint),
            ("timestamp", &self.timestamp),
            ("device", &self.device),
        ] {
            anyhow::ensure!(
                !value.trim().is_empty(),
                "run provenance {label} must not be empty"
            );
        }
        anyhow::ensure!(
            self.capture_step > 0,
            "capture_step must be one-based and greater than zero"
        );
        anyhow::ensure!(
            self.warmup_steps < self.capture_step,
            "warmup_steps ({}) must be fewer than the one-based capture_step ({})",
            self.warmup_steps,
            self.capture_step
        );
        if let Some(identity) = &self.comparison_identity {
            identity.validate()?;
        }
        Ok(())
    }
}

impl ComparisonIdentity {
    /// Validate that every declared identity dimension carries a usable value.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (label, value) in [
            ("workload_id", &self.workload_id),
            ("model_id", &self.model_id),
            ("config_id", &self.config_id),
            ("data_id", &self.data_id),
            ("seed_policy", &self.seed_policy),
            ("precision", &self.precision),
            ("device_state", &self.device_state),
        ] {
            anyhow::ensure!(
                !value.trim().is_empty(),
                "comparison identity {label} must not be empty"
            );
        }
        for (label, value) in [
            ("implementation_id", &self.implementation_id),
            ("pair_id", &self.pair_id),
        ] {
            if let Some(value) = value {
                anyhow::ensure!(
                    !value.trim().is_empty(),
                    "comparison identity {label} must not be empty when declared"
                );
            }
        }
        anyhow::ensure!(
            self.physical_batch > 0,
            "comparison identity physical_batch must be greater than zero"
        );
        anyhow::ensure!(
            self.accumulation_steps > 0,
            "comparison identity accumulation_steps must be greater than zero"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingMode {
    Host,
    DeviceSynchronized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Complete,
    Failed,
}

/// Span classification — mirrors profiler function / op / module hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Function,
    Op,
    Module,
}

impl fmt::Display for SpanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "function"),
            Self::Op => write!(f, "op"),
            Self::Module => write!(f, "module"),
        }
    }
}

/// Observed gradient presence / quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientState {
    Present,
    Missing,
    Zero,
    NonFinite,
}

impl GradientState {
    pub(crate) fn norm_is_valid(self, norm: Option<f64>) -> bool {
        match (self, norm) {
            (Self::Present, Some(norm)) => norm.is_finite() && norm > 0.0,
            (Self::Zero, Some(norm)) => norm == 0.0 && !norm.is_sign_negative(),
            (Self::Missing | Self::NonFinite, None) => true,
            _ => false,
        }
    }
}

impl fmt::Display for GradientState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Present => write!(f, "present"),
            Self::Missing => write!(f, "missing"),
            Self::Zero => write!(f, "zero"),
            Self::NonFinite => write!(f, "non_finite"),
        }
    }
}

/// Aggregated profiler statistics derived from a [`super::document::TraceDocument`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSummary {
    pub op_count: usize,
    pub total_ns: u64,
    pub span_count: usize,
    pub root_span_count: usize,
    pub max_depth: usize,
    pub alloc_count: usize,
    pub free_count: usize,
    pub logical_peak_bytes: Option<u64>,
}

/// Resolved span node assembled from `span_start` / `span_end` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    /// True only for the caller-controlled region used as the run's performance total.
    pub measured: bool,
    /// Monotonic timestamp in nanoseconds since the profile run started.
    pub start_ns: u64,
    /// True when a matching `span_end` was observed.
    #[serde(default)]
    pub closed: bool,
    /// Wall duration from `span_end` (nanoseconds).
    #[serde(default)]
    pub duration_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<ExecutionStep>,
}
