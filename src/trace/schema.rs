//! Schema types for `candle-graph/trace/4` — TensorFlow-Profiler-style span traces.

use std::fmt;

use crate::phase::ExecutionStep;

use serde::{Deserialize, Serialize};

/// Schema identifier written into every trace document and expected on parse.
pub const SCHEMA: &str = "candle-graph/trace/5";

/// Run metadata carried in the first JSONL `meta` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRunMeta {
    /// Stable identifier for this probe run (UUID, build id, or caller-defined key).
    pub run_id: String,
    /// Analyzed entrypoint, e.g. `train::leworld_loss` or `Model::forward`.
    pub entrypoint: String,
    /// Execution phase: `train` or `infer`.
    pub phase: String,
    /// ISO-8601 timestamp when the trace was captured.
    pub timestamp: String,
    /// Optional candle crate version observed at probe time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle_version: Option<String>,
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
    pub peak_bytes: u64,
}

/// Resolved span node assembled from `span_start` / `span_end` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    /// True when a matching `span_end` was observed.
    #[serde(default)]
    pub closed: bool,
    /// Wall duration from `span_end` (nanoseconds).
    #[serde(default)]
    pub duration_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<ExecutionStep>,
}
