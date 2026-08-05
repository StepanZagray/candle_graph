//! JSONL event records for `candle-graph/trace/5`.

use serde::{Deserialize, Serialize};

use crate::phase::ExecutionStep;
use super::memory::{MemoryAction, MemoryCategory};
use super::schema::{GradientState, SpanKind, TraceRunMeta, SCHEMA};

/// One JSONL record in a trace stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    /// Declares schema + run metadata (exactly one per stream).
    Meta {
        schema: String,
        #[serde(flatten)]
        run: TraceRunMeta,
    },
    SpanStart(SpanStartEvent),
    SpanEnd(SpanEndEvent),
    Op(OpEvent),
    Tensor(TensorEvent),
    Memory(MemoryEvent),
    DeviceMemory(DeviceMemoryEvent),
    Gradient(GradientEvent),
    Edge(EdgeEvent),
}

impl TraceEvent {
    /// Convenience constructor for the required first meta line.
    pub fn meta(run: TraceRunMeta) -> Self {
        Self::Meta {
            schema: SCHEMA.to_string(),
            run,
        }
    }
}

/// Opens a span in the profiler hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanStartEvent {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub name: String,
    /// Span classification (`function` / `op` / `module`). Serialized as `span_kind` because
    /// the JSONL event discriminator also uses the key `kind`.
    #[serde(rename = "span_kind")]
    pub kind: SpanKind,
    /// PyTorch-style training step (`forward` / `backward` / `optimizer`) for memory categories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<ExecutionStep>,
}

/// Closes a span opened by [`SpanStartEvent`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanEndEvent {
    pub id: String,
    /// Wall duration in nanoseconds (from the probe's span guard).
    #[serde(default)]
    pub duration_ns: u64,
}

/// Timed operation inside a span (matmul, add, function body, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpEvent {
    pub span_id: String,
    pub op_name: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default)]
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub duration_ns: u64,
    /// Monotonic timestamp for memory timeline ordering (nanoseconds since probe start).
    #[serde(default)]
    pub timestamp_ns: u64,
    /// Dense output storage bytes; derived from shape×dtype when omitted at parse time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_bytes: Option<u64>,
    /// Sum of input tensor storage (PyTorch `record_shapes` footprint).
    #[serde(default)]
    pub input_storage_bytes: u64,
}

/// Tensor snapshot associated with a span (create or metadata).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEvent {
    pub span_id: String,
    pub tensor_id: String,
    #[serde(default)]
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    #[serde(default)]
    pub requires_grad: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_bytes: Option<u64>,
    #[serde(default)]
    pub category: MemoryCategory,
}

/// Tensor allocation or deallocation (TensorFlow Memory Profile timeline).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub timestamp_ns: u64,
    pub tensor_id: String,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op_name: Option<String>,
    pub device: String,
    pub bytes: u64,
    pub action: MemoryAction,
    #[serde(default)]
    pub shape: Vec<usize>,
    pub dtype: String,
    #[serde(default)]
    pub category: MemoryCategory,
}

/// Optional device-level memory checkpoint (cudaMemGetInfo-style).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemoryEvent {
    pub timestamp_ns: u64,
    pub device: String,
    pub used_bytes: u64,
    pub free_bytes: u64,
    /// Caching allocator reserved bytes (PyTorch `memory_reserved`); optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved_bytes: Option<u64>,
}

/// Parameter gradient fact recorded during a probe run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientEvent {
    pub event_id: String,
    pub root: String,
    /// Parameter key under `root` (alias `param_key` for v4 emitters).
    #[serde(alias = "param_key")]
    pub key: String,
    pub state: GradientState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<f64>,
}

impl GradientEvent {
    pub fn param_key(&self) -> &str {
        &self.key
    }
}

/// Data-flow edge timing between two spans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeEvent {
    pub from_span: String,
    pub to_span: String,
    pub duration_ns: u64,
}
