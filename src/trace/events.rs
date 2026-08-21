//! JSONL event records for `candle-graph/trace/9`.

use serde::{Deserialize, Serialize};

use super::memory::{MemoryAction, MemoryCategory};
use super::schema::{GradientState, RunOutcome, SpanKind, TraceRunMeta, SCHEMA};
use crate::phase::ExecutionStep;

/// One JSONL record in a trace stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    /// Declares schema + run metadata (exactly one per stream).
    Meta {
        schema: String,
        #[serde(flatten)]
        run: Box<TraceRunMeta>,
    },
    SpanStart(SpanStartEvent),
    SpanEnd(SpanEndEvent),
    Op(OpEvent),
    Tensor(TensorEvent),
    Memory(MemoryEvent),
    DeviceMemory(DeviceMemoryEvent),
    DeviceInterval(DeviceIntervalEvent),
    Gradient(GradientEvent),
    Edge(EdgeEvent),
    Terminal(TerminalEvent),
}

impl TraceEvent {
    /// Convenience constructor for the required first meta line.
    pub fn meta(run: TraceRunMeta) -> Self {
        Self::Meta {
            schema: SCHEMA.to_string(),
            run: Box::new(run),
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
    /// Monotonic timestamp in nanoseconds since the profile run started.
    pub start_ns: u64,
    /// Span classification (`function` / `op` / `module`). Serialized as `span_kind` because
    /// the JSONL event discriminator also uses the key `kind`.
    #[serde(rename = "span_kind")]
    pub kind: SpanKind,
    #[serde(default)]
    pub measured: bool,
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
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub duration_ns: u64,
    /// Monotonic timestamp for memory timeline ordering (nanoseconds since probe start).
    #[serde(default)]
    pub timestamp_ns: u64,
    /// Dense output tensor footprint; this is not backing-allocation size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dense_bytes: Option<u64>,
    /// Sum of dense input tensor footprints (PyTorch `record_shapes`-style metadata).
    pub input_dense_bytes: u64,
}

/// Tensor snapshot associated with a span (create or metadata).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEvent {
    pub span_id: String,
    /// Backend tensor identity used for graph joins and deduplication.
    pub tensor_id: String,
    /// Optional caller-owned observation label; it is never used as tensor identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    #[serde(default)]
    pub requires_grad: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_bytes: Option<u64>,
    #[serde(default)]
    pub category: MemoryCategory,
}

/// Tensor allocation or deallocation (TensorFlow Memory Profile timeline).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub timestamp_ns: u64,
    /// Backend storage identity. Aliased tensor IDs share this identity.
    pub storage_id: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_bytes: Option<u64>,
    /// Caching allocator reserved bytes (PyTorch `memory_reserved`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved_bytes: Option<u64>,
    /// Independently observed device capacity; never derived from used + free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity_bytes: Option<u64>,
}

/// Resolved device interval on one device clock and stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIntervalEvent {
    pub span_id: String,
    pub device: String,
    pub stream_id: String,
    pub clock_id: String,
    pub backend: String,
    pub start_ns: u64,
    pub duration_ns: u64,
}

/// Parameter gradient fact recorded during a probe run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientEvent {
    pub event_id: String,
    pub root: String,
    /// Parameter key under `root`.
    pub key: String,
    pub state: GradientState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<f64>,
}

impl GradientEvent {
    pub fn param_key(&self) -> &str {
        &self.key
    }
}

/// A typed call-hierarchy or tensor data-flow edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "edge_kind", rename_all = "snake_case")]
pub enum EdgeEvent {
    Call {
        from_span: String,
        to_span: String,
        host_duration_ns: u64,
    },
    Data {
        from_tensor: String,
        to_tensor: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalEvent {
    pub outcome: RunOutcome,
    pub timestamp_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_event_without_label_remains_readable() {
        let event: TraceEvent = serde_json::from_str(
            r#"{"kind":"tensor","span_id":"root","tensor_id":"backend:1","shape":[1],"dtype":"f32","device":"cpu"}"#,
        )
        .unwrap();
        let TraceEvent::Tensor(tensor) = event else {
            panic!("expected tensor event");
        };
        assert_eq!(tensor.tensor_id, "backend:1");
        assert_eq!(tensor.label, None);
    }
}
