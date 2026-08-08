//! Execution graph model (`candle-graph/graph/3`).

use serde::{Deserialize, Serialize};

use crate::trace::memory::{MemoryCategory, MemoryProfile, MemorySummary};

/// Schema identifier for [`ExecutionGraph`] documents.
pub const SCHEMA: &str = "candle-graph/graph/3";

/// Hierarchical execution graph built from a trace document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub schema: String,
    /// Flat list of span and op nodes (parent links encode the tree).
    pub spans: Vec<GraphNode>,
    /// Call hierarchy and tensor data-flow edges with timing.
    pub edges: Vec<GraphEdge>,
    /// Semantic tensor checkpoints captured by the application.
    pub tensors: Vec<TensorRecord>,
    pub gradients: Vec<GradientRecord>,
    pub summary: GraphSummary,
    pub memory: MemoryProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub span_id: String,
    pub tensor_id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    pub storage_bytes: u64,
    pub category: MemoryCategory,
}

/// One span or attached operation in the execution tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub name: String,
    pub kind: GraphNodeKind,
    pub start_ns: u64,
    /// Wall time excluding nested spans/ops (TensorFlow profiler style).
    pub self_time_ns: u64,
    /// Inclusive wall time for this node.
    pub total_time_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Total bytes requested by this node's ops (TF `bytes`).
    #[serde(default)]
    pub bytes: u64,
    /// Peak live bytes in subtree (TF `peak_bytes`).
    #[serde(default)]
    pub peak_bytes: u64,
    /// Bytes still live when node finishes (TF `residual_bytes`).
    #[serde(default)]
    pub residual_bytes: u64,
    /// Op output storage (derived from shape×dtype).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Root,
    Function,
    Module,
    Op,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub kind: GraphEdgeKind,
    pub duration_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    Call,
    Data,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientRecord {
    pub root: String,
    pub key: String,
    pub state: GradientRecordState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientRecordState {
    Present,
    Missing,
    Zero,
    NonFinite,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSummary {
    pub entrypoint: String,
    /// Root span inclusive wall time in milliseconds.
    pub total_ms: f64,
    /// Top spans by self time (excluding op leaf nodes).
    pub slowest_spans: Vec<SlowSpan>,
    /// Top spans by requested bytes (TF scope -order_by bytes).
    pub heaviest_spans: Vec<HeavySpan>,
    pub memory: MemorySummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlowSpan {
    pub id: String,
    pub name: String,
    pub self_time_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeavySpan {
    pub id: String,
    pub name: String,
    pub bytes: u64,
    pub peak_bytes: u64,
}

impl ExecutionGraph {
    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.spans.iter().find(|n| n.id == id)
    }

    pub fn children(&self, parent_id: &str) -> Vec<&GraphNode> {
        self.spans
            .iter()
            .filter(|n| n.parent_id.as_deref() == Some(parent_id))
            .collect()
    }
}
