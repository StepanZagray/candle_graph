//! Execution graph model (`candle-graph/graph/4`).

use serde::{Deserialize, Serialize};

use crate::trace::memory::MemoryCategory;

/// Schema identifier for [`ExecutionGraph`] documents.
pub const SCHEMA: &str = "candle-graph/graph/4";

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub span_id: String,
    pub tensor_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_bytes: Option<u64>,
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
    /// Host wall time excluding nested spans/ops.
    pub host_self_time_ns: u64,
    /// Inclusive host wall time for this node.
    pub host_total_time_ns: u64,
    /// Overlap-safe timings kept separate for every incomparable device clock.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub device_timings: Vec<DeviceNodeTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Logical storage bytes directly allocated by this node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocated_bytes: Option<u64>,
    /// Peak logical live bytes in this node's subtree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_live_bytes: Option<u64>,
    /// Logical bytes still live when this node finishes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_bytes: Option<u64>,
    /// Dense tensor footprint (derived from shape×dtype), not backing allocation size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Root,
    Function,
    Module,
    Op,
    Tensor,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceNodeTiming {
    pub device: String,
    pub clock_id: String,
    pub busy_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphEdge {
    Call {
        from_span: String,
        to_span: String,
        host_duration_ns: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    Data {
        from_tensor: String,
        to_tensor: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
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
    pub outer_wall_time_ns: u64,
    pub slowest_host_spans: Vec<HostSpanCost>,
    pub slowest_device_spans: Vec<DeviceSpanCost>,
    /// Top spans by known logical allocation bytes.
    pub heaviest_spans: Vec<HeavySpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostSpanCost {
    pub id: String,
    pub name: String,
    pub host_self_time_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSpanCost {
    pub id: String,
    pub name: String,
    pub device: String,
    pub clock_id: String,
    pub device_busy_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeavySpan {
    pub id: String,
    pub name: String,
    pub allocated_bytes: u64,
    pub peak_live_bytes: Option<u64>,
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
