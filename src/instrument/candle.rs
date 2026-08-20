//! Candle tensor helpers — PyTorch `record_shapes` / `profile_memory` ergonomics for probes.
//!
//! Enable with the `candle` feature: `candle-graph = { path = "..", features = ["candle"] }`.

use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::instrument::SpanId;
use crate::instrument::{OpRecord, TensorRecord, TraceSession};
use crate::phase::ExecutionStep;
use crate::trace::memory::{category_for_step, dense_tensor_bytes};
use crate::trace::MemoryCategory;

/// Owned metadata captured from a Candle tensor (safe to store between probe calls).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleCapture {
    pub storage_id: String,
    pub tensor_id: String,
    pub label: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub tensor_bytes: u64,
    pub requires_grad: bool,
    pub category: MemoryCategory,
}

/// Owned op observation including output tensor metadata and input byte sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleOpCapture {
    pub op_name: String,
    pub inputs: Vec<String>,
    pub output: CandleCapture,
    pub input_tensor_bytes: u64,
    pub duration_ns: u64,
    pub timestamp_ns: u64,
}

/// Stable string form of [`Tensor::id`] for trace dedup.
pub fn tensor_id(t: &Tensor) -> String {
    format!("{id:?}", id = t.id())
}

/// Process-local identity of the backend storage shared by aliased tensor views.
pub fn storage_id(t: &Tensor) -> String {
    let (storage, _) = t.storage_and_layout();
    format!("storage:{:p}", &*storage)
}

/// Device label for JSONL (`cpu`, `cuda:0`, `metal:0`, …).
pub fn device_label(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "cpu".into(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
    }
}

/// Lowercase dtype label (`f32`, `bf16`, …).
pub fn dtype_label(dtype: DType) -> String {
    format!("{dtype:?}").to_ascii_lowercase()
}

/// Dense tensor footprint. This does not claim the size of shared backing storage.
pub fn tensor_dense_bytes(t: &Tensor) -> u64 {
    dense_tensor_bytes(t.dims(), &dtype_label(t.dtype()))
        .expect("every Candle dtype has a known byte width")
}

impl CandleCapture {
    pub fn from_tensor(t: &Tensor, step: Option<ExecutionStep>) -> Self {
        let requires_grad = t.is_variable();
        let category = if requires_grad {
            MemoryCategory::Parameter
        } else {
            category_for_step(step, false)
        };
        Self {
            storage_id: storage_id(t),
            tensor_id: tensor_id(t),
            label: None,
            shape: t.dims().to_vec(),
            dtype: dtype_label(t.dtype()),
            device: device_label(t.device()),
            tensor_bytes: tensor_dense_bytes(t),
            requires_grad,
            category,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl CandleOpCapture {
    pub fn new(
        op_name: impl Into<String>,
        inputs: Vec<String>,
        output: &Tensor,
        input_bytes: u64,
        duration_ns: u64,
        timestamp_ns: u64,
        step: Option<ExecutionStep>,
    ) -> Self {
        Self {
            op_name: op_name.into(),
            inputs,
            output: CandleCapture::from_tensor(output, step),
            input_tensor_bytes: input_bytes,
            duration_ns,
            timestamp_ns,
        }
    }
}

/// Sum dense footprints for input tensors (PyTorch input-shape-style metadata).
pub fn inputs_dense_bytes(tensors: &[&Tensor]) -> u64 {
    tensors.iter().map(|t| tensor_dense_bytes(t)).sum()
}

/// Record a Candle tensor via [`TraceSession::record_tensor`].
pub fn record_tensor(
    session: &TraceSession,
    span_id: SpanId,
    cap: &CandleCapture,
) -> anyhow::Result<()> {
    session.record_tensor(
        span_id,
        TensorRecord {
            tensor_id: &cap.tensor_id,
            label: cap.label.as_deref(),
            shape: &cap.shape,
            dtype: &cap.dtype,
            device: &cap.device,
            requires_grad: cap.requires_grad,
            dense_bytes: Some(cap.tensor_bytes),
            category: cap.category,
        },
    )
}

/// Record a timed operation from a [`CandleOpCapture`].
pub fn record_op(
    session: &TraceSession,
    span_id: SpanId,
    cap: &CandleOpCapture,
) -> anyhow::Result<()> {
    session.record_op(
        span_id,
        OpRecord {
            op_name: &cap.op_name,
            inputs: &cap.inputs,
            output: Some(&cap.output.tensor_id),
            shape: &cap.output.shape,
            dtype: &cap.output.dtype,
            device: &cap.output.device,
            duration_ns: cap.duration_ns,
            timestamp_ns: cap.timestamp_ns,
            output_dense_bytes: Some(cap.output.tensor_bytes),
            input_dense_bytes: cap.input_tensor_bytes,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn dense_tensor_footprint_matches_shape_dtype() {
        let t = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor_dense_bytes(&t), 4 * 8 * 4);
    }

    #[test]
    fn capture_from_tensor() {
        let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        let cap = CandleCapture::from_tensor(&t, Some(ExecutionStep::Forward));
        assert_eq!(cap.shape, vec![2, 3]);
        assert_eq!(cap.category, MemoryCategory::Activation);
    }
}
