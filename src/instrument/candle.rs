//! Candle tensor helpers — PyTorch `record_shapes` / `profile_memory` ergonomics for probes.
//!
//! Enable with the `candle` feature: `candle-graph = { path = "..", features = ["candle"] }`.

use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::instrument::{MemoryRecord, OpRecord, TensorRecord, TraceSession};
use crate::instrument::SpanId;
use crate::phase::ExecutionStep;
use crate::trace::memory::{category_for_step, storage_bytes};
use crate::trace::MemoryCategory;

/// Owned metadata captured from a Candle tensor (safe to store between probe calls).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleCapture {
    pub tensor_id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub storage_bytes: u64,
    pub requires_grad: bool,
    pub category: MemoryCategory,
}

/// Owned op observation including output tensor metadata and input byte sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleOpCapture {
    pub op_name: String,
    pub inputs: Vec<String>,
    pub output: CandleCapture,
    pub input_storage_bytes: u64,
    pub duration_ns: u64,
    pub timestamp_ns: u64,
    pub category: MemoryCategory,
}

/// Stable string form of [`Tensor::id`] for trace dedup.
pub fn tensor_id(t: &Tensor) -> String {
    format!("{id:?}", id = t.id())
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

/// Dense storage bytes for a Candle tensor.
pub fn tensor_storage_bytes(t: &Tensor) -> u64 {
    storage_bytes(t.dims(), &dtype_label(t.dtype()))
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
            tensor_id: tensor_id(t),
            shape: t.dims().to_vec(),
            dtype: dtype_label(t.dtype()),
            device: device_label(t.device()),
            storage_bytes: tensor_storage_bytes(t),
            requires_grad,
            category,
        }
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
            input_storage_bytes: input_bytes,
            duration_ns,
            timestamp_ns,
            category: category_for_step(step, false),
        }
    }
}

/// Sum storage bytes for input tensors (PyTorch input-shape memory).
pub fn inputs_storage_bytes(tensors: &[&Tensor]) -> u64 {
    tensors.iter().map(|t| tensor_storage_bytes(t)).sum()
}

/// Record a Candle tensor via [`TraceSession::record_tensor`].
pub fn record_tensor(session: &TraceSession, span_id: SpanId, cap: &CandleCapture) -> anyhow::Result<()> {
    session.record_tensor(
        span_id,
        TensorRecord {
            tensor_id: &cap.tensor_id,
            shape: &cap.shape,
            dtype: &cap.dtype,
            device: &cap.device,
            requires_grad: cap.requires_grad,
            storage_bytes: Some(cap.storage_bytes),
            category: cap.category,
        },
    )
}

/// Record a timed op + memory alloc from a [`CandleOpCapture`].
pub fn record_op(session: &TraceSession, span_id: SpanId, cap: &CandleOpCapture) -> anyhow::Result<()> {
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
            storage_bytes: Some(cap.output.storage_bytes),
            input_storage_bytes: cap.input_storage_bytes,
            category: Some(cap.category),
        },
    )
}

/// Record an explicit tensor free (e.g. after `zero_grad(set_to_none=true)`).
pub fn record_memory_free(
    session: &TraceSession,
    span_id: SpanId,
    cap: &CandleCapture,
    timestamp_ns: Option<u64>,
) -> anyhow::Result<()> {
    session.record_memory_free(
        span_id,
        MemoryRecord {
            tensor_id: &cap.tensor_id,
            device: &cap.device,
            bytes: cap.storage_bytes,
            dtype: &cap.dtype,
            category: cap.category,
            timestamp_ns,
            op_name: None,
            shape: &cap.shape,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn tensor_storage_matches_shape_dtype() {
        let t = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor_storage_bytes(&t), 4 * 8 * 4);
    }

    #[test]
    fn capture_from_tensor() {
        let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        let cap = CandleCapture::from_tensor(&t, Some(ExecutionStep::Forward));
        assert_eq!(cap.shape, vec![2, 3]);
        assert_eq!(cap.category, MemoryCategory::Activation);
    }
}
