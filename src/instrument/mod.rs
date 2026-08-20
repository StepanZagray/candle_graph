//! Post-run trace instrumentation — TensorFlow Profiler-style span API.
//!
//! Emits [`crate::trace`] JSONL (`candle-graph/trace/7`). Candle-free: callers supply
//! timings, shapes, and gradient facts from their own probe binary.

mod selector;
mod session;
mod span;

#[cfg(feature = "candle")]
pub mod candle;

pub use selector::CaptureSelector;
pub use session::{ProfileRun, TraceSession};
pub use span::{
    DeviceIntervalRecord, DeviceMemoryRecord, MemoryRecord, OpRecord, SpanGuard, SpanId, SpanKind,
    TensorRecord,
};
