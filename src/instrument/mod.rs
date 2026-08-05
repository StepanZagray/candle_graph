//! Post-run trace instrumentation — TensorFlow Profiler-style span API.
//!
//! Emits [`crate::trace`] JSONL (`candle-graph/trace/5`). Candle-free: callers supply
//! timings, shapes, and gradient facts from their own probe binary.

mod session;
mod span;

#[cfg(feature = "candle")]
pub mod candle;

pub use session::TraceSession;
pub use span::{MemoryRecord, OpRecord, SpanGuard, SpanId, SpanKind, TensorRecord};
