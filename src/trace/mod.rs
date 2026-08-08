//! TensorFlow-Profiler-style trace protocol (`candle-graph/trace/5`).
//!
//! Trace-only evidence: spans, timed ops, tensor snapshots, memory events, gradient facts,
//! and span edges. No static Rust analysis — importers aggregate JSONL into
//! [`document::TraceDocument`].

pub mod document;
pub mod events;
pub mod memory;
pub mod schema;

pub use document::{parse_trace, write_jsonl, TraceDocument};
pub use events::{
    DeviceMemoryEvent, EdgeEvent, GradientEvent, MemoryEvent, OpEvent, SpanEndEvent,
    SpanStartEvent, TensorEvent, TraceEvent,
};
pub use memory::{category_for_step, storage_bytes, MemoryAction, MemoryCategory};
pub use schema::{GradientState, SpanKind, SpanRecord, TraceRunMeta, TraceSummary, SCHEMA};
