//! TensorFlow-Profiler-style trace protocol (`candle-graph/trace/9`).
//!
//! Trace-only evidence: spans, timed ops, tensor snapshots, memory events, gradient facts,
//! and span edges. No static Rust analysis — importers aggregate JSONL into
//! [`document::TraceDocument`].

pub mod document;
pub mod events;
pub mod health;
pub mod memory;
pub mod schema;

pub use document::{parse_trace, write_jsonl, TraceDocument};
pub use events::{
    DeviceIntervalEvent, DeviceMemoryEvent, EdgeEvent, GradientEvent, MemoryEvent, OpEvent,
    SpanEndEvent, SpanStartEvent, TensorEvent, TerminalEvent, TraceEvent,
};
pub use health::{analyze_health, EvidenceCoverage, HealthIssue, HealthSeverity, TraceHealth};
pub use memory::{category_for_step, dense_tensor_bytes, MemoryAction, MemoryCategory};
pub use schema::{
    ComparisonIdentity, GradientState, RunOutcome, SpanKind, SpanRecord, TimingMode, TraceRunMeta,
    TraceSummary, SCHEMA,
};
