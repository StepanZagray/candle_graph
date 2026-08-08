//! TensorFlow Profiler-style execution graphs for [Candle](https://github.com/huggingface/candle) runs.
//!
//! Record a post-run JSONL trace ([`instrument::TraceSession`]), build an [`graph::ExecutionGraph`],
//! and inspect it via CLI or HTML ([`viewer::render_trace_html`]).
//!
//! There is **no static Rust analysis** — the graph comes only from what executed.

pub mod cli;
pub mod graph;
pub mod instrument;
pub mod phase;
pub mod trace;
#[cfg(feature = "visualizer")]
pub mod viewer;

pub use graph::{build_from_trace, ExecutionGraph};
#[cfg(feature = "candle")]
pub use instrument::candle;
pub use instrument::{
    MemoryRecord, OpRecord, SpanGuard, SpanId, SpanKind, TensorRecord, TraceSession,
};
pub use phase::{ExecutionPhase, ExecutionStep};
pub use trace::{parse_trace, write_jsonl, TraceDocument, TraceEvent, SCHEMA as TRACE_SCHEMA};
