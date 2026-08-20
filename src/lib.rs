//! TensorFlow Profiler-style execution graphs for [Candle](https://github.com/huggingface/candle) runs.
//!
//! Record one representative JSONL trace ([`instrument::TraceSession`]), build an [`graph::ExecutionGraph`],
//! and inspect it via CLI or HTML ([`viewer::render_evidence_html`]).
//!
//! There is **no static Rust analysis** — the graph comes only from what executed.

pub mod artifact;
pub mod capability;
pub mod cli;
pub mod comparison;
pub mod evidence;
pub mod graph;
pub mod instrument;
pub mod nsight;
pub mod phase;
pub mod timing;
pub mod trace;
#[cfg(feature = "visualizer")]
pub mod viewer;

pub use artifact::{publish_bundle, verify_bundle, BundleManifest, BundleVerificationReceipt};
pub use capability::{CaptureContract, CoverageLevel, MeasurementScope};
pub use comparison::{compare_replicates, ReplicatedComparison};
pub use evidence::{build_evidence, EvidencePacket};
pub use graph::{build_from_trace, ExecutionGraph};
#[cfg(feature = "candle")]
pub use instrument::candle;
pub use instrument::{
    CaptureSelector, DeviceIntervalRecord, DeviceMemoryRecord, MemoryRecord, OpRecord, ProfileRun,
    SpanGuard, SpanId, SpanKind, TensorRecord, TraceSession,
};
pub use nsight::{GpuEvidenceStatus, NsightEvidence};
pub use phase::{ExecutionPhase, ExecutionStep};
pub use trace::{parse_trace, write_jsonl, TraceDocument, TraceEvent, SCHEMA as TRACE_SCHEMA};
pub use trace::{ComparisonIdentity, TimingMode};
