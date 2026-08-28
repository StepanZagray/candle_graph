//! TensorFlow Profiler-style execution graphs for [Candle](https://github.com/huggingface/candle) runs.
//!
//! Record one representative JSONL trace with [`instrument::TraceSession`], qualify its structural
//! health and evidence capabilities with [`evidence::build_evidence`], and inspect it through the
//! CLI or, with the `visualizer` feature, [`viewer::render_evidence_html`].
//!
//! There is **no static Rust analysis** — the graph comes only from what executed.
//!
//! Keep these evidence planes distinct:
//!
//! - span duration is host timing unless the producer declares stronger synchronization semantics;
//! - [`timing`] device intervals retain their own device clock;
//! - [`trace::memory`] logical storage lifetimes do not imply physical allocator usage; and
//! - optional [`nsight`] evidence retains the Nsight clock and artifact provenance.
//!
//! Failed or structurally invalid captures remain diagnostic but do not produce a derived
//! [`graph::ExecutionGraph`]. Finalized evidence can be published and deeply verified through
//! [`artifact`], while eligible repeated timing comparisons require verified bundles through
//! [`comparison`].

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

pub use artifact::{
    publish_bundle, verify_bundle, BundleManifest, BundleVerificationReceipt,
    SCHEMA as BUNDLE_SCHEMA,
};
pub use capability::{
    CaptureContract, CoverageLevel, ExpectedGradient, GradientContract, GradientFamilyContract,
    GradientFamilyExpectation, MeasurementScope, GRADIENT_MANIFEST_SCHEMA,
};
pub use comparison::{
    compare_unverified_traces, compare_verified_bundles, ComparisonInputVerification,
    ComparisonInputs, ComparisonVerdict, ReplicatedComparison, TensorStatsComparison,
    TensorStatsComparisonRow, VerifiedBundleInput, SCHEMA as COMPARISON_SCHEMA,
};
pub use evidence::{build_evidence, EvidencePacket, SCHEMA as EVIDENCE_SCHEMA};
pub use graph::{build_from_trace, ExecutionGraph, SCHEMA as GRAPH_SCHEMA};
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
