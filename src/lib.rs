//! Static structure and dataflow analysis for candle-rs models.
//!
//! See `README.md` and `docs/features.md` for the feature split:
//!
//! - **`static`** (default): agent-oriented IR, bounded queries, dtype/gradient dataflow
//! - **`runtime`**: merge v3 profile traces; operation timings (avg/min/max) and gradient audit
//! - **`visualizer`**: standalone `model.html` for human exploration

pub mod analysis_cache;
pub mod cargo_context;
pub mod cli;
pub mod contracts;
pub mod dataflow;
pub mod dtype_propagate;
pub mod diagnostics;
pub mod discover;
pub mod extract;
pub mod ir;
pub mod known;
pub mod load;
pub mod model_baseline;
pub mod model_ir;
pub mod op_semantics;
pub mod phase;
#[cfg(feature = "runtime")]
pub mod profile;
pub mod query;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod verify;
#[cfg(feature = "visualizer")]
pub mod viewer;
#[cfg(feature = "visualizer")]
pub mod viewer_projection;

pub use phase::ExecutionPhase;
