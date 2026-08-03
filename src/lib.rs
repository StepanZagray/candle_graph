//! Static structure and dataflow analysis for candle-rs models.
//!
//! See `README.md` for the motivation. In short: candle's autograd graph is not inspectable at
//! runtime (`Tensor::op()` is `pub(crate)`), and several candle-nn ops sever gradient flow
//! silently, so the only way to answer "does this parameter receive a gradient" is to read the
//! source.
//!
//! It reconstructs module and parameter structure, expression-level dtype and gradient dataflow,
//! deterministic CI baselines, and a standalone interactive report.

pub mod analysis_cache;
pub mod baseline;
pub mod cargo_context;
pub mod cli;
pub mod contracts;
pub mod dataflow;
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
pub mod profile;
pub mod query;
pub mod report;
pub mod runtime;
pub mod verify;
pub mod viewer;

pub use phase::ExecutionPhase;
pub use ir::Structure;
