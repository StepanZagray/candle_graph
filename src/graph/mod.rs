//! Execution graph derived purely from trace documents.

mod build;
mod model;

pub use build::build_from_trace;
pub use model::*;
