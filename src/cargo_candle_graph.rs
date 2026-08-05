//! Cargo subcommand entrypoint: `cargo candle-graph …`.
//!
//! Installed as `cargo-candle-graph` so Cargo discovers it as `cargo candle-graph`.
//! Trace-only commands: import JSONL traces into execution graphs, summarize, query, and view.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use candle_graph::cli::trace_cli::{self, TraceQueryKind};

#[derive(Parser, Debug)]
#[command(
    name = "cargo-candle-graph",
    bin_name = "cargo candle-graph",
    about = "Import and analyze candle-graph execution trace files",
    long_about = "Trace-only CLI for candle-graph v4 JSONL profiles: build execution graphs, \
                  summarize wall time, run bounded queries, and optionally emit HTML."
)]
struct CargoArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Import a trace JSONL file and emit execution graph JSON.
    Import(ImportArgs),
    /// Render a standalone HTML visualizer from a trace (requires `visualizer` feature).
    #[cfg(feature = "visualizer")]
    View(ViewArgs),
    /// Emit profiler summary for a trace file.
    Summary(SummaryArgs),
    /// Run a bounded query against trace-derived graph data.
    Query(QueryArgs),
}

#[derive(Parser, Debug)]
struct ImportArgs {
    /// Trace JSONL file (`candle-graph/trace/5`).
    trace: PathBuf,

    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[cfg(feature = "visualizer")]
struct ViewArgs {
    /// Trace JSONL file (`candle-graph/trace/5`).
    trace: PathBuf,

    #[arg(long, value_name = "FILE")]
    output: PathBuf,
}

#[derive(Parser, Debug)]
struct SummaryArgs {
    /// Trace JSONL file (`candle-graph/trace/5`).
    trace: PathBuf,

    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    /// Trace JSONL file (`candle-graph/trace/5`).
    trace: PathBuf,

    #[arg(long, value_enum)]
    kind: CliTraceQueryKind,

    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliTraceQueryKind {
    Slowest,
    Heaviest,
    Memory,
    Efficiency,
    Spans,
    Gradients,
}

impl From<CliTraceQueryKind> for TraceQueryKind {
    fn from(kind: CliTraceQueryKind) -> Self {
        match kind {
            CliTraceQueryKind::Slowest => Self::Slowest,
            CliTraceQueryKind::Heaviest => Self::Heaviest,
            CliTraceQueryKind::Memory => Self::Memory,
            CliTraceQueryKind::Efficiency => Self::Efficiency,
            CliTraceQueryKind::Spans => Self::Spans,
            CliTraceQueryKind::Gradients => Self::Gradients,
        }
    }
}

fn main() -> Result<()> {
    let args = CargoArgs::parse();
    match args.command {
        Command::Import(import) => trace_cli::run_import(&import.trace, import.output.as_deref()),
        #[cfg(feature = "visualizer")]
        Command::View(view) => trace_cli::run_view(&view.trace, &view.output),
        Command::Summary(summary) => {
            trace_cli::run_summary(&summary.trace, summary.output.as_deref())
        }
        Command::Query(query) => trace_cli::run_query(
            &query.trace,
            query.kind.into(),
            query.output.as_deref(),
        ),
    }
}
