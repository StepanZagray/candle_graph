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
    long_about = "Trustworthy evidence packets and unified HTML from candle-graph/trace/6 runs."
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
    /// Compare a candidate profile run with an explicit baseline.
    Compare(CompareArgs),
    /// Publish JSON and Markdown evidence artifacts.
    Report(ReportArgs),
}

#[derive(Parser, Debug)]
struct ImportArgs {
    /// Trace JSONL file (`candle-graph/trace/6`).
    trace: PathBuf,

    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[cfg(feature = "visualizer")]
struct ViewArgs {
    /// Trace JSONL file (`candle-graph/trace/6`).
    trace: PathBuf,

    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, value_name = "TRACE")]
    baseline: Option<PathBuf>,

    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    baseline: PathBuf,
    candidate: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    trace: PathBuf,
    #[arg(long, value_name = "TRACE")]
    baseline: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    json: PathBuf,
    #[arg(long, value_name = "FILE")]
    markdown: PathBuf,
}

#[derive(Parser, Debug)]
struct SummaryArgs {
    /// Trace JSONL file (`candle-graph/trace/6`).
    trace: PathBuf,

    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    /// Trace JSONL file (`candle-graph/trace/6`).
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
    Tensors,
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
            CliTraceQueryKind::Tensors => Self::Tensors,
            CliTraceQueryKind::Gradients => Self::Gradients,
        }
    }
}

fn main() -> Result<()> {
    let args = CargoArgs::parse();
    match args.command {
        Command::Import(import) => trace_cli::run_import(&import.trace, import.output.as_deref()),
        #[cfg(feature = "visualizer")]
        Command::View(view) => trace_cli::run_view(
            &view.trace,
            &view.output,
            view.baseline.as_deref(),
            view.nsight_dir.as_deref(),
        ),
        Command::Summary(summary) => {
            trace_cli::run_summary(&summary.trace, summary.output.as_deref())
        }
        Command::Query(query) => {
            trace_cli::run_query(&query.trace, query.kind.into(), query.output.as_deref())
        }
        Command::Compare(compare) => trace_cli::run_compare(
            &compare.baseline,
            &compare.candidate,
            compare.output.as_deref(),
        ),
        Command::Report(report) => trace_cli::run_report(
            &report.trace,
            report.baseline.as_deref(),
            report.nsight_dir.as_deref(),
            &report.json,
            &report.markdown,
        ),
    }
}
