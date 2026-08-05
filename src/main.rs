//! Standalone `candle-graph` binary — same trace commands as `cargo candle-graph`.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use candle_graph::cli::trace_cli::{self, TraceQueryKind};

#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    about = "Import and visualize Candle execution traces",
    long_about = "TensorFlow Profiler-style graphs from post-run JSONL traces (candle-graph/trace/4)."
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Import(ImportArgs),
    #[cfg(feature = "visualizer")]
    View(ViewArgs),
    Summary(SummaryArgs),
    Query(QueryArgs),
}

#[derive(Parser, Debug)]
struct ImportArgs {
    trace: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[cfg(feature = "visualizer")]
struct ViewArgs {
    trace: PathBuf,
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
}

#[derive(Parser, Debug)]
struct SummaryArgs {
    trace: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
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
    let args = Args::parse();
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
