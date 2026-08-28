//! Standalone `candle-graph` binary — same trace commands as `cargo candle-graph`.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use candle_graph::cli::trace_cli::{self, TraceQueryKind};

#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    about = "Import and visualize Candle execution traces",
    long_about = "Capability-qualified evidence and atomic bundles from candle-graph/trace/10 runs."
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Emit a complete capability-qualified evidence packet.
    Import(ImportArgs),
    /// Render a standalone HTML visualizer from a trace (requires `visualizer` feature).
    #[cfg(feature = "visualizer")]
    View(ViewArgs),
    /// Emit a profiler summary for a raw trace or verified bundle/profile.
    Summary(SummaryArgs),
    /// Run a typed query against raw-trace or verified bundle evidence.
    Query(QueryArgs),
    /// Compare a candidate profile run with an explicit baseline.
    Compare(CompareArgs),
    /// Atomically publish a content-addressed evidence bundle.
    Report(ReportArgs),
    /// Deeply verify a published evidence bundle and emit a durable receipt.
    Verify(VerifyArgs),
}

#[derive(Parser, Debug)]
struct ImportArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    trace: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[cfg(feature = "visualizer")]
struct ViewArgs {
    /// Trace JSONL file (`candle-graph/trace/10` or readable trace/9).
    #[arg(value_name = "TRACE")]
    trace: PathBuf,
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// Finalized evidence bundle directories, unless `--unverified-traces` is set.
    #[arg(long, required = true, num_args = 1.., value_name = "BUNDLE")]
    baseline: Vec<PathBuf>,
    /// Finalized evidence bundle directories, unless `--unverified-traces` is set.
    #[arg(long, required = true, num_args = 1.., value_name = "BUNDLE")]
    candidate: Vec<PathBuf>,
    /// Treat baseline and candidate paths as raw traces; output is always diagnostic/ineligible.
    #[arg(long)]
    unverified_traces: bool,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    /// Raw trace JSONL file (`candle-graph/trace/10` or readable trace/9).
    #[arg(value_name = "TRACE")]
    trace: PathBuf,
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    bundle: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    /// Finalized evidence bundle directory.
    #[arg(value_name = "BUNDLE")]
    bundle: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct SummaryArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(long, value_enum)]
    kind: CliTraceQueryKind,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliTraceQueryKind {
    SlowestHost,
    SlowestDevice,
    Heaviest,
    Memory,
    Spans,
    Tensors,
    TensorStats,
    Gradients,
    Capabilities,
    GpuStatus,
    GpuCorrelation,
    GpuPhases,
    GpuKernels,
    GpuAttributionGaps,
}

impl From<CliTraceQueryKind> for TraceQueryKind {
    fn from(kind: CliTraceQueryKind) -> Self {
        match kind {
            CliTraceQueryKind::SlowestHost => Self::SlowestHost,
            CliTraceQueryKind::SlowestDevice => Self::SlowestDevice,
            CliTraceQueryKind::Heaviest => Self::Heaviest,
            CliTraceQueryKind::Memory => Self::Memory,
            CliTraceQueryKind::Spans => Self::Spans,
            CliTraceQueryKind::Tensors => Self::Tensors,
            CliTraceQueryKind::TensorStats => Self::TensorStats,
            CliTraceQueryKind::Gradients => Self::Gradients,
            CliTraceQueryKind::Capabilities => Self::Capabilities,
            CliTraceQueryKind::GpuStatus => Self::GpuStatus,
            CliTraceQueryKind::GpuCorrelation => Self::GpuCorrelation,
            CliTraceQueryKind::GpuPhases => Self::GpuPhases,
            CliTraceQueryKind::GpuKernels => Self::GpuKernels,
            CliTraceQueryKind::GpuAttributionGaps => Self::GpuAttributionGaps,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Import(import) => trace_cli::run_import(&import.trace, import.output.as_deref()),
        #[cfg(feature = "visualizer")]
        Command::View(view) => {
            trace_cli::run_view(&view.trace, &view.output, view.nsight_dir.as_deref())
        }
        Command::Summary(summary) => {
            trace_cli::run_summary(&summary.input, summary.output.as_deref())
        }
        Command::Query(query) => {
            trace_cli::run_query(&query.input, query.kind.into(), query.output.as_deref())
        }
        Command::Compare(compare) => trace_cli::run_compare(
            &compare.baseline,
            &compare.candidate,
            compare.unverified_traces,
            compare.output.as_deref(),
        ),
        Command::Report(report) => {
            trace_cli::run_report(&report.trace, report.nsight_dir.as_deref(), &report.bundle)
        }
        Command::Verify(verify) => trace_cli::run_verify(&verify.bundle, verify.output.as_deref()),
    }
}
