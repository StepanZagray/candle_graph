//! Standalone `candle-graph` binary — same trace commands as `cargo candle-graph`.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use candle_graph::cli::trace_cli::{self, TraceQueryKind};

#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    about = "Import and visualize Candle execution traces",
    long_about = "Capability-qualified evidence and atomic bundles from candle-graph/trace/7 runs."
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
    Compare(CompareArgs),
    Report(ReportArgs),
    Verify(VerifyArgs),
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
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    #[arg(long, required = true, num_args = 1.., value_name = "TRACE")]
    baseline: Vec<PathBuf>,
    #[arg(long, required = true, num_args = 1.., value_name = "TRACE")]
    candidate: Vec<PathBuf>,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    trace: PathBuf,
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    bundle: PathBuf,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    #[arg(value_name = "BUNDLE")]
    bundle: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
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
    SlowestHost,
    SlowestDevice,
    Heaviest,
    Memory,
    Spans,
    Tensors,
    Gradients,
    Capabilities,
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
            CliTraceQueryKind::Gradients => Self::Gradients,
            CliTraceQueryKind::Capabilities => Self::Capabilities,
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
        Command::Report(report) => {
            trace_cli::run_report(&report.trace, report.nsight_dir.as_deref(), &report.bundle)
        }
        Command::Verify(verify) => trace_cli::run_verify(&verify.bundle, verify.output.as_deref()),
    }
}
