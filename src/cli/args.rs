//! Single shared Clap definition for `candle-graph` and `cargo candle-graph`.
//!
//! Both binaries parse the same [`Cli`]; the cargo wrapper only strips the
//! forwarded subcommand name and rebrands the top-level command, so there are
//! zero duplicated subcommand definitions.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};

use super::trace_cli::{self, QueryLabelFilter, TraceQueryKind};

/// Complete typed CLI protocol shared by both binary entrypoints.
#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    version,
    about = "Import and visualize Candle execution traces",
    long_about = "Capability-qualified evidence and atomic bundles from candle-graph/trace/10 runs."
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

impl Cli {
    /// Dispatch the parsed command into the evidence CLI engine.
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Import(import) => {
                trace_cli::run_import(&import.trace, import.output.as_deref())
            }
            #[cfg(feature = "visualizer")]
            Command::View(view) => {
                trace_cli::run_view(&view.trace, &view.output, view.nsight_dir.as_deref())
            }
            Command::Summary(summary) => trace_cli::run_summary(
                &summary.input,
                summary.output.as_deref(),
                summary.require_valid,
            ),
            Command::Query(query) => {
                let filter = query.filter();
                trace_cli::run_query(
                    &query.input,
                    query.kind.into(),
                    filter.as_ref(),
                    query.output.as_deref(),
                )
            }
            Command::Overview(overview) => {
                trace_cli::run_overview(&overview.input, overview.output.as_deref())
            }
            Command::Compare(compare) => trace_cli::run_compare(
                &compare.baseline,
                &compare.candidate,
                compare.unverified_traces,
                compare.require_eligible,
                compare.output.as_deref(),
            ),
            Command::Report(report) => trace_cli::run_report(
                &report.trace,
                report.nsight_dir.as_deref(),
                &report.bundle,
                report.output.as_deref(),
            ),
            Command::Verify(verify) => {
                trace_cli::run_verify(&verify.bundle, verify.semantic, verify.output.as_deref())
            }
            Command::Protocol(protocol) => trace_cli::run_protocol(protocol.output.as_deref()),
            Command::CampaignStatus(status) => {
                trace_cli::run_campaign_status(&status.manifest, status.output.as_deref())
            }
            Command::Series(series) => trace_cli::run_series(
                series.manifest.as_deref(),
                &series.bundle,
                series.label_prefix.as_deref(),
                series.output.as_deref(),
            ),
        }
    }

    /// Parse argv for the `cargo-candle-graph` wrapper.
    ///
    /// Cargo forwards the external subcommand name through argv
    /// (`cargo candle-graph summary …` invokes
    /// `cargo-candle-graph candle-graph summary …`), so a leading
    /// `candle-graph` element is stripped before parsing. The top-level
    /// name/about are rebranded on the same shared [`Cli`] definition.
    pub fn parse_as_cargo_subcommand() -> Self {
        let mut argv: Vec<OsString> = std::env::args_os().collect();
        if argv
            .get(1)
            .is_some_and(|argument| argument == "candle-graph")
        {
            argv.remove(1);
        }
        let command = Self::command()
            .name("cargo-candle-graph")
            .bin_name("cargo candle-graph")
            .about("Import and analyze candle-graph execution trace files");
        let matches = command.get_matches_from(argv);
        match Self::from_arg_matches(&matches) {
            Ok(cli) => cli,
            Err(error) => error.exit(),
        }
    }
}

/// Bounded machine-readable catalog of every subcommand, for `protocol`.
pub fn command_catalog() -> Vec<serde_json::Value> {
    Cli::command()
        .get_subcommands()
        .map(|subcommand| {
            serde_json::json!({
                "name": subcommand.get_name(),
                "about": subcommand.get_about().map(|about| about.to_string()),
            })
        })
        .collect()
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Emit a complete capability-qualified evidence packet.
    Import(ImportArgs),
    /// Render a standalone HTML visualizer from a trace or verified bundle (requires `visualizer` feature).
    #[cfg(feature = "visualizer")]
    View(ViewArgs),
    /// Emit a profiler summary for a raw trace or verified bundle/profile.
    Summary(SummaryArgs),
    /// Run a typed query against raw-trace or verified bundle evidence.
    Query(QueryArgs),
    /// Emit a bounded first-look overview of a raw trace or verified bundle.
    Overview(OverviewArgs),
    /// Compare a candidate profile run with an explicit baseline.
    Compare(CompareArgs),
    /// Atomically publish a content-addressed evidence bundle and emit its publication receipt.
    Report(ReportArgs),
    /// Deeply verify a published evidence bundle and emit a durable receipt.
    Verify(VerifyArgs),
    /// Emit the versioned schema and command protocol of this tool.
    Protocol(ProtocolArgs),
    /// Reconcile a campaign manifest against published bundles on disk.
    CampaignStatus(CampaignStatusArgs),
    /// Build a cross-run series report from a campaign manifest or explicit bundles.
    Series(SeriesArgs),
}

#[derive(Args, Debug)]
struct ImportArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    trace: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[cfg(feature = "visualizer")]
#[derive(Args, Debug)]
struct ViewArgs {
    /// Trace JSONL file (`candle-graph/trace/10` or readable trace/9), or a verified bundle.
    #[arg(value_name = "TRACE")]
    trace: PathBuf,
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
    /// Nsight artifact directory; rejected for bundle inputs, which already bind their Nsight evidence.
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SummaryArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    /// Exit nonzero (after writing output) unless the capture is structurally valid and complete.
    #[arg(long)]
    require_valid: bool,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct QueryArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(long, value_enum)]
    kind: CliTraceQueryKind,
    /// Exact label filter (kinds: spans, tensors, tensor-stats, gradients).
    #[arg(long, value_name = "S", conflicts_with = "label_prefix")]
    label: Option<String>,
    /// Label-prefix filter (kinds: spans, tensors, tensor-stats, gradients).
    #[arg(long, value_name = "S")]
    label_prefix: Option<String>,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

impl QueryArgs {
    fn filter(&self) -> Option<QueryLabelFilter> {
        self.label
            .clone()
            .map(QueryLabelFilter::Exact)
            .or_else(|| self.label_prefix.clone().map(QueryLabelFilter::Prefix))
    }
}

#[derive(Args, Debug)]
struct OverviewArgs {
    /// Raw trace, finalized bundle/profile directory, or its `trace.jsonl`.
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
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
    /// Exit nonzero (after writing output) when the comparison verdict is ineligible.
    #[arg(long)]
    require_eligible: bool,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ReportArgs {
    /// Raw trace JSONL file (`candle-graph/trace/10` or readable trace/9).
    #[arg(value_name = "TRACE")]
    trace: PathBuf,
    #[arg(long, value_name = "DIR")]
    nsight_dir: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    bundle: PathBuf,
    /// Publication receipt destination (stdout when omitted); must be outside the new bundle.
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct VerifyArgs {
    /// Finalized evidence bundle directory.
    #[arg(value_name = "BUNDLE")]
    bundle: PathBuf,
    /// Also rederive the evidence packet from retained inputs and require an exact match.
    #[arg(long)]
    semantic: bool,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ProtocolArgs {
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct CampaignStatusArgs {
    /// Campaign manifest (`candle-graph/campaign/1`).
    #[arg(long, value_name = "FILE")]
    manifest: PathBuf,
    #[arg(long, short, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SeriesArgs {
    /// Campaign manifest; every planned capture must already be published.
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with = "bundle",
        required_unless_present = "bundle"
    )]
    manifest: Option<PathBuf>,
    /// Explicit ordered verified bundle directories.
    #[arg(long, value_name = "DIR", num_args = 1..)]
    bundle: Vec<PathBuf>,
    /// Restrict tensor-stat and gradient series to labels with this prefix.
    #[arg(long, value_name = "P")]
    label_prefix: Option<String>,
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
