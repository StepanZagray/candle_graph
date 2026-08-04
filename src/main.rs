use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use candle_graph::cli::{self, AnalyzeRequest, ModelRun, OutputFormat, ReportMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Compact summary tree.
    Tree,
    /// Full unified model IR or query response.
    Json,
    /// Standalone interactive HTML visualizer (requires `visualizer` feature).
    #[cfg(feature = "visualizer")]
    Html,
}

#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    about = "Static structure, dtype, and gradient analysis for candle-rs models",
    long_about = "Crate-wide model IR, bounded agent queries, and optional HTML visualization. \
                  Prefer `cargo candle-graph check` for Clippy-like package analysis."
)]
struct Args {
    /// Crate or directory to analyze.
    #[arg(value_name = "DIR")]
    dir: PathBuf,

    /// Optional component root, including private/internal model types.
    #[arg(long, value_name = "NAME")]
    root: Option<String>,

    #[arg(long, value_enum, default_value_t = Format::Tree)]
    format: Format,

    /// Run a bounded agent query: summary, doctor, architecture, components, tensors, …
    #[arg(long, value_name = "KIND")]
    query: Option<String>,

    /// Case-insensitive query selector (name, qualified name, key, or stable id).
    #[arg(long, value_name = "SELECTOR", requires = "query")]
    select: Option<String>,

    /// Destination selector for a `path` query.
    #[arg(long, value_name = "SELECTOR", requires = "query")]
    to: Option<String>,

    /// Maximum number of query records returned.
    #[arg(long, default_value_t = 100, requires = "query")]
    limit: usize,

    /// Deterministic query result offset for paginated drill-down.
    #[arg(long, default_value_t = 0, requires = "query")]
    offset: usize,

    /// Import runtime tensor/gradient observations (requires `runtime` feature).
    #[cfg(feature = "runtime")]
    #[arg(long, value_name = "FILE")]
    runtime_trace: Option<PathBuf>,

    /// Cargo features to resolve while discovering active cfg branches.
    #[arg(long, value_delimiter = ',', value_name = "FEATURE")]
    features: Vec<String>,

    /// Resolve all Cargo features.
    #[arg(long, conflicts_with = "no_default_features")]
    all_features: bool,

    /// Disable Cargo default features during discovery.
    #[arg(long)]
    no_default_features: bool,

    /// Cargo target triple used for metadata filtering and rustc cfg discovery.
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// Cargo package target to analyze (library or binary name).
    #[arg(long, value_name = "NAME")]
    cargo_target: Option<String>,

    /// Enable exploratory architecture/pipeline/optimizer heuristics.
    #[arg(long)]
    heuristic_architecture: bool,

    /// Write the selected report to a file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print finding diagnostics to stderr.
    #[arg(long)]
    diagnostics: bool,

    /// Exit non-zero on proven error findings.
    #[arg(long)]
    strict: bool,

    /// Compare the model fingerprint with a committed baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    /// Atomically create or replace a model fingerprint baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,

    /// Write a multi-file agent audit bundle.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["output", "query"])]
    audit_dir: Option<PathBuf>,

    /// Exit non-zero when proven findings match these rule names (comma-separated).
    #[arg(long, value_name = "RULES", value_delimiter = ',')]
    deny: Vec<String>,

    /// Reuse a cached analysis when `CANDLE_GRAPH_CACHE=1` or this flag is set.
    #[arg(long)]
    cache: bool,

    /// Checkpoint path for `--audit-dir` verification output.
    #[arg(long, value_name = "FILE", requires = "audit_dir")]
    audit_checkpoint: Option<PathBuf>,

    /// VarBuilder root for checkpoint verification inside `--audit-dir`.
    #[arg(long, value_name = "NAME", requires = "audit_dir")]
    audit_verify_root: Option<String>,

    /// Safetensors checkpoint used to verify parameters and propagate tensor dtypes.
    #[arg(long, value_name = "FILE", conflicts_with = "audit_dir")]
    checkpoint: Option<PathBuf>,

    /// VarBuilder root for `--checkpoint` verification (default: first builder or `vb`).
    #[arg(long, value_name = "NAME", requires = "checkpoint")]
    verify_root: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let format = match args.format {
        Format::Json => OutputFormat::Json,
        Format::Tree => OutputFormat::Tree,
        #[cfg(feature = "visualizer")]
        Format::Html => OutputFormat::Html,
    };
    let report = if let Some(kind) = args.query.as_deref() {
        ReportMode::Query {
            kind: kind.to_string(),
            selector: args.select.clone(),
            to: args.to.clone(),
            limit: args.limit,
            offset: args.offset,
            format,
        }
    } else {
        ReportMode::FullIr { format }
    };
    cli::run_model(&ModelRun {
        analyze: AnalyzeRequest {
            path: args.dir,
            cargo: candle_graph::cargo_context::CargoOptions {
                features: args.features,
                all_features: args.all_features,
                no_default_features: args.no_default_features,
                target: args.target,
                package_target: args.cargo_target,
            },
            #[cfg(feature = "runtime")]
            runtime_trace: args.runtime_trace,
            component_root: args.root,
            dataflow: true,
            heuristic_architecture: args.heuristic_architecture,
            use_cache: args.cache,
        },
        report,
        output: args.output,
        diagnostics: args
            .diagnostics
            .then_some(candle_graph::diagnostics::MessageFormat::Human),
        fail_on_warning_error: args.strict,
        deny_rules: args.deny.clone(),
        check_baseline: args.check,
        update_baseline: args.update_baseline,
        audit: args.audit_dir.map(|output_dir| cli::AuditBundle {
            output_dir,
            checkpoint: args.audit_checkpoint,
            verify_root: args.audit_verify_root,
            deny_rules: args.deny.clone(),
            strict: args.strict,
        }),
        checkpoint: args.checkpoint,
        verify_root: args.verify_root,
    })
}
