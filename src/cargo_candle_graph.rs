//! Cargo subcommand entrypoint: `cargo candle-graph …`.
//!
//! Installed as `cargo-candle-graph` so Cargo discovers it as `cargo candle-graph`, analogous to
//! `cargo clippy`. This drives the existing syn/Cargo-metadata analyzer; it is not a rustc lint
//! pass and does not run as a silent side-effect of `cargo build`.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use candle_graph::{
    cargo_context::CargoOptions,
    cli::{self, AnalyzeRequest, AuditBundle, ModelRun, OutputFormat, ReportMode},
    diagnostics::MessageFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliFormat {
    Json,
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliMessageFormat {
    Human,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    name = "cargo-candle-graph",
    bin_name = "cargo candle-graph",
    about = "Clippy-like static analysis for candle-rs model crates",
    long_about = "Analyze candle model architecture, structure, parameters, and tensor/dataflow \
                  facts. Invoked as `cargo candle-graph <command>`, mirroring `cargo clippy`: an \
                  explicit cargo subcommand rather than a silent `cargo build` hook."
)]
struct CargoArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze the package, emit rustc-like diagnostics, and write the full model IR.
    ///
    /// Exits non-zero on proven `Error` findings only (`Confidence::Proven`). Coverage-gap
    /// warnings and Information notes (including `compiler-semantic-evidence`) do not fail.
    Check(CheckArgs),
    /// Emit the full unified model IR without failing on warnings.
    Report(ReportArgs),
    /// Run a bounded agent query against the unified model IR.
    Query(QueryArgs),
    /// Write a multi-file audit bundle for agent / CI consumption.
    Audit(AuditArgs),
    /// Merge static analysis with a runtime v3 profile trace (timings + phase).
    #[cfg(feature = "runtime")]
    Profile(ProfileArgs),
}

#[derive(Parser, Debug)]
struct CheckArgs {
    /// Package directory to analyze. Defaults to the current directory.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    /// Diagnostic format written to stderr.
    #[arg(long, value_enum, default_value_t = CliMessageFormat::Human)]
    message_format: CliMessageFormat,

    /// Model IR output format written to stdout / `--output`.
    #[arg(long, value_enum, default_value_t = CliFormat::Json)]
    format: CliFormat,

    /// Compare the model fingerprint with a committed baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    /// Atomically create or replace a model fingerprint baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    /// Package directory to analyze. Defaults to the current directory.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_enum, default_value_t = CliFormat::Json)]
    format: CliFormat,

    /// Exit non-zero when findings include errors or warnings.
    #[arg(long)]
    strict: bool,

    /// Emit finding diagnostics to stderr.
    #[arg(long, value_enum)]
    message_format: Option<CliMessageFormat>,

    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    /// Query kind (summary, doctor, components, tensors, findings, …).
    kind: String,

    #[command(flatten)]
    common: CommonArgs,

    /// Package directory to analyze. Defaults to the current directory.
    #[arg(long, value_name = "PATH")]
    path: Option<PathBuf>,

    #[arg(long, value_name = "SELECTOR")]
    select: Option<String>,

    #[arg(long, value_name = "SELECTOR")]
    to: Option<String>,

    #[arg(long, default_value_t = 100)]
    limit: usize,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long, value_enum, default_value_t = CliFormat::Json)]
    format: CliFormat,

    #[arg(long)]
    strict: bool,
}

#[derive(Parser, Debug)]
struct AuditArgs {
    /// Output directory for the audit bundle.
    #[arg(value_name = "DIR")]
    output_dir: PathBuf,

    /// Package directory to analyze. Defaults to the current directory.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    /// Optional safetensors checkpoint for verification output.
    #[arg(long, value_name = "FILE")]
    checkpoint: Option<PathBuf>,

    /// VarBuilder root for checkpoint verification.
    #[arg(long, value_name = "NAME")]
    verify_root: Option<String>,

    /// Legacy component root for optional `world-model.json` (e.g. `MyModel`).
    #[arg(long, value_name = "TYPE")]
    legacy_root: Option<String>,

    /// Legacy entrypoint for optional `world-model.json` (e.g. `MyModel::forward`).
    #[arg(long, value_name = "FN")]
    legacy_entry: Option<String>,

    /// Exit non-zero on proven error findings or `--deny` rule matches.
    #[arg(long)]
    strict: bool,
}

#[derive(Parser, Debug)]
#[cfg(feature = "runtime")]
struct ProfileArgs {
    /// Runtime profile trace (`candle-graph/runtime/3` JSON or JSONL).
    #[arg(long, value_name = "FILE")]
    runtime_trace: PathBuf,

    /// Package directory to analyze. Defaults to the current directory.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    /// Write profile query JSON to a file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct CommonArgs {
    /// Path to `Cargo.toml` for the package under analysis.
    #[arg(long, value_name = "PATH")]
    manifest_path: Option<PathBuf>,

    /// Write the primary report to a file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Optional component root, including private/internal model types.
    #[arg(long, value_name = "NAME")]
    root: Option<String>,

    /// Import runtime tensor/gradient observations (JSON or JSONL runtime schema v1).
    #[cfg(feature = "runtime")]
    #[arg(long, value_name = "FILE")]
    runtime_trace: Option<PathBuf>,

    /// Cargo features to resolve while discovering active cfg branches.
    #[arg(long, value_delimiter = ',', value_name = "FEATURE")]
    features: Vec<String>,

    #[arg(long, conflicts_with = "no_default_features")]
    all_features: bool,

    #[arg(long)]
    no_default_features: bool,

    /// Cargo target triple used for metadata filtering and rustc cfg discovery.
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// Cargo package target to analyze (library or binary name).
    #[arg(long, value_name = "NAME")]
    cargo_target: Option<String>,

    /// Enable exploratory architecture/pipeline/optimizer heuristics (tagged Heuristic).
    #[arg(long)]
    heuristic_architecture: bool,

    /// Reuse a cached analysis when `CANDLE_GRAPH_CACHE=1` or this flag is set.
    #[arg(long)]
    cache: bool,

    /// Exit non-zero when proven findings match these rule names (comma-separated).
    #[arg(long, value_name = "RULES", value_delimiter = ',')]
    deny: Vec<String>,
}

fn main() -> Result<()> {
    let args = CargoArgs::parse();
    match args.command {
        Command::Check(check) => run_check(check),
        Command::Report(report) => run_report(report),
        Command::Query(query) => run_query(query),
        Command::Audit(audit) => run_audit(audit),
        #[cfg(feature = "runtime")]
        Command::Profile(profile) => run_profile(profile),
    }
}

fn run_check(args: CheckArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    cli::run_model(&ModelRun {
        analyze: analyze_request(&args.common, path),
        report: ReportMode::FullIr {
            format: output_format(args.format),
        },
        output: args.common.output,
        diagnostics: Some(message_format(args.message_format)),
        fail_on_warning_error: true,
        deny_rules: args.common.deny.clone(),
        check_baseline: args.check,
        update_baseline: args.update_baseline,
        audit: None,
    })
}

fn run_report(args: ReportArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    cli::run_model(&ModelRun {
        analyze: analyze_request(&args.common, path),
        report: ReportMode::FullIr {
            format: output_format(args.format),
        },
        output: args.common.output,
        diagnostics: args.message_format.map(message_format),
        fail_on_warning_error: args.strict,
        deny_rules: args.common.deny.clone(),
        check_baseline: args.check,
        update_baseline: args.update_baseline,
        audit: None,
    })
}

fn run_query(args: QueryArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    cli::run_model(&ModelRun {
        analyze: analyze_request(&args.common, path),
        report: ReportMode::Query {
            kind: args.kind,
            selector: args.select,
            to: args.to,
            limit: args.limit,
            offset: args.offset,
            format: output_format(args.format),
        },
        output: args.common.output,
        diagnostics: None,
        fail_on_warning_error: args.strict,
        deny_rules: args.common.deny.clone(),
        check_baseline: None,
        update_baseline: None,
        audit: None,
    })
}

fn run_audit(args: AuditArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    cli::run_model(&ModelRun {
        analyze: analyze_request(&args.common, path),
        report: ReportMode::FullIr {
            format: OutputFormat::Json,
        },
        output: None,
        diagnostics: None,
        fail_on_warning_error: args.strict,
        deny_rules: args.common.deny.clone(),
        check_baseline: None,
        update_baseline: None,
        audit: Some(AuditBundle {
            output_dir: args.output_dir,
            checkpoint: args.checkpoint,
            verify_root: args.verify_root,
            legacy_root: args
                .legacy_root
                .or(args.common.root.clone()),
            legacy_entry: args.legacy_entry,
            deny_rules: args.common.deny.clone(),
            strict: args.strict,
        }),
    })
}

#[cfg(feature = "runtime")]
fn run_profile(args: ProfileArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    let mut analyze = analyze_request(&args.common, path);
    analyze.runtime_trace = Some(args.runtime_trace);
    cli::run_model(&ModelRun {
        analyze,
        report: ReportMode::Query {
            kind: "profile".into(),
            selector: None,
            to: None,
            limit: 100,
            offset: 0,
            format: OutputFormat::Json,
        },
        output: args.output.or(args.common.output),
        diagnostics: None,
        fail_on_warning_error: false,
        deny_rules: args.common.deny.clone(),
        check_baseline: None,
        update_baseline: None,
        audit: None,
    })
}

fn analyze_request(common: &CommonArgs, path: PathBuf) -> AnalyzeRequest {
    AnalyzeRequest {
        path,
        cargo: CargoOptions {
            features: common.features.clone(),
            all_features: common.all_features,
            no_default_features: common.no_default_features,
            target: common.target.clone(),
            package_target: common.cargo_target.clone(),
        },
        #[cfg(feature = "runtime")]
        runtime_trace: common.runtime_trace.clone(),
        component_root: common.root.clone(),
        dataflow: true,
        heuristic_architecture: common.heuristic_architecture,
        use_cache: common.cache,
    }
}

fn output_format(format: CliFormat) -> OutputFormat {
    match format {
        CliFormat::Json => OutputFormat::Json,
        CliFormat::Tree => OutputFormat::Tree,
    }
}

fn message_format(format: CliMessageFormat) -> MessageFormat {
    match format {
        CliMessageFormat::Human => MessageFormat::Human,
        CliMessageFormat::Json => MessageFormat::Json,
    }
}
