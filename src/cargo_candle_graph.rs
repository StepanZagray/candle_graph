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
    #[cfg(feature = "visualizer")]
    Html,
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
    Check(CheckArgs),
    /// Emit the full unified model IR without failing on warnings.
    Report(ReportArgs),
    /// Run a bounded agent query against the unified model IR.
    Query(QueryArgs),
    /// Write a multi-file audit bundle for agent / CI consumption.
    Audit(AuditArgs),
    /// Emit a standalone HTML visualizer (requires `visualizer` feature).
    #[cfg(feature = "visualizer")]
    View(ViewArgs),
    /// Merge static analysis with a runtime v3 profile trace (requires `runtime` feature).
    #[cfg(feature = "runtime")]
    Profile(ProfileArgs),
}

#[derive(Parser, Debug)]
struct CheckArgs {
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_enum, default_value_t = CliMessageFormat::Human)]
    message_format: CliMessageFormat,

    #[arg(long, value_enum, default_value_t = CliFormat::Json)]
    format: CliFormat,

    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_enum, default_value_t = CliFormat::Json)]
    format: CliFormat,

    #[arg(long)]
    strict: bool,

    #[arg(long, value_enum)]
    message_format: Option<CliMessageFormat>,

    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QueryArgs {
    kind: String,

    #[command(flatten)]
    common: CommonArgs,

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
    #[arg(value_name = "DIR")]
    output_dir: PathBuf,

    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_name = "FILE")]
    checkpoint: Option<PathBuf>,

    #[arg(long, value_name = "NAME")]
    verify_root: Option<String>,

    #[arg(long)]
    strict: bool,
}

#[derive(Parser, Debug)]
#[cfg(feature = "visualizer")]
struct ViewArgs {
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_name = "FILE")]
    checkpoint: Option<PathBuf>,

    #[arg(long, value_name = "NAME")]
    verify_root: Option<String>,
}

#[derive(Parser, Debug)]
#[cfg(feature = "runtime")]
struct ProfileArgs {
    #[arg(long, value_name = "FILE")]
    runtime_trace: PathBuf,

    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct CommonArgs {
    #[arg(long, value_name = "PATH")]
    manifest_path: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    #[arg(long, value_name = "NAME")]
    root: Option<String>,

    #[cfg(feature = "runtime")]
    #[arg(long, value_name = "FILE")]
    runtime_trace: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', value_name = "FEATURE")]
    features: Vec<String>,

    #[arg(long, conflicts_with = "no_default_features")]
    all_features: bool,

    #[arg(long)]
    no_default_features: bool,

    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    #[arg(long, value_name = "NAME")]
    cargo_target: Option<String>,

    #[arg(long)]
    heuristic_architecture: bool,

    #[arg(long)]
    cache: bool,

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
        #[cfg(feature = "visualizer")]
        Command::View(view) => run_view(view),
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
        checkpoint: None,
        verify_root: None,
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
        checkpoint: None,
        verify_root: None,
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
        checkpoint: None,
        verify_root: None,
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
            deny_rules: args.common.deny.clone(),
            strict: args.strict,
        }),
        checkpoint: None,
        verify_root: None,
    })
}

#[cfg(feature = "visualizer")]
fn run_view(args: ViewArgs) -> Result<()> {
    let path =
        cli::resolve_package_path(args.path.as_deref(), args.common.manifest_path.as_deref())?;
    cli::run_model(&ModelRun {
        analyze: analyze_request(&args.common, path),
        report: ReportMode::FullIr {
            format: OutputFormat::Html,
        },
        output: args.common.output,
        diagnostics: None,
        fail_on_warning_error: false,
        deny_rules: args.common.deny.clone(),
        check_baseline: None,
        update_baseline: None,
        audit: None,
        checkpoint: args.checkpoint,
        verify_root: args.verify_root,
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
        checkpoint: None,
        verify_root: None,
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
        #[cfg(feature = "visualizer")]
        CliFormat::Html => OutputFormat::Html,
    }
}

fn message_format(format: CliMessageFormat) -> MessageFormat {
    match format {
        CliMessageFormat::Human => MessageFormat::Human,
        CliMessageFormat::Json => MessageFormat::Json,
    }
}
