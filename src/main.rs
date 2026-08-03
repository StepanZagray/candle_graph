use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use candle_graph::{
    baseline,
    cli::{self, AnalyzeRequest, ModelRun, OutputFormat, ReportMode},
    dataflow,
    extract::Extractor,
    load, report, verify, viewer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Indented module tree with parameters.
    Tree,
    /// Flat sorted list of parameter keys; the form to diff in CI.
    Keys,
    /// Structured output for agents.
    Json,
    /// Standalone interactive HTML report.
    Html,
}

#[derive(Parser, Debug)]
#[command(
    name = "candle-graph",
    about = "Static structure, dtype, and gradient analysis for candle-rs models",
    long_about = "Reconstructs a candle model's module and parameter tree and, when an entrypoint \
                  is supplied, its expression-level dtype and gradient dataflow. It reads Rust \
                  source without compiling or running the model. Prefer `cargo candle-graph check` \
                  for Clippy-like crate-wide analysis."
)]
struct Args {
    /// Crate or directory to scan for Rust sources.
    #[arg(value_name = "DIR")]
    dir: PathBuf,

    /// Root model type for the legacy single-component view, e.g. `MyModel`.
    #[arg(long, value_name = "TYPE")]
    root: Option<String>,

    /// Constructor to enter through. Defaults to `new`, then `load`, then the first method
    /// taking a VarBuilder.
    #[arg(long, value_name = "FN")]
    ctor: Option<String>,

    /// Function or inherent method to analyze for dtype and gradient flow, e.g. `train` or
    /// `MyModel::forward`.
    #[arg(long, value_name = "FN")]
    entry: Option<String>,

    #[arg(long, value_enum, default_value_t = Format::Tree)]
    format: Format,

    /// Emit the unified crate-wide model IR instead of the legacy single-root report.
    #[arg(long, conflicts_with = "query")]
    model_ir: bool,

    /// Run a bounded agent query: summary, doctor, architecture, cargo, components, component,
    /// modules, composition, pipeline, stages, artifacts, entrypoints, functions, function,
    /// parameters, parameter, tensors, tensor, operations, operation, optimizers, runtime,
    /// findings, path.
    #[arg(long, value_name = "KIND", conflicts_with = "model_ir")]
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

    /// Import runtime tensor/gradient observations (JSON or JSONL runtime schema v1).
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

    /// Cargo package target to analyze (library or binary name). Defaults to the library target,
    /// then the first ordinary binary.
    #[arg(long, value_name = "NAME")]
    cargo_target: Option<String>,

    /// Enable exploratory architecture/pipeline/optimizer heuristics (tagged Heuristic, not proven).
    #[arg(long)]
    heuristic_architecture: bool,

    /// Write the selected report to a file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Cross-check parameter keys against a safetensors checkpoint.
    #[arg(long, value_name = "FILE")]
    verify: Option<PathBuf>,

    /// Which VarBuilder root the checkpoint corresponds to. Defaults to the root model's own
    /// builder. Models with both frozen and trainable builders need one file per root.
    #[arg(long, value_name = "NAME")]
    verify_root: Option<String>,

    /// Show source locations.
    #[arg(long)]
    spans: bool,

    /// Print diagnostics to stderr (always included in `--format json`).
    #[arg(long)]
    diagnostics: bool,

    /// Exit non-zero for unresolved structure, dtype conflicts/risks, or dead trainable params.
    #[arg(long)]
    strict: bool,

    /// Compare the canonical structure/dataflow findings with a committed baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "update_baseline")]
    check: Option<PathBuf>,

    /// Atomically create or replace a canonical structure/dataflow baseline.
    #[arg(long, value_name = "FILE", conflicts_with = "check")]
    update_baseline: Option<PathBuf>,

    /// Write a multi-file agent audit bundle (summary, doctor, model-improvement, …).
    #[arg(long, value_name = "DIR", conflicts_with_all = ["output", "query", "model_ir"])]
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
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.model_ir || args.query.is_some() || args.audit_dir.is_some() {
        return run_model_mode(&args);
    }

    let root = args.root.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--root is required unless --model-ir or --query is used")
    })?;
    let cargo_context =
        candle_graph::cargo_context::CargoContext::discover(&args.dir, &cargo_options(&args)).ok();
    let mut krate = match cargo_context.as_ref() {
        Some(context) => {
            let roots = context.selected_source_roots(args.cargo_target.as_deref())?;
            load::load_from_roots(&args.dir, &roots)?
        }
        None => load::load(&args.dir)?,
    };
    if let Some(context) = cargo_context.as_ref() {
        krate.set_dependency_aliases(context.dependency_aliases.clone());
    }
    if krate.structs.is_empty() {
        anyhow::bail!("no Rust structs found under {}", args.dir.display());
    }

    let candle_nn_version = cargo_context.as_ref().and_then(|context| {
        candle_graph::op_semantics::matched_candle_version(
            context
                .candle_versions
                .get("candle-core")
                .map(String::as_str),
            context.candle_versions.get("candle-nn").map(String::as_str),
        )
        .map(str::to_string)
    });
    let mut structure = Extractor::for_candle_version(&krate, candle_nn_version.as_deref())
        .run(root, args.ctor.as_deref())?;
    let dataflow_graph = args
        .entry
        .as_deref()
        .map(|entry| {
            dataflow::analyze_with_candle_version(&krate, entry, candle_nn_version.as_deref())
        })
        .transpose()
        .with_context(|| {
            format!(
                "analyzing dataflow entrypoint `{}`",
                args.entry.as_deref().unwrap_or_default()
            )
        })?;

    let verify_report = match &args.verify {
        Some(path) => {
            let header = verify::read_header(path)?;
            let root = match &args.verify_root {
                Some(name) => name.clone(),
                None => structure
                    .root
                    .map(|id| structure.instance(id).root.clone())
                    .unwrap_or_default(),
            };
            let known = structure.roots();
            if !known.contains(&root) {
                anyhow::bail!(
                    "unknown VarBuilder root `{root}`; this model has: {}",
                    known.join(", ")
                );
            }
            Some(verify::verify(&mut structure, &header, &root))
        }
        None => None,
    };

    let dataflow_lines = dataflow_graph
        .as_ref()
        .map(|graph| report::dataflow_findings(graph, &krate))
        .unwrap_or_default();

    if let Some(path) = &args.check {
        baseline::check(&structure, path, &dataflow_lines)
            .with_context(|| format!("checking baseline {}", path.display()))?;
    }
    if let Some(path) = &args.update_baseline {
        baseline::update(&structure, path, &dataflow_lines)
            .with_context(|| format!("updating baseline {}", path.display()))?;
        eprintln!("updated baseline {}", path.display());
    }

    let structure_json = report::json(&structure, &krate, verify_report.as_ref());
    let dataflow_json = dataflow_graph
        .as_ref()
        .map(|graph| report::dataflow_json(graph, &krate));
    let rendered = match args.format {
        Format::Tree => {
            let mut text = report::tree(&structure, &krate, args.spans);
            if let Some(graph) = &dataflow_graph {
                text.push_str(&report::dataflow_text(graph, &krate));
            }
            text
        }
        Format::Keys => report::keys(&structure),
        Format::Json => {
            let mut value = structure_json.clone();
            if let Some(dataflow) = &dataflow_json {
                value
                    .as_object_mut()
                    .expect("structure report is an object")
                    .insert("dataflow".to_string(), dataflow.clone());
            }
            serde_json::to_string_pretty(&value)? + "\n"
        }
        Format::Html => viewer::render_html(&structure_json, dataflow_json.as_ref()),
    };
    cli::write_output(args.output.as_deref(), rendered.as_bytes())?;

    if let Some(report) = &verify_report {
        if args.format != Format::Json {
            eprintln!(
                "\ncheckpoint [{}]: {} tensors, {} matched, {} unexpectedly missing, {} unclaimed, \
                 {} skipped (other roots)",
                report.root,
                report.checkpoint_tensors,
                report.matched,
                report.missing_certain.len(),
                report.unclaimed.len(),
                report.skipped_other_root,
            );
            for name in report.missing_certain.iter().take(20) {
                eprintln!("  MISSING (claimed certain): {name}");
            }
            for name in report.missing_conditional.iter().take(20) {
                eprintln!("  absent, as flagged conditional: {name}");
            }
            for name in report.unclaimed.iter().take(20) {
                eprintln!("  unclaimed: {name}");
            }
        }
    }

    if args.diagnostics && args.format != Format::Json {
        for d in &structure.diagnostics {
            eprintln!("{}: {}", krate.file_label(d.span), d.message);
        }
    }

    if args.strict {
        let coverage = structure.coverage();
        let dtype_conflicts = dataflow_graph
            .as_ref()
            .map(|graph| graph.dtype_conflicts.len())
            .unwrap_or_default();
        let dtype_risks = dataflow_graph
            .as_ref()
            .map(|graph| graph.dtype_risks.len())
            .unwrap_or_default();
        let dead_params = dataflow_graph
            .as_ref()
            .map(|graph| graph.dead_params().len())
            .unwrap_or_default();
        if coverage.params_unknown > 0
            || coverage.diagnostics > 0
            || dtype_conflicts > 0
            || dtype_risks > 0
            || dead_params > 0
        {
            anyhow::bail!(
                "strict: {} unknown parameters, {} structure diagnostics, {} dtype conflicts, \
                 {} dtype risks, {} dead trainable parameters",
                coverage.params_unknown,
                coverage.diagnostics,
                dtype_conflicts,
                dtype_risks,
                dead_params,
            );
        }
    }

    Ok(())
}

fn run_model_mode(args: &Args) -> Result<()> {
    if !matches!(args.format, Format::Tree | Format::Json) {
        anyhow::bail!("--model-ir/--query support --format tree or --format json");
    }
    let format = match args.format {
        Format::Json => OutputFormat::Json,
        Format::Tree => OutputFormat::Tree,
        _ => unreachable!(),
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
            path: args.dir.clone(),
            cargo: cargo_options(args),
            #[cfg(feature = "runtime")]
            runtime_trace: args.runtime_trace.clone(),
            component_root: args.root.clone(),
            dataflow: true,
            heuristic_architecture: args.heuristic_architecture,
            use_cache: args.cache,
        },
        report,
        output: args.output.clone(),
        diagnostics: None,
        fail_on_warning_error: args.strict,
        deny_rules: args.deny.clone(),
        check_baseline: args.check.clone(),
        update_baseline: args.update_baseline.clone(),
        audit: args.audit_dir.as_ref().map(|output_dir| cli::AuditBundle {
            output_dir: output_dir.clone(),
            checkpoint: args.audit_checkpoint.clone(),
            verify_root: args.audit_verify_root.clone(),
            legacy_root: args.root.clone(),
            legacy_entry: args.entry.clone(),
            deny_rules: args.deny.clone(),
            strict: args.strict,
        }),
    })
}

fn cargo_options(args: &Args) -> candle_graph::cargo_context::CargoOptions {
    candle_graph::cargo_context::CargoOptions {
        features: args.features.clone(),
        all_features: args.all_features,
        no_default_features: args.no_default_features,
        target: args.target.clone(),
        package_target: args.cargo_target.clone(),
    }
}
