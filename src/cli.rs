//! Shared CLI engine for `candle-graph` and `cargo-candle-graph`.

use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    io::{self, Write},
    path::{Path, PathBuf},
};

use crate::{
    analysis_cache,
    cargo_context::CargoOptions,
    diagnostics::{self, MessageFormat},
    discover::{self, ScanOptions},
    model_baseline,
    model_ir::ModelIr,
    query::{self, QueryKind},
    verify,
};

/// Output format for model IR / query responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Tree,
    #[cfg(feature = "visualizer")]
    Html,
}

/// How to render analyzed model facts.
#[derive(Debug, Clone)]
pub enum ReportMode {
    FullIr {
        format: OutputFormat,
    },
    Query {
        kind: String,
        selector: Option<String>,
        to: Option<String>,
        limit: usize,
        offset: usize,
        format: OutputFormat,
    },
}

/// Inputs for a unified-model scan.
#[derive(Debug, Clone)]
pub struct AnalyzeRequest {
    pub path: PathBuf,
    pub cargo: CargoOptions,
    #[cfg(feature = "runtime")]
    pub runtime_trace: Option<PathBuf>,
    pub component_root: Option<String>,
    pub dataflow: bool,
    pub heuristic_architecture: bool,
    pub use_cache: bool,
}

/// Bundle written by `--audit-dir` / `cargo candle-graph audit`.
#[derive(Debug, Clone)]
pub struct AuditBundle {
    pub output_dir: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub verify_root: Option<String>,
    pub deny_rules: Vec<String>,
    pub strict: bool,
}

/// One invocation of the shared model-mode engine.
#[derive(Debug, Clone)]
pub struct ModelRun {
    pub analyze: AnalyzeRequest,
    pub report: ReportMode,
    pub output: Option<PathBuf>,
    /// When set, finding diagnostics are written to stderr in this format.
    pub diagnostics: Option<MessageFormat>,
    /// Exit non-zero when findings include proven Error defects.
    pub fail_on_warning_error: bool,
    /// Exit non-zero when proven findings match any of these rule names.
    pub deny_rules: Vec<String>,
    pub check_baseline: Option<PathBuf>,
    pub update_baseline: Option<PathBuf>,
    pub audit: Option<AuditBundle>,
    /// Optional safetensors checkpoint merged before report/audit (header verify only).
    pub checkpoint: Option<PathBuf>,
    /// VarBuilder root for safetensors checkpoint verification (default: first builder or `vb`).
    pub verify_root: Option<String>,
}

/// Analyze a crate into the unified model IR, optionally reusing a disk cache.
pub fn analyze_model(request: &AnalyzeRequest) -> Result<ModelIr> {
    if analysis_cache::cache_enabled(request.use_cache) {
        let canonical = request
            .path
            .canonicalize()
            .unwrap_or_else(|_| request.path.clone());
        let key = format!("{canonical:?}|{:?}", request.cargo);
        let cache_path = analysis_cache::cache_path(&key);
        if let Some(cached) = analysis_cache::load(&cache_path)? {
            return Ok(cached);
        }
        let model = analyze_model_uncached(request)?;
        analysis_cache::save(&cache_path, &model)?;
        return Ok(model);
    }
    analyze_model_uncached(request)
}

fn analyze_model_uncached(request: &AnalyzeRequest) -> Result<ModelIr> {
    let options = ScanOptions {
        cargo: request.cargo.clone(),
        #[cfg(feature = "runtime")]
        runtime_trace: request.runtime_trace.clone(),
        component_root: request.component_root.clone(),
        dataflow: request.dataflow,
        heuristic_architecture: request.heuristic_architecture,
    };
    discover::analyze(&request.path, &options)
}

/// Render a report from an already-analyzed model.
pub fn render_report(model: &ModelIr, report: &ReportMode) -> Result<String> {
    match report {
        ReportMode::FullIr { format } => match format {
            OutputFormat::Json => Ok(serde_json::to_string_pretty(model)? + "\n"),
            OutputFormat::Tree => {
                let request = query::QueryRequest::new(query::QueryKind::Summary);
                Ok(query::render_text(&query::execute(model, &request)?))
            }
            #[cfg(feature = "visualizer")]
            OutputFormat::Html => {
                let payload = crate::viewer_projection::project(model);
                Ok(crate::viewer::render_html(&payload))
            }
        },
        ReportMode::Query {
            kind,
            selector,
            to,
            limit,
            offset,
            format,
        } => {
            let mut request = query::QueryRequest::new(kind.parse()?);
            request.selector = selector.clone();
            request.to = to.clone();
            request.limit = *limit;
            request.offset = *offset;
            let response = query::execute(model, &request)?;
            match format {
                OutputFormat::Json => Ok(serde_json::to_string_pretty(&response)? + "\n"),
                OutputFormat::Tree => Ok(query::render_text(&response)),
                #[cfg(feature = "visualizer")]
                OutputFormat::Html => {
                    let payload = crate::viewer_projection::project(model);
                    Ok(crate::viewer::render_html(&payload))
                }
            }
        }
    }
}

fn write_query(model: &ModelIr, kind: QueryKind, path: &Path) -> Result<()> {
    let request = query::QueryRequest::new(kind);
    let response = query::execute(model, &request)?;
    let rendered = serde_json::to_string_pretty(&response)? + "\n";
    write_output(Some(path), rendered.as_bytes())
}

fn write_checkpoint_audit(
    model: &mut ModelIr,
    checkpoint: &Path,
    verify_root: Option<&str>,
    path: &Path,
) -> Result<()> {
    let header = verify::read_header(checkpoint)?;
    let root = verify_root
        .map(str::to_string)
        .or_else(|| {
            model
                .components
                .first()
                .and_then(|c| c.builders.first().map(|b| b.name.clone()))
        })
        .unwrap_or_else(|| "vb".to_string());
    let report = verify::verify_model(model, &header, &root);
    write_output(
        Some(path),
        (serde_json::to_string_pretty(&report)? + "\n").as_bytes(),
    )
}

fn run_audit_bundle(
    model: &mut ModelIr,
    _request: &AnalyzeRequest,
    audit: &AuditBundle,
) -> Result<()> {
    std::fs::create_dir_all(&audit.output_dir)
        .with_context(|| format!("creating {}", audit.output_dir.display()))?;
    write_query(
        model,
        QueryKind::Summary,
        &audit.output_dir.join("summary.json"),
    )?;
    write_query(
        model,
        QueryKind::Doctor,
        &audit.output_dir.join("doctor.json"),
    )?;
    write_query(
        model,
        QueryKind::ModelImprovement,
        &audit.output_dir.join("model-improvement.json"),
    )?;
    write_query(
        model,
        QueryKind::Findings,
        &audit.output_dir.join("findings.json"),
    )?;
    let rendered = render_report(
        model,
        &ReportMode::FullIr {
            format: OutputFormat::Json,
        },
    )?;
    write_output(
        Some(&audit.output_dir.join("model-ir.json")),
        rendered.as_bytes(),
    )?;
    #[cfg(feature = "runtime")]
    if let Some(runtime) = &_request.runtime_trace {
        write_query(
            model,
            QueryKind::Runtime,
            &audit.output_dir.join("runtime.json"),
        )?;
        let _ = runtime;
    }
    #[cfg(feature = "visualizer")]
    {
        let payload = crate::viewer_projection::project(model);
        write_output(
            Some(&audit.output_dir.join("model.html")),
            crate::viewer::render_html(&payload).as_bytes(),
        )?;
    }
    if let Some(checkpoint) = &audit.checkpoint {
        write_checkpoint_audit(
            model,
            checkpoint,
            audit.verify_root.as_deref(),
            &audit.output_dir.join("checkpoint.json"),
        )?;
    }
    Ok(())
}

fn enforce_exit_policy(model: &ModelIr, run: &ModelRun) -> Result<()> {
    if run.fail_on_warning_error && diagnostics::has_proven_defect_findings(model) {
        anyhow::bail!(
            "strict: unified model analysis contains proven error findings \
             (coverage gaps and non-proven warnings do not fail)"
        );
    }
    let denied = diagnostics::denied_findings(model, &run.deny_rules);
    if !denied.is_empty() {
        let rules: Vec<_> = denied.iter().map(|f| f.rule.as_str()).collect();
        anyhow::bail!(
            "deny: proven findings matched blocked rules: {}",
            rules.join(", ")
        );
    }
    if run.audit.as_ref().is_some_and(|a| a.strict)
        && (diagnostics::has_proven_defect_findings(model)
            || diagnostics::has_denied_findings(model, &run.deny_rules))
    {
        anyhow::bail!("audit strict gate failed");
    }
    Ok(())
}

fn apply_checkpoint(
    model: &mut ModelIr,
    checkpoint: &Path,
    verify_root: Option<&str>,
) -> Result<()> {
    let header = verify::read_header(checkpoint)?;
    let root = verify_root
        .map(str::to_string)
        .or_else(|| {
            model
                .components
                .first()
                .and_then(|c| c.builders.first().map(|b| b.name.clone()))
        })
        .unwrap_or_else(|| "vb".to_string());
    verify::verify_model(model, &header, &root);
    crate::dtype_propagate::propagate_tensor_dtypes(model, &HashMap::new());
    model.normalize();
    Ok(())
}

/// Run a full model-mode invocation: analyze, optional baseline, report, diagnostics, exit policy.
pub fn run_model(run: &ModelRun) -> Result<()> {
    let mut model = analyze_model(&run.analyze)?;

    if let Some(path) = &run.update_baseline {
        model_baseline::update(&model, path)
            .with_context(|| format!("updating model baseline {}", path.display()))?;
        eprintln!("updated model baseline {}", path.display());
    }
    if let Some(path) = &run.check_baseline {
        model_baseline::check(&model, path)
            .with_context(|| format!("checking model baseline {}", path.display()))?;
    }

    if run.audit.is_none() {
        if let Some(checkpoint) = &run.checkpoint {
            apply_checkpoint(&mut model, checkpoint, run.verify_root.as_deref())?;
        }
    }

    if let Some(audit) = &run.audit {
        run_audit_bundle(&mut model, &run.analyze, audit)?;
    }

    let rendered = render_report(&model, &run.report)?;
    if run.audit.is_none() || run.output.is_some() {
        write_output(run.output.as_deref(), rendered.as_bytes())?;
    }

    if let Some(format) = run.diagnostics {
        let diagnostics = diagnostics::from_model(&model);
        let text = diagnostics::render(&diagnostics, format);
        if !text.is_empty() {
            eprint!("{text}");
        }
    }

    enforce_exit_policy(&model, run)?;
    Ok(())
}

/// Resolve a package path from an optional directory / manifest path.
pub fn resolve_package_path(path: Option<&Path>, manifest_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(manifest) = manifest_path {
        if manifest
            .file_name()
            .is_some_and(|name| name == "Cargo.toml")
        {
            return Ok(manifest
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")));
        }
        return Ok(manifest.to_path_buf());
    }
    Ok(path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".")))
}

pub fn write_output(path: Option<&Path>, bytes: &[u8]) -> Result<()> {
    match path {
        Some(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        }
        None => io::stdout().lock().write_all(bytes)?,
    }
    Ok(())
}
