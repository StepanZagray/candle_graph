//! Crate-wide model discovery and unified IR assembly.
//!
//! Public API boundaries and `VarBuilder` constructors provide conservative component candidates.
//! Architecture, pipeline, artifact, and optimizer relationships require compiler-resolved value
//! flow; the former syntax/name-based implementation is quarantined until that frontend exists.
//! Optional runtime traces refine (but never erase) static facts.

#[cfg(feature = "runtime")]
use std::collections::BTreeMap;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
#[cfg(feature = "runtime")]
use std::path::PathBuf;

#[cfg(feature = "runtime")]
use anyhow::Context;
use anyhow::Result;
use quote::ToTokens;
use syn::visit::{self, Visit};

use crate::cargo_context::{CargoContext, CargoOptions};
use crate::dataflow::{self, GradState, NodeKind, NumericImpact};
use crate::extract::Extractor;
use crate::ir::{Acquisition, Certainty, CheckpointMatch};
use crate::load::{self, Crate, ImplFn, StructDef};
use crate::model_ir::{
    ArchitectureEdge, Artifact, ArtifactKind, AssemblySite, BuilderNamespace, BuilderRole,
    BuilderSourceKind, CargoSummary, Component, Confidence, DeviceFact, Evidence, EvidenceKind,
    Finding, FindingSeverity, Function, FunctionParameter, LayoutFact, ModelIr, Module, Operation,
    OptimizerMembership, Parameter, ParameterRole, PipelineStage, ShapeFact, StableId,
    StageDispatchKind, StageKind, TensorContract, TensorRole, Visibility,
};
#[cfg(feature = "runtime")]
use crate::model_ir::{EdgeTimingSummary, RuntimeSummary, TimingStats};
use crate::op_semantics::{self};
#[cfg(feature = "runtime")]
use crate::runtime::{self, ExpectedIdentity, GradientState, RuntimeTrace};

#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    pub cargo: CargoOptions,
    #[cfg(feature = "runtime")]
    pub runtime_trace: Option<PathBuf>,
    /// Optional component root, including private/internal model types.
    pub component_root: Option<String>,
    /// Analyze expression graphs for source entrypoints. Disable only for a fast symbol scan.
    pub dataflow: bool,
    /// Enable name/order-derived architecture, pipeline, artifact, and optimizer heuristics.
    /// All emitted relationships are tagged `Heuristic` and should not be treated as proven facts.
    pub heuristic_architecture: bool,
}

/// Analyze a Cargo crate or source directory into the unified model IR.
pub fn analyze(path: impl AsRef<Path>, options: &ScanOptions) -> Result<ModelIr> {
    let path = path.as_ref();
    let scan_root = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    let mut options = options.clone();
    let _stripped = options.cargo.strip_candle_graph_features();
    let cargo_result = CargoContext::discover(&scan_root, &options.cargo);
    let mut krate = match cargo_result.as_ref() {
        Ok(context) => {
            let roots = context.selected_source_roots(options.cargo.package_target.as_deref())?;
            load::load_from_roots(&scan_root, &roots)?
        }
        Err(error) if manifest_discovery_failed(error) => load::load(&scan_root)?,
        Err(error) => return Err(enrich_cargo_error(error)),
    };
    if let Ok(context) = cargo_result.as_ref() {
        krate.set_dependency_aliases(context.dependency_aliases.clone());
    }
    if krate.all_structs().next().is_none() {
        if let Err(error) = cargo_result.as_ref() {
            anyhow::bail!(
                "no Rust structs found under {} ({error:#})",
                scan_root.display()
            );
        }
        anyhow::bail!("no Rust structs found under {}", scan_root.display());
    }

    let analysis_id = match cargo_result.as_ref() {
        Ok(context) => StableId::new(
            "analysis",
            [cargo_build_id(
                context,
                options.cargo.package_target.as_deref(),
            )],
        ),
        Err(_) => StableId::new("analysis", [canonical_label(&scan_root)]),
    };
    let mut model = ModelIr::empty(analysis_id);

    let cargo = match cargo_result {
        Ok(context) => {
            model.cargo = Some(cargo_summary(
                &context,
                options.cargo.package_target.as_deref(),
            ));
            Some(context)
        }
        Err(error) => {
            push_finding(
                &mut model,
                "cargo-context",
                FindingSeverity::Warning,
                Confidence::Proven,
                format!("Cargo context unavailable: {error:#}"),
                None,
                Vec::new(),
            );
            None
        }
    };
    for diagnostic in &krate.diagnostics {
        push_finding(
            &mut model,
            "source-load",
            FindingSeverity::Warning,
            Confidence::Unknown,
            format!("incomplete source analysis: {}", diagnostic.message),
            Some(diagnostic.path.clone()),
            Vec::new(),
        );
    }

    let function_lookup = build_functions(&krate, cargo.as_ref(), &mut model);
    link_calls(&krate, &function_lookup, &mut model);
    discover_components(
        &krate,
        cargo.as_ref(),
        options.component_root.as_deref(),
        options.heuristic_architecture,
        &mut model,
    );
    discover_composition_edges(&krate, &mut model);
    discover_assembly_sites(&krate, &mut model);
    add_contracts(&krate, &mut model);
    if options.heuristic_architecture {
        discover_architecture_edges(&krate, &mut model);
        discover_subprocess_pipeline(&krate, &function_lookup, &mut model);
        discover_pipeline_and_artifacts(&krate, &function_lookup, &mut model);
        discover_optimizers(&krate, &mut model);
        apply_optimizer_roles(&mut model);
    } else {
        push_finding(
            &mut model,
            "compiler-semantic-evidence",
            FindingSeverity::Information,
            Confidence::Unknown,
            "architecture, pipeline, artifact, and optimizer relationships are unavailable until \
             compiler-resolved value-flow evidence is implemented; use --heuristic-architecture \
             for exploratory name- and call-order leads"
                .to_string(),
            None,
            Vec::new(),
        );
    }

    if options.dataflow {
        add_dataflow(&krate, &mut model);
        let builder_dtypes = crate::dtype_propagate::infer_builder_default_dtypes(&krate);
        crate::dtype_propagate::propagate_tensor_dtypes(&mut model, &builder_dtypes);
    }

    #[cfg(feature = "runtime")]
    if let Some(path) = options.runtime_trace.as_deref() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading runtime trace {}", path.display()))?;
        let trace = runtime::parse(&text)
            .with_context(|| format!("parsing runtime trace {}", path.display()))?;
        merge_runtime(&mut model, &trace);
    }

    if let Some(cargo) = cargo {
        flag_candle_semantics_version(&mut model, &cargo);
    }
    model.normalize();
    Ok(model)
}

fn manifest_discovery_failed(error: &anyhow::Error) -> bool {
    let msg = format!("{error:#}");
    msg.contains("Cargo.toml not found") || msg.contains("failed to locate Cargo.toml")
}

fn enrich_cargo_error(error: &anyhow::Error) -> anyhow::Error {
    let msg = format!("{error:#}");
    if msg.contains("does not contain this feature") {
        anyhow::Error::msg(format!(
            "{error:#}\n\
             `--features` selects Cargo features on the model crate being analyzed. \
             Names like `static`, `visualizer`, `runtime`, and `all` refer to candle-graph itself \
             (enable them when installing/building candle-graph, e.g. \
             `cargo install --path ../candle_graph --features all`). \
             For Tofy, use the `.cargo/config.toml` alias or match your GPU build with \
             `--features cuda` / `--features cudnn`."
        ))
    } else {
        anyhow::Error::msg(msg)
    }
}

fn canonical_label(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn selected_target(context: &CargoContext, requested_target: Option<&str>) -> Option<String> {
    requested_target.map(str::to_string).or_else(|| {
        context
            .targets
            .iter()
            .find(|target| target.kind.iter().any(|kind| kind == "lib"))
            .or_else(|| {
                context
                    .targets
                    .iter()
                    .find(|target| target.kind.iter().any(|kind| kind == "bin"))
            })
            .map(|target| target.name.clone())
    })
}

fn cargo_build_id(context: &CargoContext, requested_target: Option<&str>) -> String {
    let target = selected_target(context, requested_target).unwrap_or_else(|| "unknown".into());
    let mut identity = String::from("candle-graph/build/1\0");
    for part in [
        context.package_name.as_str(),
        context.package_version.as_str(),
        target.as_str(),
    ] {
        identity.push_str(part);
        identity.push('\0');
    }
    for feature in &context.active_features {
        identity.push_str("feature=");
        identity.push_str(feature);
        identity.push('\0');
    }
    for cfg in &context.cfgs {
        identity.push_str("cfg=");
        identity.push_str(cfg);
        identity.push('\0');
    }
    for (package, version) in &context.candle_versions {
        identity.push_str("candle=");
        identity.push_str(package);
        identity.push('@');
        identity.push_str(version);
        identity.push('\0');
    }

    // FNV-1a is deliberately fixed here rather than relying on Rust's unstable DefaultHasher.
    let hash = identity.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("candle-graph/build/1:{hash:016x}")
}

fn cargo_summary(context: &CargoContext, requested_target: Option<&str>) -> CargoSummary {
    CargoSummary {
        build_id: cargo_build_id(context, requested_target),
        workspace_root: context.workspace_root.to_string_lossy().into_owned(),
        manifest_path: context.manifest_path.to_string_lossy().into_owned(),
        package_name: context.package_name.clone(),
        package_version: context.package_version.clone(),
        selected_target: selected_target(context, requested_target),
        active_features: context.active_features.clone(),
        active_cfg: context.cfgs.clone(),
        candle_packages: context.candle_versions.clone(),
    }
}

fn build_functions(
    krate: &Crate,
    cargo: Option<&CargoContext>,
    model: &mut ModelIr,
) -> HashMap<String, StableId> {
    let mut lookup = HashMap::new();
    for func in krate.all_functions().chain(krate.all_methods()) {
        let cfg_active = cargo.and_then(|context| {
            crate::cargo_context::cfg_predicates_active(&func.cfg_predicates, &context.cfgs)
        });
        if cfg_active == Some(false) {
            continue;
        }
        let id = function_id(func);
        lookup.insert(func.qualified_name.clone(), id.clone());
        let tensor_signature = func.param_types.iter().any(|ty| is_tensor_type(ty))
            || is_tensor_type(&func.return_type);
        let is_loss = has_explicit_candle_loss_call(krate, func);
        // A public tensor boundary is a code-derived entrypoint candidate. Its spelling carries no
        // semantics; only its trait identity or public tensor signature matters here.
        let is_entrypoint =
            is_candle_module_entry(func) || is_loss || tensor_signature && func.visibility == "pub";
        model.functions.push(Function {
            id,
            name: func.fn_name.clone(),
            qualified_name: func.qualified_name.clone(),
            owner_type: (!func.qualified_type_name.is_empty())
                .then(|| func.qualified_type_name.clone()),
            visibility: visibility(&func.visibility),
            parameters: func
                .params
                .iter()
                .zip(&func.param_types)
                .map(|(name, type_name)| FunctionParameter {
                    name: name.clone(),
                    type_name: type_name.clone(),
                })
                .collect(),
            return_type: (func.return_type != "()").then(|| func.return_type.clone()),
            cfg_predicates: func.cfg_predicates.clone(),
            cfg_active,
            source: krate.file_label(func.span),
            calls: Vec::new(),
            tensor_inputs: Vec::new(),
            tensor_outputs: Vec::new(),
            is_entrypoint,
            is_loss,
            execution_phases: Vec::new(),
        });
    }
    lookup
}

fn is_candle_module_entry(func: &ImplFn) -> bool {
    let Some(trait_name) = func.trait_name.as_deref() else {
        return false;
    };
    let trait_leaf = trait_name.rsplit("::").next().unwrap_or(trait_name);
    matches!(
        (trait_leaf, func.fn_name.as_str()),
        ("Module", "forward")
            | ("ModuleT", "forward_t")
            | ("ModuleWithArgs", "forward")
            | ("ModuleTWithArgs", "forward_t")
    )
}

fn has_explicit_candle_loss_call(krate: &Crate, func: &ImplFn) -> bool {
    let mut collector = CallCollector::default();
    collector.visit_block(&func.block);
    collector.calls.iter().any(|call| {
        let segments = call.split("::").map(str::to_string).collect::<Vec<_>>();
        let resolved = krate
            .resolve_import_path(&func.module_path, &segments)
            .join("::");
        resolved
            .strip_prefix("candle_nn::loss::")
            .is_some_and(|leaf| {
                matches!(
                    leaf,
                    "nll" | "cross_entropy" | "mse" | "binary_cross_entropy_with_logit" | "huber"
                )
            })
    })
}

fn link_calls(krate: &Crate, lookup: &HashMap<String, StableId>, model: &mut ModelIr) {
    let mut by_bare: HashMap<String, Vec<StableId>> = HashMap::new();
    for function in &model.functions {
        by_bare
            .entry(function.name.clone())
            .or_default()
            .push(function.id.clone());
    }
    let index: HashMap<StableId, usize> = model
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.id.clone(), index))
        .collect();

    for func in krate.all_functions().chain(krate.all_methods()) {
        let mut collector = CallCollector::default();
        collector.visit_block(&func.block);
        let caller = function_id(func);
        let Some(&caller_index) = index.get(&caller) else {
            continue;
        };
        let mut calls = Vec::new();
        for call in collector.calls {
            if let Some(id) = resolve_call(&call, func, lookup, &by_bare) {
                calls.push(id);
            }
        }
        calls.sort();
        calls.dedup();
        model.functions[caller_index].calls = calls;
    }
}

fn resolve_call(
    call: &str,
    caller: &ImplFn,
    exact: &HashMap<String, StableId>,
    bare: &HashMap<String, Vec<StableId>>,
) -> Option<StableId> {
    let clean = call.trim_start_matches("crate::");
    for candidate in [
        clean.to_string(),
        qualify(&caller.module_path, clean),
        clean
            .strip_prefix("self::")
            .map(|rest| qualify(&caller.module_path, rest))
            .unwrap_or_default(),
    ] {
        if let Some(id) = exact.get(&candidate) {
            return Some(id.clone());
        }
    }
    if clean.contains("::") {
        let suffix = format!("::{clean}");
        let mut matches = exact
            .iter()
            .filter(|(name, _)| name.ends_with(&suffix))
            .map(|(_, id)| id);
        let first = matches.next().cloned();
        if first.is_some() && matches.next().is_none() {
            return first;
        }
    }
    let leaf = clean.rsplit("::").next().unwrap_or(clean);
    match bare.get(leaf).map(Vec::as_slice) {
        Some([id]) => Some(id.clone()),
        _ => None,
    }
}

fn discover_architecture_edges(krate: &Crate, model: &mut ModelIr) {
    let component_by_type: HashMap<String, StableId> = model
        .components
        .iter()
        .flat_map(|component| {
            [
                (component.name.clone(), component.id.clone()),
                (component.qualified_name.clone(), component.id.clone()),
            ]
        })
        .collect();

    let mut seen = HashSet::new();
    for function in krate.all_functions().chain(krate.all_methods()) {
        if !is_production_source(krate, function) {
            continue;
        }
        let owner_fields = krate
            .struct_candidates(&function.qualified_type_name)
            .into_iter()
            .next()
            .map(|owner| {
                owner
                    .fields
                    .iter()
                    .filter_map(|field| {
                        component_by_type
                            .get(&field.ty.base)
                            .cloned()
                            .map(|component| (field.name.clone(), component))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut collector = ComponentFlowCollector {
            component_by_type: &component_by_type,
            owner_fields,
            locals: HashMap::new(),
            sequence: Vec::new(),
        };
        let signature_sequence: Vec<StableId> = function
            .param_types
            .iter()
            .filter_map(|type_name| {
                type_base_from_text(type_name)
                    .and_then(|base| component_by_type.get(&base).cloned())
            })
            .fold(Vec::new(), |mut sequence, component| {
                if sequence.last() != Some(&component) {
                    sequence.push(component);
                }
                sequence
            });
        collector.visit_block(&function.block);
        let from_signature = signature_sequence.len() > collector.sequence.len();
        let sequence = if from_signature {
            signature_sequence
        } else {
            collector.sequence
        };
        for pair in sequence.windows(2) {
            if pair[0] == pair[1] {
                continue;
            }
            let key = (pair[0].clone(), pair[1].clone(), function_id(function));
            if !seen.insert(key.clone()) {
                continue;
            }
            model.architecture_edges.push(ArchitectureEdge {
                id: StableId::new(
                    "architecture-edge",
                    [key.0 .0.as_str(), key.1 .0.as_str(), key.2 .0.as_str()],
                ),
                from: key.0,
                to: key.1,
                via_function: key.2,
                source: krate.file_label(function.span),
                evidence: vec![heuristic_source_evidence(
                    krate.file_label(function.span),
                    if from_signature {
                        "component-typed parameters establish this interface order"
                    } else {
                        "typed component receiver calls occur in this source order"
                    },
                )],
            });
        }
    }
}

fn infer_builder_role(name: &str) -> (BuilderRole, Confidence) {
    let lower = name.to_ascii_lowercase();
    if lower.contains("train") || lower == "adapter_vb" {
        return (BuilderRole::Trainable, Confidence::Heuristic);
    }
    if lower.contains("base")
        || lower.contains("frozen")
        || lower.contains("mmap")
        || lower.contains("pretrained")
    {
        return (BuilderRole::Frozen, Confidence::Heuristic);
    }
    if lower.contains("state") || lower.contains("running") {
        return (BuilderRole::State, Confidence::Heuristic);
    }
    (BuilderRole::Unknown, Confidence::Unknown)
}

fn discover_composition_edges(krate: &Crate, model: &mut ModelIr) {
    let component_by_type: HashMap<String, StableId> = model
        .components
        .iter()
        .flat_map(|component| {
            [
                (component.name.clone(), component.id.clone()),
                (component.qualified_name.clone(), component.id.clone()),
            ]
        })
        .collect();

    let mut seen = HashSet::new();
    for component in model.components.clone() {
        let Some(def) = krate
            .struct_candidates(&component.name)
            .into_iter()
            .find(|def| def.qualified_name == component.qualified_name)
        else {
            continue;
        };
        for field in &def.fields {
            let child = unique_qualified_struct(krate, &field.ty.base)
                .and_then(|qualified| component_by_type.get(&qualified).cloned())
                .or_else(|| component_by_type.get(&field.ty.base).cloned());
            let Some(child) = child else {
                continue;
            };
            if child == component.id {
                continue;
            }
            let key = (component.id.clone(), child.clone(), field.name.clone());
            if !seen.insert(key.clone()) {
                continue;
            }
            model.architecture_edges.push(ArchitectureEdge {
                id: StableId::new(
                    "composition-edge",
                    [
                        component.id.0.as_str(),
                        child.0.as_str(),
                        field.name.as_str(),
                    ],
                ),
                from: component.id.clone(),
                to: child,
                via_function: component.constructor.clone(),
                source: krate.file_label(field.span),
                evidence: vec![Evidence {
                    kind: EvidenceKind::Source,
                    confidence: Confidence::Heuristic,
                    source: Some(krate.file_label(field.span)),
                    detail: format!(
                        "struct field `{}` embeds component type `{}`",
                        field.name, field.ty.base
                    ),
                }],
            });
        }
    }
}

fn discover_assembly_sites(krate: &Crate, model: &mut ModelIr) {
    let constructor_functions: HashMap<StableId, &Function> = model
        .functions
        .iter()
        .map(|candidate| (candidate.id.clone(), candidate))
        .collect();
    let specs: Vec<ConstructorSpec> = model
        .components
        .iter()
        .filter_map(|component| {
            let constructor = constructor_functions.get(&component.constructor)?;
            let builders = constructor
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, parameter)| parameter.type_name.contains("VarBuilder"))
                .zip(component.builders.iter())
                .map(|((index, _), builder)| (index, builder.name.clone()))
                .collect();
            Some(ConstructorSpec {
                component: component.id.clone(),
                owner: component.name.clone(),
                builders,
            })
        })
        .collect();
    if specs.is_empty() {
        return;
    }

    for func in krate.all_functions().chain(krate.all_methods()) {
        if !is_production_source(krate, func) {
            continue;
        }
        let function_id = function_id(func);
        let mut collector = AssemblyCollector {
            specs: &specs,
            function_id: function_id.clone(),
            function_name: func.qualified_name.clone(),
            source: krate.file_label(func.span),
            varmap_checkpoints: HashMap::new(),
            vb_bindings: HashMap::new(),
            sites: Vec::new(),
        };
        collector.visit_block(&func.block);
        for site in &mut collector.sites {
            if site.checkpoint_load.is_none() {
                site.checkpoint_load = site
                    .varmap
                    .as_ref()
                    .and_then(|varmap| collector.varmap_checkpoints.get(varmap).cloned());
            }
        }
        model.assembly_sites.extend(collector.sites);
    }
}

struct AssemblyCollector<'a> {
    specs: &'a [ConstructorSpec],
    function_id: StableId,
    function_name: String,
    source: String,
    varmap_checkpoints: HashMap<String, String>,
    vb_bindings: HashMap<String, BuilderExpressionFacts>,
    sites: Vec<AssemblySite>,
}

impl AssemblyCollector<'_> {
    fn resolve_builder_facts(&self, expression: &syn::Expr) -> BuilderExpressionFacts {
        match expression {
            syn::Expr::Path(path) if path.path.segments.len() == 1 => self
                .vb_bindings
                .get(&path.path.segments[0].ident.to_string())
                .cloned()
                .unwrap_or_else(empty_builder_facts),
            _ => builder_expression_facts(expression, &self.vb_bindings),
        }
    }
}

impl<'ast> Visit<'ast> for AssemblyCollector<'_> {
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let (Some(name), Some(init)) = (pat_ident(&node.pat), node.init.as_ref()) {
            let facts = builder_expression_facts(&init.expr, &self.vb_bindings);
            if facts.source_kind != BuilderSourceKind::Unknown
                || !facts.prefix_chain.is_empty()
                || facts.varmap.is_some()
            {
                self.vb_bindings.insert(name, facts);
            }
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*node.func {
            let leaf = path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_default();
            if leaf == "load_varmap_checked" || leaf.ends_with("load_varmap_checked") {
                if let (Some(varmap), Some(checkpoint)) = (
                    node.args.first().and_then(load_varmap_target),
                    node.args
                        .get(1)
                        .map(|arg| arg.to_token_stream().to_string()),
                ) {
                    self.varmap_checkpoints.insert(varmap, checkpoint);
                }
            }
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            if let Some(owner) = segments.get(segments.len().saturating_sub(2)) {
                for spec in self.specs.iter().filter(|spec| &spec.owner == owner) {
                    for (index, builder_root) in &spec.builders {
                        let Some(argument) = node.args.iter().nth(*index) else {
                            continue;
                        };
                        let facts = self.resolve_builder_facts(argument);
                        let varmap = facts.varmap.clone();
                        let checkpoint = varmap
                            .as_ref()
                            .and_then(|name| self.varmap_checkpoints.get(name).cloned());
                        self.sites.push(AssemblySite {
                            id: StableId::new(
                                "assembly-site",
                                [
                                    self.function_id.0.as_str(),
                                    spec.component.0.as_str(),
                                    builder_root.as_str(),
                                    &facts.prefix_chain.join("/"),
                                ],
                            ),
                            function: self.function_id.clone(),
                            function_name: self.function_name.clone(),
                            component: spec.component.clone(),
                            component_name: spec.owner.clone(),
                            builder_root: builder_root.clone(),
                            prefix_chain: facts.prefix_chain,
                            varmap,
                            source_kind: facts.source_kind,
                            role: facts.role,
                            checkpoint_load: checkpoint,
                            source: self.source.clone(),
                            evidence: vec![Evidence {
                                kind: EvidenceKind::Source,
                                confidence: Confidence::Heuristic,
                                source: Some(self.source.clone()),
                                detail: format!(
                                    "`{owner}::new` wired through `{builder_root}` in `{}`",
                                    self.function_name
                                ),
                            }],
                        });
                    }
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

#[derive(Clone)]
struct BuilderExpressionFacts {
    prefix_chain: Vec<String>,
    varmap: Option<String>,
    source_kind: BuilderSourceKind,
    role: BuilderRole,
}

fn empty_builder_facts() -> BuilderExpressionFacts {
    BuilderExpressionFacts {
        prefix_chain: Vec::new(),
        varmap: None,
        source_kind: BuilderSourceKind::Unknown,
        role: BuilderRole::Unknown,
    }
}

fn builder_expression_facts(
    expression: &syn::Expr,
    vb_bindings: &HashMap<String, BuilderExpressionFacts>,
) -> BuilderExpressionFacts {
    match expression {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => vb_bindings
            .get(&path.path.segments[0].ident.to_string())
            .cloned()
            .unwrap_or_else(empty_builder_facts),
        syn::Expr::MethodCall(call) if call.method == "pp" => {
            let mut facts = builder_expression_facts(&call.receiver, vb_bindings);
            if let Some(syn::Expr::Lit(lit)) = call.args.first() {
                if let syn::Lit::Str(text) = &lit.lit {
                    facts.prefix_chain.insert(0, text.value());
                }
            }
            facts
        }
        syn::Expr::Call(call) => {
            let leaf = call_leaf_name(&call.func);
            match leaf.as_deref() {
                Some("from_varmap") => BuilderExpressionFacts {
                    prefix_chain: Vec::new(),
                    varmap: call.args.first().and_then(expr_identifier).or_else(|| {
                        call.args.first().and_then(|arg| match arg {
                            syn::Expr::Reference(reference) => expr_identifier(&reference.expr),
                            _ => None,
                        })
                    }),
                    source_kind: BuilderSourceKind::VarMap,
                    role: BuilderRole::Trainable,
                },
                Some("from_mmaped_safetensors") | Some("from_buffered_safetensors") => {
                    BuilderExpressionFacts {
                        prefix_chain: Vec::new(),
                        varmap: None,
                        source_kind: if leaf.as_deref() == Some("from_mmaped_safetensors") {
                            BuilderSourceKind::MmapSafetensors
                        } else {
                            BuilderSourceKind::BufferedSafetensors
                        },
                        role: BuilderRole::Frozen,
                    }
                }
                Some("from_tensors") => BuilderExpressionFacts {
                    prefix_chain: Vec::new(),
                    varmap: None,
                    source_kind: BuilderSourceKind::FromTensors,
                    role: BuilderRole::Frozen,
                },
                _ => BuilderExpressionFacts {
                    prefix_chain: Vec::new(),
                    varmap: varmap_sources(expression, &HashMap::new())
                        .into_iter()
                        .next(),
                    source_kind: BuilderSourceKind::Unknown,
                    role: BuilderRole::Unknown,
                },
            }
        }
        syn::Expr::Reference(reference) => builder_expression_facts(&reference.expr, vb_bindings),
        syn::Expr::Try(value) => builder_expression_facts(&value.expr, vb_bindings),
        syn::Expr::Await(value) => builder_expression_facts(&value.base, vb_bindings),
        syn::Expr::Unsafe(value) => value
            .block
            .stmts
            .iter()
            .find_map(|statement| match statement {
                syn::Stmt::Expr(expression, _) => {
                    Some(builder_expression_facts(expression, vb_bindings))
                }
                _ => None,
            })
            .unwrap_or_else(empty_builder_facts),
        syn::Expr::Block(block) => block
            .block
            .stmts
            .iter()
            .find_map(|statement| match statement {
                syn::Stmt::Expr(expression, _) => {
                    Some(builder_expression_facts(expression, vb_bindings))
                }
                _ => None,
            })
            .unwrap_or_else(empty_builder_facts),
        syn::Expr::Paren(paren) => builder_expression_facts(&paren.expr, vb_bindings),
        syn::Expr::Group(group) => builder_expression_facts(&group.expr, vb_bindings),
        _ => empty_builder_facts(),
    }
}

fn call_leaf_name(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        _ => None,
    }
}

fn load_varmap_target(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Reference(reference) => expr_identifier(&reference.expr),
        _ => expr_identifier(expression),
    }
}

#[derive(Clone)]
struct SubprocessInvocation {
    wrapper_function: StableId,
    subprocess_key: String,
    cli_flags: Vec<String>,
    launcher: String,
    source: String,
}

fn discover_subprocess_pipeline(
    krate: &Crate,
    functions: &HashMap<String, StableId>,
    model: &mut ModelIr,
) {
    let launchers = subprocess_launcher_functions(krate);
    if launchers.is_empty() {
        return;
    }

    let function_by_id: HashMap<StableId, Function> = model
        .functions
        .iter()
        .map(|function| (function.id.clone(), function.clone()))
        .collect();
    let by_bare: HashMap<String, Vec<StableId>> =
        model
            .functions
            .iter()
            .fold(HashMap::new(), |mut grouped, function| {
                grouped
                    .entry(function.name.clone())
                    .or_default()
                    .push(function.id.clone());
                grouped
            });

    let mut invocations = Vec::new();
    for func in krate.all_functions().chain(krate.all_methods()) {
        if !is_production_source(krate, func) {
            continue;
        }
        let wrapper_id = function_id(func);
        let mut collector = SubprocessInvocationCollector {
            launchers: &launchers,
            wrapper_id,
            source: krate.file_label(func.span),
            invocations: Vec::new(),
        };
        collector.visit_block(&func.block);
        invocations.extend(collector.invocations);
    }

    let orchestrator = discover_orchestrator_order(krate, functions, model, &by_bare);
    if orchestrator.is_empty() && invocations.is_empty() {
        return;
    }

    let mut seen_functions = model
        .stages
        .iter()
        .map(|stage| stage.function.clone())
        .collect::<HashSet<_>>();
    let order_base = model.stages.len();

    for (order, (function_id, orchestrator_name)) in orchestrator.into_iter().enumerate() {
        if seen_functions.contains(&function_id) {
            continue;
        }
        let Some(function) = function_by_id.get(&function_id) else {
            continue;
        };
        let invocation = invocations
            .iter()
            .find(|item| item.wrapper_function == function_id);
        push_subprocess_stage(
            model,
            function,
            invocation,
            order_base + order,
            Some(orchestrator_name),
        );
        seen_functions.insert(function_id);
    }

    for invocation in invocations {
        if seen_functions.contains(&invocation.wrapper_function) {
            if let Some(stage) = model
                .stages
                .iter_mut()
                .find(|stage| stage.function == invocation.wrapper_function)
            {
                stage.dispatch = StageDispatchKind::Subprocess;
                if stage.subprocess_key.is_none() {
                    stage.subprocess_key = Some(invocation.subprocess_key.clone());
                }
                stage.name = stage
                    .subprocess_key
                    .clone()
                    .unwrap_or_else(|| stage.name.clone());
                stage.cli_flags.extend(invocation.cli_flags.iter().cloned());
                stage.cli_flags.sort();
                stage.cli_flags.dedup();
                stage.launcher = Some(invocation.launcher.clone());
                stage.evidence.push(Evidence {
                    kind: EvidenceKind::Source,
                    confidence: Confidence::Heuristic,
                    source: Some(invocation.source.clone()),
                    detail: format!(
                        "subprocess relaunch via `{}` with stage key `{}`",
                        invocation.launcher, invocation.subprocess_key
                    ),
                });
            }
            continue;
        }
        let Some(function) = function_by_id.get(&invocation.wrapper_function) else {
            continue;
        };
        push_subprocess_stage(
            model,
            function,
            Some(&invocation),
            order_base + seen_functions.len(),
            None,
        );
        seen_functions.insert(invocation.wrapper_function.clone());
    }
}

fn push_subprocess_stage(
    model: &mut ModelIr,
    function: &Function,
    invocation: Option<&SubprocessInvocation>,
    order: usize,
    orchestrator: Option<String>,
) {
    let name = invocation
        .map(|item| item.subprocess_key.clone())
        .unwrap_or_else(|| stage_display_name(function));
    let id = StableId::new("subprocess-stage", [&function.qualified_name, &name]);
    let mut evidence = vec![Evidence {
        kind: EvidenceKind::Source,
        confidence: Confidence::Heuristic,
        source: Some(function.source.clone()),
        detail: if let Some(orchestrator) = orchestrator.as_deref() {
            format!(
                "orchestrator `{orchestrator}` calls `{}` in source order",
                function.name
            )
        } else {
            format!(
                "wrapper `{}` relaunches the current executable",
                function.name
            )
        },
    }];
    if let Some(item) = invocation {
        evidence.push(Evidence {
            kind: EvidenceKind::Source,
            confidence: Confidence::Heuristic,
            source: Some(item.source.clone()),
            detail: format!(
                "subprocess relaunch via `{}` with stage key `{}`",
                item.launcher, item.subprocess_key
            ),
        });
    }
    model.stages.push(PipelineStage {
        id,
        name,
        kind: stage_kind(&function.name),
        function: function.id.clone(),
        order: Some(order),
        components: reachable_components(function, model),
        consumes: Vec::new(),
        produces: Vec::new(),
        depends_on: Vec::new(),
        source: function.source.clone(),
        evidence,
        dispatch: if invocation.is_some() {
            StageDispatchKind::Subprocess
        } else {
            StageDispatchKind::Inline
        },
        subprocess_key: invocation.map(|item| item.subprocess_key.clone()),
        cli_flags: invocation
            .map(|item| item.cli_flags.clone())
            .unwrap_or_default(),
        launcher: invocation.map(|item| item.launcher.clone()),
        orchestrator,
    });
}

fn discover_orchestrator_order(
    krate: &Crate,
    functions: &HashMap<String, StableId>,
    model: &ModelIr,
    by_bare: &HashMap<String, Vec<StableId>>,
) -> Vec<(StableId, String)> {
    let mut best = Vec::new();
    for pipeline in krate
        .all_functions()
        .filter(|function| function.fn_name == "run_pipeline")
    {
        let mut collector = OrderedCallCollector::default();
        collector.visit_block(&pipeline.block);
        let orchestrator_name = pipeline.qualified_name.clone();
        let mut ordered = Vec::new();
        for call in collector.calls {
            let Some(callee) = resolve_call(&call, pipeline, functions, by_bare) else {
                continue;
            };
            let Some(function) = model
                .functions
                .iter()
                .find(|function| function.id == callee)
            else {
                continue;
            };
            if is_orchestrator_stage_call(&function.name) {
                ordered.push((callee, orchestrator_name.clone()));
            }
        }
        if ordered.len() > best.len() {
            best = ordered;
        }
    }
    best
}

fn subprocess_launcher_functions(krate: &Crate) -> HashSet<String> {
    let mut launchers = HashSet::new();
    launchers.insert("run_stage_command".to_string());
    for func in krate.all_functions().chain(krate.all_methods()) {
        let text = func.block.to_token_stream().to_string();
        if text.contains("current_exe") && text.contains("Command") {
            launchers.insert(func.fn_name.clone());
        }
    }
    launchers
}

struct SubprocessInvocationCollector<'a> {
    launchers: &'a HashSet<String>,
    wrapper_id: StableId,
    source: String,
    invocations: Vec<SubprocessInvocation>,
}

impl<'ast> Visit<'ast> for SubprocessInvocationCollector<'_> {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let Some(leaf) = call_leaf_name(&node.func) {
            if self.launchers.contains(&leaf) {
                if let Some(subprocess_key) = node.args.first().and_then(string_literal_expr) {
                    let cli_flags = node
                        .args
                        .get(1)
                        .map(extract_cli_flags_from_expr)
                        .unwrap_or_default();
                    self.invocations.push(SubprocessInvocation {
                        wrapper_function: self.wrapper_id.clone(),
                        subprocess_key,
                        cli_flags,
                        launcher: leaf.clone(),
                        source: self.source.clone(),
                    });
                }
            } else if leaf == "run_training_stage_with_oom_recovery" {
                if let Some(subprocess_key) = node.args.get(2).and_then(string_literal_expr) {
                    let mut cli_flags = node
                        .args
                        .get(7)
                        .map(extract_cli_flags_from_expr)
                        .unwrap_or_default();
                    if cli_flags.is_empty() {
                        cli_flags = node
                            .args
                            .iter()
                            .flat_map(extract_cli_flags_from_expr)
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect();
                    }
                    self.invocations.push(SubprocessInvocation {
                        wrapper_function: self.wrapper_id.clone(),
                        subprocess_key,
                        cli_flags,
                        launcher: "run_stage_command".to_string(),
                        source: self.source.clone(),
                    });
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

#[derive(Default)]
struct OrderedCallCollector {
    calls: Vec<String>,
}

impl<'ast> Visit<'ast> for OrderedCallCollector {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*node.func {
            self.calls
                .push(path.path.to_token_stream().to_string().replace(' ', ""));
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.calls.push(node.method.to_string());
        visit::visit_expr_method_call(self, node);
    }
}

fn is_orchestrator_stage_call(name: &str) -> bool {
    is_pipeline_stage_call(name) || name.starts_with("preflight_")
}

fn string_literal_expr(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Str(text) => Some(text.value()),
            _ => None,
        },
        syn::Expr::Reference(reference) => string_literal_expr(&reference.expr),
        syn::Expr::Group(group) => string_literal_expr(&group.expr),
        syn::Expr::Paren(paren) => string_literal_expr(&paren.expr),
        _ => None,
    }
}

fn extract_cli_flags_from_expr(expression: &syn::Expr) -> Vec<String> {
    match expression {
        syn::Expr::Closure(closure) => extract_cli_flags_from_expr(&closure.body),
        syn::Expr::Block(block) => {
            let mut flags = block
                .block
                .stmts
                .iter()
                .flat_map(|statement| match statement {
                    syn::Stmt::Expr(expr, _) => extract_cli_flags_from_expr(expr),
                    syn::Stmt::Macro(macro_stmt) => {
                        flags_from_tokens(macro_stmt.mac.tokens.clone())
                    }
                    _ => Vec::new(),
                })
                .collect::<Vec<_>>();
            flags.sort();
            flags.dedup();
            flags
        }
        syn::Expr::Macro(expr_macro) => flags_from_tokens(expr_macro.mac.tokens.clone()),
        _ => extract_cli_flags(expression),
    }
}

fn flags_from_tokens(tokens: proc_macro2::TokenStream) -> Vec<String> {
    if let Ok(expression) = syn::parse2::<syn::Expr>(tokens.clone()) {
        let flags = extract_cli_flags(&expression);
        if !flags.is_empty() {
            return flags;
        }
    }
    let mut flags = tokens
        .to_string()
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || character == '-' || character == '_')
        })
        .map(|token| token.trim_matches('"'))
        .filter(|token| token.starts_with("--") && token.len() > 2)
        .map(str::to_string)
        .collect::<Vec<_>>();
    flags.sort();
    flags.dedup();
    flags
}

fn extract_cli_flags(expression: &syn::Expr) -> Vec<String> {
    let mut flags = string_literals(expression)
        .into_iter()
        .filter(|value| value.starts_with("--") && value.len() > 2)
        .collect::<Vec<_>>();
    flags.sort();
    flags.dedup();
    flags
}

struct ComponentFlowCollector<'a> {
    component_by_type: &'a HashMap<String, StableId>,
    owner_fields: HashMap<String, StableId>,
    locals: HashMap<String, StableId>,
    sequence: Vec<StableId>,
}

impl<'ast> Visit<'ast> for ComponentFlowCollector<'_> {
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let (Some(name), Some(init)) = (pat_ident(&node.pat), node.init.as_ref()) {
            if let Some(component) = constructor_component(&init.expr, self.component_by_type)
                .or_else(|| receiver_component(&init.expr, &self.locals, &self.owner_fields))
            {
                self.locals.insert(name, component);
            }
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if is_model_entry_name(&node.method.to_string()) {
            if let Some(component) =
                receiver_component(&node.receiver, &self.locals, &self.owner_fields)
            {
                if self.sequence.last() != Some(&component) {
                    self.sequence.push(component);
                }
            }
        }
        visit::visit_expr_method_call(self, node);
    }
}

fn pat_ident(pattern: &syn::Pat) -> Option<String> {
    match pattern {
        syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
        syn::Pat::Type(typed) => pat_ident(&typed.pat),
        _ => None,
    }
}

fn constructor_component(
    expression: &syn::Expr,
    components: &HashMap<String, StableId>,
) -> Option<StableId> {
    match expression {
        syn::Expr::Call(call) => {
            let syn::Expr::Path(path) = &*call.func else {
                return None;
            };
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            let owner = segments
                .get(segments.len().checked_sub(2)?)
                .map(String::as_str)?;
            components.get(owner).cloned()
        }
        syn::Expr::Try(value) => constructor_component(&value.expr, components),
        syn::Expr::Await(value) => constructor_component(&value.base, components),
        syn::Expr::Paren(value) => constructor_component(&value.expr, components),
        syn::Expr::Group(value) => constructor_component(&value.expr, components),
        _ => None,
    }
}

fn receiver_component(
    expression: &syn::Expr,
    locals: &HashMap<String, StableId>,
    owner_fields: &HashMap<String, StableId>,
) -> Option<StableId> {
    match expression {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => locals
            .get(&path.path.segments[0].ident.to_string())
            .cloned(),
        syn::Expr::Field(field)
            if matches!(
                &*field.base,
                syn::Expr::Path(path)
                    if path.path.segments.len() == 1 && path.path.segments[0].ident == "self"
            ) =>
        {
            let syn::Member::Named(name) = &field.member else {
                return None;
            };
            owner_fields.get(&name.to_string()).cloned()
        }
        syn::Expr::Reference(reference) => {
            receiver_component(&reference.expr, locals, owner_fields)
        }
        syn::Expr::Try(value) => receiver_component(&value.expr, locals, owner_fields),
        syn::Expr::Await(value) => receiver_component(&value.base, locals, owner_fields),
        syn::Expr::MethodCall(call) => receiver_component(&call.receiver, locals, owner_fields),
        syn::Expr::Paren(paren) => receiver_component(&paren.expr, locals, owner_fields),
        syn::Expr::Group(group) => receiver_component(&group.expr, locals, owner_fields),
        _ => None,
    }
}

fn type_base_from_text(text: &str) -> Option<String> {
    syn::parse_str::<syn::Type>(text)
        .ok()
        .and_then(|type_name| load::type_base_name(&type_name))
}

fn discover_components(
    krate: &Crate,
    cargo: Option<&CargoContext>,
    selected_root: Option<&str>,
    heuristic_architecture: bool,
    model: &mut ModelIr,
) {
    let exported: HashSet<String> = krate
        .public_reexports
        .iter()
        .map(|item| item.name.clone())
        .collect();
    let mut candidates = Vec::new();
    for def in krate.all_structs() {
        if let Some(selected) = selected_root {
            if def.name != selected && def.qualified_name != selected {
                continue;
            }
        }
        if cargo.and_then(|context| {
            crate::cargo_context::cfg_predicates_active(&def.cfg_predicates, &context.cfgs)
        }) == Some(false)
        {
            continue;
        }
        let methods = krate.method_candidates(&def.qualified_name, "new");
        let mut constructors = if methods.is_empty() {
            krate.method_candidates(&def.qualified_name, "load")
        } else {
            methods
        };
        if constructors.is_empty() {
            constructors = krate
                .all_methods()
                .filter(|method| {
                    method.qualified_type_name == def.qualified_name
                        && method.trait_name.is_none()
                        && !method.vb_params.is_empty()
                        && return_mentions_type(&method.return_type, &def.name)
                })
                .collect();
            constructors.sort_by(|a, b| a.qualified_name.cmp(&b.qualified_name));
        }
        let Some(ctor) = constructors.into_iter().find(|method| {
            !method.vb_params.is_empty()
                && cargo.and_then(|context| {
                    crate::cargo_context::cfg_predicates_active(
                        &method.cfg_predicates,
                        &context.cfgs,
                    )
                }) != Some(false)
        }) else {
            continue;
        };
        let is_api_boundary =
            exported.contains(&def.name) || def.module_path.is_empty() && def.visibility == "pub";
        // Any public type with a VarBuilder constructor is a model component candidate
        // (e.g. `my_model::Model`), not only crate-root re-exports.
        let is_varbuilder_component = def.visibility == "pub" && !ctor.vb_params.is_empty();
        if selected_root.is_some() || is_api_boundary || is_varbuilder_component {
            candidates.push((def, ctor));
        }
    }
    candidates.sort_by(|(a, _), (b, _)| a.qualified_name.cmp(&b.qualified_name));
    candidates.dedup_by(|(a, _), (b, _)| a.qualified_name == b.qualified_name);

    for (def, ctor) in candidates {
        if selected_root.is_some() {
            for function in &mut model.functions {
                if function.owner_type.as_deref() == Some(def.qualified_name.as_str())
                    && (function
                        .parameters
                        .iter()
                        .any(|parameter| is_tensor_type(&parameter.type_name))
                        || function.return_type.as_deref().is_some_and(is_tensor_type))
                {
                    function.is_entrypoint = true;
                }
            }
        }
        let candle_version = cargo.and_then(|context| {
            op_semantics::matched_candle_version(
                context
                    .candle_versions
                    .get("candle-core")
                    .map(String::as_str),
                context.candle_versions.get("candle-nn").map(String::as_str),
            )
        });
        add_component(
            krate,
            model,
            def,
            ctor,
            candle_version,
            heuristic_architecture,
        );
    }
}

fn return_mentions_type(return_type: &str, type_name: &str) -> bool {
    return_type
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|part| part == "Self" || part == type_name)
}

fn add_component(
    krate: &Crate,
    model: &mut ModelIr,
    def: &StructDef,
    ctor: &ImplFn,
    candle_version: Option<&str>,
    heuristic_architecture: bool,
) {
    let component_id = StableId::new("component", [&def.qualified_name]);
    let constructor_id = function_id(ctor);
    let mut component = Component {
        id: component_id.clone(),
        name: def.name.clone(),
        qualified_name: def.qualified_name.clone(),
        source: krate.file_label(def.span),
        constructor: constructor_id,
        builders: ctor
            .vb_params
            .iter()
            .filter_map(|index| ctor.params.get(*index))
            .map(|name| {
                let (role, confidence) = if heuristic_architecture {
                    infer_builder_role(name)
                } else {
                    (BuilderRole::Unknown, Confidence::Unknown)
                };
                let mut evidence = vec![source_evidence(
                    krate.file_label(ctor.span),
                    format!("constructor parameter `{name}` has VarBuilder type"),
                )];
                if role != BuilderRole::Unknown {
                    evidence.push(Evidence {
                        kind: EvidenceKind::Inferred,
                        confidence,
                        source: Some(krate.file_label(ctor.span)),
                        detail: format!("builder name `{name}` suggests role `{role:?}`"),
                    });
                }
                BuilderNamespace {
                    name: name.clone(),
                    role,
                    evidence,
                }
            })
            .collect(),
        modules: Vec::new(),
        parameters: Vec::new(),
        entrypoints: model
            .functions
            .iter()
            .filter(|function| {
                function
                    .owner_type
                    .as_deref()
                    .is_some_and(|owner| owner == def.qualified_name)
                    && function.is_entrypoint
            })
            .map(|function| function.id.clone())
            .collect(),
        evidence: vec![source_evidence(
            krate.file_label(def.span),
            "public model API type with a VarBuilder constructor",
        )],
    };

    match Extractor::for_candle_version(krate, candle_version)
        .run(&def.qualified_name, Some(&ctor.fn_name))
    {
        Ok(structure) => {
            let mut module_ids = HashMap::new();
            for instance in &structure.instances {
                let module_def = structure.def(instance.def);
                let id = StableId::new(
                    "module",
                    [
                        component_id.0.as_str(),
                        instance.root.as_str(),
                        instance.prefix.to_string().as_str(),
                        module_def.name.as_str(),
                        instance.id.0.to_string().as_str(),
                    ],
                );
                module_ids.insert(instance.id, id.clone());
                component.modules.push(id.clone());
                model.modules.push(Module {
                    id,
                    component: component_id.clone(),
                    parent: instance
                        .parent
                        .and_then(|parent| module_ids.get(&parent).cloned()),
                    type_name: module_def.name.clone(),
                    qualified_type: unique_qualified_struct(krate, &module_def.name),
                    field: instance.via_field.clone(),
                    builder_root: instance.root.clone(),
                    prefix: instance.prefix.to_string(),
                    repeat: instance
                        .repeat
                        .as_ref()
                        .map(|repeat| format!("{} in 0..{}", repeat.var, repeat.bound)),
                    source: krate.file_label(instance.origin),
                    confidence: certainty_confidence(&instance.certainty),
                });
            }
            for param in &structure.params {
                let site = structure.site(param.site);
                let Some(module) = module_ids.get(&param.owner).cloned() else {
                    continue;
                };
                let key = param.key.to_string();
                let id = StableId::new(
                    "parameter",
                    [component_id.0.as_str(), param.root.as_str(), key.as_str()],
                );
                component.parameters.push(id.clone());
                let (checkpoint_shape, checkpoint_dtype) = match &param.checkpoint {
                    CheckpointMatch::Found { shape, dtype, .. } => {
                        (Some(shape.clone()), Some(dtype.clone()))
                    }
                    _ => (None, None),
                };
                model.parameters.push(Parameter {
                    id,
                    component: component_id.clone(),
                    module,
                    key,
                    builder_root: param.root.clone(),
                    role: match site.kind {
                        crate::known::ParamKind::RunningMean
                        | crate::known::ParamKind::RunningVar => ParameterRole::RunningState,
                        _ => ParameterRole::Unknown,
                    },
                    kind: acquisition_label(&site.acquisition),
                    symbolic_shape: site.shape.clone(),
                    checkpoint_shape,
                    checkpoint_dtype,
                    source: krate.file_label(site.span),
                    uses: Vec::new(),
                    optimizer_memberships: Vec::new(),
                    evidence: vec![Evidence {
                        kind: EvidenceKind::Source,
                        confidence: certainty_confidence(&param.certainty),
                        source: Some(krate.file_label(site.span)),
                        detail: format!(
                            "parameter registered through {}",
                            acquisition_label(&site.acquisition)
                        ),
                    }],
                });
            }
            for diagnostic in structure.diagnostics {
                push_finding(
                    model,
                    "structure-unresolved",
                    FindingSeverity::Warning,
                    Confidence::Proven,
                    diagnostic.message,
                    Some(krate.file_label(diagnostic.span)),
                    vec![component_id.clone()],
                );
            }
        }
        Err(error) => push_finding(
            model,
            "component-expansion",
            FindingSeverity::Warning,
            Confidence::Proven,
            format!("could not expand {}: {error:#}", def.qualified_name),
            Some(krate.file_label(ctor.span)),
            vec![component_id.clone()],
        ),
    }
    model.components.push(component);
}

fn add_contracts(krate: &Crate, model: &mut ModelIr) {
    let analysis = crate::contracts::analyze(krate);
    let owners: HashMap<String, StableId> = model
        .functions
        .iter()
        .map(|function| (function.qualified_name.clone(), function.id.clone()))
        .collect();
    let function_indices: HashMap<StableId, usize> = model
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.id.clone(), index))
        .collect();

    for contracts in analysis.functions {
        let Some(owner) = owners.get(&contracts.qualified_name).cloned() else {
            continue;
        };
        for mut tensor in contracts.tensors {
            tensor.owner_function = owner.clone();
            if tensor.dtype.eq_ignore_ascii_case("unknown")
                || tensor.dtype.starts_with("same_as(")
                || tensor.dtype == "work_dtype"
            {
                tensor.dtype = "Unknown".to_string();
            }
            if let Some(existing) = model.tensors.iter_mut().find(|existing| {
                existing.owner_function == owner && existing.name == tensor.name
            }) {
                if existing.dtype == "Unknown" && tensor.dtype != "Unknown" {
                    existing.dtype = tensor.dtype.clone();
                    existing.evidence.extend(tensor.evidence.clone());
                }
                continue;
            }
            tensor.id = StableId::new("tensor", [owner.0.as_str(), tensor.name.as_str()]);
            let id = tensor.id.clone();
            let role = tensor.role.clone();
            if let Some(&index) = function_indices.get(&owner) {
                if matches!(role, TensorRole::Input) {
                    model.functions[index].tensor_inputs.push(id.clone());
                }
                if matches!(role, TensorRole::Output | TensorRole::Loss) {
                    model.functions[index].tensor_outputs.push(id.clone());
                }
            }
            model.tensors.push(tensor);
        }
    }
}

fn discover_pipeline_and_artifacts(
    krate: &Crate,
    functions: &HashMap<String, StableId>,
    model: &mut ModelIr,
) {
    let function_by_id: HashMap<StableId, &Function> = model
        .functions
        .iter()
        .map(|function| (function.id.clone(), function))
        .collect();
    let by_bare: HashMap<String, Vec<StableId>> =
        model
            .functions
            .iter()
            .fold(HashMap::new(), |mut grouped, function| {
                grouped
                    .entry(function.name.clone())
                    .or_default()
                    .push(function.id.clone());
                grouped
            });
    let mut ordered_stage_functions: Vec<(StableId, Option<String>)> = Vec::new();
    for pipeline in krate
        .all_functions()
        .filter(|function| function.fn_name == "run_pipeline")
    {
        let mut collector = CallCollector::default();
        collector.visit_block(&pipeline.block);
        for call in collector.calls {
            let Some(callee) = resolve_call(&call, pipeline, functions, &by_bare) else {
                continue;
            };
            let Some(target) = function_by_id.get(&callee) else {
                continue;
            };
            if !is_pipeline_stage_call(&target.name) {
                continue;
            }
            let variants = stage_variants(krate, &callee);
            if variants.is_empty() {
                ordered_stage_functions.push((callee, None));
            } else {
                ordered_stage_functions.extend(
                    variants
                        .into_iter()
                        .map(|variant| (callee.clone(), Some(variant))),
                );
            }
        }
    }
    if ordered_stage_functions.is_empty() {
        ordered_stage_functions.extend(
            model
                .functions
                .iter()
                .filter(|function| is_stage_entry_name(&function.name))
                .map(|function| (function.id.clone(), None)),
        );
    }
    ordered_stage_functions.dedup();

    let mut stage_by_function = HashMap::new();
    for (order, (function_id, variant)) in ordered_stage_functions.into_iter().enumerate() {
        let Some(function) = function_by_id.get(&function_id) else {
            continue;
        };
        let name = variant.unwrap_or_else(|| stage_display_name(function));
        let id = StableId::new("stage", [&function.qualified_name, &name]);
        stage_by_function
            .entry(function_id.clone())
            .or_insert_with(|| id.clone());
        let related_components = reachable_components(function, model);
        model.stages.push(PipelineStage {
            id,
            name,
            kind: stage_kind(&function.name),
            function: function_id,
            order: Some(order),
            components: related_components,
            consumes: Vec::new(),
            produces: Vec::new(),
            depends_on: Vec::new(),
            source: function.source.clone(),
            evidence: vec![heuristic_source_evidence(
                function.source.clone(),
                "stage entrypoint discovered from pipeline call order",
            )],
            dispatch: StageDispatchKind::Unknown,
            subprocess_key: None,
            cli_flags: Vec::new(),
            launcher: None,
            orchestrator: None,
        });
    }

    let mut facts = Vec::new();
    for func in krate.all_functions().chain(krate.all_methods()) {
        if !is_production_source(krate, func) {
            continue;
        }
        let mut visitor = ArtifactVisitor::default();
        visitor.visit_block(&func.block);
        let owner = function_id(func);
        for observed in visitor.observed {
            facts.push((owner.clone(), krate.file_label(func.span), observed));
        }
    }
    facts.sort_by(|a, b| a.2.path_expr.cmp(&b.2.path_expr));

    let mut artifact_index: HashMap<String, usize> = HashMap::new();
    for (function_id, source, observed) in facts {
        let identity = artifact_identity(&observed.path_expr);
        let index = match artifact_index.get(&identity).copied() {
            Some(index) => index,
            None => {
                let id = StableId::new("artifact", [&identity]);
                let index = model.artifacts.len();
                model.artifacts.push(Artifact {
                    id,
                    name: observed.label.clone(),
                    kind: artifact_kind(&observed.label),
                    path_expr: observed.path_expr.clone(),
                    produced_by: Vec::new(),
                    consumed_by: Vec::new(),
                    source: source.clone(),
                    evidence: vec![heuristic_source_evidence(
                        source.clone(),
                        format!("path passed to `{}`", observed.operation),
                    )],
                });
                artifact_index.insert(identity, index);
                index
            }
        };
        let stage = stage_for_function(&function_id, &stage_by_function, model);
        if let Some(stage_id) = stage {
            let artifact = &mut model.artifacts[index];
            if observed.produced {
                artifact.produced_by.push(stage_id.clone());
            } else {
                artifact.consumed_by.push(stage_id.clone());
            }
        }
    }

    for artifact in &mut model.artifacts {
        artifact.produced_by.sort();
        artifact.produced_by.dedup();
        artifact.consumed_by.sort();
        artifact.consumed_by.dedup();
    }
    infer_artifact_stage_links(model);
    let stage_orders: HashMap<StableId, usize> = model
        .stages
        .iter()
        .map(|stage| (stage.id.clone(), stage.order.unwrap_or(usize::MAX)))
        .collect();
    for stage in &mut model.stages {
        let stage_order = stage.order.unwrap_or(usize::MAX);
        for artifact in &model.artifacts {
            if artifact.produced_by.contains(&stage.id) {
                stage.produces.push(artifact.id.clone());
            }
            if artifact.consumed_by.contains(&stage.id) {
                stage.consumes.push(artifact.id.clone());
                stage.depends_on.extend(
                    artifact
                        .produced_by
                        .iter()
                        .filter(|producer| {
                            **producer != stage.id
                                && stage_orders
                                    .get(*producer)
                                    .is_some_and(|producer_order| *producer_order < stage_order)
                        })
                        .cloned(),
                );
            }
        }
        stage.consumes.sort();
        stage.consumes.dedup();
        stage.produces.sort();
        stage.produces.dedup();
        stage.depends_on.sort();
        stage.depends_on.dedup();
    }

    let _ = functions;
}

fn is_production_source(krate: &Crate, function: &ImplFn) -> bool {
    !function
        .module_path
        .split("::")
        .any(|segment| segment == "tests" || segment == "test")
        && krate.files.get(function.span.file).is_some_and(|file| {
            let rel = file.rel.replace('\\', "/");
            rel.starts_with("src/") || !rel.contains('/')
        })
}

#[derive(Clone)]
struct ConstructorSpec {
    component: StableId,
    owner: String,
    builders: Vec<(usize, String)>,
}

fn component_varmap_bindings(
    _krate: &Crate,
    function: &ImplFn,
    model: &ModelIr,
) -> HashMap<String, Vec<(StableId, String)>> {
    let constructor_functions: HashMap<StableId, &Function> = model
        .functions
        .iter()
        .map(|candidate| (candidate.id.clone(), candidate))
        .collect();
    let specs: Vec<ConstructorSpec> = model
        .components
        .iter()
        .filter_map(|component| {
            let constructor = constructor_functions.get(&component.constructor)?;
            let builders = constructor
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, parameter)| parameter.type_name.contains("VarBuilder"))
                .zip(component.builders.iter())
                .map(|((index, _), builder)| (index, builder.name.clone()))
                .collect();
            Some(ConstructorSpec {
                component: component.id.clone(),
                owner: component.name.clone(),
                builders,
            })
        })
        .collect();

    let mut builder_collector = BuilderSourceCollector::default();
    builder_collector.visit_block(&function.block);
    let mut collector = ConstructorBindingCollector {
        specs: &specs,
        builder_sources: &builder_collector.sources,
        bindings: HashMap::new(),
    };
    collector.visit_block(&function.block);
    for bindings in collector.bindings.values_mut() {
        bindings.sort();
        bindings.dedup();
    }
    collector.bindings
}

#[derive(Default)]
struct BuilderSourceCollector {
    sources: HashMap<String, BTreeSet<String>>,
}

impl<'ast> Visit<'ast> for BuilderSourceCollector {
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let (Some(name), Some(init)) = (pat_ident(&node.pat), node.init.as_ref()) {
            let varmaps = varmap_sources(&init.expr, &self.sources);
            if !varmaps.is_empty() {
                self.sources.entry(name).or_default().extend(varmaps);
            }
        }
        visit::visit_local(self, node);
    }
}

struct ConstructorBindingCollector<'a> {
    specs: &'a [ConstructorSpec],
    builder_sources: &'a HashMap<String, BTreeSet<String>>,
    bindings: HashMap<String, Vec<(StableId, String)>>,
}

impl<'ast> Visit<'ast> for ConstructorBindingCollector<'_> {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*node.func {
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            if let Some(owner) = segments.get(segments.len().saturating_sub(2)) {
                for spec in self.specs.iter().filter(|spec| &spec.owner == owner) {
                    for (index, root) in &spec.builders {
                        let Some(argument) = node.args.iter().nth(*index) else {
                            continue;
                        };
                        for varmap in varmap_sources(argument, self.builder_sources) {
                            self.bindings
                                .entry(varmap)
                                .or_default()
                                .push((spec.component.clone(), root.clone()));
                        }
                    }
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

fn varmap_sources(
    expression: &syn::Expr,
    builder_sources: &HashMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    match expression {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => {
            let name = path.path.segments[0].ident.to_string();
            builder_sources.get(&name).cloned().unwrap_or_default()
        }
        syn::Expr::Call(call) => {
            let is_from_varmap = matches!(
                &*call.func,
                syn::Expr::Path(path)
                    if path.path.segments.last().is_some_and(|segment| segment.ident == "from_varmap")
            );
            if is_from_varmap {
                call.args
                    .first()
                    .and_then(expr_identifier)
                    .into_iter()
                    .collect()
            } else {
                BTreeSet::new()
            }
        }
        syn::Expr::MethodCall(call) => varmap_sources(&call.receiver, builder_sources),
        syn::Expr::Reference(reference) => varmap_sources(&reference.expr, builder_sources),
        syn::Expr::Try(value) => varmap_sources(&value.expr, builder_sources),
        syn::Expr::Paren(paren) => varmap_sources(&paren.expr, builder_sources),
        syn::Expr::Group(group) => varmap_sources(&group.expr, builder_sources),
        _ => BTreeSet::new(),
    }
}

fn discover_optimizers(krate: &Crate, model: &mut ModelIr) {
    let stage_functions: Vec<(StableId, StableId, String)> = model
        .stages
        .iter()
        .map(|stage| {
            (
                stage.id.clone(),
                stage.function.clone(),
                stage.source.clone(),
            )
        })
        .collect();
    for func in krate.all_functions().chain(krate.all_methods()) {
        if !is_production_source(krate, func) {
            continue;
        }
        let text = func.block.to_token_stream().to_string();
        if !text.contains("all_vars") && !text.contains("named_train_vars") {
            continue;
        }
        let mut visitor = OptimizerVisitor::default();
        visitor.visit_block(&func.block);
        visitor.excludes.sort();
        visitor.excludes.dedup();
        visitor.includes.sort();
        visitor.includes.dedup();
        if visitor.optimizer.is_none() {
            continue;
        }
        let component_bindings = component_varmap_bindings(krate, func, model);
        let function_id = function_id(func);
        let exact_stage = stage_functions
            .iter()
            .find(|(_, stage_function, _)| stage_function == &function_id)
            .map(|(stage, _, _)| stage.clone())
            .or_else(|| {
                stage_functions
                    .iter()
                    .find(|(_, _, source)| {
                        source.split(':').next()
                            == Some(
                                krate
                                    .file_label(func.span)
                                    .split(':')
                                    .next()
                                    .unwrap_or_default(),
                            )
                    })
                    .map(|(stage, _, _)| stage.clone())
            });
        let mut stages: Vec<StableId> = exact_stage.into_iter().collect();
        if stages.is_empty() {
            stages.push(StableId::new("stage", [&func.qualified_name]));
        }

        for varmap in visitor.varmaps {
            let bindings = component_bindings.get(&varmap).cloned().unwrap_or_default();
            let components: Vec<StableId> = bindings
                .iter()
                .map(|(component, _)| component.clone())
                .collect();
            let components = expand_nested_components(model, components);
            let roots: Vec<String> = bindings.iter().map(|(_, root)| root.clone()).collect();
            for stage in &stages {
                let id = StableId::new(
                    "optimizer-membership",
                    [
                        stage.0.as_str(),
                        varmap.as_str(),
                        func.qualified_name.as_str(),
                    ],
                );
                model.optimizers.push(OptimizerMembership {
                    id,
                    stage: stage.clone(),
                    optimizer: visitor
                        .optimizer
                        .clone()
                        .unwrap_or_else(|| "optimizer".to_string()),
                    varmap: varmap.clone(),
                    components: components.clone(),
                    builder_roots: roots.clone(),
                    include_patterns: visitor.includes.clone(),
                    exclude_patterns: visitor.excludes.clone(),
                    conditional: (stages.len() > 1)
                        .then(|| "pipeline stage/configuration dependent".to_string()),
                    source: krate.file_label(func.span),
                    evidence: vec![heuristic_source_evidence(
                        krate.file_label(func.span),
                        "optimizer consumes variables returned by named_train_vars/all_vars",
                    )],
                });
            }
        }
    }

    let components_by_stage = model.optimizers.iter().fold(
        HashMap::<StableId, Vec<StableId>>::new(),
        |mut map, optimizer| {
            map.entry(optimizer.stage.clone())
                .or_default()
                .extend(optimizer.components.iter().cloned());
            map
        },
    );
    for stage in &mut model.stages {
        if let Some(components) = components_by_stage.get(&stage.id) {
            stage.components.extend(components.iter().cloned());
            stage.components.sort();
            stage.components.dedup();
        }
    }
}

fn expand_nested_components(model: &ModelIr, mut components: Vec<StableId>) -> Vec<StableId> {
    let qualified_components: HashMap<&str, &StableId> = model
        .components
        .iter()
        .map(|component| (component.qualified_name.as_str(), &component.id))
        .collect();
    loop {
        let mut added = Vec::new();
        for module in model
            .modules
            .iter()
            .filter(|module| components.contains(&module.component))
        {
            let Some(qualified_type) = module.qualified_type.as_deref() else {
                continue;
            };
            if let Some(component) = qualified_components.get(qualified_type) {
                if !components.contains(component) {
                    added.push((*component).clone());
                }
            }
        }
        if added.is_empty() {
            break;
        }
        components.extend(added);
        components.sort();
        components.dedup();
    }
    components
}

fn apply_optimizer_roles(model: &mut ModelIr) {
    let component_names: HashMap<StableId, String> = model
        .components
        .iter()
        .map(|component| (component.id.clone(), component.name.to_ascii_lowercase()))
        .collect();
    for parameter in &mut model.parameters {
        let mut matched = Vec::new();
        let mut excluded = false;
        let mut conditionally_excluded_component = false;
        for optimizer in &model.optimizers {
            let component_matches = optimizer.components.is_empty()
                || optimizer.components.contains(&parameter.component);
            let root_matches = optimizer.builder_roots.contains(&parameter.builder_root)
                || optimizer.builder_roots.is_empty()
                    && similar_stem(&optimizer.varmap, &parameter.builder_root);
            if !component_matches || !root_matches {
                continue;
            }
            if optimizer
                .exclude_patterns
                .iter()
                .any(|pattern| parameter.key.contains(pattern))
            {
                excluded = true;
                continue;
            }
            if optimizer.exclude_patterns.iter().any(|pattern| {
                let prefix = pattern
                    .trim_matches(|character: char| !character.is_alphanumeric())
                    .to_ascii_lowercase();
                !prefix.is_empty()
                    && component_names
                        .get(&parameter.component)
                        .is_some_and(|component| component.contains(&prefix))
            }) {
                conditionally_excluded_component = true;
            }
            if optimizer.include_patterns.is_empty()
                || optimizer
                    .include_patterns
                    .iter()
                    .any(|pattern| parameter.key.contains(pattern))
            {
                matched.push(optimizer.id.clone());
            }
        }
        parameter.optimizer_memberships = matched;
        parameter.role = if is_running_state(&parameter.key) {
            ParameterRole::RunningState
        } else if !parameter.optimizer_memberships.is_empty() && conditionally_excluded_component {
            ParameterRole::Conditional
        } else if !parameter.optimizer_memberships.is_empty() {
            ParameterRole::Optimized
        } else if excluded {
            ParameterRole::Excluded
        } else if model.optimizers.iter().any(|optimizer| {
            optimizer.components.contains(&parameter.component)
                && !optimizer.builder_roots.contains(&parameter.builder_root)
        }) {
            ParameterRole::Frozen
        } else {
            ParameterRole::Unknown
        };
        parameter.evidence.push(Evidence {
            kind: EvidenceKind::Source,
            confidence: if parameter.optimizer_memberships.is_empty() {
                Confidence::Conditional
            } else {
                Confidence::Proven
            },
            source: Some(parameter.source.clone()),
            detail: format!(
                "role {:?} derived from optimizer membership, not constructor naming",
                parameter.role
            ),
        });
    }

    for component in &mut model.components {
        for builder in &mut component.builders {
            let roles: Vec<_> = model
                .parameters
                .iter()
                .filter(|parameter| {
                    parameter.component == component.id && parameter.builder_root == builder.name
                })
                .map(|parameter| &parameter.role)
                .collect();
            builder.role = if roles
                .iter()
                .any(|role| matches!(role, ParameterRole::Optimized))
            {
                BuilderRole::Trainable
            } else if roles.iter().any(|role| {
                matches!(
                    role,
                    ParameterRole::Frozen | ParameterRole::Excluded | ParameterRole::RunningState
                )
            }) {
                BuilderRole::Frozen
            } else {
                BuilderRole::Unknown
            };
        }
    }
}

fn entrypoint_analysis_phases(
    name: &str,
    qualified_name: &str,
    is_loss: bool,
) -> Vec<crate::phase::ExecutionPhase> {
    #[cfg(feature = "runtime")]
    {
        crate::phase::entrypoint_phases(name, qualified_name, is_loss)
    }
    #[cfg(not(feature = "runtime"))]
    {
        let _ = (name, qualified_name, is_loss);
        vec![crate::phase::ExecutionPhase::Train]
    }
}

fn add_dataflow(krate: &Crate, model: &mut ModelIr) {
    let candle_nn_version = model.cargo.as_ref().and_then(|cargo| {
        op_semantics::matched_candle_version(
            cargo.candle_packages.get("candle-core").map(String::as_str),
            cargo.candle_packages.get("candle-nn").map(String::as_str),
        )
        .map(str::to_string)
    });
    let selected: Vec<(StableId, String, String, bool)> = model
        .functions
        .iter()
        .filter(|function| function.is_entrypoint && function.cfg_active != Some(false))
        .map(|function| {
            (
                function.id.clone(),
                function.name.clone(),
                function.qualified_name.clone(),
                function.is_loss,
            )
        })
        .collect();

    let function_indices: HashMap<StableId, usize> = model
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.id.clone(), index))
        .collect();
    for (function_id, name, entry, is_loss) in selected {
        let phases = entrypoint_analysis_phases(&name, &entry, is_loss);
        if let Some(&function_index) = function_indices.get(&function_id) {
            model.functions[function_index].execution_phases = phases.clone();
        }
        for phase in phases {
            let graph = match dataflow::analyze_with_phase(
                krate,
                &entry,
                candle_nn_version.as_deref(),
                phase,
            ) {
                Ok(graph) => graph,
                Err(error) => {
                    push_finding(
                        model,
                        "dataflow-entrypoint",
                        FindingSeverity::Information,
                        Confidence::Proven,
                        format!("could not analyze `{entry}`: {error:#}"),
                        None,
                        vec![function_id.clone()],
                    );
                    continue;
                }
            };
            let mut tensor_ids = vec![None; graph.nodes.len()];
            for node_id in &graph.tensor_nodes {
                let node = graph.node(*node_id);
                let id = StableId::new(
                    "tensor",
                    [
                        phase.as_str(),
                        function_id.0.as_str(),
                        node.id.0.to_string().as_str(),
                    ],
                );
                let name = node_name(&node.kind);
                let role = match &node.kind {
                    NodeKind::Param { .. } => TensorRole::Input,
                    NodeKind::Return => TensorRole::Output,
                    _ if graph.loss_nodes.contains(&node.id) || is_loss => TensorRole::Loss,
                    _ => TensorRole::Activation,
                };
                model.tensors.push(TensorContract {
                    id: id.clone(),
                    name,
                    role,
                    owner_function: function_id.clone(),
                    parameter: None,
                    shape: ShapeFact {
                        rank: shape_rank(node.shape.as_deref()),
                        dimensions: Vec::new(),
                        source_expr: node.shape.clone(),
                    },
                    dtype: if node.dtype.is_known() {
                        node.dtype.to_string()
                    } else {
                        "Unknown".to_string()
                    },
                    device: DeviceFact::Unknown,
                    layout: layout_for_node(&node.kind),
                    requires_grad: match node.grad {
                        GradState::Trainable | GradState::Differentiable => Some(true),
                        GradState::Frozen | GradState::Severed => Some(false),
                        _ => None,
                    },
                    execution_phase: Some(phase),
                    evidence: {
                        let mut evidence = vec![source_evidence(
                            krate.file_label(node.span),
                            format!(
                                "expression-level static dataflow; source type {}",
                                node.type_name.as_deref().unwrap_or("Tensor")
                            ),
                        )];
                        if node.dtype.is_known() {
                            evidence.push(source_evidence(
                                krate.file_label(node.span),
                                format!("static dtype {}", node.dtype),
                            ));
                        }
                        evidence
                    },
                });
                tensor_ids[node.id.0] = Some(id);
            }
            if let Some(&function_index) = function_indices.get(&function_id) {
                if phase == crate::phase::ExecutionPhase::Train {
                    model.functions[function_index].tensor_inputs = graph
                        .nodes
                        .iter()
                        .filter(|node| matches!(node.kind, NodeKind::Param { .. }))
                        .filter_map(|node| tensor_ids[node.id.0].clone())
                        .collect();
                    model.functions[function_index].tensor_outputs = graph
                        .entry_return
                        .and_then(|node| tensor_ids[node.0].clone())
                        .map(|tensor| vec![tensor])
                        .unwrap_or_default();
                }
            }
            for node in &graph.nodes {
                let NodeKind::Call { callee } = &node.kind else {
                    continue;
                };
                let short = callee.rsplit("::").next().unwrap_or(callee.as_str());
                let mut effect =
                    op_semantics::lookup_resolved(callee, candle_nn_version.as_deref());
                if matches!(effect.dtype, op_semantics::DtypeRule::Unknown)
                    && matches!(effect.grad, op_semantics::GradFlow::Unknown)
                {
                    effect = op_semantics::lookup_for(short, candle_nn_version.as_deref());
                }
                if matches!(effect.dtype, op_semantics::DtypeRule::Unknown)
                    && matches!(effect.grad, op_semantics::GradFlow::Unknown)
                {
                    if node.dtype.is_known() {
                        effect = op_semantics::OpEffect::inferred_preserve(short);
                    } else {
                        continue;
                    }
                }
                let Some(output) = tensor_ids[node.id.0].clone() else {
                    continue;
                };
                let inputs = graph
                    .edges
                    .iter()
                    .filter(|edge| edge.to == node.id)
                    .filter_map(|edge| tensor_ids[edge.from.0].clone())
                    .collect();
                let operation_id = StableId::new(
                    "operation",
                    [
                        phase.as_str(),
                        function_id.0.as_str(),
                        node.id.0.to_string().as_str(),
                    ],
                );
                model.operations.push(Operation {
                    id: operation_id.clone(),
                    function: function_id.clone(),
                    name: effect.name.clone(),
                    qualified_name: callee.contains("::").then(|| callee.clone()),
                    inputs,
                    output,
                    source: krate.file_label(node.span),
                    dtype_rule: format!("{:?}", effect.dtype),
                    gradient_rule: format!("{:?}", effect.grad),
                    device_rule: if effect.name == "to_device" {
                        "explicit".to_string()
                    } else {
                        "preserve".to_string()
                    },
                    shape_rule: shape_rule(&effect.name).to_string(),
                    domain_rule: effect.domain_rule_label(),
                    execution_phase: Some(phase),
                    timing: None,
                    evidence: vec![Evidence {
                        kind: EvidenceKind::Source,
                        confidence: Confidence::Proven,
                        source: Some(krate.file_label(node.span)),
                        detail: effect.note.unwrap_or("Candle operation rule").to_string(),
                    }],
                });
                link_implicit_parameter_reads(model, &graph, node.id, &function_id, &operation_id);
            }
            let dead_params = graph.dead_params();
            for conflict in &graph.dtype_conflicts {
                push_finding(
                    model,
                    "dtype-conflict",
                    FindingSeverity::Error,
                    Confidence::Proven,
                    conflict.message.clone(),
                    Some(krate.file_label(conflict.span)),
                    vec![function_id.clone()],
                );
            }
            for risk in &graph.dtype_risks {
                push_finding(
                    model,
                    "dtype-risk",
                    FindingSeverity::Warning,
                    Confidence::Conditional,
                    risk.message.clone(),
                    Some(krate.file_label(risk.span)),
                    vec![function_id.clone()],
                );
            }
            let inference_entrypoint = phase == crate::phase::ExecutionPhase::Infer;
            for violation in &graph.numeric_domain_violations {
                let (severity, confidence) = numeric_finding_severity(
                    violation.proven,
                    violation.impact,
                    inference_entrypoint,
                );
                push_finding(
                    model,
                    "numeric-domain-violation",
                    severity,
                    confidence,
                    violation.message.clone(),
                    Some(krate.file_label(violation.span)),
                    vec![function_id.clone()],
                );
            }
            for nan_shape in &graph.zero_times_infinity {
                let (severity, confidence) =
                    numeric_finding_severity(true, nan_shape.impact, inference_entrypoint);
                push_finding(
                    model,
                    "zero-times-infinity",
                    severity,
                    confidence,
                    nan_shape.message.clone(),
                    Some(krate.file_label(nan_shape.span)),
                    vec![function_id.clone()],
                );
                if nan_shape.library_cite.is_some()
                    && matches!(
                        nan_shape.impact,
                        NumericImpact::TrainingLossNaN | NumericImpact::GradientPoison
                    )
                {
                    push_finding(
                        model,
                        "unstable-library-loss",
                        FindingSeverity::Error,
                        Confidence::Proven,
                        nan_shape.message.clone(),
                        Some(krate.file_label(nan_shape.span)),
                        vec![function_id.clone()],
                    );
                }
            }
            for dead in dead_params {
                let mut related = vec![function_id.clone()];
                if let Some(tensor) = tensor_ids[dead.0].clone() {
                    related.push(tensor);
                }
                push_finding(
                    model,
                    "dead-gradient-path",
                    FindingSeverity::Warning,
                    Confidence::Conditional,
                    format!(
                        "trainable expression `{}` has no differentiable path to a loss",
                        node_name(&graph.node(dead).kind)
                    ),
                    Some(krate.file_label(graph.node(dead).span)),
                    related,
                );
            }
        } // execution phase
    }
}

fn link_implicit_parameter_reads(
    model: &mut ModelIr,
    graph: &dataflow::ExprGraph,
    operation: dataflow::NodeId,
    function_id: &StableId,
    operation_id: &StableId,
) {
    let Some(module_node) = graph
        .edges
        .iter()
        .find(|edge| edge.to == operation && edge.label.as_deref() == Some("module"))
        .map(|edge| graph.node(edge.from))
    else {
        return;
    };
    let NodeKind::Local { name } = &module_node.kind else {
        return;
    };
    let field = name.strip_prefix('.').unwrap_or(name);
    let Some(type_name) = module_node
        .type_name
        .as_deref()
        .and_then(|name| name.rsplit("::").next())
    else {
        return;
    };
    let modules = model
        .modules
        .iter()
        .filter(|module| {
            module.field.as_deref() == Some(field)
                && module.type_name.rsplit("::").next() == Some(type_name)
        })
        .map(|module| module.id.clone())
        .collect::<HashSet<_>>();
    let owner_type = model
        .functions
        .iter()
        .find(|function| &function.id == function_id)
        .and_then(|function| function.owner_type.as_deref());
    let mut components = model
        .components
        .iter()
        .filter(|component| Some(component.qualified_name.as_str()) == owner_type)
        .map(|component| component.id.clone())
        .collect::<HashSet<_>>();
    if let Some(owner_type) = owner_type {
        let owner_leaf = owner_type.rsplit("::").next().unwrap_or(owner_type);
        components.extend(
            model
                .modules
                .iter()
                .filter(|module| {
                    module.type_name.rsplit("::").next() == Some(owner_leaf)
                        || module
                            .qualified_type
                            .as_deref()
                            .and_then(|name| name.rsplit("::").next())
                            == Some(owner_leaf)
                })
                .map(|module| module.component.clone()),
        );
    }
    for parameter in &mut model.parameters {
        let field_prefix_match = components.contains(&parameter.component)
            && parameter.key.split('.').any(|segment| segment == field);
        if (modules.contains(&parameter.module) || field_prefix_match)
            && !parameter.uses.contains(operation_id)
        {
            parameter.uses.push(operation_id.clone());
        }
    }
}

#[cfg(feature = "runtime")]
fn aggregate_edge_timings(trace: &RuntimeTrace) -> Vec<EdgeTimingSummary> {
    let mut edge_durations: BTreeMap<(String, String), Vec<u64>> = BTreeMap::new();
    for edge in &trace.edge_timings {
        let key = (edge.from_static_id.clone(), edge.to_static_id.clone());
        edge_durations
            .entry(key)
            .or_default()
            .push(edge.duration_ns);
    }
    edge_durations
        .into_iter()
        .filter_map(|((from, to), durations)| {
            Some(EdgeTimingSummary {
                from: StableId(from),
                to: StableId(to),
                timing: TimingStats::from_durations(&durations)?,
            })
        })
        .collect()
}

#[cfg(feature = "runtime")]
fn merge_runtime(model: &mut ModelIr, trace: &RuntimeTrace) {
    let expected = ExpectedIdentity {
        analysis_id: Some(model.analysis_id.0.clone()),
        build_id: model.cargo.as_ref().map(|cargo| cargo.build_id.clone()),
    };
    let audit = trace.audit_with_identity(Some(&expected));
    let identity_matches = audit.identity_mismatches.is_empty();

    if identity_matches {
        let static_ids = trace
            .tensors
            .iter()
            .filter_map(|observation| observation.static_id.as_deref())
            .collect::<BTreeSet<_>>();
        let mut unknown_static_ids = 0usize;
        for static_id in static_ids {
            let Some(observation) = trace.agreed_tensor(static_id) else {
                continue;
            };
            let Some(tensor) = model
                .tensors
                .iter_mut()
                .find(|tensor| tensor.id.0 == static_id)
            else {
                unknown_static_ids += 1;
                continue;
            };
            tensor.shape.rank = Some(observation.shape.len());
            tensor.shape.dimensions = observation
                .shape
                .iter()
                .map(|dimension| crate::model_ir::Dimension {
                    name: None,
                    expr: dimension.to_string(),
                })
                .collect();
            tensor.dtype = observation.dtype.clone();
            tensor.device = parse_device(&observation.device);
            tensor.layout = if observation.contiguous {
                LayoutFact::Contiguous
            } else {
                LayoutFact::NonContiguous
            };
            tensor.requires_grad = Some(observation.requires_grad);
            tensor.evidence.push(Evidence {
                kind: EvidenceKind::Runtime,
                confidence: Confidence::Proven,
                source: observation.source.clone(),
                detail: format!("agreed runtime observation {}", observation.event_id),
            });
        }
        if unknown_static_ids > 0 {
            push_finding(
                model,
                "runtime-unmapped",
                FindingSeverity::Information,
                Confidence::Unknown,
                format!(
                    "{unknown_static_ids} runtime tensor identities did not exist in this analysis"
                ),
                None,
                Vec::new(),
            );
        }

        for parameter in &mut model.parameters {
            if let Some(gradient) = trace.gradient(&parameter.builder_root, &parameter.key) {
                parameter.evidence.push(Evidence {
                    kind: EvidenceKind::Runtime,
                    confidence: Confidence::Proven,
                    source: None,
                    detail: format!("gradient {:?}, norm {:?}", gradient.state, gradient.norm),
                });
            }
        }
        for observation in &trace.operations {
            let Some(static_id) = observation.static_id.as_deref() else {
                continue;
            };
            if let Some(operation) = model
                .operations
                .iter_mut()
                .find(|operation| operation.id.0 == static_id)
            {
                let mut detail = format!(
                    "runtime operation {} observed as {}",
                    observation.event_id, observation.op
                );
                if let Some(duration_ns) = observation.duration_ns {
                    detail.push_str(&format!(" duration_ns={duration_ns}"));
                }
                operation.evidence.push(Evidence {
                    kind: EvidenceKind::Runtime,
                    confidence: Confidence::Proven,
                    source: observation.source.clone(),
                    detail,
                });
            }
        }
        let mut op_durations: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for observation in &trace.operations {
            let Some(static_id) = observation.static_id.as_deref() else {
                continue;
            };
            let Some(duration_ns) = observation.duration_ns else {
                continue;
            };
            op_durations
                .entry(static_id.to_string())
                .or_default()
                .push(duration_ns);
        }
        for (static_id, durations) in op_durations {
            let Some(timing) = TimingStats::from_durations(&durations) else {
                continue;
            };
            if let Some(operation) = model
                .operations
                .iter_mut()
                .find(|operation| operation.id.0 == static_id)
            {
                operation.timing = Some(timing);
            }
        }
        for identity in audit
            .missing_gradients
            .iter()
            .chain(&audit.zero_gradients)
            .chain(&audit.non_finite_gradients)
        {
            let Some(gradient) = trace.gradient(&identity.root, &identity.key) else {
                continue;
            };
            let severity = if gradient.state == GradientState::NonFinite {
                FindingSeverity::Error
            } else {
                FindingSeverity::Warning
            };
            push_finding(
                model,
                "runtime-gradient",
                severity,
                Confidence::Proven,
                format!(
                    "{}:{} runtime gradient is {:?}",
                    gradient.root, gradient.key, gradient.state
                ),
                None,
                model
                    .parameters
                    .iter()
                    .filter(|parameter| {
                        parameter.builder_root == gradient.root && parameter.key == gradient.key
                    })
                    .map(|parameter| parameter.id.clone())
                    .collect(),
            );
        }
    }
    for conflict in &audit.tensor_conflicts {
        push_finding(
            model,
            "runtime-tensor-conflict",
            FindingSeverity::Error,
            Confidence::Unknown,
            format!(
                "{} has conflicting runtime {:?}: {}",
                conflict.static_id,
                conflict.kind,
                conflict.values.join(", ")
            ),
            None,
            vec![StableId(conflict.static_id.clone())],
        );
    }
    for conflict in &audit.gradient_conflicts {
        push_finding(
            model,
            "runtime-gradient-conflict",
            FindingSeverity::Error,
            Confidence::Unknown,
            format!(
                "{}:{} has contradictory gradient observations: {}",
                conflict.identity.root,
                conflict.identity.key,
                conflict.states.join(", ")
            ),
            None,
            model
                .parameters
                .iter()
                .filter(|parameter| {
                    parameter.builder_root == conflict.identity.root
                        && parameter.key == conflict.identity.key
                })
                .map(|parameter| parameter.id.clone())
                .collect(),
        );
    }
    for mismatch in &audit.identity_mismatches {
        push_finding(
            model,
            "runtime-identity",
            FindingSeverity::Error,
            Confidence::Unknown,
            format!(
                "runtime {} mismatch: expected {}, observed {}; trace was not merged",
                mismatch.field, mismatch.expected, mismatch.observed
            ),
            None,
            Vec::new(),
        );
    }
    model.coverage.runtime_observations = trace.tensors.len()
        + trace.operations.len()
        + trace.gradients.len()
        + trace.values.len()
        + trace.edge_timings.len();
    let edge_timing_summaries = aggregate_edge_timings(trace);
    let avg_operation_duration_ns = trace
        .operations
        .iter()
        .filter_map(|op| op.duration_ns)
        .reduce(|a, b| a.saturating_add(b))
        .and_then(|total| {
            let count = trace
                .operations
                .iter()
                .filter(|op| op.duration_ns.is_some())
                .count();
            (count > 0).then_some(total / count as u64)
        });
    model.runtime = Some(RuntimeSummary {
        trace_schema: trace.schema.clone(),
        entrypoint: Some(trace.run.entrypoint.clone()),
        profile: Some(trace.run.profile.clone()),
        tensor_observations: trace.tensors.len(),
        operation_observations: trace.operations.len(),
        gradient_observations: trace.gradients.len(),
        missing_gradients: audit.missing_gradients.len(),
        zero_gradients: audit.zero_gradients.len(),
        non_finite_gradients: audit.non_finite_gradients.len(),
        tensor_conflicts: audit.tensor_conflicts.len(),
        gradient_conflicts: audit.gradient_conflicts.len(),
        identity_mismatches: audit.identity_mismatches.len(),
        first_non_finite_step: audit.first_non_finite_step,
        saturating_activations: audit.saturating_activations.len(),
        value_observations: trace.values.len(),
        execution_phase: trace
            .run
            .phase
            .as_deref()
            .and_then(crate::phase::ExecutionPhase::parse),
        avg_operation_duration_ns,
        edge_timings: edge_timing_summaries,
    });
}

fn flag_candle_semantics_version(model: &mut ModelIr, cargo: &CargoContext) {
    for (package, version) in &cargo.candle_versions {
        if package == "candle-core" || package == "candle-nn" {
            let supported = op_semantics::is_audited_candle_version(version);
            if !supported {
                push_finding(
                    model,
                    "candle-semantics-version",
                    FindingSeverity::Warning,
                    Confidence::Proven,
                    format!(
                        "{package} {version} differs from the audited operation catalogs \
                         {}; \
                         unknown or changed operations remain explicit",
                        op_semantics::AUDITED_CANDLE_VERSION
                    ),
                    Some(cargo.manifest_path.to_string_lossy().into_owned()),
                    Vec::new(),
                );
            }
        }
    }
    let core = cargo.candle_versions.get("candle-core");
    let nn = cargo.candle_versions.get("candle-nn");
    if let (Some(core), Some(nn)) = (core, nn) {
        if core != nn {
            push_finding(
                model,
                "candle-semantics-version",
                FindingSeverity::Warning,
                Confidence::Proven,
                format!(
                    "candle-core {core} and candle-nn {nn} do not match; version-sensitive \
                     constructor and autograd rules remain Unknown"
                ),
                Some(cargo.manifest_path.to_string_lossy().into_owned()),
                Vec::new(),
            );
        }
    }
}

#[derive(Default)]
struct CallCollector {
    calls: Vec<String>,
}

impl<'ast> Visit<'ast> for CallCollector {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*node.func {
            self.calls
                .push(path.path.to_token_stream().to_string().replace(' ', ""));
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.calls.push(node.method.to_string());
        visit::visit_expr_method_call(self, node);
    }
}

#[derive(Debug)]
struct ArtifactObservation {
    operation: String,
    path_expr: String,
    label: String,
    produced: bool,
}

#[derive(Default)]
struct ArtifactVisitor {
    observed: Vec<ArtifactObservation>,
}

impl<'ast> Visit<'ast> for ArtifactVisitor {
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init {
            let path_expr = format!(
                "{} = {}",
                node.pat.to_token_stream(),
                init.expr.to_token_stream()
            );
            let strings = string_literals(&init.expr);
            if let Some(label) = strings
                .iter()
                .rev()
                .find(|value| is_artifact_literal(value))
                .cloned()
                .filter(|_| path_expr.len() <= 256)
            {
                self.observed.push(ArtifactObservation {
                    operation: "path_binding".to_string(),
                    path_expr,
                    label,
                    produced: false,
                });
            }
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        for field in &node.fields {
            let strings = string_literals(&field.expr);
            if let Some(label) = strings
                .iter()
                .rev()
                .find(|value| is_artifact_literal(value))
                .cloned()
            {
                self.observed.push(ArtifactObservation {
                    operation: "path_field".to_string(),
                    path_expr: format!(
                        "{} = {}",
                        field.member.to_token_stream(),
                        field.expr.to_token_stream()
                    ),
                    label,
                    produced: false,
                });
            }
        }
        visit::visit_expr_struct(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        let op = match &*node.func {
            syn::Expr::Path(path) => path.path.to_token_stream().to_string().replace(' ', ""),
            _ => String::new(),
        };
        if is_artifact_operation(&op) {
            for argument in &node.args {
                self.observe(&op, argument);
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let op = node.method.to_string();
        if is_artifact_operation(&op) {
            self.observe(&op, &node.receiver);
            for argument in &node.args {
                self.observe(&op, argument);
            }
        }
        visit::visit_expr_method_call(self, node);
    }
}

impl ArtifactVisitor {
    fn observe(&mut self, operation: &str, expr: &syn::Expr) {
        let text = expr.to_token_stream().to_string();
        let strings = string_literals(expr);
        let Some(label) = strings
            .iter()
            .rev()
            .find(|value| is_artifact_literal(value))
            .cloned()
        else {
            return;
        };
        self.observed.push(ArtifactObservation {
            operation: operation.to_string(),
            path_expr: text,
            label,
            produced: is_producer_operation(operation),
        });
    }
}

#[derive(Default)]
struct StringCollector {
    values: Vec<String>,
}

impl<'ast> Visit<'ast> for StringCollector {
    fn visit_lit_str(&mut self, node: &'ast syn::LitStr) {
        self.values.push(node.value());
    }
}

fn string_literals(expr: &syn::Expr) -> Vec<String> {
    let mut collector = StringCollector::default();
    collector.visit_expr(expr);
    collector.values
}

#[derive(Default)]
struct OptimizerVisitor {
    varmaps: BTreeSet<String>,
    includes: Vec<String>,
    excludes: Vec<String>,
    optimizer: Option<String>,
}

impl<'ast> Visit<'ast> for OptimizerVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method = node.method.to_string();
        if method == "all_vars" {
            self.varmaps
                .insert(node.receiver.to_token_stream().to_string().replace(' ', ""));
        }
        if method == "retain" {
            let mut patterns = RetainPatternVisitor::default();
            for argument in &node.args {
                patterns.visit_expr(argument);
            }
            self.includes.extend(patterns.includes);
            self.excludes.extend(patterns.excludes);
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*node.func {
            let name = path.path.to_token_stream().to_string().replace(' ', "");
            if name.ends_with("named_train_vars") {
                if let Some(varmap) = node.args.first().and_then(expr_identifier) {
                    self.varmaps.insert(varmap);
                }
            }
            if name.contains("Adam") || name.contains("Optimizer") {
                self.optimizer = Some(name);
            }
        }
        visit::visit_expr_call(self, node);
    }
}

#[derive(Default)]
struct RetainPatternVisitor {
    negated: bool,
    includes: Vec<String>,
    excludes: Vec<String>,
}

impl<'ast> Visit<'ast> for RetainPatternVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if matches!(
            node.method.to_string().as_str(),
            "contains" | "starts_with" | "ends_with"
        ) {
            if let Some(syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(value),
                ..
            })) = node.args.first()
            {
                if self.negated {
                    self.excludes.push(value.value());
                } else {
                    self.includes.push(value.value());
                }
            }
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_unary(&mut self, node: &'ast syn::ExprUnary) {
        if matches!(node.op, syn::UnOp::Not(_)) {
            self.negated = !self.negated;
            self.visit_expr(&node.expr);
            self.negated = !self.negated;
        } else {
            visit::visit_expr_unary(self, node);
        }
    }
}

fn expr_identifier(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => {
            Some(path.path.segments[0].ident.to_string())
        }
        syn::Expr::Reference(reference) => expr_identifier(&reference.expr),
        syn::Expr::Paren(paren) => expr_identifier(&paren.expr),
        syn::Expr::Group(group) => expr_identifier(&group.expr),
        syn::Expr::Try(value) => expr_identifier(&value.expr),
        _ => None,
    }
}

fn function_id(func: &ImplFn) -> StableId {
    StableId::new(
        "function",
        [
            func.qualified_name.as_str(),
            func.cfg_predicates.join(" && ").as_str(),
        ],
    )
}

fn unique_qualified_struct(krate: &Crate, name: &str) -> Option<String> {
    match krate.struct_candidates(name).as_slice() {
        [def] => Some(def.qualified_name.clone()),
        _ => None,
    }
}

fn source_evidence(source: String, detail: impl Into<String>) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        confidence: Confidence::Proven,
        source: Some(source),
        detail: detail.into(),
    }
}

fn heuristic_source_evidence(source: String, detail: impl Into<String>) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        confidence: Confidence::Heuristic,
        source: Some(source),
        detail: detail.into(),
    }
}

fn numeric_finding_severity(
    proven: bool,
    impact: NumericImpact,
    inference_entrypoint: bool,
) -> (FindingSeverity, Confidence) {
    if !proven {
        return (FindingSeverity::Warning, Confidence::Unknown);
    }
    match impact {
        NumericImpact::TrainingLossNaN | NumericImpact::GradientPoison => {
            (FindingSeverity::Error, Confidence::Proven)
        }
        NumericImpact::InferenceOutputRisk if inference_entrypoint => {
            (FindingSeverity::Error, Confidence::Proven)
        }
        NumericImpact::InferenceOutputRisk | NumericImpact::LocalOnly => {
            (FindingSeverity::Warning, Confidence::Proven)
        }
    }
}

fn push_finding(
    model: &mut ModelIr,
    rule: &str,
    severity: FindingSeverity,
    confidence: Confidence,
    message: String,
    source: Option<String>,
    related: Vec<StableId>,
) {
    model.findings.push(Finding {
        id: StableId::new(
            "finding",
            [rule, source.as_deref().unwrap_or(""), message.as_str()],
        ),
        rule: rule.to_string(),
        severity,
        confidence,
        message,
        source,
        related,
        evidence: Vec::new(),
    });
}

fn visibility(text: &str) -> Visibility {
    match text {
        "" => Visibility::Private,
        "pub" => Visibility::Public,
        "pub(crate)" => Visibility::Crate,
        value if value.starts_with("pub(") => Visibility::Restricted,
        _ => Visibility::Unknown,
    }
}

fn certainty_confidence(certainty: &Certainty) -> Confidence {
    match certainty {
        Certainty::Certain => Confidence::Proven,
        Certainty::Conditional(_) => Confidence::Conditional,
        Certainty::Unknown(_) => Confidence::Unknown,
    }
}

fn is_tensor_type(ty: &str) -> bool {
    ty.split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|segment| segment == "Tensor")
}

fn is_model_entry_name(name: &str) -> bool {
    matches!(
        name,
        "forward"
            | "forward_diff"
            | "forward_features"
            | "forward_t"
            | "encode"
            | "predict"
            | "compress"
            | "conditioning"
            | "condition"
            | "generate"
            | "decode"
            | "loss"
    ) || name.ends_with("_forward")
        || name.ends_with("_loss")
}

fn is_stage_entry_name(name: &str) -> bool {
    name == "prepare"
        || name.starts_with("try_run_")
        || name.starts_with("run_")
            && (name.contains("train") || name.contains("eval") || name.contains("prepare"))
}

fn is_pipeline_stage_call(name: &str) -> bool {
    name == "prepare_data"
        || name == "final_eval"
        || name.starts_with("train_")
        || name.starts_with("evaluate_")
        || name.starts_with("export_")
}

fn stage_variants(krate: &Crate, target: &StableId) -> Vec<String> {
    let _ = (krate, target);
    Vec::new()
}

fn stage_kind(name: &str) -> StageKind {
    if name.contains("eval") {
        StageKind::Evaluate
    } else if name.contains("prepare") || name.contains("data") {
        StageKind::Prepare
    } else if name.contains("train")
        || name.contains("bridge")
        || name == "latent"
        || name == "world"
    {
        StageKind::Train
    } else if name.contains("export") {
        StageKind::Export
    } else if name.starts_with("preflight_") {
        StageKind::Probe
    } else {
        StageKind::Unknown
    }
}

fn stage_display_name(function: &Function) -> String {
    let module = function
        .qualified_name
        .rsplit_once("::")
        .map(|(module, _)| module.rsplit("::").next().unwrap_or(module))
        .unwrap_or_default();
    if module.is_empty() || function.name.contains(module) {
        function.name.clone()
    } else {
        format!("{module}:{}", function.name)
    }
}

fn reachable_components(function: &Function, model: &ModelIr) -> Vec<StableId> {
    let mut components = Vec::new();
    let mut queue = vec![function.id.clone()];
    let mut seen = HashSet::new();
    while let Some(id) = queue.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(func) = model.functions.iter().find(|func| func.id == id) {
            if let Some(owner) = func.owner_type.as_deref() {
                components.extend(
                    model
                        .components
                        .iter()
                        .filter(|component| component.qualified_name == owner)
                        .map(|component| component.id.clone()),
                );
            }
            for edge in model
                .architecture_edges
                .iter()
                .filter(|edge| edge.via_function == id)
            {
                components.push(edge.from.clone());
                components.push(edge.to.clone());
            }
            queue.extend(func.calls.iter().cloned());
        }
    }
    components.sort();
    components.dedup();
    components
}

fn stage_for_function(
    function: &StableId,
    stages: &HashMap<StableId, StableId>,
    model: &ModelIr,
) -> Option<StableId> {
    if let Some(stage) = stages.get(function) {
        return Some(stage.clone());
    }
    model.stages.iter().find_map(|stage| {
        reachable_function(&stage.function, function, model, &mut HashSet::new())
            .then(|| stage.id.clone())
    })
}

fn reachable_function(
    current: &StableId,
    target: &StableId,
    model: &ModelIr,
    seen: &mut HashSet<StableId>,
) -> bool {
    if current == target {
        return true;
    }
    if !seen.insert(current.clone()) {
        return false;
    }
    model
        .functions
        .iter()
        .find(|function| &function.id == current)
        .is_some_and(|function| {
            function
                .calls
                .iter()
                .any(|callee| reachable_function(callee, target, model, seen))
        })
}

fn qualify(module: &str, path: &str) -> String {
    if module.is_empty() {
        path.to_string()
    } else if path.is_empty() {
        module.to_string()
    } else {
        format!("{module}::{path}")
    }
}

fn acquisition_label(acquisition: &Acquisition) -> String {
    match acquisition {
        Acquisition::Constructor { func, .. } => func.clone(),
        Acquisition::RawGet { method } => method.clone(),
    }
}

fn similar_stem(left: &str, right: &str) -> bool {
    fn stem(value: &str) -> String {
        value
            .replace("varmap", "")
            .replace("var_map", "")
            .replace("vb", "")
            .replace(['_', '&'], "")
    }
    let left = stem(left);
    let right = stem(right);
    !left.is_empty() && !right.is_empty() && (left.contains(&right) || right.contains(&left))
}

fn is_running_state(key: &str) -> bool {
    key.contains("running_mean")
        || key.contains("running_var")
        || key.contains("num_batches_tracked")
}

fn is_artifact_operation(name: &str) -> bool {
    let leaf = name.rsplit("::").next().unwrap_or(name);
    matches!(
        leaf,
        "load"
            | "save"
            | "read"
            | "write"
            | "open"
            | "mmap"
            | "from_mmaped_safetensors"
            | "from_buffered_safetensors"
            | "serialize_to_file"
            | "load_buffer"
    )
}

fn is_producer_operation(name: &str) -> bool {
    let leaf = name.rsplit("::").next().unwrap_or(name);
    matches!(leaf, "save" | "write" | "serialize_to_file")
}

fn is_artifact_literal(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        ".safetensors",
        ".safetensor",
        ".json",
        ".jsonl",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".parquet",
        ".arrow",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
}

fn artifact_identity(path_expr: &str) -> String {
    path_expr
        .split_once('=')
        .map(|(binding, _)| binding.trim().replace(' ', ""))
        .filter(|binding| {
            !binding.is_empty()
                && binding
                    .chars()
                    .all(|character| character.is_alphanumeric() || character == '_')
        })
        .unwrap_or_else(|| path_expr.to_string())
}

fn infer_artifact_stage_links(model: &mut ModelIr) {
    let _ = model;
}

fn artifact_kind(label: &str) -> ArtifactKind {
    let lower = label.to_ascii_lowercase();
    if lower.ends_with(".safetensors")
        || lower.ends_with(".bin")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".gguf")
    {
        ArtifactKind::Checkpoint
    } else if lower.contains("eval") || lower.contains("report") || lower.contains("metric") {
        ArtifactKind::EvaluationReport
    } else if lower.ends_with(".parquet") || lower.ends_with(".arrow") {
        ArtifactKind::Dataset
    } else if lower.ends_with(".json") {
        ArtifactKind::Configuration
    } else {
        ArtifactKind::Unknown
    }
}

fn node_name(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Param { name } | NodeKind::Local { name } => name.clone(),
        NodeKind::Call { callee } => format!("result_of_{callee}"),
        NodeKind::Literal { text } => text.clone(),
        NodeKind::Phi => "branch_join".to_string(),
        NodeKind::Return => "return".to_string(),
        NodeKind::Unknown { reason } => format!("unknown:{reason}"),
    }
}

fn shape_rank(shape: Option<&str>) -> Option<usize> {
    let shape = shape?;
    let trimmed = shape.trim();
    if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        Some(
            trimmed[1..trimmed.len() - 1]
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .count(),
        )
    } else {
        None
    }
}

fn layout_for_node(kind: &NodeKind) -> LayoutFact {
    match kind {
        NodeKind::Call { callee } if callee.ends_with("contiguous") => LayoutFact::Contiguous,
        NodeKind::Call { callee }
            if ["transpose", "permute", "narrow", "t"]
                .iter()
                .any(|op| callee.ends_with(op)) =>
        {
            LayoutFact::Strided
        }
        _ => LayoutFact::Unknown,
    }
}

fn shape_rule(op: &str) -> &'static str {
    match op {
        "reshape" | "broadcast_as" | "expand" => "explicit_argument",
        "flatten_all" | "flatten_to" | "flatten_from" => "flatten",
        "squeeze" => "remove_dimension",
        "unsqueeze" => "insert_dimension",
        "transpose" | "permute" | "t" => "permute_dimensions",
        "narrow" => "slice_dimension",
        "matmul" | "broadcast_matmul" => "matrix_product",
        _ => "preserve_or_operation_defined",
    }
}

#[cfg(feature = "runtime")]
fn parse_device(device: &str) -> DeviceFact {
    let lower = device.to_ascii_lowercase();
    if lower == "cpu" {
        DeviceFact::Cpu
    } else if lower.starts_with("cuda") {
        let ordinal = lower
            .split([':', '(', ')'])
            .find_map(|part| part.parse::<u32>().ok());
        DeviceFact::Cuda { ordinal }
    } else if lower.starts_with("metal") {
        DeviceFact::Metal
    } else {
        DeviceFact::Unknown
    }
}
