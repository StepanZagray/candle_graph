//! Bounded queries over [`crate::model_ir::ModelIr`].
//!
//! Agents should not have to ingest an entire model graph to answer a local question. Queries
//! therefore return compact, deterministically ordered records and make truncation explicit.
//! Listing kinds omit tensor contracts and evidence payloads; narrow singular queries (or a
//! selector on evidence-bearing lists) unlock detail, with `drill_down` hints pointing the way.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::model_ir::{Finding, Function, ModelIr, StableId, TensorContract, ExecutionPhase};

pub const QUERY_SCHEMA: &str = "candle-graph/query/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    Summary,
    Doctor,
    Architecture,
    Cargo,
    Components,
    Component,
    Modules,
    Composition,
    Assembly,
    Pipeline,
    Stages,
    Artifacts,
    Entrypoints,
    Functions,
    Function,
    Parameters,
    Parameter,
    Tensors,
    Tensor,
    Operations,
    Operation,
    Optimizers,
    Runtime,
    Findings,
    Path,
    /// Compact agent-oriented rollup for model repair and improvement workflows.
    ModelImprovement,
    /// Phase-filtered graph: tensors + operations for training autograd paths.
    GraphTrain,
    /// Phase-filtered graph: tensors + operations for inference no-grad paths.
    GraphInfer,
    /// Runtime profile rollup: operation and edge timings merged from a v3 trace.
    Profile,
}

impl std::str::FromStr for QueryKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.replace('-', "_").as_str() {
            "summary" => Ok(Self::Summary),
            "doctor" | "trust" | "coverage" => Ok(Self::Doctor),
            "architecture" | "model" => Ok(Self::Architecture),
            "cargo" | "cfg" | "features" => Ok(Self::Cargo),
            "components" => Ok(Self::Components),
            "component" => Ok(Self::Component),
            "modules" | "module" => Ok(Self::Modules),
            "composition" | "edges" | "contains" => Ok(Self::Composition),
            "assembly" | "wiring" | "checkpoint_assembly" => Ok(Self::Assembly),
            "pipeline" => Ok(Self::Pipeline),
            "stages" => Ok(Self::Stages),
            "artifacts" => Ok(Self::Artifacts),
            "entrypoints" => Ok(Self::Entrypoints),
            "functions" => Ok(Self::Functions),
            "function" => Ok(Self::Function),
            "parameters" => Ok(Self::Parameters),
            "parameter" => Ok(Self::Parameter),
            "tensors" => Ok(Self::Tensors),
            "tensor" => Ok(Self::Tensor),
            "operations" | "ops" => Ok(Self::Operations),
            "operation" | "op" => Ok(Self::Operation),
            "optimizers" | "optimizer" => Ok(Self::Optimizers),
            "runtime" | "gradient_audit" | "gradients" => {
                require_runtime_feature("runtime")?;
                Ok(Self::Runtime)
            }
            "findings" | "diagnostics" => Ok(Self::Findings),
            "path" | "trace" => Ok(Self::Path),
            "model_improvement" | "model-improvement" | "improvement" | "agent" => {
                Ok(Self::ModelImprovement)
            }
            "graph_train" | "graph-train" | "train_graph" | "train-graph" => {
                require_runtime_feature("graph_train")?;
                Ok(Self::GraphTrain)
            }
            "graph_infer" | "graph-infer" | "infer_graph" | "infer-graph" => {
                require_runtime_feature("graph_infer")?;
                Ok(Self::GraphInfer)
            }
            "profile" | "profiler" | "timings" => {
                require_runtime_feature("profile")?;
                Ok(Self::Profile)
            }
            other => bail!("unknown query kind `{other}`"),
        }
    }
}

fn require_runtime_feature(kind: &str) -> Result<()> {
    #[cfg(not(feature = "runtime"))]
    {
        bail!(
            "query kind `{kind}` requires the `runtime` crate feature; rebuild with `--features runtime`"
        );
    }
    #[cfg(feature = "runtime")]
    {
        let _ = kind;
        Ok(())
    }
}

fn ensure_runtime_query(kind: QueryKind) -> Result<()> {
    if matches!(
        kind,
        QueryKind::Runtime | QueryKind::GraphTrain | QueryKind::GraphInfer | QueryKind::Profile
    ) {
        require_runtime_feature(match kind {
            QueryKind::Runtime => "runtime",
            QueryKind::GraphTrain => "graph_train",
            QueryKind::GraphInfer => "graph_infer",
            QueryKind::Profile => "profile",
            _ => unreachable!(),
        })?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryRequest {
    pub kind: QueryKind,
    pub selector: Option<String>,
    pub to: Option<String>,
    pub limit: usize,
    /// Deterministic start index into the sorted result set; exposed as CLI `--offset`.
    #[serde(default)]
    pub offset: usize,
}

impl QueryRequest {
    pub fn new(kind: QueryKind) -> Self {
        Self {
            kind,
            selector: None,
            to: None,
            limit: 100,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResponse {
    pub schema: String,
    pub analysis_id: StableId,
    pub kind: QueryKind,
    pub selector: Option<String>,
    pub total: usize,
    pub returned: usize,
    #[serde(default)]
    pub offset: usize,
    pub truncated: bool,
    pub items: Vec<Value>,
}

pub fn execute(model: &ModelIr, request: &QueryRequest) -> Result<QueryResponse> {
    if request.limit == 0 {
        bail!("query limit must be greater than zero");
    }
    ensure_runtime_query(request.kind)?;
    let mut items = match request.kind {
        QueryKind::Summary => vec![summary(model)],
        QueryKind::Doctor => vec![doctor(model)],
        QueryKind::Architecture => vec![architecture(model)],
        QueryKind::Cargo => vec![json!({
            "id": "cargo",
            "context": model.cargo,
            "drill_down": [
                {"kind": "summary"},
                {"kind": "functions"},
            ],
        })],
        QueryKind::Components => model
            .components
            .iter()
            .map(|component| {
                json!({
                    "id": component.id,
                    "name": component.name,
                    "qualified_name": component.qualified_name,
                    "source": component.source,
                    "builders": component.builders.iter().map(|builder| json!({
                        "name": builder.name,
                        "role": builder.role,
                    })).collect::<Vec<_>>(),
                    "modules": component.modules.len(),
                    "parameters": component.parameters.len(),
                    "entrypoints": component.entrypoints,
                    "drill_down": [
                        {"kind": "component", "select": component.qualified_name},
                        {"kind": "modules", "select": component.qualified_name},
                        {"kind": "entrypoints", "select": component.name},
                        {"kind": "parameters", "select": component.name},
                    ],
                })
            })
            .collect(),
        QueryKind::Component => {
            let selector = required_selector(request)?;
            model
                .components
                .iter()
                .filter(|item| {
                    matches_text(selector, [&item.id.0, &item.name, &item.qualified_name])
                })
                .map(|component| {
                    let mut value = serde_json::to_value(component)?;
                    if let Some(object) = value.as_object_mut() {
                        object.insert(
                            "drill_down".into(),
                            json!([
                                {"kind": "modules", "select": component.qualified_name},
                                {"kind": "entrypoints", "select": component.name},
                                {"kind": "parameters", "select": component.name},
                                {"kind": "functions", "select": component.name},
                                {"kind": "tensors", "select": component.name},
                                {"kind": "composition", "select": component.qualified_name},
                            ]),
                        );
                    }
                    Ok(value)
                })
                .collect::<serde_json::Result<Vec<_>>>()?
        }
        QueryKind::Modules => {
            let component_names = component_name_lookup(model);
            model
                .modules
                .iter()
                .filter(|module| {
                    let component_name = component_names
                        .get(&module.component)
                        .map(String::as_str)
                        .unwrap_or("");
                    optional_matches(
                        request.selector.as_deref(),
                        [
                            &module.id.0,
                            &module.type_name,
                            module.qualified_type.as_deref().unwrap_or(""),
                            module.field.as_deref().unwrap_or(""),
                            &module.prefix,
                            &module.builder_root,
                            component_name,
                        ],
                    )
                })
                .map(|module| module_listing(module, &component_names))
                .collect()
        }
        QueryKind::Composition => model
            .architecture_edges
            .iter()
            .filter(|edge| edge.id.0.starts_with("composition-edge:"))
            .filter(|edge| {
                composition_matches(
                    model,
                    request.selector.as_deref(),
                    edge.from.clone(),
                    edge.to.clone(),
                )
            })
            .map(|edge| composition_listing(model, edge))
            .collect(),
        QueryKind::Assembly => model
            .assembly_sites
            .iter()
            .filter(|site| {
                optional_matches(
                    request.selector.as_deref(),
                    [
                        &site.id.0,
                        &site.component_name,
                        &site.function_name,
                        &site.builder_root,
                        site.varmap.as_deref().unwrap_or(""),
                        site.checkpoint_load.as_deref().unwrap_or(""),
                    ],
                )
            })
            .map(assembly_listing)
            .collect(),
        QueryKind::Pipeline => vec![json!({
            "id": "pipeline",
            "stages": model.stages.iter().map(|stage| json!({
                "id": stage.id,
                "name": stage.name,
                "kind": stage.kind,
                "order": stage.order,
                "dispatch": stage.dispatch,
                "subprocess_key": stage.subprocess_key,
                "cli_flags": stage.cli_flags,
            })).collect::<Vec<_>>(),
            "subprocess_stages": model.coverage.subprocess_stages,
            "artifacts": model.artifacts.iter().map(|artifact| json!({
                "id": artifact.id,
                "name": artifact.name,
            })).collect::<Vec<_>>(),
            "optimizers": model.optimizers.len(),
            "drill_down": [
                {"kind": "stages"},
                {"kind": "artifacts"},
                {"kind": "optimizers"},
            ],
        })],
        QueryKind::Stages => model
            .stages
            .iter()
            .filter(|stage| {
                optional_matches(request.selector.as_deref(), [&stage.id.0, &stage.name])
            })
            .map(|stage| {
                json!({
                    "id": stage.id,
                    "name": stage.name,
                    "kind": stage.kind,
                    "order": stage.order,
                    "dispatch": stage.dispatch,
                    "subprocess_key": stage.subprocess_key,
                    "cli_flags": stage.cli_flags,
                    "launcher": stage.launcher,
                    "orchestrator": stage.orchestrator,
                    "source": stage.source,
                    "components": stage.components.len(),
                    "drill_down": [
                        {"kind": "function", "select": stage.source},
                    ],
                })
            })
            .collect(),
        QueryKind::Artifacts => model
            .artifacts
            .iter()
            .filter(|artifact| {
                optional_matches(
                    request.selector.as_deref(),
                    [&artifact.id.0, &artifact.name, &artifact.path_expr],
                )
            })
            .map(serde_json::to_value)
            .collect::<serde_json::Result<Vec<_>>>()?,
        QueryKind::Entrypoints => {
            let mut entrypoints: Vec<_> = model
                .functions
                .iter()
                .filter(|function| function.is_entrypoint)
                .filter(|function| {
                    optional_matches(
                        request.selector.as_deref(),
                        [&function.id.0, &function.name, &function.qualified_name],
                    )
                })
                .map(|function| (function, is_component_entrypoint(function, model)))
                .collect();
            entrypoints.sort_by(|(left, left_component), (right, right_component)| {
                right_component
                    .cmp(left_component)
                    .then_with(|| left.qualified_name.cmp(&right.qualified_name))
            });
            entrypoints
                .into_iter()
                .map(|(function, is_component)| function_listing(function, is_component))
                .collect()
        }
        QueryKind::Functions => model
            .functions
            .iter()
            .filter(|function| {
                optional_matches(
                    request.selector.as_deref(),
                    [&function.id.0, &function.name, &function.qualified_name],
                )
            })
            .map(|function| function_listing(function, is_component_entrypoint(function, model)))
            .collect(),
        QueryKind::Function => {
            let selector = required_selector(request)?;
            model
                .functions
                .iter()
                .filter(|function| {
                    matches_text(
                        selector,
                        [&function.id.0, &function.name, &function.qualified_name],
                    )
                })
                .map(function_detail)
                .collect()
        }
        QueryKind::Parameters => model
            .parameters
            .iter()
            .filter(|parameter| {
                optional_matches(
                    request.selector.as_deref(),
                    [
                        &parameter.id.0,
                        &parameter.key,
                        &parameter.builder_root,
                        &parameter.kind,
                    ],
                )
            })
            .map(|parameter| {
                json!({
                    "id": parameter.id,
                    "component": parameter.component,
                    "module": parameter.module,
                    "key": parameter.key,
                    "builder_root": parameter.builder_root,
                    "role": parameter.role,
                    "kind": parameter.kind,
                    "symbolic_shape": parameter.symbolic_shape,
                    "checkpoint_shape": parameter.checkpoint_shape,
                    "checkpoint_dtype": parameter.checkpoint_dtype,
                    "source": parameter.source,
                    "uses": parameter.uses,
                    "optimizer_memberships": parameter.optimizer_memberships,
                    "drill_down": [
                        {"kind": "parameter", "select": parameter.id.0},
                    ],
                })
            })
            .collect(),
        QueryKind::Parameter => {
            let selector = required_selector(request)?;
            model
                .parameters
                .iter()
                .filter(|parameter| {
                    matches_text(
                        selector,
                        [
                            &parameter.id.0,
                            &parameter.key,
                            &parameter.builder_root,
                            &format!("{}:{}", parameter.builder_root, parameter.key),
                        ],
                    )
                })
                .map(serde_json::to_value)
                .collect::<serde_json::Result<Vec<_>>>()?
        }
        QueryKind::Tensors => model
            .tensors
            .iter()
            .filter(|tensor| tensor_matches(model, tensor, request.selector.as_deref()))
            .map(|tensor| tensor_listing(model, tensor))
            .collect(),
        QueryKind::Tensor => {
            let selector = required_selector(request)?;
            model
                .tensors
                .iter()
                .filter(|tensor| tensor_matches(model, tensor, Some(selector)))
                .map(serde_json::to_value)
                .collect::<serde_json::Result<Vec<_>>>()?
        }
        QueryKind::Operations => model
            .operations
            .iter()
            .filter(|operation| {
                let function_name = model
                    .functions
                    .iter()
                    .find(|function| function.id == operation.function)
                    .map(|function| function.qualified_name.as_str())
                    .unwrap_or_default();
                optional_matches(
                    request.selector.as_deref(),
                    [
                        &operation.id.0,
                        &operation.name,
                        operation.qualified_name.as_deref().unwrap_or_default(),
                        &operation.function.0,
                        function_name,
                    ],
                )
            })
            .map(|operation| {
                json!({
                    "id": operation.id,
                    "name": operation.name,
                    "qualified_name": operation.qualified_name,
                    "function": operation.function,
                    "inputs": operation.inputs.len(),
                    "output": operation.output,
                    "source": operation.source,
                    "drill_down": [
                        {"kind": "operation", "select": operation.id.0},
                        {"kind": "function", "select": operation.function.0},
                    ],
                })
            })
            .collect(),
        QueryKind::Operation => {
            let selector = required_selector(request)?;
            model
                .operations
                .iter()
                .filter(|operation| {
                    matches_text(
                        selector,
                        [
                            &operation.id.0,
                            &operation.name,
                            operation.qualified_name.as_deref().unwrap_or_default(),
                        ],
                    )
                })
                .map(serde_json::to_value)
                .collect::<serde_json::Result<Vec<_>>>()?
        }
        QueryKind::Optimizers => model
            .optimizers
            .iter()
            .filter(|optimizer| {
                optional_matches(
                    request.selector.as_deref(),
                    [
                        &optimizer.id.0,
                        &optimizer.optimizer,
                        &optimizer.varmap,
                        &optimizer.stage.0,
                    ],
                )
            })
            .map(serde_json::to_value)
            .collect::<serde_json::Result<Vec<_>>>()?,
        QueryKind::Runtime => vec![runtime_query(model)],
        QueryKind::Findings => model
            .findings
            .iter()
            .filter(|finding| {
                optional_matches(
                    request.selector.as_deref(),
                    [&finding.id.0, &finding.rule, &finding.message],
                )
            })
            .map(|finding| {
                if request.selector.as_deref() == Some(finding.id.0.as_str()) {
                    serde_json::to_value(finding).unwrap_or_else(|_| json!({}))
                } else {
                    finding_listing(finding)
                }
            })
            .collect(),
        QueryKind::ModelImprovement => vec![model_improvement(model)],
        QueryKind::GraphTrain => vec![phase_graph(model, ExecutionPhase::Train)],
        QueryKind::GraphInfer => vec![phase_graph(model, ExecutionPhase::Infer)],
        QueryKind::Profile => vec![profile_query(model)],
        QueryKind::Path => {
            let from = required_selector(request)?;
            let to = request
                .to
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("path query requires `to`"))?;
            let from = resolve_id(model, from)?;
            let to = resolve_id(model, to)?;
            shortest_path(model, &from, &to)?
                .into_iter()
                .map(|id| describe_id(model, &id))
                .collect()
        }
    };

    if matches!(
        request.kind,
        QueryKind::Component
            | QueryKind::Function
            | QueryKind::Parameter
            | QueryKind::Tensor
            | QueryKind::Operation
    ) && items.len() > 1
    {
        bail!(
            "{:?} selector matched {} records; use an exact stable id from the compact listing",
            request.kind,
            items.len()
        );
    }

    if !matches!(
        request.kind,
        QueryKind::Summary
            | QueryKind::Doctor
            | QueryKind::Architecture
            | QueryKind::Cargo
            | QueryKind::Pipeline
            | QueryKind::Stages
            | QueryKind::Runtime
            | QueryKind::ModelImprovement
            | QueryKind::GraphTrain
            | QueryKind::GraphInfer
            | QueryKind::Profile
    ) {
        items.sort_by_key(stable_value_key);
    }
    let total = items.len();
    let offset = request.offset.min(total);
    items = items.into_iter().skip(offset).take(request.limit).collect();
    Ok(QueryResponse {
        schema: QUERY_SCHEMA.to_string(),
        analysis_id: model.analysis_id.clone(),
        kind: request.kind,
        selector: request.selector.clone(),
        total,
        returned: items.len(),
        offset,
        truncated: offset + items.len() < total,
        items,
    })
}

pub fn render_text(response: &QueryResponse) -> String {
    let mut out = format!(
        "query {:?}: {} result{}",
        response.kind,
        response.total,
        if response.total == 1 { "" } else { "s" }
    );
    if response.offset > 0 || response.truncated {
        out.push_str(&format!(
            " (offset {}, showing {})",
            response.offset, response.returned
        ));
    }
    out.push('\n');
    for item in &response.items {
        if let Some(object) = item.as_object() {
            let id = object
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| object.get("name").and_then(Value::as_str))
                .unwrap_or("-");
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| object.get("key").and_then(Value::as_str))
                .or_else(|| object.get("message").and_then(Value::as_str));
            out.push_str("  ");
            out.push_str(id);
            if let Some(name) = name.filter(|name| *name != id) {
                out.push_str("  ");
                out.push_str(name);
            }
            if let Some(source) = object.get("source").and_then(Value::as_str) {
                out.push_str("  @");
                out.push_str(source);
            }
            if let Some(hints) = object.get("drill_down").and_then(Value::as_array) {
                let kinds = hints
                    .iter()
                    .filter_map(|hint| hint.get("kind").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(", ");
                if !kinds.is_empty() {
                    out.push_str("  → ");
                    out.push_str(&kinds);
                }
            }
            out.push('\n');
        } else {
            out.push_str("  ");
            out.push_str(&item.to_string());
            out.push('\n');
        }
    }
    out
}

fn summary(model: &ModelIr) -> Value {
    json!({
        "id": "summary",
        "schema": model.schema,
        "analysis_id": model.analysis_id,
        "cargo": model.cargo.as_ref().map(|cargo| json!({
            "build_id": cargo.build_id,
            "package": cargo.package_name,
            "version": cargo.package_version,
            "target": cargo.selected_target,
            "active_features": cargo.active_features,
            "candle_packages": cargo.candle_packages,
        })),
        "coverage": model.coverage,
        "components": model.components.iter().map(|c| &c.name).collect::<Vec<_>>(),
        "pipeline": model.stages.iter().map(|stage| json!({
            "id": stage.id,
            "name": stage.name,
            "kind": stage.kind,
            "order": stage.order,
            "dispatch": stage.dispatch,
            "subprocess_key": stage.subprocess_key,
            "cli_flags": stage.cli_flags,
        })).collect::<Vec<_>>(),
        "finding_counts": finding_counts(model),
        "runtime": model.runtime,
        "drill_down": [
            {"kind": "architecture"},
            {"kind": "composition"},
            {"kind": "components"},
            {"kind": "modules"},
            {"kind": "functions"},
            {"kind": "entrypoints"},
            {"kind": "tensors"},
            {"kind": "findings"},
            {"kind": "cargo"},
            {"kind": "doctor"},
        ],
    })
}

fn doctor(model: &ModelIr) -> Value {
    let mut by_rule = BTreeMap::new();
    let mut unknown = 0usize;
    for finding in &model.findings {
        *by_rule.entry(finding.rule.clone()).or_insert(0usize) += 1;
        if matches!(
            finding.confidence,
            crate::model_ir::Confidence::Unknown | crate::model_ir::Confidence::Heuristic
        ) {
            unknown += 1;
        }
    }
    let source_incomplete = by_rule.get("source-load").copied().unwrap_or(0);
    let semantic_version_gaps = by_rule
        .get("candle-semantics-version")
        .copied()
        .unwrap_or(0);
    let dtype_risks = by_rule.get("dtype-risk").copied().unwrap_or(0);
    let dtype_conflicts = by_rule.get("dtype-conflict").copied().unwrap_or(0);
    let tensor_dtype_pct = pct(model.coverage.tensors_with_dtype, model.coverage.tensors);
    let tensor_shape_pct = pct(model.coverage.tensors_with_shape, model.coverage.tensors);
    let tensor_device_pct = pct(model.coverage.tensors_with_device, model.coverage.tensors);
    json!({
        "id": "doctor",
        "analysis_id": model.analysis_id,
        "cargo_available": model.cargo.is_some(),
        "coverage": model.coverage,
        "coverage_quality": {
            "tensor_dtype_pct": tensor_dtype_pct,
            "tensor_shape_pct": tensor_shape_pct,
            "tensor_device_pct": tensor_device_pct,
            "dtype_risks": dtype_risks,
            "dtype_conflicts": dtype_conflicts,
            "component_entrypoints": model.coverage.component_entrypoints,
            "total_entrypoints": model.coverage.entrypoints,
            "composition_edges": model.coverage.composition_edges,
            "assembly_sites": model.coverage.assembly_sites,
            "subprocess_stages": model.coverage.subprocess_stages,
        },
        "trust": {
            "source_complete": source_incomplete == 0,
            "candle_catalog_matched": semantic_version_gaps == 0,
            "compiler_resolved": false,
            "runtime_evidence": model.runtime.is_some(),
            "unknown_or_heuristic_findings": unknown,
            "actionable_warnings": source_incomplete > 0
                || semantic_version_gaps > 0
                || dtype_risks > 0
                || dtype_conflicts > 0,
        },
        "finding_counts_by_rule": by_rule,
        "limitations": [
            "Rust names and types are source-resolved, not rustc DefIds",
            "macros and unresolved dynamic dispatch remain explicit Unknown evidence",
            "call-order pipeline/optimizer relationships require compiler-resolved value flow",
            "composition edges follow struct-field types with Heuristic confidence",
        ],
        "drill_down": [
            {"kind": "cargo"},
            {"kind": "findings"},
            {"kind": "composition"},
            {"kind": "assembly"},
            {"kind": "entrypoints"},
            {"kind": "modules"},
        ],
    })
}

fn runtime_query(model: &ModelIr) -> Value {
    json!({
        "id": "runtime",
        "summary": model.runtime,
        "gradient_finding_count": model.findings.iter()
            .filter(|finding| finding.rule.starts_with("runtime-"))
            .count(),
        "drill_down": [
            {"kind": "findings", "select": "runtime-"},
            {"kind": "tensors"},
        ],
    })
}

#[cfg(feature = "runtime")]
fn phase_graph(model: &ModelIr, phase: ExecutionPhase) -> Value {
    let tensors: Vec<Value> = model
        .tensors
        .iter()
        .filter(|tensor| tensor.execution_phase == Some(phase))
        .map(|tensor| {
            json!({
                "id": tensor.id,
                "name": tensor.name,
                "role": format!("{:?}", tensor.role),
                "requires_grad": tensor.requires_grad,
                "owner_function": tensor.owner_function,
            })
        })
        .collect();
    let operations: Vec<Value> = model
        .operations
        .iter()
        .filter(|operation| operation.execution_phase == Some(phase))
        .map(|operation| {
            json!({
                "id": operation.id,
                "name": operation.name,
                "function": operation.function,
                "inputs": operation.inputs,
                "output": operation.output,
                "avg_duration_ns": operation.avg_duration_ns,
                "timing_samples": operation.timing_samples,
            })
        })
        .collect();
    json!({
        "id": format!("graph-{}", phase.as_str()),
        "phase": phase,
        "tensor_count": tensors.len(),
        "operation_count": operations.len(),
        "tensors": tensors,
        "operations": operations,
        "drill_down": [
            {"kind": "tensors"},
            {"kind": "operations"},
            {"kind": "profile"},
        ],
    })
}

#[cfg(not(feature = "runtime"))]
fn phase_graph(_model: &ModelIr, _phase: ExecutionPhase) -> Value {
    unreachable!("runtime queries require the `runtime` crate feature")
}

#[cfg(feature = "runtime")]
fn profile_query(model: &ModelIr) -> Value {
    let mut slowest: Vec<Value> = model
        .operations
        .iter()
        .filter_map(|operation| {
            operation.avg_duration_ns.map(|duration| {
                json!({
                    "id": operation.id,
                    "name": operation.name,
                    "phase": operation.execution_phase,
                    "avg_duration_ns": duration,
                    "samples": operation.timing_samples,
                })
            })
        })
        .collect();
    slowest.sort_by(|left, right| {
        right["avg_duration_ns"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&left["avg_duration_ns"].as_u64().unwrap_or(0))
    });
    slowest.truncate(50);
    json!({
        "id": "profile",
        "runtime": model.runtime,
        "slowest_operations": slowest,
        "drill_down": [
            {"kind": "graph-train"},
            {"kind": "graph-infer"},
            {"kind": "runtime"},
        ],
    })
}

#[cfg(not(feature = "runtime"))]
fn profile_query(_model: &ModelIr) -> Value {
    unreachable!("runtime queries require the `runtime` crate feature")
}

fn model_improvement(model: &ModelIr) -> Value {
    use crate::model_ir::{Confidence, FindingSeverity};

    let proven_errors: Vec<Value> = model
        .findings
        .iter()
        .filter(|f| {
            matches!(f.severity, FindingSeverity::Error)
                && matches!(f.confidence, Confidence::Proven)
        })
        .map(finding_listing)
        .collect();

    let numeric_hazards: Vec<Value> = model
        .findings
        .iter()
        .filter(|f| {
            matches!(
                f.rule.as_str(),
                "numeric-domain-violation"
                    | "zero-times-infinity"
                    | "unstable-library-loss"
            ) && matches!(f.confidence, Confidence::Proven)
        })
        .map(finding_listing)
        .collect();

    let coverage_gaps: Vec<String> = model
        .findings
        .iter()
        .filter(|f| {
            matches!(f.confidence, Confidence::Unknown | Confidence::Heuristic)
                && !matches!(f.severity, FindingSeverity::Information)
        })
        .map(|f| format!("{}: {}", f.rule, f.message))
        .take(20)
        .collect();

    let gradient_gaps = model.runtime.as_ref().map(|rt| {
        json!({
            "missing": rt.missing_gradients,
            "zero": rt.zero_gradients,
            "non_finite": rt.non_finite_gradients,
            "first_non_finite_step": rt.first_non_finite_step,
            "saturating_activations": rt.saturating_activations,
            "value_observations": rt.value_observations,
        })
    });

    let mut suggested = vec![
        json!({"kind": "doctor"}),
        json!({"kind": "findings"}),
    ];
    if model.components.is_empty() {
        suggested.push(json!({"kind": "components"}));
    } else {
        for component in model.components.iter().take(3) {
            suggested.push(json!({
                "kind": "component",
                "select": component.qualified_name,
            }));
        }
    }
    if model.runtime.is_some() {
        suggested.push(json!({"kind": "runtime"}));
    }

    json!({
        "id": "model-improvement",
        "analysis_id": model.analysis_id,
        "trust": doctor(model).get("trust").cloned().unwrap_or(json!({})),
        "proven_errors": proven_errors,
        "proven_error_count": proven_errors.len(),
        "numeric_hazards": numeric_hazards,
        "gradient_gaps": gradient_gaps,
        "coverage_gaps": coverage_gaps,
        "components": model.components.iter().map(|c| &c.name).collect::<Vec<_>>(),
        "parameter_count": model.parameters.len(),
        "suggested_next_queries": suggested,
        "drill_down": [
            {"kind": "doctor"},
            {"kind": "findings"},
            {"kind": "model-improvement"},
        ],
    })
}

fn architecture(model: &ModelIr) -> Value {
    json!({
        "id": "architecture",
        "components": model.components.iter().map(|component| json!({
            "id": component.id,
            "name": component.name,
            "qualified_name": component.qualified_name,
            "source": component.source,
            "modules": component.modules.len(),
            "parameters": component.parameters.len(),
            "entrypoints": component.entrypoints.len(),
            "drill_down": [
                {"kind": "component", "select": component.qualified_name},
            ],
        })).collect::<Vec<_>>(),
        "edges": model.architecture_edges.iter().map(|edge| json!({
            "id": edge.id,
            "from": edge.from,
            "to": edge.to,
            "via_function": edge.via_function,
            "kind": if edge.id.0.starts_with("composition-edge:") {
                "composition"
            } else {
                "call_flow"
            },
        })).collect::<Vec<_>>(),
        "composition_edges": model.coverage.composition_edges,
        "stages": model.stages.iter().map(|stage| json!({
            "id": stage.id,
            "name": stage.name,
            "kind": stage.kind,
            "order": stage.order,
        })).collect::<Vec<_>>(),
        "artifacts": model.artifacts.iter().map(|artifact| json!({
            "id": artifact.id,
            "name": artifact.name,
        })).collect::<Vec<_>>(),
        "entrypoints": model.functions.iter()
            .filter(|function| function.is_entrypoint)
            .count(),
        "drill_down": [
            {"kind": "components"},
            {"kind": "composition"},
            {"kind": "modules"},
            {"kind": "functions"},
            {"kind": "tensors"},
            {"kind": "findings"},
        ],
    })
}

fn function_listing(function: &Function, is_component_entrypoint: bool) -> Value {
    json!({
        "id": function.id,
        "name": function.name,
        "qualified_name": function.qualified_name,
        "owner_type": function.owner_type,
        "visibility": function.visibility,
        "source": function.source,
        "is_entrypoint": function.is_entrypoint,
        "is_component_entrypoint": is_component_entrypoint,
        "is_loss": function.is_loss,
        "cfg_active": function.cfg_active,
        "calls": function.calls.len(),
        "tensor_inputs": function.tensor_inputs.len(),
        "tensor_outputs": function.tensor_outputs.len(),
        "drill_down": [
            {"kind": "function", "select": function.qualified_name},
            {"kind": "tensors", "select": function.qualified_name},
            {"kind": "operations", "select": function.id.0},
        ],
    })
}

fn function_detail(function: &Function) -> Value {
    let mut drill_down = vec![
        json!({"kind": "tensors", "select": function.qualified_name}),
        json!({"kind": "operations", "select": function.id.0}),
    ];
    if let Some(id) = function.tensor_inputs.first() {
        drill_down.push(json!({"kind": "tensor", "select": id.0}));
    } else if let Some(id) = function.tensor_outputs.first() {
        drill_down.push(json!({"kind": "tensor", "select": id.0}));
    }
    json!({
        "id": function.id,
        "name": function.name,
        "qualified_name": function.qualified_name,
        "owner_type": function.owner_type,
        "visibility": function.visibility,
        "parameters": function.parameters,
        "return_type": function.return_type,
        "cfg_predicates": function.cfg_predicates,
        "cfg_active": function.cfg_active,
        "source": function.source,
        "calls": function.calls,
        "tensor_inputs": function.tensor_inputs,
        "tensor_outputs": function.tensor_outputs,
        "is_entrypoint": function.is_entrypoint,
        "is_loss": function.is_loss,
        "drill_down": drill_down,
    })
}

fn tensor_listing(model: &ModelIr, tensor: &TensorContract) -> Value {
    let owner = owner_name(model, tensor);
    json!({
        "id": tensor.id,
        "name": tensor.name,
        "role": tensor.role,
        "owner_function": tensor.owner_function,
        "owner": owner,
        "dtype": tensor.dtype,
        "shape_rank": tensor.shape.rank,
        "requires_grad": tensor.requires_grad,
        "drill_down": [
            {"kind": "tensor", "select": tensor.id.0},
            {"kind": "function", "select": owner},
        ],
    })
}

fn finding_listing(finding: &Finding) -> Value {
    json!({
        "id": finding.id,
        "rule": finding.rule,
        "severity": finding.severity,
        "confidence": finding.confidence,
        "message": finding.message,
        "source": finding.source,
        "related": finding.related.len(),
        "drill_down": [
            {"kind": "findings", "select": finding.id.0},
        ],
    })
}

fn owner_name(model: &ModelIr, tensor: &TensorContract) -> String {
    model
        .functions
        .iter()
        .find(|function| function.id == tensor.owner_function)
        .map(|function| function.qualified_name.clone())
        .unwrap_or_default()
}

fn tensor_matches(model: &ModelIr, tensor: &TensorContract, selector: Option<&str>) -> bool {
    let owner = owner_name(model, tensor);
    optional_matches(
        selector,
        [
            tensor.id.0.as_str(),
            tensor.name.as_str(),
            tensor.owner_function.0.as_str(),
            owner.as_str(),
        ],
    )
}

fn finding_counts(model: &ModelIr) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for finding in &model.findings {
        *counts
            .entry(format!("{:?}", finding.severity).to_lowercase())
            .or_insert(0) += 1;
    }
    counts
}

fn required_selector(request: &QueryRequest) -> Result<&str> {
    request
        .selector
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{:?} query requires a selector", request.kind))
}

fn optional_matches<const N: usize>(selector: Option<&str>, values: [&str; N]) -> bool {
    selector.is_none_or(|selector| matches_text(selector, values))
}

fn matches_text<const N: usize>(selector: &str, values: [&str; N]) -> bool {
    let selector = selector.to_ascii_lowercase();
    values
        .iter()
        .any(|value| value.to_ascii_lowercase().contains(&selector))
}

fn stable_value_key(value: &Value) -> String {
    let key = value
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| value.get("name").and_then(Value::as_str))
        .or_else(|| value.get("key").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string();
    if let Some((prefix, suffix)) = key.rsplit_once(':') {
        if let Ok(sequence) = suffix.parse::<u64>() {
            return format!("{prefix}:{sequence:020}");
        }
    }
    key
}

fn resolve_id(model: &ModelIr, selector: &str) -> Result<StableId> {
    let mut hits = all_ids(model)
        .into_iter()
        .filter(|(id, labels)| {
            matches_text(
                selector,
                [
                    id.0.as_str(),
                    labels.first().map(String::as_str).unwrap_or_default(),
                ],
            )
        })
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    hits.sort();
    hits.dedup();
    match hits.as_slice() {
        [id] => Ok(id.clone()),
        [] => bail!("selector `{selector}` did not match any model object"),
        _ => bail!(
            "selector `{selector}` is ambiguous; matched {} objects",
            hits.len()
        ),
    }
}

fn all_ids(model: &ModelIr) -> Vec<(StableId, Vec<String>)> {
    let mut values = Vec::new();
    values.extend(
        model
            .components
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone(), v.qualified_name.clone()])),
    );
    values.extend(
        model
            .functions
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone(), v.qualified_name.clone()])),
    );
    values.extend(
        model
            .parameters
            .iter()
            .map(|v| (v.id.clone(), vec![v.key.clone()])),
    );
    values.extend(
        model
            .tensors
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone()])),
    );
    values.extend(
        model
            .operations
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone()])),
    );
    values.extend(
        model
            .stages
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone()])),
    );
    values.extend(
        model
            .artifacts
            .iter()
            .map(|v| (v.id.clone(), vec![v.name.clone(), v.path_expr.clone()])),
    );
    values
}

fn shortest_path(model: &ModelIr, from: &StableId, to: &StableId) -> Result<Vec<StableId>> {
    let mut adjacency: BTreeMap<StableId, BTreeSet<StableId>> = BTreeMap::new();
    for function in &model.functions {
        for callee in &function.calls {
            connect(&mut adjacency, &function.id, callee);
        }
        for input in &function.tensor_inputs {
            connect(&mut adjacency, input, &function.id);
        }
        for output in &function.tensor_outputs {
            connect(&mut adjacency, &function.id, output);
        }
    }
    for edge in &model.architecture_edges {
        connect(&mut adjacency, &edge.from, &edge.to);
    }
    for operation in &model.operations {
        for input in &operation.inputs {
            connect(&mut adjacency, input, &operation.id);
        }
        connect(&mut adjacency, &operation.id, &operation.output);
    }
    for parameter in &model.parameters {
        for use_id in &parameter.uses {
            connect(&mut adjacency, &parameter.id, use_id);
        }
        for optimizer in &parameter.optimizer_memberships {
            connect(&mut adjacency, optimizer, &parameter.id);
        }
    }
    for stage in &model.stages {
        for dependency in &stage.depends_on {
            connect(&mut adjacency, dependency, &stage.id);
        }
        connect(&mut adjacency, &stage.id, &stage.function);
        for artifact in &stage.consumes {
            connect(&mut adjacency, artifact, &stage.id);
        }
        for artifact in &stage.produces {
            connect(&mut adjacency, &stage.id, artifact);
        }
    }

    let mut queue = VecDeque::from([from.clone()]);
    let mut previous: BTreeMap<StableId, Option<StableId>> = BTreeMap::from([(from.clone(), None)]);
    while let Some(current) = queue.pop_front() {
        if current == *to {
            let mut path = Vec::new();
            let mut cursor = Some(current);
            while let Some(id) = cursor {
                cursor = previous.get(&id).cloned().flatten();
                path.push(id);
            }
            path.reverse();
            return Ok(path);
        }
        for next in adjacency.get(&current).into_iter().flatten() {
            if previous.contains_key(next) {
                continue;
            }
            previous.insert(next.clone(), Some(current.clone()));
            queue.push_back(next.clone());
        }
    }
    bail!("no path found from `{from}` to `{to}`")
}

fn connect(adjacency: &mut BTreeMap<StableId, BTreeSet<StableId>>, from: &StableId, to: &StableId) {
    adjacency
        .entry(from.clone())
        .or_default()
        .insert(to.clone());
}

fn describe_id(model: &ModelIr, id: &StableId) -> Value {
    if let Some(value) = model.components.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "component", "name": value.name, "source": value.source});
    }
    if let Some(value) = model.functions.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "function", "name": value.qualified_name, "source": value.source});
    }
    if let Some(value) = model.parameters.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "parameter", "name": value.key, "source": value.source});
    }
    if let Some(value) = model.tensors.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "tensor", "name": value.name});
    }
    if let Some(value) = model.operations.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "operation", "name": value.name, "source": value.source});
    }
    if let Some(value) = model.stages.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "stage", "name": value.name, "source": value.source});
    }
    if let Some(value) = model.artifacts.iter().find(|v| v.id == *id) {
        return json!({"id": id, "kind": "artifact", "name": value.name, "source": value.source});
    }
    json!({"id": id, "kind": "unknown"})
}

fn pct(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        ((part as f64 / total as f64) * 1000.0).round() / 10.0
    }
}

fn component_name_lookup(model: &ModelIr) -> BTreeMap<StableId, String> {
    model
        .components
        .iter()
        .map(|component| (component.id.clone(), component.name.clone()))
        .collect()
}

fn is_component_entrypoint(function: &Function, model: &ModelIr) -> bool {
    function.owner_type.as_ref().is_some_and(|owner| {
        model
            .components
            .iter()
            .any(|component| component.qualified_name == *owner || component.name == *owner)
    })
}

fn module_listing(
    module: &crate::model_ir::Module,
    component_names: &BTreeMap<StableId, String>,
) -> Value {
    let component_name = component_names
        .get(&module.component)
        .cloned()
        .unwrap_or_default();
    let mut drill_down = vec![json!({"kind": "parameters", "select": module.prefix})];
    if module.qualified_type.as_deref().is_some_and(|type_name| {
        component_names.values().any(|name| name == type_name)
            || type_name.contains("::")
                && component_names
                    .values()
                    .any(|name| type_name.ends_with(name))
    }) {
        drill_down.push(json!({
            "kind": "composition",
            "select": module.qualified_type.clone().unwrap_or(module.type_name.clone()),
        }));
    }
    json!({
        "id": module.id,
        "component": module.component,
        "component_name": component_name,
        "parent": module.parent,
        "type_name": module.type_name,
        "qualified_type": module.qualified_type,
        "field": module.field,
        "builder_root": module.builder_root,
        "prefix": module.prefix,
        "repeat": module.repeat,
        "source": module.source,
        "confidence": module.confidence,
        "drill_down": drill_down,
    })
}

fn assembly_listing(site: &crate::model_ir::AssemblySite) -> Value {
    json!({
        "id": site.id,
        "function_name": site.function_name,
        "component_name": site.component_name,
        "component": site.component,
        "builder_root": site.builder_root,
        "prefix_chain": site.prefix_chain,
        "varmap": site.varmap,
        "source_kind": site.source_kind,
        "role": site.role,
        "checkpoint_load": site.checkpoint_load,
        "source": site.source,
        "drill_down": [
            {"kind": "component", "select": site.component_name},
            {"kind": "parameters", "select": site.component_name},
            {"kind": "function", "select": site.function_name},
        ],
    })
}

fn composition_listing(model: &ModelIr, edge: &crate::model_ir::ArchitectureEdge) -> Value {
    let from = model
        .components
        .iter()
        .find(|component| component.id == edge.from);
    let to = model
        .components
        .iter()
        .find(|component| component.id == edge.to);
    let mut drill_down = Vec::new();
    if let Some(component) = from {
        drill_down.push(json!({"kind": "component", "select": component.qualified_name}));
        drill_down.push(json!({"kind": "modules", "select": component.name}));
    }
    if let Some(component) = to {
        drill_down.push(json!({"kind": "component", "select": component.qualified_name}));
    }
    json!({
        "id": edge.id,
        "from": from.map(|component| json!({
            "id": component.id,
            "name": component.name,
            "qualified_name": component.qualified_name,
        })).unwrap_or_else(|| json!({"id": edge.from})),
        "to": to.map(|component| json!({
            "id": component.id,
            "name": component.name,
            "qualified_name": component.qualified_name,
        })).unwrap_or_else(|| json!({"id": edge.to})),
        "via_function": edge.via_function,
        "source": edge.source,
        "confidence": edge.evidence.first().map(|evidence| &evidence.confidence),
        "detail": edge.evidence.first().map(|evidence| &evidence.detail),
        "drill_down": drill_down,
    })
}

fn composition_matches(
    model: &ModelIr,
    selector: Option<&str>,
    from: StableId,
    to: StableId,
) -> bool {
    let Some(selector) = selector else {
        return true;
    };
    let from_labels = component_labels(model, &from);
    let to_labels = component_labels(model, &to);
    matches_text(selector, from_labels) || matches_text(selector, to_labels)
}

fn component_labels<'a>(model: &'a ModelIr, id: &StableId) -> [&'a str; 3] {
    if let Some(component) = model
        .components
        .iter()
        .find(|component| component.id == *id)
    {
        [
            component.id.0.as_str(),
            component.name.as_str(),
            component.qualified_name.as_str(),
        ]
    } else {
        ["", "", ""]
    }
}
