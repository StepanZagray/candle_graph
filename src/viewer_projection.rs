//! Project [`ModelIr`] into `candle-graph/viewer/1` for the interactive HTML visualizer.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::dtype_propagate::tensor_dtype_is_proven_for_display;

use crate::model_ir::{
    FindingSeverity, ModelIr, Module, Operation, Parameter, ParameterRole, PipelineStage, StableId,
    TensorContract, TensorRole,
};
use crate::phase::ExecutionPhase;

pub const VIEWER_SCHEMA: &str = "candle-graph/viewer/1";

/// Build the viewer payload consumed by [`crate::viewer::render_html`].
pub fn project(model: &ModelIr) -> Value {
    let architecture = architecture_view(model);
    let dataflow_train = dataflow_view(model, ExecutionPhase::Train);
    let dataflow_infer = dataflow_view(model, ExecutionPhase::Infer);
    let pipeline = pipeline_view(model);
    let findings = findings_view(model);
    json!({
        "schema": VIEWER_SCHEMA,
        "analysis_id": model.analysis_id,
        "coverage": model.coverage,
        "package": model.cargo.as_ref().map(|cargo| cargo.package_name.clone()),
        "runtime": model.runtime,
        "summary": summary_view(model, &architecture, &dataflow_train, &dataflow_infer, &pipeline, &findings),
        "agent_context": agent_context_view(),
        "views": {
            "structure": structure_view(model),
            "architecture": architecture,
            "dataflow_train": dataflow_train,
            "dataflow_infer": dataflow_infer,
            "pipeline": pipeline,
            "findings": findings,
        },
        "diagnostics": diagnostics_view(model),
    })
}

fn summary_view(
    model: &ModelIr,
    architecture: &Value,
    dataflow_train: &Value,
    dataflow_infer: &Value,
    pipeline: &Value,
    _findings: &Value,
) -> Value {
    let entrypoints: Vec<String> = model
        .functions
        .iter()
        .filter(|f| f.is_entrypoint)
        .map(|f| f.qualified_name.clone())
        .collect();
    let trainable = model
        .parameters
        .iter()
        .filter(|p| matches!(p.role, ParameterRole::Optimized))
        .count();
    let mut severity_counts: BTreeMap<String, usize> = BTreeMap::new();
    for finding in &model.findings {
        let key = format!("{:?}", finding.severity);
        *severity_counts.entry(key).or_default() += 1;
    }
    json!({
        "model_name": model.cargo.as_ref().map(|c| c.package_name.clone()),
        "entrypoints": entrypoints,
        "trainable_parameters": trainable,
        "total_parameters": model.parameters.len(),
        "findings_by_severity": severity_counts,
        "views": {
            "architecture": view_stats(architecture, "Module hierarchy and composition edges"),
            "dataflow_train": view_stats(dataflow_train, "Training-phase tensor/operation dataflow"),
            "dataflow_infer": view_stats(dataflow_infer, "Inference-phase tensor/operation dataflow"),
            "pipeline": view_stats(pipeline, "Training pipeline stages and dependencies"),
        },
    })
}

fn view_stats(view: &Value, description: &str) -> Value {
    json!({
        "nodes": view["nodes"].as_array().map(|a| a.len()).unwrap_or(0),
        "edges": view["edges"].as_array().map(|a| a.len()).unwrap_or(0),
        "functions": view["functions"].as_array().map(|a| a.len()),
        "description": description,
    })
}

fn agent_context_view() -> Value {
    json!({
        "purpose": "Static analysis of a candle-rs model: module structure, gradient dataflow, and pipeline stages.",
        "schema": VIEWER_SCHEMA,
        "views": {
            "architecture": "Hierarchical module/component graph. Use for understanding model structure.",
            "dataflow_train": "Layered DAG of tensors and operations during training. Filter by function for detail.",
            "dataflow_infer": "Same as dataflow_train but for inference phase.",
            "pipeline": "Training pipeline stage ordering and dependencies.",
            "findings": "Static analysis findings with related node IDs for cross-referencing.",
        },
        "grad_states": {
            "Trainable": "Parameter receives gradients during training",
            "Frozen": "Parameter excluded from gradient updates",
            "Differentiable": "Tensor participates in autograd",
            "Severed": "Gradient flow is cut (e.g. detach, stop_gradient)",
            "LayoutDependent": "Gradient depends on tensor layout/strides",
            "Unknown": "Could not determine gradient state statically",
        },
        "node_kinds": {
            "component": "Top-level model component",
            "module": "Sub-module within a component",
            "operation": "Tensor operation (matmul, relu, etc.)",
            "parameter": "Learnable or frozen model parameter",
            "tensor": "Intermediate or input tensor",
            "stage": "Pipeline stage (train, eval, checkpoint, etc.)",
        },
    })
}

fn structure_view(model: &ModelIr) -> Value {
    let component_names = component_name_lookup(model);
    let modules: Vec<Value> = model
        .modules
        .iter()
        .map(|module| {
            json!({
                "id": module.id,
                "component": component_names.get(&module.component).cloned().unwrap_or_default(),
                "parent": module.parent,
                "type_name": module.type_name,
                "field": module.field,
                "builder_root": module.builder_root,
                "prefix": module.prefix,
                "repeat": module.repeat,
                "source": module.source,
                "confidence": format!("{:?}", module.confidence),
            })
        })
        .collect();
    let parameters: Vec<Value> = model
        .parameters
        .iter()
        .map(|parameter| {
            let mut node = parameter_node(model, parameter, None);
            if let Some(obj) = node.as_object_mut() {
                obj.insert("module_id".into(), json!(parameter.module));
            }
            node
        })
        .collect();
    json!({
        "modules": modules,
        "parameters": parameters,
    })
}

fn architecture_view(model: &ModelIr) -> Value {
    let names = component_name_lookup(model);
    let param_counts: HashMap<&StableId, usize> = model
        .parameters
        .iter()
        .map(|parameter| &parameter.module)
        .fold(HashMap::new(), |mut counts, module| {
            *counts.entry(module).or_default() += 1;
            counts
        });

    let mut nodes: Vec<Value> = model
        .components
        .iter()
        .map(|component| {
            json!({
                "id": component.id,
                "label": component.name,
                "short_label": component.name,
                "qualified_name": component.qualified_name,
                "source": component.source,
                "kind": "component",
                "modules": component.modules.len(),
                "parameters": component.parameters.len(),
            })
        })
        .collect();

    for module in &model.modules {
        let label = module_label(module);
        let params = param_counts.get(&module.id).copied().unwrap_or(0);
        nodes.push(json!({
            "id": module.id,
            "label": label,
            "short_label": module.type_name.rsplit("::").next().unwrap_or(&module.type_name),
            "qualified_name": module.type_name,
            "type_name": module.type_name,
            "field": module.field,
            "source": module.source,
            "kind": "module",
            "builder_root": module.builder_root,
            "parameters": params,
            "confidence": format!("{:?}", module.confidence),
        }));
    }

    let mut edges: Vec<Value> = model
        .architecture_edges
        .iter()
        .map(|edge| {
            let kind = if edge.id.0.starts_with("composition-edge:") {
                "composition"
            } else {
                "call_flow"
            };
            json!({
                "id": edge.id,
                "from": edge.from,
                "to": edge.to,
                "label": format!(
                    "{} → {}",
                    names.get(&edge.from).cloned().unwrap_or_else(|| edge.from.0.clone()),
                    names.get(&edge.to).cloned().unwrap_or_else(|| edge.to.0.clone())
                ),
                "kind": kind,
                "via_function": edge.via_function,
                "source": edge.source,
                "confidence": edge
                    .evidence
                    .first()
                    .map(|evidence| format!("{:?}", evidence.confidence))
                    .unwrap_or_else(|| "Unknown".to_string()),
            })
        })
        .collect();

    for module in &model.modules {
        if let Some(parent) = &module.parent {
            edges.push(json!({
                "id": format!("module-edge:{}:{}", parent.0, module.id.0),
                "from": parent,
                "to": module.id,
                "label": module.field.clone().unwrap_or_else(|| "child".into()),
                "kind": "composition",
                "source": module.source,
            }));
        } else if model.components.len() == 1 {
            let component = &model.components[0];
            edges.push(json!({
                "id": format!("component-edge:{}:{}", component.id.0, module.id.0),
                "from": component.id,
                "to": module.id,
                "label": "root",
                "kind": "composition",
                "source": module.source,
            }));
        }
    }

    json!({ "nodes": nodes, "edges": edges })
}

fn dataflow_view(model: &ModelIr, phase: ExecutionPhase) -> Value {
    let tensors: Vec<&TensorContract> = model
        .tensors
        .iter()
        .filter(|tensor| tensor.execution_phase == Some(phase))
        .collect();
    let operations: Vec<&Operation> = model
        .operations
        .iter()
        .filter(|operation| operation.execution_phase == Some(phase))
        .collect();
    if tensors.is_empty() && operations.is_empty() {
        return json!({ "phase": phase, "functions": [], "nodes": [], "edges": [] });
    }

    let param_by_tensor: HashMap<&StableId, &Parameter> = model
        .parameters
        .iter()
        .flat_map(|parameter| {
            model
                .tensors
                .iter()
                .filter(|tensor| tensor.parameter.as_ref() == Some(&parameter.id))
                .map(move |tensor| (&tensor.id, parameter))
        })
        .collect();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for tensor in &tensors {
        let parameter = param_by_tensor.get(&tensor.id).copied();
        let mut node = tensor_node(tensor, parameter);
        if let Some(obj) = node.as_object_mut() {
            obj.insert("function".into(), json!(tensor.owner_function));
        }
        nodes.push(node);
    }

    for operation in &operations {
        let op_id = operation.id.0.clone();
        let mut op_node = json!({
            "id": operation.id,
            "label": operation.name,
            "short_label": operation.name,
            "qualified_name": operation.qualified_name,
            "kind": "operation",
            "function": operation.function,
            "source": operation.source,
            "dtype_rule": operation.dtype_rule,
            "gradient_rule": operation.gradient_rule,
            "timing": operation.timing,
        });
        if let Some(timing) = operation.timing {
            op_node["avg_duration_ms"] = json!((timing.avg_ns as f64) / 1_000_000.0);
        }
        nodes.push(op_node);

        let grad = edge_grad_state(&operation.gradient_rule);
        let edge_kind = if grad == "Severed" {
            "severing"
        } else {
            "data"
        };

        for (index, input) in operation.inputs.iter().enumerate() {
            edges.push(json!({
                "id": format!("{op_id}:in:{index}"),
                "from": input,
                "to": operation.id,
                "kind": edge_kind,
                "grad_state": grad,
                "label": "in",
            }));
        }
        edges.push(json!({
            "id": format!("{op_id}:out"),
            "from": operation.id,
            "to": operation.output,
            "kind": edge_kind,
            "grad_state": grad,
            "label": "out",
        }));
    }

    json!({
        "phase": phase,
        "functions": dataflow_functions(model, phase),
        "nodes": nodes,
        "edges": edges,
    })
}

fn dataflow_functions(model: &ModelIr, phase: ExecutionPhase) -> Vec<Value> {
    let mut op_counts: HashMap<&StableId, usize> = HashMap::new();
    let mut tensor_counts: HashMap<&StableId, usize> = HashMap::new();
    for operation in &model.operations {
        if operation.execution_phase == Some(phase) {
            *op_counts.entry(&operation.function).or_default() += 1;
        }
    }
    for tensor in &model.tensors {
        if tensor.execution_phase == Some(phase) {
            *tensor_counts.entry(&tensor.owner_function).or_default() += 1;
        }
    }

    let mut functions: Vec<Value> = model
        .functions
        .iter()
        .filter_map(|function| {
            let ops = op_counts.get(&function.id).copied().unwrap_or(0);
            let tensors = tensor_counts.get(&function.id).copied().unwrap_or(0);
            if ops == 0 && tensors == 0 {
                return None;
            }
            Some(json!({
                "id": function.id,
                "label": function.qualified_name,
                "short_label": function_short_label(&function.qualified_name),
                "is_entrypoint": function.is_entrypoint,
                "operations": ops,
                "tensors": tensors,
            }))
        })
        .collect();
    functions.sort_by(|a, b| {
        b["operations"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["operations"].as_u64().unwrap_or(0))
            .then_with(|| {
                a["label"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["label"].as_str().unwrap_or(""))
            })
    });
    functions
}

fn module_label(module: &Module) -> String {
    let short = module
        .type_name
        .rsplit("::")
        .next()
        .unwrap_or(module.type_name.as_str());
    match &module.field {
        Some(field) => format!("{field}: {short}"),
        None => short.to_string(),
    }
}

fn function_short_label(qualified_name: &str) -> String {
    qualified_name
        .rsplit("::")
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("::")
}

fn tensor_short_label(name: &str) -> String {
    let mut label = name.strip_prefix("result_of_").unwrap_or(name);
    if let Some(base) = label.strip_suffix("::forward") {
        label = base;
    }
    label
        .rsplit('.')
        .next()
        .unwrap_or(label)
        .replace('_', " ")
}

fn pipeline_view(model: &ModelIr) -> Value {
    let nodes: Vec<Value> = model.stages.iter().map(stage_node).collect();
    let mut edges = Vec::new();
    for stage in &model.stages {
        for dependency in &stage.depends_on {
            edges.push(json!({
                "id": format!("{}:depends:{}", stage.id.0, dependency.0),
                "from": dependency,
                "to": stage.id,
                "kind": "depends_on",
                "label": "depends",
            }));
        }
        if let Some(order) = stage.order {
            for other in &model.stages {
                if other.order == Some(order + 1) {
                    edges.push(json!({
                        "id": format!("{}:order:{}", stage.id.0, other.id.0),
                        "from": stage.id,
                        "to": other.id,
                        "kind": "sequence",
                        "label": "next",
                    }));
                }
            }
        }
    }
    json!({ "nodes": nodes, "edges": edges })
}

fn suggest_view_for_finding(finding: &crate::model_ir::Finding) -> &'static str {
    let rule = finding.rule.to_lowercase();
    if rule.contains("pipeline") || rule.contains("stage") {
        "pipeline"
    } else if rule.contains("grad") || rule.contains("tensor") || rule.contains("operation") {
        "dataflow_train"
    } else {
        "architecture"
    }
}

fn findings_view(model: &ModelIr) -> Value {
    let items: Vec<Value> = model
        .findings
        .iter()
        .map(|finding| {
            json!({
                "id": finding.id,
                "rule": finding.rule,
                "severity": format!("{:?}", finding.severity),
                "confidence": format!("{:?}", finding.confidence),
                "message": finding.message,
                "source": finding.source,
                "related": finding.related,
                "suggested_view": suggest_view_for_finding(finding),
            })
        })
        .collect();
    json!({ "items": items })
}

fn diagnostics_view(model: &ModelIr) -> Value {
    let items: Vec<Value> = model
        .findings
        .iter()
        .filter(|finding| {
            matches!(
                finding.severity,
                FindingSeverity::Error | FindingSeverity::Warning
            )
        })
        .map(|finding| {
            json!({
                "at": finding.source,
                "message": finding.message,
                "rule": finding.rule,
                "severity": format!("{:?}", finding.severity),
            })
        })
        .collect();
    json!(items)
}

fn component_name_lookup(model: &ModelIr) -> BTreeMap<StableId, String> {
    model
        .components
        .iter()
        .map(|component| (component.id.clone(), component.name.clone()))
        .collect()
}

fn parameter_node(
    model: &ModelIr,
    parameter: &Parameter,
    tensor: Option<&TensorContract>,
) -> Value {
    let module = model
        .modules
        .iter()
        .find(|module| module.id == parameter.module)
        .map(|module| module.prefix.clone())
        .unwrap_or_default();
    json!({
        "id": parameter.id,
        "label": parameter.key,
        "short_label": parameter.key.rsplit('.').next().unwrap_or(&parameter.key),
        "key": parameter.key,
        "kind": "parameter",
        "builder_root": parameter.builder_root,
        "role": format!("{:?}", parameter.role),
        "param_kind": parameter.kind,
        "symbolic_shape": parameter.symbolic_shape,
        "module_prefix": module,
        "source": parameter.source,
        "grad_state": parameter_grad_state(parameter, tensor),
        "dtype": parameter_dtype(parameter, tensor),
        "shape": shape_label(tensor),
    })
}

fn parameter_dtype(parameter: &Parameter, tensor: Option<&TensorContract>) -> Value {
    if let Some(tensor) = tensor {
        if tensor_dtype_is_proven_for_display(tensor) {
            return json!(tensor.dtype);
        }
    }
    parameter
        .checkpoint_dtype
        .as_ref()
        .map(|dtype| json!(dtype))
        .unwrap_or(Value::Null)
}

fn viewer_dtype(tensor: Option<&TensorContract>) -> Value {
    match tensor {
        Some(tensor) if tensor_dtype_is_proven_for_display(tensor) => json!(tensor.dtype),
        _ => Value::Null,
    }
}

fn tensor_node(tensor: &TensorContract, parameter: Option<&Parameter>) -> Value {
    json!({
        "id": tensor.id,
        "label": tensor.name,
        "short_label": tensor_short_label(&tensor.name),
        "kind": if parameter.is_some() { "parameter" } else { "tensor" },
        "role": format!("{:?}", tensor.role),
        "dtype": viewer_dtype(Some(tensor)),
        "shape": shape_label(Some(tensor)),
        "builder_root": parameter.map(|p| p.builder_root.as_str()),
        "source": tensor.owner_function,
        "grad_state": tensor_grad_state(tensor, parameter),
        "requires_grad": tensor.requires_grad,
        "repeat_group": repeat_group(tensor, parameter),
    })
}

fn stage_node(stage: &PipelineStage) -> Value {
    json!({
        "id": stage.id,
        "label": stage.name,
        "kind": "stage",
        "stage_kind": format!("{:?}", stage.kind),
        "order": stage.order,
        "source": stage.source,
        "dispatch": format!("{:?}", stage.dispatch),
        "subprocess_key": stage.subprocess_key,
    })
}

fn shape_label(tensor: Option<&TensorContract>) -> Value {
    match tensor {
        Some(tensor) if !tensor.shape.dimensions.is_empty() => {
            json!(tensor
                .shape
                .dimensions
                .iter()
                .map(|d| d.expr.as_str())
                .collect::<Vec<_>>())
        }
        Some(tensor) if tensor.shape.rank.is_some() => {
            json!(format!("rank {}", tensor.shape.rank.unwrap_or(0)))
        }
        Some(tensor) => tensor
            .shape
            .source_expr
            .as_ref()
            .map(|expr| json!(expr))
            .unwrap_or(json!("—")),
        None => json!("—"),
    }
}

fn parameter_grad_state(parameter: &Parameter, tensor: Option<&TensorContract>) -> &'static str {
    match parameter.role {
        ParameterRole::Frozen | ParameterRole::Excluded => return "Frozen",
        ParameterRole::RunningState => return "LayoutDependent",
        ParameterRole::Optimized => {}
        ParameterRole::Conditional | ParameterRole::Unknown => {}
    }
    tensor
        .and_then(|tensor| tensor.requires_grad)
        .map(|requires| if requires { "Trainable" } else { "Frozen" })
        .unwrap_or("Unknown")
}

fn tensor_grad_state(tensor: &TensorContract, parameter: Option<&Parameter>) -> &'static str {
    if let Some(parameter) = parameter {
        return parameter_grad_state(parameter, Some(tensor));
    }
    if tensor.role == TensorRole::Loss {
        return "Differentiable";
    }
    match tensor.requires_grad {
        Some(true) => "Trainable",
        Some(false) => "Frozen",
        None => "Unknown",
    }
}

fn edge_grad_state(gradient_rule: &str) -> &'static str {
    if gradient_rule.contains("Severs") {
        "Severed"
    } else if gradient_rule.contains("LayoutDependent") {
        "LayoutDependent"
    } else if gradient_rule.contains("Propagates") {
        "Differentiable"
    } else {
        "Unknown"
    }
}

fn repeat_group(tensor: &TensorContract, parameter: Option<&Parameter>) -> Option<String> {
    if let Some(parameter) = parameter {
        if parameter.key.contains("{") || parameter.key.contains("index") {
            return Some(parameter.key.split('.').next()?.to_string());
        }
    }
    if tensor.name.contains("block") || tensor.name.contains("layer") {
        return Some("layers".to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ir::{
        Component, Confidence, DeviceFact, Evidence, EvidenceKind, Function, LayoutFact,
        ModelCoverage, Module, Parameter, ShapeFact, StableId,
    };

    fn sample_model() -> ModelIr {
        let component_id = StableId::new("component", ["Root"]);
        let module_id = StableId::new("module", ["Root", ""]);
        let param_id = StableId::new("parameter", ["Root", "head.weight"]);
        let tensor_id = StableId::new("tensor", ["train", "Root::forward", "0"]);
        let op_id = StableId::new("operation", ["train", "Root::forward", "1"]);
        let function_id = StableId::new("function", ["Root::forward"]);
        ModelIr {
            schema: crate::model_ir::MODEL_IR_SCHEMA.to_string(),
            analysis_id: StableId::new("analysis", ["test"]),
            cargo: None,
            coverage: ModelCoverage {
                components: 1,
                modules: 1,
                parameters: 1,
                tensors: 1,
                operations: 1,
                ..ModelCoverage::default()
            },
            components: vec![Component {
                id: component_id.clone(),
                name: "Root".into(),
                qualified_name: "Root".into(),
                source: "model.rs:1".into(),
                constructor: StableId::new("function", ["Root::new"]),
                builders: vec![],
                modules: vec![module_id.clone()],
                parameters: vec![param_id.clone()],
                entrypoints: vec![function_id.clone()],
                evidence: vec![],
            }],
            architecture_edges: vec![],
            modules: vec![Module {
                id: module_id,
                component: component_id,
                parent: None,
                type_name: "Root".into(),
                qualified_type: None,
                field: None,
                builder_root: "vb".into(),
                prefix: String::new(),
                repeat: None,
                source: "model.rs:1".into(),
                confidence: Confidence::Proven,
            }],
            parameters: vec![Parameter {
                id: param_id.clone(),
                component: StableId::new("component", ["Root"]),
                module: StableId::new("module", ["Root", ""]),
                key: "head.weight".into(),
                builder_root: "vb".into(),
                role: ParameterRole::Optimized,
                kind: "linear".into(),
                symbolic_shape: Some("[8, 8]".into()),
                checkpoint_shape: None,
                checkpoint_dtype: None,
                source: "model.rs:5".into(),
                uses: vec![op_id.clone()],
                optimizer_memberships: vec![],
                evidence: vec![],
            }],
            functions: vec![Function {
                id: function_id,
                name: "forward".into(),
                qualified_name: "Root::forward".into(),
                owner_type: Some("Root".into()),
                visibility: crate::model_ir::Visibility::Public,
                parameters: vec![],
                return_type: None,
                cfg_predicates: vec![],
                cfg_active: None,
                source: "model.rs:10".into(),
                calls: vec![],
                tensor_inputs: vec![tensor_id.clone()],
                tensor_outputs: vec![],
                is_entrypoint: true,
                is_loss: false,
                execution_phases: vec![ExecutionPhase::Train],
            }],
            tensors: vec![TensorContract {
                id: tensor_id.clone(),
                name: "head.weight".into(),
                role: TensorRole::Parameter,
                owner_function: StableId::new("function", ["Root::forward"]),
                parameter: Some(param_id),
                shape: ShapeFact {
                    rank: Some(2),
                    dimensions: vec![],
                    source_expr: Some("[8, 8]".into()),
                },
                dtype: "F32".into(),
                device: DeviceFact::Unknown,
                layout: LayoutFact::Unknown,
                requires_grad: Some(true),
                execution_phase: Some(ExecutionPhase::Train),
                evidence: vec![Evidence {
                    kind: EvidenceKind::Source,
                    confidence: Confidence::Proven,
                    source: Some("model.rs:5".into()),
                    detail: "test".into(),
                }],
            }],
            operations: vec![Operation {
                id: op_id,
                function: StableId::new("function", ["Root::forward"]),
                name: "matmul".into(),
                qualified_name: None,
                inputs: vec![tensor_id],
                output: StableId::new("tensor", ["train", "Root::forward", "2"]),
                source: "model.rs:11".into(),
                dtype_rule: "preserve".into(),
                gradient_rule: "Propagates".into(),
                device_rule: "preserve".into(),
                shape_rule: "unknown".into(),
                domain_rule: String::new(),
                execution_phase: Some(ExecutionPhase::Train),
                timing: None,
                evidence: vec![],
            }],
            stages: vec![],
            artifacts: vec![],
            optimizers: vec![],
            assembly_sites: vec![],
            findings: vec![],
            runtime: None,
        }
    }

    #[test]
    fn project_emits_viewer_schema_and_dataflow_edges() {
        let payload = project(&sample_model());
        assert_eq!(payload["schema"], VIEWER_SCHEMA);
        let train = &payload["views"]["dataflow_train"];
        assert_eq!(train["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(train["edges"].as_array().unwrap().len(), 2);
        assert_eq!(
            payload["views"]["structure"]["modules"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }
}
