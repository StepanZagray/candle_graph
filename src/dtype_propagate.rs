//! Fixpoint dtype propagation over the unified model IR operation graph.
//!
//! Only seeds from facts already present in the IR (dataflow-resolved dtypes, checkpoint
//! parameter dtypes) and walks catalog `Preserve` / `SameAsInputs` / `Fixed` edges.
//! Does not invent dtypes for unconnected tensors.

use std::collections::HashMap;

use crate::model_ir::{Confidence, Evidence, EvidenceKind, ModelIr, StableId};

/// Infer `builder_root -> dtype` from `VarBuilder::from_*` sites in the analyzed crate.
pub fn infer_builder_default_dtypes(krate: &crate::load::Crate) -> HashMap<String, String> {
    use syn::visit::Visit;
    struct Visitor {
        dtypes: HashMap<String, String>,
    }
    impl<'ast> Visit<'ast> for Visitor {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            let Some(init) = local.init.as_ref() else {
                return;
            };
            let Some(binding) = pat_ident(&local.pat) else {
                return;
            };
            if let Some(dtype) = varbuilder_default_dtype(&init.expr) {
                self.dtypes.insert(binding, dtype);
            }
        }
    }
    let mut visitor = Visitor {
        dtypes: HashMap::new(),
    };
    for func in krate.all_functions().chain(krate.all_methods()) {
        visitor.visit_block(&func.block);
    }
    visitor.dtypes
}

/// Whether a tensor's dtype is backed by analysis evidence (may include conditional propagation).
pub fn tensor_dtype_is_evidence_backed(tensor: &crate::model_ir::TensorContract) -> bool {
    if tensor.dtype == "Unknown" {
        return false;
    }
    tensor.evidence.iter().any(dtype_evidence_is_backing)
}

/// Whether a tensor's dtype is safe to show in the visualizer (proven facts only).
pub fn tensor_dtype_is_proven_for_display(tensor: &crate::model_ir::TensorContract) -> bool {
    if tensor.dtype == "Unknown" {
        return false;
    }
    tensor
        .evidence
        .iter()
        .any(|evidence| matches!(evidence.confidence, Confidence::Proven) && dtype_evidence_is_backing(evidence))
}

fn dtype_evidence_is_backing(evidence: &Evidence) -> bool {
    if evidence.detail.contains("model-wide homogeneous") {
        return false;
    }
    matches!(evidence.confidence, Confidence::Proven | Confidence::Conditional)
        && (evidence.detail.contains("static dtype")
            || evidence.detail.contains("checkpoint")
            || evidence.detail.contains("via operation")
            || evidence.detail.contains("agreed runtime observation")
            || evidence.detail.contains("Tensor::")
            || evidence.detail.contains("to_dtype")
            || evidence.detail.contains("DType::"))
}

/// Propagate dtypes through operations until fixpoint, then write back to tensors.
pub fn propagate_tensor_dtypes(model: &mut ModelIr, _builder_dtypes: &HashMap<String, String>) {
    let mut known: HashMap<StableId, String> = model
        .tensors
        .iter()
        .filter(|tensor| tensor_dtype_is_evidence_backed(tensor))
        .map(|tensor| (tensor.id.clone(), tensor.dtype.clone()))
        .collect();

    seed_from_checkpoint_parameters(model, &mut known);

    loop {
        let mut changed = false;
        for op in &model.operations {
            let Some(output) = resolve_output_dtype(op, &known) else {
                continue;
            };
            if known.get(&op.output) != Some(&output) {
                known.insert(op.output.clone(), output);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for tensor in &mut model.tensors {
        if tensor.dtype != "Unknown" {
            continue;
        }
        let Some(dtype) = known.get(&tensor.id) else {
            continue;
        };
        let via_op = model.operations.iter().find(|op| op.output == tensor.id);
        tensor.dtype = dtype.clone();
        tensor.evidence.push(Evidence {
            kind: EvidenceKind::Source,
            confidence: Confidence::Conditional,
            source: via_op.map(|op| op.source.clone()),
            detail: format!(
                "dtype {dtype} via operation `{}` from evidenced upstream dtype",
                via_op.map(|op| op.name.as_str()).unwrap_or("?")
            ),
        });
    }
}

fn seed_from_checkpoint_parameters(
    model: &ModelIr,
    known: &mut HashMap<StableId, String>,
) {
    for parameter in &model.parameters {
        let Some(dtype) = parameter.checkpoint_dtype.as_ref() else {
            continue;
        };
        for operation_id in &parameter.uses {
            let Some(operation) = model
                .operations
                .iter()
                .find(|operation| &operation.id == operation_id)
            else {
                continue;
            };
            known
                .entry(operation.output.clone())
                .or_insert_with(|| dtype.clone());
        }
    }
}

fn resolve_output_dtype(
    op: &crate::model_ir::Operation,
    known: &HashMap<StableId, String>,
) -> Option<String> {
    if let Some(rule) = op.dtype_rule.strip_prefix("Fixed(").and_then(|rest| rest.strip_suffix(')'))
    {
        return Some(normalize_dtype_label(rule));
    }
    match op.dtype_rule.as_str() {
        "Preserve" => op
            .inputs
            .iter()
            .find_map(|input| known.get(input))
            .cloned()
            .map(|dtype| normalize_dtype_label(&dtype)),
        "SameAsInputs" => {
            let dtypes: Vec<&String> = op.inputs.iter().filter_map(|input| known.get(input)).collect();
            if dtypes.is_empty() || dtypes.len() != op.inputs.len() {
                return None;
            }
            let first = normalize_dtype_label(dtypes[0]);
            if dtypes
                .iter()
                .all(|dtype| normalize_dtype_label(dtype) == first)
            {
                Some(first)
            } else {
                None
            }
        }
        "Explicit" => known.get(&op.output).cloned(),
        _ => None,
    }
}

fn normalize_dtype_label(dtype: &str) -> String {
    match dtype.trim() {
        "BF16" | "Bf16" | "bf16" => "BF16".into(),
        "F32" | "f32" => "F32".into(),
        "F64" | "f64" => "F64".into(),
        "F16" | "f16" => "F16".into(),
        other if !other.is_empty() && !other.eq_ignore_ascii_case("unknown") => {
            other.to_ascii_uppercase()
        }
        _ => "Unknown".into(),
    }
}

fn varbuilder_default_dtype(expression: &syn::Expr) -> Option<String> {
    let syn::Expr::Call(call) = expression else {
        return None;
    };
    let leaf = call_leaf_name(&call.func)?;
    if !matches!(
        leaf.as_str(),
        "from_varmap" | "from_mmaped_safetensors" | "from_buffered_safetensors"
    ) {
        return None;
    }
    call.args
        .get(1)
        .and_then(dtype_from_syn_expr)
        .map(|dtype| normalize_dtype_label(&dtype.to_string()))
        .filter(|dtype| dtype != "Unknown")
}

fn dtype_from_syn_expr(expression: &syn::Expr) -> Option<crate::op_semantics::AbstractDtype> {
    let expression = strip_expr(expression);
    if let syn::Expr::Path(path) = expression {
        let segments: Vec<_> = path.path.segments.iter().collect();
        if segments.len() >= 2 && segments[segments.len() - 2].ident == "DType" {
            return Some(crate::op_semantics::AbstractDtype::parse(
                &segments[segments.len() - 1].ident.to_string(),
            ));
        }
    }
    None
}

fn call_leaf_name(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        syn::Expr::Field(field) => call_leaf_name(&field.base),
        syn::Expr::Call(call) => call_leaf_name(&call.func),
        _ => None,
    }
}

fn strip_expr(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Group(group) => strip_expr(&group.expr),
        syn::Expr::Paren(paren) => strip_expr(&paren.expr),
        syn::Expr::Reference(reference) => strip_expr(&reference.expr),
        syn::Expr::Try(value) => strip_expr(&value.expr),
        other => other,
    }
}

fn pat_ident(pat: &syn::Pat) -> Option<String> {
    match pat {
        syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
        syn::Pat::Type(pat) => pat_ident(&pat.pat),
        syn::Pat::Reference(pat) => pat_ident(&pat.pat),
        syn::Pat::TupleStruct(pat) if pat.elems.len() == 1 => pat_ident(&pat.elems[0]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ir::{
        DeviceFact, LayoutFact, Operation, ParameterRole, ShapeFact, StableId, TensorContract,
        TensorRole,
    };

    #[test]
    fn preserve_rule_propagates_from_evidence_backed_seed() {
        let input = StableId::new("tensor", ["in"]);
        let output = StableId::new("tensor", ["out"]);
        let op = StableId::new("operation", ["op"]);
        let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
        model.tensors.push(TensorContract {
            id: input.clone(),
            name: "x".into(),
            role: TensorRole::Input,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "F32".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: vec![Evidence {
                kind: EvidenceKind::Source,
                confidence: Confidence::Proven,
                source: Some("test".into()),
                detail: "static dtype F32".into(),
            }],
        });
        model.tensors.push(TensorContract {
            id: output.clone(),
            name: "y".into(),
            role: TensorRole::Activation,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "Unknown".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: Vec::new(),
        });
        model.operations.push(Operation {
            id: op,
            function: StableId::new("function", ["f"]),
            name: "relu".into(),
            qualified_name: None,
            inputs: vec![input],
            output: output.clone(),
            source: "test".into(),
            dtype_rule: "Preserve".into(),
            gradient_rule: "Propagates".into(),
            device_rule: "unknown".into(),
            shape_rule: "unknown".into(),
            domain_rule: String::new(),
            execution_phase: None,
            timing: None,
            evidence: Vec::new(),
        });
        propagate_tensor_dtypes(&mut model, &HashMap::new());
        assert_eq!(model.tensors[1].dtype, "F32");
        assert!(tensor_dtype_is_evidence_backed(&model.tensors[1]));
        assert!(!tensor_dtype_is_proven_for_display(&model.tensors[1]));
    }

    #[test]
    fn propagated_dtype_is_not_shown_without_proven_evidence() {
        let input = StableId::new("tensor", ["in"]);
        let mid = StableId::new("tensor", ["mid"]);
        let output = StableId::new("tensor", ["out"]);
        let op1 = StableId::new("operation", ["op1"]);
        let op2 = StableId::new("operation", ["op2"]);
        let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
        model.tensors.push(TensorContract {
            id: input.clone(),
            name: "x".into(),
            role: TensorRole::Input,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "F32".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: vec![Evidence {
                kind: EvidenceKind::Source,
                confidence: Confidence::Proven,
                source: Some("test".into()),
                detail: "static dtype F32".into(),
            }],
        });
        model.tensors.push(TensorContract {
            id: mid.clone(),
            name: "y".into(),
            role: TensorRole::Activation,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "Unknown".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: Vec::new(),
        });
        model.tensors.push(TensorContract {
            id: output.clone(),
            name: "z".into(),
            role: TensorRole::Activation,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "Unknown".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: Vec::new(),
        });
        for (name, input_id, output_id, op_id) in [
            ("relu", input.clone(), mid.clone(), op1),
            ("relu", mid.clone(), output.clone(), op2),
        ] {
            model.operations.push(Operation {
                id: op_id,
                function: StableId::new("function", ["f"]),
                name: name.into(),
                qualified_name: None,
                inputs: vec![input_id],
                output: output_id,
                source: "test".into(),
                dtype_rule: "Preserve".into(),
                gradient_rule: "Propagates".into(),
                device_rule: "unknown".into(),
                shape_rule: "unknown".into(),
                domain_rule: String::new(),
                execution_phase: None,
                timing: None,
                evidence: Vec::new(),
            });
        }
        propagate_tensor_dtypes(&mut model, &HashMap::new());
        assert_eq!(model.tensors[1].dtype, "F32");
        assert!(tensor_dtype_is_evidence_backed(&model.tensors[1]));
        assert!(!tensor_dtype_is_proven_for_display(&model.tensors[1]));
        assert_eq!(model.tensors[2].dtype, "F32");
        assert!(!tensor_dtype_is_proven_for_display(&model.tensors[2]));
    }

    #[test]
    fn checkpoint_parameter_seeds_linked_operation_output() {
        let output = StableId::new("tensor", ["out"]);
        let op = StableId::new("operation", ["linear"]);
        let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
        model.parameters.push(crate::model_ir::Parameter {
            id: StableId::new("parameter", ["vb", "w"]),
            component: StableId::new("component", ["m"]),
            module: StableId::new("module", ["enc"]),
            key: "enc.weight".into(),
            builder_root: "vb".into(),
            role: ParameterRole::Optimized,
            kind: "get".into(),
            symbolic_shape: None,
            checkpoint_shape: None,
            checkpoint_dtype: Some("F32".into()),
            source: "test".into(),
            uses: vec![op.clone()],
            optimizer_memberships: Vec::new(),
            evidence: Vec::new(),
        });
        model.tensors.push(TensorContract {
            id: output.clone(),
            name: "hidden".into(),
            role: TensorRole::Activation,
            owner_function: StableId::new("function", ["f"]),
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "Unknown".into(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: Vec::new(),
        });
        model.operations.push(Operation {
            id: op,
            function: StableId::new("function", ["f"]),
            name: "forward".into(),
            qualified_name: Some("Linear::forward".into()),
            inputs: vec![StableId::new("tensor", ["x"])],
            output: output.clone(),
            source: "test".into(),
            dtype_rule: "Preserve".into(),
            gradient_rule: "Propagates".into(),
            device_rule: "unknown".into(),
            shape_rule: "unknown".into(),
            domain_rule: String::new(),
            execution_phase: None,
            timing: None,
            evidence: Vec::new(),
        });
        propagate_tensor_dtypes(&mut model, &HashMap::new());
        assert_eq!(model.tensors[0].dtype, "F32");
    }

    #[test]
    fn entrypoint_inputs_are_not_blanket_seeded() {
        let func = StableId::new("function", ["forward"]);
        let input = StableId::new("tensor", ["frames"]);
        let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
        model.functions.push(crate::model_ir::Function {
            id: func.clone(),
            name: "forward".into(),
            qualified_name: "Model::forward".into(),
            owner_type: Some("Model".into()),
            visibility: crate::model_ir::Visibility::Public,
            parameters: Vec::new(),
            return_type: None,
            cfg_predicates: Vec::new(),
            cfg_active: Some(true),
            source: "test".into(),
            calls: Vec::new(),
            tensor_inputs: vec![input.clone()],
            tensor_outputs: Vec::new(),
            is_entrypoint: true,
            is_loss: false,
            execution_phases: Vec::new(),
        });
        model.tensors.push(TensorContract {
            id: input,
            name: "frames".into(),
            role: TensorRole::Input,
            owner_function: func,
            parameter: None,
            shape: ShapeFact::default(),
            dtype: "Unknown".into(),
            device: crate::model_ir::DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            execution_phase: None,
            evidence: Vec::new(),
        });
        let mut builder_dtypes = HashMap::new();
        builder_dtypes.insert("vb".into(), "F32".into());
        propagate_tensor_dtypes(&mut model, &builder_dtypes);
        assert_eq!(model.tensors[0].dtype, "Unknown");
    }
}
