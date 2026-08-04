use candle_graph::viewer::{embed_json, escape_for_script, render_html};
use candle_graph::viewer_projection::{project, VIEWER_SCHEMA};
use candle_graph::{
    model_ir::{
        Component, Confidence, DeviceFact, Evidence, EvidenceKind, Function, LayoutFact,
        ModelCoverage, ModelIr, Module, Operation, Parameter, ParameterRole, ShapeFact, StableId,
        TensorContract, TensorRole,
    },
    phase::ExecutionPhase,
};

fn sample_model() -> ModelIr {
    let component_id = StableId::new("component", ["Root"]);
    let module_id = StableId::new("module", ["Root", ""]);
    let param_id = StableId::new("parameter", ["Root", "head.weight"]);
    let tensor_id = StableId::new("tensor", ["train", "Root::forward", "0"]);
    let output_id = StableId::new("tensor", ["train", "Root::forward", "2"]);
    let op_id = StableId::new("operation", ["train", "Root::forward", "1"]);
    let function_id = StableId::new("function", ["Root::forward"]);
    ModelIr {
        schema: candle_graph::model_ir::MODEL_IR_SCHEMA.to_string(),
        analysis_id: StableId::new("analysis", ["test"]),
        cargo: None,
        coverage: ModelCoverage {
            components: 1,
            modules: 1,
            parameters: 1,
            tensors: 1,
            operations: 1,
            diagnostics: 1,
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
            visibility: candle_graph::model_ir::Visibility::Public,
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
                detail: "sample </script><script>alert(1)</script>".into(),
            }],
        }],
        operations: vec![Operation {
            id: op_id,
            function: StableId::new("function", ["Root::forward"]),
            name: "matmul".into(),
            qualified_name: None,
            inputs: vec![tensor_id],
            output: output_id,
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
fn escape_for_script_neutralizes_script_breakouts() {
    let raw = "foo</script><script>alert(1)</script>bar";
    let esc = escape_for_script(raw);
    assert!(!esc.to_lowercase().contains("</script>"));
    assert!(esc.contains("\\u003c"));
}

#[test]
fn embed_json_escapes_angle_brackets_in_values() {
    let v = serde_json::json!({"msg": "</script><img onerror=alert(1)>"});
    let embedded = embed_json(&v);
    assert!(!embedded.to_lowercase().contains("</script>"));
}

#[test]
fn render_html_escapes_payload() {
    let payload = project(&sample_model());
    let html = render_html(&payload);
    assert!(html.contains(r#"id="cg-payload""#));
    let after_payload = html.split(r#"id="cg-payload""#).nth(1).unwrap();
    let script_payload = after_payload
        .split("</script>")
        .next()
        .unwrap()
        .to_lowercase();
    assert!(!script_payload.contains("</script"));
}

#[test]
fn render_html_includes_landmarks_and_data_attributes() {
    let html = render_html(&project(&sample_model()));

    assert!(html.contains(r#"data-viewer="candle-graph""#));
    assert!(html.contains(r#"data-pane="sidebar""#));
    assert!(html.contains(r#"data-pane="canvas""#));
    assert!(html.contains(r#"data-pane="inspector""#));
    assert!(html.contains(r#"data-view-tabs"#));
    assert!(html.contains(r#"data-module-tree"#));
    assert!(html.contains(r#"data-findings-list"#));
    assert!(html.contains(r#"data-canvas"#));
    assert!(html.contains(r#"data-inspector"#));
    assert!(html.contains(r#"data-coverage"#));
    assert!(html.contains(r#"data-legend"#));
    assert!(html.contains(r#"data-theme-toggle"#));

    for state in [
        "Trainable",
        "Frozen",
        "Differentiable",
        "Severed",
        "LayoutDependent",
        "Unknown",
    ] {
        assert!(
            html.contains(&format!(r#"data-legend-item="{state}""#)),
            "missing legend item {state}"
        );
    }

    assert!(!html.contains("https://"));
    assert!(!html.contains("<script src="));
    assert!(html.contains("canvas-hint"));
    assert!(html.contains("clearSelection"));
}

#[test]
fn render_html_is_complete_document() {
    let html = render_html(&project(&sample_model()));
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("</html>"));
    assert!(html.contains("JSON.parse(document.getElementById(\"cg-payload\")"));
    assert!(html.contains("<kbd>U</kbd> clear"));
}

#[test]
fn project_uses_viewer_schema() {
    let payload = project(&sample_model());
    assert_eq!(payload["schema"], VIEWER_SCHEMA);
    assert!(!payload["views"]["dataflow_train"]["edges"]
        .as_array()
        .unwrap()
        .is_empty());
}
