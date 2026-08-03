use candle_graph::viewer::{embed_json, escape_for_script, render_html};
use serde_json::json;

fn sample_structure() -> serde_json::Value {
    json!({
        "schema": "candle-graph/structure/1",
        "coverage": {
            "instances": 2,
            "params": 1,
            "params_certain": 1,
            "params_conditional": 0,
            "params_unknown": 0,
            "diagnostics": 1
        },
        "diagnostics": [{
            "at": "model.rs:1",
            "message": "sample </script><script>alert(1)</script> diagnostic",
            "key": null
        }],
        "modules": [
            {
                "prefix": "",
                "root": "vb",
                "prefix_derived": false,
                "type": "Root",
                "field": null,
                "parent": null,
                "repeat": null,
                "certainty": {"kind": "certain"},
                "at": "model.rs:1"
            },
            {
                "prefix": "block_{index}",
                "root": "vb",
                "prefix_derived": true,
                "type": "Block",
                "field": "blocks",
                "parent": "",
                "repeat": {"var": "index", "bound": "depth"},
                "certainty": {"kind": "certain"},
                "at": "model.rs:10"
            }
        ],
        "parameters": [{
            "key": "block_{index}.weight",
            "root": "vb",
            "template": true,
            "kind": "linear",
            "shape": "[dim, dim]",
            "acquired_via": "linear",
            "module": "Block",
            "module_prefix": "block_{index}",
            "certainty": {"kind": "certain"},
            "checkpoint": "NotChecked",
            "at": "model.rs:12"
        }]
    })
}

fn sample_dataflow() -> serde_json::Value {
    json!({
        "schema": "candle-graph/dataflow/1",
        "nodes": [
            {
                "id": "n0",
                "label": "weight",
                "kind": "param",
                "dtype": "BF16",
                "shape": [8, 8],
                "root": "vb",
                "certainty": {"kind": "certain"},
                "grad_state": "Trainable",
                "at": "model.rs:12",
                "repeat_group": "blocks"
            },
            {
                "id": "n1",
                "label": "weight#1",
                "kind": "param",
                "dtype": "BF16",
                "shape": [8, 8],
                "root": "vb",
                "grad_state": "Frozen",
                "at": "model.rs:12",
                "repeat_group": "blocks"
            },
            {
                "id": "n2",
                "label": "loss",
                "kind": "op",
                "dtype": "F32",
                "grad_state": "Severed",
                "at": "model.rs:40"
            }
        ],
        "edges": [
            {"id": "e0", "from": "n0", "to": "n2", "kind": "data"},
            {"id": "e1", "from": "n1", "to": "n2", "kind": "severing", "grad_state": "Severed"}
        ],
        "diagnostics": []
    })
}

#[test]
fn escape_for_script_neutralizes_script_breakouts() {
    let raw = "foo</script><script>alert(1)</script>bar";
    let esc = escape_for_script(raw);
    assert!(!esc.to_lowercase().contains("</script>"));
    assert!(esc.contains("\\u003c"));
    assert!(esc.contains("script"));
}

#[test]
fn embed_json_escapes_angle_brackets_in_values() {
    let v = json!({"msg": "</script><img onerror=alert(1)>"});
    let embedded = embed_json(&v);
    assert!(!embedded.to_lowercase().contains("</script>"));
    assert!(embedded.contains("\\u003c/script\\u003e") || embedded.contains("\\u003c"));
}

#[test]
fn render_html_escapes_structure_payload() {
    let html = render_html(&sample_structure(), None);
    assert!(html.contains(r#"id="cg-structure""#));
    // Raw breakout sequence must not appear as closable script markup in the payload.
    let after_structure = html.split(r#"id="cg-structure""#).nth(1).unwrap();
    let payload = after_structure
        .split("</script>")
        .next()
        .unwrap()
        .to_lowercase();
    assert!(
        !payload.contains("</script"),
        "structure JSON payload must not contain a raw script closer"
    );
    assert!(html.contains("\\u003c/script\\u003e") || html.contains("\\u003cscript\\u003e"));
}

#[test]
fn render_html_includes_landmarks_and_data_attributes() {
    let html = render_html(&sample_structure(), None);

    assert!(html.contains(r#"data-viewer="candle-graph""#));
    assert!(html.contains(r#"data-pane="hierarchy""#));
    assert!(html.contains(r#"data-pane="canvas""#));
    assert!(html.contains(r#"data-pane="inspector""#));
    assert!(html.contains(r#"data-module-tree"#));
    assert!(html.contains(r#"data-module-search"#));
    assert!(html.contains(r#"data-canvas"#));
    assert!(html.contains(r#"data-inspector"#));
    assert!(html.contains(r#"data-coverage"#));
    assert!(html.contains(r#"data-diagnostics"#));
    assert!(html.contains(r#"data-legend"#));
    assert!(html.contains(r#"data-theme-toggle"#));
    assert!(html.contains(r#"data-empty-dataflow="true""#));
    assert!(html.contains(r#"data-empty-state"#));
    assert!(html.contains(r#"role="tree""#));
    assert!(html.contains(r#"role="banner""#));
    assert!(html.contains(r#"aria-label="Module hierarchy""#));
    assert!(html.contains(r#"aria-label="Dataflow canvas""#));
    assert!(html.contains(r#"aria-label="Inspector""#));

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

    for field in ["source", "shape", "dtype", "root", "certainty", "grad"] {
        assert!(
            html.contains(&format!(r#"data-field="{field}""#)),
            "missing inspector field {field}"
        );
    }

    // No external network / CDN dependencies.
    assert!(!html.contains("https://"));
    assert!(!html.contains("http://"));
    assert!(!html.contains("cdn."));
}

#[test]
fn render_html_with_dataflow_clears_empty_marker_and_embeds_graph() {
    let html = render_html(&sample_structure(), Some(&sample_dataflow()));
    assert!(!html.contains(r#"data-empty-dataflow="true""#));
    assert!(html.contains(r#"id="cg-dataflow""#));
    assert!(html.contains("Trainable"));
    assert!(html.contains("repeat_group"));
    assert!(html.contains("n0"));
}

#[test]
fn render_html_is_complete_document() {
    let html = render_html(&sample_structure(), None);
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("<html"));
    assert!(html.contains("</html>"));
    assert!(html.contains("<style>"));
    assert!(html.contains("const S=JSON.parse"));
}
