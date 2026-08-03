//! Output formats.
//!
//! Two consumers, two shapes. The tree is for humans. The JSON is for agents, and is
//! deliberately *not* a raw dump: it leads with coverage and diagnostics so a reader learns how
//! much of the model was actually seen before reading any of it.

use crate::ir::*;
use crate::load::Crate;
use crate::verify::VerifyReport;
use crate::{
    dataflow::{EdgeKind, ExprGraph, GradState, NodeKind},
    op_semantics::AbstractDtype,
};

/// Human-facing indented tree.
pub fn tree(structure: &Structure, krate: &Crate, show_spans: bool) -> String {
    let mut out = String::new();
    let Some(root) = structure.root else {
        return "no root instance\n".to_string();
    };
    render(structure, krate, root, 0, show_spans, &mut out);

    let coverage = structure.coverage();
    out.push_str(&format!(
        "\n{} instances, {} parameters ({} certain, {} conditional, {} unknown), {} diagnostics\n",
        coverage.instances,
        coverage.params,
        coverage.params_certain,
        coverage.params_conditional,
        coverage.params_unknown,
        coverage.diagnostics,
    ));
    out
}

fn render(
    structure: &Structure,
    krate: &Crate,
    id: ModuleInstanceId,
    depth: usize,
    show_spans: bool,
    out: &mut String,
) {
    let instance = structure.instance(id);
    let def = structure.def(instance.def);
    let pad = "  ".repeat(depth);

    let label = match &instance.via_field {
        Some(field) => format!("{field}: {}", def.name),
        None => def.name.clone(),
    };
    let prefix = if instance.prefix.is_empty() {
        "<root>".to_string()
    } else {
        instance.prefix.to_string()
    };

    out.push_str(&format!("{pad}{label}  [{}/{prefix}]", instance.root));
    if instance.prefix_derived {
        out.push('~');
    }
    if let Some(repeat) = &instance.repeat {
        out.push_str(&format!("  x{{{}}} over {}", repeat.var, repeat.bound));
    }
    if let Certainty::Conditional(reason) = &instance.certainty {
        out.push_str(&format!("  (conditional: {reason})"));
    }
    if show_spans {
        out.push_str(&format!("  @{}", krate.file_label(instance.origin)));
    }
    out.push('\n');

    for param_id in structure
        .params
        .iter()
        .filter(|p| p.owner == id)
        .map(|p| p.id)
    {
        let param = &structure.params[param_id.0];
        let site = structure.site(param.site);
        // Show the key relative to the owning module's prefix, so a leaf reads
        // `input_layernorm.weight` rather than a bare `weight` that could be any of several.
        let leaf = relative(&param.key, &instance.prefix);

        out.push_str(&format!("{pad}  - {leaf}"));
        if let Some(shape) = &site.shape {
            out.push_str(&format!("  shape={shape}"));
        }
        match &param.checkpoint {
            CheckpointMatch::Found { shape, dtype, .. } => {
                out.push_str(&format!("  ckpt={shape:?} {dtype}"));
            }
            CheckpointMatch::FoundMany { count, .. } => {
                out.push_str(&format!("  ckpt={count} tensors"));
            }
            CheckpointMatch::Missing => out.push_str("  ckpt=MISSING"),
            CheckpointMatch::NotChecked => {}
        }
        if let Certainty::Conditional(reason) = &param.certainty {
            out.push_str(&format!("  (conditional: {reason})"));
        }
        if show_spans {
            out.push_str(&format!("  @{}", krate.file_label(site.span)));
        }
        out.push('\n');
    }

    for child in &instance.children {
        render(structure, krate, *child, depth + 1, show_spans, out);
    }
}

/// Strip a module prefix from a parameter key for display. Falls back to the full key when the
/// prefix does not match, which is the honest thing to show if the two ever diverge.
fn relative(key: &Key, prefix: &Key) -> String {
    if key.segs.len() > prefix.segs.len() && key.segs[..prefix.segs.len()] == prefix.segs[..] {
        let rest: Vec<String> = key.segs[prefix.segs.len()..]
            .iter()
            .map(|s| s.to_string())
            .collect();
        return rest.join(".");
    }
    key.to_string()
}

/// Flat parameter list, one dotted key per line. This is the form to diff in CI.
pub fn keys(structure: &Structure) -> String {
    let mut lines: Vec<String> = structure
        .params
        .iter()
        .map(|p| {
            let marker = match &p.certainty {
                Certainty::Certain => "",
                Certainty::Conditional(_) => "  # conditional",
                Certainty::Unknown(_) => "  # unknown",
            };
            format!("{}{marker}", p.key)
        })
        .collect();
    lines.sort();
    lines.dedup();
    lines.join("\n") + "\n"
}

/// Agent-facing JSON. Coverage and diagnostics come first on purpose.
pub fn json(
    structure: &Structure,
    krate: &Crate,
    verify: Option<&VerifyReport>,
) -> serde_json::Value {
    let diagnostics: Vec<serde_json::Value> = structure
        .diagnostics
        .iter()
        .map(|d| {
            serde_json::json!({
                "at": krate.file_label(d.span),
                "message": d.message,
                "key": d.key.as_ref().map(|k| k.to_string()),
            })
        })
        .collect();

    let params: Vec<serde_json::Value> = structure
        .params
        .iter()
        .map(|p| {
            let site = structure.site(p.site);
            let owner = structure.instance(p.owner);
            serde_json::json!({
                "key": p.key.to_string(),
                "root": p.root,
                "template": p.key.is_template(),
                "kind": site.kind,
                "shape": site.shape,
                "acquired_via": site.acquisition,
                "module": structure.def(owner.def).name,
                "module_prefix": owner.prefix.to_string(),
                "certainty": p.certainty,
                "checkpoint": p.checkpoint,
                "at": krate.file_label(site.span),
            })
        })
        .collect();

    let modules: Vec<serde_json::Value> = structure
        .instances
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id.0,
                "prefix": m.prefix.to_string(),
                "root": m.root,
                "prefix_derived": m.prefix_derived,
                "type": structure.def(m.def).name,
                "field": m.via_field,
                "parent": m.parent.map(|p| structure.instance(p).prefix.to_string()),
                "parent_id": m.parent.map(|p| p.0),
                "repeat": m.repeat,
                "certainty": m.certainty,
                "at": krate.file_label(m.origin),
            })
        })
        .collect();

    serde_json::json!({
        "schema": "candle-graph/structure/1",
        "coverage": structure.coverage(),
        "diagnostics": diagnostics,
        "verify": verify,
        "modules": modules,
        "parameters": params,
    })
}

/// Viewer- and agent-facing dataflow JSON with stable labels and human-readable source locations.
pub fn dataflow_json(graph: &ExprGraph, krate: &Crate) -> serde_json::Value {
    let nodes: Vec<serde_json::Value> = graph
        .nodes
        .iter()
        .map(|node| {
            let (kind, label) = node_kind(&node.kind);
            serde_json::json!({
                "id": format!("n{}", node.id.0),
                "kind": kind,
                "label": label,
                "shape": node.shape,
                "dtype": dtype_name(node.dtype),
                "grad_state": grad_name(node.grad),
                "at": krate.file_label(node.span),
            })
        })
        .collect();
    let edges: Vec<serde_json::Value> = graph
        .edges
        .iter()
        .map(|edge| {
            serde_json::json!({
                "id": format!("e{}", edge.id.0),
                "from": format!("n{}", edge.from.0),
                "to": format!("n{}", edge.to.0),
                "kind": edge_kind(edge.kind),
                "label": edge.label,
            })
        })
        .collect();
    let conflicts: Vec<serde_json::Value> = graph
        .dtype_conflicts
        .iter()
        .map(|conflict| {
            serde_json::json!({
                "node": format!("n{}", conflict.edge_or_node.0),
                "op": conflict.op,
                "left": dtype_name(conflict.left),
                "right": dtype_name(conflict.right),
                "at": krate.file_label(conflict.span),
                "message": conflict.message,
            })
        })
        .collect();
    let risks: Vec<serde_json::Value> = graph
        .dtype_risks
        .iter()
        .map(|risk| {
            serde_json::json!({
                "node": format!("n{}", risk.edge_or_node.0),
                "op": risk.op,
                "known": dtype_name(risk.known),
                "at": krate.file_label(risk.span),
                "message": risk.message,
            })
        })
        .collect();
    let diagnostics: Vec<serde_json::Value> = graph
        .diagnostics
        .iter()
        .map(|diagnostic| {
            serde_json::json!({
                "at": krate.file_label(diagnostic.span),
                "message": diagnostic.message,
            })
        })
        .collect();

    serde_json::json!({
        "schema": "candle-graph/dataflow/1",
        "coverage": {
            "nodes": graph.nodes.len(),
            "edges": graph.edges.len(),
            "parameters": graph.param_nodes.len(),
            "losses": graph.loss_nodes.len(),
            "dead_parameters": graph.dead_params().len(),
            "severing_edges": graph.severing_edges().len(),
            "dtype_conflicts": graph.dtype_conflicts.len(),
            "dtype_risks": graph.dtype_risks.len(),
            "diagnostics": graph.diagnostics.len(),
        },
        "entry_return": graph.entry_return.map(|id| format!("n{}", id.0)),
        "loss_nodes": graph.loss_nodes.iter().map(|id| format!("n{}", id.0)).collect::<Vec<_>>(),
        "parameter_nodes": graph.param_nodes.iter().map(|id| format!("n{}", id.0)).collect::<Vec<_>>(),
        "dead_parameters": graph.dead_params().iter().map(|id| format!("n{}", id.0)).collect::<Vec<_>>(),
        "severing_edges": graph.severing_edges().iter().map(|id| format!("e{}", id.0)).collect::<Vec<_>>(),
        "dtype_conflicts": conflicts,
        "dtype_risks": risks,
        "diagnostics": diagnostics,
        "nodes": nodes,
        "edges": edges,
    })
}

/// Stable, sorted findings used in human summaries and CI baselines.
pub fn dataflow_findings(graph: &ExprGraph, krate: &Crate) -> Vec<String> {
    let mut findings = Vec::new();
    for conflict in &graph.dtype_conflicts {
        findings.push(format!(
            "dtype-conflict\t{}\t{}\t{} vs {}\t{}",
            krate.file_label(conflict.span),
            conflict.op,
            dtype_name(conflict.left),
            dtype_name(conflict.right),
            conflict.message
        ));
    }
    for risk in &graph.dtype_risks {
        findings.push(format!(
            "dtype-risk\t{}\t{}\tknown {} + unknown\t{}",
            krate.file_label(risk.span),
            risk.op,
            dtype_name(risk.known),
            risk.message
        ));
    }
    for id in graph.dead_params() {
        let node = graph.node(id);
        let (_, label) = node_kind(&node.kind);
        findings.push(format!(
            "dead-parameter\t{}\t{}\t{}",
            krate.file_label(node.span),
            label,
            grad_name(node.grad)
        ));
    }
    for edge_id in graph.severing_edges() {
        let edge = graph.edge(edge_id);
        let target = graph.node(edge.to);
        let (_, label) = node_kind(&target.kind);
        findings.push(format!(
            "severing-edge\t{}\t{}\t{}",
            krate.file_label(target.span),
            label,
            edge.label.as_deref().unwrap_or("data")
        ));
    }
    findings.sort();
    findings.dedup();
    findings
}

/// Compact human-readable dataflow summary.
pub fn dataflow_text(graph: &ExprGraph, krate: &Crate) -> String {
    let findings = dataflow_findings(graph, krate);
    let mut out = format!(
        "\ndataflow: {} nodes, {} edges, {} loss sinks, {} dtype conflicts, {} dtype risks, {} dead trainable parameters, {} severing edges\n",
        graph.nodes.len(),
        graph.edges.len(),
        graph.loss_nodes.len(),
        graph.dtype_conflicts.len(),
        graph.dtype_risks.len(),
        graph.dead_params().len(),
        graph.severing_edges().len(),
    );
    for finding in findings {
        out.push_str("  ");
        out.push_str(&finding.replace('\t', "  "));
        out.push('\n');
    }
    out
}

fn node_kind(kind: &NodeKind) -> (&'static str, String) {
    match kind {
        NodeKind::Param { name } => ("parameter", name.clone()),
        NodeKind::Local { name } => ("local", name.clone()),
        NodeKind::Call { callee } => ("operation", callee.clone()),
        NodeKind::Literal { text } => ("literal", text.clone()),
        NodeKind::Phi => ("phi", "branch join".to_string()),
        NodeKind::Return => ("return", "return".to_string()),
        NodeKind::Unknown { reason } => ("unknown", reason.clone()),
    }
}

fn grad_name(grad: GradState) -> &'static str {
    match grad {
        GradState::Trainable => "Trainable",
        GradState::Frozen => "Frozen",
        GradState::Differentiable => "Differentiable",
        GradState::Severed => "Severed",
        GradState::LayoutDependent => "LayoutDependent",
        GradState::Unknown => "Unknown",
    }
}

fn dtype_name(dtype: AbstractDtype) -> &'static str {
    match dtype {
        AbstractDtype::F64 => "F64",
        AbstractDtype::F32 => "F32",
        AbstractDtype::F16 => "F16",
        AbstractDtype::Bf16 => "BF16",
        AbstractDtype::I16 => "I16",
        AbstractDtype::I32 => "I32",
        AbstractDtype::I64 => "I64",
        AbstractDtype::U32 => "U32",
        AbstractDtype::U8 => "U8",
        AbstractDtype::F8E4M3 => "F8E4M3",
        AbstractDtype::F6E2M3 => "F6E2M3",
        AbstractDtype::F6E3M2 => "F6E3M2",
        AbstractDtype::F4 => "F4",
        AbstractDtype::F8E8M0 => "F8E8M0",
        AbstractDtype::Unknown => "Unknown",
    }
}

fn edge_kind(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::Data => "data",
        EdgeKind::Severing => "severing",
        EdgeKind::Control => "control",
    }
}
