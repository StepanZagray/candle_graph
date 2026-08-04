//! Expression-level dataflow, dtype, and gradient-connectivity analysis.
//!
//! Builds a serializable graph over ordinary crate-local free functions and inherent methods
//! starting from a chosen entrypoint. Transfer rules live in [`crate::op_semantics`]; anything
//! not covered is left [`GradState::Unknown`] / [`AbstractDtype::Unknown`] rather than guessed.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::Serialize;
use syn::spanned::Spanned;

use crate::ir::SrcSpan;
use crate::load::{self, Crate, ImplFn};
use crate::op_semantics::{
    self, affine_domain, domain_includes_zero, domain_violation, library_body, AbstractDtype,
    BodyAtom, DomainRequirement, DomainViolationConfidence, DtypeRule, GradFlow, LibraryBody,
    NumericDomain, OpEffect,
};
use crate::phase::ExecutionPhase;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
        pub struct $name(pub usize);
    };
}

id_type!(NodeId);
id_type!(EdgeId);

/// Gradient / trainability state attached to an expression node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GradState {
    /// A leaf that is expected to receive gradients (e.g. `Var` / trainable parameter).
    Trainable,
    /// A leaf that must not receive gradients (frozen weight, constant).
    Frozen,
    /// An intermediate on a differentiable path.
    Differentiable,
    /// Gradient flow was cut (`detach`, known `apply_op*_no_bwd`, …).
    Severed,
    /// Gradient exists only under layout assumptions we cannot prove.
    LayoutDependent,
    /// Not enough information.
    Unknown,
}

/// What an expression node represents in source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeKind {
    /// Function / method parameter.
    Param { name: String },
    /// `let` binding.
    Local { name: String },
    /// Result of a call or method call.
    Call { callee: String },
    /// Literal / constructor-ish value (dtype may still be known from `DType::…`).
    Literal { text: String },
    /// Branch join (phi) of mutually exclusive arms.
    Phi,
    /// Return value of the analyzed entry / callee.
    Return,
    /// Placeholder when the expression could not be modeled.
    Unknown { reason: String },
}

/// Edge role in the expression graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Ordinary data dependence (operand → result).
    Data,
    /// Data dependence that severs autograd.
    Severing,
    /// Control / branch contribution into a phi.
    Control,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExprNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub span: SrcSpan,
    /// Symbolic shape text when observed; never invented.
    pub shape: Option<String>,
    pub dtype: AbstractDtype,
    pub grad: GradState,
    /// Float-range domain after rounding. Unknown until a catalog rule assigns one.
    pub domain: NumericDomain,
    /// Best-effort resolved source type. `None` remains explicitly unknown.
    pub type_name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExprEdge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    /// Operand slot or note, e.g. `lhs`, `rhs`, `self`.
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DtypeConflict {
    pub edge_or_node: NodeId,
    pub op: String,
    pub left: AbstractDtype,
    pub right: AbstractDtype,
    pub span: SrcSpan,
    pub message: String,
}

/// A same-dtype op where at least one operand is known and another remains unknown.
///
/// This is intentionally separate from [`DtypeConflict`]: it highlights a runtime mismatch risk
/// without claiming that the unknown operand definitely differs.
#[derive(Debug, Clone, Serialize)]
pub struct DtypeRisk {
    pub edge_or_node: NodeId,
    pub op: String,
    pub known: AbstractDtype,
    pub span: SrcSpan,
    pub message: String,
}

/// How a numeric hazard can interfere with training or inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NumericImpact {
    /// Hazard is on a loss sink (or reaches one): training can abort with NaN loss.
    TrainingLossNaN,
    /// Hazard lies on a trainable → loss path: gradients can be poisoned.
    GradientPoison,
    /// Hazard reaches the entry return of a non-loss forward.
    InferenceOutputRisk,
    /// Unstable value proven locally but not shown to reach loss or return.
    #[default]
    LocalOnly,
}

impl NumericImpact {
    pub fn label(self) -> &'static str {
        match self {
            Self::TrainingLossNaN => "training impact: loss can become NaN when values saturate",
            Self::GradientPoison => {
                "training impact: gradients can be poisoned by non-finite values"
            }
            Self::InferenceOutputRisk => "inference impact: forward outputs can become non-finite",
            Self::LocalOnly => "local numeric hazard (not shown to reach loss or return)",
        }
    }

    pub fn is_training_failure(self) -> bool {
        matches!(self, Self::TrainingLossNaN | Self::GradientPoison)
    }
}

/// A partial function applied to a domain that can attain a forbidden endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct NumericDomainViolation {
    pub edge_or_node: NodeId,
    pub op: String,
    pub requires: String,
    pub producer_domain: String,
    pub proven: bool,
    pub impact: NumericImpact,
    pub span: SrcSpan,
    pub message: String,
    /// Source citation when the hazard came from an expanded library body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_cite: Option<String>,
}

/// Multiply that can evaluate `0 * ±inf` (silent NaN) rather than a loud `-inf` loss.
#[derive(Debug, Clone, Serialize)]
pub struct ZeroTimesInfinity {
    pub edge_or_node: NodeId,
    pub impact: NumericImpact,
    pub span: SrcSpan,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_cite: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DataflowDiagnostic {
    pub span: SrcSpan,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExprGraph {
    pub nodes: Vec<ExprNode>,
    pub edges: Vec<ExprEdge>,
    /// Entrypoint return node, when analysis produced one.
    pub entry_return: Option<NodeId>,
    /// Nodes that are loss sinks (`cross_entropy`, `mse`, …).
    pub loss_nodes: Vec<NodeId>,
    /// Trainable / frozen parameter leaves discovered during analysis.
    pub param_nodes: Vec<NodeId>,
    /// Nodes proven to be Candle tensor values. Non-tensor expressions stay out of ModelIr.
    pub tensor_nodes: Vec<NodeId>,
    pub dtype_conflicts: Vec<DtypeConflict>,
    pub dtype_risks: Vec<DtypeRisk>,
    pub numeric_domain_violations: Vec<NumericDomainViolation>,
    pub zero_times_infinity: Vec<ZeroTimesInfinity>,
    pub diagnostics: Vec<DataflowDiagnostic>,
}

impl ExprGraph {
    pub fn node(&self, id: NodeId) -> &ExprNode {
        &self.nodes[id.0]
    }

    #[allow(dead_code)]
    pub fn edge(&self, id: EdgeId) -> &ExprEdge {
        &self.edges[id.0]
    }

    /// Trainable leaves with no reverse data path to any loss node.
    pub fn dead_params(&self) -> Vec<NodeId> {
        let reachable = self.nodes_reaching_losses();
        self.param_nodes
            .iter()
            .copied()
            .filter(|id| {
                matches!(self.node(*id).grad, GradState::Trainable) && !reachable.contains(id)
            })
            .collect()
    }

    /// Sever autograd connectivity as in inference / `no_grad` mode.
    pub fn apply_inference_mode(&mut self) {
        for node in &mut self.nodes {
            if matches!(
                node.grad,
                GradState::Trainable | GradState::Differentiable | GradState::LayoutDependent
            ) {
                node.grad = GradState::Severed;
            }
        }
    }

    /// Edges marked as severing autograd.
    pub fn severing_edges(&self) -> Vec<EdgeId> {
        self.edges
            .iter()
            .filter(|e| matches!(e.kind, EdgeKind::Severing))
            .map(|e| e.id)
            .collect()
    }

    /// Same-dtype operand mismatches recorded during transfer.
    pub fn dtype_conflicts(&self) -> &[DtypeConflict] {
        &self.dtype_conflicts
    }

    /// Partially-known operands at an op that requires matching dtypes.
    pub fn dtype_risks(&self) -> &[DtypeRisk] {
        &self.dtype_risks
    }

    /// Differentiable paths from `from` to `to`; severing edges are intentionally excluded.
    /// Caps path count to keep recursion honest on dense graphs.
    pub fn paths_to(&self, from: NodeId, to: NodeId) -> Vec<Vec<NodeId>> {
        const MAX_PATHS: usize = 64;
        const MAX_DEPTH: usize = 128;
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for e in &self.edges {
            if matches!(e.kind, EdgeKind::Severing) {
                continue;
            }
            adj.entry(e.from).or_default().push(e.to);
        }
        let mut out = Vec::new();
        let mut stack = vec![from];
        let mut visiting = HashSet::new();
        visiting.insert(from);
        self.dfs_paths(
            from,
            to,
            &adj,
            &mut stack,
            &mut visiting,
            &mut out,
            MAX_PATHS,
            MAX_DEPTH,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs_paths(
        &self,
        cur: NodeId,
        to: NodeId,
        adj: &HashMap<NodeId, Vec<NodeId>>,
        stack: &mut Vec<NodeId>,
        visiting: &mut HashSet<NodeId>,
        out: &mut Vec<Vec<NodeId>>,
        max_paths: usize,
        max_depth: usize,
    ) {
        if out.len() >= max_paths || stack.len() > max_depth {
            return;
        }
        if cur == to {
            out.push(stack.clone());
            return;
        }
        let Some(nexts) = adj.get(&cur) else {
            return;
        };
        for &n in nexts {
            if !visiting.insert(n) {
                continue;
            }
            stack.push(n);
            self.dfs_paths(n, to, adj, stack, visiting, out, max_paths, max_depth);
            stack.pop();
            visiting.remove(&n);
        }
    }

    /// Nodes that can reach a loss following edges forward (operand → result → … → loss).
    fn nodes_reaching_losses(&self) -> HashSet<NodeId> {
        // Reverse adjacency: result → operands, then BFS from losses.
        let mut rev: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for e in &self.edges {
            if matches!(e.kind, EdgeKind::Severing) {
                // Severed edges do not carry gradient; they do not count for "alive".
                continue;
            }
            rev.entry(e.to).or_default().push(e.from);
        }
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        for &loss in &self.loss_nodes {
            if seen.insert(loss) {
                q.push_back(loss);
            }
        }
        while let Some(n) = q.pop_front() {
            if let Some(preds) = rev.get(&n) {
                for &p in preds {
                    if seen.insert(p) {
                        q.push_back(p);
                    }
                }
            }
        }
        seen
    }

    /// Forward-reachable nodes from `start` along non-severing edges.
    pub fn reachable_from(&self, start: NodeId) -> HashSet<NodeId> {
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for e in &self.edges {
            if matches!(e.kind, EdgeKind::Severing) {
                continue;
            }
            adj.entry(e.from).or_default().push(e.to);
        }
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        seen.insert(start);
        q.push_back(start);
        while let Some(n) = q.pop_front() {
            if let Some(nexts) = adj.get(&n) {
                for &next in nexts {
                    if seen.insert(next) {
                        q.push_back(next);
                    }
                }
            }
        }
        seen
    }

    /// Classify numeric hazards by whether they can fail training or interfere with inference.
    pub fn classify_numeric_impacts(&mut self) {
        let can_reach_loss = self.nodes_reaching_losses();
        let loss_nodes: HashSet<NodeId> = self.loss_nodes.iter().copied().collect();
        let trainable: Vec<NodeId> = self
            .param_nodes
            .iter()
            .copied()
            .filter(|id| matches!(self.node(*id).grad, GradState::Trainable))
            .collect();
        let can_reach_return = {
            let mut set = HashSet::new();
            if let Some(ret) = self.entry_return {
                let mut rev: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
                for e in &self.edges {
                    if matches!(e.kind, EdgeKind::Severing) {
                        continue;
                    }
                    rev.entry(e.to).or_default().push(e.from);
                }
                let mut q = VecDeque::new();
                set.insert(ret);
                q.push_back(ret);
                while let Some(n) = q.pop_front() {
                    if let Some(preds) = rev.get(&n) {
                        for &p in preds {
                            if set.insert(p) {
                                q.push_back(p);
                            }
                        }
                    }
                }
            }
            set
        };

        let from_trainable: HashSet<NodeId> = trainable
            .iter()
            .flat_map(|param| self.reachable_from(*param))
            .collect();

        let classify = |node: NodeId| -> NumericImpact {
            let reaches_loss = loss_nodes.contains(&node) || can_reach_loss.contains(&node);
            if reaches_loss {
                return NumericImpact::TrainingLossNaN;
            }
            if from_trainable.contains(&node) {
                return NumericImpact::GradientPoison;
            }
            if can_reach_return.contains(&node) {
                return NumericImpact::InferenceOutputRisk;
            }
            NumericImpact::LocalOnly
        };

        let domain_nodes: Vec<NodeId> = self
            .numeric_domain_violations
            .iter()
            .map(|finding| finding.edge_or_node)
            .collect();
        let domain_impacts: Vec<NumericImpact> =
            domain_nodes.iter().copied().map(classify).collect();
        for (finding, impact) in self
            .numeric_domain_violations
            .iter_mut()
            .zip(domain_impacts)
        {
            finding.impact = impact;
            if !finding.message.contains("impact:") {
                finding
                    .message
                    .push_str(&format!(" ({})", finding.impact.label()));
            }
        }

        let zti_nodes: Vec<NodeId> = self
            .zero_times_infinity
            .iter()
            .map(|finding| finding.edge_or_node)
            .collect();
        let zti_impacts: Vec<NumericImpact> = zti_nodes.iter().copied().map(classify).collect();
        for (finding, impact) in self.zero_times_infinity.iter_mut().zip(zti_impacts) {
            finding.impact = impact;
            if !finding.message.contains("impact:") {
                finding
                    .message
                    .push_str(&format!(" ({})", finding.impact.label()));
            }
        }
    }
}

/// Analyze `entrypoint` against a loaded crate.
///
/// `entrypoint` is either a free function name (`forward`) or an inherent method
/// (`Model::forward`).
pub fn analyze(krate: &Crate, entrypoint: &str) -> anyhow::Result<ExprGraph> {
    analyze_with_candle_version(krate, entrypoint, None)
}

/// Analyze with the resolved candle-nn version used to gate version-specific gradient rules.
pub fn analyze_with_candle_version(
    krate: &Crate,
    entrypoint: &str,
    candle_nn_version: Option<&str>,
) -> anyhow::Result<ExprGraph> {
    analyze_with_phase(krate, entrypoint, candle_nn_version, ExecutionPhase::Train)
}

/// Analyze an entrypoint for a specific execution phase (train autograd vs inference no-grad).
pub fn analyze_with_phase(
    krate: &Crate,
    entrypoint: &str,
    candle_nn_version: Option<&str>,
    phase: ExecutionPhase,
) -> anyhow::Result<ExprGraph> {
    let mut analyzer = Analyzer::new(krate, candle_nn_version);
    analyzer.run(entrypoint)?;
    if phase == ExecutionPhase::Infer {
        analyzer.graph.apply_inference_mode();
    }
    analyzer.graph.classify_numeric_impacts();
    Ok(analyzer.graph)
}

struct Analyzer<'a> {
    krate: &'a Crate,
    graph: ExprGraph,
    /// Best-effort source types for expression nodes. Entries are only added from explicit
    /// signatures, struct fields, or known Candle tensor operations; absence means unknown.
    node_types: HashMap<NodeId, String>,
    /// Recursion guard: `(type_name, fn_name)` currently on the call stack.
    call_stack: HashSet<(String, String)>,
    file_hint: usize,
    module_path: String,
    candle_nn_version: Option<String>,
    /// True while expanding an audited library body (prevents recursive expansion).
    expanding_library_body: bool,
    /// Citation attached to numeric findings produced by the active expansion.
    expansion_cite: Option<&'static str>,
}

impl<'a> Analyzer<'a> {
    fn new(krate: &'a Crate, candle_nn_version: Option<&str>) -> Self {
        Self {
            krate,
            graph: ExprGraph::default(),
            node_types: HashMap::new(),
            call_stack: HashSet::new(),
            file_hint: 0,
            module_path: String::new(),
            candle_nn_version: candle_nn_version.map(str::to_string),
            expanding_library_body: false,
            expansion_cite: None,
        }
    }

    fn run(&mut self, entrypoint: &str) -> anyhow::Result<()> {
        let (func, type_name) = resolve_entrypoint(self.krate, entrypoint)?;
        self.file_hint = func.span.file;
        self.module_path = func.module_path.clone();
        let key = (type_name.clone(), func.fn_name.clone());
        self.call_stack.insert(key.clone());
        let ret = self.analyze_function(func, &type_name, None, &[])?;
        self.call_stack.remove(&key);
        self.graph.entry_return = ret;
        self.finalize_tensor_nodes();
        Ok(())
    }

    fn add_node(
        &mut self,
        kind: NodeKind,
        span: SrcSpan,
        dtype: AbstractDtype,
        grad: GradState,
        shape: Option<String>,
    ) -> NodeId {
        self.add_node_with_domain(kind, span, dtype, grad, shape, NumericDomain::Unknown)
    }

    fn add_node_with_domain(
        &mut self,
        kind: NodeKind,
        span: SrcSpan,
        dtype: AbstractDtype,
        grad: GradState,
        shape: Option<String>,
        domain: NumericDomain,
    ) -> NodeId {
        let id = NodeId(self.graph.nodes.len());
        self.graph.nodes.push(ExprNode {
            id,
            kind,
            span,
            shape,
            dtype,
            grad,
            domain,
            type_name: None,
        });
        id
    }

    fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        kind: EdgeKind,
        label: Option<String>,
    ) -> EdgeId {
        let id = EdgeId(self.graph.edges.len());
        self.graph.edges.push(ExprEdge {
            id,
            from,
            to,
            kind,
            label,
        });
        id
    }

    fn diagnose(&mut self, span: SrcSpan, message: impl Into<String>) {
        self.graph.diagnostics.push(DataflowDiagnostic {
            span,
            message: message.into(),
        });
    }

    fn set_node_type(&mut self, id: NodeId, ty: impl Into<String>) {
        let ty = ty.into();
        if !ty.is_empty() && ty != "()" {
            self.node_types.insert(id, ty.clone());
            self.graph.nodes[id.0].type_name = Some(ty);
        }
    }

    fn node_type(&self, id: NodeId) -> Option<&str> {
        self.node_types.get(&id).map(String::as_str)
    }

    fn finalize_tensor_nodes(&mut self) {
        let mut tensor = self
            .graph
            .nodes
            .iter()
            .map(|node| {
                node.type_name
                    .as_deref()
                    .is_some_and(is_candle_tensor_receiver)
            })
            .collect::<Vec<_>>();
        loop {
            let mut changed = false;
            for node in &self.graph.nodes {
                if tensor[node.id.0] || !matches!(node.kind, NodeKind::Phi) {
                    continue;
                }
                let inputs = self
                    .graph
                    .edges
                    .iter()
                    .filter(|edge| edge.to == node.id)
                    .map(|edge| edge.from)
                    .collect::<Vec<_>>();
                if !inputs.is_empty() && inputs.iter().all(|input| tensor[input.0]) {
                    tensor[node.id.0] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        self.graph.tensor_nodes = tensor
            .into_iter()
            .enumerate()
            .filter_map(|(index, is_tensor)| is_tensor.then_some(NodeId(index)))
            .collect();
    }

    fn analyze_function(
        &mut self,
        func: &ImplFn,
        type_name: &str,
        receiver: Option<NodeId>,
        args: &[NodeId],
    ) -> anyhow::Result<Option<NodeId>> {
        let mut env: HashMap<String, NodeId> = HashMap::new();
        let mut arg_i = 0usize;
        let _ = type_name;

        for (param_index, pname) in func.params.iter().enumerate() {
            let param_type = func
                .param_types
                .get(param_index)
                .map(String::as_str)
                .unwrap_or_default();
            if pname == "self" {
                let id = if let Some(recv) = receiver {
                    recv
                } else {
                    self.add_node(
                        NodeKind::Param {
                            name: "self".into(),
                        },
                        func.span,
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                };
                self.set_node_type(id, type_name);
                env.insert("self".to_string(), id);
                continue;
            }

            let id = if arg_i < args.len() {
                let src = args[arg_i];
                arg_i += 1;
                // Only an explicit `Var` parameter type proves trainability. Tensor names such as
                // `weight` do not: frozen and trainable tensors share the same type.
                let g = hint_param_grad(param_type, self.graph.node(src).grad);
                if g != self.graph.node(src).grad {
                    self.graph.nodes[src.0].grad = g;
                }
                if matches!(
                    self.graph.node(src).grad,
                    GradState::Trainable | GradState::Frozen
                ) && !self.graph.param_nodes.contains(&src)
                {
                    self.graph.param_nodes.push(src);
                }
                src
            } else {
                let grad = hint_param_grad(param_type, GradState::Unknown);
                let id = self.add_node(
                    NodeKind::Param {
                        name: pname.clone(),
                    },
                    func.span,
                    AbstractDtype::Unknown,
                    grad,
                    None,
                );
                if matches!(grad, GradState::Trainable | GradState::Frozen) {
                    self.graph.param_nodes.push(id);
                }
                id
            };
            if let Some(base) = source_type_base(param_type) {
                self.set_node_type(id, base);
            }
            env.insert(pname.clone(), id);
        }

        let mut last: Option<NodeId> = None;
        for stmt in &func.block.stmts {
            last = self.stmt(&mut env, stmt)?;
        }
        Ok(last)
    }

    fn stmt(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        stmt: &syn::Stmt,
    ) -> anyhow::Result<Option<NodeId>> {
        match stmt {
            syn::Stmt::Local(local) => {
                let init = match &local.init {
                    Some(local_init) => self.expr(env, &local_init.expr)?,
                    None => {
                        let span = span_of(self.file_hint, local.pat.span());
                        self.add_node(
                            NodeKind::Unknown {
                                reason: "uninitialized local".into(),
                            },
                            span,
                            AbstractDtype::Unknown,
                            GradState::Unknown,
                            None,
                        )
                    }
                };
                bind_pat(env, &local.pat, init);
                Ok(Some(init))
            }
            syn::Stmt::Expr(expr, _) => Ok(Some(self.expr(env, expr)?)),
            syn::Stmt::Item(_) => Ok(None),
            syn::Stmt::Macro(m) => {
                let span = span_of(self.file_hint, m.mac.path.span());
                self.diagnose(span, "macro statement not expanded; left Unknown");
                Ok(Some(self.add_node(
                    NodeKind::Unknown {
                        reason: "macro".into(),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Unknown,
                    None,
                )))
            }
        }
    }

    fn expr(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        expr: &syn::Expr,
    ) -> anyhow::Result<NodeId> {
        match expr {
            syn::Expr::Path(p) => self.expr_path(env, p),
            syn::Expr::Lit(l) => {
                let span = span_of(self.file_hint, l.lit.span());
                Ok(self.add_node(
                    NodeKind::Literal {
                        text: lit_text(&l.lit),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Frozen,
                    None,
                ))
            }
            syn::Expr::Reference(r) => self.expr(env, &r.expr),
            syn::Expr::Paren(p) => self.expr(env, &p.expr),
            syn::Expr::Group(g) => self.expr(env, &g.expr),
            syn::Expr::Try(t) => self.expr(env, &t.expr),
            syn::Expr::Unary(u) => self.expr(env, &u.expr),
            syn::Expr::Field(f) => {
                let base = self.expr(env, &f.base)?;
                let span = span_of(self.file_hint, f.member.span());
                let name = match &f.member {
                    syn::Member::Named(id) => id.to_string(),
                    syn::Member::Unnamed(i) => i.index.to_string(),
                };
                // Field projection: dtype/grad Unknown unless we know more — honest default.
                let id = self.add_node(
                    NodeKind::Local {
                        name: format!(".{name}"),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Unknown,
                    None,
                );
                self.add_edge(base, id, EdgeKind::Data, Some("field".into()));
                if let Some(owner_type) = self.node_type(base).map(ToString::to_string) {
                    let candidates = self.krate.struct_candidates(&owner_type);
                    if let [owner] = candidates.as_slice() {
                        if let Some(field) = owner.fields.iter().find(|field| field.name == name) {
                            if !field.ty.base.is_empty() {
                                self.set_node_type(id, field.ty.base.clone());
                            }
                        }
                    }
                }
                // Do not infer trainability from a struct field called `weight`: candle models
                // commonly store frozen base weights and trainable adapters side by side. Without
                // a resolved builder root, Unknown is safer than a false dead-parameter report.
                Ok(id)
            }
            syn::Expr::MethodCall(m) => self.method_call(env, m),
            syn::Expr::Call(c) => self.func_call(env, c),
            syn::Expr::Binary(b) => self.binary(env, b),
            syn::Expr::If(i) => self.expr_if(env, i),
            syn::Expr::Match(m) => self.expr_match(env, m),
            syn::Expr::ForLoop(loop_expr) => {
                // Analyze one symbolic iteration. This preserves the body dataflow without
                // pretending to know the runtime trip count.
                let iterator = self.expr(env, &loop_expr.expr)?;
                let mut loop_env = env.clone();
                bind_pat(&mut loop_env, &loop_expr.pat, iterator);
                let last = self.expr_block(&mut loop_env, &loop_expr.body)?;
                Ok(last.unwrap_or_else(|| {
                    self.add_node(
                        NodeKind::Literal { text: "()".into() },
                        span_of(self.file_hint, loop_expr.for_token.span),
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                }))
            }
            syn::Expr::While(loop_expr) => {
                let _condition = self.expr(env, &loop_expr.cond)?;
                let last = self.expr_block(env, &loop_expr.body)?;
                Ok(last.unwrap_or_else(|| {
                    self.add_node(
                        NodeKind::Literal { text: "()".into() },
                        span_of(self.file_hint, loop_expr.while_token.span),
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                }))
            }
            syn::Expr::Loop(loop_expr) => {
                let last = self.expr_block(env, &loop_expr.body)?;
                Ok(last.unwrap_or_else(|| {
                    self.add_node(
                        NodeKind::Literal { text: "()".into() },
                        span_of(self.file_hint, loop_expr.loop_token.span),
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                }))
            }
            syn::Expr::Block(b) => {
                let last = self.expr_block(env, &b.block)?;
                Ok(last.unwrap_or_else(|| {
                    let span = span_of(self.file_hint, b.block.brace_token.span.join());
                    self.add_node(
                        NodeKind::Literal { text: "()".into() },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Frozen,
                        None,
                    )
                }))
            }
            syn::Expr::Return(r) => match &r.expr {
                Some(e) => self.expr(env, e),
                None => {
                    let span = span_of(self.file_hint, r.return_token.span);
                    Ok(self.add_node(
                        NodeKind::Return,
                        span,
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    ))
                }
            },
            syn::Expr::Closure(c) => {
                // Analyze closure body in a forked env; captures stay as outer bindings.
                let mut nested = env.clone();
                let last = match &*c.body {
                    syn::Expr::Block(b) => self.expr_block(&mut nested, &b.block)?,
                    other => Some(self.expr(&mut nested, other)?),
                };
                Ok(last.unwrap_or_else(|| {
                    let span = span_of(self.file_hint, c.or1_token.span);
                    self.add_node(
                        NodeKind::Unknown {
                            reason: "empty closure".into(),
                        },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                }))
            }
            syn::Expr::Tuple(t) => {
                let mut last = None;
                for e in &t.elems {
                    last = Some(self.expr(env, e)?);
                }
                Ok(last.unwrap_or_else(|| {
                    let span = SrcSpan::UNKNOWN;
                    self.add_node(
                        NodeKind::Literal { text: "()".into() },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Frozen,
                        None,
                    )
                }))
            }
            syn::Expr::Macro(m) => {
                let span = span_of(self.file_hint, m.mac.path.span());
                // Recognize DType::… inside simple macros? Leave Unknown.
                self.diagnose(
                    span,
                    format!("macro {} not expanded", path_last(&m.mac.path)),
                );
                Ok(self.add_node(
                    NodeKind::Unknown {
                        reason: "macro".into(),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Unknown,
                    None,
                ))
            }
            syn::Expr::Await(a) => self.expr(env, &a.base),
            syn::Expr::Assign(a) => {
                let val = self.expr(env, &a.right)?;
                if let syn::Expr::Path(p) = &*a.left {
                    if let Some(name) = path_ident(p) {
                        env.insert(name, val);
                    }
                }
                Ok(val)
            }
            other => {
                let span = SrcSpan {
                    file: self.file_hint,
                    line: 0,
                    col: 0,
                };
                self.diagnose(
                    span,
                    format!(
                        "unsupported expression {}; left Unknown",
                        expr_kind_name(other)
                    ),
                );
                Ok(self.add_node(
                    NodeKind::Unknown {
                        reason: expr_kind_name(other).into(),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Unknown,
                    None,
                ))
            }
        }
    }

    fn expr_path(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        p: &syn::ExprPath,
    ) -> anyhow::Result<NodeId> {
        if let Some(name) = path_ident(p) {
            if let Some(id) = env.get(&name) {
                return Ok(*id);
            }
        }
        let text = path_text(&p.path);
        let span = span_of(self.file_hint, p.path.span());
        // Only an actual `DType::<variant>` path is dtype evidence. A constant or user type whose
        // name happens to end in `F32` must not affect tensor inference.
        let segments: Vec<_> = p.path.segments.iter().collect();
        if segments.len() >= 2 && segments[segments.len() - 2].ident == "DType" {
            let dtype = AbstractDtype::parse(&segments[segments.len() - 1].ident.to_string());
            return Ok(self.add_node(
                NodeKind::Literal { text },
                span,
                dtype,
                GradState::Frozen,
                None,
            ));
        }
        Ok(self.add_node(
            NodeKind::Unknown {
                reason: format!("unresolved path {text}"),
            },
            span,
            AbstractDtype::Unknown,
            GradState::Unknown,
            None,
        ))
    }

    fn expr_block(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        block: &syn::Block,
    ) -> anyhow::Result<Option<NodeId>> {
        let mut nested = env.clone();
        let mut last = None;
        for stmt in &block.stmts {
            last = self.stmt(&mut nested, stmt)?;
        }
        // Write back assignments to names that existed in the outer env.
        for (k, v) in nested {
            if env.contains_key(&k) {
                env.insert(k, v);
            }
        }
        Ok(last)
    }

    fn expr_if(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        i: &syn::ExprIf,
    ) -> anyhow::Result<NodeId> {
        let _cond = self.expr(env, &i.cond)?;
        let original_env = env.clone();
        let mut then_env = env.clone();
        let then_v = self.expr_block(&mut then_env, &i.then_branch)?;
        let mut else_env = original_env.clone();
        let else_v = match &i.else_branch {
            Some((_, e)) => Some(self.expr(&mut else_env, e)?),
            None => None,
        };
        let span = span_of(self.file_hint, i.if_token.span);
        self.merge_branch_envs(env, &original_env, &[then_env, else_env], span);
        let phi = self.add_node(
            NodeKind::Phi,
            span,
            AbstractDtype::Unknown,
            GradState::Unknown,
            None,
        );
        let mut dtypes = Vec::new();
        let mut grads = Vec::new();
        if let Some(t) = then_v {
            self.add_edge(t, phi, EdgeKind::Control, Some("then".into()));
            dtypes.push(self.graph.node(t).dtype);
            grads.push(self.graph.node(t).grad);
        }
        if let Some(e) = else_v {
            self.add_edge(e, phi, EdgeKind::Control, Some("else".into()));
            dtypes.push(self.graph.node(e).dtype);
            grads.push(self.graph.node(e).grad);
        }
        self.graph.nodes[phi.0].dtype = join_dtypes(&dtypes);
        self.graph.nodes[phi.0].grad = join_grads(&grads);
        Ok(phi)
    }

    fn merge_branch_envs(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        original: &HashMap<String, NodeId>,
        branches: &[HashMap<String, NodeId>],
        span: SrcSpan,
    ) {
        for (name, original_id) in original {
            let values = branches
                .iter()
                .map(|branch| branch.get(name).copied().unwrap_or(*original_id))
                .collect::<Vec<_>>();
            if values.iter().all(|value| *value == values[0]) {
                env.insert(name.clone(), values[0]);
                continue;
            }
            let phi = self.add_node(
                NodeKind::Phi,
                span,
                join_dtypes(
                    &values
                        .iter()
                        .map(|value| self.graph.node(*value).dtype)
                        .collect::<Vec<_>>(),
                ),
                join_grads(
                    &values
                        .iter()
                        .map(|value| self.graph.node(*value).grad)
                        .collect::<Vec<_>>(),
                ),
                None,
            );
            for (index, value) in values.iter().enumerate() {
                self.add_edge(
                    *value,
                    phi,
                    EdgeKind::Control,
                    Some(format!("branch{index}")),
                );
            }
            let types = values
                .iter()
                .filter_map(|value| self.node_type(*value))
                .collect::<Vec<_>>();
            if let Some(first) = types.first().copied() {
                if types.len() == values.len() && types.iter().all(|value| *value == first) {
                    self.set_node_type(phi, first.to_string());
                }
            }
            env.insert(name.clone(), phi);
        }
    }

    fn expr_match(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        m: &syn::ExprMatch,
    ) -> anyhow::Result<NodeId> {
        let _scrut = self.expr(env, &m.expr)?;
        let span = span_of(self.file_hint, m.match_token.span);
        let phi = self.add_node(
            NodeKind::Phi,
            span,
            AbstractDtype::Unknown,
            GradState::Unknown,
            None,
        );
        let mut dtypes = Vec::new();
        let mut grads = Vec::new();
        for arm in &m.arms {
            let mut arm_env = env.clone();
            let v = self.expr(&mut arm_env, &arm.body)?;
            self.add_edge(v, phi, EdgeKind::Control, Some("arm".into()));
            dtypes.push(self.graph.node(v).dtype);
            grads.push(self.graph.node(v).grad);
        }
        self.graph.nodes[phi.0].dtype = join_dtypes(&dtypes);
        self.graph.nodes[phi.0].grad = join_grads(&grads);
        Ok(phi)
    }

    fn binary(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        b: &syn::ExprBinary,
    ) -> anyhow::Result<NodeId> {
        let left = self.expr(env, &b.left)?;
        let right = self.expr(env, &b.right)?;
        let op = match b.op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => "div",
            _ => {
                let span = span_of(self.file_hint, span_binop(&b.op));
                let id = self.add_node(
                    NodeKind::Call {
                        callee: "binary".into(),
                    },
                    span,
                    AbstractDtype::Unknown,
                    GradState::Unknown,
                    None,
                );
                self.add_edge(left, id, EdgeKind::Data, Some("lhs".into()));
                self.add_edge(right, id, EdgeKind::Data, Some("rhs".into()));
                return Ok(id);
            }
        };
        let span = span_of(self.file_hint, span_binop(&b.op));
        if is_scalar_literal(&b.left) ^ is_scalar_literal(&b.right) {
            let tensor = if is_scalar_literal(&b.left) {
                right
            } else {
                left
            };
            let id = self.add_node(
                NodeKind::Call {
                    callee: op.to_string(),
                },
                span,
                self.graph.node(tensor).dtype,
                self.graph.node(tensor).grad,
                self.graph.node(tensor).shape.clone(),
            );
            self.set_node_type(id, "Tensor");
            self.add_edge(left, id, EdgeKind::Data, Some("lhs".into()));
            self.add_edge(right, id, EdgeKind::Data, Some("rhs".into()));
            return Ok(id);
        }
        Ok(self.apply_op(op, span, &[left, right], None))
    }

    fn method_call(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        m: &syn::ExprMethodCall,
    ) -> anyhow::Result<NodeId> {
        let receiver = self.expr(env, &m.receiver)?;
        let mut args = Vec::with_capacity(m.args.len());
        for a in &m.args {
            args.push(self.expr(env, a)?);
        }
        let method = m.method.to_string();
        let span = span_of(self.file_hint, m.method.span());

        if method == "dtype"
            && self
                .node_type(receiver)
                .is_some_and(is_candle_tensor_receiver)
        {
            let dtype = self.graph.node(receiver).dtype;
            return Ok(self.add_node(
                NodeKind::Literal {
                    text: "dtype()".into(),
                },
                span,
                dtype,
                GradState::Frozen,
                None,
            ));
        }

        if self
            .node_type(receiver)
            .is_some_and(|receiver_type| receiver_type.contains("VarBuilder"))
        {
            if matches!(method.as_str(), "pp" | "push_prefix" | "clone") {
                let dtype = self.graph.node(receiver).dtype;
                let id = self.add_node(
                    NodeKind::Call {
                        callee: method.clone(),
                    },
                    span,
                    dtype,
                    GradState::Frozen,
                    None,
                );
                self.set_node_type(id, "VarBuilder");
                self.add_edge(receiver, id, EdgeKind::Data, Some("self".into()));
                for (index, arg) in args.iter().enumerate() {
                    self.add_edge(*arg, id, EdgeKind::Data, Some(format!("arg{index}")));
                }
                return Ok(id);
            }
            if matches!(
                method.as_str(),
                "get"
                    | "get_with_hints"
                    | "get_with_hints_dtype"
                    | "get_unchecked"
                    | "get_unchecked_dtype"
                    | "get_with_dtype"
            ) {
                let dtype = self.graph.node(receiver).dtype;
                if dtype.is_known() {
                    let id = self.add_node(
                        NodeKind::Call {
                            callee: method.clone(),
                        },
                        span,
                        dtype,
                        GradState::Trainable,
                        None,
                    );
                    self.set_node_type(id, "Tensor");
                    self.add_edge(receiver, id, EdgeKind::Data, Some("builder".into()));
                    for (index, arg) in args.iter().enumerate() {
                        self.add_edge(*arg, id, EdgeKind::Data, Some(format!("arg{index}")));
                    }
                    return Ok(id);
                }
            }
        }

        // Inherent crate-local method on a known type? Only when receiver looks like `self`
        // or we can recover a type name — keep interprocedural for `Self::` style via func_call.
        // For `self.foo(...)` try type of `self` from env naming: look up methods by scanning.
        if let Some(ret) = self.try_interproc_method(&method, receiver, &args, span)? {
            return Ok(ret);
        }

        if let Some(receiver_type) = self.node_type(receiver).map(str::to_string) {
            let effect = op_semantics::lookup_method(
                &receiver_type,
                &method,
                self.candle_nn_version.as_deref(),
            );
            if !matches!(effect.dtype, DtypeRule::Unknown)
                || !matches!(effect.grad, GradFlow::Unknown)
            {
                let label = effect.name.clone();
                let result = self.apply_effect(&label, effect, span, &args, None);
                if !is_candle_tensor_receiver(&receiver_type) {
                    self.add_edge(receiver, result, EdgeKind::Data, Some("module".into()));
                }
                return Ok(result);
            }
        }

        let mut operands = vec![receiver];
        operands.extend(args.iter().copied());
        if !self
            .node_type(receiver)
            .is_some_and(is_candle_tensor_receiver)
        {
            let id = self.add_node(
                NodeKind::Call {
                    callee: method.clone(),
                },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            );
            self.add_edge(receiver, id, EdgeKind::Data, Some("self".into()));
            for (index, arg) in args.iter().enumerate() {
                self.add_edge(*arg, id, EdgeKind::Data, Some(format!("arg{index}")));
            }
            self.diagnose(
                span,
                format!(
                    "receiver type for method `{method}` is not proven to be Tensor or a \
                     crate-local type; transfer semantics left unknown"
                ),
            );
            return Ok(id);
        }
        let explicit = if method == "to_dtype" {
            args.first().map(|id| self.graph.node(*id).dtype)
        } else {
            None
        };
        Ok(self.apply_op(&method, span, &operands, explicit))
    }

    fn try_interproc_method(
        &mut self,
        method: &str,
        receiver: NodeId,
        args: &[NodeId],
        span: SrcSpan,
    ) -> anyhow::Result<Option<NodeId>> {
        let Some(receiver_type) = self.node_type(receiver).map(ToString::to_string) else {
            let count = self
                .krate
                .all_methods()
                .filter(|func| func.fn_name == method)
                .count();
            if count > 0 {
                self.diagnose(
                    span,
                    format!(
                        "method `{method}` has {count} crate-local candidate(s), but the receiver \
                         type is unknown; not inlined"
                    ),
                );
            }
            return Ok(None);
        };
        if is_candle_tensor_receiver(&receiver_type) {
            return Ok(None);
        }

        let candidates = self.krate.method_candidates(&receiver_type, method);
        let func = match candidates.as_slice() {
            [] => return Ok(None),
            [func] => *func,
            _ => {
                self.diagnose(
                    span,
                    format!(
                        "ambiguous method `{receiver_type}::{method}` ({} candidates); not inlined",
                        candidates.len()
                    ),
                );
                return Ok(None);
            }
        };

        let key = (func.type_name.clone(), func.fn_name.clone());
        if self.call_stack.contains(&key) {
            self.diagnose(
                span,
                format!("recursion guard: {}::{} already on stack", key.0, key.1),
            );
            let id = self.add_node(
                NodeKind::Call {
                    callee: format!("{}::{method}", func.type_name),
                },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            );
            self.add_edge(receiver, id, EdgeKind::Data, Some("self".into()));
            for (i, a) in args.iter().enumerate() {
                self.add_edge(*a, id, EdgeKind::Data, Some(format!("arg{i}")));
            }
            return Ok(Some(id));
        }

        self.call_stack.insert(key.clone());
        let prev_file = self.file_hint;
        let prev_module = self.module_path.clone();
        self.file_hint = func.span.file;
        self.module_path = func.module_path.clone();
        let ret = self.analyze_function(func, &func.type_name, Some(receiver), args)?;
        self.file_hint = prev_file;
        self.module_path = prev_module;
        self.call_stack.remove(&key);

        let out = ret.unwrap_or_else(|| {
            self.add_node(
                NodeKind::Call {
                    callee: format!("{}::{method}", func.type_name),
                },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            )
        });
        // Link only the callee result. Direct actual-argument edges would bypass detach/no-bwd
        // operations inside the callee and create false live-gradient paths.
        let call_node = self.add_node(
            NodeKind::Call {
                callee: format!("{}::{method}", func.type_name),
            },
            span,
            self.graph.node(out).dtype,
            self.graph.node(out).grad,
            self.graph.node(out).shape.clone(),
        );
        if let Some(return_type) = resolved_return_type(func, &func.type_name) {
            self.set_node_type(call_node, return_type);
        }
        self.add_edge(out, call_node, EdgeKind::Data, Some("return".into()));
        Ok(Some(call_node))
    }

    fn func_call(
        &mut self,
        env: &mut HashMap<String, NodeId>,
        c: &syn::ExprCall,
    ) -> anyhow::Result<NodeId> {
        let mut args = Vec::with_capacity(c.args.len());
        for a in &c.args {
            args.push(self.expr(env, a)?);
        }
        let span = match &*c.func {
            syn::Expr::Path(p) => span_of(self.file_hint, p.path.span()),
            _ => SrcSpan {
                file: self.file_hint,
                line: 0,
                col: 0,
            },
        };

        // Path call: Type::method or free function or candle_nn::loss::cross_entropy
        if let syn::Expr::Path(p) = &*c.func {
            let source_segments: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            let segs = self
                .krate
                .resolve_import_path(&self.module_path, &source_segments);
            let last = segs.last().cloned().unwrap_or_default();

            if matches!(
                last.as_str(),
                "from_varmap" | "from_mmaped_safetensors" | "from_buffered_safetensors"
            ) && segs.iter().any(|segment| segment == "VarBuilder")
            {
                let dtype = c
                    .args
                    .get(1)
                    .map(|expression| self.expr(env, expression))
                    .transpose()?
                    .map(|node| self.graph.node(node).dtype)
                    .filter(|dtype| dtype.is_known())
                    .unwrap_or(AbstractDtype::Unknown);
                let id = self.add_node(
                    NodeKind::Call {
                        callee: segs.join("::"),
                    },
                    span,
                    dtype,
                    GradState::Frozen,
                    None,
                );
                self.set_node_type(id, "VarBuilder");
                for (index, arg) in args.iter().enumerate() {
                    self.add_edge(*arg, id, EdgeKind::Data, Some(format!("arg{index}")));
                }
                return Ok(id);
            }

            if is_tensor_constructor_name(&last) && segs.iter().any(|segment| segment == "Tensor") {
                return self.tensor_constructor(&last, &args, &c.args, span);
            }

            // Inherent Type::method
            if segs.len() >= 2 {
                let type_name = normalize_qualified_segments(&segs[..segs.len() - 1]);
                let candidates = self.krate.method_candidates(&type_name, &last);
                match candidates.as_slice() {
                    [func] => return self.call_crate_fn(func, &type_name, None, &args, span),
                    [] => {}
                    _ => {
                        self.diagnose(
                            span,
                            format!(
                                "call `{type_name}::{last}` is ambiguous ({} definitions); not \
                                 inlined",
                                candidates.len()
                            ),
                        );
                    }
                }
            }

            // Free function
            let function_name = normalize_qualified_segments(&segs);
            let candidates = self.krate.function_candidates(&function_name);
            if let [func] = candidates.as_slice() {
                // Prefer candle_nn loss / op names when path mentions candle_nn — still apply
                // transfer rules on the known last segment either way.
                if op_semantics::lookup_for(&last, self.candle_nn_version.as_deref())
                    .note
                    .is_some()
                    || matches!(
                        op_semantics::lookup_for(&last, self.candle_nn_version.as_deref()).dtype,
                        DtypeRule::SameAsInputs
                            | DtypeRule::Preserve
                            | DtypeRule::Explicit
                            | DtypeRule::Fixed(_)
                    )
                {
                    // If it's a known op AND a local function, the local body wins for
                    // interprocedural detail; still record the op effect on the call node.
                }
                return self.call_crate_fn(func, "", None, &args, span);
            } else if candidates.len() > 1 {
                self.diagnose(
                    span,
                    format!(
                        "free function `{function_name}` is ambiguous ({} definitions); not inlined",
                        candidates.len()
                    ),
                );
            }

            // Known library op (candle_nn::loss::cross_entropy, etc.)
            let effect = op_semantics::lookup_for(&last, self.candle_nn_version.as_deref());
            if is_explicit_candle_path(&segs)
                && (effect.note.is_some()
                    || !matches!(effect.grad, GradFlow::Unknown)
                    || !matches!(effect.dtype, DtypeRule::Unknown))
            {
                let explicit = if last == "to_dtype" {
                    args.first().map(|id| self.graph.node(*id).dtype)
                } else {
                    None
                };
                return Ok(self.apply_op(&last, span, &args, explicit));
            }

            // Unknown external call
            let id = self.add_node(
                NodeKind::Call {
                    callee: segs.join("::"),
                },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            );
            for (i, a) in args.iter().enumerate() {
                self.add_edge(*a, id, EdgeKind::Data, Some(format!("arg{i}")));
            }
            return Ok(id);
        }

        let callee = self.expr(env, &c.func)?;
        let id = self.add_node(
            NodeKind::Call {
                callee: "call".into(),
            },
            span,
            AbstractDtype::Unknown,
            GradState::Unknown,
            None,
        );
        self.add_edge(callee, id, EdgeKind::Data, Some("callee".into()));
        for (i, a) in args.iter().enumerate() {
            self.add_edge(*a, id, EdgeKind::Data, Some(format!("arg{i}")));
        }
        Ok(id)
    }

    fn tensor_constructor(
        &mut self,
        name: &str,
        arg_nodes: &[NodeId],
        arg_exprs: &syn::punctuated::Punctuated<syn::Expr, syn::token::Comma>,
        span: SrcSpan,
    ) -> anyhow::Result<NodeId> {
        let dtype = match name {
            "zeros" | "ones" | "zeros_like" | "ones_like" => {
                let from_syn = arg_exprs
                    .iter()
                    .nth(if name.ends_with("_like") { 0 } else { 1 })
                    .and_then(dtype_from_syn_expr)
                    .filter(|dtype| dtype.is_known());
                let from_node = if name.ends_with("_like") {
                    arg_nodes
                        .first()
                        .map(|node| self.graph.node(*node).dtype)
                        .filter(|dtype| dtype.is_known())
                } else {
                    arg_nodes
                        .get(1)
                        .map(|node| self.graph.node(*node).dtype)
                        .filter(|dtype| dtype.is_known())
                };
                from_node.or(from_syn).unwrap_or(AbstractDtype::Unknown)
            }
            "from_vec" | "new" => arg_exprs
                .first()
                .and_then(dtype_from_collection_expr)
                .or_else(|| {
                    arg_nodes
                        .first()
                        .map(|node| self.graph.node(*node).dtype)
                })
                .filter(|dtype| dtype.is_known())
                .unwrap_or(AbstractDtype::Unknown),
            "arange" | "arange_step" => arg_exprs
                .first()
                .and_then(dtype_from_scalar_expr)
                .filter(|dtype| dtype.is_known())
                .unwrap_or(AbstractDtype::Unknown),
            "rand" | "randn" => arg_exprs
                .first()
                .and_then(dtype_from_scalar_expr)
                .filter(|dtype| dtype.is_known())
                .unwrap_or(AbstractDtype::Unknown),
            _ => AbstractDtype::Unknown,
        };
        let id = self.add_node(
            NodeKind::Call {
                callee: format!("Tensor::{name}"),
            },
            span,
            dtype,
            GradState::Frozen,
            None,
        );
        self.set_node_type(id, "Tensor");
        for (index, arg) in arg_nodes.iter().enumerate() {
            self.add_edge(*arg, id, EdgeKind::Data, Some(format!("arg{index}")));
        }
        Ok(id)
    }

    fn call_crate_fn(
        &mut self,
        func: &ImplFn,
        type_name: &str,
        receiver: Option<NodeId>,
        args: &[NodeId],
        span: SrcSpan,
    ) -> anyhow::Result<NodeId> {
        let key = (func.type_name.clone(), func.fn_name.clone());
        let label = if type_name.is_empty() {
            func.fn_name.clone()
        } else {
            format!("{type_name}::{}", func.fn_name)
        };

        if self.call_stack.contains(&key) {
            self.diagnose(span, format!("recursion guard: {label} already on stack"));
            let id = self.add_node(
                NodeKind::Call { callee: label },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            );
            for (i, a) in args.iter().enumerate() {
                self.add_edge(*a, id, EdgeKind::Data, Some(format!("arg{i}")));
            }
            return Ok(id);
        }

        // Also apply transfer rule when the function name itself is a known op (e.g. a
        // thin local wrapper is uncommon; known candle_nn names take the op path above).
        self.call_stack.insert(key.clone());
        let prev_file = self.file_hint;
        let prev_module = self.module_path.clone();
        self.file_hint = func.span.file;
        self.module_path = func.module_path.clone();
        let ret = self.analyze_function(func, &func.type_name, receiver, args)?;
        self.file_hint = prev_file;
        self.module_path = prev_module;
        self.call_stack.remove(&key);

        let out = ret.unwrap_or_else(|| {
            self.add_node(
                NodeKind::Call {
                    callee: label.clone(),
                },
                span,
                AbstractDtype::Unknown,
                GradState::Unknown,
                None,
            )
        });
        let call_node = self.add_node(
            NodeKind::Call { callee: label },
            span,
            self.graph.node(out).dtype,
            self.graph.node(out).grad,
            self.graph.node(out).shape.clone(),
        );
        if let Some(return_type) = resolved_return_type(func, type_name) {
            self.set_node_type(call_node, return_type);
        }
        self.add_edge(out, call_node, EdgeKind::Data, Some("return".into()));
        Ok(call_node)
    }

    /// Apply a named op transfer rule to `operands` (receiver first for methods).
    fn apply_op(
        &mut self,
        op: &str,
        span: SrcSpan,
        operands: &[NodeId],
        explicit_dtype: Option<AbstractDtype>,
    ) -> NodeId {
        let effect = op_semantics::lookup_for(op, self.candle_nn_version.as_deref());
        self.apply_effect(op, effect, span, operands, explicit_dtype)
    }

    fn apply_effect(
        &mut self,
        op: &str,
        effect: OpEffect,
        span: SrcSpan,
        operands: &[NodeId],
        explicit_dtype: Option<AbstractDtype>,
    ) -> NodeId {
        let (dtype, grad, edge_kind) = transfer(&effect, operands, explicit_dtype, |id| {
            let n = &self.graph.nodes[id.0];
            (n.dtype, n.grad)
        });
        let domain = self.result_domain(&effect, op, operands);

        if matches!(effect.dtype, DtypeRule::SameAsInputs) {
            let operand_dtypes: Vec<AbstractDtype> = operands
                .iter()
                .map(|id| self.graph.node(*id).dtype)
                .collect();
            let known: Vec<AbstractDtype> = operand_dtypes
                .iter()
                .copied()
                .filter(|d| d.is_known())
                .collect();
            if let Some((a, b)) = first_dtype_mismatch(&known) {
                let id = self.add_node_with_domain(
                    NodeKind::Call {
                        callee: op.to_string(),
                    },
                    span,
                    AbstractDtype::Unknown, // honest: conflicting inputs
                    grad,
                    None,
                    domain,
                );
                self.graph.dtype_conflicts.push(DtypeConflict {
                    edge_or_node: id,
                    op: op.to_string(),
                    left: a,
                    right: b,
                    span,
                    message: format!("{op} requires same dtype, got {a} vs {b}"),
                });
                for (i, &src) in operands.iter().enumerate() {
                    let label = operand_label(op, i, operands.len());
                    self.add_edge(src, id, edge_kind, Some(label));
                }
                if effect.is_loss {
                    self.graph.loss_nodes.push(id);
                }
                self.set_node_type(id, "Tensor");
                self.finish_numeric_call(id, op, &effect, operands, span);
                return id;
            }
            if let Some(known_dtype) = known.first().copied().filter(|_| {
                operand_dtypes
                    .iter()
                    .any(|dtype| !matches!(dtype, AbstractDtype::Unknown))
                    && operand_dtypes
                        .iter()
                        .any(|dtype| matches!(dtype, AbstractDtype::Unknown))
            }) {
                let id = self.add_node_with_domain(
                    NodeKind::Call {
                        callee: op.to_string(),
                    },
                    span,
                    dtype,
                    grad,
                    None,
                    domain,
                );
                self.graph.dtype_risks.push(DtypeRisk {
                    edge_or_node: id,
                    op: op.to_string(),
                    known: known_dtype,
                    span,
                    message: format!(
                        "{op} requires matching dtypes; one operand is {known_dtype} and another is unknown"
                    ),
                });
                for (i, &src) in operands.iter().enumerate() {
                    let label = operand_label(op, i, operands.len());
                    self.add_edge(src, id, edge_kind, Some(label));
                }
                if effect.is_loss {
                    self.graph.loss_nodes.push(id);
                }
                self.set_node_type(id, "Tensor");
                self.finish_numeric_call(id, op, &effect, operands, span);
                return id;
            }
        }

        let id = self.add_node_with_domain(
            NodeKind::Call {
                callee: op.to_string(),
            },
            span,
            dtype,
            grad,
            None,
            domain,
        );
        for (i, &src) in operands.iter().enumerate() {
            let label = operand_label(op, i, operands.len());
            self.add_edge(src, id, edge_kind, Some(label));
        }
        if effect.is_loss {
            self.graph.loss_nodes.push(id);
        }
        if !matches!(effect.dtype, DtypeRule::Unknown)
            || !matches!(effect.grad, GradFlow::Unknown)
            || operands.iter().any(|operand| {
                self.node_type(*operand)
                    .is_some_and(is_candle_tensor_receiver)
            }) && !is_non_tensor_tensor_method(op)
        {
            self.set_node_type(id, "Tensor");
        }
        if let Some(note) = effect.note {
            if matches!(effect.grad, GradFlow::Unknown) {
                self.diagnose(span, note);
            }
        }
        self.finish_numeric_call(id, op, &effect, operands, span);
        id
    }

    fn finish_numeric_call(
        &mut self,
        id: NodeId,
        op: &str,
        effect: &OpEffect,
        operands: &[NodeId],
        span: SrcSpan,
    ) {
        self.record_numeric_effects(id, op, effect, operands, span);
        if !self.expanding_library_body {
            if let Some(body) = library_body(op, self.candle_nn_version.as_deref()) {
                self.expand_library_body(id, body, operands, span);
            }
        }
    }

    /// Expand an audited library body into synthetic ops judged by the same domain pass.
    fn expand_library_body(
        &mut self,
        outer: NodeId,
        body: &LibraryBody,
        args: &[NodeId],
        span: SrcSpan,
    ) {
        self.expanding_library_body = true;
        self.expansion_cite = Some(body.cite);
        let mut vals: Vec<NodeId> = Vec::with_capacity(body.steps.len());
        for step in body.steps {
            let id = match *step {
                BodyAtom::Arg(index) => args.get(index).copied().unwrap_or_else(|| {
                    self.add_node(
                        NodeKind::Unknown {
                            reason: format!("missing library body arg {index}"),
                        },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Unknown,
                        None,
                    )
                }),
                BodyAtom::Assume { src, domain } => {
                    let src = vals[src as usize];
                    let id = self.add_node_with_domain(
                        NodeKind::Call {
                            callee: "library_domain_assume".into(),
                        },
                        span,
                        self.graph.node(src).dtype,
                        self.graph.node(src).grad,
                        None,
                        domain,
                    );
                    self.add_edge(src, id, EdgeKind::Data, Some("assume".into()));
                    self.set_node_type(id, "Tensor");
                    id
                }
                BodyAtom::Unary { op, src } => {
                    let src = vals[src as usize];
                    self.apply_op(op, span, &[src], None)
                }
                BodyAtom::Binary { op, left, right } => {
                    let left = vals[left as usize];
                    let right = vals[right as usize];
                    self.apply_op(op, span, &[left, right], None)
                }
                BodyAtom::Affine { src, mul, add } => {
                    let src = vals[src as usize];
                    let mul_node = self.add_node(
                        NodeKind::Literal {
                            text: format!("{mul}"),
                        },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Frozen,
                        None,
                    );
                    let add_node = self.add_node(
                        NodeKind::Literal {
                            text: format!("{add}"),
                        },
                        span,
                        AbstractDtype::Unknown,
                        GradState::Frozen,
                        None,
                    );
                    self.apply_op("affine", span, &[src, mul_node, add_node], None)
                }
            };
            vals.push(id);
        }
        if let Some(&last) = vals.last() {
            self.add_edge(last, outer, EdgeKind::Data, Some("expanded_body".into()));
            self.graph.nodes[outer.0].domain = self.graph.node(last).domain;
        }
        // Point findings at the user call site while keeping synthetic nodes for domain facts.
        for finding in &mut self.graph.numeric_domain_violations {
            if finding.library_cite.as_deref() == Some(body.cite) {
                finding.edge_or_node = outer;
                finding.span = span;
            }
        }
        for finding in &mut self.graph.zero_times_infinity {
            if finding.library_cite.as_deref() == Some(body.cite) {
                finding.edge_or_node = outer;
                finding.span = span;
            }
        }
        self.expansion_cite = None;
        self.expanding_library_body = false;
    }

    fn record_numeric_effects(
        &mut self,
        id: NodeId,
        op: &str,
        effect: &OpEffect,
        operands: &[NodeId],
        span: SrcSpan,
    ) {
        let library_cite = self.expansion_cite.map(str::to_string);
        if let Some(operand) = required_operand(op, effect.requires, operands) {
            let producer_domain = self.graph.node(operand).domain;
            if let Some(confidence) = domain_violation(effect.requires, producer_domain) {
                let proven = matches!(confidence, DomainViolationConfidence::Proven);
                let mut message = format!(
                    "`{}` requires {:?} operand, but producer domain is {:?}{}",
                    effect.name,
                    effect.requires,
                    producer_domain,
                    if proven {
                        " with no discharging guard"
                    } else {
                        " (producer domain unknown)"
                    }
                );
                if let Some(cite) = &library_cite {
                    message.push_str(&format!(" [expanded from {cite}]"));
                }
                self.graph
                    .numeric_domain_violations
                    .push(NumericDomainViolation {
                        edge_or_node: id,
                        op: effect.name.clone(),
                        requires: format!("{:?}", effect.requires),
                        producer_domain: format!("{producer_domain:?}"),
                        proven,
                        impact: NumericImpact::LocalOnly,
                        span,
                        message,
                        library_cite: library_cite.clone(),
                    });
            }
        }

        let bare = op.rsplit("::").next().unwrap_or(op);
        if matches!(bare, "mul" | "broadcast_mul") && operands.len() >= 2 {
            if let Some(mut message) =
                zero_times_infinity_message(&self.graph, operands[0], operands[1])
            {
                if let Some(cite) = &library_cite {
                    message.push_str(&format!(" [expanded from {cite}]"));
                }
                self.graph.zero_times_infinity.push(ZeroTimesInfinity {
                    edge_or_node: id,
                    impact: NumericImpact::LocalOnly,
                    span,
                    message,
                    library_cite,
                });
            }
        }
    }

    fn result_domain(&self, effect: &OpEffect, op: &str, operands: &[NodeId]) -> NumericDomain {
        let bare = op.rsplit("::").next().unwrap_or(op);
        if bare == "affine" {
            let operand = operands
                .first()
                .map(|id| self.graph.node(*id).domain)
                .unwrap_or(NumericDomain::Unknown);
            // Method form: receiver, mul, add — epsilon guard needs mul > 0 and add > 0.
            let mul = operands
                .get(1)
                .and_then(|id| literal_f64(&self.graph.node(*id).kind));
            let add = operands
                .get(2)
                .and_then(|id| literal_f64(&self.graph.node(*id).kind));
            return affine_domain(operand, mul, add);
        }
        match bare {
            "mul" | "broadcast_mul" | "add" | "broadcast_add" if operands.len() >= 2 => {
                op_semantics::join_domain(
                    self.graph.node(operands[0]).domain,
                    self.graph.node(operands[1]).domain,
                )
            }
            _ => effect.domain,
        }
    }
}

fn required_operand(op: &str, requires: DomainRequirement, operands: &[NodeId]) -> Option<NodeId> {
    if matches!(requires, DomainRequirement::None) || operands.is_empty() {
        return None;
    }
    let bare = op.rsplit("::").next().unwrap_or(op);
    // For division the non-zero requirement applies to the divisor.
    if matches!(bare, "div" | "broadcast_div") && operands.len() >= 2 {
        return Some(operands[1]);
    }
    Some(operands[0])
}

fn literal_f64(kind: &NodeKind) -> Option<f64> {
    let NodeKind::Literal { text } = kind else {
        return None;
    };
    let trimmed = text.trim().trim_end_matches('_');
    let trimmed = trimmed
        .strip_suffix("f64")
        .or_else(|| trimmed.strip_suffix("f32"))
        .or_else(|| trimmed.strip_suffix("f16"))
        .unwrap_or(trimmed);
    trimmed.parse::<f64>().ok()
}

fn is_undischarged_log(graph: &ExprGraph, node: NodeId) -> bool {
    let node = graph.node(node);
    let NodeKind::Call { callee } = &node.kind else {
        return false;
    };
    let bare = callee.rsplit("::").next().unwrap_or(callee.as_str());
    if bare != "log" {
        return false;
    }
    let Some(operand) = graph
        .edges
        .iter()
        .find(|edge| edge.to == node.id && edge.label.as_deref() != Some("module"))
        .map(|edge| edge.from)
    else {
        return false;
    };
    domain_violation(
        DomainRequirement::StrictlyPositive,
        graph.node(operand).domain,
    )
    .is_some()
}

fn zero_times_infinity_message(graph: &ExprGraph, left: NodeId, right: NodeId) -> Option<String> {
    let left_zero = domain_includes_zero(graph.node(left).domain);
    let right_zero = domain_includes_zero(graph.node(right).domain);
    let left_log = is_undischarged_log(graph, left);
    let right_log = is_undischarged_log(graph, right);
    if (left_zero && right_log) || (right_zero && left_log) {
        Some(
            "multiply combines a domain that includes 0 with an undischarged `log`, \
             which yields `0 * -inf = NaN` rather than a loud `-inf` loss"
                .to_string(),
        )
    } else {
        None
    }
}

fn transfer(
    effect: &OpEffect,
    operands: &[NodeId],
    explicit: Option<AbstractDtype>,
    lookup: impl Fn(NodeId) -> (AbstractDtype, GradState),
) -> (AbstractDtype, GradState, EdgeKind) {
    let dtypes: Vec<AbstractDtype> = operands.iter().map(|id| lookup(*id).0).collect();
    let grads: Vec<GradState> = operands.iter().map(|id| lookup(*id).1).collect();

    let dtype = match effect.dtype {
        DtypeRule::Preserve => dtypes.first().copied().unwrap_or(AbstractDtype::Unknown),
        DtypeRule::SameAsInputs => {
            let known: Vec<_> = dtypes.iter().copied().filter(|d| d.is_known()).collect();
            if known.is_empty() {
                AbstractDtype::Unknown
            } else if known.iter().all(|d| *d == known[0]) {
                known[0]
            } else {
                AbstractDtype::Unknown
            }
        }
        DtypeRule::Explicit => explicit.unwrap_or(AbstractDtype::Unknown),
        DtypeRule::Fixed(dtype) => dtype,
        DtypeRule::Unknown => AbstractDtype::Unknown,
    };

    let (grad, edge_kind) = match effect.grad {
        GradFlow::Severs => (GradState::Severed, EdgeKind::Severing),
        GradFlow::LayoutDependent => (GradState::LayoutDependent, EdgeKind::Data),
        GradFlow::Propagates => (propagate_grad(&grads), EdgeKind::Data),
        GradFlow::Unknown => (GradState::Unknown, EdgeKind::Data),
    };

    // Once severed, stay severed even if rule said propagates — handled by edge kind on inputs.
    let grad = if grads.iter().any(|g| matches!(g, GradState::Severed))
        && matches!(effect.grad, GradFlow::Propagates)
    {
        // Inputs already severed: result is severed for connectivity purposes.
        GradState::Severed
    } else {
        grad
    };

    (dtype, grad, edge_kind)
}

fn propagate_grad(grads: &[GradState]) -> GradState {
    if grads.is_empty() {
        return GradState::Unknown;
    }
    if grads.iter().any(|g| matches!(g, GradState::Severed)) {
        return GradState::Severed;
    }
    if grads
        .iter()
        .any(|g| matches!(g, GradState::LayoutDependent))
    {
        return GradState::LayoutDependent;
    }
    if grads
        .iter()
        .any(|g| matches!(g, GradState::Trainable | GradState::Differentiable))
    {
        return GradState::Differentiable;
    }
    if grads.iter().all(|g| matches!(g, GradState::Frozen)) {
        return GradState::Frozen;
    }
    GradState::Unknown
}

fn first_dtype_mismatch(known: &[AbstractDtype]) -> Option<(AbstractDtype, AbstractDtype)> {
    let first = *known.first()?;
    known
        .iter()
        .copied()
        .find(|d| *d != first)
        .map(|other| (first, other))
}

fn operand_label(op: &str, index: usize, len: usize) -> String {
    if len == 2 {
        return if index == 0 {
            "lhs".into()
        } else {
            "rhs".into()
        };
    }
    if index == 0
        && !matches!(
            op,
            "cross_entropy" | "nll" | "mse" | "huber" | "binary_cross_entropy_with_logit"
        )
    {
        // method receiver
        return "self".into();
    }
    format!("arg{index}")
}

fn join_dtypes(ds: &[AbstractDtype]) -> AbstractDtype {
    let known: Vec<_> = ds.iter().copied().filter(|d| d.is_known()).collect();
    if known.is_empty() {
        AbstractDtype::Unknown
    } else if known.iter().all(|d| *d == known[0]) {
        known[0]
    } else {
        AbstractDtype::Unknown
    }
}

fn join_grads(gs: &[GradState]) -> GradState {
    if gs.is_empty() {
        return GradState::Unknown;
    }
    let first = gs[0];
    if gs.iter().all(|g| *g == first) {
        first
    } else {
        GradState::Unknown
    }
}

fn hint_param_grad(type_text: &str, incoming: GradState) -> GradState {
    if !matches!(incoming, GradState::Unknown) {
        return incoming;
    }
    match source_type_base(type_text).as_deref() {
        Some("Var" | "candle_core::Var") => GradState::Trainable,
        _ => GradState::Unknown,
    }
}

fn source_type_base(text: &str) -> Option<String> {
    let ty = syn::parse_str::<syn::Type>(text).ok()?;
    innermost_type_name(&ty)
}

fn resolved_return_type(function: &ImplFn, owner_type: &str) -> Option<String> {
    let ty = syn::parse_str::<syn::Type>(&function.return_type).ok()?;
    let base = result_inner_base(&ty)?;
    if base == "Self" {
        Some(owner_type.to_string())
    } else {
        Some(base)
    }
}

fn result_inner_base(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Reference(reference) => result_inner_base(&reference.elem),
        syn::Type::Paren(paren) => result_inner_base(&paren.elem),
        syn::Type::Group(group) => result_inner_base(&group.elem),
        syn::Type::Path(path) => {
            let segment = path.path.segments.last()?;
            if matches!(
                segment.ident.to_string().as_str(),
                "Result" | "Option" | "Box" | "Arc"
            ) {
                let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
                    return None;
                };
                return arguments.args.iter().find_map(|argument| match argument {
                    syn::GenericArgument::Type(inner) => result_inner_base(inner),
                    _ => None,
                });
            }
            Some(
                path.path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
            )
        }
        _ => None,
    }
}

fn is_tensor_constructor_name(name: &str) -> bool {
    matches!(
        name,
        "zeros" | "ones" | "from_vec" | "new" | "arange" | "arange_step" | "rand" | "randn"
        | "zeros_like" | "ones_like"
    )
}

fn is_candle_tensor_receiver(type_name: &str) -> bool {
    matches!(
        type_name.trim_start_matches('&').trim().rsplit("::").next(),
        Some("Tensor" | "Var")
    )
}

fn is_non_tensor_tensor_method(method: &str) -> bool {
    matches!(
        method.rsplit("::").next().unwrap_or(method),
        "device"
            | "dtype"
            | "layout"
            | "rank"
            | "dims"
            | "dim"
            | "dims1"
            | "dims2"
            | "dims3"
            | "dims4"
            | "dims5"
            | "elem_count"
            | "stride"
            | "is_contiguous"
            | "to_scalar"
            | "to_vec0"
            | "to_vec1"
            | "to_vec2"
            | "to_vec3"
    )
}

fn normalize_qualified_segments(segments: &[String]) -> String {
    segments
        .iter()
        .map(String::as_str)
        .skip_while(|segment| matches!(*segment, "crate" | "self"))
        .collect::<Vec<_>>()
        .join("::")
}

fn is_explicit_candle_path(segments: &[String]) -> bool {
    matches!(
        segments.first().map(String::as_str),
        Some(
            "candle" | "candle_core" | "candle_nn" | "candle_transformers" | "nn" | "ops" | "loss"
        )
    )
}

fn is_scalar_literal(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Lit(literal) => matches!(
            literal.lit,
            syn::Lit::Int(_)
                | syn::Lit::Float(_)
                | syn::Lit::Bool(_)
                | syn::Lit::Byte(_)
                | syn::Lit::Char(_)
        ),
        syn::Expr::Unary(unary) => is_scalar_literal(&unary.expr),
        syn::Expr::Paren(paren) => is_scalar_literal(&paren.expr),
        syn::Expr::Group(group) => is_scalar_literal(&group.expr),
        _ => false,
    }
}

fn innermost_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Reference(reference) => innermost_type_name(&reference.elem),
        syn::Type::Paren(paren) => innermost_type_name(&paren.elem),
        syn::Type::Group(group) => innermost_type_name(&group.elem),
        syn::Type::Path(path) => {
            let segment = path.path.segments.last()?;
            if let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments {
                if let Some(inner) = arguments.args.iter().find_map(|argument| match argument {
                    syn::GenericArgument::Type(inner) => Some(inner),
                    _ => None,
                }) {
                    return innermost_type_name(inner);
                }
            }
            Some(
                path.path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
            )
        }
        syn::Type::Tuple(tuple) if tuple.elems.len() == 1 => {
            tuple.elems.first().and_then(innermost_type_name)
        }
        _ => None,
    }
}

fn resolve_entrypoint<'a>(
    krate: &'a Crate,
    entrypoint: &str,
) -> anyhow::Result<(&'a ImplFn, String)> {
    let function_candidates = krate.function_candidates(entrypoint);
    match function_candidates.as_slice() {
        [func] => return Ok((func, String::new())),
        [] => {}
        _ => {
            return Err(anyhow::anyhow!(
                "entrypoint `{entrypoint}` is ambiguous ({} free functions)",
                function_candidates.len()
            ))
        }
    }
    if let Some((ty, method)) = entrypoint.rsplit_once("::") {
        let candidates = krate.method_candidates(ty, method);
        return match candidates.as_slice() {
            [func] => Ok((func, ty.to_string())),
            [] => Err(anyhow::anyhow!("entrypoint `{entrypoint}` not found")),
            _ => Err(anyhow::anyhow!(
                "entrypoint `{entrypoint}` is ambiguous ({} methods); select active Cargo cfg",
                candidates.len()
            )),
        };
    }
    // Bare method name unique across all loaded impls?
    let hits: Vec<_> = krate
        .all_methods()
        .filter(|func| func.fn_name == entrypoint)
        .collect();
    match hits.len() {
        1 => {
            let func = hits[0];
            Ok((func, func.qualified_type_name.clone()))
        }
        0 => Err(anyhow::anyhow!("entrypoint `{entrypoint}` not found")),
        _ => Err(anyhow::anyhow!(
            "entrypoint `{entrypoint}` is ambiguous; use Type::method"
        )),
    }
}

fn bind_pat(env: &mut HashMap<String, NodeId>, pat: &syn::Pat, value: NodeId) {
    match pat {
        syn::Pat::Ident(id) => {
            env.insert(id.ident.to_string(), value);
        }
        syn::Pat::Type(t) => bind_pat(env, &t.pat, value),
        syn::Pat::Reference(r) => bind_pat(env, &r.pat, value),
        syn::Pat::Tuple(t) => {
            // Without splitting the value node, bind each subpat to the same node (honest Unknown
            // would be worse for simple `(a, b) = …` where we lack tuple projection).
            for p in &t.elems {
                bind_pat(env, p, value);
            }
        }
        syn::Pat::TupleStruct(t) => {
            for p in &t.elems {
                bind_pat(env, p, value);
            }
        }
        syn::Pat::Slice(s) => {
            for p in &s.elems {
                bind_pat(env, p, value);
            }
        }
        syn::Pat::Wild(_) => {}
        _ => {}
    }
}

fn path_ident(p: &syn::ExprPath) -> Option<String> {
    if p.path.segments.len() == 1 {
        Some(p.path.segments[0].ident.to_string())
    } else {
        None
    }
}

fn path_last(path: &syn::Path) -> String {
    path.segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default()
}

fn path_text(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn lit_text(lit: &syn::Lit) -> String {
    match lit {
        syn::Lit::Str(s) => s.value(),
        syn::Lit::Int(i) => i.to_string(),
        syn::Lit::Float(f) => f.to_string(),
        syn::Lit::Bool(b) => b.value.to_string(),
        _ => load::type_text(lit),
    }
}

fn span_of(file: usize, span: proc_macro2::Span) -> SrcSpan {
    load::span_of(file, span)
}

fn span_binop(op: &syn::BinOp) -> proc_macro2::Span {
    op.span()
}

fn expr_kind_name(expr: &syn::Expr) -> &'static str {
    match expr {
        syn::Expr::Array(_) => "array",
        syn::Expr::Async(_) => "async",
        syn::Expr::Cast(_) => "cast",
        syn::Expr::Index(_) => "index",
        syn::Expr::Range(_) => "range",
        syn::Expr::Struct(_) => "struct",
        syn::Expr::Repeat(_) => "repeat",
        syn::Expr::Unsafe(_) => "unsafe",
        syn::Expr::While(_) => "while",
        syn::Expr::Loop(_) => "loop",
        syn::Expr::ForLoop(_) => "for",
        syn::Expr::Break(_) => "break",
        syn::Expr::Continue(_) => "continue",
        syn::Expr::Yield(_) => "yield",
        _ => "expr",
    }
}

fn dtype_from_syn_expr(expression: &syn::Expr) -> Option<AbstractDtype> {
    let expression = strip_dataflow_expr(expression);
    if let syn::Expr::Path(path) = expression {
        let segments: Vec<_> = path.path.segments.iter().collect();
        if segments.len() >= 2 && segments[segments.len() - 2].ident == "DType" {
            return Some(AbstractDtype::parse(
                &segments[segments.len() - 1].ident.to_string(),
            ));
        }
    }
    None
}

fn dtype_from_scalar_expr(expression: &syn::Expr) -> Option<AbstractDtype> {
    match strip_dataflow_expr(expression) {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Float(value) => Some(match value.suffix() {
                "" | "f64" => AbstractDtype::F64,
                "f32" => AbstractDtype::F32,
                _ => AbstractDtype::Unknown,
            }),
            syn::Lit::Int(value) => Some(match value.suffix() {
                "" | "i64" => AbstractDtype::I64,
                "i32" => AbstractDtype::I32,
                "u32" => AbstractDtype::U32,
                "u8" => AbstractDtype::U8,
                _ => AbstractDtype::Unknown,
            }),
            _ => None,
        },
        other => dtype_from_syn_expr(other),
    }
}

fn dtype_from_collection_expr(expression: &syn::Expr) -> Option<AbstractDtype> {
    match strip_dataflow_expr(expression) {
        syn::Expr::Array(array) => array.elems.first().and_then(dtype_from_scalar_expr),
        syn::Expr::Repeat(repeat) => dtype_from_scalar_expr(&repeat.expr),
        other => dtype_from_scalar_expr(other),
    }
}

fn strip_dataflow_expr(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Group(group) => strip_dataflow_expr(&group.expr),
        syn::Expr::Paren(paren) => strip_dataflow_expr(&paren.expr),
        syn::Expr::Reference(reference) => strip_dataflow_expr(&reference.expr),
        syn::Expr::Try(value) => strip_dataflow_expr(&value.expr),
        other => other,
    }
}
