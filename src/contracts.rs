//! Conservative tensor-contract inference from qualified Rust functions.
//!
//! This pass intentionally does not try to become a Rust type checker. It starts from
//! [`crate::load::ImplFn`] signatures, follows local tensor expressions, and records only facts
//! justified by syntax or a small set of Candle operations. Unknown values stay unknown.

use std::collections::{BTreeMap, HashMap};

use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::spanned::Spanned;

use crate::load::{Crate, ImplFn};
use crate::model_ir::{
    Confidence, DeviceFact, Dimension, Evidence, EvidenceKind, LayoutFact, ShapeFact, StableId,
    TensorContract, TensorRole,
};

/// Schema identifier for the standalone contract pass.
pub const CONTRACT_SCHEMA: &str = "candle-graph/contracts/1";

/// Contract facts grouped by their fully-qualified owner function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractAnalysis {
    pub schema_version: String,
    pub functions: Vec<FunctionContracts>,
}

/// Tensor facts found in one function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionContracts {
    pub qualified_name: String,
    pub tensors: Vec<TensorContract>,
}

/// Analyze every uniquely indexed free function and inherent method.
pub fn analyze(krate: &Crate) -> ContractAnalysis {
    let mut functions: BTreeMap<String, &ImplFn> = BTreeMap::new();
    for function in krate.qualified_functions.values() {
        functions.insert(function.qualified_name.clone(), function);
    }
    for function in krate.qualified_methods.values() {
        functions.insert(function.qualified_name.clone(), function);
    }

    ContractAnalysis {
        schema_version: CONTRACT_SCHEMA.to_string(),
        functions: functions
            .into_values()
            .map(|function| analyze_function(krate, function))
            .collect(),
    }
}

/// Analyze one already-resolved function.
pub fn analyze_function(krate: &Crate, function: &ImplFn) -> FunctionContracts {
    FunctionAnalyzer::new(krate, function).run()
}

/// Analyze all definitions matching a bare or fully-qualified free-function name.
///
/// Bare-name collisions are retained. Callers that need exactly one result should supply the
/// qualified name and verify the returned length.
pub fn functions_named(krate: &Crate, name: &str) -> Vec<FunctionContracts> {
    krate
        .function_candidates(name)
        .into_iter()
        .map(|function| analyze_function(krate, function))
        .collect()
}

/// Analyze all inherent methods matching a bare or qualified owner.
pub fn methods_named(krate: &Crate, owner: &str, method: &str) -> Vec<FunctionContracts> {
    krate
        .method_candidates(owner, method)
        .into_iter()
        .map(|function| analyze_function(krate, function))
        .collect()
}

#[derive(Clone)]
struct Fact {
    shape: ShapeFact,
    dtype: String,
    device: DeviceFact,
    layout: LayoutFact,
    requires_grad: Option<bool>,
    evidence: Vec<Evidence>,
}

impl Default for Fact {
    fn default() -> Self {
        Self {
            shape: ShapeFact::default(),
            dtype: "unknown".to_string(),
            device: DeviceFact::Unknown,
            layout: LayoutFact::Unknown,
            requires_grad: None,
            evidence: Vec::new(),
        }
    }
}

struct FunctionAnalyzer<'a> {
    krate: &'a Crate,
    function: &'a ImplFn,
    facts: BTreeMap<String, (TensorRole, Fact)>,
    shapes: HashMap<String, ShapeFact>,
    output_count: usize,
}

impl<'a> FunctionAnalyzer<'a> {
    fn new(krate: &'a Crate, function: &'a ImplFn) -> Self {
        Self {
            krate,
            function,
            facts: BTreeMap::new(),
            shapes: HashMap::new(),
            output_count: 0,
        }
    }

    fn run(mut self) -> FunctionContracts {
        for (name, ty) in self
            .function
            .params
            .iter()
            .zip(self.function.param_types.iter())
        {
            if name != "self" && type_contains_tensor(ty) {
                let mut fact = Fact::default();
                fact.evidence.push(self.evidence(
                    self.function.span.line,
                    Confidence::Proven,
                    format!("parameter `{name}` has Tensor type `{ty}`"),
                ));
                self.facts.insert(name.clone(), (TensorRole::Input, fact));
            }
        }

        self.observe_dimension_bindings(&self.function.block);
        self.process_block(&self.function.block, true);

        let owner = StableId::new("function", [&self.function.qualified_name]);
        let tensors = self
            .facts
            .into_iter()
            .map(|(name, (role, fact))| TensorContract {
                id: StableId::new("tensor", [&self.function.qualified_name, &name]),
                name,
                role,
                owner_function: owner.clone(),
                parameter: None,
                shape: fact.shape,
                dtype: fact.dtype,
                device: fact.device,
                layout: fact.layout,
                requires_grad: fact.requires_grad,
                execution_phase: None,
                evidence: fact.evidence,
            })
            .collect();

        FunctionContracts {
            qualified_name: self.function.qualified_name.clone(),
            tensors,
        }
    }

    fn evidence(&self, line: usize, confidence: Confidence, detail: impl Into<String>) -> Evidence {
        let source = self
            .krate
            .files
            .get(self.function.span.file)
            .map(|file| format!("{}:{line}", file.rel));
        Evidence {
            kind: EvidenceKind::Source,
            confidence,
            source,
            detail: detail.into(),
        }
    }

    fn expr_evidence(
        &self,
        expr: &syn::Expr,
        confidence: Confidence,
        detail: impl Into<String>,
    ) -> Evidence {
        self.evidence(expr.span().start().line, confidence, detail)
    }

    fn observe_dimension_bindings(&mut self, block: &syn::Block) {
        for statement in &block.stmts {
            match statement {
                syn::Stmt::Local(local) => self.observe_dimension_local(local),
                syn::Stmt::Expr(expr, _) => self.observe_dimension_expr(expr),
                syn::Stmt::Item(_) | syn::Stmt::Macro(_) => {}
            }
        }
    }

    fn observe_dimension_expr(&mut self, expr: &syn::Expr) {
        match strip_expr(expr) {
            syn::Expr::Block(block) => self.observe_dimension_bindings(&block.block),
            syn::Expr::If(expr_if) => {
                self.observe_dimension_bindings(&expr_if.then_branch);
                if let Some((_, otherwise)) = &expr_if.else_branch {
                    self.observe_dimension_expr(otherwise);
                }
            }
            syn::Expr::ForLoop(loop_expr) => self.observe_dimension_bindings(&loop_expr.body),
            syn::Expr::While(loop_expr) => self.observe_dimension_bindings(&loop_expr.body),
            syn::Expr::Loop(loop_expr) => self.observe_dimension_bindings(&loop_expr.body),
            syn::Expr::Match(expr_match) => {
                for arm in &expr_match.arms {
                    self.observe_dimension_expr(&arm.body);
                }
            }
            _ => {}
        }
    }

    fn observe_dimension_local(&mut self, local: &syn::Local) {
        let Some(init) = &local.init else {
            return;
        };
        let Some((tensor, method, args)) = dimension_call(&init.expr) else {
            self.observe_dimension_expr(&init.expr);
            return;
        };
        if !self.facts.contains_key(&tensor) {
            return;
        }

        let pattern_rank = if method == "dims" {
            match &local.pat {
                syn::Pat::Slice(slice) => Some(slice.elems.len()),
                syn::Pat::Type(typed) => match &*typed.pat {
                    syn::Pat::Slice(slice) => Some(slice.elems.len()),
                    _ => None,
                },
                _ => None,
            }
        } else {
            dims_rank(&method)
        };
        if let Some(rank) = pattern_rank {
            let names = pattern_names(&local.pat);
            if names.len() == rank {
                let shape = ShapeFact {
                    rank: Some(rank),
                    dimensions: names.iter().map(|name| dimension(name)).collect(),
                    source_expr: Some(format!("{tensor}.{method}()")),
                };
                let evidence = self.evidence(
                    local.span().start().line,
                    Confidence::Proven,
                    format!(
                        "`{tensor}.{method}()` destructures into [{}]",
                        names.join(", ")
                    ),
                );
                if let Some((_, fact)) = self.facts.get_mut(&tensor) {
                    fact.shape = shape;
                    fact.evidence.push(evidence);
                }
            }
            return;
        }

        if method == "dim" {
            let Some(axis) = args.first().and_then(|expr| literal_usize(expr)) else {
                return;
            };
            let names = pattern_names(&local.pat);
            let Some(name) = names.first() else {
                return;
            };
            let evidence = self.evidence(
                local.span().start().line,
                Confidence::Proven,
                format!("`{name}` is axis {axis} of `{tensor}`"),
            );
            if let Some((_, fact)) = self.facts.get_mut(&tensor) {
                if fact.shape.rank.is_some_and(|rank| axis < rank) {
                    while fact.shape.dimensions.len() < fact.shape.rank.unwrap_or(0) {
                        fact.shape
                            .dimensions
                            .push(dimension(&format!("axis_{}", fact.shape.dimensions.len())));
                    }
                    fact.shape.dimensions[axis] = dimension(name);
                }
                fact.evidence.push(evidence);
            }
        }
    }

    fn process_block(&mut self, block: &syn::Block, capture_tail: bool) {
        for (index, statement) in block.stmts.iter().enumerate() {
            let is_tail = capture_tail && index + 1 == block.stmts.len();
            match statement {
                syn::Stmt::Local(local) => self.process_local(local),
                syn::Stmt::Expr(expr, semicolon) => {
                    self.process_nested(expr);
                    if is_tail
                        && semicolon.is_none()
                        && type_contains_tensor(&self.function.return_type)
                    {
                        self.record_outputs(expr);
                    }
                }
                syn::Stmt::Item(_) | syn::Stmt::Macro(_) => {}
            }
        }
    }

    fn process_nested(&mut self, expr: &syn::Expr) {
        match strip_expr(expr) {
            syn::Expr::Block(block) => self.process_block(&block.block, false),
            syn::Expr::If(expr_if) => {
                self.process_block(&expr_if.then_branch, false);
                if let Some((_, otherwise)) = &expr_if.else_branch {
                    self.process_nested(otherwise);
                }
            }
            syn::Expr::ForLoop(loop_expr) => self.process_block(&loop_expr.body, false),
            syn::Expr::While(loop_expr) => self.process_block(&loop_expr.body, false),
            syn::Expr::Loop(loop_expr) => self.process_block(&loop_expr.body, false),
            syn::Expr::Match(expr_match) => {
                for arm in &expr_match.arms {
                    self.process_nested(&arm.body);
                }
            }
            syn::Expr::Return(return_expr) => {
                if let Some(value) = &return_expr.expr {
                    self.record_outputs(value);
                }
            }
            _ => {}
        }
    }

    fn process_local(&mut self, local: &syn::Local) {
        // The initial dimension pre-pass sees tensor parameters. Re-check while walking so
        // dimensions of tensor-valued locals declared earlier in the block are also captured.
        self.observe_dimension_local(local);
        let Some(init) = &local.init else {
            return;
        };
        self.process_nested(&init.expr);

        let names = pattern_names(&local.pat);
        if names.len() != 1 {
            return;
        }
        let name = names[0].clone();

        if let Some(shape) = self.shape_expr(&init.expr) {
            self.shapes.insert(name.clone(), shape);
        }

        if let Some(mut fact) = self.infer_expr(&init.expr) {
            fact.evidence.push(self.evidence(
                local.span().start().line,
                Confidence::Proven,
                format!(
                    "local tensor `{name}` is derived from `{}`",
                    expr_text(&init.expr)
                ),
            ));
            self.facts.insert(name, (TensorRole::Activation, fact));
        }
    }

    fn record_outputs(&mut self, expr: &syn::Expr) {
        let expr = unwrap_result_expr(expr);
        if let syn::Expr::Tuple(tuple) = strip_expr(expr) {
            for element in &tuple.elems {
                self.record_output(element);
            }
        } else {
            self.record_output(expr);
        }
    }

    fn record_output(&mut self, expr: &syn::Expr) {
        let Some(mut fact) = self.infer_expr(expr) else {
            return;
        };
        let name = if self.output_count == 0 {
            "return".to_string()
        } else {
            format!("return.{}", self.output_count)
        };
        self.output_count += 1;
        fact.evidence.push(self.expr_evidence(
            expr,
            Confidence::Proven,
            format!("function returns tensor expression `{}`", expr_text(expr)),
        ));
        self.facts.insert(name, (TensorRole::Output, fact));
    }

    fn infer_expr(&self, expr: &syn::Expr) -> Option<Fact> {
        let expr = strip_expr(expr);
        match expr {
            syn::Expr::Path(path) => {
                let name = path.path.get_ident()?.to_string();
                self.facts.get(&name).map(|(_, fact)| fact.clone())
            }
            syn::Expr::MethodCall(call) => self.infer_method(call),
            syn::Expr::Call(call) => self.infer_call(call),
            syn::Expr::Binary(binary) => {
                let mut fact = self
                    .infer_expr(&binary.left)
                    .or_else(|| self.infer_expr(&binary.right))?;
                fact.evidence.push(self.expr_evidence(
                    expr,
                    Confidence::Conditional,
                    format!(
                        "binary `{}` preserves the selected tensor operand contract",
                        binary.op.to_token_stream()
                    ),
                ));
                Some(fact)
            }
            syn::Expr::Unary(unary) => self.infer_expr(&unary.expr),
            syn::Expr::Index(index) => self.infer_expr(&index.expr),
            syn::Expr::If(expr_if) => {
                let then_fact = expr_if
                    .then_branch
                    .stmts
                    .last()
                    .and_then(|stmt| match stmt {
                        syn::Stmt::Expr(expr, None) => self.infer_expr(expr),
                        _ => None,
                    });
                let else_fact = expr_if
                    .else_branch
                    .as_ref()
                    .and_then(|(_, expr)| self.infer_expr(expr));
                merge_facts(then_fact, else_fact)
            }
            syn::Expr::Match(expr_match) => expr_match
                .arms
                .iter()
                .filter_map(|arm| self.infer_expr(&arm.body))
                .reduce(merge_two_facts),
            _ => None,
        }
    }

    fn infer_method(&self, call: &syn::ExprMethodCall) -> Option<Fact> {
        let method = call.method.to_string();

        // Extraction methods return host values, not tensors.
        if method.starts_with("to_vec")
            || matches!(
                method.as_str(),
                "dims"
                    | "dims1"
                    | "dims2"
                    | "dims3"
                    | "dims4"
                    | "dims5"
                    | "dim"
                    | "dtype"
                    | "device"
                    | "rank"
                    | "elem_count"
                    | "to_scalar"
            )
        {
            return None;
        }

        let receiver_name = root_ident(&call.receiver);
        let mut fact = self.infer_expr(&call.receiver).or_else(|| {
            if matches!(
                method.as_str(),
                "forward" | "forward_t" | "forward_diff" | "apply" | "apply_t"
            ) {
                call.args.iter().find_map(|arg| self.infer_expr(arg))
            } else {
                None
            }
        })?;

        match method.as_str() {
            "to_dtype" => {
                if let Some(dtype) = call.args.first() {
                    fact.dtype = dtype_fact(dtype);
                    fact.evidence.push(self.expr_evidence(
                        &syn::Expr::MethodCall(call.clone()),
                        Confidence::Proven,
                        format!("`{method}` sets dtype to `{}`", fact.dtype),
                    ));
                }
            }
            "to_device" => {
                if let Some(device) = call.args.first() {
                    fact.device = device_fact(device);
                    fact.evidence.push(self.expr_evidence(
                        &syn::Expr::MethodCall(call.clone()),
                        Confidence::Proven,
                        format!("`to_device` sets device from `{}`", expr_text(device)),
                    ));
                }
            }
            "contiguous" => {
                fact.layout = LayoutFact::Contiguous;
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    "`contiguous` materializes contiguous layout",
                ));
            }
            "transpose" | "t" => {
                if method == "transpose" && call.args.len() == 2 {
                    if let (Some(left), Some(right)) = (
                        call.args.first().and_then(literal_usize),
                        call.args.iter().nth(1).and_then(literal_usize),
                    ) {
                        if fact.shape.rank.is_some()
                            && left < fact.shape.dimensions.len()
                            && right < fact.shape.dimensions.len()
                        {
                            fact.shape.dimensions.swap(left, right);
                        }
                    }
                } else if method == "t" && fact.shape.dimensions.len() >= 2 {
                    let last = fact.shape.dimensions.len() - 1;
                    fact.shape.dimensions.swap(last - 1, last);
                }
                fact.layout = LayoutFact::Strided;
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    format!("`{method}` produces a strided view"),
                ));
            }
            "permute" => {
                if let Some(order) = call.args.first().and_then(index_list) {
                    if order.len() == fact.shape.dimensions.len()
                        && order.iter().all(|axis| *axis < order.len())
                    {
                        let prior = fact.shape.dimensions.clone();
                        fact.shape.dimensions =
                            order.into_iter().map(|axis| prior[axis].clone()).collect();
                    }
                }
                fact.layout = LayoutFact::Strided;
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    "`permute` produces a strided view",
                ));
            }
            "narrow" => {
                if let (Some(axis), Some(length)) = (
                    call.args.first().and_then(literal_usize),
                    call.args.iter().nth(2),
                ) {
                    if axis < fact.shape.dimensions.len() {
                        fact.shape.dimensions[axis] = dimension(&expr_text(length));
                    }
                }
                fact.layout = LayoutFact::Strided;
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    "`narrow` preserves rank and creates a strided view",
                ));
            }
            "reshape" | "broadcast_as" => {
                if let Some(shape) = call.args.first().and_then(|expr| self.shape_expr(expr)) {
                    fact.shape = shape;
                }
                if method == "broadcast_as" {
                    fact.layout = LayoutFact::Strided;
                }
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    format!("`{method}` applies an explicit symbolic shape"),
                ));
            }
            "detach" | "as_detached_tensor" => {
                fact.requires_grad = Some(false);
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Proven,
                    "explicit detach disables gradient tracking",
                ));
            }
            "clone" | "map_err" | "unwrap" | "expect" | "as_ref" => {}
            "forward" | "forward_t" | "forward_diff" | "apply" | "apply_t" => {
                if let Some(name) = receiver_name {
                    fact.evidence.push(self.expr_evidence(
                        &syn::Expr::MethodCall(call.clone()),
                        Confidence::Heuristic,
                        format!(
                            "`{name}.{method}` propagates dtype/device from its first tensor input; shape remains conservative"
                        ),
                    ));
                    fact.shape = ShapeFact::default();
                    fact.layout = LayoutFact::Unknown;
                }
            }
            _ => {
                fact.evidence.push(self.expr_evidence(
                    &syn::Expr::MethodCall(call.clone()),
                    Confidence::Conditional,
                    format!("`{method}` conservatively preserves receiver metadata"),
                ));
            }
        }
        Some(fact)
    }

    fn infer_call(&self, call: &syn::ExprCall) -> Option<Fact> {
        let name = call_path_name(&call.func)?;
        let short = name.rsplit("::").next().unwrap_or(&name);

        if matches!(short, "Ok" | "Some") {
            return call.args.first().and_then(|expr| self.infer_expr(expr));
        }

        if !name.contains("Tensor") {
            return None;
        }

        let mut fact = Fact::default();
        match short {
            "zeros" | "ones" => {
                fact.shape = call
                    .args
                    .first()
                    .and_then(|expr| self.shape_expr(expr))
                    .unwrap_or_default();
                if let Some(dtype) = call.args.iter().nth(1) {
                    fact.dtype = dtype_fact(dtype);
                }
                if let Some(device) = call.args.iter().nth(2) {
                    fact.device = device_fact(device);
                }
                fact.layout = LayoutFact::Contiguous;
                fact.requires_grad = Some(false);
            }
            "rand" | "randn" => {
                fact.shape = call
                    .args
                    .iter()
                    .nth(2)
                    .and_then(|expr| self.shape_expr(expr))
                    .unwrap_or_default();
                if let Some(value) = call.args.first() {
                    fact.dtype = scalar_dtype(value).unwrap_or_else(|| "unknown".to_string());
                }
                if let Some(device) = call.args.iter().nth(3) {
                    fact.device = device_fact(device);
                }
                fact.layout = LayoutFact::Contiguous;
                fact.requires_grad = Some(false);
            }
            "from_vec" => {
                fact.shape = call
                    .args
                    .iter()
                    .nth(1)
                    .and_then(|expr| self.shape_expr(expr))
                    .unwrap_or_default();
                if let Some(values) = call.args.first() {
                    fact.dtype = collection_dtype(values).unwrap_or_else(|| "unknown".to_string());
                }
                if let Some(device) = call.args.iter().nth(2) {
                    fact.device = device_fact(device);
                }
                fact.layout = LayoutFact::Contiguous;
                fact.requires_grad = Some(false);
            }
            "new" => {
                if let Some(value) = call.args.first() {
                    fact.dtype = scalar_dtype(value)
                        .or_else(|| collection_dtype(value))
                        .unwrap_or_else(|| "unknown".to_string());
                }
                if let Some(device) = call.args.iter().nth(1) {
                    fact.device = device_fact(device);
                }
                fact.layout = LayoutFact::Contiguous;
                fact.requires_grad = Some(false);
            }
            "arange" | "arange_step" => {
                if let Some(value) = call.args.first() {
                    fact.dtype = scalar_dtype(value).unwrap_or_else(|| "unknown".to_string());
                }
                let device_index = if short == "arange" { 2 } else { 3 };
                if let Some(device) = call.args.iter().nth(device_index) {
                    fact.device = device_fact(device);
                }
                fact.shape = ShapeFact {
                    rank: Some(1),
                    dimensions: vec![dimension("range_len")],
                    source_expr: Some(expr_text(syn::Expr::Call(call.clone()))),
                };
                fact.layout = LayoutFact::Contiguous;
                fact.requires_grad = Some(false);
            }
            _ => return None,
        }
        fact.evidence.push(self.expr_evidence(
            &syn::Expr::Call(call.clone()),
            Confidence::Proven,
            format!("Tensor::{short} constructor"),
        ));
        Some(fact)
    }

    fn shape_expr(&self, expr: &syn::Expr) -> Option<ShapeFact> {
        let expr = strip_expr(expr);
        if let syn::Expr::Path(path) = expr {
            if let Some(name) = path.path.get_ident() {
                if let Some(shape) = self.shapes.get(&name.to_string()) {
                    return Some(shape.clone());
                }
            }
        }

        let dimensions: Vec<Dimension> = match expr {
            syn::Expr::Tuple(tuple) => tuple
                .elems
                .iter()
                .map(|expr| dimension(&expr_text(expr)))
                .collect(),
            syn::Expr::Array(array) => array
                .elems
                .iter()
                .map(|expr| dimension(&expr_text(expr)))
                .collect(),
            syn::Expr::Macro(mac) if mac.mac.path.is_ident("vec") => {
                let parser =
                    syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
                use syn::parse::Parser;
                parser
                    .parse2(mac.mac.tokens.clone())
                    .ok()?
                    .iter()
                    .map(|expr| dimension(&expr_text(expr)))
                    .collect()
            }
            _ => return None,
        };
        Some(ShapeFact {
            rank: Some(dimensions.len()),
            dimensions,
            source_expr: Some(expr_text(expr)),
        })
    }
}

fn type_contains_tensor(ty: &str) -> bool {
    ty.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|part| part == "Tensor")
}

fn strip_expr(mut expr: &syn::Expr) -> &syn::Expr {
    loop {
        expr = match expr {
            syn::Expr::Try(value) => &value.expr,
            syn::Expr::Await(value) => &value.base,
            syn::Expr::Paren(value) => &value.expr,
            syn::Expr::Group(value) => &value.expr,
            syn::Expr::Reference(value) => &value.expr,
            _ => return expr,
        };
    }
}

fn unwrap_result_expr(expr: &syn::Expr) -> &syn::Expr {
    let expr = strip_expr(expr);
    if let syn::Expr::Call(call) = expr {
        if matches!(call_path_name(&call.func).as_deref(), Some("Ok" | "Some")) {
            if let Some(inner) = call.args.first() {
                return strip_expr(inner);
            }
        }
    }
    expr
}

fn root_ident(expr: &syn::Expr) -> Option<String> {
    match strip_expr(expr) {
        syn::Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
        syn::Expr::MethodCall(call) => root_ident(&call.receiver),
        syn::Expr::Field(field) => root_ident(&field.base),
        _ => None,
    }
}

fn call_path_name(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = strip_expr(expr) else {
        return None;
    };
    Some(
        path.path
            .segments
            .iter()
            .map(|part| part.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
    )
}

fn dimension_call(expr: &syn::Expr) -> Option<(String, String, Vec<&syn::Expr>)> {
    let syn::Expr::MethodCall(call) = strip_expr(expr) else {
        return None;
    };
    let method = call.method.to_string();
    if matches!(method.as_str(), "unwrap" | "expect" | "map_err") {
        return dimension_call(&call.receiver);
    }
    if !matches!(
        method.as_str(),
        "dims" | "dims1" | "dims2" | "dims3" | "dims4" | "dims5" | "dim"
    ) {
        return None;
    }
    Some((
        root_ident(&call.receiver)?,
        method,
        call.args.iter().collect(),
    ))
}

fn dims_rank(method: &str) -> Option<usize> {
    match method {
        "dims1" => Some(1),
        "dims2" => Some(2),
        "dims3" => Some(3),
        "dims4" => Some(4),
        "dims5" => Some(5),
        _ => None,
    }
}

fn pattern_names(pattern: &syn::Pat) -> Vec<String> {
    match pattern {
        syn::Pat::Ident(ident) => vec![ident.ident.to_string()],
        syn::Pat::Tuple(tuple) => tuple.elems.iter().flat_map(pattern_names).collect(),
        syn::Pat::TupleStruct(tuple) => tuple.elems.iter().flat_map(pattern_names).collect(),
        syn::Pat::Slice(slice) => slice.elems.iter().flat_map(pattern_names).collect(),
        syn::Pat::Paren(paren) => pattern_names(&paren.pat),
        syn::Pat::Type(typed) => pattern_names(&typed.pat),
        syn::Pat::Reference(reference) => pattern_names(&reference.pat),
        syn::Pat::Wild(_) => vec!["_".to_string()],
        _ => Vec::new(),
    }
}

fn dimension(expr: &str) -> Dimension {
    Dimension {
        name: semantic_dimension(expr),
        expr: normalize(expr),
    }
}

fn semantic_dimension(expr: &str) -> Option<String> {
    let normalized = normalize(expr).to_ascii_lowercase();
    let label = match normalized.as_str() {
        "b" | "batch" | "batch_size" => "batch",
        "t" | "tokens" | "seq" | "seq_len" | "positions" | "l" => "tokens",
        "s" | "slots" | "num_slots" => "slots",
        "a" | "adapter_slots" | "output_slots" | "num_output_slots" => "adapter_slots",
        "d" | "dim" | "hidden" | "hidden_size" | "model_dim" | "hq" => "hidden",
        "v" | "vocab" | "vocab_size" => "vocab",
        "_" => return None,
        _ => return Some(normalized),
    };
    Some(label.to_string())
}

fn expr_text(value: impl ToTokens) -> String {
    normalize(&value.to_token_stream().to_string())
}

fn normalize(text: &str) -> String {
    text.replace(" :: ", "::")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace(" ,", ",")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn dtype_fact(expr: &syn::Expr) -> String {
    let expr = strip_expr(expr);
    if let syn::Expr::MethodCall(call) = expr {
        if call.method == "dtype" {
            if let Some(name) = root_ident(&call.receiver) {
                return format!("same_as({name})");
            }
        }
    }
    let text = expr_text(expr);
    text.rsplit("::").next().unwrap_or(&text).to_string()
}

fn device_fact(expr: &syn::Expr) -> DeviceFact {
    let expr = strip_expr(expr);
    if let syn::Expr::MethodCall(call) = expr {
        if call.method == "device" {
            if let Some(name) = root_ident(&call.receiver) {
                return DeviceFact::SameAs(name);
            }
        }
    }
    let text = expr_text(expr);
    if text.ends_with("Device::Cpu") || text == "Device::Cpu" {
        DeviceFact::Cpu
    } else if text.contains("new_cuda") {
        DeviceFact::Cuda {
            ordinal: call_last_literal(expr).map(|value| value as u32),
        }
    } else if text.contains("new_metal") {
        DeviceFact::Metal
    } else {
        DeviceFact::SameAs(text.trim_start_matches('&').to_string())
    }
}

fn call_last_literal(expr: &syn::Expr) -> Option<usize> {
    match strip_expr(expr) {
        syn::Expr::Call(call) => call.args.last().and_then(literal_usize),
        syn::Expr::MethodCall(call) => call.args.last().and_then(literal_usize),
        _ => None,
    }
}

fn scalar_dtype(expr: &syn::Expr) -> Option<String> {
    match strip_expr(expr) {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Float(value),
            ..
        }) => Some(
            match value.suffix() {
                "" | "f64" => "F64",
                "f32" => "F32",
                suffix => suffix,
            }
            .to_ascii_uppercase(),
        ),
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(value),
            ..
        }) => Some(
            match value.suffix() {
                "" | "i32" => "I32",
                "i16" => "I16",
                "u8" => "U8",
                "u32" => "U32",
                "i64" => "I64",
                suffix => suffix,
            }
            .to_ascii_uppercase(),
        ),
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Bool(_),
            ..
        }) => None,
        _ => None,
    }
}

fn collection_dtype(expr: &syn::Expr) -> Option<String> {
    match strip_expr(expr) {
        syn::Expr::Array(array) => array.elems.first().and_then(scalar_dtype),
        syn::Expr::Macro(mac) if mac.mac.path.is_ident("vec") => {
            use syn::parse::Parser;
            let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
            parser
                .parse2(mac.mac.tokens.clone())
                .ok()?
                .first()
                .and_then(scalar_dtype)
        }
        _ => None,
    }
}

fn literal_usize(expr: &syn::Expr) -> Option<usize> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(value),
        ..
    }) = strip_expr(expr)
    else {
        return None;
    };
    value.base10_parse().ok()
}

fn index_list(expr: &syn::Expr) -> Option<Vec<usize>> {
    match strip_expr(expr) {
        syn::Expr::Array(array) => array.elems.iter().map(literal_usize).collect(),
        syn::Expr::Tuple(tuple) => tuple.elems.iter().map(literal_usize).collect(),
        _ => None,
    }
}

fn merge_facts(left: Option<Fact>, right: Option<Fact>) -> Option<Fact> {
    match (left, right) {
        (Some(left), Some(right)) => Some(merge_two_facts(left, right)),
        (Some(mut fact), None) | (None, Some(mut fact)) => {
            fact.shape = ShapeFact::default();
            fact.dtype = "unknown".to_string();
            fact.device = DeviceFact::Unknown;
            fact.layout = LayoutFact::Unknown;
            fact.requires_grad = None;
            for evidence in &mut fact.evidence {
                evidence.confidence = Confidence::Conditional;
            }
            Some(fact)
        }
        (None, None) => None,
    }
}

fn merge_two_facts(mut left: Fact, right: Fact) -> Fact {
    if left.shape != right.shape {
        left.shape = ShapeFact::default();
    }
    if left.dtype != right.dtype {
        left.dtype = "unknown".to_string();
    }
    if left.device != right.device {
        left.device = DeviceFact::Unknown;
    }
    if left.layout != right.layout {
        left.layout = LayoutFact::Unknown;
    }
    if left.requires_grad != right.requires_grad {
        left.requires_grad = None;
    }
    left.evidence.extend(right.evidence);
    left
}
