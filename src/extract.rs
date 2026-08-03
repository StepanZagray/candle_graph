//! The structure extraction pass.
//!
//! Walks a model's constructor, threading `VarBuilder` prefixes through local bindings and into
//! nested constructors, and records every parameter-registering call it finds.
//!
//! # Supported subset
//!
//! This is a restricted-dialect analyzer, not a Rust compiler. It understands:
//!
//! * `vb.pp("lit")`, `vb.pp(format!("stem.{i}"))`, `set_prefix`, `root`, and chains of those
//! * multiple `VarBuilder` parameters per constructor, tracked as distinct namespaces
//! * `let` bindings whose initializer resolves to a `VarBuilder`
//! * known candle-nn constructors (see [`crate::known`]) and raw `vb.get*` calls
//! * `Type::method(.., vb_expr, ..)` calls into inherent methods of crate-local structs
//! * `cond.then(|| vb.pp(..))`, the idiomatic optional-builder form
//! * struct literals of crate-local types, as grouping nodes
//! * `for` loops and `(a..b).map(|i| ..)` closures, recorded as repeats
//! * `if` / `match` / `Option` branches, whose contents are marked conditional
//!
//! Anything else becomes a [`Diagnostic`] rather than a guess. In particular the analyzer never
//! assumes an unresolved call registers no parameters; it says it does not know.

use std::collections::HashMap;

use crate::ir::*;
use crate::known::{self, ParamKind, PrefixOp};
use crate::load::{Container, Crate};

/// Guards against pathological or cyclic model definitions.
const MAX_DEPTH: usize = 32;
const MAX_INSTANCES: usize = 20_000;

/// A resolved `VarBuilder`: which namespace it belongs to, where in it we are, and whether it
/// exists at all.
///
/// Certainty rides on the builder rather than on the call site because a constructor can take
/// several builders with different certainties — a constructor may take a
/// unconditional `vb` and an `Option`al LoRA builder. Merging them at the call site would mark
/// the whole attention block conditional, which is false.
#[derive(Clone, Debug)]
struct VbVal {
    /// Name of the constructor parameter this builder ultimately came from, e.g. `base_vb`.
    root: String,
    key: Key,
    certainty: Certainty,
}

/// Lexical scope.
///
/// Closures are tracked because candle model code defines local constructor helpers
/// (`let linear = |i, o, vb| if cfg.attention_bias { nn::linear(..) } else { .. }`). Such a
/// binding shadows the candle-nn function of the same name, so matching calls against the known
/// constructor table by name alone would silently attribute the wrong parameter set — in that
/// example, reporting an unconditional `bias` that is really config-gated.
#[derive(Clone, Default)]
struct Env {
    vb: HashMap<String, VbVal>,
    closures: HashMap<String, syn::ExprClosure>,
}

impl Env {
    /// Bind a closure's parameters for an inlined call. A parameter that cannot be resolved
    /// *removes* any outer binding of the same name rather than leaving it visible: the
    /// closure parameter shadows it in real Rust, and inheriting the outer value would compute
    /// a confidently wrong prefix.
    fn bind_param(&mut self, name: String, value: Option<VbVal>) {
        match value {
            Some(val) => {
                self.vb.insert(name, val);
            }
            None => {
                self.vb.remove(&name);
            }
        }
    }
}

pub struct Extractor<'a> {
    krate: &'a Crate,
    known_candle_constructors: bool,
    out: Structure,
    defs: HashMap<(String, String), ModuleDefId>,
    sites: HashMap<(ModuleDefId, usize, usize, usize, String), ParamSiteId>,
    stack: Vec<(String, String)>,
    truncated: bool,
}

#[derive(Clone)]
struct Ctx {
    def: ModuleDefId,
    instance: ModuleInstanceId,
    /// File containing the body currently being walked. `syn` spans carry a line but not a
    /// file, so it has to be threaded down explicitly.
    file: usize,
    conditional: Option<String>,
    repeat: Option<Repeat>,
    depth: usize,
}

impl Ctx {
    fn certainty(&self) -> Certainty {
        match &self.conditional {
            Some(reason) => Certainty::Conditional(reason.clone()),
            None => Certainty::Certain,
        }
    }

    fn in_branch(&self, reason: impl Into<String>) -> Self {
        let mut next = self.clone();
        // Keep the outermost reason: it is the one a reader needs to see first.
        if next.conditional.is_none() {
            next.conditional = Some(reason.into());
        }
        next
    }
}

impl<'a> Extractor<'a> {
    pub fn new(krate: &'a Crate) -> Self {
        Self {
            krate,
            // Direct users of the extraction API opt into the analyzer's current catalog. Crate
            // discovery uses `for_candle_version` so metadata can disable stale constructor
            // assumptions.
            known_candle_constructors: true,
            out: Structure::default(),
            defs: HashMap::new(),
            sites: HashMap::new(),
            stack: Vec::new(),
            truncated: false,
        }
    }

    /// Build an extractor whose candle-nn constructor catalog is enabled only for the audited
    /// version. Raw `VarBuilder::get*` sites and crate-local constructors remain analyzable when
    /// the dependency version is absent or unsupported.
    pub fn for_candle_version(krate: &'a Crate, version: Option<&str>) -> Self {
        let mut extractor = Self::new(krate);
        extractor.known_candle_constructors =
            version.is_some_and(crate::op_semantics::is_audited_candle_version);
        extractor
    }

    pub fn run(mut self, root_type: &str, ctor: Option<&str>) -> anyhow::Result<Structure> {
        let ctor_name = match ctor {
            Some(name) => name.to_string(),
            None => self.find_entry_ctor(root_type)?,
        };

        let candidates: Vec<_> = self
            .krate
            .method_candidates(root_type, &ctor_name)
            .into_iter()
            .filter(|func| func.trait_name.is_none())
            .collect();
        let func = match candidates.as_slice() {
            [func] => *func,
            [] => {
                return Err(anyhow::anyhow!(
                    "`{root_type}::{ctor_name}` not found in crate"
                ))
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "`{root_type}::{ctor_name}` is ambiguous ({} definitions); use a \
                     module-qualified root and active Cargo cfg",
                    candidates.len()
                ))
            }
        };

        let def = self.def_id(root_type, &ctor_name, func.span);
        let primary_root = func
            .vb_params
            .first()
            .and_then(|i| func.params.get(*i))
            .cloned()
            .unwrap_or_else(|| "vb".to_string());

        let root = self.out.add_instance(
            def,
            None,
            None,
            Key::default(),
            primary_root,
            false,
            None,
            func.span,
            Certainty::Certain,
        );
        self.out.root = Some(root);

        // Every VarBuilder parameter starts at its own empty prefix in its own namespace.
        let mut env = Env::default();
        for index in &func.vb_params {
            if let Some(name) = func.params.get(*index) {
                env.vb.insert(
                    name.clone(),
                    VbVal {
                        root: name.clone(),
                        key: Key::default(),
                        certainty: Certainty::Certain,
                    },
                );
            }
        }

        let ctx = Ctx {
            def,
            instance: root,
            file: func.span.file,
            conditional: None,
            repeat: None,
            depth: 0,
        };
        self.stack.push((root_type.to_string(), ctor_name));
        let block = func.block.clone();
        self.walk_block(&block, &mut env, &ctx);
        self.stack.pop();

        if self.truncated {
            self.out.diagnose(
                SrcSpan::UNKNOWN,
                format!("analysis truncated at {MAX_INSTANCES} instances; output is incomplete"),
                None,
            );
        }
        self.out.dedupe_params();
        self.out.derive_prefixes();
        Ok(self.out)
    }

    fn find_entry_ctor(&self, root_type: &str) -> anyhow::Result<String> {
        // Prefer `new`, then `load`, so the common case needs no flag; otherwise pick the
        // alphabetically first candidate so the choice is at least deterministic.
        for preferred in ["new", "load"] {
            let candidates: Vec<_> = self
                .krate
                .method_candidates(root_type, preferred)
                .into_iter()
                .filter(|func| func.trait_name.is_none() && !func.vb_params.is_empty())
                .collect();
            match candidates.len() {
                0 => {}
                1 => return Ok(preferred.to_string()),
                count => {
                    return Err(anyhow::anyhow!(
                        "`{root_type}::{preferred}` is ambiguous ({count} active-or-cfg-gated \
                         definitions); select a module-qualified root and Cargo configuration"
                    ))
                }
            }
        }
        let mut candidates: Vec<String> = self
            .krate
            .all_methods()
            .filter(|func| {
                let owner_matches = if root_type.contains("::") {
                    func.qualified_type_name == root_type
                } else {
                    func.type_name == root_type
                };
                owner_matches && func.trait_name.is_none() && !func.vb_params.is_empty()
            })
            .map(|func| func.fn_name.clone())
            .collect();
        candidates.sort();
        candidates.dedup();
        candidates.first().cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "no constructor taking a VarBuilder found on `{root_type}`; \
                 pass --ctor to choose one explicitly"
            )
        })
    }

    fn def_id(&mut self, type_name: &str, ctor: &str, span: SrcSpan) -> ModuleDefId {
        let key = (type_name.to_string(), ctor.to_string());
        if let Some(id) = self.defs.get(&key) {
            return *id;
        }
        let id = self
            .out
            .add_def(type_name.to_string(), Some(ctor.to_string()), span);
        self.defs.insert(key, id);
        id
    }

    // ---------------------------------------------------------------- statements

    fn walk_block(&mut self, block: &syn::Block, env: &mut Env, ctx: &Ctx) {
        for stmt in &block.stmts {
            self.walk_stmt(stmt, env, ctx);
        }
    }

    fn walk_stmt(&mut self, stmt: &syn::Stmt, env: &mut Env, ctx: &Ctx) {
        match stmt {
            syn::Stmt::Local(local) => {
                let Some(init) = &local.init else { return };
                // A closure binding is recorded, not walked. Its body runs at each call site
                // with the arguments bound there; walking it here would attribute parameters to
                // whatever `vb` happened to be in scope at the definition.
                if let syn::Expr::Closure(closure) = unwrap_expr(&init.expr) {
                    if let Some(name) = binding_name(&local.pat) {
                        env.closures.insert(name, closure.clone());
                        return;
                    }
                }
                if let Some(val) = self.eval_vb(&init.expr, env, ctx) {
                    if let Some(name) = binding_name(&local.pat) {
                        env.vb.insert(name, val);
                        return;
                    }
                }
                self.walk_expr(&init.expr, env, ctx);
            }
            syn::Stmt::Expr(expr, _) => self.walk_expr(expr, env, ctx),
            syn::Stmt::Item(_) => {}
            syn::Stmt::Macro(m) => {
                if macro_cannot_register_params(&m.mac.path) {
                    return;
                }
                self.out.diagnose(
                    crate::load::span_of(ctx.file, m.mac.path.segments[0].ident.span()),
                    format!(
                        "macro `{}!` not expanded; any parameters it registers are invisible",
                        crate::load::type_text(&m.mac.path)
                    ),
                    None,
                );
            }
        }
    }

    // ---------------------------------------------------------------- expressions

    fn walk_expr(&mut self, expr: &syn::Expr, env: &mut Env, ctx: &Ctx) {
        if self.out.instances.len() > MAX_INSTANCES {
            self.truncated = true;
            return;
        }
        // A call to a local closure must be inlined before the known-constructor table is
        // consulted, because a local binding shadows the candle-nn function of the same name.
        if self.try_local_closure(expr, env, ctx) {
            return;
        }
        if self.try_param_site(expr, env, ctx) {
            return;
        }
        if self.try_free_function(expr, env, ctx) {
            return;
        }
        if self.try_submodule(expr, env, ctx, None) {
            return;
        }

        match unwrap_expr(expr) {
            syn::Expr::Struct(s) => self.walk_struct_literal(s, env, ctx),
            syn::Expr::ForLoop(f) => {
                let mut inner = ctx.clone();
                inner.repeat = Some(Repeat {
                    var: binding_name(&f.pat).unwrap_or_else(|| "_".to_string()),
                    bound: crate::load::type_text(&f.expr),
                });
                let mut scoped = env.clone();
                self.walk_block(&f.body, &mut scoped, &inner);
            }
            syn::Expr::While(w) => {
                self.walk_expr(&w.cond, env, ctx);
                let mut inner = ctx.clone();
                inner.repeat = Some(Repeat {
                    var: "_".to_string(),
                    bound: format!("while {}", crate::load::type_text(&w.cond)),
                });
                let mut scoped = env.clone();
                self.walk_block(&w.body, &mut scoped, &inner);
            }
            syn::Expr::Loop(l) => {
                let mut inner = ctx.clone();
                inner.repeat = Some(Repeat {
                    var: "_".to_string(),
                    bound: "loop".to_string(),
                });
                let mut scoped = env.clone();
                self.walk_block(&l.body, &mut scoped, &inner);
            }
            syn::Expr::If(i) => {
                let branch = ctx.in_branch(format!("if {}", crate::load::type_text(&i.cond)));
                let mut scoped = env.clone();
                self.walk_block(&i.then_branch, &mut scoped, &branch);
                if let Some((_, alt)) = &i.else_branch {
                    let mut scoped = env.clone();
                    self.walk_expr(alt, &mut scoped, &branch);
                }
            }
            syn::Expr::Match(m) => {
                let branch = ctx.in_branch(format!("match {}", crate::load::type_text(&m.expr)));
                for arm in &m.arms {
                    let mut scoped = env.clone();
                    self.walk_expr(&arm.body, &mut scoped, &branch);
                }
            }
            syn::Expr::Closure(c) => {
                // Reached without a receiver to bind from, so the parameters shadow into
                // nothing rather than picking up an unrelated outer builder.
                let mut scoped = env.clone();
                for param in &c.inputs {
                    if let Some(name) = binding_name(param) {
                        scoped.bind_param(name, None);
                    }
                }
                self.walk_expr(&c.body, &mut scoped, ctx);
            }
            syn::Expr::MethodCall(mc) => {
                // `(0..n).map(|i| Layer::new(..))` is the iterator spelling of a layer stack;
                // recording a repeat keeps such families visible.
                let mut inner = ctx.clone();
                // `cond.then(|| nn::linear(..))` gates a whole constructor, not just a
                // builder. Without this, an untaken branch's parameters are reported as
                // certain — tied embeddings can mean a separate `lm_head` does not exist.
                if matches!(mc.method.to_string().as_str(), "then" | "then_some") {
                    inner = inner.in_branch(format!(
                        "only when {}",
                        crate::load::type_text(&mc.receiver)
                    ));
                }
                // Only a genuine iteration is a repeat. `opt_vb.as_ref().map(..)` is an
                // `Option` combinator over a single value and must not be reported as a
                // layer family.
                if mc.method == "map"
                    && inner.repeat.is_none()
                    && looks_like_iteration(&mc.receiver)
                {
                    inner.repeat = Some(Repeat {
                        var: "_".to_string(),
                        bound: crate::load::type_text(&mc.receiver),
                    });
                }
                // `opt_vb.as_ref().map(|vb| Lora::new(.., vb.pp("q")))` — the closure parameter
                // shadows any outer `vb`, and its value is the receiver's builder. Binding it
                // is what puts the LoRA parameters under the LoRA builder's prefix instead of
                // the enclosing module's.
                let receiver_vb = self.eval_vb(&mc.receiver, env, &inner);
                self.walk_expr(&mc.receiver, env, &inner);
                for arg in &mc.args {
                    if let syn::Expr::Closure(closure) = unwrap_expr(arg) {
                        let mut scoped = env.clone();
                        for (index, param) in closure.inputs.iter().enumerate() {
                            if let Some(name) = binding_name(param) {
                                let value = if index == 0 {
                                    receiver_vb.clone()
                                } else {
                                    None
                                };
                                scoped.bind_param(name, value);
                            }
                        }
                        self.walk_expr(&closure.body, &mut scoped, &inner);
                    } else {
                        self.walk_expr(arg, env, &inner);
                    }
                }
            }
            other => self.walk_children(other, env, ctx),
        }
    }

    /// Inline a call to a locally bound closure, binding its parameters from the call site.
    fn try_local_closure(&mut self, expr: &syn::Expr, env: &mut Env, ctx: &Ctx) -> bool {
        let syn::Expr::Call(call) = unwrap_expr(expr) else {
            return false;
        };
        let Some(path) = call_path(&call.func) else {
            return false;
        };
        if path.len() != 1 {
            return false;
        }
        let Some(closure) = env.closures.get(&path[0]).cloned() else {
            return false;
        };

        let mut scoped = env.clone();
        // Drop the binding inside its own body so a self-referential helper cannot loop.
        scoped.closures.remove(&path[0]);
        for (param, arg) in closure.inputs.iter().zip(call.args.iter()) {
            if let Some(name) = binding_name(param) {
                let value = self.eval_vb(arg, env, ctx);
                scoped.bind_param(name, value);
            }
        }
        // Parameters the call does not supply are unbound, not inherited.
        for param in closure.inputs.iter().skip(call.args.len()) {
            if let Some(name) = binding_name(param) {
                scoped.bind_param(name, None);
            }
        }

        self.walk_expr(&closure.body, &mut scoped, ctx);
        true
    }

    /// A struct literal of a crate-local type becomes a grouping instance. It owns no builder,
    /// so its prefix is derived afterwards from its descendants rather than invented.
    fn walk_struct_literal(&mut self, s: &syn::ExprStruct, env: &mut Env, ctx: &Ctx) {
        let type_name = s.path.segments.last().map(|seg| seg.ident.to_string());
        let own_type = self.out.def(ctx.def).name.clone();

        let group = match &type_name {
            // `Self { .. }` / `Foo { .. }` inside `Foo::new` is the constructor's own return
            // value, not a nested module.
            Some(name)
                if name != "Self"
                    && *name != own_type
                    && self.krate.struct_candidates(name).len() == 1 =>
            {
                let def = self.def_id(name, "<struct literal>", span_of_expr(ctx, &s.path));
                let id = self.out.add_instance(
                    def,
                    Some(ctx.instance),
                    None,
                    Key::default(),
                    String::new(),
                    true,
                    ctx.repeat.clone(),
                    span_of_expr(ctx, &s.path),
                    ctx.certainty(),
                );
                Some((def, id))
            }
            _ => None,
        };

        let inner = match group {
            Some((def, instance)) => Ctx {
                def,
                instance,
                ..ctx.clone()
            },
            None => ctx.clone(),
        };

        for field in &s.fields {
            let name = match &field.member {
                syn::Member::Named(id) => Some(id.to_string()),
                syn::Member::Unnamed(i) => Some(i.index.to_string()),
            };
            self.walk_field_expr(&field.expr, env, &inner, name);
        }
    }

    fn walk_field_expr(
        &mut self,
        expr: &syn::Expr,
        env: &mut Env,
        ctx: &Ctx,
        field: Option<String>,
    ) {
        // An `Option` field means the module may not exist at runtime.
        let ctx = match &field {
            Some(name) if self.field_is_option(ctx, name) => {
                &ctx.in_branch(format!("Option field `{name}`"))
            }
            _ => ctx,
        };

        // Same ordering rule as `walk_expr`: a local closure shadows the candle-nn function of
        // the same name, so it has to be inlined before the constructor table is consulted.
        if self.try_local_closure(expr, env, ctx) {
            return;
        }
        if self.try_param_site(expr, env, ctx) {
            return;
        }
        if self.try_free_function(expr, env, ctx) {
            return;
        }
        if self.try_submodule(expr, env, ctx, field) {
            return;
        }
        self.walk_expr(expr, env, ctx);
    }

    fn field_is_option(&self, ctx: &Ctx, field: &str) -> bool {
        let candidates = self.krate.struct_candidates(&self.out.def(ctx.def).name);
        candidates
            .first()
            .filter(|_| candidates.len() == 1)
            .and_then(|s| s.fields.iter().find(|f| f.name == field))
            .map(|f| f.ty.container == Container::Option)
            .unwrap_or(false)
    }

    fn walk_children(&mut self, expr: &syn::Expr, env: &mut Env, ctx: &Ctx) {
        use syn::Expr as E;
        match expr {
            E::Call(c) => {
                if c.args
                    .iter()
                    .any(|arg| self.eval_vb(arg, env, ctx).is_some())
                {
                    let target = call_path(&c.func)
                        .map(|path| path.join("::"))
                        .unwrap_or_else(|| crate::load::type_text(&c.func));
                    self.out.diagnose(
                        span_of_expr(ctx, c),
                        format!(
                            "unresolved call `{target}` receives a VarBuilder; any parameters it \
                             registers are unknown"
                        ),
                        None,
                    );
                }
                for a in &c.args {
                    self.walk_expr(a, env, ctx);
                }
            }
            E::Try(t) => self.walk_expr(&t.expr, env, ctx),
            E::Reference(r) => self.walk_expr(&r.expr, env, ctx),
            E::Paren(p) => self.walk_expr(&p.expr, env, ctx),
            E::Group(g) => self.walk_expr(&g.expr, env, ctx),
            E::Block(b) => {
                let mut scoped = env.clone();
                self.walk_block(&b.block, &mut scoped, ctx);
            }
            E::Unsafe(u) => {
                let mut scoped = env.clone();
                self.walk_block(&u.block, &mut scoped, ctx);
            }
            E::Tuple(t) => {
                for e in &t.elems {
                    self.walk_expr(e, env, ctx);
                }
            }
            E::Array(a) => {
                for e in &a.elems {
                    self.walk_expr(e, env, ctx);
                }
            }
            E::Return(r) => {
                if let Some(e) = &r.expr {
                    self.walk_expr(e, env, ctx);
                }
            }
            E::Let(l) => self.walk_expr(&l.expr, env, ctx),
            E::Assign(a) => self.walk_expr(&a.right, env, ctx),
            E::Binary(b) => {
                self.walk_expr(&b.left, env, ctx);
                self.walk_expr(&b.right, env, ctx);
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------- parameter sites

    fn try_param_site(&mut self, expr: &syn::Expr, env: &mut Env, ctx: &Ctx) -> bool {
        let expr = unwrap_expr(expr);

        // `nn::linear(a, b, vb.pp("q"))`
        if let syn::Expr::Call(call) = expr {
            let Some(path) = call_path(&call.func) else {
                return false;
            };
            let Some(func) = path.last() else {
                return false;
            };
            // Defence in depth: never match a bare name against the candle-nn table while a
            // local binding of that name is in scope.
            if path.len() == 1 && env.closures.contains_key(func) {
                return false;
            }
            let resolved_path = self.krate.resolve_unambiguous_import_path(&path);
            let Some(ctor) = self
                .known_candle_constructors
                .then(|| known_constructor(resolved_path.as_slice(), func))
                .flatten()
            else {
                return false;
            };

            let span = span_of_expr(ctx, expr);
            let Some(vb) = self.resolve_vb_arg(&call.args, ctor.vb_arg, env, ctx) else {
                self.out.diagnose(
                    span,
                    format!(
                        "`{}` call whose VarBuilder argument could not be resolved to a prefix",
                        path.join("::")
                    ),
                    None,
                );
                return true;
            };

            let rnn = rnn_config(func, &call.args);
            for leaf in ctor.leaves {
                // A config that visibly opts out of the biases proves those tensors are absent,
                // so recording them at all would invent parameters.
                if leaf.kind == ParamKind::Bias && rnn.biases == Some(false) {
                    continue;
                }
                let shape = constructor_leaf_shape(func, leaf.name, &call.args);
                // A visibly default config resolves the otherwise-conditional biases to present.
                let unconditional = leaf.unconditional
                    || (leaf.kind == ParamKind::Bias && rnn.biases == Some(true));
                let leaf_certainty = combine(
                    &vb.certainty,
                    &ctx.certainty(),
                    unconditional,
                    &format!(
                        "`{func}` registers `{}` only in some configurations",
                        leaf.name
                    ),
                );
                // `LSTM::new` formats its names from the config, so an unresolved config licenses
                // only the layer/direction *family*, never the plain `_l0` spelling.
                let named_by_config = leaf.config_named && !rnn.default_names;
                let leaf_seg = if named_by_config {
                    self.out.diagnose(
                        span,
                        format!(
                            "`{func}` names `{}` from its config argument; unresolved \
                             `layer_idx`/`direction` leaves a tensor-name family",
                            leaf.name
                        ),
                        None,
                    );
                    KeySeg::Template {
                        text: config_named_family(leaf.name),
                    }
                } else {
                    KeySeg::Literal(leaf.name.to_string())
                };
                let site = self.site_for(
                    ctx,
                    span,
                    leaf.name,
                    Acquisition::Constructor {
                        func: path.join("::"),
                        cite: ctor.cite,
                    },
                    Key::default().push(leaf_seg.clone()),
                    leaf.kind,
                    shape,
                    leaf_certainty.clone(),
                );
                let key = vb.key.push(leaf_seg);
                self.out
                    .add_param(site, ctx.instance, key, vb.root.clone(), leaf_certainty);
            }
            return true;
        }

        // `vb.get((a, b), "weight")`
        if let syn::Expr::MethodCall(mc) = expr {
            let method = mc.method.to_string();
            let Some(name_arg) = known::raw_get_name_arg(&method) else {
                return false;
            };
            let Some(vb) = self.eval_vb(&mc.receiver, env, ctx) else {
                return false;
            };
            let span = span_of_expr(ctx, expr);
            let Some(name_expr) = mc.args.iter().nth(name_arg) else {
                return false;
            };
            let Some(name) = string_literal(name_expr) else {
                self.out.diagnose(
                    span,
                    format!("`{method}` with a non-literal tensor name; key is unknown"),
                    Some(vb.key.clone()),
                );
                return true;
            };

            let shape = if name_arg > 0 {
                mc.args.first().map(crate::load::type_text)
            } else {
                None
            };
            let leaf_certainty = combine(&vb.certainty, &ctx.certainty(), true, "");
            let site = self.site_for(
                ctx,
                span,
                &name,
                Acquisition::RawGet { method },
                Key::default().push_literal(&name),
                ParamKind::Raw,
                shape,
                leaf_certainty.clone(),
            );
            let key = vb.key.push_literal(&name);
            self.out
                .add_param(site, ctx.instance, key, vb.root, leaf_certainty);
            return true;
        }

        false
    }

    #[allow(clippy::too_many_arguments)]
    fn site_for(
        &mut self,
        ctx: &Ctx,
        span: SrcSpan,
        leaf: &str,
        acquisition: Acquisition,
        relative_key: Key,
        kind: ParamKind,
        shape: Option<String>,
        certainty: Certainty,
    ) -> ParamSiteId {
        // One site per source position per leaf name, so a def instantiated many times reuses
        // its sites while `weight` and `bias` from one call stay distinct.
        let cache_key = (ctx.def, span.file, span.line, span.col, leaf.to_string());
        if let Some(id) = self.sites.get(&cache_key) {
            return *id;
        }
        let id = self.out.add_site(
            ctx.def,
            acquisition,
            relative_key,
            kind,
            shape,
            span,
            certainty,
        );
        self.sites.insert(cache_key, id);
        id
    }

    // ---------------------------------------------------------------- nested modules

    /// Inline a crate-local free helper that accepts one or more `VarBuilder`s.
    ///
    /// Helpers such as `fn build_projection(vb: VarBuilder) -> Result<Linear>` are common in
    /// model crates. They do not form a named module instance on their own, so their parameter
    /// sites remain owned by the calling instance while their source file and builder bindings
    /// come from the helper body.
    fn try_free_function(&mut self, expr: &syn::Expr, env: &mut Env, ctx: &Ctx) -> bool {
        let syn::Expr::Call(call) = unwrap_expr(expr) else {
            return false;
        };
        let Some(path) = call_path(&call.func) else {
            return false;
        };
        let Some(name) = path.last().cloned() else {
            return false;
        };
        let lookup_name = normalize_qualified_path(&path);
        let candidates = self.krate.function_candidates(&lookup_name);
        let func = match candidates.as_slice() {
            [func] => *func,
            [] => return false,
            _ => {
                if candidates.iter().any(|func| !func.vb_params.is_empty()) {
                    self.out.diagnose(
                        span_of_expr(ctx, expr),
                        format!(
                            "free function `{lookup_name}` is ambiguous ({} definitions); \
                             parameters cannot be attributed safely",
                            candidates.len()
                        ),
                        None,
                    );
                    return true;
                }
                return false;
            }
        };
        if func.vb_params.is_empty() {
            return false;
        }

        let span = span_of_expr(ctx, expr);
        let stack_key = ("<free>".to_string(), name.clone());
        if ctx.depth >= MAX_DEPTH {
            self.out.diagnose(
                span,
                format!("recursion depth {MAX_DEPTH} exceeded at free function `{name}`"),
                None,
            );
            return true;
        }
        if self.stack.contains(&stack_key) {
            self.out.diagnose(
                span,
                format!("cycle through free function `{name}`; body not expanded"),
                None,
            );
            return true;
        }

        let mut helper_env = Env::default();
        let mut resolved = 0usize;
        for index in &func.vb_params {
            let Some(param) = func.params.get(*index) else {
                continue;
            };
            let Some(arg) = call.args.iter().nth(*index) else {
                self.out.diagnose(
                    span,
                    format!("free function `{name}` is missing VarBuilder argument `{param}`"),
                    None,
                );
                continue;
            };
            match self.eval_vb(arg, env, ctx) {
                Some(value) => {
                    helper_env.vb.insert(param.clone(), value);
                    resolved += 1;
                }
                None => self.out.diagnose(
                    span,
                    format!(
                        "free function `{name}` argument `{param}` is a VarBuilder whose prefix \
                         could not be resolved"
                    ),
                    None,
                ),
            }
        }
        if resolved == 0 {
            return true;
        }

        let helper_ctx = Ctx {
            file: func.span.file,
            depth: ctx.depth + 1,
            ..ctx.clone()
        };
        self.stack.push(stack_key);
        let block = func.block.clone();
        self.walk_block(&block, &mut helper_env, &helper_ctx);
        self.stack.pop();
        true
    }

    /// Follow `Type::ctor(.., vb_expr, ..)` into a crate-local inherent method.
    fn try_submodule(
        &mut self,
        expr: &syn::Expr,
        env: &mut Env,
        ctx: &Ctx,
        field: Option<String>,
    ) -> bool {
        let expr = unwrap_expr(expr);
        let syn::Expr::Call(call) = expr else {
            return false;
        };
        let Some(path) = call_path(&call.func) else {
            return false;
        };
        if path.len() < 2 {
            return false;
        }
        let ctor_name = path[path.len() - 1].clone();
        let raw_type = path[..path.len() - 1].join("::");
        let type_name = if raw_type == "Self" {
            self.out.def(ctx.def).name.clone()
        } else {
            normalize_qualified_name(&raw_type)
        };

        let candidates: Vec<_> = self
            .krate
            .method_candidates(&type_name, &ctor_name)
            .into_iter()
            .filter(|func| func.trait_name.is_none())
            .collect();
        let func = match candidates.as_slice() {
            [func] => *func,
            [] => {
                if call
                    .args
                    .iter()
                    .any(|arg| self.eval_vb(arg, env, ctx).is_some())
                {
                    self.out.diagnose(
                        span_of_expr(ctx, expr),
                        format!(
                            "unresolved constructor `{type_name}::{ctor_name}` receives a \
                             VarBuilder; its parameters are unknown"
                        ),
                        None,
                    );
                    return true;
                }
                return false;
            }
            _ => {
                self.out.diagnose(
                    span_of_expr(ctx, expr),
                    format!(
                        "constructor `{type_name}::{ctor_name}` is ambiguous ({} definitions); \
                         subtree not expanded",
                        candidates.len()
                    ),
                    None,
                );
                return true;
            }
        };
        if func.vb_params.is_empty() {
            return false;
        }

        let span = span_of_expr(ctx, expr);

        // Bind every VarBuilder parameter of the callee from its corresponding argument. A
        // constructor taking a frozen and a trainable builder needs both, and they are
        // separate namespaces.
        let mut bindings: Vec<(String, VbVal)> = Vec::new();
        let mut primary: Option<VbVal> = None;

        for index in &func.vb_params {
            let Some(arg) = call.args.iter().nth(*index) else {
                continue;
            };
            let Some(name) = func.params.get(*index) else {
                continue;
            };
            match self.eval_vb(arg, env, ctx) {
                Some(val) => {
                    // The *first* resolvable builder defines the instance. Later builders may
                    // be conditional (an optional LoRA adapter) without making the module
                    // itself conditional; their certainty stays on their own bindings.
                    if primary.is_none() {
                        primary = Some(val.clone());
                    }
                    bindings.push((name.clone(), val));
                }
                None => {
                    self.out.diagnose(
                        span,
                        format!(
                            "`{type_name}::{ctor_name}` argument `{name}` is a VarBuilder whose \
                             prefix could not be resolved; parameters under it are missing"
                        ),
                        None,
                    );
                }
            }
        }

        let Some(primary) = primary else {
            self.out.diagnose(
                span,
                format!("`{type_name}::{ctor_name}` VarBuilder arguments not resolvable"),
                None,
            );
            return true;
        };

        if ctx.depth >= MAX_DEPTH {
            self.out.diagnose(
                span,
                format!("recursion depth {MAX_DEPTH} exceeded at `{type_name}::{ctor_name}`"),
                Some(primary.key),
            );
            return true;
        }
        if self.stack.contains(&(type_name.clone(), ctor_name.clone())) {
            self.out.diagnose(
                span,
                format!("cycle through `{type_name}::{ctor_name}`; subtree not expanded"),
                Some(primary.key),
            );
            return true;
        }

        let certainty = merge(ctx.certainty(), primary.certainty.clone());
        // `Self::new` commonly delegates to `Self::new_impl`. A same-type helper used as the
        // current constructor expression is the current instance, not a nested model module.
        if raw_type == "Self" && field.is_none() {
            let mut helper_env = Env::default();
            for (name, val) in bindings {
                helper_env.vb.insert(name, val);
            }
            let helper_ctx = Ctx {
                file: func.span.file,
                conditional: match &certainty {
                    Certainty::Conditional(reason) => Some(reason.clone()),
                    _ => ctx.conditional.clone(),
                },
                depth: ctx.depth + 1,
                ..ctx.clone()
            };
            self.stack.push((type_name, ctor_name));
            let block = func.block.clone();
            self.walk_block(&block, &mut helper_env, &helper_ctx);
            self.stack.pop();
            return true;
        }

        let child_def = self.def_id(&type_name, &ctor_name, func.span);
        let child = self.out.add_instance(
            child_def,
            Some(ctx.instance),
            field,
            primary.key.clone(),
            primary.root.clone(),
            false,
            ctx.repeat.clone(),
            span,
            certainty.clone(),
        );

        let mut child_env = Env::default();
        for (name, val) in bindings {
            child_env.vb.insert(name, val);
        }
        let child_ctx = Ctx {
            def: child_def,
            instance: child,
            file: func.span.file,
            conditional: match &certainty {
                Certainty::Conditional(reason) => Some(reason.clone()),
                _ => None,
            },
            // The child's own prefix already carries any dynamic segment, so re-marking the
            // repeat inside would double-count it.
            repeat: None,
            depth: ctx.depth + 1,
        };

        self.stack.push((type_name, ctor_name));
        let block = func.block.clone();
        self.walk_block(&block, &mut child_env, &child_ctx);
        self.stack.pop();
        true
    }

    // ---------------------------------------------------------------- VarBuilder algebra

    fn resolve_vb_arg(
        &mut self,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        index: usize,
        env: &Env,
        ctx: &Ctx,
    ) -> Option<VbVal> {
        args.iter()
            .nth(index)
            .and_then(|arg| self.eval_vb(arg, env, ctx))
    }

    fn eval_vb(&mut self, expr: &syn::Expr, env: &Env, ctx: &Ctx) -> Option<VbVal> {
        let expr = unwrap_expr(expr);
        match expr {
            syn::Expr::Path(p) => {
                let name = p.path.segments.last()?.ident.to_string();
                env.vb.get(&name).cloned()
            }
            syn::Expr::MethodCall(mc) => {
                let method = mc.method.to_string();

                // `cond.then(|| vb.pp(..))` / `cond.then_some(vb)` — the idiomatic optional
                // builder. The prefix is real but its existence is conditional.
                if method == "then" || method == "then_some" {
                    let arg = mc.args.first()?;
                    let inner = match unwrap_expr(arg) {
                        syn::Expr::Closure(c) => c.body.as_ref(),
                        other => other,
                    };
                    let val = self.eval_vb(inner, env, ctx)?;
                    let reason = format!("only when {}", crate::load::type_text(&mc.receiver));
                    return Some(VbVal {
                        certainty: merge(val.certainty, Certainty::Conditional(reason)),
                        ..val
                    });
                }

                // `opt_vb.as_ref().map(|vb| vb.pp("q"))` — an optional builder threaded through
                // `Option` combinators. The inner closure body is the builder expression, with
                // the closure parameter bound to the receiver's builder.
                if method == "map" || method == "and_then" {
                    let base = self.eval_vb(&mc.receiver, env, ctx)?;
                    let arg = mc.args.first()?;
                    let syn::Expr::Closure(closure) = unwrap_expr(arg) else {
                        return None;
                    };
                    let param = closure.inputs.first().and_then(binding_name)?;
                    let mut scoped = env.clone();
                    scoped.vb.insert(param, base.clone());
                    let val = self.eval_vb(&closure.body, &scoped, ctx)?;
                    return Some(VbVal {
                        certainty: merge(val.certainty, base.certainty),
                        ..val
                    });
                }

                // These return the same logical builder with metadata or ownership adjusted.
                // For example, a model may use `vb.pp("norm").to_dtype(F32)` for batch norm.
                if matches!(
                    method.as_str(),
                    "as_ref" | "as_mut" | "clone" | "to_owned" | "to_dtype"
                ) {
                    return self.eval_vb(&mc.receiver, env, ctx);
                }

                let op = known::prefix_method(&method)?;
                let base = self.eval_vb(&mc.receiver, env, ctx)?;
                match op {
                    PrefixOp::Root => Some(VbVal {
                        key: Key::default(),
                        ..base
                    }),
                    PrefixOp::Push | PrefixOp::Replace => {
                        let arg = mc.args.first()?;
                        let (segs, seg_certainty) = self.eval_prefix_arg(arg, ctx);
                        let start = if op == PrefixOp::Replace {
                            Key::default()
                        } else {
                            base.key
                        };
                        Some(VbVal {
                            root: base.root,
                            key: start.extend(&segs),
                            certainty: merge(base.certainty, seg_certainty),
                        })
                    }
                }
            }
            syn::Expr::Call(c) => {
                let path = call_path(&c.func)?;
                if path.last().map(String::as_str) == Some("Some") && c.args.len() == 1 {
                    return self.eval_vb(&c.args[0], env, ctx);
                }
                None
            }
            _ => None,
        }
    }

    fn eval_prefix_arg(&mut self, expr: &syn::Expr, ctx: &Ctx) -> (Vec<KeySeg>, Certainty) {
        let expr = unwrap_expr(expr);

        if let Some(text) = string_literal(expr) {
            return (literal_segs(&text), Certainty::Certain);
        }

        // `format!("layers.{index}")` — the only macro interpreted, deliberately.
        if let syn::Expr::Macro(m) = expr {
            if m.mac
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .as_deref()
                == Some("format")
            {
                if let Some(segs) = format_segs(&m.mac.tokens.to_string()) {
                    return (segs, Certainty::Certain);
                }
            }
        }

        // Anything else: the prefix level exists but its text is a runtime value. Recording it
        // as dynamic keeps key arity correct instead of silently dropping a level.
        let text = crate::load::type_text(expr);
        self.out.diagnose(
            span_of_expr(ctx, expr),
            format!("prefix argument `{text}` is not a literal or `format!`; segment is dynamic"),
            None,
        );
        (vec![KeySeg::Dynamic { expr: text }], Certainty::Certain)
    }
}

// -------------------------------------------------------------------- helpers

fn combine(a: &Certainty, b: &Certainty, unconditional: bool, reason: &str) -> Certainty {
    let merged = merge(a.clone(), b.clone());
    if unconditional {
        merged
    } else {
        match merged {
            Certainty::Certain => Certainty::Conditional(reason.to_string()),
            other => other,
        }
    }
}

/// Least-certain-wins join. `Unknown` dominates `Conditional` dominates `Certain`, so nothing
/// is ever reported as more certain than its least certain ancestor.
fn merge(a: Certainty, b: Certainty) -> Certainty {
    match (&a, &b) {
        (Certainty::Unknown(_), _) => a,
        (_, Certainty::Unknown(_)) => b,
        (Certainty::Conditional(_), _) => a,
        (_, Certainty::Conditional(_)) => b,
        _ => Certainty::Certain,
    }
}

fn literal_segs(text: &str) -> Vec<KeySeg> {
    text.split('.')
        .filter(|p| !p.is_empty())
        .map(|p| KeySeg::Literal(p.to_string()))
        .collect()
}

/// Parse a `format!` invocation into key segments.
///
/// Handles the shapes that appear in candle model code: `"layers.{i}"` with inline captures and
/// `"layers.{}"` with a positional argument. Returns `None` when the template is not a plain
/// string literal, so the caller falls back to a dynamic segment.
fn format_segs(tokens: &str) -> Option<Vec<KeySeg>> {
    let tokens = tokens.trim();
    let rest = tokens.strip_prefix('"')?;
    let end = find_unescaped_quote(rest)?;
    let template = &rest[..end];
    let args: Vec<String> = rest[end + 1..]
        .trim_start()
        .trim_start_matches(',')
        .split(',')
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();

    let mut positional = args.into_iter();
    let mut segs = Vec::new();
    for part in template.split('.').filter(|p| !p.is_empty()) {
        if let Some(inner) = brace_content(part) {
            let expr = if inner.is_empty() {
                positional.next().unwrap_or_else(|| "_".to_string())
            } else {
                // `{index}` and `{index:?}` both name a capture.
                inner.split(':').next().unwrap_or(inner).to_string()
            };
            segs.push(KeySeg::Dynamic { expr });
        } else if part.contains('{') {
            // Mixed literal and placeholder in one segment, e.g. `block{i}`. Keeping the
            // template text preserves the information without pretending to resolve it.
            segs.push(KeySeg::Template {
                text: part.to_string(),
            });
        } else {
            segs.push(KeySeg::Literal(part.to_string()));
        }
    }
    Some(segs)
}

fn brace_content(part: &str) -> Option<&str> {
    let inner = part.strip_prefix('{')?.strip_suffix('}')?;
    (!inner.contains('{')).then_some(inner)
}

fn find_unescaped_quote(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Macros whose expansion cannot contain a parameter-registering expression.
///
/// Unknown/custom macros still produce a diagnostic because they may hide arbitrary model
/// construction. These built-ins and common error/logging macros only control flow, validate,
/// or emit text, so warning about invisible parameters is a false positive.
fn macro_cannot_register_params(path: &syn::Path) -> bool {
    let Some(name) = path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
    else {
        return false;
    };
    matches!(
        name.as_str(),
        "assert"
            | "assert_eq"
            | "assert_ne"
            | "debug_assert"
            | "debug_assert_eq"
            | "debug_assert_ne"
            | "bail"
            | "ensure"
            | "panic"
            | "todo"
            | "unimplemented"
            | "unreachable"
            | "print"
            | "println"
            | "eprint"
            | "eprintln"
            | "dbg"
    )
}

fn string_literal(expr: &syn::Expr) -> Option<String> {
    match unwrap_expr(expr) {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Str(s) => Some(s.value()),
            _ => None,
        },
        _ => None,
    }
}

/// Strip `?`, `&`, parens and groups, which appear constantly and carry no meaning here.
fn unwrap_expr(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Try(t) => unwrap_expr(&t.expr),
        syn::Expr::Reference(r) => unwrap_expr(&r.expr),
        syn::Expr::Paren(p) => unwrap_expr(&p.expr),
        syn::Expr::Group(g) => unwrap_expr(&g.expr),
        other => other,
    }
}

/// Dotted segments of a call target, e.g. `nn::linear` -> `["nn", "linear"]`.
fn call_path(expr: &syn::Expr) -> Option<Vec<String>> {
    match unwrap_expr(expr) {
        syn::Expr::Path(p) => Some(
            p.path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect(),
        ),
        _ => None,
    }
}

/// Only apply Candle constructor semantics when the call is visibly in the candle-nn namespace.
///
/// A last-segment-only match (`other_crate::linear`) can confidently invent parameters. Bare
/// imported functions need compiler/use resolution and are therefore left unresolved by this
/// syntax-only pass.
fn known_constructor(path: &[String], func: &str) -> Option<&'static crate::known::Constructor> {
    let namespaced = path.len() >= 2
        && matches!(
            path.first().map(String::as_str),
            Some("candle_nn" | "nn" | "candle")
        );
    namespaced.then(|| known::lookup(func)).flatten()
}

/// What a candle-nn 0.11.0 `LSTMConfig`/`GRUConfig` argument proves about the registered tensors.
///
/// Both RNN constructors read the bias presence out of the config (`b_ih_init`/`b_hh_init`,
/// rnn.rs:156-176 and :321-326), and `LSTM::new` additionally formats every tensor name from
/// `layer_idx`/`direction` (rnn.rs:139-147). Only a visibly default config licenses either fact,
/// so anything else — a local binding, a struct literal with overrides — stays unresolved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RnnConfig {
    /// `Some(true)`/`Some(false)` when the bias inits are provably present/absent.
    biases: Option<bool>,
    /// True when the config visibly leaves `layer_idx: 0` and `Direction::Forward` in place.
    default_names: bool,
}

fn rnn_config(
    func: &str,
    args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
) -> RnnConfig {
    if !matches!(func, "lstm" | "gru") {
        return RnnConfig::default();
    }
    // rnn.rs:189 / :345 — the config is the third positional argument of both constructors.
    let Some(text) = args.iter().nth(2).map(crate::load::type_text) else {
        return RnnConfig::default();
    };
    let text = text.replace(' ', "");
    // `Default::default()`, `LSTMConfig::default()` and `GRUConfig::default_no_bias()` all reduce
    // to their final segment; a struct literal or a binding does not.
    match text.rsplit("::").next().unwrap_or(&text) {
        "default()" => RnnConfig {
            biases: Some(true),
            default_names: true,
        },
        "default_no_bias()" => RnnConfig {
            biases: Some(false),
            default_names: true,
        },
        _ => RnnConfig::default(),
    }
}

/// Turn a default-config leaf name into the family its config can range over.
///
/// `weight_ih_l0` becomes `weight_ih_l{layer_idx}{direction}`, which matches every layer index
/// and both the forward and `_reverse` spellings (rnn.rs:141-147).
fn config_named_family(leaf: &str) -> String {
    match leaf.strip_suffix("l0") {
        Some(stem) => format!("{stem}l{{layer_idx}}{{direction}}"),
        None => leaf.to_string(),
    }
}

fn constructor_leaf_shape(
    func: &str,
    leaf: &str,
    args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
) -> Option<String> {
    let arg = |index: usize| args.iter().nth(index).map(crate::load::type_text);
    let pair =
        |left: Option<String>, right: Option<String>| Some(format!("({}, {})", left?, right?));
    match (func, leaf) {
        ("linear" | "linear_no_bias" | "linear_b", "weight") => pair(arg(1), arg(0)),
        ("linear" | "linear_b", "bias") => arg(1),
        ("embedding", "weight") => pair(arg(0), arg(1)),
        (
            "layer_norm" | "layer_norm_no_bias" | "rms_norm" | "batch_norm" | "group_norm",
            "weight" | "bias" | "running_mean" | "running_var",
        ) => {
            if func == "group_norm" {
                arg(1)
            } else {
                arg(0)
            }
        }
        ("prelu", "weight") => Some(format!("({}.unwrap_or(1),)", arg(0)?)),
        // rnn.rs:145-176 (LSTM, 4 gates) and :310-326 (GRU, 3 gates); arg 0 is `in_dim`, arg 1
        // `hidden_dim`. `_ih_` maps the input, `_hh_` the recurrent state.
        ("lstm" | "gru", leaf) if leaf.starts_with("weight_") || leaf.starts_with("bias_") => {
            let gates = if func == "lstm" { 4 } else { 3 };
            let rows = format!("{gates} * {}", arg(1)?);
            match (leaf.starts_with("weight_"), leaf.contains("_ih_")) {
                (true, true) => Some(format!("({rows}, {})", arg(0)?)),
                (true, false) => Some(format!("({rows}, {})", arg(1)?)),
                (false, _) => Some(format!("({rows},)")),
            }
        }
        ("conv1d" | "conv1d_no_bias", "weight") => Some(format!(
            "({}, {} / groups({}), {})",
            arg(1)?,
            arg(0)?,
            arg(3)?,
            arg(2)?
        )),
        ("conv2d" | "conv2d_no_bias", "weight") => Some(format!(
            "({}, {} / groups({}), {}, {})",
            arg(1)?,
            arg(0)?,
            arg(3)?,
            arg(2)?,
            arg(2)?
        )),
        ("conv_transpose1d" | "conv_transpose1d_no_bias", "weight") => Some(format!(
            "({}, {} / groups({}), {})",
            arg(0)?,
            arg(1)?,
            arg(3)?,
            arg(2)?
        )),
        ("conv_transpose2d" | "conv_transpose2d_no_bias", "weight") => Some(format!(
            "({}, {}, {}, {})",
            arg(0)?,
            arg(1)?,
            arg(2)?,
            arg(2)?
        )),
        (
            "conv1d"
            | "conv1d_no_bias"
            | "conv2d"
            | "conv2d_no_bias"
            | "conv_transpose1d"
            | "conv_transpose1d_no_bias"
            | "conv_transpose2d"
            | "conv_transpose2d_no_bias",
            "bias",
        ) => arg(1),
        _ => None,
    }
}

fn normalize_qualified_path(path: &[String]) -> String {
    let mut path = path;
    while matches!(path.first().map(String::as_str), Some("crate" | "self")) {
        path = &path[1..];
    }
    path.join("::")
}

fn normalize_qualified_name(name: &str) -> String {
    normalize_qualified_path(
        &name
            .split("::")
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    )
}

/// Whether an expression is plausibly an iterator, as opposed to an `Option`. Used to decide
/// whether a `.map(..)` denotes a repeated construction.
fn looks_like_iteration(expr: &syn::Expr) -> bool {
    match unwrap_expr(expr) {
        syn::Expr::Range(_) => true,
        syn::Expr::MethodCall(mc) => matches!(
            mc.method.to_string().as_str(),
            "iter"
                | "into_iter"
                | "iter_mut"
                | "enumerate"
                | "zip"
                | "take"
                | "skip"
                | "rev"
                | "filter"
                | "chain"
                | "step_by"
                | "windows"
                | "chunks"
        ),
        _ => false,
    }
}

fn binding_name(pat: &syn::Pat) -> Option<String> {
    match pat {
        syn::Pat::Ident(id) => Some(id.ident.to_string()),
        syn::Pat::Type(t) => binding_name(&t.pat),
        _ => None,
    }
}

/// `syn` spans carry a line and column but no file, so the file comes from the walk context.
fn span_of_expr<T: syn::spanned::Spanned>(ctx: &Ctx, node: &T) -> SrcSpan {
    let start = node.span().start();
    SrcSpan {
        file: ctx.file,
        line: start.line,
        col: start.column,
    }
}
