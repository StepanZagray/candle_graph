//! The analysis IR.
//!
//! Two layers, deliberately separated: a *structure* layer, which is what this milestone
//! builds, and a *dataflow* layer, which will hang off the same arenas later.
//!
//! Four splits keep the two compatible, and all four exist because collapsing them is
//! expensive to undo:
//!
//! * [`ModuleDef`] (a Rust type) vs [`ModuleInstance`] (that type built at one `VarBuilder`
//!   prefix). One `SelfAttention` def yields 28 instances in a 28-layer model.
//! * [`ParamSite`] (a `vb.get`/constructor call in source) vs [`Param`] (the tensor that call
//!   produces at one instance). One site under a loop yields many parameters.
//! * Struct containment vs prefix nesting. They usually agree and are not required to: a
//!   constructor may re-prefix with `vb.root()` or hand a sibling's builder down.
//! * A parameter's logical identity vs its storage identity. Tied weights make these differ.
//!
//! The dataflow layer will reference [`ParamId`] and [`ModuleInstanceId`] directly rather than
//! rediscovering parameters by string, which is why paths are never the primary key.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;

/// A source location. `file` indexes [`crate::load::Crate::files`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SrcSpan {
    pub file: usize,
    pub line: usize,
    pub col: usize,
}

impl SrcSpan {
    pub const UNKNOWN: SrcSpan = SrcSpan {
        file: usize::MAX,
        line: 0,
        col: 0,
    };
}

/// The outcome of any lookup the analyzer performs.
///
/// Generic on purpose: every resolution step (field type, callee, prefix) returns one of these,
/// so "we did not know" is representable everywhere and cannot decay into a default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "resolution", rename_all = "snake_case")]
pub enum Resolved<T> {
    /// Exactly one answer, statically certain.
    Exact(T),
    /// Several possible answers and no way to choose. Never silently pick the first.
    Ambiguous(Vec<T>),
    /// Nothing could be determined. Carries why, for the diagnostic stream.
    Unresolved(String),
}

impl<T> Resolved<T> {
    pub fn exact(&self) -> Option<&T> {
        match self {
            Resolved::Exact(v) => Some(v),
            _ => None,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Resolved::Exact(_))
    }
}

/// Certainty attached to a node that exists but may be conditional.
///
/// Distinct from [`Resolved`]: that answers "did the lookup succeed", this answers "does this
/// thing definitely exist at runtime".
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "reason", rename_all = "snake_case")]
pub enum Certainty {
    /// Unconditionally present.
    Certain,
    /// Present only under a condition the analyzer could not evaluate (a config-dependent
    /// bias, an `Option` field, a branch).
    Conditional(String),
    /// The analyzer could not model the construct at all; reported so the hole is visible.
    Unknown(String),
}

impl Certainty {
    pub fn is_certain(&self) -> bool {
        matches!(self, Certainty::Certain)
    }
}

/// One element of a parameter key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "seg", rename_all = "snake_case")]
pub enum KeySeg {
    /// A literal prefix from `vb.pp("name")` or a leaf tensor name.
    Literal(String),
    /// A segment whose value is a runtime expression, e.g. the `{index}` in
    /// `vb.pp(format!("layers.{index}"))`.
    ///
    /// The source text is kept verbatim; it is never evaluated. Concrete indices come only
    /// from matching a real checkpoint, never from a guess.
    Dynamic { expr: String },
    /// A single dotted component containing both literal text and one or more runtime
    /// placeholders, e.g. `pre_local_block_{i}`.
    ///
    /// This is distinct from [`KeySeg::Dynamic`] so display preserves the literal portion
    /// instead of adding another pair of braces around the whole component.
    Template { text: String },
}

impl fmt::Display for KeySeg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeySeg::Literal(n) => write!(f, "{n}"),
            KeySeg::Dynamic { expr } => write!(f, "{{{expr}}}"),
            KeySeg::Template { text } => write!(f, "{text}"),
        }
    }
}

/// A dotted key such as `model.layers.{index}.self_attn.q_proj.weight`.
///
/// This is a *display and matching* type, not the primary key. Identity is [`ParamId`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Key {
    pub segs: Vec<KeySeg>,
}

impl Key {
    /// Parse a dotted checkpoint key string, treating `{…}` segments as templates.
    pub fn from_dotted(text: &str) -> Self {
        Key {
            segs: text
                .split('.')
                .filter(|part| !part.is_empty())
                .map(|part| {
                    if part.contains('{') && part.contains('}') {
                        KeySeg::Template {
                            text: part.to_string(),
                        }
                    } else {
                        KeySeg::Literal(part.to_string())
                    }
                })
                .collect(),
        }
    }

    /// Append a prefix as written at a `pp()` call. A single call may carry dots
    /// (`pp("lora_layers.0")`), so the text is split to keep key algebra uniform.
    pub fn push_literal(&self, text: &str) -> Self {
        let mut next = self.clone();
        for part in text.split('.').filter(|p| !p.is_empty()) {
            next.segs.push(KeySeg::Literal(part.to_string()));
        }
        next
    }

    pub fn push(&self, seg: KeySeg) -> Self {
        let mut next = self.clone();
        next.segs.push(seg);
        next
    }

    pub fn extend(&self, segs: &[KeySeg]) -> Self {
        let mut next = self.clone();
        next.segs.extend_from_slice(segs);
        next
    }

    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// True when the key contains a dynamic segment and so denotes a family of tensors.
    pub fn is_template(&self) -> bool {
        self.segs
            .iter()
            .any(|s| matches!(s, KeySeg::Dynamic { .. } | KeySeg::Template { .. }))
    }

    /// Match against a concrete checkpoint tensor name. A dynamic segment matches exactly one
    /// dotted component — candle's `VarBuilder::path` joins with `.` (var_builder.rs:186), so
    /// one `pp` level is one component.
    pub fn matches(&self, concrete: &str) -> bool {
        let parts: Vec<&str> = concrete.split('.').filter(|p| !p.is_empty()).collect();
        if parts.len() != self.segs.len() {
            return false;
        }
        self.segs.iter().zip(parts).all(|(seg, part)| match seg {
            KeySeg::Literal(n) => n == part,
            KeySeg::Dynamic { .. } => true,
            KeySeg::Template { text } => template_segment_matches(text, part),
        })
    }
}

fn template_segment_matches(template: &str, concrete: &str) -> bool {
    let mut literals = Vec::new();
    let mut cursor = 0usize;
    while let Some(open_rel) = template[cursor..].find('{') {
        let open = cursor + open_rel;
        let Some(close_rel) = template[open + 1..].find('}') else {
            return template == concrete;
        };
        let close = open + 1 + close_rel;
        literals.push(&template[cursor..open]);
        cursor = close + 1;
    }
    if literals.is_empty() {
        return template == concrete;
    }
    literals.push(&template[cursor..]);

    let starts_with_wildcard = template.starts_with('{');
    let ends_with_wildcard = template.ends_with('}');
    let mut position = 0usize;
    for (index, literal) in literals.iter().enumerate() {
        if literal.is_empty() {
            continue;
        }
        if index == 0 && !starts_with_wildcard {
            if !concrete.starts_with(literal) {
                return false;
            }
            position = literal.len();
            continue;
        }
        let Some(found) = concrete[position..].find(literal) else {
            return false;
        };
        position += found + literal.len();
    }
    ends_with_wildcard
        || literals
            .last()
            .is_some_and(|suffix| concrete.ends_with(suffix))
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined: Vec<String> = self.segs.iter().map(|s| s.to_string()).collect();
        write!(f, "{}", joined.join("."))
    }
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
        pub struct $name(pub usize);
    };
}

id_type!(ModuleDefId);
id_type!(ModuleInstanceId);
id_type!(ParamSiteId);
id_type!(ParamId);

/// A Rust type that constructs parameters, i.e. a module definition.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleDef {
    pub id: ModuleDefId,
    /// Type name as written, e.g. `SelfAttention`.
    pub name: String,
    /// Constructor followed to discover this def's contents, e.g. `SelfAttention::new`.
    pub ctor: Option<String>,
    pub span: SrcSpan,
    pub sites: Vec<ParamSiteId>,
}

/// A repeated construction, from a `for` loop or an iterator chain over layers.
#[derive(Debug, Clone, Serialize)]
pub struct Repeat {
    /// Loop variable, e.g. `index`.
    pub var: String,
    /// Source text of the bound, e.g. `cfg.num_hidden_layers`. Never evaluated.
    pub bound: String,
}

/// A module definition built at one specific `VarBuilder` prefix.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleInstance {
    pub id: ModuleInstanceId,
    pub def: ModuleDefId,
    pub parent: Option<ModuleInstanceId>,
    /// Field name in the parent struct, when this came from a field initializer. Absent when
    /// containment and prefix nesting diverge.
    pub via_field: Option<String>,
    /// Prefix at this instance. May be a template.
    pub prefix: Key,
    /// Which `VarBuilder` root this prefix belongs to, named after the constructor parameter
    /// it entered through (e.g. `base_vb` vs `train_vb`). Two prefixes are only comparable
    /// within the same root: distinct roots are distinct namespaces, and in practice one is
    /// frozen mmapped weights while another is a trainable `VarMap`.
    pub root: String,
    /// True when the prefix was not written in source but computed as the longest common
    /// prefix of this instance's descendants. Grouping structs (`Layer { .. }` literals) have
    /// no builder of their own, so their prefix is derived rather than observed.
    pub prefix_derived: bool,
    pub repeat: Option<Repeat>,
    pub origin: SrcSpan,
    pub children: Vec<ModuleInstanceId>,
    pub certainty: Certainty,
}

/// How a parameter is acquired.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "via", rename_all = "snake_case")]
pub enum Acquisition {
    /// Through a known candle-nn constructor, e.g. `candle_nn::linear`.
    Constructor { func: String, cite: &'static str },
    /// Through a raw `vb.get(..)` / `get_with_hints(..)` in user code.
    RawGet { method: String },
}

/// A parameter-registering call site in source, relative to its owning def.
#[derive(Debug, Clone, Serialize)]
pub struct ParamSite {
    pub id: ParamSiteId,
    pub owner: ModuleDefId,
    pub acquisition: Acquisition,
    /// Key relative to the `VarBuilder` handed to the owning constructor.
    pub relative_key: Key,
    pub kind: crate::known::ParamKind,
    /// Source text of the shape argument, when one was supplied. Symbolic on purpose: the
    /// dimensions are config expressions and inventing numbers would be a lie.
    pub shape: Option<String>,
    pub span: SrcSpan,
    pub certainty: Certainty,
}

/// Whether a parameter was found in a checkpoint. Evidence about the checkpoint, not ground
/// truth about the model: a tensor may be absent because the checkpoint is stale, and present
/// because something else wrote it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "match", rename_all = "snake_case")]
pub enum CheckpointMatch {
    NotChecked,
    /// Matched exactly one tensor.
    Found {
        name: String,
        shape: Vec<usize>,
        dtype: String,
    },
    /// A template matched several tensors; this is the expected case for layer families.
    FoundMany {
        count: usize,
        sample: String,
    },
    Missing,
}

/// One tensor: a site realised at one instance.
#[derive(Debug, Clone, Serialize)]
pub struct Param {
    pub id: ParamId,
    pub site: ParamSiteId,
    pub owner: ModuleInstanceId,
    /// Fully qualified key, prefix + relative key.
    pub key: Key,
    /// `VarBuilder` root this tensor lives under. See [`ModuleInstance::root`].
    pub root: String,
    pub certainty: Certainty,
    pub checkpoint: CheckpointMatch,
}

/// Something the analyzer could not model, surfaced rather than swallowed.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub span: SrcSpan,
    pub message: String,
    pub key: Option<Key>,
}

/// Quantitative analysis coverage. Reported in every output so a reader can judge how much of
/// the model the tool actually saw.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Coverage {
    pub instances: usize,
    pub params: usize,
    pub params_certain: usize,
    pub params_conditional: usize,
    pub params_unknown: usize,
    pub diagnostics: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct Structure {
    pub defs: Vec<ModuleDef>,
    pub instances: Vec<ModuleInstance>,
    pub sites: Vec<ParamSite>,
    pub params: Vec<Param>,
    pub root: Option<ModuleInstanceId>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Structure {
    pub fn def(&self, id: ModuleDefId) -> &ModuleDef {
        &self.defs[id.0]
    }

    pub fn instance(&self, id: ModuleInstanceId) -> &ModuleInstance {
        &self.instances[id.0]
    }

    pub fn site(&self, id: ParamSiteId) -> &ParamSite {
        &self.sites[id.0]
    }

    pub fn add_def(&mut self, name: String, ctor: Option<String>, span: SrcSpan) -> ModuleDefId {
        let id = ModuleDefId(self.defs.len());
        self.defs.push(ModuleDef {
            id,
            name,
            ctor,
            span,
            sites: Vec::new(),
        });
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_instance(
        &mut self,
        def: ModuleDefId,
        parent: Option<ModuleInstanceId>,
        via_field: Option<String>,
        prefix: Key,
        root: String,
        prefix_derived: bool,
        repeat: Option<Repeat>,
        origin: SrcSpan,
        certainty: Certainty,
    ) -> ModuleInstanceId {
        let id = ModuleInstanceId(self.instances.len());
        self.instances.push(ModuleInstance {
            id,
            def,
            parent,
            via_field,
            prefix,
            root,
            prefix_derived,
            repeat,
            origin,
            children: Vec::new(),
            certainty,
        });
        if let Some(p) = parent {
            self.instances[p.0].children.push(id);
        }
        id
    }

    /// Fill in the prefixes of grouping instances as the longest common prefix of everything
    /// beneath them. Runs bottom-up so nested grouping nodes resolve correctly.
    ///
    /// Keys are grouped by `VarBuilder` root first, because distinct roots are distinct
    /// namespaces: a `Layer` holding both frozen base weights and a trainable cross-attention
    /// adapter has no meaningful prefix spanning the two, and computing one across them would
    /// collapse to empty and lose the grouping entirely. The most populated root wins, and the
    /// derived prefix is reported as belonging to it.
    pub fn derive_prefixes(&mut self) {
        let order: Vec<ModuleInstanceId> =
            (0..self.instances.len()).map(ModuleInstanceId).collect();
        for id in order.into_iter().rev() {
            if !self.instances[id.0].prefix_derived {
                continue;
            }

            let mut by_root: BTreeMap<String, Vec<Key>> = BTreeMap::new();
            for p in self.params.iter().filter(|p| p.owner == id) {
                by_root
                    .entry(p.root.clone())
                    .or_default()
                    .push(p.key.clone());
            }
            for child in self.instances[id.0].children.clone() {
                let child = &self.instances[child.0];
                if !child.prefix.is_empty() && !child.root.is_empty() {
                    by_root
                        .entry(child.root.clone())
                        .or_default()
                        .push(child.prefix.clone());
                }
            }

            let Some((root, keys)) = by_root.into_iter().max_by_key(|(_, k)| k.len()) else {
                continue;
            };
            if let Some(common) = longest_common_prefix(&keys) {
                self.instances[id.0].prefix = common;
                self.instances[id.0].root = root;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_site(
        &mut self,
        owner: ModuleDefId,
        acquisition: Acquisition,
        relative_key: Key,
        kind: crate::known::ParamKind,
        shape: Option<String>,
        span: SrcSpan,
        certainty: Certainty,
    ) -> ParamSiteId {
        let id = ParamSiteId(self.sites.len());
        self.sites.push(ParamSite {
            id,
            owner,
            acquisition,
            relative_key,
            kind,
            shape,
            span,
            certainty,
        });
        self.defs[owner.0].sites.push(id);
        id
    }

    pub fn add_param(
        &mut self,
        site: ParamSiteId,
        owner: ModuleInstanceId,
        key: Key,
        root: String,
        certainty: Certainty,
    ) -> ParamId {
        let id = ParamId(self.params.len());
        self.params.push(Param {
            id,
            site,
            owner,
            key,
            root,
            certainty,
            checkpoint: CheckpointMatch::NotChecked,
        });
        id
    }

    pub fn diagnose(&mut self, span: SrcSpan, message: impl Into<String>, key: Option<Key>) {
        self.diagnostics.push(Diagnostic {
            span,
            message: message.into(),
            key,
        });
    }

    /// Collapse parameters that resolve to the same tensor.
    ///
    /// Branches produce duplicates: `if cfg.attention_bias { linear(..) } else {
    /// linear_no_bias(..) }` yields `weight` from both arms. When duplicates disagree about
    /// certainty the least certain wins, because the analyzer does not track which arms are
    /// mutually exclusive and must not claim more than it can prove.
    pub fn dedupe_params(&mut self) {
        let mut seen: BTreeMap<(String, String), ParamId> = BTreeMap::new();
        let mut keep: Vec<bool> = vec![true; self.params.len()];

        for (index, should_keep) in keep.iter_mut().enumerate() {
            let ident = (
                self.params[index].root.clone(),
                self.params[index].key.to_string(),
            );
            match seen.get(&ident) {
                Some(first) => {
                    let first = *first;
                    *should_keep = false;
                    let incoming = self.params[index].certainty.clone();
                    let existing = self.params[first.0].certainty.clone();
                    self.params[first.0].certainty = least_certain(existing, incoming);
                }
                None => {
                    seen.insert(ident, ParamId(index));
                }
            }
        }

        let mut next = 0usize;
        let mut remap: Vec<Option<ParamId>> = vec![None; self.params.len()];
        let mut kept = Vec::new();
        for (index, param) in self.params.drain(..).enumerate() {
            if keep[index] {
                remap[index] = Some(ParamId(next));
                let mut param = param;
                param.id = ParamId(next);
                kept.push(param);
                next += 1;
            }
        }
        self.params = kept;
        let _ = remap;
    }

    /// Distinct `VarBuilder` roots in declaration order.
    pub fn roots(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for p in &self.params {
            if !seen.contains(&p.root) {
                seen.push(p.root.clone());
            }
        }
        seen
    }

    pub fn coverage(&self) -> Coverage {
        let mut c = Coverage {
            instances: self.instances.len(),
            params: self.params.len(),
            diagnostics: self.diagnostics.len(),
            ..Default::default()
        };
        for p in &self.params {
            match p.certainty {
                Certainty::Certain => c.params_certain += 1,
                Certainty::Conditional(_) => c.params_conditional += 1,
                Certainty::Unknown(_) => c.params_unknown += 1,
            }
        }
        c
    }
}

/// Least-certain-wins join, used when merging duplicate parameters.
fn least_certain(a: Certainty, b: Certainty) -> Certainty {
    match (&a, &b) {
        (Certainty::Unknown(_), _) => a,
        (_, Certainty::Unknown(_)) => b,
        (Certainty::Conditional(_), _) => a,
        (_, Certainty::Conditional(_)) => b,
        _ => Certainty::Certain,
    }
}

/// Longest common prefix of a set of keys, comparing segments structurally.
fn longest_common_prefix(keys: &[Key]) -> Option<Key> {
    let first = keys.first()?;
    let mut len = first.segs.len();
    for key in &keys[1..] {
        let shared = first
            .segs
            .iter()
            .zip(&key.segs)
            .take_while(|(a, b)| a == b)
            .count();
        len = len.min(shared);
    }
    (len > 0).then(|| Key {
        segs: first.segs[..len].to_vec(),
    })
}
