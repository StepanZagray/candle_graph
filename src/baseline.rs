//! Deterministic structure baselines for CI.
//!
//! Renders [`crate::ir::Structure`] as stable, line-oriented text that omits arena
//! integer IDs. Baselines can be loaded, compared (added / removed / changed), checked, and
//! updated atomically. Optional pre-normalized dataflow lines may be appended without depending
//! on a dataflow analysis module.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::ir::{Acquisition, Certainty, ModuleInstanceId, Structure};
use crate::known::ParamKind;

/// Schema line written at the top of every baseline file.
pub const SCHEMA: &str = "candle-graph/baseline/1";

const HEADER: &str = "# candle-graph/baseline/1";

/// One module instance in canonical form (no arena IDs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleEntry {
    /// Hierarchy path from the builder root instance, e.g. `Root/blocks:Block`.
    pub path: String,
    pub root: String,
    pub type_name: String,
    pub field: Option<String>,
    pub prefix: String,
    pub repeat: Option<RepeatEntry>,
    pub certainty: String,
}

/// Canonical repeat descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatEntry {
    pub var: String,
    pub bound: String,
}

/// One parameter in canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamEntry {
    pub key: String,
    pub root: String,
    pub kind: String,
    pub certainty: String,
    pub source: String,
}

/// Parsed / rendered baseline document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Baseline {
    pub modules: Vec<ModuleEntry>,
    pub params: Vec<ParamEntry>,
    /// Pre-normalized dataflow finding lines (sorted, unique).
    pub dataflow: Vec<String>,
}

/// A record that exists under the same identity in both sides but differs in attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedEntry {
    pub kind: DiffKind,
    pub identity: String,
    pub expected: String,
    pub actual: String,
}

/// Which section a diff line belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)] // Dataflow reserved for attribute-level dataflow diffs later.
pub enum DiffKind {
    Module,
    Param,
    Dataflow,
}

impl fmt::Display for DiffKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffKind::Module => write!(f, "module"),
            DiffKind::Param => write!(f, "param"),
            DiffKind::Dataflow => write!(f, "dataflow"),
        }
    }
}

/// Line-oriented comparison of two baselines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BaselineDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<ChangedEntry>,
}

impl BaselineDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

impl fmt::Display for BaselineDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return writeln!(f, "baselines match");
        }
        if !self.added.is_empty() {
            writeln!(f, "### added")?;
            for line in &self.added {
                writeln!(f, "+ {line}")?;
            }
        }
        if !self.removed.is_empty() {
            writeln!(f, "### removed")?;
            for line in &self.removed {
                writeln!(f, "- {line}")?;
            }
        }
        if !self.changed.is_empty() {
            writeln!(f, "### changed")?;
            for change in &self.changed {
                writeln!(
                    f,
                    "! {} {}\n- {}\n+ {}",
                    change.kind, change.identity, change.expected, change.actual
                )?;
            }
        }
        Ok(())
    }
}

/// Errors from loading, parsing, checking, or updating a baseline.
#[derive(Debug)]
pub enum BaselineError {
    /// Baseline file does not exist.
    Missing(PathBuf),
    /// Baseline exists but is not a valid document.
    Invalid {
        path: Option<PathBuf>,
        message: String,
    },
    /// Filesystem failure during load/update.
    Io { path: PathBuf, source: io::Error },
    /// Actual structure does not match the expected baseline.
    Mismatch(BaselineDiff),
}

impl fmt::Display for BaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaselineError::Missing(path) => {
                write!(f, "baseline not found: {}", path.display())
            }
            BaselineError::Invalid { path, message } => match path {
                Some(p) => write!(f, "invalid baseline {}: {message}", p.display()),
                None => write!(f, "invalid baseline: {message}"),
            },
            BaselineError::Io { path, source } => {
                write!(f, "baseline I/O error at {}: {source}", path.display())
            }
            BaselineError::Mismatch(diff) => {
                write!(f, "baseline mismatch:\n{diff}")
            }
        }
    }
}

impl std::error::Error for BaselineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BaselineError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl Baseline {
    /// Build a deterministic baseline from a structure plus optional dataflow lines.
    ///
    /// `dataflow_lines` are treated as already-normalized findings: trimmed, empty lines
    /// dropped, then sorted and deduplicated. This keeps the hook independent of any future
    /// dataflow module.
    pub fn from_structure(structure: &Structure, dataflow_lines: &[impl AsRef<str>]) -> Self {
        let mut modules: Vec<ModuleEntry> = structure
            .instances
            .iter()
            .map(|inst| {
                let def = structure.def(inst.def);
                ModuleEntry {
                    path: module_path(structure, inst.id),
                    root: inst.root.clone(),
                    type_name: def.name.clone(),
                    field: inst.via_field.clone(),
                    prefix: inst.prefix.to_string(),
                    repeat: inst.repeat.as_ref().map(|r| RepeatEntry {
                        var: r.var.clone(),
                        bound: r.bound.clone(),
                    }),
                    certainty: certainty_str(&inst.certainty),
                }
            })
            .collect();
        modules.sort_by(|a, b| {
            (&a.path, &a.root, &a.prefix, &a.type_name).cmp(&(
                &b.path,
                &b.root,
                &b.prefix,
                &b.type_name,
            ))
        });

        let mut params: Vec<ParamEntry> = structure
            .params
            .iter()
            .map(|param| {
                let site = structure.site(param.site);
                ParamEntry {
                    key: param.key.to_string(),
                    root: param.root.clone(),
                    kind: kind_str(site.kind),
                    certainty: certainty_str(&param.certainty),
                    source: acquisition_str(&site.acquisition),
                }
            })
            .collect();
        params.sort_by(|a, b| (&a.root, &a.key, &a.kind).cmp(&(&b.root, &b.key, &b.kind)));

        let mut dataflow: Vec<String> = dataflow_lines
            .iter()
            .map(|s| s.as_ref().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        dataflow.sort();
        dataflow.dedup();

        Baseline {
            modules,
            params,
            dataflow,
        }
    }

    /// Render canonical stable text.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(HEADER);
        out.push('\n');
        out.push('\n');

        for module in &self.modules {
            out.push_str(&module_line(module));
            out.push('\n');
        }
        if !self.modules.is_empty() && (!self.params.is_empty() || !self.dataflow.is_empty()) {
            out.push('\n');
        }
        for param in &self.params {
            out.push_str(&param_line(param));
            out.push('\n');
        }
        if !self.params.is_empty() && !self.dataflow.is_empty() {
            out.push('\n');
        }
        for line in &self.dataflow {
            out.push_str("dataflow\t");
            out.push_str(&escape_field(line));
            out.push('\n');
        }
        out
    }

    /// Parse a baseline document from text.
    pub fn parse(text: &str) -> Result<Self, BaselineError> {
        let mut lines = text.lines().map(str::trim_end).peekable();

        // Skip leading blank lines; require schema header.
        while matches!(lines.peek(), Some(l) if l.trim().is_empty()) {
            lines.next();
        }
        let header = lines.next().ok_or_else(|| BaselineError::Invalid {
            path: None,
            message: "empty baseline".into(),
        })?;
        let header = header.trim();
        if header != HEADER && header.trim_start_matches('#').trim() != SCHEMA {
            return Err(BaselineError::Invalid {
                path: None,
                message: format!("expected header `{HEADER}`, found `{header}`"),
            });
        }

        let mut modules = Vec::new();
        let mut params = Vec::new();
        let mut dataflow = Vec::new();

        for (idx, raw) in lines.enumerate() {
            let line_no = idx + 2; // header consumed as line 1 conceptually
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('\t');
            let kind = parts.next().ok_or_else(|| BaselineError::Invalid {
                path: None,
                message: format!("line {line_no}: missing section kind"),
            })?;
            match kind {
                "module" => {
                    let fields = parse_fields(parts, line_no)?;
                    modules.push(ModuleEntry {
                        path: required_field(&fields, "path", line_no)?,
                        root: required_field(&fields, "root", line_no)?,
                        type_name: required_field(&fields, "type", line_no)?,
                        field: optional_field(&fields, "field"),
                        prefix: fields.get("prefix").cloned().unwrap_or_default(),
                        repeat: parse_repeat(fields.get("repeat").map(String::as_str), line_no)?,
                        certainty: required_field(&fields, "certainty", line_no)?,
                    });
                }
                "param" => {
                    let fields = parse_fields(parts, line_no)?;
                    params.push(ParamEntry {
                        key: required_field(&fields, "key", line_no)?,
                        root: required_field(&fields, "root", line_no)?,
                        kind: required_field(&fields, "kind", line_no)?,
                        certainty: required_field(&fields, "certainty", line_no)?,
                        source: required_field(&fields, "source", line_no)?,
                    });
                }
                "dataflow" => {
                    let rest: Vec<&str> = parts.collect();
                    if rest.is_empty() {
                        return Err(BaselineError::Invalid {
                            path: None,
                            message: format!("line {line_no}: dataflow line missing payload"),
                        });
                    }
                    // Prefer `text=` field form; otherwise join remaining tabs as payload.
                    let payload = if rest.len() == 1 && !rest[0].starts_with("text=") {
                        unescape_field(rest[0])
                    } else {
                        let fields = parse_fields(rest.into_iter(), line_no)?;
                        match fields.get("text") {
                            Some(t) => t.clone(),
                            None => {
                                return Err(BaselineError::Invalid {
                                    path: None,
                                    message: format!("line {line_no}: dataflow line missing text"),
                                });
                            }
                        }
                    };
                    if !payload.is_empty() {
                        dataflow.push(payload);
                    }
                }
                other => {
                    return Err(BaselineError::Invalid {
                        path: None,
                        message: format!("line {line_no}: unknown section `{other}`"),
                    });
                }
            }
        }

        modules.sort_by(|a, b| {
            (&a.path, &a.root, &a.prefix, &a.type_name).cmp(&(
                &b.path,
                &b.root,
                &b.prefix,
                &b.type_name,
            ))
        });
        params.sort_by(|a, b| (&a.root, &a.key, &a.kind).cmp(&(&b.root, &b.key, &b.kind)));
        dataflow.sort();
        dataflow.dedup();

        Ok(Baseline {
            modules,
            params,
            dataflow,
        })
    }
}

/// Render canonical baseline text for `structure`, optionally appending dataflow lines.
pub fn render(structure: &Structure, dataflow_lines: &[impl AsRef<str>]) -> String {
    Baseline::from_structure(structure, dataflow_lines).render()
}

/// Parse baseline text.
pub fn parse(text: &str) -> Result<Baseline, BaselineError> {
    Baseline::parse(text)
}

/// Load a baseline from disk. Missing files yield [`BaselineError::Missing`].
pub fn load(path: impl AsRef<Path>) -> Result<Baseline, BaselineError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            BaselineError::Missing(path.to_path_buf())
        } else {
            BaselineError::Io {
                path: path.to_path_buf(),
                source: err,
            }
        }
    })?;
    Baseline::parse(&text).map_err(|err| match err {
        BaselineError::Invalid { message, .. } => BaselineError::Invalid {
            path: Some(path.to_path_buf()),
            message,
        },
        other => other,
    })
}

/// Compare actual vs expected. Identities that exist on both sides with different attributes
/// appear under `changed`; purely new/gone records under added/removed.
pub fn compare(actual: &Baseline, expected: &Baseline) -> BaselineDiff {
    let mut diff = BaselineDiff::default();

    let exp_modules: BTreeMap<String, &ModuleEntry> = expected
        .modules
        .iter()
        .map(|m| (module_identity(m), m))
        .collect();
    let act_modules: BTreeMap<String, &ModuleEntry> = actual
        .modules
        .iter()
        .map(|m| (module_identity(m), m))
        .collect();

    let all_module_ids: BTreeSet<_> = exp_modules
        .keys()
        .chain(act_modules.keys())
        .cloned()
        .collect();
    for id in all_module_ids {
        match (act_modules.get(&id), exp_modules.get(&id)) {
            (Some(a), Some(e)) => {
                let al = module_line(a);
                let el = module_line(e);
                if al != el {
                    diff.changed.push(ChangedEntry {
                        kind: DiffKind::Module,
                        identity: id,
                        expected: el,
                        actual: al,
                    });
                }
            }
            (Some(a), None) => diff.added.push(module_line(a)),
            (None, Some(e)) => diff.removed.push(module_line(e)),
            (None, None) => unreachable!(),
        }
    }

    let exp_params: BTreeMap<String, &ParamEntry> = expected
        .params
        .iter()
        .map(|p| (param_identity(p), p))
        .collect();
    let act_params: BTreeMap<String, &ParamEntry> = actual
        .params
        .iter()
        .map(|p| (param_identity(p), p))
        .collect();

    let all_param_ids: BTreeSet<_> = exp_params
        .keys()
        .chain(act_params.keys())
        .cloned()
        .collect();
    for id in all_param_ids {
        match (act_params.get(&id), exp_params.get(&id)) {
            (Some(a), Some(e)) => {
                let al = param_line(a);
                let el = param_line(e);
                if al != el {
                    diff.changed.push(ChangedEntry {
                        kind: DiffKind::Param,
                        identity: id,
                        expected: el,
                        actual: al,
                    });
                }
            }
            (Some(a), None) => diff.added.push(param_line(a)),
            (None, Some(e)) => diff.removed.push(param_line(e)),
            (None, None) => unreachable!(),
        }
    }

    let exp_df: BTreeSet<&str> = expected.dataflow.iter().map(String::as_str).collect();
    let act_df: BTreeSet<&str> = actual.dataflow.iter().map(String::as_str).collect();
    for line in act_df.difference(&exp_df) {
        diff.added.push(format!("dataflow\t{}", escape_field(line)));
    }
    for line in exp_df.difference(&act_df) {
        diff.removed
            .push(format!("dataflow\t{}", escape_field(line)));
    }

    diff.added.sort();
    diff.removed.sort();
    diff.changed
        .sort_by(|a, b| (&a.kind, &a.identity).cmp(&(&b.kind, &b.identity)));

    diff
}

/// Check that `structure` (+ optional dataflow lines) matches the baseline at `path`.
pub fn check(
    structure: &Structure,
    path: impl AsRef<Path>,
    dataflow_lines: &[impl AsRef<str>],
) -> Result<(), BaselineError> {
    let expected = load(path)?;
    let actual = Baseline::from_structure(structure, dataflow_lines);
    let diff = compare(&actual, &expected);
    if diff.is_empty() {
        Ok(())
    } else {
        Err(BaselineError::Mismatch(diff))
    }
}

/// Atomically write a fresh baseline for `structure` to `path` (temp file + rename).
pub fn update(
    structure: &Structure,
    path: impl AsRef<Path>,
    dataflow_lines: &[impl AsRef<str>],
) -> Result<(), BaselineError> {
    let path = path.as_ref();
    let text = render(structure, dataflow_lines);
    atomic_write(path, text.as_bytes())
}

/// Write `bytes` to `path` via a same-directory temporary file and rename.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), BaselineError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| BaselineError::Io {
        path: parent.to_path_buf(),
        source,
    })?;

    let mut tmp_name = path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| "baseline".into());
    tmp_name.push(".tmp");
    let tmp_path = parent.join(tmp_name);

    let write_tmp = || -> io::Result<()> {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    };
    if let Err(source) = write_tmp() {
        let _ = fs::remove_file(&tmp_path);
        return Err(BaselineError::Io {
            path: tmp_path,
            source,
        });
    }

    if let Err(source) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(BaselineError::Io {
            path: path.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn module_identity(m: &ModuleEntry) -> String {
    format!("{}\t{}", m.path, m.root)
}

fn param_identity(p: &ParamEntry) -> String {
    format!("{}\t{}", p.root, p.key)
}

fn module_line(m: &ModuleEntry) -> String {
    let field = m.field.as_deref().unwrap_or("");
    let repeat = match &m.repeat {
        Some(r) => format!("{} over {}", r.var, r.bound),
        None => String::new(),
    };
    format!(
        "module\tpath={}\troot={}\ttype={}\tfield={}\tprefix={}\trepeat={}\tcertainty={}",
        escape_field(&m.path),
        escape_field(&m.root),
        escape_field(&m.type_name),
        escape_field(field),
        escape_field(&m.prefix),
        escape_field(&repeat),
        escape_field(&m.certainty),
    )
}

fn param_line(p: &ParamEntry) -> String {
    format!(
        "param\tkey={}\troot={}\tkind={}\tcertainty={}\tsource={}",
        escape_field(&p.key),
        escape_field(&p.root),
        escape_field(&p.kind),
        escape_field(&p.certainty),
        escape_field(&p.source),
    )
}

fn module_path(structure: &Structure, id: ModuleInstanceId) -> String {
    let mut parts = Vec::new();
    let mut cur = Some(id);
    while let Some(cid) = cur {
        let inst = structure.instance(cid);
        let type_name = structure.def(inst.def).name.as_str();
        let seg = match &inst.via_field {
            Some(field) => format!("{field}:{type_name}"),
            None if inst.parent.is_some() && !inst.prefix.is_empty() => {
                format!("{type_name}@{}", inst.prefix)
            }
            None => type_name.to_string(),
        };
        parts.push(seg);
        cur = inst.parent;
    }
    parts.reverse();
    parts.join("/")
}

fn certainty_str(c: &Certainty) -> String {
    match c {
        Certainty::Certain => "certain".to_string(),
        Certainty::Conditional(reason) => format!("conditional:{reason}"),
        Certainty::Unknown(reason) => format!("unknown:{reason}"),
    }
}

fn kind_str(kind: ParamKind) -> String {
    match kind {
        ParamKind::Weight => "weight",
        ParamKind::Bias => "bias",
        ParamKind::RunningMean => "running_mean",
        ParamKind::RunningVar => "running_var",
        ParamKind::Raw => "raw",
    }
    .to_string()
}

fn acquisition_str(acq: &Acquisition) -> String {
    match acq {
        Acquisition::Constructor { func, .. } => format!("constructor:{func}"),
        Acquisition::RawGet { method } => format!("raw_get:{method}"),
    }
}

fn escape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn unescape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn parse_fields<'a>(
    parts: impl Iterator<Item = &'a str>,
    line_no: usize,
) -> Result<BTreeMap<String, String>, BaselineError> {
    let mut fields = BTreeMap::new();
    for part in parts {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').ok_or_else(|| BaselineError::Invalid {
            path: None,
            message: format!("line {line_no}: expected key=value, found `{part}`"),
        })?;
        fields.insert(key.to_string(), unescape_field(value));
    }
    Ok(fields)
}

fn required_field(
    fields: &BTreeMap<String, String>,
    key: &str,
    line_no: usize,
) -> Result<String, BaselineError> {
    fields
        .get(key)
        .cloned()
        .ok_or_else(|| BaselineError::Invalid {
            path: None,
            message: format!("line {line_no}: missing field `{key}`"),
        })
}

fn optional_field(fields: &BTreeMap<String, String>, key: &str) -> Option<String> {
    fields.get(key).cloned().filter(|v| !v.is_empty())
}

fn parse_repeat(raw: Option<&str>, line_no: usize) -> Result<Option<RepeatEntry>, BaselineError> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let (var, bound) = raw
        .split_once(" over ")
        .ok_or_else(|| BaselineError::Invalid {
            path: None,
            message: format!(
                "line {line_no}: repeat must look like `var over bound`, found `{raw}`"
            ),
        })?;
    Ok(Some(RepeatEntry {
        var: var.to_string(),
        bound: bound.to_string(),
    }))
}
