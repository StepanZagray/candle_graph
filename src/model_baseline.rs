//! Deterministic ModelIr fingerprints for crate-wide CI checks.
//!
//! Unlike the legacy structure baseline, this captures stable identities from the unified
//! `candle-graph/model/1` IR: components, parameters, entrypoints, and finding codes.

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::model_ir::{FindingSeverity, ModelIr};

/// Schema line written at the top of every model baseline file.
pub const SCHEMA: &str = "candle-graph/model-baseline/1";

const HEADER: &str = "# candle-graph/model-baseline/1";

/// Parsed / rendered model baseline document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelBaseline {
    pub components: Vec<String>,
    pub parameters: Vec<String>,
    pub entrypoints: Vec<String>,
    pub findings: Vec<String>,
}

/// Line-oriented comparison of two model baselines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelBaselineDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl ModelBaselineDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

impl fmt::Display for ModelBaselineDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.removed {
            writeln!(f, "- {line}")?;
        }
        for line in &self.added {
            writeln!(f, "+ {line}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ModelBaselineError {
    Io { path: PathBuf, source: io::Error },
    Parse { path: PathBuf, message: String },
    Mismatch(ModelBaselineDiff),
}

impl fmt::Display for ModelBaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Parse { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Mismatch(diff) => write!(f, "model baseline mismatch:\n{diff}"),
        }
    }
}

impl std::error::Error for ModelBaselineError {}

impl ModelBaseline {
    pub fn from_model(model: &ModelIr) -> Self {
        let mut components: Vec<String> = model
            .components
            .iter()
            .map(|component| {
                format!(
                    "component\t{}\t{}",
                    escape_field(&component.qualified_name),
                    escape_field(&component.id.0)
                )
            })
            .collect();
        let mut parameters: Vec<String> = model
            .parameters
            .iter()
            .map(|parameter| {
                format!(
                    "parameter\t{}\t{}\t{}",
                    escape_field(&parameter.builder_root),
                    escape_field(&parameter.key),
                    escape_field(&parameter.kind)
                )
            })
            .collect();
        let mut entrypoints: Vec<String> = model
            .functions
            .iter()
            .filter(|function| function.is_entrypoint)
            .map(|function| format!("entrypoint\t{}", escape_field(&function.qualified_name)))
            .collect();
        let mut findings: Vec<String> = model
            .findings
            .iter()
            .map(|finding| {
                format!(
                    "finding\t{}\t{}",
                    escape_field(&finding.rule),
                    severity_name(&finding.severity)
                )
            })
            .collect();

        components.sort();
        components.dedup();
        parameters.sort();
        parameters.dedup();
        entrypoints.sort();
        entrypoints.dedup();
        findings.sort();
        findings.dedup();

        Self {
            components,
            parameters,
            entrypoints,
            findings,
        }
    }

    pub fn render(&self) -> String {
        let mut lines = vec![HEADER.to_string()];
        lines.extend(self.components.iter().cloned());
        lines.extend(self.parameters.iter().cloned());
        lines.extend(self.entrypoints.iter().cloned());
        lines.extend(self.findings.iter().cloned());
        lines.push(String::new());
        lines.join("\n")
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut baseline = Self::default();
        let mut saw_header = false;
        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim_end();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                if line == HEADER {
                    saw_header = true;
                    continue;
                }
                return Err(format!("line {}: unsupported header `{line}`", index + 1));
            }
            let mut parts = line.split('\t');
            let kind = parts.next().unwrap_or_default();
            match kind {
                "component" => {
                    let qualified = unescape_field(parts.next().unwrap_or_default());
                    let id = unescape_field(parts.next().unwrap_or_default());
                    baseline.components.push(format!(
                        "component\t{}\t{}",
                        escape_field(&qualified),
                        escape_field(&id)
                    ));
                }
                "parameter" => {
                    let root = unescape_field(parts.next().unwrap_or_default());
                    let key = unescape_field(parts.next().unwrap_or_default());
                    let param_kind = unescape_field(parts.next().unwrap_or_default());
                    baseline.parameters.push(format!(
                        "parameter\t{}\t{}\t{}",
                        escape_field(&root),
                        escape_field(&key),
                        escape_field(&param_kind)
                    ));
                }
                "entrypoint" => {
                    let name = unescape_field(parts.next().unwrap_or_default());
                    baseline
                        .entrypoints
                        .push(format!("entrypoint\t{}", escape_field(&name)));
                }
                "finding" => {
                    let rule = unescape_field(parts.next().unwrap_or_default());
                    let severity = parts.next().unwrap_or_default();
                    baseline
                        .findings
                        .push(format!("finding\t{}\t{severity}", escape_field(&rule)));
                }
                other => {
                    return Err(format!("line {}: unknown record kind `{other}`", index + 1));
                }
            }
        }
        if !saw_header {
            return Err(format!("missing `{HEADER}` header"));
        }
        baseline.components.sort();
        baseline.components.dedup();
        baseline.parameters.sort();
        baseline.parameters.dedup();
        baseline.entrypoints.sort();
        baseline.entrypoints.dedup();
        baseline.findings.sort();
        baseline.findings.dedup();
        Ok(baseline)
    }
}

/// Compare two baselines as sorted identity lines.
pub fn compare(actual: &ModelBaseline, expected: &ModelBaseline) -> ModelBaselineDiff {
    let actual_lines = all_lines(actual);
    let expected_lines = all_lines(expected);
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < actual_lines.len() && j < expected_lines.len() {
        match actual_lines[i].cmp(&expected_lines[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                added.push(actual_lines[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                removed.push(expected_lines[j].clone());
                j += 1;
            }
        }
    }
    added.extend(actual_lines[i..].iter().cloned());
    removed.extend(expected_lines[j..].iter().cloned());
    ModelBaselineDiff { added, removed }
}

pub fn load(path: impl AsRef<Path>) -> Result<ModelBaseline, ModelBaselineError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ModelBaselineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    ModelBaseline::parse(&text).map_err(|message| ModelBaselineError::Parse {
        path: path.to_path_buf(),
        message,
    })
}

pub fn check(model: &ModelIr, path: impl AsRef<Path>) -> Result<(), ModelBaselineError> {
    let expected = load(path)?;
    let actual = ModelBaseline::from_model(model);
    let diff = compare(&actual, &expected);
    if diff.is_empty() {
        Ok(())
    } else {
        Err(ModelBaselineError::Mismatch(diff))
    }
}

pub fn update(model: &ModelIr, path: impl AsRef<Path>) -> Result<(), ModelBaselineError> {
    let path = path.as_ref();
    let text = ModelBaseline::from_model(model).render();
    atomic_write(path, text.as_bytes())
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ModelBaselineError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| ModelBaselineError::Io {
        path: parent.to_path_buf(),
        source,
    })?;

    let mut tmp_name = path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| "model-baseline".into());
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
        return Err(ModelBaselineError::Io {
            path: tmp_path,
            source,
        });
    }
    fs::rename(&tmp_path, path).map_err(|source| ModelBaselineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn all_lines(baseline: &ModelBaseline) -> Vec<String> {
    let mut lines = Vec::new();
    lines.extend(baseline.components.iter().cloned());
    lines.extend(baseline.parameters.iter().cloned());
    lines.extend(baseline.entrypoints.iter().cloned());
    lines.extend(baseline.findings.iter().cloned());
    lines
}

fn severity_name(severity: &FindingSeverity) -> &'static str {
    match severity {
        FindingSeverity::Error => "error",
        FindingSeverity::Warning => "warning",
        FindingSeverity::Information => "information",
    }
}

fn escape_field(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn unescape_field(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ir::{
        Confidence, Finding, FindingSeverity, Function, ModelIr, StableId, Visibility,
    };

    fn sample_model() -> ModelIr {
        let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
        model.functions.push(Function {
            id: StableId::new("fn", ["Root::forward"]),
            name: "forward".into(),
            qualified_name: "Root::forward".into(),
            owner_type: Some("Root".into()),
            visibility: Visibility::Public,
            parameters: Vec::new(),
            return_type: None,
            cfg_predicates: Vec::new(),
            cfg_active: Some(true),
            source: "model.rs".into(),
            calls: Vec::new(),
            tensor_inputs: Vec::new(),
            tensor_outputs: Vec::new(),
            is_entrypoint: true,
            is_loss: false,
        });
        model.findings.push(Finding {
            id: StableId::new("finding", ["compiler-semantic-evidence"]),
            rule: "compiler-semantic-evidence".into(),
            severity: FindingSeverity::Information,
            confidence: Confidence::Unknown,
            message: "pending compiler frontend".into(),
            source: None,
            related: Vec::new(),
            evidence: Vec::new(),
        });
        model
    }

    #[test]
    fn round_trip_render_parse() {
        let baseline = ModelBaseline::from_model(&sample_model());
        let text = baseline.render();
        assert!(text.starts_with(HEADER));
        let parsed = ModelBaseline::parse(&text).unwrap();
        assert_eq!(parsed, baseline);
    }
}
