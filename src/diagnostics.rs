//! Rustc-like diagnostics rendered from unified model findings.

use serde::Serialize;

use crate::model_ir::{Finding, FindingSeverity, ModelIr};

/// How finding diagnostics are rendered on stderr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    pub confidence: String,
}

impl Diagnostic {
    pub fn from_finding(finding: &Finding) -> Self {
        let (file, line, column) = parse_source_location(finding.source.as_deref());
        Self {
            code: finding.rule.clone(),
            severity: severity_label(&finding.severity),
            message: finding.message.clone(),
            file,
            line,
            column,
            confidence: format!("{:?}", finding.confidence).to_ascii_lowercase(),
        }
    }
}

/// Collect diagnostics for every finding in `model`, sorted by (file, line, code, message).
pub fn from_model(model: &ModelIr) -> Vec<Diagnostic> {
    let mut diagnostics: Vec<Diagnostic> = model
        .findings
        .iter()
        .map(Diagnostic::from_finding)
        .collect();
    diagnostics.sort_by(|a, b| {
        (
            a.file.as_deref().unwrap_or(""),
            a.line.unwrap_or(0),
            a.column.unwrap_or(0),
            a.code.as_str(),
            a.message.as_str(),
        )
            .cmp(&(
                b.file.as_deref().unwrap_or(""),
                b.line.unwrap_or(0),
                b.column.unwrap_or(0),
                b.code.as_str(),
                b.message.as_str(),
            ))
    });
    diagnostics
}

/// Render diagnostics for stderr.
pub fn render(diagnostics: &[Diagnostic], format: MessageFormat) -> String {
    match format {
        MessageFormat::Human => render_human(diagnostics),
        MessageFormat::Json => {
            let mut out = serde_json::to_string_pretty(diagnostics).unwrap_or_else(|_| "[]".into());
            out.push('\n');
            out
        }
    }
}

fn render_human(diagnostics: &[Diagnostic]) -> String {
    let mut out = String::new();
    for diagnostic in diagnostics {
        out.push_str(diagnostic.severity);
        out.push_str(": ");
        out.push_str(&diagnostic.message);
        out.push('\n');
        if let Some(file) = &diagnostic.file {
            out.push_str("  --> ");
            out.push_str(file);
            if let Some(line) = diagnostic.line {
                out.push(':');
                out.push_str(&line.to_string());
                if let Some(column) = diagnostic.column {
                    out.push(':');
                    out.push_str(&column.to_string());
                }
            }
            out.push('\n');
        }
        out.push_str("  = code: ");
        out.push_str(&diagnostic.code);
        out.push('\n');
    }
    if !diagnostics.is_empty() {
        let errors = diagnostics.iter().filter(|d| d.severity == "error").count();
        let warnings = diagnostics
            .iter()
            .filter(|d| d.severity == "warning")
            .count();
        let notes = diagnostics.iter().filter(|d| d.severity == "note").count();
        out.push('\n');
        out.push_str(&format!(
            "candle-graph check: {errors} error(s), {warnings} warning(s), {notes} note(s)\n"
        ));
    }
    out
}

fn severity_label(severity: &FindingSeverity) -> &'static str {
    match severity {
        FindingSeverity::Error => "error",
        FindingSeverity::Warning => "warning",
        FindingSeverity::Information => "note",
    }
}

/// Parse `path`, `path:line`, or `path:line:col` from a finding source string.
fn parse_source_location(source: Option<&str>) -> (Option<String>, Option<usize>, Option<usize>) {
    let Some(source) = source.filter(|value| !value.is_empty()) else {
        return (None, None, None);
    };
    let parts: Vec<&str> = source.rsplitn(3, ':').collect();
    match parts.as_slice() {
        [col, line, file]
            if line.parse::<usize>().is_ok()
                && col.parse::<usize>().is_ok()
                && (file.contains('/') || file.contains('\\') || file.ends_with(".rs")) =>
        {
            (
                Some((*file).to_string()),
                line.parse().ok(),
                col.parse().ok(),
            )
        }
        [line, file] if line.parse::<usize>().is_ok() => {
            (Some((*file).to_string()), line.parse().ok(), None)
        }
        _ => (Some(source.to_string()), None, None),
    }
}

/// True when model findings include Error or Warning severity.
pub fn has_failing_findings(model: &ModelIr) -> bool {
    model.findings.iter().any(|finding| {
        matches!(
            finding.severity,
            FindingSeverity::Error | FindingSeverity::Warning
        )
    })
}

/// True when model findings include a proven defect (`Error` + `Confidence::Proven`).
///
/// Coverage gaps (`Unknown` / `Warning`) must not fail `--strict` / `cargo candle-graph check`.
pub fn has_proven_defect_findings(model: &ModelIr) -> bool {
    use crate::model_ir::Confidence;
    model.findings.iter().any(|finding| {
        matches!(finding.severity, FindingSeverity::Error)
            && matches!(finding.confidence, Confidence::Proven)
    })
}

/// Findings that match any `--deny` rule with proven confidence.
///
/// Explicit deny rules gate even when severity is `Warning` (e.g. proven
/// `zero-times-infinity` on a loss path classified as local-only before deny).
pub fn denied_findings<'a>(model: &'a ModelIr, deny_rules: &[String]) -> Vec<&'a Finding> {
    use crate::model_ir::Confidence;
    if deny_rules.is_empty() {
        return Vec::new();
    }
    model
        .findings
        .iter()
        .filter(|finding| {
            matches!(finding.confidence, Confidence::Proven)
                && deny_rules
                    .iter()
                    .any(|rule| rule_matches(rule, &finding.rule))
        })
        .collect()
}

pub fn has_denied_findings(model: &ModelIr, deny_rules: &[String]) -> bool {
    !denied_findings(model, deny_rules).is_empty()
}

fn rule_matches(rule: &str, finding_rule: &str) -> bool {
    let rule = rule.trim().to_ascii_lowercase();
    let finding_rule = finding_rule.to_ascii_lowercase();
    rule == finding_rule || finding_rule.starts_with(&format!("{rule}-"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ir::{Confidence, StableId};

    #[test]
    fn parses_rustc_style_locations() {
        let (file, line, col) = parse_source_location(Some("src/model.rs:12:4"));
        assert_eq!(file.as_deref(), Some("src/model.rs"));
        assert_eq!(line, Some(12));
        assert_eq!(col, Some(4));
    }

    #[test]
    fn human_render_includes_summary() {
        let finding = Finding {
            id: StableId::new("finding", ["demo"]),
            rule: "demo-rule".into(),
            severity: FindingSeverity::Warning,
            confidence: Confidence::Proven,
            message: "something looks off".into(),
            source: Some("src/lib.rs:1:1".into()),
            related: Vec::new(),
            evidence: Vec::new(),
        };
        let text = render(&[Diagnostic::from_finding(&finding)], MessageFormat::Human);
        assert!(text.contains("warning: something looks off"));
        assert!(text.contains("--> src/lib.rs:1:1"));
        assert!(text.contains("1 warning(s)"));
    }
}
