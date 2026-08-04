//! Bounded Cargo/config discovery for analyzing a crate in its real feature/cfg context.
//!
//! Discovers the nearest `Cargo.toml`, runs `cargo metadata` and `rustc --print cfg` via
//! [`std::process::Command`] (never a shell), and returns a deterministic, serializable snapshot.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Feature / target selection passed through to `cargo metadata` and `rustc --print cfg`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CargoOptions {
    /// Explicit features forwarded as `--features`.
    pub features: Vec<String>,
    /// Forwarded as `--all-features`.
    pub all_features: bool,
    /// Forwarded as `--no-default-features`.
    pub no_default_features: bool,
    /// Optional `--filter-platform` / `rustc --target` triple.
    pub target: Option<String>,
    /// Optional Cargo target name (`lib`/binary target), independent of the target triple.
    pub package_target: Option<String>,
}

/// Names in candle-graph's own `[features]` table — not valid on arbitrary model crates.
const CANDLE_GRAPH_FEATURES: &[&str] = &["static", "visualizer", "runtime", "all"];

impl CargoOptions {
    /// Strip feature flags that refer to candle-graph itself, not the crate under analysis.
    ///
    /// Users often run `cargo candle-graph view --features visualizer` (or `--features all`)
    /// intending to enable the HTML visualizer on candle-graph. Those flags must not be
    /// forwarded to `cargo metadata` for the analyzed model crate.
    pub fn strip_candle_graph_features(&mut self) -> Vec<String> {
        let mut stripped = Vec::new();
        self.features.retain(|feature| {
            if CANDLE_GRAPH_FEATURES.contains(&feature.as_str()) {
                stripped.push(feature.clone());
                false
            } else {
                true
            }
        });
        stripped
    }
}

/// One compile target of the selected package (lib, bin, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CargoTarget {
    pub name: String,
    pub kind: Vec<String>,
    pub src_path: PathBuf,
}

/// Deterministic snapshot of the Cargo package / workspace / feature / cfg context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CargoContext {
    pub package_name: String,
    pub package_version: String,
    pub package_id: String,
    pub manifest_path: PathBuf,
    pub workspace_root: PathBuf,
    pub target_directory: PathBuf,
    /// Targets of the selected package, sorted by `(name, kind, src_path)`.
    pub targets: Vec<CargoTarget>,
    /// Resolved active features for the selected package, sorted.
    pub active_features: Vec<String>,
    /// Versions of every package whose name starts with `candle-`, keyed by name.
    pub candle_versions: BTreeMap<String, String>,
    /// Rust crate identifier (including dependency renames) to Cargo package name.
    pub dependency_aliases: BTreeMap<String, String>,
    /// Active `rustc --print cfg` lines plus `feature="…"` for each active feature, sorted.
    pub cfgs: Vec<String>,
}

/// Discover Cargo context for `path`, which may be a crate root or a nested source directory.
pub fn discover(path: impl AsRef<Path>, options: &CargoOptions) -> Result<CargoContext> {
    CargoContext::discover(path, options)
}

impl CargoContext {
    /// Discover Cargo context for `path`, which may be a crate root or a nested source directory.
    pub fn discover(path: impl AsRef<Path>, options: &CargoOptions) -> Result<Self> {
        let path = path.as_ref();
        let manifest_path = find_manifest(path)
            .with_context(|| format!("failed to locate Cargo.toml from {}", path.display()))?;

        let metadata = run_cargo_metadata(&manifest_path, options)?;
        let package = select_package(&metadata, &manifest_path)?;

        let package_name = package
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("package missing name"))?
            .to_string();
        let package_version = package
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("package missing version"))?
            .to_string();
        let package_id = package
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("package missing id"))?
            .to_string();

        let manifest_path = path_from_json(package, "manifest_path")?;
        let workspace_root = path_from_root(&metadata, "workspace_root")?;
        let target_directory = path_from_root(&metadata, "target_directory")?;

        let mut targets = parse_targets(package)?;
        targets
            .sort_by(|a, b| (&a.name, &a.kind, &a.src_path).cmp(&(&b.name, &b.kind, &b.src_path)));

        let mut active_features = resolve_active_features(&metadata, &package_id)?;
        active_features.sort();
        active_features.dedup();

        let candle_versions = collect_candle_versions(&metadata, &package_id)?;
        let dependency_aliases = collect_dependency_aliases(package)?;

        let mut cfgs = collect_rustc_cfgs(options.target.as_deref())?;
        for feature in &active_features {
            cfgs.push(format!("feature=\"{feature}\""));
        }
        cfgs.sort();
        cfgs.dedup();

        Ok(Self {
            package_name,
            package_version,
            package_id,
            manifest_path,
            workspace_root,
            target_directory,
            targets,
            active_features,
            candle_versions,
            dependency_aliases,
            cfgs,
        })
    }

    /// Crate roots selected for source analysis.
    ///
    /// By default a library target is preferred because binaries normally consume it. Packages
    /// without a library select their first ordinary binary. Tests/examples/benches are included
    /// only when explicitly selected by target name.
    pub fn selected_source_roots(&self, requested: Option<&str>) -> Result<Vec<PathBuf>> {
        if let Some(name) = requested {
            let selected = self
                .targets
                .iter()
                .filter(|target| target.name == name)
                .map(|target| target.src_path.clone())
                .collect::<Vec<_>>();
            if selected.is_empty() {
                bail!(
                    "Cargo target `{name}` not found; available targets: {}",
                    self.targets
                        .iter()
                        .map(|target| target.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            return Ok(selected);
        }

        if let Some(library) = self
            .targets
            .iter()
            .find(|target| target.kind.iter().any(|kind| kind == "lib"))
        {
            return Ok(vec![library.src_path.clone()]);
        }
        if let Some(binary) = self
            .targets
            .iter()
            .find(|target| target.kind.iter().any(|kind| kind == "bin"))
        {
            return Ok(vec![binary.src_path.clone()]);
        }
        bail!(
            "package `{}` has no library or binary target; select a target explicitly",
            self.package_name
        )
    }
}

fn collect_dependency_aliases(package: &Value) -> Result<BTreeMap<String, String>> {
    let dependencies = package
        .get("dependencies")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("package missing dependencies array"))?;
    let mut aliases = BTreeMap::new();
    for dependency in dependencies {
        let Some(name) = dependency.get("name").and_then(Value::as_str) else {
            continue;
        };
        let alias = dependency
            .get("rename")
            .and_then(Value::as_str)
            .unwrap_or(name)
            .replace('-', "_");
        aliases.insert(alias, name.to_string());
    }
    Ok(aliases)
}

/// Evaluate item-level `cfg` predicates against an active Cargo/rustc cfg snapshot.
///
/// `Some(true)` and `Some(false)` are exact for the standard `all`, `any`, `not`, key/value,
/// and bare-name forms. `None` means the source used a predicate form this bounded evaluator
/// does not understand; callers must preserve that branch rather than guessing.
pub fn cfg_predicates_active(predicates: &[String], active_cfg: &[String]) -> Option<bool> {
    let active = active_cfg
        .iter()
        .map(|item| normalize_cfg(item))
        .collect::<std::collections::HashSet<_>>();
    let mut unknown = false;
    for predicate in predicates {
        match eval_cfg(&normalize_cfg(predicate), &active) {
            Some(false) => return Some(false),
            Some(true) => {}
            None => unknown = true,
        }
    }
    (!unknown).then_some(true)
}

fn eval_cfg(predicate: &str, active: &std::collections::HashSet<String>) -> Option<bool> {
    if let Some(arguments) = outer_arguments(predicate, "all") {
        let parts = split_cfg_arguments(arguments)?;
        let mut unknown = false;
        for part in parts {
            match eval_cfg(part, active) {
                Some(false) => return Some(false),
                Some(true) => {}
                None => unknown = true,
            }
        }
        return (!unknown).then_some(true);
    }
    if let Some(arguments) = outer_arguments(predicate, "any") {
        let parts = split_cfg_arguments(arguments)?;
        let mut unknown = false;
        for part in parts {
            match eval_cfg(part, active) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        return (!unknown).then_some(false);
    }
    if let Some(arguments) = outer_arguments(predicate, "not") {
        let parts = split_cfg_arguments(arguments)?;
        let [inner] = parts.as_slice() else {
            return None;
        };
        return eval_cfg(inner, active).map(|value| !value);
    }
    if predicate.is_empty()
        || predicate.contains('(')
        || predicate.contains(')')
        || predicate.contains(',')
    {
        None
    } else {
        Some(active.contains(predicate))
    }
}

fn normalize_cfg(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut quoted = false;
    for character in value.chars() {
        if character == '"' {
            quoted = !quoted;
            normalized.push(character);
        } else if quoted || !character.is_whitespace() {
            normalized.push(character);
        }
    }
    normalized
}

fn outer_arguments<'a>(value: &'a str, name: &str) -> Option<&'a str> {
    value
        .strip_prefix(name)?
        .strip_prefix('(')?
        .strip_suffix(')')
}

fn split_cfg_arguments(value: &str) -> Option<Vec<&str>> {
    if value.is_empty() {
        return Some(Vec::new());
    }
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut start = 0usize;
    for (index, character) in value.char_indices() {
        match character {
            '"' => quoted = !quoted,
            '(' if !quoted => depth = depth.checked_add(1)?,
            ')' if !quoted => depth = depth.checked_sub(1)?,
            ',' if !quoted && depth == 0 => {
                parts.push(&value[start..index]);
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    if quoted || depth != 0 {
        return None;
    }
    parts.push(&value[start..]);
    Some(parts)
}

/// Walk upward from `start` until a `Cargo.toml` is found.
fn find_manifest(start: &Path) -> Result<PathBuf> {
    if !start.exists() {
        bail!("path does not exist: {}", start.display());
    }

    let mut dir = if start.is_file() {
        start
            .parent()
            .ok_or_else(|| anyhow!("path has no parent: {}", start.display()))?
            .to_path_buf()
    } else {
        start.to_path_buf()
    };

    // Prefer a stable absolute base when possible.
    if let Ok(canon) = dir.canonicalize() {
        dir = canon;
    }

    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.is_file() {
            return Ok(candidate);
        }
        if !dir.pop() {
            bail!("Cargo.toml not found starting from {}", start.display());
        }
    }
}

fn run_cargo_metadata(manifest_path: &Path, options: &CargoOptions) -> Result<Value> {
    let mut cmd = Command::new("cargo");
    cmd.arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(manifest_path);

    if options.all_features {
        cmd.arg("--all-features");
    }
    if options.no_default_features {
        cmd.arg("--no-default-features");
    }
    if !options.features.is_empty() {
        cmd.arg("--features").arg(options.features.join(","));
    }
    if let Some(target) = options.target.as_deref() {
        cmd.arg("--filter-platform").arg(target);
    }

    let output = cmd.output().with_context(|| {
        format!(
            "failed to spawn cargo metadata for {}",
            manifest_path.display()
        )
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "cargo metadata failed for {} (status {}): {}",
            manifest_path.display(),
            output.status,
            stderr.trim()
        );
    }

    let stdout = String::from_utf8(output.stdout).context("cargo metadata stdout was not UTF-8")?;
    serde_json::from_str(&stdout).context("failed to parse cargo metadata JSON")
}

fn select_package<'a>(metadata: &'a Value, manifest_path: &Path) -> Result<&'a Value> {
    let packages = metadata
        .get("packages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("cargo metadata missing packages array"))?;

    let want = normalize_path(manifest_path);

    for package in packages {
        let Some(mp) = package.get("manifest_path").and_then(|v| v.as_str()) else {
            continue;
        };
        if normalize_path(Path::new(mp)) == want {
            return Ok(package);
        }
    }

    bail!(
        "no package in cargo metadata matched manifest {}",
        manifest_path.display()
    )
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn path_from_json(obj: &Value, key: &str) -> Result<PathBuf> {
    let s = obj
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing string field `{key}`"))?;
    Ok(PathBuf::from(s))
}

fn path_from_root(metadata: &Value, key: &str) -> Result<PathBuf> {
    path_from_json(metadata, key).with_context(|| format!("cargo metadata missing `{key}`"))
}

fn parse_targets(package: &Value) -> Result<Vec<CargoTarget>> {
    let targets = package
        .get("targets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("package missing targets array"))?;

    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let name = target
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target missing name"))?
            .to_string();
        let kind = target
            .get("kind")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("target missing kind"))?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<Vec<_>>();
        let src_path = path_from_json(target, "src_path")?;
        out.push(CargoTarget {
            name,
            kind,
            src_path,
        });
    }
    Ok(out)
}

fn resolve_active_features(metadata: &Value, package_id: &str) -> Result<Vec<String>> {
    let resolve = metadata
        .get("resolve")
        .ok_or_else(|| anyhow!("cargo metadata missing resolve"))?;
    let nodes = resolve
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("cargo metadata resolve missing nodes"))?;

    for node in nodes {
        let id = node.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if id == package_id {
            let features = node
                .get("features")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("resolve node missing features for {package_id}"))?;
            return Ok(features
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect());
        }
    }

    bail!("resolve node not found for package id {package_id}")
}

fn collect_candle_versions(
    metadata: &Value,
    selected_package_id: &str,
) -> Result<BTreeMap<String, String>> {
    let packages = metadata
        .get("packages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("cargo metadata missing packages array"))?;

    let mut versions: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for package in packages {
        if package.get("id").and_then(Value::as_str) == Some(selected_package_id) {
            continue;
        }
        let name = match package.get("name").and_then(|v| v.as_str()) {
            Some(n) if n.starts_with("candle-") => n,
            _ => continue,
        };
        let version = package
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("package `{name}` missing version"))?;
        versions
            .entry(name.to_string())
            .or_default()
            .insert(version.to_string());
    }
    Ok(versions
        .into_iter()
        .map(|(name, versions)| (name, versions.into_iter().collect::<Vec<_>>().join(",")))
        .collect())
}

fn collect_rustc_cfgs(target: Option<&str>) -> Result<Vec<String>> {
    let mut cmd = Command::new("rustc");
    cmd.arg("--print").arg("cfg");
    if let Some(triple) = target {
        cmd.arg("--target").arg(triple);
    }

    let output = cmd.output().context("failed to spawn rustc --print cfg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "rustc --print cfg failed (status {}): {}",
            output.status,
            stderr.trim()
        );
    }

    let stdout =
        String::from_utf8(output.stdout).context("rustc --print cfg stdout was not UTF-8")?;
    let mut cfgs = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    cfgs.sort();
    cfgs.dedup();
    Ok(cfgs)
}
