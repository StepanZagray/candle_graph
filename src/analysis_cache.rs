//! Optional on-disk cache for analyzed [`crate::model_ir::ModelIr`] documents.
//!
//! Enabled when `CANDLE_GRAPH_CACHE=1` or `--cache` is passed. Cache files live under
//! `CANDLE_GRAPH_CACHE_DIR` (default: `$XDG_CACHE_HOME/candle-graph` or `/tmp/candle-graph-cache`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::model_ir::ModelIr;

pub fn cache_enabled(explicit: bool) -> bool {
    explicit || std::env::var("CANDLE_GRAPH_CACHE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

pub fn cache_dir() -> PathBuf {
    std::env::var("CANDLE_GRAPH_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("XDG_CACHE_HOME")
                .map(|home| PathBuf::from(home).join("candle-graph"))
                .or_else(|_| std::env::var("HOME").map(|home| PathBuf::from(home).join(".cache/candle-graph")))
        })
        .unwrap_or_else(|_| PathBuf::from("/tmp/candle-graph-cache"))
}

pub fn cache_path(analysis_id: &str) -> PathBuf {
    let safe = analysis_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    cache_dir().join(format!("{safe}.json"))
}

pub fn load(path: &Path) -> Result<Option<ModelIr>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading analysis cache {}", path.display()))?;
    let model: ModelIr =
        serde_json::from_str(&text).context("parsing cached model IR")?;
    Ok(Some(model))
}

pub fn save(path: &Path, model: &ModelIr) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating cache dir {}", parent.display()))?;
    }
    let text = serde_json::to_string(model).context("serializing model IR for cache")?;
    std::fs::write(path, text).with_context(|| format!("writing analysis cache {}", path.display()))?;
    Ok(())
}
