//! Trace-file CLI helpers.

pub mod trace_cli;

use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::Path;

pub fn write_output(path: Option<&Path>, bytes: &[u8]) -> Result<()> {
    match path {
        Some(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        }
        None => io::stdout().lock().write_all(bytes)?,
    }
    Ok(())
}
