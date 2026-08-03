//! Runtime profiler: attach during train or inference to emit timed operation/edge traces.
//!
//! Writes `candle-graph/runtime/3` JSONL compatible with [`crate::runtime`] and mergeable
//! into the unified model IR via static `static_id` correlation.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::phase::ExecutionPhase;
use crate::runtime::{
    EdgeTimingObservation, OperationObservation, RunMetadata, RuntimeTraceWriter, SCHEMA_V3,
};

/// Configuration for a profiling session started alongside train or inference.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    pub entrypoint: String,
    pub phase: ExecutionPhase,
    pub profile: String,
    pub cargo_features: Vec<String>,
    pub analysis_id: Option<String>,
    pub build_id: Option<String>,
}

impl ProfileConfig {
    pub fn train(entrypoint: impl Into<String>) -> Self {
        Self {
            entrypoint: entrypoint.into(),
            phase: ExecutionPhase::Train,
            profile: "release".into(),
            cargo_features: Vec::new(),
            analysis_id: None,
            build_id: None,
        }
    }

    pub fn infer(entrypoint: impl Into<String>) -> Self {
        Self {
            entrypoint: entrypoint.into(),
            phase: ExecutionPhase::Infer,
            profile: "release".into(),
            cargo_features: Vec::new(),
            analysis_id: None,
            build_id: None,
        }
    }
}

struct ActiveOp {
    static_id: String,
    op: String,
    inputs: Vec<String>,
    started: Instant,
}

/// Streaming profiler session writing JSONL runtime v3 events.
pub struct ProfileSession {
    path: PathBuf,
    writer: RuntimeTraceWriter<io::BufWriter<std::fs::File>>,
    phase: ExecutionPhase,
    step: u64,
    active: HashMap<String, ActiveOp>,
    event_counter: u64,
}

impl ProfileSession {
    /// Open (or truncate) a JSONL profile trace at `path`.
    pub fn open(path: impl AsRef<Path>, config: ProfileConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create profile dir {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("open profile trace {}", path.display()))?;
        let run = RunMetadata {
            entrypoint: config.entrypoint,
            profile: config.profile,
            cargo_features: config.cargo_features,
            cfg: Vec::new(),
            analysis_id: config.analysis_id,
            build_id: config.build_id,
            phase: Some(config.phase.as_str().to_string()),
        };
        let writer = RuntimeTraceWriter::new_with_schema(io::BufWriter::new(file), SCHEMA_V3, run)?;
        Ok(Self {
            path,
            writer,
            phase: config.phase,
            step: 0,
            active: HashMap::new(),
            event_counter: 0,
        })
    }

    pub fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn set_step(&mut self, step: u64) {
        self.step = step;
    }

    /// Begin timing an operation; returns an `event_id` for [`Self::end_operation`].
    pub fn begin_operation(
        &mut self,
        static_id: impl Into<String>,
        op: impl Into<String>,
        inputs: &[String],
    ) -> Result<String> {
        self.event_counter += 1;
        let event_id = format!("{}-op-{}-{}", self.phase.as_str(), self.step, self.event_counter);
        self.active.insert(
            event_id.clone(),
            ActiveOp {
                static_id: static_id.into(),
                op: op.into(),
                inputs: inputs.to_vec(),
                started: Instant::now(),
            },
        );
        Ok(event_id)
    }

    /// End a timed operation and emit a runtime v3 operation observation.
    pub fn end_operation(&mut self, event_id: &str, output: Option<String>) -> Result<u64> {
        let active = self
            .active
            .remove(event_id)
            .with_context(|| format!("unknown profile operation `{event_id}`"))?;
        let duration_ns = active.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.writer.operation(OperationObservation {
            event_id: event_id.to_string(),
            op: active.op,
            static_id: Some(active.static_id),
            source: None,
            inputs: active.inputs,
            output,
            step: Some(self.step),
            duration_ns: Some(duration_ns),
        })?;
        Ok(duration_ns)
    }

    /// Record average-friendly edge timing between two static tensor/operation ids.
    pub fn record_edge_timing(
        &mut self,
        from_static_id: impl Into<String>,
        to_static_id: impl Into<String>,
        duration_ns: u64,
    ) -> Result<()> {
        self.event_counter += 1;
        let event_id = format!(
            "{}-edge-{}-{}",
            self.phase.as_str(),
            self.step,
            self.event_counter
        );
        self.writer.edge_timing(EdgeTimingObservation {
            event_id,
            from_static_id: from_static_id.into(),
            to_static_id: to_static_id.into(),
            duration_ns,
            step: Some(self.step),
        })
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    pub fn finish(self) -> Result<PathBuf> {
        self.writer.finish()?;
        Ok(self.path)
    }
}

/// Convenience RAII guard for [`ProfileSession::begin_operation`] / [`ProfileSession::end_operation`].
pub struct TimedOperation<'a> {
    session: &'a mut ProfileSession,
    event_id: String,
}

impl<'a> TimedOperation<'a> {
    pub fn begin(
        session: &'a mut ProfileSession,
        static_id: impl Into<String>,
        op: impl Into<String>,
        inputs: &[String],
    ) -> Result<Self> {
        let event_id = session.begin_operation(static_id, op, inputs)?;
        Ok(Self { session, event_id })
    }
}

impl Drop for TimedOperation<'_> {
    fn drop(&mut self) {
        let _ = self.session.end_operation(&self.event_id, None);
    }
}
