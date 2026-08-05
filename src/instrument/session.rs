//! Post-run trace profiler session — emits `candle-graph/trace/5` JSONL.
//!
//! Not for hot training loops: run once after a representative forward/loss pass.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::phase::ExecutionPhase;
use crate::trace::memory::category_for_step;
use crate::trace::events::{
    DeviceMemoryEvent, GradientEvent, MemoryEvent, OpEvent, SpanEndEvent, SpanStartEvent,
    TensorEvent, TraceEvent,
};
use crate::trace::memory::{resolve_storage_bytes, MemoryAction};
use crate::trace::schema::{GradientState, TraceRunMeta};

use super::span::{MemoryRecord, OpRecord, SpanGuard, SpanId, SpanKind, TensorRecord};

/// Streaming trace session writing TensorFlow-Profiler-style span JSONL.
pub struct TraceSession {
    path: PathBuf,
    inner: RefCell<SessionInner>,
}

struct SessionInner {
    writer: io::BufWriter<File>,
    span_stack: Vec<u64>,
    span_steps: Vec<Option<crate::phase::ExecutionStep>>,
    next_span_id: u64,
    next_event_id: u64,
    id_buf: String,
    probe_started: Instant,
}

impl TraceSession {
    /// Open (or truncate) a JSONL trace at `path` and write the required `meta` event.
    pub fn open(
        path: impl AsRef<Path>,
        entrypoint: impl Into<String>,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create trace dir {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("open trace {}", path.display()))?;
        let mut writer = io::BufWriter::new(file);
        let entrypoint = entrypoint.into();
        let run_id = new_run_id();
        let meta = TraceRunMeta {
            run_id: run_id.clone(),
            entrypoint: entrypoint.clone(),
            phase: phase.as_str().to_string(),
            timestamp: utc_iso8601_now(),
            candle_version: None,
        };
        write_event(
            &mut writer,
            &TraceEvent::meta(meta),
        )?;
        Ok(Self {
            path,
            inner: RefCell::new(SessionInner {
                writer,
                span_stack: Vec::with_capacity(16),
                span_steps: Vec::with_capacity(16),
                next_span_id: 0,
                next_event_id: 0,
                id_buf: String::with_capacity(24),
                probe_started: Instant::now(),
            }),
        })
    }

    /// Begin a nested span; parent is the top of the session span stack (TF Profiler call tree).
    pub fn begin_span(&self, name: impl Into<String>, kind: SpanKind) -> SpanGuard<'_> {
        self.begin_span_inner(name, kind, None)
    }

    /// Begin a span tagged with a PyTorch-style training step (`forward` / `backward` / `optimizer`).
    pub fn begin_step_span(
        &self,
        name: impl Into<String>,
        step: crate::phase::ExecutionStep,
        kind: SpanKind,
    ) -> SpanGuard<'_> {
        self.begin_span_inner(name, kind, Some(step))
    }

    fn begin_span_inner(
        &self,
        name: impl Into<String>,
        kind: SpanKind,
        step: Option<crate::phase::ExecutionStep>,
    ) -> SpanGuard<'_> {
        let mut inner = self.inner.borrow_mut();
        inner.next_span_id += 1;
        let span_id = inner.next_span_id;
        let parent_id = inner
            .span_stack
            .last()
            .copied()
            .map(span_id_string);

        format_span_id(&mut inner.id_buf, span_id);
        let id_str = inner.id_buf.clone();

        write_event(
            &mut inner.writer,
            &TraceEvent::SpanStart(SpanStartEvent {
                id: id_str,
                parent_id,
                name: name.into(),
                kind,
                step,
            }),
        )
        .expect("span_start write");

        inner.span_stack.push(span_id);
        inner.span_steps.push(step);

        SpanGuard {
            session: self,
            id: SpanId(span_id),
            started: Instant::now(),
        }
    }

    fn current_step(&self) -> Option<crate::phase::ExecutionStep> {
        self.inner
            .borrow()
            .span_steps
            .iter()
            .rev()
            .find_map(|step| *step)
    }

    pub(crate) fn end_span(&self, id: SpanId, duration_ns: u64) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        let expected = inner
            .span_stack
            .pop()
            .with_context(|| format!("span stack underflow closing span {}", id.0))?;
        inner.span_steps.pop();
        anyhow::ensure!(
            expected == id.0,
            "span_end id `{}` does not match open span `{}`",
            id.0,
            expected
        );

        format_span_id(&mut inner.id_buf, id.0);
        let span_id = inner.id_buf.clone();
        write_event(
            &mut inner.writer,
            &TraceEvent::SpanEnd(SpanEndEvent {
                id: span_id,
                duration_ns,
            }),
        )
    }

    fn elapsed_ns(&self) -> u64 {
        self.inner
            .borrow()
            .probe_started
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64
    }

    /// Record a timed op observation attached to `span_id`.
    pub fn record_op(&self, span_id: SpanId, op: OpRecord<'_>) -> Result<()> {
        let storage_bytes = resolve_storage_bytes(op.storage_bytes, op.shape, op.dtype);
        let timestamp_ns = if op.timestamp_ns > 0 {
            op.timestamp_ns
        } else {
            self.elapsed_ns()
        };
        let category = op
            .category
            .unwrap_or_else(|| category_for_step(self.current_step(), false));
        {
            let mut inner = self.inner.borrow_mut();
            write_event(
                &mut inner.writer,
                &TraceEvent::Op(OpEvent {
                    span_id: span_id_string(span_id.0),
                    op_name: op.op_name.into(),
                    inputs: op.inputs.to_vec(),
                    output: op.output.map(str::to_string),
                    shape: op.shape.to_vec(),
                    dtype: op.dtype.into(),
                    device: op.device.into(),
                    duration_ns: op.duration_ns,
                    timestamp_ns,
                    storage_bytes: Some(storage_bytes),
                    input_storage_bytes: op.input_storage_bytes,
                }),
            )?;
        }

        if storage_bytes > 0 {
            if let Some(output) = op.output {
                self.record_memory_alloc(
                    span_id,
                    MemoryRecord {
                        tensor_id: output,
                        device: op.device,
                        bytes: storage_bytes,
                        dtype: op.dtype,
                        category,
                        timestamp_ns: Some(timestamp_ns),
                        op_name: Some(op.op_name),
                        shape: op.shape,
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Record tensor metadata and an allocation event.
    pub fn record_tensor(&self, span_id: SpanId, tensor: TensorRecord<'_>) -> Result<()> {
        let storage_bytes = resolve_storage_bytes(tensor.storage_bytes, tensor.shape, tensor.dtype);
        let timestamp_ns = self.elapsed_ns();
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Tensor(TensorEvent {
                span_id: span_id_string(span_id.0),
                tensor_id: tensor.tensor_id.into(),
                shape: tensor.shape.to_vec(),
                dtype: tensor.dtype.into(),
                device: tensor.device.into(),
                requires_grad: tensor.requires_grad,
                storage_bytes: Some(storage_bytes),
                category: tensor.category,
            }),
        )?;
        drop(inner);

        if storage_bytes > 0 {
            self.record_memory_alloc(
                span_id,
                MemoryRecord {
                    tensor_id: tensor.tensor_id,
                    device: tensor.device,
                    bytes: storage_bytes,
                    dtype: tensor.dtype,
                    category: tensor.category,
                    timestamp_ns: Some(timestamp_ns),
                    op_name: None,
                    shape: tensor.shape,
                },
            )?;
        }
        Ok(())
    }

    /// Record an explicit tensor allocation (TensorFlow memory timeline).
    pub fn record_memory_alloc(&self, span_id: SpanId, mem: MemoryRecord<'_>) -> Result<()> {
        let timestamp_ns = mem.timestamp_ns.unwrap_or_else(|| self.elapsed_ns());
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Memory(MemoryEvent {
                timestamp_ns,
                tensor_id: mem.tensor_id.into(),
                span_id: span_id_string(span_id.0),
                op_name: mem.op_name.map(str::to_string),
                device: mem.device.into(),
                bytes: mem.bytes,
                action: MemoryAction::Alloc,
                shape: mem.shape.to_vec(),
                dtype: mem.dtype.into(),
                category: mem.category,
            }),
        )
    }

    /// Record an explicit tensor deallocation.
    pub fn record_memory_free(&self, span_id: SpanId, mem: MemoryRecord<'_>) -> Result<()> {
        let timestamp_ns = mem.timestamp_ns.unwrap_or_else(|| self.elapsed_ns());
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Memory(MemoryEvent {
                timestamp_ns,
                tensor_id: mem.tensor_id.into(),
                span_id: span_id_string(span_id.0),
                op_name: mem.op_name.map(str::to_string),
                device: mem.device.into(),
                bytes: mem.bytes,
                action: MemoryAction::Free,
                shape: mem.shape.to_vec(),
                dtype: mem.dtype.into(),
                category: mem.category,
            }),
        )
    }

    /// Record a device-level memory checkpoint (cudaMemGetInfo-style).
    pub fn record_device_memory(
        &self,
        device: impl Into<String>,
        used_bytes: u64,
        free_bytes: u64,
        timestamp_ns: Option<u64>,
    ) -> Result<()> {
        let timestamp_ns = timestamp_ns.unwrap_or_else(|| self.elapsed_ns());
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::DeviceMemory(DeviceMemoryEvent {
                timestamp_ns,
                device: device.into(),
                used_bytes,
                free_bytes,
                reserved_bytes: None,
            }),
        )
    }

    /// Record one parameter gradient fact from a probe run.
    pub fn record_gradient(
        &self,
        root: impl Into<String>,
        key: impl Into<String>,
        state: GradientState,
        norm: Option<f64>,
    ) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        inner.next_event_id += 1;
        let event_id = format!("gradient-{}", inner.next_event_id);
        write_event(
            &mut inner.writer,
            &TraceEvent::Gradient(GradientEvent {
                event_id,
                root: root.into(),
                key: key.into(),
                state,
                step: Some(0),
                norm,
            }),
        )
    }

    pub fn flush(&self) -> Result<()> {
        self.inner
            .borrow_mut()
            .writer
            .flush()
            .context("flushing trace JSONL")
    }

    /// Flush and return the trace path.
    pub fn finish(self) -> Result<PathBuf> {
        self.inner
            .borrow_mut()
            .writer
            .flush()
            .context("flushing trace JSONL")?;
        Ok(self.path)
    }
}

fn write_event<W: Write, T: Serialize>(writer: &mut W, event: &T) -> Result<()> {
    let mut line = serde_json::to_vec(event).context("serializing trace JSONL event")?;
    line.push(b'\n');
    writer
        .write_all(&line)
        .context("writing trace JSONL event")
}

fn format_span_id(buf: &mut String, id: u64) {
    buf.clear();
    use std::fmt::Write as _;
    let _ = write!(buf, "s{id}");
}

fn span_id_string(id: u64) -> String {
    format!("s{id}")
}

fn new_run_id() -> String {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("run-{pid}-{nanos}")
}

fn utc_iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch");
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let (year, month, day, hour, minute, second) = unix_secs_to_utc(secs);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z"
    )
}

/// Convert Unix seconds to UTC calendar components (Gregorian).
fn unix_secs_to_utc(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    const SECS_PER_DAY: u64 = 86_400;
    let days = secs / SECS_PER_DAY;
    let rem = secs % SECS_PER_DAY;
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;

    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::document::parse_trace;
    use crate::trace::events::SpanStartEvent;
    use std::io::{BufRead, BufReader};
    use serde_json::Value;

    fn temp_trace(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "candle-graph-trace-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn span_end_durations(path: &Path) -> Vec<(String, u64)> {
        let file = std::fs::File::open(path).unwrap();
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| line.unwrap())
            .filter_map(|line| {
                let value: Value = serde_json::from_str(&line).unwrap();
                if value.get("kind")?.as_str()? != "span_end" {
                    return None;
                }
                Some((
                    value["id"].as_str().unwrap().to_string(),
                    value["duration_ns"].as_u64().unwrap(),
                ))
            })
            .collect()
    }

    fn read_events(path: &Path) -> Vec<TraceEvent> {
        let file = std::fs::File::open(path).unwrap();
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| {
                let line = line.unwrap();
                serde_json::from_str(&line).unwrap_or_else(|err| {
                    panic!("invalid JSONL line `{line}`: {err}");
                })
            })
            .collect()
    }

    #[test]
    fn nested_spans_emit_parent_hierarchy_and_durations() {
        let path = temp_trace("nested");
        let session =
            TraceSession::open(&path, "model::forward", ExecutionPhase::Train).unwrap();

        let inner_id = {
            let _outer = session.begin_span("Model::forward", SpanKind::Function);
            std::thread::sleep(std::time::Duration::from_micros(50));
            let inner = session.begin_span("matmul", SpanKind::Op);
            std::thread::sleep(std::time::Duration::from_micros(50));
            inner.id
        };

        session
            .record_op(
                inner_id,
                OpRecord {
                    op_name: "matmul",
                    inputs: &["a".into(), "b".into()],
                    output: Some("c"),
                    shape: &[8, 8],
                    dtype: "f32",
                    device: "cpu",
                    duration_ns: 1200,
                    timestamp_ns: 0,
                    storage_bytes: None,
                    input_storage_bytes: 0,
                    category: None,
                },
            )
            .unwrap();

        session.finish().unwrap();

        let events = read_events(&path);
        assert!(matches!(events.first(), Some(TraceEvent::Meta { .. })));

        let starts: Vec<&SpanStartEvent> = events
            .iter()
            .filter_map(|event| match event {
                TraceEvent::SpanStart(start) => Some(start),
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0].name, "Model::forward");
        assert!(starts[0].parent_id.is_none());
        assert_eq!(starts[1].name, "matmul");
        assert_eq!(starts[1].parent_id.as_deref(), Some("s1"));

        let ends = span_end_durations(&path);
        assert_eq!(ends.len(), 2);
        assert!(ends[0].1 > 0);

        let doc = parse_trace(&path).unwrap();
        assert_eq!(doc.run.entrypoint, "model::forward");
        assert_eq!(doc.ops.len(), 1);
        assert_eq!(doc.ops[0].storage_bytes, Some(8 * 8 * 4));
        assert!(!doc.memory.is_empty());
    }

    #[test]
    fn record_gradient_round_trips_through_trace_parser() {
        let path = temp_trace("gradient");
        let session =
            TraceSession::open(&path, "train::loss", ExecutionPhase::Train).unwrap();
        session
            .record_gradient(
                "vb",
                "encoder.weight",
                GradientState::Present,
                Some(0.42),
            )
            .unwrap();
        session.finish().unwrap();

        let doc = parse_trace(&path).unwrap();
        assert_eq!(doc.gradients.len(), 1);
        assert_eq!(doc.gradients[0].key, "encoder.weight");
    }
}
