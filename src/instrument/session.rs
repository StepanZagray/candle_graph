//! Representative-run profiler session — emits `candle-graph/trace/9` JSONL.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::capability::CaptureContract;
use crate::phase::ExecutionPhase;
use crate::trace::events::{
    DeviceIntervalEvent, DeviceMemoryEvent, EdgeEvent, GradientEvent, MemoryEvent, OpEvent,
    SpanEndEvent, SpanStartEvent, TensorEvent, TerminalEvent, TraceEvent,
};
use crate::trace::memory::{resolve_dense_tensor_bytes, MemoryAction};
use crate::trace::schema::{
    ComparisonIdentity, GradientState, RunOutcome, TimingMode, TraceRunMeta,
};

use super::span::{
    DeviceIntervalRecord, DeviceMemoryRecord, MemoryRecord, OpRecord, SpanGuard, SpanId, SpanKind,
    TensorRecord,
};

/// Streaming trace session writing TensorFlow-Profiler-style span JSONL.
pub struct TraceSession {
    path: PathBuf,
    inner: RefCell<SessionInner>,
}

/// Required provenance for one representative profile run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileRun {
    pub entrypoint: String,
    pub correlation_id: String,
    pub phase: ExecutionPhase,
    /// One-based selected update or inference invocation.
    pub capture_step: u64,
    pub warmup_steps: u64,
    pub device: String,
    pub measured_region_device_synchronized: bool,
    pub timing_mode: TimingMode,
    pub capture_contract: CaptureContract,
    pub comparison_identity: Option<ComparisonIdentity>,
    pub tags: BTreeMap<String, String>,
}

impl ProfileRun {
    pub fn training(
        entrypoint: impl Into<String>,
        capture_step: u64,
        device: impl Into<String>,
    ) -> Self {
        let entrypoint = entrypoint.into();
        Self {
            correlation_id: format!("{entrypoint}/update-{capture_step}"),
            entrypoint,
            phase: ExecutionPhase::Train,
            capture_step,
            warmup_steps: capture_step.saturating_sub(1),
            device: device.into(),
            measured_region_device_synchronized: false,
            timing_mode: TimingMode::Host,
            capture_contract: CaptureContract::default(),
            comparison_identity: None,
            tags: BTreeMap::new(),
        }
    }

    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn correlation_id(mut self, value: impl Into<String>) -> Self {
        self.correlation_id = value.into();
        self
    }

    pub fn device_synchronized(mut self) -> Self {
        self.timing_mode = TimingMode::DeviceSynchronized;
        self.measured_region_device_synchronized = true;
        self
    }

    /// Mark only the caller-controlled measured region as device-synchronized.
    /// Nested semantic spans remain host-timed.
    pub fn measured_region_device_synchronized(mut self) -> Self {
        self.measured_region_device_synchronized = true;
        self
    }

    pub fn inference(
        entrypoint: impl Into<String>,
        capture_step: u64,
        device: impl Into<String>,
    ) -> Self {
        let entrypoint = entrypoint.into();
        Self {
            correlation_id: format!("{entrypoint}/inference-{capture_step}"),
            entrypoint,
            phase: ExecutionPhase::Infer,
            capture_step,
            warmup_steps: capture_step.saturating_sub(1),
            device: device.into(),
            measured_region_device_synchronized: false,
            timing_mode: TimingMode::Host,
            capture_contract: CaptureContract::default(),
            comparison_identity: None,
            tags: BTreeMap::new(),
        }
    }

    pub fn capture_contract(mut self, contract: CaptureContract) -> Self {
        self.capture_contract = contract;
        self
    }

    pub fn comparison_identity(mut self, identity: ComparisonIdentity) -> Self {
        self.comparison_identity = Some(identity);
        self
    }
}

struct SessionInner {
    writer: io::BufWriter<File>,
    span_stack: Vec<u64>,
    next_span_id: u64,
    next_event_id: u64,
    id_buf: String,
    probe_started: Instant,
    sticky_error: Option<String>,
}

impl TraceSession {
    /// Open a trace and own its single root span until [`Self::finish`].
    pub fn open(path: impl AsRef<Path>, run: ProfileRun) -> Result<Self> {
        anyhow::ensure!(
            run.capture_step > 0,
            "capture_step must be one-based and greater than zero"
        );
        run.capture_contract
            .validate()
            .context("validate capture contract")?;
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
        let run_id = new_run_id();
        let meta = TraceRunMeta {
            run_id: run_id.clone(),
            correlation_id: run.correlation_id,
            entrypoint: run.entrypoint.clone(),
            phase: run.phase,
            timestamp: utc_iso8601_now(),
            capture_step: run.capture_step,
            warmup_steps: run.warmup_steps,
            device: run.device,
            measured_region_device_synchronized: run.measured_region_device_synchronized,
            timing_mode: run.timing_mode,
            capture_contract: run.capture_contract,
            comparison_identity: run.comparison_identity,
            tags: run.tags,
            candle_version: None,
        };
        write_event(&mut writer, &TraceEvent::meta(meta))?;
        write_event(
            &mut writer,
            &TraceEvent::SpanStart(SpanStartEvent {
                id: span_id_string(1),
                parent_id: None,
                name: run.entrypoint,
                start_ns: 0,
                kind: SpanKind::Function,
                measured: false,
                step: None,
            }),
        )?;
        Ok(Self {
            path,
            inner: RefCell::new(SessionInner {
                writer,
                span_stack: vec![1],
                next_span_id: 1,
                next_event_id: 0,
                id_buf: String::with_capacity(24),
                probe_started: Instant::now(),
                sticky_error: None,
            }),
        })
    }

    /// Begin a nested span; parent is the top of the session span stack (TF Profiler call tree).
    pub fn begin_span(&self, name: impl Into<String>, kind: SpanKind) -> SpanGuard<'_> {
        self.begin_span_inner(name, kind, None, false)
    }

    /// Begin the single caller-controlled region used for total-time comparisons.
    pub fn begin_measurement(&self, name: impl Into<String>) -> SpanGuard<'_> {
        self.begin_span_inner(name, SpanKind::Function, None, true)
    }

    /// Begin a span tagged with a PyTorch-style training step (`forward` / `backward` / `optimizer`).
    pub fn begin_step_span(
        &self,
        name: impl Into<String>,
        step: crate::phase::ExecutionStep,
        kind: SpanKind,
    ) -> SpanGuard<'_> {
        self.begin_span_inner(name, kind, Some(step), false)
    }

    /// Record already-completed host work without changing the live span stack.
    ///
    /// `start_ns` is a monotonic offset from this session's start and `duration_ns` must be
    /// positive. The interval must have completed before this call. The closed span is attached
    /// directly to the session root, so it may overlap live nested spans without implying a
    /// synchronous parent/child call relationship.
    pub fn record_completed_host_span(
        &self,
        name: impl Into<String>,
        kind: SpanKind,
        start_ns: u64,
        duration_ns: u64,
    ) -> Result<SpanId> {
        anyhow::ensure!(duration_ns > 0, "completed host span duration must be positive");
        let end_ns = start_ns
            .checked_add(duration_ns)
            .context("completed host span interval overflows u64 nanoseconds")?;
        anyhow::ensure!(
            end_ns <= self.elapsed_ns(),
            "completed host span ends in the future relative to the trace session"
        );

        let mut inner = self.inner.borrow_mut();
        if let Some(error) = &inner.sticky_error {
            anyhow::bail!("trace session previously failed: {error}");
        }
        let root_id = inner
            .span_stack
            .first()
            .copied()
            .context("trace session root span is missing")?;
        inner.next_span_id += 1;
        let span_id = SpanId(inner.next_span_id);
        let id = span_id_string(span_id.0);

        if let Err(error) = write_event(
            &mut inner.writer,
            &TraceEvent::SpanStart(SpanStartEvent {
                id: id.clone(),
                parent_id: Some(span_id_string(root_id)),
                name: name.into(),
                start_ns,
                kind,
                measured: false,
                step: None,
            }),
        ) {
            inner.sticky_error.get_or_insert_with(|| error.to_string());
            return Err(error);
        }
        if let Err(error) = write_event(
            &mut inner.writer,
            &TraceEvent::SpanEnd(SpanEndEvent { id, duration_ns }),
        ) {
            inner.sticky_error.get_or_insert_with(|| error.to_string());
            return Err(error);
        }
        Ok(span_id)
    }

    fn begin_span_inner(
        &self,
        name: impl Into<String>,
        kind: SpanKind,
        step: Option<crate::phase::ExecutionStep>,
        measured: bool,
    ) -> SpanGuard<'_> {
        let started = Instant::now();
        let start_ns = self.elapsed_ns();
        let mut inner = self.inner.borrow_mut();
        inner.next_span_id += 1;
        let span_id = inner.next_span_id;
        let parent_id = inner.span_stack.last().copied().map(span_id_string);

        format_span_id(&mut inner.id_buf, span_id);
        let id_str = inner.id_buf.clone();

        if let Err(error) = write_event(
            &mut inner.writer,
            &TraceEvent::SpanStart(SpanStartEvent {
                id: id_str,
                parent_id,
                name: name.into(),
                start_ns,
                kind,
                measured,
                step,
            }),
        ) {
            inner.sticky_error.get_or_insert_with(|| error.to_string());
        }

        inner.span_stack.push(span_id);

        SpanGuard {
            session: self,
            id: SpanId(span_id),
            started,
        }
    }

    pub(crate) fn end_span(&self, id: SpanId, duration_ns: u64) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        let expected = inner
            .span_stack
            .last()
            .copied()
            .with_context(|| format!("span stack underflow closing span {}", id.0))?;
        anyhow::ensure!(
            expected == id.0,
            "span_end id `{}` does not match open span `{}`",
            id.0,
            expected
        );
        inner.span_stack.pop();

        format_span_id(&mut inner.id_buf, id.0);
        let span_id = inner.id_buf.clone();
        if let Err(error) = write_event(
            &mut inner.writer,
            &TraceEvent::SpanEnd(SpanEndEvent {
                id: span_id,
                duration_ns,
            }),
        ) {
            inner.sticky_error.get_or_insert_with(|| error.to_string());
            return Err(error);
        }
        Ok(())
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.inner
            .borrow()
            .probe_started
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64
    }

    /// Convert an [`Instant`] from this process into this trace session's
    /// monotonic nanosecond clock without an independently sampled anchor.
    pub fn host_timestamp_ns(&self, instant: Instant) -> Result<u64> {
        let probe_started = self.inner.borrow().probe_started;
        let elapsed = instant
            .checked_duration_since(probe_started)
            .context("host instant predates the trace session")?;
        Ok(elapsed.as_nanos().min(u64::MAX as u128) as u64)
    }

    /// Record a timed op observation attached to `span_id`.
    pub fn record_op(&self, span_id: SpanId, op: OpRecord<'_>) -> Result<()> {
        let output_dense_bytes =
            resolve_dense_tensor_bytes(op.output_dense_bytes, op.shape, op.dtype);
        let timestamp_ns = if op.timestamp_ns > 0 {
            op.timestamp_ns
        } else {
            self.elapsed_ns().saturating_sub(op.duration_ns)
        };
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
                    output_dense_bytes,
                    input_dense_bytes: op.input_dense_bytes,
                }),
            )?;
        }

        Ok(())
    }

    /// Record tensor metadata. Logical allocation lifetimes require explicit memory events.
    pub fn record_tensor(&self, span_id: SpanId, tensor: TensorRecord<'_>) -> Result<()> {
        let dense_bytes =
            resolve_dense_tensor_bytes(tensor.dense_bytes, tensor.shape, tensor.dtype);
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Tensor(TensorEvent {
                span_id: span_id_string(span_id.0),
                tensor_id: tensor.tensor_id.into(),
                label: tensor.label.map(str::to_string),
                shape: tensor.shape.to_vec(),
                dtype: tensor.dtype.into(),
                device: tensor.device.into(),
                requires_grad: tensor.requires_grad,
                dense_bytes,
                category: tensor.category,
            }),
        )?;
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
                storage_id: mem.storage_id.into(),
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
                storage_id: mem.storage_id.into(),
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
    pub fn record_device_memory(&self, sample: DeviceMemoryRecord<'_>) -> Result<()> {
        anyhow::ensure!(
            sample.used_bytes.is_some()
                || sample.free_bytes.is_some()
                || sample.reserved_bytes.is_some()
                || sample.capacity_bytes.is_some(),
            "device-memory checkpoint must contain at least one observation"
        );
        let timestamp_ns = sample.timestamp_ns.unwrap_or_else(|| self.elapsed_ns());
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::DeviceMemory(DeviceMemoryEvent {
                timestamp_ns,
                device: sample.device.into(),
                used_bytes: sample.used_bytes,
                free_bytes: sample.free_bytes,
                reserved_bytes: sample.reserved_bytes,
                capacity_bytes: sample.capacity_bytes,
            }),
        )
    }

    pub fn record_device_interval(
        &self,
        span_id: SpanId,
        interval: DeviceIntervalRecord<'_>,
    ) -> Result<()> {
        anyhow::ensure!(
            interval.duration_ns > 0,
            "device interval duration must be positive"
        );
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::DeviceInterval(DeviceIntervalEvent {
                span_id: span_id_string(span_id.0),
                device: interval.device.into(),
                stream_id: interval.stream_id.into(),
                clock_id: interval.clock_id.into(),
                backend: interval.backend.into(),
                start_ns: interval.start_ns,
                duration_ns: interval.duration_ns,
            }),
        )
    }

    pub fn record_call_edge(&self, from: SpanId, to: SpanId, duration_ns: u64) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Edge(EdgeEvent::Call {
                from_span: span_id_string(from.0),
                to_span: span_id_string(to.0),
                host_duration_ns: duration_ns,
            }),
        )
    }

    pub fn record_data_edge(&self, from_tensor: &str, to_tensor: &str) -> Result<()> {
        anyhow::ensure!(
            !from_tensor.is_empty() && !to_tensor.is_empty(),
            "data-edge tensor IDs cannot be empty"
        );
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Edge(EdgeEvent::Data {
                from_tensor: from_tensor.into(),
                to_tensor: to_tensor.into(),
            }),
        )
    }

    /// Record one parameter gradient fact from a probe run.
    ///
    /// `Present` requires a finite positive norm, `Zero` requires positive zero, and `Missing` or
    /// `NonFinite` require `None`. Exact-contract captures emit one event per `(root, key)`.
    pub fn record_gradient(
        &self,
        root: impl Into<String>,
        key: impl Into<String>,
        state: GradientState,
        norm: Option<f64>,
    ) -> Result<()> {
        let root = root.into();
        let key = key.into();
        anyhow::ensure!(
            !root.trim().is_empty() && !key.trim().is_empty(),
            "gradient roots and parameter keys must not be empty"
        );
        anyhow::ensure!(
            state.norm_is_valid(norm),
            "gradient state `{state}` is inconsistent with norm {norm:?}"
        );
        let mut inner = self.inner.borrow_mut();
        inner.next_event_id += 1;
        let event_id = format!("gradient-{}", inner.next_event_id);
        write_event(
            &mut inner.writer,
            &TraceEvent::Gradient(GradientEvent {
                event_id,
                root,
                key,
                state,
                norm,
            }),
        )
    }

    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if let Some(error) = &inner.sticky_error {
            anyhow::bail!("trace session previously failed: {error}");
        }
        inner.writer.flush().context("flushing trace JSONL")
    }

    /// Close the owned root span, flush, and return the trace path.
    pub fn finish(self) -> Result<PathBuf> {
        let duration_ns = self.elapsed_ns();
        {
            let mut inner = self.inner.borrow_mut();
            if let Some(error) = &inner.sticky_error {
                anyhow::bail!("trace session previously failed: {error}");
            }
            anyhow::ensure!(
                inner.span_stack.as_slice() == [1],
                "cannot finish trace with {} nested spans still open",
                inner.span_stack.len().saturating_sub(1)
            );
            inner.span_stack.pop();
            write_event(
                &mut inner.writer,
                &TraceEvent::SpanEnd(SpanEndEvent {
                    id: span_id_string(1),
                    duration_ns,
                }),
            )?;
            write_event(
                &mut inner.writer,
                &TraceEvent::Terminal(TerminalEvent {
                    outcome: RunOutcome::Complete,
                    timestamp_ns: duration_ns,
                    reason: None,
                }),
            )?;
            inner.writer.flush().context("flushing trace JSONL")?;
        }
        Ok(self.path)
    }

    /// Finalize a diagnosable partial trace without presenting it as complete evidence.
    pub fn finish_failed(self, reason: impl Into<String>) -> Result<PathBuf> {
        let reason = reason.into();
        anyhow::ensure!(!reason.trim().is_empty(), "failure reason cannot be empty");
        let timestamp_ns = self.elapsed_ns();
        let mut inner = self.inner.borrow_mut();
        write_event(
            &mut inner.writer,
            &TraceEvent::Terminal(TerminalEvent {
                outcome: RunOutcome::Failed,
                timestamp_ns,
                reason: Some(reason),
            }),
        )?;
        inner
            .writer
            .flush()
            .context("flushing failed trace JSONL")?;
        drop(inner);
        Ok(self.path.clone())
    }
}

fn write_event<W: Write, T: Serialize>(writer: &mut W, event: &T) -> Result<()> {
    let mut line = serde_json::to_vec(event).context("serializing trace JSONL event")?;
    line.push(b'\n');
    writer.write_all(&line).context("writing trace JSONL event")
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
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
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
    use crate::trace::health::analyze_health;
    use serde_json::Value;
    use std::io::{BufRead, BufReader};

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
            TraceSession::open(&path, ProfileRun::training("model::forward", 1, "cpu")).unwrap();

        let inner_id = {
            let _outer = session.begin_measurement("Model::forward");
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
                    output_dense_bytes: None,
                    input_dense_bytes: 0,
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
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0].name, "model::forward");
        assert!(starts[0].parent_id.is_none());
        assert_eq!(starts[1].name, "Model::forward");
        assert_eq!(starts[1].parent_id.as_deref(), Some("s1"));
        assert_eq!(starts[2].name, "matmul");
        assert_eq!(starts[2].parent_id.as_deref(), Some("s2"));

        let ends = span_end_durations(&path);
        assert_eq!(ends.len(), 3);
        assert!(ends[0].1 > 0);

        let doc = parse_trace(&path).unwrap();
        assert_eq!(doc.run.entrypoint, "model::forward");
        assert_eq!(doc.ops.len(), 1);
        assert_eq!(doc.ops[0].output_dense_bytes, Some(8 * 8 * 4));
        assert!(
            doc.memory.is_empty(),
            "op metadata must not fabricate tensor lifetime"
        );
    }

    #[test]
    fn measured_region_sync_does_not_overstate_nested_span_timing() {
        let run = ProfileRun::training("train::update", 2, "cuda:0")
            .measured_region_device_synchronized();

        assert!(run.measured_region_device_synchronized);
        assert_eq!(run.timing_mode, TimingMode::Host);
    }

    #[test]
    fn record_gradient_round_trips_through_trace_parser() {
        let path = temp_trace("gradient");
        let session =
            TraceSession::open(&path, ProfileRun::training("train::loss", 1, "cpu")).unwrap();
        session
            .record_gradient("vb", "encoder.weight", GradientState::Present, Some(0.42))
            .unwrap();
        session.finish().unwrap();

        let doc = parse_trace(&path).unwrap();
        assert_eq!(doc.gradients.len(), 1);
        assert_eq!(doc.gradients[0].key, "encoder.weight");
    }

    #[test]
    fn completed_host_span_is_root_attached_stack_neutral_and_overlap_valid() {
        let path = temp_trace("completed-host-span");
        let session =
            TraceSession::open(&path, ProfileRun::inference("serve::request", 1, "cpu")).unwrap();

        let measured = session.begin_measurement("request");
        let live = session.begin_span("main-thread", SpanKind::Function);
        let stack_before = session.inner.borrow().span_stack.clone();
        let start_ns = session.elapsed_ns();
        let end_ns = loop {
            let elapsed = session.elapsed_ns();
            if elapsed > start_ns {
                break elapsed;
            }
            std::hint::spin_loop();
        };
        let duration_ns = end_ns - start_ns;

        let completed_id = session
            .record_completed_host_span(
                "background-work",
                SpanKind::Function,
                start_ns,
                duration_ns,
            )
            .unwrap();
        assert_eq!(session.inner.borrow().span_stack, stack_before);

        drop(live);
        drop(measured);
        session.finish().unwrap();

        let doc = parse_trace(&path).unwrap();
        let completed = doc
            .spans
            .iter()
            .find(|span| span.id == span_id_string(completed_id.0))
            .unwrap();
        let measured = doc.spans.iter().find(|span| span.measured).unwrap();
        assert_eq!(completed.parent_id.as_deref(), Some("s1"));
        assert_eq!(completed.start_ns, start_ns);
        assert_eq!(completed.duration_ns, duration_ns);
        assert!(completed.closed);
        assert!(completed.start_ns < measured.start_ns.saturating_add(measured.duration_ns));
        assert!(measured.start_ns < completed.start_ns.saturating_add(completed.duration_ns));

        let health = analyze_health(&doc);
        assert!(health.structurally_valid, "{:?}", health.issues);
        assert!(health.capture_complete);
    }

    #[test]
    fn completed_host_span_requires_positive_duration() {
        let path = temp_trace("completed-host-span-zero");
        let session =
            TraceSession::open(&path, ProfileRun::inference("serve::request", 1, "cpu")).unwrap();
        let error = session
            .record_completed_host_span("empty", SpanKind::Function, 0, 0)
            .unwrap_err();
        assert!(error.to_string().contains("duration must be positive"));

        let overflow = session
            .record_completed_host_span(
                "overflow",
                SpanKind::Function,
                u64::MAX,
                1,
            )
            .unwrap_err();
        assert!(overflow.to_string().contains("overflows"));

        let future = session
            .record_completed_host_span(
                "future",
                SpanKind::Function,
                session.elapsed_ns().saturating_add(60_000_000_000),
                1,
            )
            .unwrap_err();
        assert!(future.to_string().contains("ends in the future"));

        let before_session = session
            .inner
            .borrow()
            .probe_started
            .checked_sub(std::time::Duration::from_nanos(1))
            .unwrap();
        assert!(session.host_timestamp_ns(before_session).is_err());
        let now = Instant::now();
        assert!(session.host_timestamp_ns(now).unwrap() <= session.elapsed_ns());
        session.finish().unwrap();
    }
}
