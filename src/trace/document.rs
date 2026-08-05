//! Aggregate trace document and JSONL I/O for `candle-graph/trace/5`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::events::{
    DeviceMemoryEvent, EdgeEvent, GradientEvent, MemoryEvent, OpEvent, SpanEndEvent,
    SpanStartEvent, TensorEvent, TraceEvent,
};
use super::memory::{resolve_storage_bytes, MemoryAction};
use super::schema::{SpanRecord, TraceRunMeta, TraceSummary, SCHEMA};

/// Full trace document assembled from JSONL events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceDocument {
    pub schema: String,
    pub run: TraceRunMeta,
    #[serde(default)]
    pub spans: Vec<SpanRecord>,
    #[serde(default)]
    pub ops: Vec<OpEvent>,
    #[serde(default)]
    pub tensors: Vec<TensorEvent>,
    #[serde(default)]
    pub memory: Vec<MemoryEvent>,
    #[serde(default)]
    pub device_memory: Vec<DeviceMemoryEvent>,
    #[serde(default)]
    pub gradients: Vec<GradientEvent>,
    #[serde(default)]
    pub edges: Vec<EdgeEvent>,
}

impl TraceDocument {
    /// Build a document from an ordered event stream (meta must be first).
    pub fn from_events(events: impl IntoIterator<Item = TraceEvent>) -> Result<Self> {
        let mut schema: Option<String> = None;
        let mut run: Option<TraceRunMeta> = None;
        let mut span_starts: BTreeMap<String, SpanStartEvent> = BTreeMap::new();
        let mut span_durations: HashMap<String, u64> = HashMap::new();
        let mut span_closed: HashSet<String> = HashSet::new();
        let mut ops = Vec::new();
        let mut tensors = Vec::new();
        let mut memory = Vec::new();
        let mut device_memory = Vec::new();
        let mut gradients = Vec::new();
        let mut edges = Vec::new();

        for (index, event) in events.into_iter().enumerate() {
            match event {
                TraceEvent::Meta { schema: s, run: meta } => {
                    if schema.is_some() || run.is_some() {
                        bail!(
                            "duplicate meta event at index {index}; only one meta record is allowed"
                        );
                    }
                    schema = Some(s);
                    run = Some(meta);
                }
                TraceEvent::SpanStart(start) => {
                    if span_starts.contains_key(&start.id) {
                        bail!("duplicate span_start id `{}` at index {index}", start.id);
                    }
                    span_starts.insert(start.id.clone(), start);
                }
                TraceEvent::SpanEnd(SpanEndEvent { id, duration_ns }) => {
                    if !span_starts.contains_key(&id) {
                        bail!("span_end for unknown span `{id}` at index {index}");
                    }
                    span_closed.insert(id.clone());
                    span_durations.insert(id, duration_ns);
                }
                TraceEvent::Op(mut op) => {
                    op.storage_bytes = Some(resolve_storage_bytes(
                        op.storage_bytes,
                        &op.shape,
                        &op.dtype,
                    ));
                    ops.push(op);
                }
                TraceEvent::Tensor(mut tensor) => {
                    tensor.storage_bytes = Some(resolve_storage_bytes(
                        tensor.storage_bytes,
                        &tensor.shape,
                        &tensor.dtype,
                    ));
                    tensors.push(tensor);
                }
                TraceEvent::Memory(mem) => memory.push(mem),
                TraceEvent::DeviceMemory(snapshot) => device_memory.push(snapshot),
                TraceEvent::Gradient(gradient) => gradients.push(gradient),
                TraceEvent::Edge(edge) => edges.push(edge),
            }
        }

        let schema = schema.unwrap_or_else(|| SCHEMA.to_string());
        let run = run.context("trace stream is missing a meta event with run metadata")?;

        if schema != SCHEMA {
            bail!(
                "unsupported trace schema {schema:?}; expected {SCHEMA:?}"
            );
        }

        let mut spans: Vec<SpanRecord> = span_starts
            .into_iter()
            .map(|(id, start)| SpanRecord {
                id: id.clone(),
                parent_id: start.parent_id,
                name: start.name,
                kind: start.kind,
                closed: span_closed.contains(&id),
                duration_ns: span_durations.get(&id).copied().unwrap_or(0),
                step: start.step,
            })
            .collect();
        spans.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(Self {
            schema,
            run,
            spans,
            ops,
            tensors,
            memory,
            device_memory,
            gradients,
            edges,
        })
    }

    /// Profiler summary: op count, total wall time, span tree shape, memory totals.
    pub fn build_summary(&self) -> TraceSummary {
        let op_count = self.ops.len();
        let total_ns = self.ops.iter().map(|op| op.duration_ns).sum();
        let span_count = self.spans.len();
        let root_span_count = self
            .spans
            .iter()
            .filter(|span| span.parent_id.is_none())
            .count();

        let parent_by_id: HashMap<&str, Option<&str>> = self
            .spans
            .iter()
            .map(|span| (span.id.as_str(), span.parent_id.as_deref()))
            .collect();

        let mut max_depth = 0usize;
        for span in &self.spans {
            let mut depth = 0usize;
            let mut current_parent = span.parent_id.as_deref();
            let mut seen = HashSet::new();
            while let Some(parent_id) = current_parent {
                if !seen.insert(parent_id) {
                    break;
                }
                depth += 1;
                current_parent = parent_by_id
                    .get(parent_id)
                    .copied()
                    .flatten();
            }
            max_depth = max_depth.max(depth);
        }

        let alloc_count = self
            .memory
            .iter()
            .filter(|event| event.action == MemoryAction::Alloc)
            .count();
        let free_count = self
            .memory
            .iter()
            .filter(|event| event.action == MemoryAction::Free)
            .count();

        let peak_bytes = super::memory::analyze_memory(self).summary.peak_bytes;

        TraceSummary {
            op_count,
            total_ns,
            span_count,
            root_span_count,
            max_depth,
            alloc_count,
            free_count,
            peak_bytes,
        }
    }

    /// Flatten the document back into JSONL events (meta first, then body in stable order).
    pub fn to_events(&self) -> Vec<TraceEvent> {
        let mut events = vec![TraceEvent::Meta {
            schema: self.schema.clone(),
            run: self.run.clone(),
        }];

        let mut span_ids: Vec<_> = self.spans.iter().map(|span| span.id.as_str()).collect();
        span_ids.sort_unstable();
        for id in span_ids {
            let span = self
                .spans
                .iter()
                .find(|span| span.id == id)
                .expect("sorted id must exist");
            events.push(TraceEvent::SpanStart(SpanStartEvent {
                id: span.id.clone(),
                parent_id: span.parent_id.clone(),
                name: span.name.clone(),
                kind: span.kind,
                step: span.step,
            }));
            if span.closed {
                events.push(TraceEvent::SpanEnd(SpanEndEvent {
                    id: span.id.clone(),
                    duration_ns: span.duration_ns,
                }));
            }
        }

        events.extend(self.ops.iter().cloned().map(TraceEvent::Op));
        events.extend(self.tensors.iter().cloned().map(TraceEvent::Tensor));
        events.extend(self.memory.iter().cloned().map(TraceEvent::Memory));
        events.extend(
            self.device_memory
                .iter()
                .cloned()
                .map(TraceEvent::DeviceMemory),
        );
        events.extend(self.gradients.iter().cloned().map(TraceEvent::Gradient));
        events.extend(self.edges.iter().cloned().map(TraceEvent::Edge));
        events
    }
}

/// Parse a JSONL trace file into a [`TraceDocument`].
pub fn parse_trace(path: impl AsRef<Path>) -> Result<TraceDocument> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("open trace file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!("read trace JSONL line {} from {}", line_no + 1, path.display())
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event: TraceEvent = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "parse trace JSONL line {} from {}",
                line_no + 1,
                path.display()
            )
        })?;
        events.push(event);
    }

    TraceDocument::from_events(events)
}

/// Write JSONL events to `path` (creates parent directories when needed).
pub fn write_jsonl(path: impl AsRef<Path>, events: &[TraceEvent]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace dir {}", parent.display()))?;
    }
    let mut file = File::create(path)
        .with_context(|| format!("create trace file {}", path.display()))?;
    for event in events {
        let mut line = serde_json::to_vec(event).context("serialize trace JSONL event")?;
        line.push(b'\n');
        file.write_all(&line)
            .with_context(|| format!("write trace JSONL to {}", path.display()))?;
    }
    file.flush()
        .with_context(|| format!("flush trace JSONL to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::events::TraceEvent;
    use crate::trace::memory::MemoryCategory;
    use crate::trace::schema::{GradientState, SpanKind};

    fn sample_meta() -> TraceRunMeta {
        TraceRunMeta {
            run_id: "run-1".into(),
            entrypoint: "demo::train::loss".into(),
            phase: "train".into(),
            timestamp: "2026-08-04T18:00:00Z".into(),
            candle_version: Some("0.8.0".into()),
        }
    }

    fn sample_events() -> Vec<TraceEvent> {
        vec![
            TraceEvent::meta(sample_meta()),
            TraceEvent::SpanStart(SpanStartEvent {
                id: "span-root".into(),
                parent_id: None,
                name: "demo::train::loss".into(),
                kind: SpanKind::Function,
                step: None,
            }),
            TraceEvent::SpanStart(SpanStartEvent {
                id: "span-op".into(),
                parent_id: Some("span-root".into()),
                name: "matmul".into(),
                kind: SpanKind::Op,
                step: None,
            }),
            TraceEvent::Op(OpEvent {
                span_id: "span-op".into(),
                op_name: "matmul".into(),
                inputs: vec!["t0".into(), "t1".into()],
                output: Some("t2".into()),
                shape: vec![32, 32],
                dtype: "f32".into(),
                device: "cpu".into(),
                duration_ns: 1200,
                timestamp_ns: 1200,
                storage_bytes: None,
                input_storage_bytes: 0,
            }),
            TraceEvent::Memory(super::super::events::MemoryEvent {
                timestamp_ns: 1200,
                tensor_id: "t2".into(),
                span_id: "span-op".into(),
                op_name: Some("matmul".into()),
                device: "cpu".into(),
                bytes: 32 * 32 * 4,
                action: MemoryAction::Alloc,
                shape: vec![32, 32],
                dtype: "f32".into(),
                category: MemoryCategory::Activation,
            }),
            TraceEvent::Edge(EdgeEvent {
                from_span: "span-root".into(),
                to_span: "span-op".into(),
                duration_ns: 1200,
            }),
            TraceEvent::Gradient(GradientEvent {
                event_id: "grad-1".into(),
                root: "vb".into(),
                key: "encoder.weight".into(),
                state: GradientState::Present,
                step: Some(0),
                norm: Some(0.42),
            }),
            TraceEvent::SpanEnd(SpanEndEvent {
                id: "span-op".into(),
                duration_ns: 0,
            }),
            TraceEvent::SpanEnd(SpanEndEvent {
                id: "span-root".into(),
                duration_ns: 0,
            }),
        ]
    }

    #[test]
    fn from_events_builds_document_and_summary() {
        let doc = TraceDocument::from_events(sample_events()).unwrap();
        assert_eq!(doc.schema, SCHEMA);
        assert_eq!(doc.run.entrypoint, "demo::train::loss");
        assert_eq!(doc.spans.len(), 2);
        assert!(doc.spans.iter().all(|span| span.closed));
        assert_eq!(doc.ops.len(), 1);
        assert_eq!(doc.ops[0].storage_bytes, Some(32 * 32 * 4));
        assert_eq!(doc.memory.len(), 1);
        assert_eq!(doc.edges.len(), 1);
        assert_eq!(doc.gradients.len(), 1);
        assert_eq!(doc.gradients[0].param_key(), "encoder.weight");

        let summary = doc.build_summary();
        assert_eq!(summary.op_count, 1);
        assert_eq!(summary.total_ns, 1200);
        assert_eq!(summary.span_count, 2);
        assert_eq!(summary.root_span_count, 1);
        assert_eq!(summary.max_depth, 1);
        assert_eq!(summary.alloc_count, 1);
        assert_eq!(summary.peak_bytes, 32 * 32 * 4);
    }

    #[test]
    fn jsonl_roundtrip_via_temp_file() {
        let dir = std::env::temp_dir().join(format!(
            "candle-graph-trace5-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.jsonl");

        let events = sample_events();
        write_jsonl(&path, &events).unwrap();
        let parsed = parse_trace(&path).unwrap();
        assert_eq!(parsed, TraceDocument::from_events(events.clone()).unwrap());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gradient_accepts_param_key_alias() {
        let line = r#"{"kind":"gradient","event_id":"g1","root":"vb","param_key":"w","state":"present","norm":1.0}"#;
        let event: TraceEvent = serde_json::from_str(line).unwrap();
        let TraceEvent::Gradient(g) = event else {
            panic!("expected gradient event");
        };
        assert_eq!(g.key, "w");
    }

    #[test]
    fn rejects_unknown_schema() {
        let events = vec![TraceEvent::Meta {
            schema: "candle-graph/trace/4".into(),
            run: sample_meta(),
        }];
        let err = TraceDocument::from_events(events).unwrap_err();
        assert!(err.to_string().contains("unsupported trace schema"));
    }

    #[test]
    fn rejects_span_end_without_start() {
        let events = vec![
            TraceEvent::meta(sample_meta()),
            TraceEvent::SpanEnd(SpanEndEvent {
                id: "missing".into(),
                duration_ns: 0,
            }),
        ];
        let err = TraceDocument::from_events(events).unwrap_err();
        assert!(err.to_string().contains("unknown span"));
    }
}
