//! Memory timeline analysis — TensorFlow Memory Profile model.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use super::document::TraceDocument;

/// TensorFlow-profiler-style memory category for timeline breakdown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Parameter,
    Activation,
    Gradient,
    Optimizer,
    #[default]
    Other,
}

/// Allocation or deallocation recorded during a probe run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAction {
    Alloc,
    Free,
}

/// Derive element size in bytes from a Candle-style dtype label (`f32`, `F32`, …).
pub fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "u8" | "i8" | "bool" => Some(1),
        "u16" | "i16" | "f16" | "bf16" => Some(2),
        "u32" | "i32" | "f32" => Some(4),
        "u64" | "i64" | "f64" => Some(8),
        "f8" => Some(1),
        _ => None,
    }
}

/// Product of shape dimensions; returns `0` for empty shape.
pub fn elem_count(shape: &[usize]) -> u64 {
    shape
        .iter()
        .fold(1u64, |acc, dim| acc.saturating_mul(*dim as u64))
}

/// Storage bytes for a dense tensor (`elem_count × dtype_bytes`).
pub fn storage_bytes(shape: &[usize], dtype: &str) -> u64 {
    let count = elem_count(shape);
    if count == 0 {
        return 0;
    }
    dtype_size_bytes(dtype)
        .map(|sz| count.saturating_mul(sz as u64))
        .unwrap_or(0)
}

/// Resolve explicit bytes or derive from shape/dtype.
pub fn resolve_storage_bytes(explicit: Option<u64>, shape: &[usize], dtype: &str) -> u64 {
    explicit.unwrap_or_else(|| storage_bytes(shape, dtype))
}

/// Map a training step (+ tensor flags) to a PyTorch-style memory category.
pub fn category_for_step(
    step: Option<crate::phase::ExecutionStep>,
    requires_grad: bool,
) -> MemoryCategory {
    match step {
        Some(crate::phase::ExecutionStep::Backward) => MemoryCategory::Gradient,
        Some(crate::phase::ExecutionStep::Optimizer) => MemoryCategory::Optimizer,
        Some(crate::phase::ExecutionStep::Forward) => MemoryCategory::Activation,
        None if requires_grad => MemoryCategory::Parameter,
        None => MemoryCategory::Activation,
    }
}

fn category_key(category: MemoryCategory) -> String {
    format!("{category:?}").to_ascii_lowercase()
}

/// Aggregated memory statistics for one probe run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySummary {
    pub alloc_count: u64,
    pub free_count: u64,
    /// Sum of all allocation sizes (can exceed peak when tensors are freed).
    pub total_alloc_bytes: u64,
    pub peak_bytes: u64,
    pub peak_timestamp_ns: u64,
    pub peak_device: String,
    /// Live bytes by category at global peak (PyTorch memory timeline breakdown).
    #[serde(default)]
    pub peak_by_category: BTreeMap<String, u64>,
    /// Activation bytes still live when backward starts (autograd retention hint).
    #[serde(default)]
    pub autograd_retained_bytes: u64,
}

/// One point on the memory-vs-time curve for a device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryTimelinePoint {
    pub timestamp_ns: u64,
    pub device: String,
    pub live_bytes: u64,
    pub heap_bytes: u64,
    pub free_bytes: u64,
    pub by_category: BTreeMap<String, u64>,
}

/// Active allocation at the global peak (TensorFlow breakdown table).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveAllocation {
    pub tensor_id: String,
    pub span_id: String,
    pub op_name: Option<String>,
    pub bytes: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub category: MemoryCategory,
}

/// Per-device memory stats.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemoryStats {
    pub device: String,
    pub capacity_bytes: u64,
    pub peak_bytes: u64,
    pub peak_timestamp_ns: u64,
    pub alloc_count: u64,
    pub free_count: u64,
}

/// Full memory profile reconstructed from trace evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub summary: MemorySummary,
    pub timeline: Vec<MemoryTimelinePoint>,
    pub peak_breakdown: Vec<LiveAllocation>,
    pub by_device: Vec<DeviceMemoryStats>,
}

#[derive(Debug, Clone)]
struct TimelineEvent {
    timestamp_ns: u64,
    device: String,
    action: MemoryAction,
    bytes: u64,
    category: MemoryCategory,
    tensor_id: String,
    span_id: String,
    op_name: Option<String>,
    shape: Vec<usize>,
    dtype: String,
}

/// Build a TensorFlow-style memory profile from a parsed trace document.
pub fn analyze_memory(doc: &TraceDocument) -> MemoryProfile {
    let events = collect_timeline_events(doc);
    let span_steps: HashMap<String, crate::phase::ExecutionStep> = doc
        .spans
        .iter()
        .filter_map(|span| span.step.map(|step| (span.id.clone(), step)))
        .collect();
    if events.is_empty() {
        return MemoryProfile {
            summary: MemorySummary::default(),
            timeline: Vec::new(),
            peak_breakdown: Vec::new(),
            by_device: Vec::new(),
        };
    }

    let mut by_device_live: HashMap<String, u64> = HashMap::new();
    let mut by_device_category: HashMap<String, BTreeMap<String, u64>> = HashMap::new();
    let mut live_tensors: HashMap<String, LiveAllocation> = HashMap::new();

    let mut global_peak = 0u64;
    let mut global_peak_ts = 0u64;
    let mut global_peak_device = String::new();
    let mut peak_breakdown = Vec::new();

    let mut device_peaks: HashMap<String, (u64, u64)> = HashMap::new();
    let mut device_capacity: HashMap<String, u64> = HashMap::new();
    let mut device_alloc_count: HashMap<String, u64> = HashMap::new();
    let mut device_free_count: HashMap<String, u64> = HashMap::new();

    let mut timeline = Vec::new();
    let mut alloc_count = 0u64;
    let mut free_count = 0u64;
    let mut total_alloc_bytes = 0u64;
    let mut peak_by_category: BTreeMap<String, u64> = BTreeMap::new();
    let mut autograd_retained_bytes = 0u64;
    let mut backward_seen = false;

    for event in &events {
        let device = event.device.clone();
        let live = by_device_live.entry(device.clone()).or_insert(0);
        let categories = by_device_category.entry(device.clone()).or_default();
        let cat_key = category_key(event.category);

        match event.action {
            MemoryAction::Alloc => {
                alloc_count += 1;
                *device_alloc_count.entry(device.clone()).or_insert(0) += 1;
                total_alloc_bytes = total_alloc_bytes.saturating_add(event.bytes);
                *live = live.saturating_add(event.bytes);
                {
                    let cat = categories.entry(cat_key).or_insert(0);
                    *cat = cat.saturating_add(event.bytes);
                }
                live_tensors.insert(
                    event.tensor_id.clone(),
                    LiveAllocation {
                        tensor_id: event.tensor_id.clone(),
                        span_id: event.span_id.clone(),
                        op_name: event.op_name.clone(),
                        bytes: event.bytes,
                        shape: event.shape.clone(),
                        dtype: event.dtype.clone(),
                        device: device.clone(),
                        category: event.category,
                    },
                );
            }
            MemoryAction::Free => {
                free_count += 1;
                *device_free_count.entry(device.clone()).or_insert(0) += 1;
                if let Some(alloc) = live_tensors.remove(&event.tensor_id) {
                    *live = live.saturating_sub(alloc.bytes);
                    let key = category_key(alloc.category);
                    if let Some(cat_live) = categories.get_mut(&key) {
                        *cat_live = cat_live.saturating_sub(alloc.bytes);
                    }
                } else {
                    *live = live.saturating_sub(event.bytes);
                }
            }
        }

        let heap = *live;
        let capacity = device_capacity.get(&device).copied().unwrap_or(0);
        let free = capacity.saturating_sub(heap);

        timeline.push(MemoryTimelinePoint {
            timestamp_ns: event.timestamp_ns,
            device: device.clone(),
            live_bytes: heap,
            heap_bytes: heap,
            free_bytes: free,
            by_category: categories.clone(),
        });

        let (peak, peak_ts) = device_peaks.entry(device.clone()).or_insert((0, 0));
        if heap > *peak {
            *peak = heap;
            *peak_ts = event.timestamp_ns;
        }

        if heap > global_peak {
            global_peak = heap;
            global_peak_ts = event.timestamp_ns;
            global_peak_device = device.clone();
            peak_breakdown = live_tensors.values().cloned().collect();
            peak_breakdown.sort_by(|a, b| {
                b.bytes
                    .cmp(&a.bytes)
                    .then_with(|| a.tensor_id.cmp(&b.tensor_id))
            });
            peak_by_category = categories.clone();
        }

        if !backward_seen {
            let in_backward = span_steps
                .get(&event.span_id)
                .is_some_and(|step| *step == crate::phase::ExecutionStep::Backward);
            if in_backward {
                backward_seen = true;
                autograd_retained_bytes = categories
                    .get(&category_key(MemoryCategory::Activation))
                    .copied()
                    .unwrap_or(0);
            }
        }
    }

    for snapshot in &doc.device_memory {
        let entry = device_capacity.entry(snapshot.device.clone()).or_insert(0);
        *entry = (*entry).max(snapshot.used_bytes.saturating_add(snapshot.free_bytes));
    }

    let mut by_device: Vec<DeviceMemoryStats> = device_peaks
        .into_iter()
        .map(
            |(device, (peak_bytes, peak_timestamp_ns))| DeviceMemoryStats {
                device: device.clone(),
                capacity_bytes: device_capacity.get(&device).copied().unwrap_or(0),
                peak_bytes,
                peak_timestamp_ns,
                alloc_count: device_alloc_count.get(&device).copied().unwrap_or(0),
                free_count: device_free_count.get(&device).copied().unwrap_or(0),
            },
        )
        .collect();
    by_device.sort_by(|a, b| a.device.cmp(&b.device));

    MemoryProfile {
        summary: MemorySummary {
            alloc_count,
            free_count,
            total_alloc_bytes,
            peak_bytes: global_peak,
            peak_timestamp_ns: global_peak_ts,
            peak_device: global_peak_device,
            peak_by_category,
            autograd_retained_bytes,
        },
        timeline,
        peak_breakdown,
        by_device,
    }
}

fn collect_timeline_events(doc: &TraceDocument) -> Vec<TimelineEvent> {
    let mut events: Vec<TimelineEvent> = Vec::new();

    for mem in &doc.memory {
        events.push(TimelineEvent {
            timestamp_ns: mem.timestamp_ns,
            device: mem.device.clone(),
            action: mem.action,
            bytes: mem.bytes,
            category: mem.category,
            tensor_id: mem.tensor_id.clone(),
            span_id: mem.span_id.clone(),
            op_name: mem.op_name.clone(),
            shape: mem.shape.clone(),
            dtype: mem.dtype.clone(),
        });
    }

    events.sort_by(|a, b| {
        a.timestamp_ns
            .cmp(&b.timestamp_ns)
            .then_with(|| format!("{:?}", a.action).cmp(&format!("{:?}", b.action)))
            .then_with(|| a.tensor_id.cmp(&b.tensor_id))
    });
    events
}

/// Per-span / per-op TensorFlow memory metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMemoryMetrics {
    /// Total bytes requested by ops directly on this node.
    pub bytes: u64,
    /// Peak live bytes in this node's subtree during execution.
    pub peak_bytes: u64,
    /// Bytes still live when this node finishes (not deallocated).
    pub residual_bytes: u64,
    /// Output storage for op leaf nodes.
    pub storage_bytes: u64,
}

/// Attribute memory metrics to graph node ids (span ids and `{span}/op/{n}` op ids).
pub fn node_memory_metrics(
    doc: &TraceDocument,
    op_node_ids: &HashMap<(String, usize), String>,
) -> HashMap<String, NodeMemoryMetrics> {
    let profile = analyze_memory(doc);
    let mut metrics: HashMap<String, NodeMemoryMetrics> = HashMap::new();

    let mut per_span_index: HashMap<String, usize> = HashMap::new();
    for op in &doc.ops {
        let index = *per_span_index.entry(op.span_id.clone()).or_insert(0);
        per_span_index.insert(op.span_id.clone(), index + 1);

        let bytes = op
            .storage_bytes
            .unwrap_or_else(|| resolve_storage_bytes(None, &op.shape, &op.dtype));
        if bytes == 0 {
            continue;
        }

        let node_id = op_node_ids
            .get(&(op.span_id.clone(), index))
            .cloned()
            .unwrap_or_else(|| format!("{}/op/{}", op.span_id, index));

        let entry = metrics.entry(node_id).or_default();
        entry.storage_bytes = bytes;
        entry.bytes = bytes;

        let span_entry = metrics.entry(op.span_id.clone()).or_default();
        span_entry.bytes = span_entry.bytes.saturating_add(bytes);
    }

    for live in &profile.peak_breakdown {
        if let Some(entry) = metrics.get_mut(&live.span_id) {
            entry.peak_bytes = entry.peak_bytes.max(live.bytes);
        }
    }

    for span in &doc.spans {
        rollup_span_memory(span.id.as_str(), &doc.spans, &mut metrics);
    }

    metrics
}

fn rollup_span_memory(
    span_id: &str,
    spans: &[super::schema::SpanRecord],
    metrics: &mut HashMap<String, NodeMemoryMetrics>,
) {
    let children: Vec<_> = spans
        .iter()
        .filter(|s| s.parent_id.as_deref() == Some(span_id))
        .collect();

    let mut child_peak = 0u64;
    for child in &children {
        rollup_span_memory(&child.id, spans, metrics);
        if let Some(m) = metrics.get(&child.id) {
            child_peak = child_peak.max(m.peak_bytes);
        }
    }

    let own = metrics.get(span_id).cloned().unwrap_or_default();
    let entry = metrics.entry(span_id.to_string()).or_default();
    entry.peak_bytes = entry.peak_bytes.max(own.bytes).max(child_peak);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::events::{MemoryEvent, OpEvent};
    use crate::trace::schema::{SpanKind, SpanRecord, TraceRunMeta, SCHEMA};

    fn doc_with_ops_and_memory() -> TraceDocument {
        TraceDocument {
            schema: SCHEMA.to_string(),
            run: TraceRunMeta {
                run_id: "r".into(),
                correlation_id: "test/update-1".into(),
                entrypoint: "test".into(),
                phase: crate::phase::ExecutionPhase::Train,
                timestamp: "2026-01-01T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                timing_mode: crate::trace::TimingMode::Host,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "s1".into(),
                parent_id: None,
                name: "forward".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 0,
                closed: true,
                duration_ns: 1000,
                step: None,
            }],
            ops: vec![OpEvent {
                span_id: "s1".into(),
                op_name: "matmul".into(),
                inputs: vec![],
                output: Some("out".into()),
                shape: vec![100, 100],
                dtype: "f32".into(),
                device: "cpu".into(),
                duration_ns: 500,
                timestamp_ns: 500,
                storage_bytes: None,
                input_storage_bytes: 0,
            }],
            tensors: vec![],
            memory: vec![
                MemoryEvent {
                    timestamp_ns: 500,
                    tensor_id: "out".into(),
                    span_id: "s1".into(),
                    op_name: Some("matmul".into()),
                    device: "cpu".into(),
                    bytes: 100 * 100 * 4,
                    action: MemoryAction::Alloc,
                    shape: vec![100, 100],
                    dtype: "f32".into(),
                    category: MemoryCategory::Activation,
                },
                MemoryEvent {
                    timestamp_ns: 1000,
                    tensor_id: "out".into(),
                    span_id: "s1".into(),
                    op_name: Some("matmul".into()),
                    device: "cpu".into(),
                    bytes: 100 * 100 * 4,
                    action: MemoryAction::Free,
                    shape: vec![100, 100],
                    dtype: "f32".into(),
                    category: MemoryCategory::Activation,
                },
            ],
            device_memory: vec![],
            gradients: vec![],
            edges: vec![],
        }
    }

    #[test]
    fn explicit_memory_events_compute_peak() {
        let profile = analyze_memory(&doc_with_ops_and_memory());
        assert_eq!(profile.summary.peak_bytes, 100 * 100 * 4);
        assert_eq!(profile.summary.alloc_count, 1);
        assert_eq!(profile.summary.free_count, 1);
        assert_eq!(profile.peak_breakdown.len(), 1);
    }

    #[test]
    fn does_not_invent_lifetimes_from_storage_metadata() {
        let mut doc = doc_with_ops_and_memory();
        doc.memory.clear();
        let profile = analyze_memory(&doc);
        assert_eq!(profile.summary.peak_bytes, 0);
        assert!(profile.timeline.is_empty());
    }
}
