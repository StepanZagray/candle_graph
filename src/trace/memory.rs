//! Logical storage-lifetime and physical device-memory analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::document::TraceDocument;

/// Semantic category attached to a logical storage lifetime.
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

/// Derive element size in bytes from a Candle-style dtype label (`f32`, `F32`, ...).
pub fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "u8" | "i8" | "bool" | "f8" => Some(1),
        "u16" | "i16" | "f16" | "bf16" => Some(2),
        "u32" | "i32" | "f32" => Some(4),
        "u64" | "i64" | "f64" => Some(8),
        _ => None,
    }
}

/// Product of shape dimensions; an empty shape is a scalar with one element.
pub fn elem_count(shape: &[usize]) -> u64 {
    shape
        .iter()
        .fold(1u64, |acc, dim| acc.saturating_mul(*dim as u64))
}

/// Dense tensor footprint (`elem_count * dtype_bytes`), not backing-allocation size.
pub fn dense_tensor_bytes(shape: &[usize], dtype: &str) -> Option<u64> {
    dtype_size_bytes(dtype).map(|size| elem_count(shape).saturating_mul(size as u64))
}

/// Resolve an explicit dense footprint or derive it from shape and dtype.
pub fn resolve_dense_tensor_bytes(
    explicit: Option<u64>,
    shape: &[usize],
    dtype: &str,
) -> Option<u64> {
    explicit.or_else(|| dense_tensor_bytes(shape, dtype))
}

/// Map a training step and tensor flags to a semantic memory category.
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

/// One backend storage lifetime. Tensor aliases are metadata on the storage, not allocations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalStorageLifetime {
    pub device: String,
    pub storage_id: String,
    pub tensor_ids: Vec<String>,
    pub allocation_span_id: String,
    pub op_name: Option<String>,
    pub bytes: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub category: MemoryCategory,
    pub start_timestamp_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_timestamp_ns: Option<u64>,
}

/// Simultaneous logical-memory state after all events at one timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalMemoryTimelinePoint {
    pub timestamp_ns: u64,
    pub live_bytes: u64,
    pub live_bytes_by_device: BTreeMap<String, u64>,
    pub live_bytes_by_category: BTreeMap<String, u64>,
}

/// The simultaneous global logical-memory peak.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalMemoryPeak {
    pub timestamp_ns: u64,
    pub live_bytes: u64,
    pub live_bytes_by_device: BTreeMap<String, u64>,
    pub live_bytes_by_category: BTreeMap<String, u64>,
    pub live_allocations: Vec<LogicalStorageLifetime>,
}

/// Logical-memory statistics for one device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalDeviceMemoryStats {
    pub device: String,
    pub storage_allocation_count: u64,
    pub matched_storage_free_count: u64,
    pub total_allocated_bytes: u64,
    /// `None` means no allocation was observed for this device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_live_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_timestamp_ns: Option<u64>,
}

/// Evidence derived only from `(device, storage_id)` allocation lifetimes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalMemoryProfile {
    pub storage_allocation_count: u64,
    pub matched_storage_free_count: u64,
    pub total_allocated_bytes: u64,
    /// `None` means the stream contained no allocation from which to calculate a peak.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak: Option<LogicalMemoryPeak>,
    pub timeline: Vec<LogicalMemoryTimelinePoint>,
    pub lifetimes: Vec<LogicalStorageLifetime>,
    pub by_device: Vec<LogicalDeviceMemoryStats>,
}

/// One physical device or allocator checkpoint, retained without logical inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalMemoryTimelinePoint {
    pub timestamp_ns: u64,
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_bytes: Option<u64>,
}

/// Observed physical-memory extrema for one device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDeviceMemoryStats {
    pub device: String,
    pub sample_count: u64,
    pub used_sample_count: u64,
    pub free_sample_count: u64,
    pub reserved_sample_count: u64,
    pub capacity_sample_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_used_timestamp_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_reserved_timestamp_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_free_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_free_timestamp_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_observed_capacity_bytes: Option<u64>,
}

/// Physical memory evidence derived only from explicit device samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalMemoryProfile {
    pub timeline: Vec<PhysicalMemoryTimelinePoint>,
    pub by_device: Vec<PhysicalDeviceMemoryStats>,
}

/// Memory evidence for a trace. An absent plane is represented by `None`, never zeroes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProfile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical: Option<LogicalMemoryProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical: Option<PhysicalMemoryProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StorageKey {
    device: String,
    storage_id: String,
}

#[derive(Debug, Clone)]
struct ActiveStorage {
    key: StorageKey,
    tensor_ids: BTreeSet<String>,
    allocation_span_id: String,
    op_name: Option<String>,
    bytes: u64,
    shape: Vec<usize>,
    dtype: String,
    category: MemoryCategory,
    start_timestamp_ns: u64,
}

impl ActiveStorage {
    fn lifetime(&self, end_timestamp_ns: Option<u64>) -> LogicalStorageLifetime {
        LogicalStorageLifetime {
            device: self.key.device.clone(),
            storage_id: self.key.storage_id.clone(),
            tensor_ids: self.tensor_ids.iter().cloned().collect(),
            allocation_span_id: self.allocation_span_id.clone(),
            op_name: self.op_name.clone(),
            bytes: self.bytes,
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            category: self.category,
            start_timestamp_ns: self.start_timestamp_ns,
            end_timestamp_ns,
        }
    }
}

#[derive(Debug, Default)]
struct LogicalDeviceAccumulator {
    allocation_count: u64,
    free_count: u64,
    total_allocated_bytes: u64,
    peak_live_bytes: Option<u64>,
    peak_timestamp_ns: Option<u64>,
}

/// Analyze the logical and physical memory planes independently.
pub fn analyze_memory(doc: &TraceDocument) -> MemoryProfile {
    MemoryProfile {
        logical: analyze_logical_memory(doc),
        physical: analyze_physical_memory(doc),
    }
}

fn analyze_logical_memory(doc: &TraceDocument) -> Option<LogicalMemoryProfile> {
    if doc.memory.is_empty() {
        return None;
    }

    let mut events: Vec<_> = doc.memory.iter().enumerate().collect();
    events.sort_by_key(|(index, event)| (event.timestamp_ns, *index));

    let mut active: BTreeMap<StorageKey, ActiveStorage> = BTreeMap::new();
    let mut lifetimes = Vec::new();
    let mut timeline = Vec::new();
    let mut devices: BTreeMap<String, LogicalDeviceAccumulator> = BTreeMap::new();
    let mut total_allocation_count = 0u64;
    let mut total_free_count = 0u64;
    let mut total_allocated_bytes = 0u64;
    let mut peak: Option<LogicalMemoryPeak> = None;

    let mut cursor = 0usize;
    while cursor < events.len() {
        let timestamp_ns = events[cursor].1.timestamp_ns;
        let end = events[cursor..]
            .iter()
            .position(|(_, event)| event.timestamp_ns != timestamp_ns)
            .map_or(events.len(), |offset| cursor + offset);

        for (_, event) in &events[cursor..end] {
            let key = StorageKey {
                device: event.device.clone(),
                storage_id: event.storage_id.clone(),
            };
            let device_stats = devices.entry(event.device.clone()).or_default();
            match event.action {
                MemoryAction::Alloc => {
                    if let Some(existing) = active.get_mut(&key) {
                        existing.tensor_ids.insert(event.tensor_id.clone());
                    } else {
                        let mut tensor_ids = BTreeSet::new();
                        tensor_ids.insert(event.tensor_id.clone());
                        active.insert(
                            key.clone(),
                            ActiveStorage {
                                key,
                                tensor_ids,
                                allocation_span_id: event.span_id.clone(),
                                op_name: event.op_name.clone(),
                                bytes: event.bytes,
                                shape: event.shape.clone(),
                                dtype: event.dtype.clone(),
                                category: event.category,
                                start_timestamp_ns: event.timestamp_ns,
                            },
                        );
                        total_allocation_count += 1;
                        total_allocated_bytes = total_allocated_bytes.saturating_add(event.bytes);
                        device_stats.allocation_count += 1;
                        device_stats.total_allocated_bytes = device_stats
                            .total_allocated_bytes
                            .saturating_add(event.bytes);
                    }
                }
                MemoryAction::Free => {
                    if let Some(mut storage) = active.remove(&key) {
                        storage.tensor_ids.insert(event.tensor_id.clone());
                        lifetimes.push(storage.lifetime(Some(event.timestamp_ns)));
                        total_free_count += 1;
                        device_stats.free_count += 1;
                    }
                }
            }
        }

        let point = logical_timeline_point(timestamp_ns, &active);
        for (device, stats) in &mut devices {
            let live = point.live_bytes_by_device.get(device).copied().unwrap_or(0);
            if stats.allocation_count > 0
                && stats.peak_live_bytes.is_none_or(|current| live > current)
            {
                stats.peak_live_bytes = Some(live);
                stats.peak_timestamp_ns = Some(timestamp_ns);
            }
        }
        if total_allocation_count > 0
            && peak
                .as_ref()
                .is_none_or(|current| point.live_bytes > current.live_bytes)
        {
            let mut live_allocations: Vec<_> = active
                .values()
                .map(|storage| storage.lifetime(None))
                .collect();
            sort_lifetimes(&mut live_allocations);
            peak = Some(LogicalMemoryPeak {
                timestamp_ns,
                live_bytes: point.live_bytes,
                live_bytes_by_device: point.live_bytes_by_device.clone(),
                live_bytes_by_category: point.live_bytes_by_category.clone(),
                live_allocations,
            });
        }
        timeline.push(point);
        cursor = end;
    }

    lifetimes.extend(active.values().map(|storage| storage.lifetime(None)));
    sort_lifetimes(&mut lifetimes);
    let by_device = devices
        .into_iter()
        .map(|(device, stats)| LogicalDeviceMemoryStats {
            device,
            storage_allocation_count: stats.allocation_count,
            matched_storage_free_count: stats.free_count,
            total_allocated_bytes: stats.total_allocated_bytes,
            peak_live_bytes: stats.peak_live_bytes,
            peak_timestamp_ns: stats.peak_timestamp_ns,
        })
        .collect();

    Some(LogicalMemoryProfile {
        storage_allocation_count: total_allocation_count,
        matched_storage_free_count: total_free_count,
        total_allocated_bytes,
        peak,
        timeline,
        lifetimes,
        by_device,
    })
}

fn logical_timeline_point(
    timestamp_ns: u64,
    active: &BTreeMap<StorageKey, ActiveStorage>,
) -> LogicalMemoryTimelinePoint {
    let mut live_bytes = 0u64;
    let mut live_bytes_by_device = BTreeMap::new();
    let mut live_bytes_by_category = BTreeMap::new();
    for storage in active.values() {
        live_bytes = live_bytes.saturating_add(storage.bytes);
        let device = live_bytes_by_device
            .entry(storage.key.device.clone())
            .or_insert(0u64);
        *device = device.saturating_add(storage.bytes);
        let category = live_bytes_by_category
            .entry(category_key(storage.category))
            .or_insert(0u64);
        *category = category.saturating_add(storage.bytes);
    }
    LogicalMemoryTimelinePoint {
        timestamp_ns,
        live_bytes,
        live_bytes_by_device,
        live_bytes_by_category,
    }
}

fn sort_lifetimes(lifetimes: &mut [LogicalStorageLifetime]) {
    lifetimes.sort_by(|a, b| {
        a.start_timestamp_ns
            .cmp(&b.start_timestamp_ns)
            .then_with(|| a.device.cmp(&b.device))
            .then_with(|| a.storage_id.cmp(&b.storage_id))
    });
}

fn analyze_physical_memory(doc: &TraceDocument) -> Option<PhysicalMemoryProfile> {
    if doc.device_memory.is_empty() {
        return None;
    }

    let mut samples: Vec<_> = doc.device_memory.iter().collect();
    samples.sort_by(|a, b| {
        a.timestamp_ns
            .cmp(&b.timestamp_ns)
            .then_with(|| a.device.cmp(&b.device))
    });
    let timeline = samples
        .iter()
        .map(|sample| PhysicalMemoryTimelinePoint {
            timestamp_ns: sample.timestamp_ns,
            device: sample.device.clone(),
            used_bytes: sample.used_bytes,
            free_bytes: sample.free_bytes,
            reserved_bytes: sample.reserved_bytes,
            capacity_bytes: sample.capacity_bytes,
        })
        .collect();

    let mut by_device_samples: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for sample in samples {
        by_device_samples
            .entry(sample.device.as_str())
            .or_default()
            .push(sample);
    }
    let by_device = by_device_samples
        .into_iter()
        .map(|(device, samples)| {
            let peak_used = samples
                .iter()
                .filter_map(|sample| sample.used_bytes.map(|bytes| (bytes, sample.timestamp_ns)))
                .max_by_key(|(bytes, timestamp)| (*bytes, std::cmp::Reverse(*timestamp)));
            let minimum_free = samples
                .iter()
                .filter_map(|sample| sample.free_bytes.map(|bytes| (bytes, sample.timestamp_ns)))
                .min_by_key(|(bytes, timestamp)| (*bytes, *timestamp));
            let peak_reserved = samples
                .iter()
                .filter_map(|sample| {
                    sample
                        .reserved_bytes
                        .map(|bytes| (bytes, sample.timestamp_ns))
                })
                .max_by_key(|(bytes, timestamp)| (*bytes, std::cmp::Reverse(*timestamp)));
            PhysicalDeviceMemoryStats {
                device: device.to_string(),
                sample_count: samples.len() as u64,
                used_sample_count: samples
                    .iter()
                    .filter(|sample| sample.used_bytes.is_some())
                    .count() as u64,
                free_sample_count: samples
                    .iter()
                    .filter(|sample| sample.free_bytes.is_some())
                    .count() as u64,
                reserved_sample_count: samples
                    .iter()
                    .filter(|sample| sample.reserved_bytes.is_some())
                    .count() as u64,
                capacity_sample_count: samples
                    .iter()
                    .filter(|sample| sample.capacity_bytes.is_some())
                    .count() as u64,
                peak_used_bytes: peak_used.map(|(bytes, _)| bytes),
                peak_used_timestamp_ns: peak_used.map(|(_, timestamp)| timestamp),
                peak_reserved_bytes: peak_reserved.map(|(bytes, _)| bytes),
                peak_reserved_timestamp_ns: peak_reserved.map(|(_, timestamp)| timestamp),
                minimum_free_bytes: minimum_free.map(|(bytes, _)| bytes),
                minimum_free_timestamp_ns: minimum_free.map(|(_, timestamp)| timestamp),
                maximum_observed_capacity_bytes: samples
                    .iter()
                    .filter_map(|sample| sample.capacity_bytes)
                    .max(),
            }
        })
        .collect();

    Some(PhysicalMemoryProfile {
        timeline,
        by_device,
    })
}

/// Per-node logical memory evidence. `None` means that metric was not observed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMemoryMetrics {
    /// Unique storage bytes allocated directly by this span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_allocated_bytes: Option<u64>,
    /// Simultaneous live bytes from this span and descendants, clipped to the span interval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtree_peak_live_bytes: Option<u64>,
    /// Subtree storage still live at the end of the span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtree_residual_bytes: Option<u64>,
    /// Dense output tensor footprint metadata for an operation node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dense_bytes: Option<u64>,
}

/// Attribute logical memory evidence to span ids and `{span}/op/{n}` operation ids.
pub fn node_memory_metrics(
    doc: &TraceDocument,
    op_node_ids: &HashMap<(String, usize), String>,
) -> HashMap<String, NodeMemoryMetrics> {
    let profile = analyze_memory(doc);
    let lifetimes = profile
        .logical
        .as_ref()
        .map(|logical| logical.lifetimes.as_slice());
    let mut metrics: HashMap<String, NodeMemoryMetrics> = HashMap::new();

    let mut per_span_index: HashMap<String, usize> = HashMap::new();
    for op in &doc.ops {
        let index = per_span_index.entry(op.span_id.clone()).or_insert(0);
        let op_index = *index;
        *index += 1;
        let Some(bytes) = resolve_dense_tensor_bytes(op.output_dense_bytes, &op.shape, &op.dtype)
        else {
            continue;
        };
        let node_id = op_node_ids
            .get(&(op.span_id.clone(), op_index))
            .cloned()
            .unwrap_or_else(|| format!("{}/op/{op_index}", op.span_id));
        metrics.entry(node_id).or_default().output_dense_bytes = Some(bytes);
    }

    let Some(lifetimes) = lifetimes else {
        return metrics;
    };

    let parent_by_id: HashMap<_, _> = doc
        .spans
        .iter()
        .map(|span| (span.id.as_str(), span.parent_id.as_deref()))
        .collect();
    for span in &doc.spans {
        let direct_allocated_bytes = lifetimes
            .iter()
            .filter(|lifetime| lifetime.allocation_span_id == span.id)
            .fold(0u64, |total, lifetime| total.saturating_add(lifetime.bytes));

        let subtree_lifetimes: Vec<_> = lifetimes
            .iter()
            .filter(|lifetime| span_contains(&parent_by_id, &span.id, &lifetime.allocation_span_id))
            .collect();
        let entry = metrics.entry(span.id.clone()).or_default();
        entry.direct_allocated_bytes = Some(direct_allocated_bytes);

        if span.closed {
            let interval_end = span.start_ns.saturating_add(span.duration_ns);
            let (peak, residual) =
                clipped_lifetime_metrics(&subtree_lifetimes, span.start_ns, interval_end);
            entry.subtree_peak_live_bytes = Some(peak);
            entry.subtree_residual_bytes = Some(residual);
        }
    }

    metrics
}

fn span_contains(
    parent_by_id: &HashMap<&str, Option<&str>>,
    ancestor_id: &str,
    descendant_id: &str,
) -> bool {
    let mut current = Some(descendant_id);
    let mut visited = BTreeSet::new();
    while let Some(id) = current {
        if id == ancestor_id {
            return true;
        }
        if !visited.insert(id) {
            return false;
        }
        current = parent_by_id.get(id).copied().flatten();
    }
    false
}

fn clipped_lifetime_metrics(
    lifetimes: &[&LogicalStorageLifetime],
    interval_start: u64,
    interval_end: u64,
) -> (u64, u64) {
    if interval_end <= interval_start {
        return (0, 0);
    }

    let mut live_at_start = 0u64;
    let mut changes: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    let mut residual = 0u64;
    for lifetime in lifetimes {
        let lifetime_end = lifetime.end_timestamp_ns.unwrap_or(u64::MAX);
        if lifetime.start_timestamp_ns >= interval_end || lifetime_end <= interval_start {
            continue;
        }
        if lifetime.start_timestamp_ns <= interval_start {
            live_at_start = live_at_start.saturating_add(lifetime.bytes);
        } else {
            let change = changes.entry(lifetime.start_timestamp_ns).or_default();
            change.1 = change.1.saturating_add(lifetime.bytes);
        }
        if lifetime_end > interval_start && lifetime_end < interval_end {
            let change = changes.entry(lifetime_end).or_default();
            change.0 = change.0.saturating_add(lifetime.bytes);
        }
        if lifetime.start_timestamp_ns < interval_end && lifetime_end > interval_end {
            residual = residual.saturating_add(lifetime.bytes);
        }
    }

    let mut live = live_at_start;
    let mut peak = live;
    for (_, (freed, allocated)) in changes {
        live = live.saturating_sub(freed).saturating_add(allocated);
        peak = peak.max(live);
    }
    (peak, residual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CaptureContract;
    use crate::trace::events::{DeviceMemoryEvent, MemoryEvent, TerminalEvent};
    use crate::trace::schema::{RunOutcome, SpanKind, SpanRecord, TraceRunMeta, SCHEMA};

    fn empty_doc() -> TraceDocument {
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
                measured_region_device_synchronized: false,
                timing_mode: crate::trace::TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![],
            ops: vec![],
            tensors: vec![],
            tensor_stats: vec![],
            memory: vec![],
            device_memory: vec![],
            device_intervals: vec![],
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 100,
                reason: None,
            },
        }
    }

    fn memory(
        timestamp_ns: u64,
        device: &str,
        storage_id: &str,
        tensor_id: &str,
        span_id: &str,
        bytes: u64,
        action: MemoryAction,
    ) -> MemoryEvent {
        MemoryEvent {
            timestamp_ns,
            storage_id: storage_id.into(),
            tensor_id: tensor_id.into(),
            span_id: span_id.into(),
            op_name: None,
            device: device.into(),
            bytes,
            action,
            shape: vec![bytes as usize],
            dtype: "u8".into(),
            category: MemoryCategory::Activation,
        }
    }

    #[test]
    fn physical_only_samples_remain_a_separate_evidence_plane() {
        let mut doc = empty_doc();
        doc.device_memory = vec![
            DeviceMemoryEvent {
                timestamp_ns: 10,
                device: "cuda:0".into(),
                used_bytes: Some(100),
                free_bytes: Some(900),
                reserved_bytes: None,
                capacity_bytes: Some(1_000),
            },
            DeviceMemoryEvent {
                timestamp_ns: 20,
                device: "cuda:0".into(),
                used_bytes: Some(300),
                free_bytes: Some(700),
                reserved_bytes: Some(400),
                capacity_bytes: Some(1_000),
            },
        ];

        let profile = analyze_memory(&doc);
        assert!(profile.logical.is_none());
        let physical = profile.physical.unwrap();
        assert_eq!(physical.timeline.len(), 2);
        assert_eq!(physical.by_device[0].peak_used_bytes, Some(300));
        assert_eq!(physical.by_device[0].peak_reserved_bytes, Some(400));
        assert_eq!(physical.by_device[0].minimum_free_bytes, Some(700));
        assert_eq!(
            physical.by_device[0].maximum_observed_capacity_bytes,
            Some(1_000)
        );
    }

    #[test]
    fn physical_unknowns_are_not_derived_from_other_measurements() {
        let mut doc = empty_doc();
        doc.device_memory = vec![DeviceMemoryEvent {
            timestamp_ns: 10,
            device: "cuda:0".into(),
            used_bytes: None,
            free_bytes: None,
            reserved_bytes: Some(400),
            capacity_bytes: None,
        }];
        let stats = &analyze_memory(&doc).physical.unwrap().by_device[0];
        assert_eq!(stats.used_sample_count, 0);
        assert_eq!(stats.peak_used_bytes, None);
        assert_eq!(stats.minimum_free_bytes, None);
        assert_eq!(stats.maximum_observed_capacity_bytes, None);
    }

    #[test]
    fn storage_identity_is_scoped_by_device() {
        let mut doc = empty_doc();
        doc.memory = vec![
            memory(10, "cpu", "shared", "cpu-t", "s", 10, MemoryAction::Alloc),
            memory(
                10,
                "cuda:0",
                "shared",
                "gpu-t",
                "s",
                20,
                MemoryAction::Alloc,
            ),
        ];

        let logical = analyze_memory(&doc).logical.unwrap();
        assert_eq!(logical.storage_allocation_count, 2);
        assert_eq!(logical.peak.unwrap().live_bytes, 30);
        assert_eq!(logical.by_device.len(), 2);
    }

    #[test]
    fn aliases_share_one_storage_lifetime_and_preserve_tensor_ids() {
        let mut doc = empty_doc();
        doc.memory = vec![
            memory(10, "cpu", "a", "base", "s", 64, MemoryAction::Alloc),
            memory(20, "cpu", "a", "view", "s", 64, MemoryAction::Alloc),
            memory(40, "cpu", "a", "view", "s", 64, MemoryAction::Free),
        ];

        let logical = analyze_memory(&doc).logical.unwrap();
        assert_eq!(logical.storage_allocation_count, 1);
        assert_eq!(logical.matched_storage_free_count, 1);
        assert_eq!(logical.peak.unwrap().live_bytes, 64);
        assert_eq!(logical.lifetimes[0].tensor_ids, vec!["base", "view"]);
        assert_eq!(logical.lifetimes[0].end_timestamp_ns, Some(40));
    }

    #[test]
    fn peak_is_the_simultaneous_sum_not_the_largest_allocation() {
        let mut doc = empty_doc();
        doc.memory = vec![
            memory(10, "cpu", "a", "a", "s", 40, MemoryAction::Alloc),
            memory(20, "cpu", "b", "b", "s", 70, MemoryAction::Alloc),
            memory(30, "cpu", "a", "a", "s", 40, MemoryAction::Free),
        ];

        let logical = analyze_memory(&doc).logical.unwrap();
        let peak = logical.peak.unwrap();
        assert_eq!(peak.live_bytes, 110);
        assert_eq!(peak.timestamp_ns, 20);
        assert_eq!(logical.by_device[0].peak_live_bytes, Some(110));
    }

    #[test]
    fn subtree_peak_and_residual_are_clipped_to_span_interval() {
        let mut doc = empty_doc();
        doc.spans = vec![
            SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "root".into(),
                kind: SpanKind::Function,
                measured: true,
                start_ns: 10,
                closed: true,
                duration_ns: 30,
                step: None,
            },
            SpanRecord {
                id: "child".into(),
                parent_id: Some("root".into()),
                name: "child".into(),
                kind: SpanKind::Function,
                measured: false,
                start_ns: 15,
                closed: true,
                duration_ns: 15,
                step: None,
            },
        ];
        doc.memory = vec![
            memory(16, "cpu", "a", "a", "child", 40, MemoryAction::Alloc),
            memory(20, "cpu", "b", "b", "child", 60, MemoryAction::Alloc),
            memory(25, "cpu", "a", "a", "child", 40, MemoryAction::Free),
            memory(45, "cpu", "b", "b", "child", 60, MemoryAction::Free),
        ];

        let metrics = node_memory_metrics(&doc, &HashMap::new());
        let child = &metrics["child"];
        assert_eq!(child.subtree_peak_live_bytes, Some(100));
        assert_eq!(child.subtree_residual_bytes, Some(60));
        let root = &metrics["root"];
        assert_eq!(root.subtree_peak_live_bytes, Some(100));
        assert_eq!(root.subtree_residual_bytes, Some(60));
    }
}
