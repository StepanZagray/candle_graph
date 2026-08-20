//! Separate host and device timing planes with overlap-safe device aggregation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::capability::CoverageLevel;
use crate::trace::TraceDocument;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSpanTiming {
    pub span_id: String,
    pub device: String,
    pub clock_id: String,
    pub backends: Vec<String>,
    pub streams: Vec<String>,
    pub interval_count: usize,
    /// Union of all intervals for this span on this device clock.
    pub busy_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceClockTiming {
    pub device: String,
    pub clock_id: String,
    pub interval_count: usize,
    /// Union across streams; overlapping device work is counted once.
    pub busy_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingProfile {
    pub device_coverage: CoverageLevel,
    pub device_spans: Vec<DeviceSpanTiming>,
    pub device_clocks: Vec<DeviceClockTiming>,
}

pub fn analyze_timing(doc: &TraceDocument) -> TimingProfile {
    let mut by_span: BTreeMap<(String, String, String), Vec<_>> = BTreeMap::new();
    let mut by_clock: BTreeMap<(String, String), Vec<_>> = BTreeMap::new();
    for interval in &doc.device_intervals {
        by_span
            .entry((
                interval.span_id.clone(),
                interval.device.clone(),
                interval.clock_id.clone(),
            ))
            .or_default()
            .push(interval);
        by_clock
            .entry((interval.device.clone(), interval.clock_id.clone()))
            .or_default()
            .push(interval);
    }

    let device_spans = by_span
        .into_iter()
        .map(|((span_id, device, clock_id), intervals)| {
            let mut backends = intervals
                .iter()
                .map(|interval| interval.backend.clone())
                .collect::<Vec<_>>();
            backends.sort();
            backends.dedup();
            let mut streams = intervals
                .iter()
                .map(|interval| interval.stream_id.clone())
                .collect::<Vec<_>>();
            streams.sort();
            streams.dedup();
            DeviceSpanTiming {
                span_id,
                device,
                clock_id,
                backends,
                streams,
                interval_count: intervals.len(),
                busy_ns: interval_union_ns(intervals.iter().map(|item| {
                    (
                        item.start_ns,
                        item.start_ns.saturating_add(item.duration_ns),
                    )
                })),
            }
        })
        .collect();

    let device_clocks = by_clock
        .into_iter()
        .map(|((device, clock_id), intervals)| DeviceClockTiming {
            device,
            clock_id,
            interval_count: intervals.len(),
            busy_ns: interval_union_ns(intervals.iter().map(|item| {
                (
                    item.start_ns,
                    item.start_ns.saturating_add(item.duration_ns),
                )
            })),
        })
        .collect();

    TimingProfile {
        device_coverage: doc
            .run
            .capture_contract
            .device_timing
            .with_observations(doc.device_intervals.len()),
        device_spans,
        device_clocks,
    }
}

fn interval_union_ns(intervals: impl IntoIterator<Item = (u64, u64)>) -> u64 {
    let mut intervals = intervals.into_iter().collect::<Vec<_>>();
    intervals.sort_unstable();
    let mut total = 0u64;
    let mut current: Option<(u64, u64)> = None;
    for (start, end) in intervals {
        match current {
            None => current = Some((start, end)),
            Some((left, right)) if start <= right => current = Some((left, right.max(end))),
            Some((left, right)) => {
                total = total.saturating_add(right.saturating_sub(left));
                current = Some((start, end));
            }
        }
    }
    if let Some((start, end)) = current {
        total = total.saturating_add(end.saturating_sub(start));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::interval_union_ns;

    #[test]
    fn overlapping_intervals_are_counted_once() {
        assert_eq!(interval_union_ns([(0, 10), (5, 20), (30, 35)]), 25);
    }
}
