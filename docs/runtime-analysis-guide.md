# Runtime evidence guide

## Select one representative invocation

Use `CaptureSelector::new(n)?.is_selected(invocation)` to gate a configurable one-based update or
inference invocation before opening a session. Record a typed
`ComparisonIdentity` and `CaptureContract`; set `implementation_id` to a stable generic build or
implementation identity. Free-form tags are descriptive and never establish comparison
compatibility. Instrumentation is inactive outside the selected invocation.

On CUDA, use `device_synchronized` only when every reported semantic span is bounded by device
synchronization. If only the single measured region is synchronized, use
`measured_region_device_synchronized`; `timing_mode` then remains `host`, so nested host spans are
not presented as completed GPU work. Nsight or another device profiler is still required for
kernel-level attribution.

## Required shape

Every trace contains one session envelope and exactly one measured region. Training captures tag
forward, backward, and optimizer spans with `ExecutionStep`. Use the same stable semantic labels in
NVTX, for example `trainer/update-000000000001/forward`.

```rust
let session = TraceSession::open(path, ProfileRun::training("train::update", 1, "cpu"))?;
{
    let _update = session.begin_measurement("workload/update-000000000001");
    let _forward = session.begin_step_span(
        "workload/update-000000000001/forward",
        ExecutionStep::Forward,
        SpanKind::Function,
    );
    // forward
}
session.finish()?;
```

`SpanGuard::id()` attaches op, tensor, and device-interval evidence. Tensor metadata does not imply
an allocation lifetime. Only explicit paired `record_memory_alloc`/`record_memory_free` events,
keyed by `(device, storage_id)`, drive logical live and peak memory. `record_device_memory` accepts
a `DeviceMemoryRecord` whose used/free/reserved/capacity values are independently optional. It
never derives capacity from used plus free. Neither plane fills gaps in the other.

With feature `candle`, `CandleCapture::from_tensor` records process-local backend tensor and
storage identities plus a dense tensor footprint. `with_label` may attach a semantic observation
label without replacing backend tensor identity. The footprint is shape metadata, not
backing-allocation size; callers must pass the backend's actual allocation bytes to explicit memory
events. Aliases with the same `(device, storage_id)` then share one logical lifetime.

## Publish evidence

```bash
cargo candle-graph summary application.jsonl
cargo candle-graph query application.jsonl --kind slowest-host
cargo candle-graph query application.jsonl --kind slowest-device
cargo candle-graph report application.jsonl --nsight-dir nsight --bundle evidence-bundle
cargo candle-graph verify evidence-bundle --output verification.json
cargo candle-graph view application.jsonl --nsight-dir nsight --output viewer.html
cargo candle-graph compare \
  --baseline base-1.jsonl base-2.jsonl base-3.jsonl base-4.jsonl base-5.jsonl \
  --candidate next-1.jsonl next-2.jsonl next-3.jsonl next-4.jsonl next-5.jsonl
```

The packet reports structural validity separately from capability states. A failed terminal event
remains inspectable but has no derived graph. Comparison requires at least five unique complete
baseline runs and five unique complete candidate runs with identical typed identities and timing
semantics. Each cohort must have one non-empty implementation ID, but the baseline and candidate
IDs may intentionally differ and are reported separately. It retains raw samples, median, p95,
MAD, and a deterministic 95% bootstrap interval; direction is confirmed only when the interval
excludes zero.

`report --bundle DIR` writes the trace, packet, Markdown, viewer (when enabled), Nsight inputs, and
content hashes under a sibling temporary directory, then atomically renames it into place. An
existing destination is rejected.
Nsight input directories are flat: every entry must be a regular file. Directory entries,
symbolic links, special files, and directory enumeration errors reject publication.
`verify` recursively rejects missing, modified, injected, symbolic-link, or special-file entries
and emits a receipt containing the exact `bundle.json` SHA-256.

## Nsight normalization

Retain `.nsys-rep`, export supported reports with NVIDIA's `nsys stats --format csv`, and include a
`capture-manifest.json` using schema `candle-graph/nsight-capture/1`, containing the trace
run/correlation IDs, required labels, tool/hardware metadata, and hashes for every retained artifact.
Exact semantic NVTX labels identify phases; correlation/device/context/stream identifiers join GPU
rows to projected ranges, and declared operation counts must match those joined rows. Correlation is
checked in both directions before display truncation. Candle,
device-event, and Nsight clocks are never overlaid as one clock.

Run `cargo bench --bench instrumentation_overhead --all-features` to measure the disabled capture
gate and active event serialization on the current machine.

Do not parse exported SQLite: its schema is not the candle-graph interface.
