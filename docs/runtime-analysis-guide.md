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
Required semantic labels form an exact cardinality contract: declarations must be non-empty and
unique, and every declared label must occur exactly once in a successfully completed trace. A
missing or repeated required label makes a complete capture structurally invalid; failed captures
retain the same finding as a diagnostic warning.

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

For already-completed host work that ran concurrently with the live call stack, retain its
`std::time::Instant`, convert it with `TraceSession::host_timestamp_ns`, and later call
`record_completed_host_span`. The explicit positive duration must end no later than the recording
call. This emits a closed span attached to the session root without pushing or popping live span
guards, so overlapping intervals remain siblings rather than asserting a synchronous call
relationship.

With feature `candle`, `CandleCapture::from_tensor` records process-local backend tensor and
storage identities plus a dense tensor footprint. `with_label` may attach a semantic observation
label without replacing backend tensor identity. The footprint is shape metadata, not
backing-allocation size; callers must pass the backend's actual allocation bytes to explicit memory
events. Aliases with the same `(device, storage_id)` then share one logical lifetime.

## Qualify complete gradient coverage

Construct a `GradientContract` from the final model `VarMap` before opening `TraceSession`. Read
the named keys through `varmap.data()`, sort them, map every key to a caller-owned family, then pass
the resulting contract in `CaptureContract { gradients: CoverageLevel::Complete,
gradient_contract: Some(contract), .. }`. A complete capture records every declared `(root, key)`
exactly once. The README contains a complete generic VarMap example.

Family policies have precise meanings:

- `active(family, min_present)` requires at least that many finite, positive-norm gradients and
  therefore rejects an all-zero active family.
- `inactive(family)` requires every member to be explicitly `Missing`.
- `data_conditional(family, min_present)` permits every member to be `Missing`; once any gradient
  attaches, at least `min_present` members must be finite and positive.

Encode `Present` with a finite norm greater than zero, `Zero` with positive zero (`0.0`, never
`-0.0`), and `Missing` or `NonFinite` with no norm. The ordered manifest digest hashes the
`GRADIENT_MANIFEST_SCHEMA` bytes, one NUL byte, then every root/key/family byte string prefixed by
its little-endian `u64` length. Current public protocol constants are `TRACE_SCHEMA` (trace/9),
`EVIDENCE_SCHEMA` (evidence/3), and `COMPARISON_SCHEMA` (comparison/4).

## Publish evidence

```bash
cargo candle-graph summary application.jsonl
cargo candle-graph query application.jsonl --kind slowest-host
cargo candle-graph query application.jsonl --kind slowest-device
cargo candle-graph report application.jsonl --nsight-dir nsight --bundle evidence-bundle
cargo candle-graph verify evidence-bundle --output verification.json
cargo candle-graph view application.jsonl --nsight-dir nsight --output viewer.html
cargo candle-graph compare \
  --baseline base-1.bundle base-2.bundle base-3.bundle base-4.bundle base-5.bundle \
  --candidate next-1.bundle next-2.bundle next-3.bundle next-4.bundle next-5.bundle
```

The packet reports structural validity separately from capability states. A failed terminal event
remains inspectable but has no derived graph. Comparison deeply verifies every supplied finalized
bundle, then requires at least five unique complete baseline runs and five unique complete
candidate runs with identical typed identities and timing semantics. Each cohort must have one
non-empty implementation ID, but the baseline and candidate IDs may intentionally differ and are
reported separately. It retains raw samples, median, p95,
MAD, and a deterministic 95% bootstrap interval; direction is confirmed only when the interval
excludes zero. `compare --unverified-traces` accepts raw JSONL paths for diagnostics, records that
trust state in comparison/4, and always withholds eligibility and confidence intervals.

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
run/correlation IDs, required application labels, their GPU-expected/CPU-only partition,
tool/hardware metadata, and hashes for every retained artifact. Older manifests without an explicit
partition retain the legacy meaning that every required label is GPU-expected. Official
`nvtx_gpu_proj_trace` rows use `Projected Start`, `Projected Duration`, `Orig Start`, and
`Orig Duration`; they establish GPU projection without correlation/device/context/stream columns.
When optional join identifiers and a CUDA GPU timeline are both available, declared operation
counts must match the joined rows. A CPU-only label appearing in the projection report makes
correlation incomplete. Correlation is checked in both directions before display truncation. Candle,
device-event, and Nsight clocks are never overlaid as one clock.

Run `cargo bench --bench instrumentation_overhead --all-features` to measure the disabled capture
gate and active event serialization on the current machine.

Do not parse exported SQLite: its schema is not the candle-graph interface.
