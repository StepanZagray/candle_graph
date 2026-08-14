# Runtime evidence guide

## Select one representative invocation

Capture a configurable one-based update, defaulting to update 1. Record `capture_step`, warmup
count, device, timing mode, batch/accumulation, precision, workload, and source revision so an agent
can decide whether two runs are comparable. Instrumentation is inactive outside the selected run.

On CUDA, use `device_synchronized` only when every reported semantic span is bounded by device
synchronization. If only the single measured region is synchronized, use
`measured_region_device_synchronized`; `timing_mode` then remains `host`, so nested host spans are
not presented as completed GPU work. Nsight or another device profiler is still required for
kernel-level attribution.

## Required shape

Every trace contains one session envelope and exactly one measured region. Training captures tag
forward, backward, and optimizer spans with `ExecutionStep`. Use the same stable semantic labels in
NVTX, for example `tofy.p2/update-000000000001/forward`.

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

`SpanGuard::id()` attaches op or tensor metadata. Tensor/storage metadata does not imply allocation
lifetime. Only explicit paired `record_memory_alloc`/`record_memory_free` events drive live and peak
memory; `record_device_memory` supplies allocator/device checkpoints.

With feature `candle`, `CandleCapture::from_tensor` and `candle::record_tensor` capture shape,
dtype, device, storage footprint, and variable status without inventing frees.

## Publish evidence

```bash
cargo candle-graph summary application.jsonl
cargo candle-graph query application.jsonl --kind slowest
cargo candle-graph report application.jsonl \
  --baseline baseline.jsonl \
  --nsight-dir nsight \
  --json evidence.json \
  --markdown EVIDENCE.md
cargo candle-graph view application.jsonl \
  --baseline baseline.jsonl \
  --nsight-dir nsight \
  --output viewer.html
```

The packet declares structural trust, section coverage, known gaps, structured numeric facts,
comparison compatibility, and per-report Nsight coverage/truncation. Absence of Nsight never
invalidates application evidence.

## Nsight normalization

Retain `.nsys-rep`, then export supported reports with NVIDIA's `nsys stats --format csv`. Pass the
directory to candle-graph. Exact semantic NVTX labels join projected GPU ranges to Candle phases;
the packet explicitly states that Candle and Nsight clocks are not overlaid as one clock.

Do not parse exported SQLite: its schema is not the candle-graph interface.
