# candle-graph

> Independent tool — not affiliated with candle-rs or Hugging Face.

candle-graph captures one representative Candle execution and turns it into trustworthy evidence
for humans and building agents: semantic timing, tensor and gradient facts, explicit coverage gaps,
baseline comparisons, and optional NVIDIA Nsight GPU evidence in one standalone HTML viewer.

There is no static Rust analysis. Every claim comes from one concrete run.

## What agents get

- `summary`: bounded provenance, trace health, findings, gaps, and totals.
- `query`: slow spans/ops, storage footprints, explicit memory lifecycle, span tree, tensors, or gradients.
- `compare`: repeated semantic spans aggregated by path against an explicit baseline.
- `report`: the same evidence packet as JSON and concise Markdown.
- `view`: Evidence, Trace, Span costs, Memory, and GPU views in one offline HTML file.

Invalid hierarchy, incomplete spans, cycles, and inconsistent timings fail before graph analysis.
Missing optional evidence is reported as a gap rather than appearing as an empty result.

## Capture one update

```rust
use candle_graph::{ExecutionStep, ProfileRun, SpanKind, TraceSession};

let run = ProfileRun::training("my_crate::train::update", 1, "cuda:0")
    .measured_region_device_synchronized()
    .tag("physical_batch", "128")
    .tag("precision", "f32");
let session = TraceSession::open("application.jsonl", run)?;

{
    let _update = session.begin_measurement("my_crate/update-000000000001");
    {
        let _forward = session.begin_step_span(
            "my_crate/update-000000000001/forward",
            ExecutionStep::Forward,
            SpanKind::Function,
        );
        // representative forward pass
    }
    // backward and optimizer use the corresponding ExecutionStep values.
}

session.finish()?;
```

`TraceSession` owns an outer envelope; `begin_measurement` marks the caller-controlled region used
for comparisons, excluding trace finalization overhead. Capture exactly one selected update, not
every hot-loop iteration.

## Analyze

```bash
cargo candle-graph summary application.jsonl
cargo candle-graph query application.jsonl --kind gradients
cargo candle-graph query application.jsonl --kind tensors
cargo candle-graph compare baseline.jsonl application.jsonl --output comparison.json
cargo candle-graph report application.jsonl \
  --json evidence.json --markdown EVIDENCE.md
cargo candle-graph view application.jsonl --output viewer.html
```

Add normalized official `nsys stats --format csv` reports without parsing Nsight's unstable SQLite
export:

```bash
cargo candle-graph report application.jsonl --nsight-dir nsight \
  --json evidence.json --markdown EVIDENCE.md
cargo candle-graph view application.jsonl --nsight-dir nsight --output viewer.html
```

The raw `.nsys-rep` remains the source artifact. Supported reports are `cuda_gpu_trace`,
`cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum`, and `nvtx_gpu_proj_trace`.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/6` | Execution JSONL with provenance, measured region, start times, and facts |
| `candle-graph/graph/3` | Validated execution graph |
| `candle-graph/evidence/1` | Agent/human packet: health, facts, gaps, graph, comparison, GPU evidence |
| `candle-graph/comparison/1` | Baseline deltas by semantic path, memory, and gradient |
| `candle-graph/viewer/4` | Offline unified viewer payload |

See [CONTEXT.md](CONTEXT.md), [runtime guide](docs/runtime-analysis-guide.md),
[features](docs/features.md), and [visualizer](docs/visualizer.md).

## Development

```bash
cargo fmt -- --check
cargo test --all-features
cargo clippy --all-features -- -D warnings
```
