# candle-graph

> Independent tool — not affiliated with candle-rs or Hugging Face.

candle-graph captures one representative Candle execution and turns it into trustworthy evidence
for humans and building agents: capability-qualified host/device timing, tensor and gradient facts,
logical and physical memory evidence, repeated comparisons, and provenance-bound NVIDIA Nsight
evidence in one standalone HTML viewer.

There is no static Rust analysis. Every claim comes from one concrete run.

## What agents get

- `summary`: bounded provenance, structural outcome, capability matrix, findings, gaps, and totals.
- `query`: host/device costs, storage lifetimes, physical memory, spans, tensors, or gradients.
- `compare`: at least five compatible baseline and candidate runs with raw samples and a 95% bootstrap interval.
- `report`: an atomically published, content-addressed evidence bundle.
- `verify`: deep bundle verification with a durable manifest-digest receipt.
- `view`: Evidence, Trace, Span costs, Memory, and GPU views in one offline HTML file.

Invalid hierarchy, incomplete spans, cycles, and inconsistent timings fail before graph analysis.
Failed captures remain diagnosable. Missing evidence is unknown, never a silent zero.

## Capture one update

```rust
use candle_graph::{
    CaptureContract, CaptureSelector, ComparisonIdentity, CoverageLevel, ExecutionStep,
    MeasurementScope, ProfileRun, SpanKind, TraceSession,
};

let update_number = 1;
let selector = CaptureSelector::new(1)?;
if !selector.is_selected(update_number) {
    return Ok(());
}
let run = ProfileRun::training("my_crate::train::update", 1, "cuda:0")
    .measured_region_device_synchronized()
    .capture_contract(CaptureContract {
        measurement_scope: MeasurementScope::ProductionEquivalent,
        operations: CoverageLevel::Partial,
        tensors: CoverageLevel::Partial,
        gradients: CoverageLevel::Partial,
        required_semantic_labels: vec!["my_crate/update-000000000001/forward".into()],
        ..CaptureContract::default()
    })
    .comparison_identity(ComparisonIdentity {
        workload_id: "train".into(), model_id: "model-v1".into(), config_id: "default".into(),
        data_id: "batch-set-a".into(), seed_policy: "fixed-42".into(), physical_batch: 128,
        accumulation_steps: 1, precision: "f32".into(), device_state: "exclusive".into(),
        pair_id: None,
    });
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
cargo candle-graph compare \
  --baseline base-1.jsonl base-2.jsonl base-3.jsonl base-4.jsonl base-5.jsonl \
  --candidate next-1.jsonl next-2.jsonl next-3.jsonl next-4.jsonl next-5.jsonl \
  --output comparison.json
cargo candle-graph report application.jsonl --bundle evidence-bundle
cargo candle-graph verify evidence-bundle --output verification.json
cargo candle-graph view application.jsonl --output viewer.html
```

Add normalized official `nsys stats --format csv` reports without parsing Nsight's unstable SQLite
export:

```bash
cargo candle-graph report application.jsonl --nsight-dir nsight \
  --bundle evidence-bundle
cargo candle-graph view application.jsonl --nsight-dir nsight --output viewer.html
```

The bundle retains the raw `.nsys-rep` and normalized inputs. A `capture-manifest.json` with schema
`candle-graph/nsight-capture/1` binds their hashes and run/correlation IDs to the trace. Supported reports are `cuda_gpu_trace`,
`cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum`, and `nvtx_gpu_proj_trace`.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/7` | Execution JSONL with capture contract, timing/memory planes, and terminal outcome |
| `candle-graph/graph/4` | Validated call/data graph with tensor nodes |
| `candle-graph/evidence/2` | Capability-qualified packet with typed facts and explicit unknowns |
| `candle-graph/comparison/2` | Fail-closed replicated outer-wall comparison |
| `candle-graph/viewer/5` | Offline unified viewer payload |
| `candle-graph/nsight-capture/1` | Raw-report and CSV provenance binding |
| `candle-graph/bundle/1` | Content-addressed atomic evidence bundle |
| `candle-graph/bundle-verification/1` | Deep bundle verification receipt |

See [CONTEXT.md](CONTEXT.md), [runtime guide](docs/runtime-analysis-guide.md),
[features](docs/features.md), and [visualizer](docs/visualizer.md).

## Development

```bash
cargo fmt -- --check
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo bench --bench instrumentation_overhead --all-features
```
