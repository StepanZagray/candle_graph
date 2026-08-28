# candle-graph

> Independent project; not affiliated with candle-rs or Hugging Face.

`candle-graph` records one selected Candle training update or inference invocation and turns that
runtime trace into inspectable evidence. It can produce JSON for tools, a Markdown report, a
content-addressed bundle, and one offline HTML viewer.

The crate does **not** inspect Rust source or reconstruct execution statically. It can only report
events that ran and that the application chose to record.

## What it records

| Evidence | Meaning |
| --- | --- |
| Host spans | A semantic call hierarchy and one caller-owned measured region |
| Operations and tensors | Executed op metadata, backend tensor identities, shapes, dtypes, and dense footprints |
| Tensor statistics | Caller-labelled RMS, absolute maximum, mean, and non-finite counts |
| Gradients | Per-parameter state and norm, optionally checked against an exact manifest |
| Logical memory | Explicit storage allocation/free lifetimes keyed by device and storage identity |
| Physical memory | Independent device or allocator checkpoints such as used, free, and reserved bytes |
| Device timing | Explicit intervals that remain separate from host time |
| GPU evidence | Optional normalized NVIDIA Nsight Systems CSV reports bound to the trace by a manifest |

Absent evidence stays absent. Tensor size metadata does not become an allocation, host duration
does not become GPU duration, and an observed event does not silently upgrade partial coverage to
complete coverage.

## Install

From a checkout:

```bash
cargo install --path . --locked
```

The default `visualizer` feature includes the `view` command and adds `viewer.html` to published
bundles. Add `--features all` to install the optional Candle tensor helpers as well. See
[Cargo features](docs/features.md) for the exact feature boundaries.

The package installs both `candle-graph` and the Cargo subcommand form used below:

```bash
cargo candle-graph --help
```

## Capture one invocation

Open a session only for the selected one-based invocation. Put exactly one measured region inside
the session; that region supplies the outer-wall duration used by comparisons.

```rust
use anyhow::Result;
use candle_graph::{
    CaptureContract, CaptureSelector, CoverageLevel, ExecutionStep, MeasurementScope, ProfileRun,
    SpanKind, TraceSession,
};

fn maybe_capture_update(update: u64) -> Result<()> {
    let selector = CaptureSelector::new(10)?;
    if !selector.is_selected(update) {
        return Ok(());
    }

    let prefix = format!("trainer/update-{update:012}");
    let forward_label = format!("{prefix}/forward");
    let contract = CaptureContract {
        measurement_scope: MeasurementScope::ProfiledWork,
        operations: CoverageLevel::Partial,
        tensors: CoverageLevel::Partial,
        required_semantic_labels: vec![forward_label.clone()],
        ..CaptureContract::default()
    };
    let run = ProfileRun::training("trainer::update", update, "cpu")
        .capture_contract(contract);
    let session = TraceSession::open("application.jsonl", run)?;

    {
        let _measured = session.begin_measurement(&prefix);
        {
            let _forward = session.begin_step_span(
                &forward_label,
                ExecutionStep::Forward,
                SpanKind::Function,
            );
            // Run and record the representative forward pass here.
        }
        // Add backward and optimizer spans for a training capture.
    }

    session.finish()?;
    Ok(())
}
```

For a CUDA comparison, synchronize immediately before and after the measured region and mark the
run with `measured_region_device_synchronized()`. Use `device_synchronized()` only if every nested
semantic span is individually synchronized; otherwise its duration is host launch time, not
completed GPU work.

The [runtime evidence guide](docs/runtime-analysis-guide.md) covers comparison identity, semantic
labels, Candle helpers, tensor statistics, gradient contracts, memory, concurrent host spans, and
failed captures.

## Inspect the trace

```bash
# Human-readable overview on stdout.
cargo candle-graph summary application.jsonl

# Focused JSON queries.
cargo candle-graph query application.jsonl --kind slowest-host
cargo candle-graph query application.jsonl --kind tensor-stats
cargo candle-graph query application.jsonl --kind memory --output memory.json

# One offline HTML file (default feature).
cargo candle-graph view application.jsonl --output viewer.html

# Atomic bundle containing trace, evidence, report, hashes, and viewer.
cargo candle-graph report application.jsonl --bundle evidence-bundle
cargo candle-graph verify evidence-bundle --output verification.json
```

`summary`, `import`, and `query` accept either a raw trace or a finalized bundle. Prefer a bundle
when one exists: it is deeply verified and preserves its bound Nsight evidence. Some focused
queries have fixed row caps, but collection queries such as `memory`, `spans`, `tensors`,
`tensor-stats`, and `gradients` can grow with the trace. The
[CLI reference](docs/cli-reference.md) lists every command, input type, query, and size property.

## Add Nsight Systems evidence

Export supported reports with `nsys stats --format csv`; do not depend on Nsight's internal SQLite
schema. Put the retained `.nsys-rep`, CSV files, and `capture-manifest.json` in one flat directory:

```bash
cargo candle-graph report application.jsonl \
  --nsight-dir nsight \
  --bundle evidence-bundle

cargo candle-graph query evidence-bundle --kind gpu-status
cargo candle-graph query evidence-bundle --kind gpu-correlation
cargo candle-graph query evidence-bundle --kind gpu-phases
```

Supported reports are `cuda_gpu_trace`, `cuda_gpu_kern_sum`, `cuda_api_sum`,
`cuda_gpu_mem_time_sum`, and `nvtx_gpu_proj_trace`. The manifest binds artifact hashes, the trace
run ID, the correlation ID, and the semantic-label contract. Missing reports remain individually
unavailable; their row counts are not inferred as zero.

## Compare implementations

Timing verdicts require at least five independent, complete, production-equivalent bundles in each
cohort. Every bundle is verified, every run keeps its raw sample, and compatible runs receive a
deterministic 95% bootstrap interval.

```bash
cargo candle-graph compare \
  --baseline base-1.bundle base-2.bundle base-3.bundle base-4.bundle base-5.bundle \
  --candidate next-1.bundle next-2.bundle next-3.bundle next-4.bundle next-5.bundle \
  --output comparison.json
```

The capture contract, timing semantics, workload identity, model/config/data identity, batch and
precision settings, and device state must match. Each cohort has one stable implementation ID;
the baseline and candidate IDs may differ. Raw traces can be inspected with
`--unverified-traces`, but that result is always diagnostic and ineligible for a timing verdict.

## Trust model

- A complete trace must have one root, one measured region, closed spans, a valid hierarchy, and a
  successful terminal event.
- Failed captures remain parseable only when the caller explicitly finishes them with
  `finish_failed(reason)`; they do not produce a derived execution graph or normal findings.
- Coverage levels are producer declarations qualified by observed evidence. Exact gradient
  coverage additionally requires a digest-bound parameter manifest and family validation.
- Logical storage lifetime, physical memory, host time, device-event time, and Nsight time are
  separate evidence planes.
- A bundle manifest binds every published file. `verify` rejects missing, changed, undeclared
  files, symbolic links, and special files.

## Documentation

- [Documentation map](docs/README.md)
- [Runtime evidence guide](docs/runtime-analysis-guide.md)
- [CLI reference](docs/cli-reference.md)
- [Schema and compatibility reference](docs/schemas.md)
- [Cargo features](docs/features.md)
- [HTML visualizer](docs/visualizer.md)
- [Product context and glossary](CONTEXT.md)

## Development

```bash
cargo fmt -- --check
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo bench --bench instrumentation_overhead --all-features
```
