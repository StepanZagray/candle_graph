# candle-graph

> Independent tool — not affiliated with candle-rs or Hugging Face.

candle-graph captures one representative Candle execution and turns it into trustworthy evidence
for humans and building agents: capability-qualified host/device timing, tensor and gradient facts,
logical and physical memory evidence, repeated comparisons, and provenance-bound NVIDIA Nsight
evidence in one standalone HTML viewer.

There is no static Rust analysis. Every claim comes from one concrete run.

## What agents get

- `summary`: bounded provenance, structural outcome, capability matrix, GPU status, findings, gaps, and totals.
- `query`: host/device costs, storage lifetimes, physical memory, spans, tensors, gradients, or bounded GPU evidence.
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
        implementation_id: Some("build-v1".into()),
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

## Declare exact gradient coverage from a VarMap

Build the manifest after model construction, sort the `VarMap` keys because its map iteration is
not stable, and supply an application-owned family policy. Only add family contracts that have
members:

```rust
use std::collections::BTreeMap;
use candle_graph::{
    CaptureContract, CoverageLevel, ExpectedGradient, GradientContract,
    GradientFamilyContract, GradientFamilyExpectation, MeasurementScope,
};
use candle_nn::VarMap;

fn contract_from_varmap(varmap: &VarMap) -> anyhow::Result<GradientContract> {
    let mut keys = varmap
        .data()
        .lock()
        .expect("VarMap lock poisoned")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();

    let mut policies = BTreeMap::new();
    let expected = keys
        .into_iter()
        .map(|key| {
            // Replace these prefixes with the application's actual parameter-family policy.
            let (family, expectation, min_present) = if key.starts_with("frozen.") {
                ("frozen", GradientFamilyExpectation::Inactive, 0)
            } else if key.starts_with("conditional.") {
                ("conditional", GradientFamilyExpectation::DataConditional, 1)
            } else {
                ("trainable", GradientFamilyExpectation::Active, 1)
            };
            policies.insert(family.to_owned(), (expectation, min_present));
            ExpectedGradient::new("parameters", key, family)
        })
        .collect();
    let families = policies
        .into_iter()
        .map(|(family, (expectation, min_present))| match expectation {
            GradientFamilyExpectation::Active => {
                GradientFamilyContract::active(family, min_present)
            }
            GradientFamilyExpectation::Inactive => GradientFamilyContract::inactive(family),
            GradientFamilyExpectation::DataConditional => {
                GradientFamilyContract::data_conditional(family, min_present)
            }
        })
        .collect();
    GradientContract::new(expected, families)
}

let gradient_contract = contract_from_varmap(&varmap)?;
let capture_contract = CaptureContract {
    measurement_scope: MeasurementScope::ProductionEquivalent,
    gradients: CoverageLevel::Complete,
    gradient_contract: Some(gradient_contract),
    ..CaptureContract::default()
};
```

Record every declared `(root, key)` exactly once. `Present` requires a finite positive norm,
`Zero` requires positive `0.0`, and `Missing`/`NonFinite` carry no numeric norm. Complete coverage
is granted only after exact key, digest, state, and family validation. The public schema constants
are `TRACE_SCHEMA`, `EVIDENCE_SCHEMA`, `COMPARISON_SCHEMA`, `BUNDLE_SCHEMA`, and
`GRADIENT_MANIFEST_SCHEMA`.

## Analyze

```bash
cargo candle-graph summary evidence-bundle
cargo candle-graph query application.jsonl --kind slowest-host
cargo candle-graph query evidence-bundle --kind gpu-status
cargo candle-graph query evidence-bundle --kind gpu-correlation
cargo candle-graph query evidence-bundle --kind gpu-phases
cargo candle-graph query evidence-bundle --kind gpu-kernels
cargo candle-graph query evidence-bundle --kind gpu-attribution-gaps
cargo candle-graph query application.jsonl --kind gradients
cargo candle-graph query application.jsonl --kind tensors
cargo candle-graph report base-1.jsonl --bundle base-1.bundle
# Publish the other baseline/candidate runs the same way, then compare finalized bundles.
cargo candle-graph compare \
  --baseline base-1.bundle base-2.bundle base-3.bundle base-4.bundle base-5.bundle \
  --candidate next-1.bundle next-2.bundle next-3.bundle next-4.bundle next-5.bundle \
  --output comparison.json
cargo candle-graph report application.jsonl --bundle evidence-bundle
cargo candle-graph verify evidence-bundle --output verification.json
cargo candle-graph view application.jsonl --output viewer.html
```

`summary` and `query` accept a raw trace, a finalized bundle/profile directory, or that bundle's
`trace.jsonl`. Bundle inputs are deeply verified and read from their verified `evidence.json`, so
identity-bound normalized Nsight evidence is retained. The bundle is deeply verified before and
after reading its trace and packet, and both point-in-time receipts must match. This narrows the
verification/read race but does not make mutable storage atomic. A parent directory containing
`evidence.json` or `nsight/` without `bundle.json` is rejected, regardless of the input filename,
instead of being silently treated as trace-only. Raw trace inputs from unaugmented directories
remain supported; their GPU status is explicitly `unavailable` because no Nsight source was
supplied.

GPU queries are bounded. `gpu-phases` separates projected Nsight ranges from exact joined GPU-busy
attribution, and `gpu-attribution-gaps` reports missing, unexpected, CPU-only, duplicate, and
matched-but-unattributed semantic labels. Candle host, device-event, and Nsight clocks remain
separate in every result.

`compare` deeply verifies every bundle immediately before reading its bound trace. Raw trace
comparison remains available through `--unverified-traces`, but its result is always marked
diagnostic and ineligible for a performance verdict.

The `slowest-host` headline covers both the measured subtree and structurally separate host spans
that overlap the measured interval. Every entry declares `scope` as `measured_subtree` or
`concurrent_overlap` and reports full plus overlap-clipped duration/self-time fields. Concurrent
entries retain their real parent links; overlap-clipped values are attribution, not durations to
sum with measured wall time.

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
Required application labels can be partitioned into `gpu_expected_semantic_labels` and
`cpu_only_semantic_labels`; legacy contracts with neither field continue to classify every required
label as GPU-expected. Projected CPU-only labels are reported as correlation failures.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/9` | Execution JSONL with exact gradient contracts, timing/memory planes, and terminal outcome |
| `candle-graph/graph/4` | Validated call/data graph with tensor nodes and measured-scope host attribution |
| `candle-graph/evidence/3` | Capability-qualified packet with typed gradient-contract facts and explicit unknowns |
| `candle-graph/comparison/4` | Bundle-verified replicated outer-wall comparison with input receipts |
| `candle-graph/viewer/5` | Offline unified viewer payload |
| `candle-graph/gradient-manifest/1` | Ordered `(root, key, family)` digest domain |
| `candle-graph/nsight-capture/1` | Raw-report and CSV provenance binding |
| `candle-graph/bundle/1` | Content-addressed atomic evidence bundle |
| `candle-graph/bundle-verification/1` | Deep bundle verification receipt |

Version 0.9 rejects trace/8. Producers declaring complete gradient coverage must supply a
`GradientContract`; otherwise use partial or none. Comparison/4 takes finalized bundle directories
by default. Evidence/3 and comparison/4 consumers must accept their new contract/provenance fields.

See [CONTEXT.md](CONTEXT.md), [runtime guide](docs/runtime-analysis-guide.md),
[features](docs/features.md), and [visualizer](docs/visualizer.md).

## Development

```bash
cargo fmt -- --check
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo bench --bench instrumentation_overhead --all-features
```
