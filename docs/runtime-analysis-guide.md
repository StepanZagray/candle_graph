# Runtime evidence guide

This guide describes how to produce evidence that `candle-graph` can qualify. The central rule is
simple: instrument one representative runtime invocation and describe honestly what the probe did
and did not cover.

## 1. Select before opening a session

`CaptureSelector` uses one-based invocation numbers. Put the gate outside the hot-path session so
unselected invocations do not open files or serialize events:

```rust
let selector = CaptureSelector::new(selected_invocation)?;
if !selector.is_selected(current_invocation) {
    return run_without_capture();
}
```

`ProfileRun::training` and `ProfileRun::inference` record the entrypoint, selected invocation,
warmup count, phase, device, and correlation ID. Use `tag` only for descriptive provenance; tags do
not establish comparison compatibility.

## 2. Declare the capture contract

`CaptureContract` is a producer assertion. Each coverage field describes the **whole measured
region**, not merely whether the trace happens to contain one event.

| Field | Declare `complete` only when… |
| --- | --- |
| `operations` | every operation in scope is represented |
| `tensors` | every tensor checkpoint required by the application policy is represented |
| `gradients` | every exact manifest entry and family rule is represented and validated |
| `logical_memory` | all storage allocations/frees in scope are represented |
| `physical_memory` | the application's complete checkpoint policy ran |
| `device_timing` | all device intervals required by the adapter policy are represented |

Use `partial` for a deliberate sample and `none` when the producer did not attempt that evidence.
Observed events can raise `none` to partial during analysis, but they never prove complete coverage.

`measurement_scope` has three meanings:

- `unknown`: representativeness was not established;
- `profiled_work`: valid diagnostic work, but not claimed equivalent to production; and
- `production_equivalent`: same relevant workload conditions as a normal run. This is required for
  an eligible timing comparison.

## 3. Record one measured region

Every successful trace has one session envelope and exactly one measured region. Session setup and
evidence publication are outside that region.

```rust
let session = TraceSession::open(
    path,
    ProfileRun::training("trainer::update", update, "cpu")
        .capture_contract(contract),
)?;

{
    let _measured = session.begin_measurement("trainer/update-000000000010");
    {
        let _forward = session.begin_step_span(
            "trainer/update-000000000010/forward",
            ExecutionStep::Forward,
            SpanKind::Function,
        );
        // forward pass
    }
    // backward and optimizer spans
}

session.finish()?;
```

`SpanGuard` is RAII: dropping it emits `span_end`. Drop nested guards before finishing the session.
`SpanGuard::id()` is the opaque identity used by op, tensor, memory, and device-interval recording
methods.

### Semantic labels

`required_semantic_labels` is an exact cardinality contract. Every label must be non-empty and
unique, and every declared label must name exactly one span in a successful trace. Reuse the same
labels for NVTX ranges.

For GPU correlation, either leave both classification lists empty (legacy behavior: every required
label is GPU-expected) or partition every required label exactly once between:

- `gpu_expected_semantic_labels`; and
- `cpu_only_semantic_labels`.

A missing, repeated, or incorrectly partitioned required label invalidates a successful capture.
The same condition remains a diagnostic warning when the terminal outcome is failed.

## 4. Keep timing planes separate

### Host timing

Span guards measure host wall intervals. On an asynchronous backend, this is often launch latency,
not completed device work.

For CPU execution, host outer-wall time is comparison-eligible without a device synchronization
flag. For CUDA or another non-CPU device, synchronize immediately before and after the measured
region, then call `measured_region_device_synchronized()` on `ProfileRun`.

`device_synchronized()` is stronger: it sets `timing_mode` to `device_synchronized` and asserts
that every reported semantic span is bounded by device synchronization. Do not use it when only the
outer measured region is synchronized.

### Device intervals

`record_device_interval` accepts intervals from a device timing adapter. They are grouped by
`(device, clock_id)` and unioned across overlapping streams so simultaneous work is counted once.
They are never aligned or added to host time unless the producer has an explicit clock mapping;
`candle-graph` does not invent one.

### Concurrent host work

For already-completed work that ran concurrently with the live call stack:

1. retain its `std::time::Instant` start;
2. convert it with `TraceSession::host_timestamp_ns`; and
3. call `record_completed_host_span` after the interval ends.

The positive interval is attached to the session root without modifying the span stack. Host
headlines report it as `concurrent_overlap`, clipped to the measured interval. It keeps its real
parent and must not be added to measured wall time. Normal descendants of the measured span are
reported as `measured_subtree`.

## 5. Record tensor and operation metadata

With the `candle` feature, `CandleCapture::from_tensor` extracts:

- process-local backend tensor and storage identities;
- shape, dtype, device, and variable state;
- a semantic memory category; and
- the dense tensor footprint.

Use `with_label` for a caller-owned observation label. The label never replaces backend tensor
identity. `candle::record_tensor` and `candle::record_op` translate the owned capture values into
session events.

Dense footprint is `element_count × dtype_size`. It is useful shape metadata, but it is not the
size of shared or padded backing storage. Record actual allocation bytes separately.

### Numerical tensor statistics

`TraceSession::record_tensor_stats` (feature `candle`) reduces a caller-labelled tensor to element
count, non-finite count, RMS, absolute maximum, and mean. Labels are the join keys used by repeated
comparisons, so make them stable and unique within a run. If a label repeats within a run, every
event is still averaged into the comparison and the per-row sample counts expose the repetition.

The current API accepts the wire span ID as a string:

```rust
let forward = session.begin_step_span(label, ExecutionStep::Forward, SpanKind::Function);
let span_id = format!("s{}", forward.id().raw());
session.record_tensor_stats(&span_id, "model/block-0/output", &output)?;
```

These are real Candle reductions. On an accelerator they can add kernels, transfers, and scalar
synchronization to the selected invocation. Record the same checkpoints in both comparison
cohorts and interpret performance results as the instrumented workload. If any element is NaN or
infinite, `non_finite` is authoritative; non-finite aggregate values are serialized as zero so the
event remains valid JSON.

## 6. Record memory as two independent planes

### Logical storage lifetime

Pair `record_memory_alloc` and `record_memory_free` by `(device, storage_id)`. Aliased tensor IDs
may share one live storage; repeating an allocation for the same tensor/storage pair is invalid.
Pass the backend's actual allocation bytes rather than the tensor's dense footprint.

Logical analysis produces simultaneous live bytes, peak composition, per-device totals, and
residual storage. A storage with no observed free remains a retained allocation and creates a
warning rather than a fabricated end time.

### Physical memory

`record_device_memory` accepts a `DeviceMemoryRecord` whose `used_bytes`, `free_bytes`,
`reserved_bytes`, and `capacity_bytes` fields are independently optional. At least one must be
present. Capacity is never derived from used plus free, and physical checkpoints never fill gaps in
logical lifetime evidence.

## 7. Qualify exact gradient coverage

Complete gradient coverage requires a `GradientContract` created from the final parameter set
before opening the session. The expected vector is caller-ordered; sort `VarMap` keys because map
iteration order is not a stable protocol.

```rust
use std::collections::BTreeMap;

use candle_graph::{
    CaptureContract, CoverageLevel, ExpectedGradient, GradientContract, GradientFamilyContract,
    GradientFamilyExpectation, MeasurementScope,
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
            // Replace these prefixes with the application's parameter-family policy.
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
let contract = CaptureContract {
    measurement_scope: MeasurementScope::ProductionEquivalent,
    gradients: CoverageLevel::Complete,
    gradient_contract: Some(gradient_contract),
    ..CaptureContract::default()
};
```

Record every declared `(root, key)` exactly once with `TraceSession::record_gradient`:

- `Present` requires a finite norm greater than zero;
- `Zero` requires positive `0.0`, never `-0.0`;
- `Missing` and `NonFinite` require no numeric norm;
- `active(family, min_present)` requires enough positive finite gradients;
- `inactive(family)` requires every member to be explicitly missing; and
- `data_conditional(family, min_present)` allows all members to be missing, but requires enough
  positive finite gradients once any are present.

The digest algorithm and schema constant locations are in the
[schema reference](schemas.md#gradient-manifest-digest).

## 8. Add comparison identity

Timing comparison needs a typed identity on every run:

```rust
use candle_graph::ComparisonIdentity;

let identity = ComparisonIdentity {
    implementation_id: Some("git:abc123".into()),
    workload_id: "training-step".into(),
    model_id: "transformer-small".into(),
    config_id: "default".into(),
    data_id: "batch-set-a".into(),
    seed_policy: "fixed-42".into(),
    physical_batch: 128,
    accumulation_steps: 1,
    precision: "f32".into(),
    device_state: "exclusive".into(),
    pair_id: None,
};

let run = run.comparison_identity(identity);
```

Use a stable implementation/build identity within a cohort. `pair_id` is optional; if it appears
on any run, it must appear exactly once in each cohort with matching sets.

## 9. Finish failures explicitly

Call `finish()` only after successful work. When the profiled invocation returns an error that the
application can catch, call `finish_failed(reason)` to retain a terminal failure reason and partial
structural evidence.

A panic, abort, process kill, or I/O failure cannot be converted automatically into a valid failed
trace. Such a file may lack its required terminal event and fail parsing. Arrange application-level
error handling when diagnosable failure artifacts are required.

## Capture checklist

- The selector runs before `TraceSession::open`.
- The capture contains exactly one measured region.
- Coverage levels describe the entire measured region honestly.
- Required semantic labels are stable, unique, and matched in the trace and NVTX.
- A non-CPU comparison synchronizes the boundaries of the measured region.
- Tensor footprint is not reported as allocation size.
- Logical and physical memory observations remain separate.
- Complete gradients have an exact ordered manifest and one event per key.
- Comparison identity fields describe the real workload and environment.
- Success calls `finish`; caught failure calls `finish_failed`.

After capture, publish and inspect evidence with the [CLI reference](cli-reference.md).
