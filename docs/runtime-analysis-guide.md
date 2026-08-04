# Runtime analysis guide

This guide explains how to use candle-graph runtime evidence correctly: what it proves, what it
does not, and how to combine static analysis, gradient traces, phase-specific graphs, and optional
profiling.

For field-level schema details see [runtime-protocol.md](runtime-protocol.md). For the query map see
[agent-query-api.md](agent-query-api.md).

## Mental model

Static analysis is the source of truth for **structure** (components, parameters, tensor contracts,
dataflow, numeric-domain hazards). Runtime evidence is an optional **refinement layer** that records
what one concrete execution actually observed.

```
┌─────────────────────┐     ┌──────────────────────────┐
│  Static scan        │     │  Runtime trace (optional) │
│  (always first)     │────▶│  JSON or JSONL            │
│  model IR + findings│     │  v1 / v2 / v3             │
└─────────────────────┘     └──────────────────────────┘
           │                            │
           └──────── merge ─────────────┘
                        │
                        ▼
              refined IR + runtime query
              checkpoint audit + gradient facts
              profile timings (v3 only)
```

**Rules that never change:**

1. Run static analysis before trusting runtime output.
2. Match Cargo features/target to the binary that produced the trace.
3. Prefer fully qualified selectors and stable tensor IDs when correlating static ↔ runtime.
4. Treat conflicting observations as `unknown` — the importer does not pick a silent winner.
5. One healthy step-0 gradient trace does **not** prove training stays finite later.

## Schema versions

| Schema | Purpose | Typical transport |
| --- | --- | --- |
| `candle-graph/runtime/1` | Tensor metadata + per-parameter gradient facts | Single JSON doc or JSONL |
| `candle-graph/runtime/2` | Same as v1 plus optional `step` on tensors/gradients/values (time series) | JSON doc or JSONL |
| `candle-graph/runtime/3` | v2 fields plus timed `operations`, `edge_timings`, and `run.phase` (`train` / `infer`) | JSONL from `ProfileSession` |

All versions share the same import path (`--runtime-trace`). v3 events merge into the unified model
IR and power the `profile`, `graph-train`, and `graph-infer` queries.

## Recommended workflow

### 1. Static gate (required)

```bash
cargo candle-graph check /path/to/model --features cuda
cargo candle-graph query doctor --path /path/to/model
cargo candle-graph query findings --path /path/to/model
```

Read `doctor.json` coverage before acting on individual tensor records. Fix or acknowledge proven
static findings before spending time on runtime correlation.

### 2. Checkpoint + gradient audit (common)

When a training or smoke run exists, pass checkpoint keys and a runtime trace together:

```bash
cargo candle-graph audit runs/analyzer \
  --checkpoint runs/train/model.safetensors \
  --runtime-trace runs/train/runtime.json \
  --strict
```

Or query runtime evidence directly:

```bash
cargo candle-graph query runtime \
  --path /path/to/model \
  --runtime-trace runs/train/runtime.json
```

The audit checks:

- safetensors header keys vs static parameter discovery (`missing` / `unclaimed` tensors);
- gradient facts keyed by `(root, key)` — `present`, `missing`, `zero`, or `non_finite`;
- optional tensor observations that refine shape/dtype/device when `static_id` is set.

Use the same `--verify-root` / component `--root` namespace the trainer uses for `VarBuilder` (e.g. `vb`).

### 3. Phase-specific static graphs (no GPU required)

Training and inference often differ: inference severs trainable gradient edges in the static graph.
Query each phase separately:

```bash
cargo candle-graph query graph-train --path /path/to/model
cargo candle-graph query graph-infer --path /path/to/model
```

Entrypoints are tagged automatically (`forward` → train + infer, `leworld_loss` → train only,
`eval_step` → infer only). Tensor and operation IDs are prefixed with the phase
(`tensor:train:…`, `tensor:infer:…`).

Compare tensor counts and operation lists between phases before optimizing inference-only paths.

### 4. Profiling (optional, offline)

Profiling is **opt-in instrumentation** in the model crate, not something the analyzer injects into
a long training loop. Emit a v3 JSONL trace from a short synthetic or CPU rollout, then merge:

```bash
cargo candle-graph profile --runtime-trace my-profile.jsonl --path /path/to/model
cargo candle-graph query profile --path /path/to/model --runtime-trace my-profile.jsonl
```

The `profile` query returns slowest operations, edge timing rollups, and drill-down hints for
`graph-train` / `graph-infer`.

## CLI quick reference

| Goal | Command |
| --- | --- |
| Import trace into any query | `--runtime-trace FILE` on `check`, `report`, `query`, `audit` |
| Runtime rollup | `cargo candle-graph query runtime --runtime-trace FILE …` |
| Train-phase graph | `cargo candle-graph query graph-train …` |
| Infer-phase graph | `cargo candle-graph query graph-infer …` |
| Timings merge | `cargo candle-graph profile --runtime-trace FILE …` |
| Agent bundle | `cargo candle-graph audit OUT_DIR --checkpoint … --runtime-trace …` |

Legacy binary equivalent:

```bash
cargo run --release -- /path/to/model \
  --runtime-trace run.runtime.json \
  --query runtime --format json
```

## Emitting traces from Rust

### Gradient / tensor trace (v1 or v2)

Use `RuntimeTraceWriter` from a **short, bounded** rollout — smoke test, single training step on
CPU, or dedicated probe binary. No Candle dependency in the analyzer crate; your model crate writes
the file.

```rust
use candle_graph::runtime::{
    GradientFact, GradientState, RunMetadata, RuntimeTraceWriter, TensorObservation,
};

let file = std::fs::File::create("run.runtime.json")?;
let mut trace = RuntimeTraceWriter::new(file, RunMetadata {
    entrypoint: "model::train_step".into(),
    profile: "release".into(),
    cargo_features: vec!["cuda".into()],
    cfg: vec![],
    analysis_id: None, // set from `query summary` when available
    build_id: None,
    phase: None,
})?;

trace.tensor(TensorObservation {
    event_id: "step-0-loss".into(),
    static_id: None, // copy from `query tensors` when correlating
    source: Some("train_step total loss".into()),
    step: Some(0),
    shape: loss.dims().to_vec(),
    dtype: format!("{:?}", loss.dtype()),
    device: "cuda:0".into(),
    contiguous: loss.is_contiguous(),
    requires_grad: true,
    storage_id: None,
})?;

trace.gradient(GradientFact {
    event_id: "step-0-head-weight".into(),
    root: "vb".into(),
    key: "head.weight".into(),
    state: GradientState::Present,
    step: Some(0),
    norm: Some(grad_norm),
})?;

trace.finish()?;
```

For streaming multi-step capture, write JSONL: first event is `kind: "meta"`, then append
`tensor`, `gradient`, and `value` events. See [runtime-protocol.md](runtime-protocol.md).

### Timed profile trace (v3)

Use `ProfileSession` when you control the call sites and can bracket operations with stable static
IDs from a prior scan:

```rust
use candle_graph::profile::{ProfileConfig, ProfileSession, TimedOperation};

let mut session = ProfileSession::open(
    "forward-profile.jsonl",
    ProfileConfig::infer("WorldModel::forward"),
)?;

session.set_step(0);

{
    let _timed = TimedOperation::begin(
        &mut session,
        "operation:infer:WorldModel::forward:3", // static id from model IR
        "matmul",
        &[],
    )?;
    // ... run forward ...
}

session.record_edge_timing(
    "tensor:infer:WorldModel::forward:2",
    "tensor:infer:WorldModel::forward:4",
    edge_ns,
)?;

session.finish()?;
```

Use `ProfileConfig::train(…)` for training-phase profiles. Set `analysis_id` / `build_id` on the
config when you need strict identity matching against the active scan.

## JSON document vs JSONL

| Format | When to use |
| --- | --- |
| **Single JSON** (`runtime.json`) | One-shot snapshot: first training step, smoke eval, CI artifact |
| **JSONL** | Multi-step series, streaming profiler, long-but-bounded probes |

Import treats both the same. JSONL merges events in order; duplicate `event_id` values are rejected.

## Correlating static IDs

1. Run `cargo candle-graph query tensors --select 'MyModel::forward' --limit 50`.
2. Copy the `id` field into `TensorObservation.static_id` or profiler `begin_operation` arguments.
3. Re-import with `--runtime-trace` and run `query tensor --select '<id>'` to see refined evidence.

Without `static_id`, gradient audit still works via `(root, key)`. Tensor refinement requires the
static link.

Optional identity fields (`analysis_id`, `build_id` in `run`) prevent silently merging a trace from
a different code revision. When both sides supply them, mismatches surface as `unknown` evidence.

## What runtime analysis proves

| Claim | Supported? |
| --- | --- |
| Parameter X had a finite non-zero gradient in this run | Yes, via `(root, key)` + `present` |
| Parameter Y received no gradient | Yes, `missing` |
| Gradient was NaN/Inf when recorded | Yes, `non_finite` |
| Observed tensor shape/dtype/device at probe time | Yes, with optional `static_id` refinement |
| Operation took ~N ns on this run | Yes, v3 `duration_ns` / `profile` query |
| Train vs infer graph differ | Yes, static `graph-train` / `graph-infer` (+ v3 phase tag) |
| Model will never NaN later in training | **No** |
| Forward value stayed inside a safe numeric domain | Only if probe emits `values` observations |
| Architecture / pipeline order | **No** — static (+ `--heuristic-architecture`) only |

Static analysis proves **composition hazards** (e.g. saturated sigmoid feeding `log`). Runtime proves
the hazard was **exercised** in a particular run — not that it will or won't happen again.

## Anti-patterns

1. **Skipping static audit** and relying on a single healthy gradient snapshot.
2. **Instrumenting the hot training loop** with per-step JSONL unless you accept I/O overhead; prefer
   one-shot `runtime.json`, offline probes, or external GPU profilers (Nsight) for throughput work.
3. **Mismatched features** — tracing a `--features cuda` binary but analyzing with default features.
4. **Expecting value-range proof** without emitting `ValueObservation` records (min/max/abs_max).
5. **Treating heuristic architecture as proven** because a runtime trace exists.

## Consumer integration pattern

Downstream crates (trainers, eval harnesses) should treat candle-graph as an **optional audit
dependency**, not a training-time requirement:

- Emit at most one lightweight artifact during normal runs (e.g. step-0 `runtime.json`).
- Keep heavy JSONL / `ProfileSession` in dedicated smoke or benchmark binaries.
- Wire audit through Cargo (`cargo candle-graph audit …` or a thin wrapper subcommand).
- Document which entrypoint and `verify-root` the trace used so agents can replay the audit.

Example audit wrapper flags: `--checkpoint`, `--runtime-trace`, `--verify-root vb`,
`--strict`, `--deny numeric-domain-violation,…`.

## Related reading

- [runtime-protocol.md](runtime-protocol.md) — JSON/JSONL fields, gradient states, conflict rules
- [agent-query-api.md](agent-query-api.md) — full query table and progressive disclosure
- [numeric-domain.md](numeric-domain.md) — static float-range hazards vs runtime value observations
- [compiler-evidence-design.md](compiler-evidence-design.md) — what static analysis cannot yet derive
