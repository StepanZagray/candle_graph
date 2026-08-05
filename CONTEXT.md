# candle-graph — product context

## Goal

Help people and agents understand **what a Candle model actually did** in one concrete run — span
hierarchy, op timings, tensor metadata, and gradient facts — without reading all the Rust and
without pretending static source analysis can reconstruct tensor flow.

This is the same contract as **TensorFlow Profiler**: the graph is built from **execution evidence**,
not from parsing arbitrary Rust.

## Trace-only model

```
┌─────────────────────────────────────┐
│  Probe binary (post-run / smoke)    │
│  TraceSession + SpanGuard           │
│  one forward / loss / eval pass     │
└──────────────────┬──────────────────┘
                   │  profile.jsonl (candle-graph/trace/5)
                   ▼
┌─────────────────────────────────────┐
│  build_from_trace → ExecutionGraph  │
│  self_time = total − children       │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   CLI (summary/query)   model.html (viewer/2)
```

| Question | Answer from |
| --- | --- |
| Which functions ran and for how long? | **Trace spans** |
| Which ops were slow on this batch shape? | **Trace ops + self-time** |
| Which ops/spans used the most memory? | **Trace memory events + bytes rollup** |
| Peak live tensors at OOM timestamp? | **Memory timeline + peak breakdown** |
| How long did each call/data edge take? | **Trace edges (ms labels in viewer)** |
| Did trainable params get gradients? | **Trace gradient events** |
| Module/parameter inventory from source alone? | **Out of scope** — use checkpoint tools or your trainer |

## Profiler UX targets

1. **Zero hot-loop overhead** — never wrap every training step; run the probe once after training or in CI smoke.
2. **Nested spans** — `Function` → `Module` → `Op` hierarchy like TF profiler.
3. **Self vs total time** — hot nodes highlight actual work, not time spent in children.
4. **Milliseconds on edges** — every call/data edge in HTML shows wall time.
5. **Bounded queries** — `slowest`, `heaviest`, `memory`, `spans`, `gradients` for agents without loading full graph JSON.

## Primary workflows

### Model authors (emit trace)

```rust
use candle_graph::{ExecutionPhase, SpanKind, TraceSession};
```

See [`docs/runtime-analysis-guide.md`](docs/runtime-analysis-guide.md).

### Humans (HTML)

```bash
cargo candle-graph view profile.jsonl --output model.html
```

### Agents / CI (JSON)

```bash
cargo candle-graph summary profile.jsonl
cargo candle-graph query profile.jsonl --kind slowest
cargo candle-graph query profile.jsonl --kind heaviest
cargo candle-graph query profile.jsonl --kind memory
cargo candle-graph query profile.jsonl --kind efficiency
```

## Related repos

- **Tofy** — sibling consumer; should emit `profile.jsonl` from a post-run probe and open with `cargo candle-graph view`.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/5` | JSONL execution trace (spans, ops, memory events) |
| `candle-graph/graph/2` | Derived execution graph + memory profile |
| `candle-graph/viewer/3` | HTML visualizer payload |
| `candle-graph/trace-query/1` | Bounded query responses |
