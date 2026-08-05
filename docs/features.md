# Cargo features

Only three feature names exist. There is no `static` or `runtime` feature.

| Feature | Default | Purpose |
| --- | --- | --- |
| **`default`** | — | Enables `visualizer` (see `[features] default` in `Cargo.toml`) |
| **`visualizer`** | yes | Standalone `model.html` from trace files |
| **`candle`** | | Candle tensor capture helpers (`CandleCapture`, `record_op`) |
| **`all`** | | Alias for `visualizer` |

```toml
[features]
default = ["visualizer"]
visualizer = []
all = ["visualizer"]
```

```bash
cargo install --path . --features all --locked
```

Build without HTML (library + CLI import/summary/query only):

```bash
cargo build --no-default-features
```

## Modules (always available)

### Trace (`instrument` + `trace`)

Emit and parse `candle-graph/trace/5` JSONL from a **probe binary** — not from the training hot loop.

| API | Use |
| --- | --- |
| `TraceSession::open` | Start a trace file with run metadata |
| `begin_span` / `SpanGuard` | Nested function/module/op spans |
| `record_op` | Timed op + auto memory alloc from shape×dtype |
| `record_tensor` | Tensor metadata + alloc event |
| `record_memory_alloc` / `record_memory_free` | Explicit lifecycle (TF memory timeline) |
| `record_device_memory` | Device checkpoint (cudaMemGetInfo-style) |
| `record_gradient` | Step-0 (or probe) gradient audit facts |
| `finish` | Flush JSONL |

### Graph (`graph`)

| API | Use |
| --- | --- |
| `build_from_trace` | Parsed trace document → `ExecutionGraph` |
| Self-time rollup | Total wall time minus nested span/op children |

### CLI

All commands take a **trace file**, not a Rust crate path:

```bash
cargo candle-graph import TRACE.jsonl
cargo candle-graph summary TRACE.jsonl
cargo candle-graph query profile.jsonl --kind slowest|heaviest|memory|efficiency|spans|gradients
cargo candle-graph view TRACE.jsonl --output model.html   # requires visualizer
```

### Visualizer (`visualizer` feature)

Embeds `candle-graph/viewer/3` JSON into one HTML file. Shows span hierarchy sidebar, self-time or **memory heat** on nodes, **Memory** tab (timeline + summary), peak breakdown table, and **ms labels on every edge**.

## Recommended bundles

| User | Build | Typical command |
| --- | --- | --- |
| Trainer author | default or `--no-default-features` | `TraceSession` in post-run smoke binary |
| Human reviewer | `all` | `cargo candle-graph view profile.jsonl --output model.html` |
| Agent / CI | `--no-default-features` | `cargo candle-graph query profile.jsonl --kind slowest` |
