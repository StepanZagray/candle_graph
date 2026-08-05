# candle-graph

> Independent tool — **not** affiliated with [candle-rs](https://github.com/huggingface/candle) or Hugging Face.

TensorFlow Profiler-style execution graphs for Candle: record a **post-run JSONL trace**, build an
execution graph with **milliseconds on every edge**, and inspect it via CLI or HTML.

Read [`CONTEXT.md`](CONTEXT.md) for the product model.

## What it does

There is **no static Rust analysis**. The graph comes only from what actually executed in one
representative forward/loss/eval pass.

| Piece | Delivers |
| --- | --- |
| **`instrument::TraceSession`** | Emit `candle-graph/trace/4` JSONL from a probe binary |
| **`graph::build_from_trace`** | Hierarchical span tree + self-time (TF profiler style) |
| **CLI** | `import`, `summary`, `query`, `view` on trace files |
| **`visualizer`** | Standalone `model.html` with ms labels on call/data edges |

Training hot loops stay clean: instrument **after** training or in a dedicated smoke binary, not
inside every step.

## Install

```bash
cargo install --path . --features all --locked
```

## Quick start

### 1. Emit a trace from your probe binary

```rust
use candle_graph::{ExecutionPhase, OpRecord, SpanKind, TraceSession};

let session = TraceSession::open("profile.jsonl", "my_crate::train::loss", ExecutionPhase::Train)?;

{
    let _forward = session.begin_span("Model::forward", SpanKind::Function);
    // ... one representative forward ...
}

session.record_gradient("vb", "encoder.weight", candle_graph::trace::schema::GradientState::Present, Some(0.42))?;
session.finish()?;
```

Use [`SpanGuard`](src/instrument/span.rs) (RAII) for nested spans; call [`record_op`](src/instrument/session.rs)
for timed ops inside a span.

### 2. CLI

```bash
cargo candle-graph summary profile.jsonl
cargo candle-graph import profile.jsonl --output graph.json
cargo candle-graph query profile.jsonl --kind slowest
cargo candle-graph view profile.jsonl --output model.html
```

### 3. Library

```rust
use candle_graph::{build_from_trace, parse_trace};

let doc = parse_trace("profile.jsonl")?;
// build_from_trace accepts the graph event stream; see trace_cli for the adapter
```

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/4` | JSONL trace (meta, spans, ops, gradients, edges) |
| `candle-graph/graph/1` | Execution graph JSON |
| `candle-graph/viewer/2` | Embedded HTML payload |
| `candle-graph/trace-query/1` | Bounded CLI query output |

## Docs

| Doc | Contents |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | Product goals, trace-only model |
| [docs/runtime-analysis-guide.md](docs/runtime-analysis-guide.md) | Probe + trace workflow |
| [docs/features.md](docs/features.md) | Cargo features |
| [docs/visualizer.md](docs/visualizer.md) | HTML trace viewer |

## Development

```bash
cargo fmt -- --check
cargo test --features all
cargo clippy --features all -- -D warnings
```
