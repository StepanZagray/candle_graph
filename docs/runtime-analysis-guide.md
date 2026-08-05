# Runtime trace guide

candle-graph is **trace-only**. There is no static Rust analysis. Follow this workflow to produce
trustworthy execution graphs.

## When to trace

| Moment | Overhead | What to emit |
| --- | --- | --- |
| **Training hot loop** | None | Nothing (optional: step-0 gradient audit only) |
| **After training / CI smoke** | One extra forward+backward+optim pass | Full span + op + memory trace |
| **Benchmark binary** | Acceptable | Timed ops + nested spans |

Never wrap every optimizer step in spans.

## Minimal probe

```rust
use candle_graph::{
    ExecutionPhase, ExecutionStep, OpRecord, SpanKind, TraceSession,
};
use candle_graph::trace::schema::GradientState;

fn profile_one_step() -> anyhow::Result<()> {
    let session = TraceSession::open(
        "profile.jsonl",
        "my_crate::train::training_loss",
        ExecutionPhase::Train,
    )?;

    {
        let _loss = session.begin_span("training_loss", SpanKind::Function);
        {
            let _fwd = session.begin_step_span("forward", ExecutionStep::Forward, SpanKind::Function);
            // forward pass + record_op per op
        }
        {
            let _bwd = session.begin_step_span("backward", ExecutionStep::Backward, SpanKind::Function);
            // backward pass
        }
        {
            let _opt = session.begin_step_span("optimizer", ExecutionStep::Optimizer, SpanKind::Function);
            // optimizer.step + zero_grad
        }
    }

    session.record_gradient("vb", "encoder.weight", GradientState::Present, Some(0.42))?;
    session.finish()?;
    Ok(())
}
```

## Candle helpers (`candle` feature)

With `candle-graph = { features = ["candle"] }`:

```rust
use candle_graph::candle::{inputs_storage_bytes, record_op, CandleOpCapture};
use candle_graph::{ExecutionStep, TraceSession};

let cap = CandleOpCapture::new(
    "matmul",
    vec![candle_graph::candle::tensor_id(&a), candle_graph::candle::tensor_id(&b)],
    &y,
    inputs_storage_bytes(&[&a, &b]),
    duration_ns,
    0,
    Some(ExecutionStep::Forward),
);
record_op(&session, span.id, &cap)?;
```

This mirrors PyTorch `record_shapes=True` + `profile_memory=True` without pulling Candle into the default build.

## Nested spans

```rust
let _model = session.begin_step_span("Model::forward", ExecutionStep::Forward, SpanKind::Function);
{
    let matmul = session.begin_span("matmul", SpanKind::Op);
    session.record_op(
        matmul.id,
        OpRecord {
            op_name: "matmul",
            inputs: &["x".into(), "w".into()],
            output: Some("y"),
            shape: &[32, 768],
            dtype: "f32",
            device: "cuda:0",
            duration_ns: 1_200_000,
            timestamp_ns: 0,
            storage_bytes: None,
            input_storage_bytes: 32 * 768 * 4 + 768 * 768 * 4,
            category: None, // inferred from forward step
        },
    )?;
}
```

## Inspect the trace

```bash
cargo candle-graph summary profile.jsonl
cargo candle-graph query profile.jsonl --kind slowest
cargo candle-graph query profile.jsonl --kind heaviest
cargo candle-graph query profile.jsonl --kind memory
cargo candle-graph query profile.jsonl --kind efficiency
cargo candle-graph view profile.jsonl --output model.html
```

## Trace schema (`candle-graph/trace/5`)

JSONL events (one JSON object per line):

| Event | Purpose |
| --- | --- |
| `meta` | Schema, entrypoint, phase, timestamp |
| `span_start` / `span_end` | Nested hierarchy + optional `step` (`forward`/`backward`/`optimizer`) |
| `op` | Timed op with shape/dtype, output + input bytes |
| `memory` | Explicit alloc/free (PyTorch memory timeline) |
| `device_memory` | Device checkpoint (`used` / `free` / optional `reserved`) |
| `tensor` | Tensor snapshot with category |
| `gradient` | Parameter gradient fact |
| `edge` | Call or data edge timing between spans |

Parse in Rust with `parse_trace(path)` or build a graph with the CLI `import` command.

## Tofy integration

Tofy should:

1. Run one post-training probe with forward **and** backward in `TraceSession`.
2. Use `begin_step_span` for forward/backward/optimizer slices.
3. Write `profile.jsonl` next to checkpoints.
4. Open `cargo candle-graph view profile.jsonl --output model.html` for human review.

## Migration from v0.3

v0.4 removed static Rust analysis and the old `static` / `runtime` Cargo features. Use trace files and the commands above instead.
