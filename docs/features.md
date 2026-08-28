# Cargo features

`candle-graph` keeps trace and evidence processing independent of Candle and keeps HTML generation
optional.

| Feature | Default | Adds |
| --- | --- | --- |
| `visualizer` | yes | `viewer` module, `view` CLI command, and `viewer.html` in published bundles |
| `candle` | no | `candle-core` plus tensor/op capture helpers and `TraceSession::record_tensor_stats` |
| `all` | no | Alias for `visualizer` and `candle` |

The exact manifest definition is:

```toml
[features]
default = ["visualizer"]
visualizer = []
candle = ["dep:candle-core"]
all = ["visualizer", "candle"]
```

## Core without optional features

With `--no-default-features`, the crate still provides:

- JSONL trace types, parsing, and instrumentation;
- health, timing, and memory analysis;
- graph and evidence construction;
- replicated comparison;
- atomic bundle publication and verification; and
- official Nsight CSV normalization.

The binaries still compile, but the `view` subcommand is absent and `report` omits `viewer.html`.

```bash
cargo build --no-default-features
```

## Candle helpers

Enable `candle` when the instrumented application wants direct `candle_core::Tensor` helpers:

```toml
[dependencies]
candle-graph = { path = "../candle_graph", features = ["candle"] }
```

The optional `candle-core` dependency disables Candle's default features. `candle-graph` does not
select CUDA, Metal, MKL, or Accelerate for the application; configure the application's own Candle
dependencies for the intended backend.

`CandleCapture` reports backend tensor/storage identities and a dense tensor footprint. It does not
observe allocator lifetime or backing allocation size. Use explicit memory events for those facts.

## Local installation profiles

```bash
# CLI plus offline viewer.
cargo install --path . --locked

# Viewer and Candle helpers.
cargo install --path . --features all --locked
```
