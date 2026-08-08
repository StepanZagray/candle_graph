# Cargo features

| Feature | Default | Purpose |
| --- | --- | --- |
| `visualizer` | yes | Standalone viewer/4 HTML |
| `candle` | no | Candle tensor/storage capture helpers |
| `all` | no | `visualizer` plus `candle` |

```toml
[features]
default = ["visualizer"]
visualizer = []
candle = ["dep:candle-core"]
all = ["visualizer", "candle"]
```

Core trace parsing, health analysis, graph building, comparisons, evidence JSON/Markdown, and
official Nsight CSV normalization are always available. HTML rendering requires `visualizer`.

```bash
cargo install --path . --features all --locked
cargo build --no-default-features
```

`record_op` and `record_tensor` report metadata/footprint. Explicit memory lifecycle and device
memory checkpoints remain separate interfaces so peak memory is never inferred from metadata.
