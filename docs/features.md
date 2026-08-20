# Cargo features

| Feature | Default | Purpose |
| --- | --- | --- |
| `visualizer` | yes | Standalone viewer/5 HTML |
| `candle` | no | Candle tensor/storage capture helpers |
| `all` | no | `visualizer` plus `candle` |

```toml
[features]
default = ["visualizer"]
visualizer = []
candle = ["dep:candle-core"]
all = ["visualizer", "candle"]
```

Core trace parsing, health analysis, graph building, replicated comparisons, evidence
JSON/Markdown, atomic bundles, and official Nsight CSV normalization are always available. HTML
rendering requires `visualizer`.

```bash
cargo install --path . --features all --locked
cargo build --no-default-features
```

`record_op` and `record_tensor` report backend tensor identities, optional semantic labels, and
dense tensor footprints, not backing allocation sizes. Logical storage lifetimes, physical
device checkpoints, and device timing intervals remain separate interfaces; unknown values remain
absent rather than becoming zero.
