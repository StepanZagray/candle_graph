# Cargo features

candle-graph splits capabilities into three optional Cargo features. Only **`static`**
is on by default.

| Feature | Build | Purpose |
| --- | --- | --- |
| **`static`** | `default` | Agent-oriented **static graph**: components, modules, parameters, tensor contracts, train/infer dataflow, bounded query API, CI baselines. No GPU, no HTML. |
| **`runtime`** | `--features runtime` | **Profiler layer**: import `candle-graph/runtime/3` JSONL traces; merge timings and gradient facts into the IR; `profile` / `runtime` queries. |
| **`visualizer`** | `--features visualizer` | **Human HTML** (`model.html`): architecture, train/infer dataflow, findings — standalone file, no CDN. |

Install everything:

```bash
cargo install --path . --features all --locked
```

Agent / CI (smallest build):

```bash
cargo install --path . --features static --locked
```

## What each layer provides

### Static (`static`)

Always run this first. Output is `candle-graph/model/1` plus `candle-graph/query/1`.

- Module tree, parameters, checkpoint key matching
- Per-phase dataflow graphs (`graph-train`, `graph-infer` queries)
- Dtype / gradient / numeric-domain findings
- Deterministic pagination and fully qualified selectors

Operations have **structure and semantics only** — no wall-clock times until a runtime trace is merged.

### Runtime (`runtime`)

Requires a trace from training or inference (see [runtime-analysis-guide.md](runtime-analysis-guide.md)).

After `--runtime-trace path.jsonl`:

- Each matched operation gets [`TimingStats`](../src/model_ir.rs): **samples**, **avg_ns**, **min_ns**, **max_ns**
- Edge timings between static tensor/operation ids roll up the same way
- Gradient audit (`present` / `missing` / `zero` / `non_finite`)
- Optional tensor observation refinement (shape, dtype, device)

CLI:

```bash
cargo candle-graph profile /path/to/model --runtime-trace run.jsonl --output profile.json
cargo candle-graph query profile --path /path/to/model --runtime-trace run.jsonl
```

Library profiling API: [`ProfileSession`](../../src/profile.rs) writes v3 JSONL during a run.

### Visualizer (`visualizer`)

Embeds the static (and runtime-enriched, if merged) IR into one HTML file:

```bash
cargo candle-graph view /path/to/model --output model.html
cargo candle-graph view /path/to/model --runtime-trace run.jsonl --output model.html
```

When timings are present, operation nodes show **avg · min–max** in the graph and inspector.

## `--features` on analyze commands

Flags like `cuda` or `cudnn` select **Cargo features on the model crate** being analyzed.

Names `static`, `runtime`, `visualizer`, and `all` refer to **candle-graph itself** and are
stripped before `cargo metadata` — they do not get forwarded to your model crate. Build
candle-graph with the feature you need instead.

## Feature matrix

| Capability | static | + runtime | + visualizer |
| --- | --- | --- | --- |
| `cargo candle-graph check/report/query` | yes | yes | yes |
| `--runtime-trace` / `profile` subcommand | — | yes | yes (if built with runtime) |
| Operation timing stats in IR | — | yes | yes (when trace merged) |
| `cargo candle-graph view` / HTML output | — | — | yes |
| [`ProfileSession`](../../src/profile.rs) | — | yes | — |
