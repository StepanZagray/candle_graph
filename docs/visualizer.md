# HTML visualizer

The `visualizer` feature emits a standalone `model.html` file: one document with embedded
CSS, JavaScript, and the `candle-graph/viewer/1` JSON payload. No CDN or network access
required.

## Build / install

Enable the feature when building or installing candle-graph:

```bash
cargo install --path . --features visualizer --locked
# or
cargo install --path . --features all --locked
```

Open the visualizer:

```bash
cargo candle-graph view --path /path/to/model --output model.html
```

### `--features` on the command line

`--features` on `cargo candle-graph …` selects **Cargo features on the model crate being
analyzed** (e.g. `cuda`, `cudnn`), not candle-graph itself.

| Name | Meaning |
|------|---------|
| `static`, `visualizer`, `runtime`, `all` | candle-graph build flags — **ignored** when passed to analyze commands |

Build candle-graph with `visualizer` or `all` instead. For Tofy, the `.cargo/config.toml`
alias already passes `--features all` when compiling the subcommand.

## UI

| Tab | Content |
|-----|---------|
| **Architecture** | Module tree (sidebar) + L→R hierarchy graph with layer bands |
| **Train / Infer** | Function-scoped dataflow — pick a function on large models |
| **Pipeline** | Stage dependency graph (top-to-bottom) |
| **Findings** | Click to jump to the suggested view and highlight nodes |

Graph nodes render as readable cards (wrapped labels, kind badges, color-coded borders)
on a dot-grid canvas. Layer bands alternate by dagre rank for easier left-to-right scanning.

Controls: resizable panes, node search (centers match), floating zoom (+/−/fit), keyboard
(`+`/`−` zoom, `F` fit, `0` reset), hover tooltips, collapsible legend, Export SVG,
Preview (print mode), Fit / Reset zoom.

Fully offline: system fonts only, vendored dagre — no CDN or network fetches.

## Graph layout

Graph layout uses **[dagre](https://github.com/dagrejs/dagre)** (`@dagrejs/dagre` 3.1),
vendored as `src/viewer/dagre.min.js` (~48 KB). Dagre implements the Sugiyama layered DAG
pipeline: rank assignment (network simplex), crossing minimization, coordinate assignment,
and orthogonal edge routing with precomputed `points` polylines.

| View | Dagre config |
|------|----------------|
| Architecture | `rankdir: LR`, network-simplex ranker |
| Train / Infer | `rankdir: LR` |
| Pipeline | `rankdir: TB` |

Cycles are broken with dagre's greedy acyclic adjustment before layout.

### Agent payload

Embedded JSON includes `summary`, `agent_context`, and per-node `short_label` /
`qualified_name`. Agents can read `<script id="cg-payload">` or use the query API.

## Third-party

| File | License |
|------|---------|
| `src/viewer/dagre.min.js` | MIT (@dagrejs/dagre) |
