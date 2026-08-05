# HTML trace visualizer

Standalone `model.html`: embedded CSS, JavaScript, and `candle-graph/viewer/2` JSON. No CDN.

```bash
cargo candle-graph view profile.jsonl --output model.html
```

## Build / install

```bash
cargo install --path . --features visualizer --locked
```

## UI

| Area | Content |
| --- | --- |
| **Views tabs** | Call graph, timeline, gradient summary |
| **Span hierarchy** | Nested function/module/op tree with self-time |
| **Canvas** | Dagre layout; **ms label on every edge** |
| **Inspector** | Selected span/op: self time, total time, shape, dtype |

Controls: resizable panes, span search, floating zoom (+/−/fit), keyboard shortcuts, hover tooltips,
collapsible legend, Export SVG, theme toggle.

Fully offline: system fonts only, vendored dagre — no CDN or network fetches.

## Graph layout

Graph layout uses **[dagre](https://github.com/dagrejs/dagre)** vendored as `src/viewer/dagre.min.js`.
Self-time heat colors nodes (cool = mostly waiting on children, hot = self work).

## Payload schema

`candle-graph/viewer/2` — produced by [`trace_view::project`](../src/viewer/trace_view.rs) from an
[`ExecutionGraph`](../src/graph/model.rs).

Key fields:

- `summary` — entrypoint, total ms, slowest spans
- `views.call_graph` — nodes + edges with `duration_ms` on edges
- `span_tree` — hierarchical sidebar data

## Source files

| File | Role |
| --- | --- |
| `src/viewer.rs` | HTML shell + embed JSON |
| `src/viewer/trace_view.rs` | Graph → viewer/2 projection |
| `src/viewer/app_trace.js` | Trace viewer runtime |
| `src/viewer/style.css` | Shared styles |
| `src/viewer/layout.js` | Dagre wrapper |

The legacy static-analysis viewer (`app.js`, viewer/1) was removed in v0.4.
