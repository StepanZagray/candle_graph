# Offline evidence visualizer

The default `visualizer` feature renders an `EvidencePacket` as one self-contained HTML document.
The file embeds `candle-graph/viewer/5` JSON, CSS, JavaScript, and dagre; it makes no CDN or network
requests.

```bash
cargo candle-graph view application.jsonl --output viewer.html

# Include normalized GPU evidence while building the page.
cargo candle-graph view application.jsonl \
  --nsight-dir nsight \
  --output viewer.html
```

A bundle published with the feature enabled includes the same viewer as `viewer.html`.

## Views

| View | Contents |
| --- | --- |
| Evidence | Provenance, trace health, capabilities, findings, gaps, tensor metadata, and gradients |
| Trace | Semantic call/data graph, tensor nodes, host timings, memory heat, search, and SVG export |
| Span costs | All graph nodes ranked by host total time, with device timing and memory columns |
| Memory | Logical live-storage timeline, peak breakdown, and independent physical observations |
| GPU | Nsight artifact provenance, report tables, semantic correlation, phase attribution, and diagnostics |

A failed or structurally invalid capture still opens on the Evidence view, but graph-dependent
views have empty states because no `ExecutionGraph` was derived. Missing GPU evidence has an
explicit unavailable state rather than an empty-success table.

GPU summary tables are global. Phase GPU attribution is exact only when a projected NVTX row has
usable join identifiers, matches GPU timeline rows, and its declared operation count equals the
joined count. Projection-only rows remain a separate, qualified view. Candle, device-event, and
Nsight clocks are never drawn as one aligned clock.

## Interaction and accessibility

- The view switcher implements ARIA tab semantics and arrow-key navigation.
- The trace hierarchy is keyboard navigable and searchable.
- Graph nodes expose keyboard focus and accessible labels.
- Evidence tables scroll horizontally on narrow screens.
- Trace-only controls are hidden in other views.
- The document includes a skip link, light/dark theme toggle, and textual unavailable/error states.

## Source map

| File | Responsibility |
| --- | --- |
| `src/viewer.rs` | HTML shell, embedded assets, and script-safe JSON escaping |
| `src/viewer/trace_view.rs` | `EvidencePacket` to viewer/5 projection |
| `src/viewer/app_trace.js` | View rendering and interactions |
| `src/viewer/style.css` | Responsive visual system |
| `src/viewer/layout.js` | Graph layout adapter |
| `src/viewer/dagre.min.js` | Embedded dagre dependency |

The visualizer is vanilla HTML/CSS/JavaScript. Changes should preserve the standalone, offline
artifact rather than introducing a build-time frontend framework.
