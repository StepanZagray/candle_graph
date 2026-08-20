# Unified evidence visualizer

`viewer.html` embeds `candle-graph/viewer/5`, CSS, JavaScript, and dagre in one offline document.

```bash
cargo candle-graph view application.jsonl --nsight-dir nsight --output viewer.html
```

| View | Purpose |
| --- | --- |
| Evidence | Structural outcome, capability matrix, provenance, typed facts, qualified findings, and gaps |
| Trace | Semantic call/data graph with explicit tensor nodes and host durations |
| Span costs | Host-cost ranking plus separate device-clock interval unions |
| Memory | Logical storage lifetimes and independent physical device observations |
| GPU | Hashed Nsight artifacts, manifest binding, correlation ledger, phase attribution, and global summaries |

GPU global summaries are labelled as global. Per-phase attribution is only claimed when projected
ranges have exact semantic keys, join to GPU rows through Nsight identifiers, and their declared
operation counts match. Provenance or correlation failures remain visibly diagnostic even when
normalized rows can still be inspected. The UI states that Candle and Nsight clocks are separate.

Tabs implement ARIA tab semantics and keyboard navigation. Trace-only controls are hidden outside
the graph view, unavailable GPU evidence has an explanatory text state, and wide evidence tables
scroll on narrow displays.

Source locality:

| File | Role |
| --- | --- |
| `src/viewer.rs` | Standalone HTML shell |
| `src/viewer/trace_view.rs` | EvidencePacket → viewer/5 projection |
| `src/viewer/app_trace.js` | Accessible view interactions |
| `src/viewer/style.css` | Responsive visual system |
| `src/viewer/layout.js` | Dagre trace-only layout |
