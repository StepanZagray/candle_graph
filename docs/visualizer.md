# Unified evidence visualizer

`viewer.html` embeds `candle-graph/viewer/4`, CSS, JavaScript, and dagre in one offline document.

```bash
cargo candle-graph view application.jsonl \
  --baseline baseline.jsonl --nsight-dir nsight --output viewer.html
```

| View | Purpose |
| --- | --- |
| Evidence | Trust status, provenance, coverage, structured facts, tensors, gradients, findings, gaps, baseline deltas |
| Trace | Semantic call/data graph with self/total time and edge durations |
| Span costs | Duration-sorted semantic spans with raw nanoseconds in the embedded packet |
| Memory | Explicit allocation lifecycle and device-memory observations |
| GPU | Nsight status, coverage, parser diagnostics, truncation limits, exact NVTX join status, projected ranges, and kernel/API/memory summaries |

GPU global summaries are labelled as global. Per-phase correlation is only claimed when
`nvtx_gpu_proj_trace` supplies projected ranges with exact semantic keys. The UI states that the
Candle and Nsight clocks are separate.

Tabs implement ARIA tab semantics and keyboard navigation. Trace-only controls are hidden outside
the graph view, unavailable GPU evidence has an explanatory text state, and wide evidence tables
scroll on narrow displays.

Source locality:

| File | Role |
| --- | --- |
| `src/viewer.rs` | Standalone HTML shell |
| `src/viewer/trace_view.rs` | EvidencePacket → viewer/4 projection |
| `src/viewer/app_trace.js` | Accessible view interactions |
| `src/viewer/style.css` | Responsive visual system |
| `src/viewer/layout.js` | Dagre trace-only layout |
