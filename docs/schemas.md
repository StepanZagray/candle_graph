# Schemas and compatibility

Wire schema versions are independent: a trace revision, graph revision, and viewer revision do not
advance as one lockstep version. Consumers should check the `schema` field rather than infer a wire
format from the crate package version.

## Current formats

| Schema | Producer | Purpose |
| --- | --- | --- |
| `candle-graph/trace/10` | `TraceSession`, `write_jsonl` | Ordered execution-evidence JSONL; adds caller-labelled tensor statistics to trace/9 |
| `candle-graph/graph/5` | `build_from_trace` | Validated call/data graph with measured-scope and concurrent-overlap host timing |
| `candle-graph/evidence/4` | `build_evidence`, `import`, `report` | Provenance, health, capabilities, facts, gaps, optional graph, timing, memory, and GPU evidence |
| `candle-graph/protocol/1` | `protocol` | Tool identity, schema-version map, and command list |
| `candle-graph/overview/1` | `overview` | Hard-bounded first-look envelope; truncated lists carry total/displayed/truncated |
| `candle-graph/summary/5` | `summary` | CLI overview envelope; adds `tool` to summary/4 |
| `candle-graph/trace-query/5` | `query` | Typed CLI query envelope; adds `tool` and `filter` to trace-query/4 |
| `candle-graph/comparison/6` | `compare` | Repeated-run comparison; replaces free-text reasons with typed `{code, message}` objects (breaking) |
| `candle-graph/viewer/5` | `render_evidence_html` | JSON payload embedded in the offline viewer |
| `candle-graph/gradient-manifest/1` | `GradientContract` | Digest domain for ordered `(root, key, family)` entries |
| `candle-graph/nsight-capture/1` | application/tooling | Nsight artifact hashes, tool/hardware facts, IDs, and semantic-label contract |
| `candle-graph/bundle/1` | `report`, `publish_bundle` | Content manifest for one evidence directory |
| `candle-graph/bundle-verification/1` | `verify_bundle` | Point-in-time deep-verification receipt |
| `candle-graph/verify/1` | `verify` | CLI verification envelope: `{schema, tool, receipt, semantic}` |
| `candle-graph/publication/1` | `report`, `CaptureRun` | Typed publication receipt: status, bundle path, run ID, verification receipt |
| `candle-graph/campaign/1` | application/tooling | Producer-declared campaign plan of capture steps and bundle paths |
| `candle-graph/campaign-status/1` | `campaign-status` | Deterministic reconciliation of a campaign plan against published bundles |
| `candle-graph/series/1` | `series`, `build_series` | Metric trajectories across verified bundles, keyed by capture step |

The root library exports `TRACE_SCHEMA`, `GRAPH_SCHEMA`, `EVIDENCE_SCHEMA`, `COMPARISON_SCHEMA`,
`BUNDLE_SCHEMA`, `GRADIENT_MANIFEST_SCHEMA`, `PUBLICATION_SCHEMA`, `CAMPAIGN_SCHEMA`,
`CAMPAIGN_STATUS_SCHEMA`, and `SERIES_SCHEMA`. The viewer constant is available as
`candle_graph::viewer::trace_view::SCHEMA`; the bundle verification constant is
`candle_graph::artifact::VERIFICATION_SCHEMA`.

## Trace stream shape

A trace is newline-delimited JSON. Its first non-empty line is one `meta` event and its final line
is one `terminal` event. The body may contain:

- `span_start` / `span_end`;
- `op` and `tensor` metadata;
- `tensor_stats` numerical summaries;
- logical `memory` events and physical `device_memory` samples;
- `device_interval` timing;
- `gradient` facts; and
- typed call or tensor-data `edge` events.

`TraceDocument::from_events` validates stream ordering and basic event invariants. Structural
health analysis then validates the hierarchy, references, interval placement, memory pairing,
semantic labels, and exact gradient contract. A graph is built only for a successful, structurally
valid capture.

## Read compatibility

The trace parser accepts `trace/10` and the immediately previous `trace/9`. New sessions emit
`trace/10`. Trace/9 does not define `tensor_stats`; the parser rejects a document labelled
trace/9 that carries `tensor_stats` events, so producers emitting tensor statistics must declare
trace/10.

Evidence packets are stricter: deserialization and `validate_schema` require exactly
`evidence/4`, and an embedded graph must be exactly `graph/5`. Bundle and comparison consumers
likewise require their current schema revisions. Regenerate derived artifacts from a readable raw
trace instead of editing schema strings in place.

## Gradient manifest digest

The digest begins with the UTF-8 bytes of `candle-graph/gradient-manifest/1`, followed by one NUL
byte. It then hashes each caller-ordered manifest entry and, within that entry, `root`, `key`, and
`family`. Every string is prefixed by its byte length as a little-endian `u64`.

Ordering is part of the contract. Sort parameter keys (or otherwise define a stable application
order) before constructing `GradientContract` and reuse that same policy across runs.
