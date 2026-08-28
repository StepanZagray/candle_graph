# CLI reference

The installed executables expose the same commands through one shared parser:

```bash
cargo candle-graph <command> ...
candle-graph <command> ...
```

`--version` prints the crate version. JSON commands write to stdout unless `--output FILE` is
supplied. `report` writes a new bundle directory and `view` always requires an output file.

Agents should call `protocol` first to bind to schema versions, then `overview` for a bounded
first look at any input.

## Inputs

Three input forms appear in the CLI:

| Input | Accepted by | Behavior |
| --- | --- | --- |
| Raw `*.jsonl` trace | all trace-reading commands | Rebuilds evidence from the trace; GPU evidence is unavailable unless `view`/`report` also receives `--nsight-dir` |
| Directory containing only `trace.jsonl` | `import`, `overview`, `summary`, `query` | Treated as a raw-trace directory |
| Finalized bundle directory, or its top-level `trace.jsonl` | `import`, `overview`, `summary`, `query`, `view` | Deeply verifies the bundle and reads its bound `evidence.json` |

A raw-trace directory that contains `evidence.json` or `nsight/` but no `bundle.json` is rejected.
This prevents the CLI from silently discarding augmented GPU evidence and rebuilding a trace-only
packet.

`report` takes a raw trace plus an optional flat Nsight directory. `view` takes either a raw trace
(with optional `--nsight-dir`) or a finalized bundle (which rejects `--nsight-dir`). `compare`
takes bundle directories by default. `campaign-status` and `series` read campaign manifests and
published bundles.

## Commands

### `protocol`

Emit `candle-graph/protocol/1`: the tool identity (`package`, `version`), a `schemas` map, and the
command list.

```bash
cargo candle-graph protocol --output protocol.json
```

The `schemas` map declares `trace_write` (trace/10), `trace_read` ([trace/9, trace/10]), graph/5,
evidence/4, comparison/6, bundle/1, bundle-verification/1, publication/1, campaign/1,
campaign-status/1, series/1, summary/5, trace-query/5, overview/1, gradient-manifest/1,
nsight-capture/1, and viewer/5. Agents should call this first to bind to versions instead of
inferring them from the package version.

### `overview`

Emit `candle-graph/overview/1`: a hard-bounded first look at any input. This is the recommended
first command for agents; it replaces `summary` in that role.

```bash
cargo candle-graph overview evidence-bundle --output overview.json
```

The output holds provenance, health, capabilities, bounded findings and gaps, collection counts,
and timing/memory/GPU headlines. It never emits unbounded collections; every truncated list
carries `total`/`displayed`/`truncated` so agents can decide which focused `query` to run next.

### `import`

Build or load the complete evidence packet and serialize `candle-graph/evidence/4`.

```bash
cargo candle-graph import application.jsonl --output evidence.json
```

The output contains the full packet and can be as large as the trace-derived graph and timelines.

### `summary`

Emit `candle-graph/summary/5`: input provenance, health, capabilities, findings, gaps, graph
headlines, tensor-stat counts, timing, memory, and GPU report status, plus a `tool` identity.

```bash
cargo candle-graph summary evidence-bundle --output summary.json
```

This is a semantic summary, not a strict byte-bounded response. In particular, its timing and
memory members retain their complete derived collections; use `overview` for a bounded response.

`--require-valid` exits non-zero — after writing the output — when the capture is not structurally
valid and complete, so scripted pipelines can gate on capture health.

### `query`

Emit `candle-graph/trace-query/5` (adds `tool` and `filter` envelope fields) for one evidence
slice:

```bash
cargo candle-graph query INPUT --kind KIND [--label S | --label-prefix S] [--output FILE]
```

| Kind | Result | Requires a complete graph? | Size |
| --- | --- | --- | --- |
| `slowest-host` | Measured-region outer wall time and ranked measured/concurrent host spans | yes | capped by graph summary |
| `slowest-device` | Per-span device busy-time rankings | yes | capped by graph summary |
| `heaviest` | Memory-heavy spans, plus operations whose recorded op name uniquely attributes logical allocations | yes | capped |
| `memory` | Logical and physical memory profiles | no | full collection |
| `spans` | Graph span/op/tensor nodes and edges | yes | full collection |
| `tensors` | Recorded tensor metadata | yes | full collection |
| `tensor-stats` | Ordered caller-labelled numerical summaries | no | full collection |
| `gradients` | Recorded gradient facts | yes | full collection |
| `capabilities` | Capability matrix | no | fixed fields |
| `gpu-status` | Per-report availability and bounded diagnostics | no | bounded diagnostics |
| `gpu-correlation` | Semantic-label correlation ledger | no | label lists capped in query output |
| `gpu-phases` | Projected phase rows and exact joined attribution | no | rows capped at 50 |
| `gpu-kernels` | Kernel summary rows | no | rows capped at 50 |
| `gpu-attribution-gaps` | Missing, unexpected, CPU-only, duplicate, and unattributed labels | no | label lists capped |

`--label` (exact match) and `--label-prefix` narrow the returned rows for the collection kinds:
`spans` filters on the span name, `tensors` on the observation label or tensor ID, `tensor-stats`
on the label, and `gradients` on the key or the `root/key` composite. Other kinds reject the
flags. Filters only narrow rows; unfiltered collection queries remain complete exports.

Graph-dependent queries reject failed or structurally invalid captures. Memory, tensor statistics,
capabilities, and GPU diagnostics remain available because they can help explain a failed capture.

`slowest-host` ranks overlap-clipped self time. Every row declares either `measured_subtree` or
`concurrent_overlap`; concurrent rows retain their real parent and must not be added to measured
wall time.

### `view`

Build a standalone HTML file from a raw trace or a finalized bundle:

```bash
cargo candle-graph view application.jsonl \
  --nsight-dir nsight \
  --output viewer.html
```

A finalized bundle input is loaded through the verified-bundle loader and rejects `--nsight-dir`
(the bundle already binds its Nsight inputs). Raw traces keep the existing unverified path.

The command exists only with the default `visualizer` feature. See the
[visualizer reference](visualizer.md).

### `report`

Publish a new content-addressed directory and emit a `candle-graph/publication/1` receipt:

```bash
cargo candle-graph report application.jsonl \
  --nsight-dir nsight \
  --bundle evidence-bundle
```

Publication stages files in a sibling temporary directory, verifies the staged bundle, syncs it,
and renames it into place. The destination must not already exist. The receipt carries the
publication status, bundle path, run ID, and the deep-verification receipt. With `visualizer`
enabled, the bundle contains:

```text
evidence-bundle/
├── bundle.json
├── trace.jsonl
├── evidence.json
├── report.md
├── viewer.html
└── nsight/          # only when supplied
```

The Nsight input directory must be flat and contain only regular files. Directories, symbolic
links, special files, and enumeration failures reject publication.

### `verify`

Deeply verify a published bundle and emit a `candle-graph/verify/1` envelope
`{schema, tool, receipt, semantic}`:

```bash
cargo candle-graph verify evidence-bundle --semantic --output verification.json
```

Content verification checks every declared file's bytes and hashes; the embedded
`candle-graph/bundle-verification/1` receipt includes the manifest digest, run ID, verified file
count, and verified byte count. `--semantic` additionally rederives the evidence packet from the
retained trace (and the retained Nsight inputs when present) and fails closed if it differs from
the published `evidence.json`.

### `compare`

Compare repeated baseline and candidate cohorts and emit `candle-graph/comparison/6`:

```bash
cargo candle-graph compare \
  --baseline base-1.bundle base-2.bundle base-3.bundle base-4.bundle base-5.bundle \
  --candidate next-1.bundle next-2.bundle next-3.bundle next-4.bundle next-5.bundle \
  --output comparison.json
```

An eligible timing result requires:

- at least five unique run IDs per cohort and no duplicate run ID across cohorts;
- complete, structurally valid, production-equivalent captures with in-domain provenance
  (non-empty identifiers, a one-based capture step exceeding the warmup count, and non-zero
  batch/accumulation identity values);
- a synchronized measured region for a non-CPU device;
- identical entrypoint, phase, device, timing mode, synchronization flag, warmup count, selected
  invocation, and capture contract;
- matching workload/model/config/data/seed/batch/precision/device-state identity;
- one non-empty implementation ID within each cohort; and
- deeply verified bundle inputs.

Ineligibility causes are typed: `reasons` is a list of `{code, message}` objects whose `code`
values are stable snake_case identifiers (for example `insufficient_runs`, `unverified_inputs`,
`identity_conditions_differ`), so agents match on codes rather than message text.

`--require-eligible` exits non-zero — after writing the output — when the verdict is `ineligible`,
naming the distinct reason codes.

If every run has a `pair_id`, the pair-ID sets must match exactly and the interval is computed from
paired deltas. Without pair IDs the cohorts are bootstrapped independently. A 95% interval wholly
below zero means the candidate is faster; one wholly above zero means it is slower; an interval
crossing zero is inconclusive.

The result also includes a numerical mechanism comparison, even when timing is ineligible. For each
label, every tensor-stat event is averaged (duplicate labels within a run included) and non-finite
counts are summed across all events. Each matched row reports `samples_a`/`samples_b` (events
averaged) and `runs_a`/`runs_b` (cohort runs containing the label), alongside the cohort totals
`baseline_runs`/`candidate_runs`, so partial cohort coverage is visible directly in the output.

`--unverified-traces` accepts raw JSONL paths but always returns an ineligible diagnostic result
without a confidence interval.

### `campaign-status`

Reconcile a producer-declared campaign plan against published bundles and emit
`candle-graph/campaign-status/1`:

```bash
cargo candle-graph campaign-status --manifest campaign.json --output status.json
```

The manifest is a `candle-graph/campaign/1` document: `campaign_id`, `entrypoint`, and `planned`
entries pairing a `capture_step` with a bundle path relative to the manifest's directory. Each
planned capture reports one state — `missing`, `published`, `failed_run`, `verification_failed`,
or `identity_mismatch` — plus overall counts. The output is deterministic and sorted by capture
step. The command is stateless: it reads artifacts and never supervises training.

### `series`

Assemble metric trajectories across verified bundles of one entrypoint and emit
`candle-graph/series/1`:

```bash
cargo candle-graph series --manifest campaign.json --output series.json
cargo candle-graph series --bundle step-100 step-200 step-300 \
  --label-prefix loss/ --output series.json
```

Every input bundle passes deep content verification and all traces must share one entrypoint and
phase; inputs are sorted by `capture_step`, which is the series coordinate. The report holds the
verified inputs, an outer-wall-time series, per-label tensor-stat/scalar series, and per-`root/key`
gradient-norm series. `--label-prefix` filters tensor-stat labels and gradient `root/key`
composites.

Manifest mode fails closed if any planned capture is not published: run `campaign-status` first,
or pass explicit `--bundle` paths for the published subset.

## Agent exit-code behavior

Commands exit zero when they produce their declared output, even when that output reports gaps or
an ineligible verdict — the JSON is the result. The strict flags convert specific facts into
non-zero exits after the output is written: `summary --require-valid` (capture not structurally
valid and complete) and `compare --require-eligible` (verdict `ineligible`, naming the distinct
reason codes). `series` in manifest mode and `verify --semantic` fail closed with a non-zero exit
instead of emitting a weakened result.

## Bundle write safety

Treat a finalized bundle as immutable. `overview`, `summary`, `query`, `import`, `view`, `verify`,
and `compare` all reject output paths that resolve into an input bundle, including lexical `..`
escapes and existing symbolic-link aliases. `report` refuses to publish a new bundle inside an
existing bundle directory.

Verification is point-in-time. It detects bundle contents at the moment of each check; it is not a
filesystem snapshot and does not make mutable storage transactional.
