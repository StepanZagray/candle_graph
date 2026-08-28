# CLI reference

The installed executables expose the same commands:

```bash
cargo candle-graph <command> ...
candle-graph <command> ...
```

JSON commands write to stdout unless `--output FILE` is supplied. `report` writes a new bundle
directory and `view` always requires an output file.

## Inputs

Three input forms appear in the CLI:

| Input | Accepted by | Behavior |
| --- | --- | --- |
| Raw `*.jsonl` trace | all trace-reading commands | Rebuilds evidence from the trace; GPU evidence is unavailable unless `view`/`report` also receives `--nsight-dir` |
| Directory containing only `trace.jsonl` | `import`, `summary`, `query` | Treated as a raw-trace directory |
| Finalized bundle directory, or its top-level `trace.jsonl` | `import`, `summary`, `query` | Deeply verifies the bundle and reads its bound `evidence.json` |

A raw-trace directory that contains `evidence.json` or `nsight/` but no `bundle.json` is rejected.
This prevents the CLI from silently discarding augmented GPU evidence and rebuilding a trace-only
packet.

`view` and `report` take a raw trace plus an optional flat Nsight directory. `compare` takes bundle
directories by default.

## Commands

### `import`

Build or load the complete evidence packet and serialize `candle-graph/evidence/4`.

```bash
cargo candle-graph import application.jsonl --output evidence.json
```

The output contains the full packet and can be as large as the trace-derived graph and timelines.

### `summary`

Emit `candle-graph/summary/4`: input provenance, health, capabilities, findings, gaps, graph
headlines, tensor-stat counts, timing, memory, and GPU report status.

```bash
cargo candle-graph summary evidence-bundle --output summary.json
```

This is a semantic summary, not a strict byte-bounded response. In particular, its timing and
memory members retain their complete derived collections.

### `query`

Emit `candle-graph/trace-query/4` for one evidence slice:

```bash
cargo candle-graph query INPUT --kind KIND [--output FILE]
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

Graph-dependent queries reject failed or structurally invalid captures. Memory, tensor statistics,
capabilities, and GPU diagnostics remain available because they can help explain a failed capture.

`slowest-host` ranks overlap-clipped self time. Every row declares either `measured_subtree` or
`concurrent_overlap`; concurrent rows retain their real parent and must not be added to measured
wall time.

### `view`

Build a standalone HTML file from a raw trace and optional Nsight directory:

```bash
cargo candle-graph view application.jsonl \
  --nsight-dir nsight \
  --output viewer.html
```

The command exists only with the default `visualizer` feature. See the
[visualizer reference](visualizer.md).

### `report`

Publish a new content-addressed directory:

```bash
cargo candle-graph report application.jsonl \
  --nsight-dir nsight \
  --bundle evidence-bundle
```

Publication stages files in a sibling temporary directory, verifies the staged bundle, syncs it,
and renames it into place. The destination must not already exist. With `visualizer` enabled, the
bundle contains:

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

Deeply verify every declared bundle file and emit `candle-graph/bundle-verification/1`:

```bash
cargo candle-graph verify evidence-bundle --output verification.json
```

The receipt includes the manifest digest, run ID, verified file count, and verified byte count.

### `compare`

Compare repeated baseline and candidate cohorts:

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

## Bundle write safety

Treat a finalized bundle as immutable. `summary`, `query`, `import`, `view`, `verify`, and
`compare` all reject output paths that resolve into an input bundle, including lexical `..`
escapes and existing symbolic-link aliases. `report` refuses to publish a new bundle inside an
existing bundle directory.

Verification is point-in-time. It detects bundle contents at the moment of each check; it is not a
filesystem snapshot and does not make mutable storage transactional.
