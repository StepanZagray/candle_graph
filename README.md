# candle-graph

> **Disclaimer:** `candle-graph` is an independent tool. It is **not**
> official and is **not affiliated with, endorsed by, or maintained by** [candle-rs](https://github.com/huggingface/candle)
> or [Hugging Face](https://huggingface.co).

`candle-graph` reconstructs parameter structure, tensor contracts, and gradient connectivity of a
[Candle](https://github.com/huggingface/candle) model crate. Its versioned,
crate-wide model IR and bounded query API are designed so coding agents can answer focused
questions without loading an entire source tree into context.

The analyzer combines conservative Rust-source analysis with the Cargo configuration selected for
the scan. An optional runtime trace can refine static tensor facts and audit gradients. It never
loads checkpoint tensor payloads or requires a GPU for static analysis.

- `candle-graph/model/1`: one unified IR for components, architecture edges, functions, modules,
  parameters, tensor contracts, stages, artifacts, optimizers, Cargo context, findings, and runtime
  evidence;
- `candle-graph/query/1`: deterministic, bounded agent queries with fully qualified selectors,
  compact-by-default progressive disclosure, and `limit`/`offset` pagination;
- `candle-graph/runtime/1`: JSON/JSONL tensor observations and per-parameter gradient facts;
  see [docs/runtime-analysis-guide.md](docs/runtime-analysis-guide.md) for the full workflow and
  v2 (time series) / v3 (profiler) extensions;
- `candle-graph/viewer/1`: standalone interactive HTML visualizer (requires the `visualizer` feature).

See [docs/features.md](docs/features.md) for the **`static`** / **`runtime`** / **`visualizer`**
split: static IR for agents, runtime profiler timings, HTML for humans.

## Status

Primary crate-wide UX is the Clippy-like Cargo subcommand `cargo candle-graph`
(`check` / `report` / `query` / `audit` / `view`) over the conservative analyzer. It is an explicit
cargo command, not a rustc lint pass and not a silent `cargo build` hook.

The agent-analysis path supports:

- automatic component discovery from public API boundaries and `VarBuilder`/`VarBuilderArgs`
  constructors, with `--root` for private model roots;
- public tensor-boundary and Candle-trait entrypoint candidates with fully qualified identities;
- symbolic tensor rank/shape, dtype, device, layout, and `requires_grad` contracts;
- Cargo package, target, feature, Candle-version, active `cfg`, and per-function cfg-activity facts;
- audited Candle 0.11.0 operation rules that distinguish similarly named APIs such as
  `RmsNorm::forward` and `RmsNorm::forward_diff`;
- optional runtime tensor refinement and missing/zero/non-finite gradient findings;
- checkpoint-header verification and model-IR CI baselines.

Analysis is deliberately conservative. Unsupported expressions remain `Unknown`; a same-dtype
operation with one known and one unknown operand is a `dtype-risk`, while two known differing
dtypes are a `dtype-conflict`. Call-order architecture, pipeline stages, artifacts, and optimizer
membership require `--heuristic-architecture` or the future compiler frontend. Struct-field
**composition** edges are always available via `--query composition` with `Heuristic` confidence.
In particular, names such as `run_pipeline`, `train_*`, `save`, and `all_vars`, filename suffixes,
and source order do not create facts without that flag.

The compiler-backed design needed to derive those relationships without annotations or loading
model weights is documented in
[docs/compiler-evidence-design.md](docs/compiler-evidence-design.md).

## Usage

Install once so Cargo discovers the Clippy-like subcommand (`cargo-candle-graph` →
`cargo candle-graph`):

```bash
cargo install --path . --features all --locked
```

Check a model crate the way you would run `cargo clippy` — an explicit cargo command, not a
silent side-effect of `cargo build`. Diagnostics go to stderr; the full `candle-graph/model/1`
document goes to stdout (or `--output`):

```bash
cargo candle-graph check /path/to/model
cargo candle-graph check /path/to/model --features cuda --message-format json
cargo candle-graph report /path/to/model --output model-ir.json
cargo candle-graph query summary --path /path/to/model
```

`check` / `--strict` exit non-zero on proven `Error` findings only. Coverage-gap warnings and
Information notes such as `compiler-semantic-evidence` stay visible but do not fail the gate.
Commit a crate-wide fingerprint with `--update-baseline` / `--check` on that same command.

Numeric domain: the analyzer expands audited candle-nn bodies (e.g. 0.11.0
`binary_cross_entropy_with_logit`) into the same float-range pass used for local
`sigmoid?.log()?` compositions, then labels whether the hazard can NaN a loss / poison
gradients or risk inference outputs. An `affine(mul>0, add>0)` epsilon guard discharges
`StrictlyPositive` and must not false-positive.

The `candle-graph` binary mirrors the same model-mode engine for scripts:

```bash
cargo run --release --features all -- /path/to/model --format json
```

Ask focused questions. Listings are compact (counts/IDs, `drill_down` hints, no tensor evidence);
narrow with `--select` / singular kinds when you need contracts or evidence:

```bash
cargo candle-graph query summary --path /path/to/model
cargo candle-graph query doctor --path /path/to/model
cargo candle-graph query components --path /path/to/model
cargo candle-graph query composition --path /path/to/model --format json
cargo candle-graph query assembly --select model --path /path/to/model --format json
cargo candle-graph query pipeline --path /path/to/model --format json
cargo candle-graph query entrypoints --path /path/to/model --format json
cargo candle-graph query tensors --select 'MyModel::forward' \
  --path /path/to/model --limit 20 --offset 0 --format json
cargo candle-graph query tensor --select '<tensor-stable-id>' \
  --path /path/to/model --format json
cargo candle-graph query cargo --path /path/to/model --format json
cargo candle-graph query findings --path /path/to/model --format json
cargo candle-graph query graph-train --path /path/to/model --format json
cargo candle-graph query graph-infer --path /path/to/model --format json
```

Use fully qualified selectors when bare names collide:

```bash
cargo candle-graph query function --select 'model::MyModel::forward' \
  --path /path/to/model --format json
```

Select Cargo features/target exactly as the analyzed build does:

```bash
cargo candle-graph query cargo \
  --path /path/to/model \
  --features cuda,flash-attn \
  --target x86_64-unknown-linux-gnu \
  --format json
```

The analyzer follows the selected Cargo crate root and reachable `mod` declarations. It defaults
to the library target, then the first ordinary binary; use `--cargo-target <name>` for another
binary/example/test target. The `--target` option selects a Rust target triple.

Import an instrumented small-run trace:

```bash
cargo candle-graph query runtime \
  --path /path/to/model \
  --runtime-trace run.runtime.jsonl \
  --format json
```

Open the interactive multi-view HTML visualizer (requires candle-graph built with the
`visualizer` or `all` feature):

```bash
cargo candle-graph view --path /path/to/model --output model.html
```

`--features` on this command selects Cargo features on the **model crate** (e.g. `cuda`),
not candle-graph. Names like `static`, `visualizer`, `runtime`, and `all` are candle-graph build flags
and are ignored when passed here. See [docs/features.md](docs/features.md) and [docs/visualizer.md](docs/visualizer.md).

Write a multi-file audit bundle (summary, doctor, findings, checkpoint verification, …):

```bash
cargo candle-graph audit runs/analyzer \
  --path /path/to/model \
  --checkpoint /path/to/model.safetensors \
  --runtime-trace runs/train/runtime.json \
  --verify-root vb \
  --strict
```

See [docs/agent-query-api.md](docs/agent-query-api.md) for the query map,
[docs/visualizer.md](docs/visualizer.md) for the HTML visualizer,
[docs/runtime-protocol.md](docs/runtime-protocol.md) for trace field definitions, and
[docs/runtime-analysis-guide.md](docs/runtime-analysis-guide.md) for the recommended runtime workflow
(gradient audit, phase graphs, offline profiling).

Checkpoint verification reads only the safetensors length prefix and JSON header. A template such
as `model.layers.{index}.self_attn.q_proj.weight` matches all concrete layer indices.

Create and check a canonical CI baseline:

```bash
cargo candle-graph check /path/to/model \
  --update-baseline model.candle-graph-baseline

cargo candle-graph check /path/to/model \
  --check model.candle-graph-baseline \
  --strict
```

`--strict` rejects proven dtype conflicts, numeric-domain violations, and other proven error
findings. Coverage-gap warnings and heuristic architecture notes stay visible but do not fail the
gate by themselves.

Run `cargo candle-graph --help` for all options.

## Supported source patterns

The analyzer intentionally understands a restricted, explicit Rust dialect:

- `vb.pp(...)`, `push_prefix`, `set_prefix`, `root`, and metadata-preserving builder methods;
- audited candle-nn 0.11.0 constructors and raw `vb.get*` calls;
- crate-local inherent constructors and free helper functions taking `VarBuilder`;
- multiple and optional builders, including `then`, `map`, and `and_then`;
- struct literals, `if`/`match` branches, `for` loops, and iterator `map`;
- literal and `format!` prefixes, including mixed segments such as `block_{index}`.
- ordinary tensor calls, local bindings, branches, symbolic loop iterations, and crate-local calls;
- dtype-preserving, explicit-cast, and same-dtype candle operations;
- version-gated Candle operation semantics. Rules not audited for the resolved Candle version stay
  unknown instead of silently borrowing another version's behavior.

Unsupported or ambiguous constructs produce diagnostics instead of silently inventing a result.
Unknown custom macros, non-literal tensor names, unresolved builders, cycles, and analysis limits
are therefore visible in every JSON report.

## Output semantics

Parameter keys are namespaced by their originating constructor argument (`root`). Two identical
keys under different roots are different tensors. Certainty is one of:

- `certain`: always registered on the analyzed path;
- `conditional`: gated by a branch, optional field/builder, or constructor configuration;
- `unknown`: reserved for constructs whose existence cannot be modeled safely.

Every model fact carries a stable identity and source/evidence where applicable. Function identity
is based on its fully qualified name and active `cfg` predicates rather than absolute paths or line
numbers, so ordinary source edits do not invalidate runtime correlation. Coverage counts and the
compact `doctor` query let consumers assess an analysis before trusting individual records.
`dead_params()`, `severing_edges()`,
`dtype_conflicts()`, `dtype_risks()`, and `paths_to()` remain available through the library API.

## Development

```bash
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

The integration suite covers the unified IR/query/runtime schemas, Cargo feature/cfg discovery,
qualified symbol collisions, tensor contracts, optimizer membership, versioned Candle semantics,
formatted layer families, multiple builder roots, interprocedural calls, BF16/F32 joins,
no-backward severing, deterministic baselines, the CLI, HTML safety, and safetensors headers.
