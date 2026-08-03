# candle-graph

`candle-graph` reconstructs parameter structure, tensor contracts, and gradient connectivity of a
[Candle](https://github.com/huggingface/candle) model crate. Its versioned,
crate-wide model IR and bounded query API are designed so coding agents can answer focused
questions without loading an entire source tree into context.

The analyzer combines conservative Rust-source analysis with the Cargo configuration selected for
the scan. With the optional `runtime` crate feature, a runtime trace can refine static tensor facts
and audit gradients during training and inference. It never loads checkpoint tensor payloads or
requires a GPU for static analysis.

## Features

- **Default (static)**: compile-time structure, dataflow, baselines, verification, and agent queries.
- **`runtime`**: train/inference phase graphs, runtime trace import (`--runtime-trace`), gradient
  audit, and the `profile` profiler / `cargo candle-graph profile` command.

Enable runtime support when depending on this crate:

```toml
candle-graph = { version = "0.2.5", features = ["runtime"] }
```

Or when installing the CLI:

```bash
cargo install candle-graph --features runtime
```

- `candle-graph/model/1`: one unified IR for components, architecture edges, functions, modules,
  parameters, tensor contracts, stages, artifacts, optimizers, Cargo context, findings, and runtime
  evidence;
- `candle-graph/query/1`: deterministic, bounded agent queries with fully qualified selectors,
  compact-by-default progressive disclosure, and `limit`/`offset` pagination;
- `candle-graph/runtime/1`: JSON/JSONL tensor observations and per-parameter gradient facts;
  see [docs/runtime-analysis-guide.md](docs/runtime-analysis-guide.md) for the full workflow and
  v2 (time series) / v3 (profiler) extensions;
- the original single-root tree, key list, baseline, checkpoint-header verification, dataflow
  report, and standalone HTML viewer.

## Status

Primary crate-wide UX is the Clippy-like Cargo subcommand `cargo candle-graph`
(`check` / `report` / `query`) over the existing conservative analyzer. It is an explicit
cargo command, not a rustc lint pass and not a silent `cargo build` hook.

The agent-analysis path now supports:

- automatic component discovery from public API boundaries and `VarBuilder`/`VarBuilderArgs`
  constructors, with `--root` for private model roots;
- public tensor-boundary and Candle-trait entrypoint candidates with fully qualified identities;
- symbolic tensor rank/shape, dtype, device, layout, and `requires_grad` contracts;
- Cargo package, target, feature, Candle-version, active `cfg`, and per-function cfg-activity facts;
- audited Candle 0.11.0 operation rules that distinguish similarly named APIs such as
  `RmsNorm::forward` and `RmsNorm::forward_diff`;
- optional runtime tensor refinement and missing/zero/non-finite gradient findings.

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
cargo install --path . --locked
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

The legacy `candle-graph` binary remains available (including for existing audit scripts):

```bash
cargo run --release -- /path/to/model --model-ir --format json
```

Ask focused questions. Listings are compact (counts/IDs, `drill_down` hints, no tensor evidence);
narrow with `--select` / singular kinds when you need contracts or evidence:

```bash
cargo candle-graph query summary --path /path/to/model
cargo candle-graph query doctor --path /path/to/model
cargo candle-graph query components --path /path/to/model
cargo run --release -- /path/to/model --query composition --format json
cargo run --release -- /path/to/model --query assembly --select model --format json
cargo run --release -- /path/to/model --query pipeline --format json
cargo run --release -- /path/to/model --query entrypoints --format json
cargo run --release -- /path/to/model \
  --query tensors --select 'MyModel::forward' \
  --limit 20 --offset 0 --format json
cargo run --release -- /path/to/model \
  --query tensor --select '<tensor-stable-id>' --format json
cargo run --release -- /path/to/model --query cargo --format json
cargo run --release -- /path/to/model --query findings --format json
```

Use fully qualified selectors when bare names collide:

```bash
cargo run --release -- /path/to/model \
  --query function --select 'model::MyModel::forward' \
  --format json
```


Select Cargo features/target exactly as the analyzed build does:

```bash
cargo run --release -- /path/to/model \
  --features cuda,flash-attn \
  --target x86_64-unknown-linux-gnu \
  --query cargo --format json
```

The analyzer follows the selected Cargo crate root and reachable `mod` declarations. It defaults
to the library target, then the first ordinary binary; use `--cargo-target <name>` for another
binary/example/test target. The existing `--target` option selects a Rust target triple.

Import an instrumented small-run trace:

```bash
cargo run --release -- /path/to/model \
  --runtime-trace run.runtime.jsonl \
  --query runtime --format json
```

See [docs/agent-query-api.md](docs/agent-query-api.md) for the query map,
[docs/runtime-protocol.md](docs/runtime-protocol.md) for trace field definitions, and
[docs/runtime-analysis-guide.md](docs/runtime-analysis-guide.md) for the recommended runtime workflow
(gradient audit, phase graphs, offline profiling).

The original single-component workflows remain available:

```bash
cargo run --release -- /path/to/model/src --root MyModel
cargo run --release -- /path/to/model --root MyModel \
  --entry MyModel::forward --format html --output model-graph.html
cargo run --release -- /path/to/model/src --root MyModel \
  --verify /path/to/model.safetensors --verify-root base_vb
```

Checkpoint verification reads only the safetensors length prefix and JSON header. A template such
as `model.layers.{index}.self_attn.q_proj.weight` matches all concrete layer indices.

Create and check a canonical CI baseline:

```bash
cargo run --release -- /path/to/model/src \
  --root MyModel \
  --entry train \
  --update-baseline model.candle-graph

cargo run --release -- /path/to/model/src \
  --root MyModel \
  --entry train \
  --check model.candle-graph \
  --strict
```

`--strict` rejects unknown structural parameters, structure diagnostics, proven dtype conflicts,
potential dtype risks, and dead trainable parameters. Known severing edges remain visible findings
but do not fail strict mode by themselves because deliberate detaches are valid.

Run `candle-graph --help` for all options.

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
