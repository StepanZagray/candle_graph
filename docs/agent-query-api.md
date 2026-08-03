# Agent query API

The default agent workflow is progressive disclosure: start with compact listings, then narrow
with an explicit selector when tensor contracts or evidence payloads are required.

```bash
cargo run --release -- /path/to/crate \
  --query <kind> [--select <qualified-name-or-id>] \
  [--to <qualified-name-or-id>] [--limit 100] [--offset 0] --format json
```

Every response uses `candle-graph/query/1` and reports `total`, `returned`, `offset`, and
`truncated`. The CLI exposes `--limit` and `--offset` for deterministic pagination of the sorted
result set. Selectors are
case-insensitive substrings over stable IDs and the most relevant names. Prefer a fully qualified
Rust path or stable ID whenever a bare name returns multiple definitions.

Compact-by-default rules:

- `summary`, `architecture`, `pipeline`, `components`, `functions`, `entrypoints`, and `modules`
  omit individual tensor contracts and evidence arrays. Counts or IDs stand in for nested detail.
- `tensors` returns compact rows (`id`, role, owner, dtype, `shape_rank`, …) without `shape` or
  `evidence`. Full tensor contracts require `tensor --select <id-or-name>`.
- `findings` omits `evidence`; evidence is returned only when `--select` is the exact finding ID.
- Compact records include a `drill_down` array of `{kind, select?}` hints for the next query.

| Question | Query |
| --- | --- |
| What was discovered and how complete is it? | `summary` |
| Can I trust this scan, and where are its evidence gaps? | `doctor` |
| What is the compact structure map? | `architecture` |
| What are the model components? | `components` |
| What is one component made of? | `component --select <type>` |
| Which public tensor/Candle-trait boundaries are callable? | `entrypoints`, `functions` |
| What does one function declare (tensor IDs, not contracts)? | `function --select <path>` |
| Which parameters and running-state registrations exist? | `parameters` |
| Which tensors exist (compact)? | `tensors` |
| What shape/dtype/device/layout/evidence does one tensor have? | `tensor --select <id>` |
| Which operations occur in a function? | `operations --select <function-id>` |
| What are one operation's inputs, output, rules, and evidence? | `operation --select <id>` |
| What submodule tree does a component contain? | `modules --select <type>` |
| Which components nest inside another (struct fields)? | `composition`, `composition --select <type>` |
| How are components wired from VarMaps/mmap at runtime? | `assembly`, `assembly --select bridge` |
| What is the training pipeline order and subprocess CLI? | `pipeline`, `stages`, `stages --select latent` |
| Which Cargo features, cfg values, target, and Candle versions apply? | `cargo` |
| What did a runtime probe observe? | `runtime` |
| What is uncertain or broken? | `findings`, then `findings --select <exact-id>` |
| Is there a static tensor/dataflow path between two selectors? | `path --select <from> --to <to>` |

Example calls for a modular model crate:

```bash
cargo run --release -- /path/to/model --query summary --format json
cargo run --release -- /path/to/model --query doctor --format json
cargo run --release -- /path/to/model --query components --format json
cargo run --release -- /path/to/model --query composition --format json
cargo run --release -- /path/to/model --query assembly --select model --format json
cargo run --release -- /path/to/model --query pipeline --format json
cargo run --release -- /path/to/model --query stages --select train --format json
cargo run --release -- /path/to/model --query modules --select MyModel --format json
cargo run --release -- /path/to/model --query entrypoints --select MyModel --format json
cargo run --release -- /path/to/model --query entrypoints --format json
cargo run --release -- /path/to/model --query parameters --select head --format json
cargo run --release -- /path/to/model --query tensors \
  --select 'MyModel::forward' --limit 20 --format json
cargo run --release -- /path/to/model --query tensor \
  --select '<tensor-stable-id>' --format json
```

Recommended agent workflow for modular model development:

1. `summary` then `doctor` — check `coverage_quality` (tensor contract coverage, dtype risks,
   `component_entrypoints` vs `total_entrypoints`) before trusting individual records.
2. `components` — enumerate public VarBuilder-backed model types and their builder namespaces.
3. `composition --select <Parent>` — see which components nest via struct fields (Heuristic).
4. `modules --select <Parent>` — inspect prefix tree, repeats, and submodule types without
   loading full `component` payloads.
5. `entrypoints --select <Component>` — prefer `is_component_entrypoint: true` rows; crate-level
   data helpers remain visible but sort last.
6. `function` / `tensors` / `tensor` — drill into one boundary with progressive disclosure.
7. `assembly --select bridge` — see VarMap/mmap wiring, builder roots, prefix chains, and
   checkpoint load expressions in task code (Heuristic).
8. Optional `--heuristic-architecture` — exploratory name/call-order pipeline, subprocess,
   artifact, optimizer, and builder-role leads. These remain `Heuristic`, never proven facts.

Evidence should drive agent decisions:

- `proven` follows directly from the evidence source represented by the current frontend.
- `conditional` depends on a branch, configuration, optimizer filter, or incomplete static fact.
- `heuristic` is a useful discovery lead, not a correctness guarantee.
- `unknown` must not be silently converted into a concrete shape, dtype, device, or training role.

Architecture, pipeline, artifact, and optimizer *call-order* queries return no relationships unless
`--heuristic-architecture` is set. Struct-field **composition** edges (`composition` query) are
always emitted with `Heuristic` confidence when a component embeds another discovered component.
The model IR includes a `compiler-semantic-evidence` finding explaining the call-order gap. Those
facts require the compiler-resolved value-flow frontend described in
[compiler-evidence-design.md](compiler-evidence-design.md); source names, order, and filename
suffixes are deliberately not used as substitutes.

`--model-ir --format json` emits the complete `candle-graph/model/1` document when a bounded query
is insufficient. It is the same normalized IR used by every query; the query layer does not run a
separate discovery implementation.
