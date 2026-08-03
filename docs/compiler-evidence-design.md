# Compiler-derived model discovery (target design)

> Status: planned frontend, not a description of the current syntax/type-table analyzer. The
> unified model IR, query API, Cargo context, qualified identities, versioned Candle rule catalog,
> and runtime import are already in place; the compiler-derived fact stream below is the remaining
> path to exact Rust semantics.

## Decision

The analyzer does not accept an architecture manifest, semantic annotations, naming
conventions, or a runtime execution as a source of truth.

Architecture and training facts must be derived from the program Cargo actually
type-checks. Runtime traces are optional observations that refine concrete shapes,
devices, branches, and gradient values; they never establish the static model
structure.

This also means analysis does not load checkpoint tensors. The cost is related to
compiling and analyzing the Rust program, not to the model's parameter count. A model
whose full weights do not fit on the developer's machine must still be statically
analyzable.

## Correctness rule

Every semantic fact in the model IR must have a derivation from compiler-resolved
facts and an audited Candle operation rule:

- `Proven`: follows for every execution represented by the active Cargo target,
  features, and `cfg` configuration.
- `Conditional`: follows only under a recorded control-flow predicate or bounded
  dynamic-dispatch alternative.
- `Unknown`: the compiler evidence is insufficient.

Name similarity, filename suffixes, source-order guesses, public naming conventions,
and string literals are not evidence. The analyzer must emit `Unknown` or omit a
relationship instead of guessing. In particular, method names such as `forward`,
`save`, `all_vars`, or `train_*` have no built-in semantic meaning unless name
resolution proves which trait or function they refer to.

## Static frontend

The frontend is driven by Cargo using the selected package, target, features, target
triple, and active `cfg` values. It consumes compiler-resolved program facts rather
than reparsing an unconfigured directory of `.rs` files.

The minimum compiler fact stream is:

1. Stable definition identity for functions, methods, traits, implementations,
   structs, fields, locals, and call sites.
2. Resolved type and trait identity at every relevant expression.
3. Expanded active code after module loading, macro expansion, and `cfg` selection.
4. Control-flow graphs and place/value flow for function bodies.
5. Generic substitutions and the finite set of statically known dispatch targets.

The durable analyzer IR must use definition identities internally. Display paths are
metadata, not lookup keys.

## Derived model facts

### Parameters and modules

A parameter is created only by a call resolved to an audited Candle parameter
registration API. `VarBuilder` namespaces and sub-builders are tracked by value/place
identity through moves, borrows, fields, returns, closures, and interprocedural calls.

A module is a parameter-owning value or type in that ownership graph. A top-level
component is a maximal parameter-owning API value in the analyzed construction graph.
This definition does not depend on type or method names.

### Entrypoints and architecture

An entrypoint is a method resolved to an audited Candle trait implementation, or an
externally reachable function selected by the analyzed Cargo target. Arbitrary
`forward`/`encode`/`loss` names are not entrypoints.

Architecture edges come from tensor/value flow between component-owned operations.
Function parameter order, field declaration order, and lexical call order alone do
not establish direction.

### Training roles

`Optimized` requires value-flow proof from a particular parameter/`VarMap` to the
resolved optimizer constructor or parameter-group API. Include/exclude behavior is
represented as its actual control-flow predicate.

`Frozen` means that the analyzer proves a parameter cannot reach any optimizer in the
analyzed training roots. Absence from an incompletely analyzed call graph is
`Unknown`, not `Frozen`. Running state comes only from resolved stateful Candle APIs,
not key substrings such as `running_mean`.

### Pipelines and artifacts

A pipeline is a dependency graph derived from typed data flow and resolved I/O calls.
Stage labels are presentation only; names such as `run_pipeline`, `train_*`, or
`final_eval` never create stages.

An artifact requires a call resolved to an audited producer/consumer API plus
place/value flow for its path or handle. A string ending in `.safetensors`, `.bin`, or
`.parquet` does not by itself create an artifact.

## Optional runtime evidence

A trace may be generated from a small configuration, synthetic inputs, or an
instrumented partial entrypoint. It may add:

- concrete tensor shapes, dtypes, layouts, and devices;
- the branch and dispatch target observed in that run;
- measured gradient presence, norm, and finiteness.

Runtime evidence is labelled as one observed execution. It does not upgrade a
conditional static fact to universally proven, and full model execution is never
required.

## Implementation sequence

1. Make the existing syntax frontend conservative: remove name/suffix/stem-derived
   architecture, pipeline, artifact, and optimizer claims.
2. Add definition identities and resolved call/type facts from a Cargo/compiler
   frontend while retaining the current model IR and query API.
3. Reimplement `VarBuilder`, parameter ownership, component, and optimizer analysis as
   interprocedural place/value dataflow.
4. Derive tensor architecture and artifact dependencies from the same resolved
   control-flow graph.
5. Version the audited Candle semantic catalog from the resolved Cargo packages.
6. Track coverage and `Unknown` causes so unsupported Rust constructs are visible and
   cannot silently become confident claims.
7. Keep runtime import as an independent, optional refinement layer.
