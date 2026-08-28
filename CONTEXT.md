# candle-graph product context

## Goal

Help people and building agents understand what a Candle model actually did in one representative
run, without pretending static source inspection can reconstruct tensor execution.

## Domain language

**Profile run**: one selected training update or inference invocation, with enough provenance and
timing semantics to judge whether it represents the intended workload.

**Session envelope**: the lifetime owned by `TraceSession`. It contains trace setup and one measured
region; publication happens after it.

**Measured region**: the caller-controlled interval used for outer-wall totals and comparisons.
Host headlines include its normal descendants and structurally separate host spans that overlap
the interval. Overlapping work is labelled concurrent rather than reparented into the measured call
tree.

**Capture contract**: the producer's typed declaration of which evidence classes cover the entire
measured region. An observed event proves presence, but only the producer can declare its coverage
policy complete.

**Semantic label contract**: the exact set of required application span names. The set may be
partitioned into labels expected to project onto the GPU and labels known to be CPU-only. Older
contracts with no explicit partition treat every required label as GPU-expected.

**Gradient contract**: a SHA-256-bound, caller-ordered manifest of expected `(root, key, family)`
parameters, plus an active, inactive, or data-conditional expectation for every family. Complete
gradient coverage means the observed events match the manifest and all state, norm, and family
rules.

**Evidence capability**: a machine-checkable statement that one class of conclusion is complete,
partial, unavailable, or invalid for an evidence packet.

**Evidence packet**: the shared deep interface for CLI output, reports, bundles, and the HTML
viewer: provenance, health, capabilities, facts, gaps, an optional validated graph, timing, memory,
and optional GPU evidence.

**Timing plane**: one clock and interval semantics. Host spans, device-event intervals, and
Nsight-projected intervals remain separate timing planes even when they describe the same phase.

**Logical memory evidence**: explicit backend storage lifetimes keyed by device and storage
identity, including simultaneous live bytes and residual storage.

**Physical memory evidence**: observed device or allocator used, reserved, free, and capacity
bytes. Every measurement is independently optional; capacity is never inferred from another
field, tensor metadata, or logical lifetime evidence.

**GPU evidence**: optional normalized Nsight kernel, runtime, transfer, timeline, and projected
NVTX facts. Its absence never invalidates application evidence.

**Evidence bundle**: one atomically published directory whose manifest binds every input and
derived file to the profile run.

**Replicated comparison**: a comparison of at least five compatible independent baseline bundles
and five compatible independent candidate bundles. It retains every timing sample and binds every
input to a point-in-time bundle verification receipt.

**Failed capture**: a terminated profile run with an explicit failure reason and partial structural
evidence. It is diagnosable but cannot support a normal graph, findings, or timing verdict.

## Evidence flow

```text
representative update or inference call
  ├─ TraceSession envelope
  ├─ one measured semantic region
  └─ matching NVTX labels (optional)
                 │
                 ▼
      application.jsonl (trace/10)
                 │
        structural validation ──────┐
                 │                  │
                 ▼                  │
             graph/5                │
                                    │
 official nsys stats CSV ─ normalize┤
                                    ▼
                            evidence/4 packet
                              ├─ CLI JSON / Markdown
                              ├─ atomic bundle/1 + verification receipt
                              ├─ comparison/5 across repeated bundles
                              └─ viewer/5 HTML
```

Analysis never proceeds through an invalid hierarchy. Failed traces keep diagnostic evidence but
do not receive a graph. Optional evidence is represented by capabilities and reasons rather than
silent empty-success values. Tensor footprint and observed memory lifetime are different facts.

## Product invariants

1. Instrumentation is inactive outside the caller-selected invocation.
2. One caller-owned measured region supplies the run total.
3. Required provenance and typed timing semantics precede comparisons.
4. Structural trust precedes graph construction and graph-dependent findings.
5. Agent-facing JSON states evidence gaps and output truncation explicitly.
6. Timing comparisons fail closed and retain raw samples plus the confidence interval.
7. One accessible offline UI presents application and optional GPU evidence.
8. Nsight normalization consumes supported official CSV and retains the raw `.nsys-rep`.
9. Complete gradient coverage requires an exact manifest and family validation.
10. Eligible comparisons use finalized bundles and retain a verification receipt for each input.

## Non-goals

- Static Rust analysis or reconstruction of operations that were not observed.
- Treating host launch duration as completed asynchronous device work.
- Inferring physical allocation size from shape metadata.
- Aligning Candle and Nsight clocks without an explicit mapping.
- Declaring unverified raw traces eligible for a performance verdict.
- Parsing Nsight's internal SQLite schema as a stable interface.

## Current protocols

| Schema | Role |
| --- | --- |
| `candle-graph/trace/10` | Execution JSONL with tensor statistics and exact gradient contracts |
| `candle-graph/graph/5` | Validated call/data graph and measured host scopes |
| `candle-graph/evidence/4` | Capability-qualified application and GPU evidence packet |
| `candle-graph/comparison/5` | Replicated timing and tensor-stat comparison |
| `candle-graph/viewer/5` | Embedded offline UI payload |
| `candle-graph/gradient-manifest/1` | Ordered gradient-manifest digest domain |
| `candle-graph/nsight-capture/1` | Nsight artifact and run-identity manifest |
| `candle-graph/bundle/1` | Content-addressed evidence bundle |
| `candle-graph/bundle-verification/1` | Point-in-time deep-verification receipt |

The parser reads trace/10 and trace/9; new sessions emit trace/10. Derived packet consumers require
their current schema. See [schemas and compatibility](docs/schemas.md) for the complete list.
