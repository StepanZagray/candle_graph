# candle-graph — product context

## Goal

Help people and building agents understand what a Candle model actually did in one representative
run without pretending static source inspection can reconstruct tensor execution.

## Language

**Profile run**: one selected model update or inference invocation, with enough provenance and
timing semantics to judge representativeness.

**Measured region**: the caller-controlled part of a profile run used for totals and comparisons;
session setup and evidence publication are outside it.

**GPU evidence**: optional normalized Nsight kernel, runtime, transfer, timeline, and projected NVTX
facts. Its absence never invalidates application evidence.

**Evidence packet**: the single deep interface for agents, reports, and the HTML viewer:
provenance, structural health, capability states, typed facts, explicit gaps, optional validated
graph, separate timing and memory planes, and optional GPU evidence.

**Capture contract**: the producer's typed declaration of which evidence classes cover the whole
measured region. Observed event counts can prove presence, but never upgrade coverage to complete.

**Gradient contract**: a SHA-256-bound, caller-ordered manifest of expected `(root, key, family)`
parameters plus an active, inactive, or data-conditional expectation for every family. Complete
gradient coverage means the observed event set matched this contract exactly and passed state/norm
and family validation.

**Evidence capability**: a machine-checkable statement that one class of conclusion is supported,
partial, unavailable, or invalid for an evidence packet.

**Timing plane**: one clock and interval semantics for timing evidence. Host, device-event, and
Nsight-projected intervals are separate timing planes even when they describe the same phase.

**Logical memory evidence**: storage lifetimes identified by device and storage identity, including
simultaneous live bytes and residual storage.

**Physical memory evidence**: observed device or allocator used, reserved, free, and capacity bytes.
Each measurement is independently optional. Capacity is never derived from used plus free, tensor
metadata, or logical storage lifetimes.

**Replicated comparison**: a metric-scoped comparison of at least five compatible independent
baseline runs and five compatible independent candidate runs, retaining every raw sample. An
eligible comparison reads traces only from bundles deeply verified immediately beforehand.

**Evidence bundle**: one atomically published directory whose manifest binds every input and
derived artifact to the profile run.

**Failed capture**: a terminated profile run with an explicit failure reason and partial structural
evidence. It is diagnosable but cannot support normal findings or comparisons.

## Evidence flow

```text
representative update
  ├─ TraceSession envelope
  ├─ one measured semantic region
  └─ matching NVTX labels (optional)
            │
            ▼
 application.jsonl (trace/9) ── validate ── graph/4
            │                                │
 official nsys stats CSV ── normalize ───────┤
            │                                ▼
            └──────────────────── evidence/3 packet
                                      ├─ bounded JSON / Markdown for agents
                                      ├─ atomic bundle/1 + verification receipt
                                      ├─ comparison/4 across verified repeated bundles
                                      └─ viewer/5 HTML for humans
```

Analysis never proceeds through an invalid hierarchy. Optional evidence is represented by coverage
and reasons, never silent empty arrays. Tensor footprint and observed memory lifetime are distinct.

## Product targets

1. Zero overhead outside one selected invocation.
2. One caller-owned measured region with forward/backward/optimizer semantics.
3. Required provenance and typed timing mode before comparisons.
4. Structural trust before graph construction or findings.
5. Bounded agent queries and durable JSON/Markdown evidence.
6. Fail-closed repeated-run timing comparisons with raw samples and confidence intervals.
7. One accessible offline UI for application and optional GPU evidence.
8. Stable normalization from official Nsight CSV, retaining raw `.nsys-rep`.
9. Exact gradient-manifest and family validation before claiming complete gradient coverage.
10. Bundle verification receipts on every comparison input; raw traces are diagnostic-only.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/9` | Execution evidence stream with exact gradient contract and terminal outcome |
| `candle-graph/graph/4` | Validated call/data graph with separate timing planes |
| `candle-graph/evidence/3` | Capability-qualified agent/human packet with gradient-contract facts |
| `candle-graph/comparison/4` | Replicated outer-wall comparison with verified bundle receipts |
| `candle-graph/viewer/5` | Embedded offline UI payload |
| `candle-graph/gradient-manifest/1` | Ordered parameter-manifest digest domain |
| `candle-graph/nsight-capture/1` | Nsight raw/normalized artifact provenance |
| `candle-graph/bundle/1` | Content-addressed atomic evidence bundle |
| `candle-graph/bundle-verification/1` | Deep verification receipt bound to one manifest digest |

## 0.9 schema migration

- Trace/8 is rejected. Emit trace/9; `gradients: complete` now requires `gradient_contract`.
- Evidence/3 carries trace/9 provenance and typed gradient manifest/family facts.
- Comparison/4 embeds per-run bundle manifest receipts. The default CLI inputs are bundle
  directories; `--unverified-traces` accepts raw traces but always produces an ineligible result.
