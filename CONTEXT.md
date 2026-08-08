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

**Evidence packet**: the single deep interface for agents, reports, comparisons, and the HTML
viewer: provenance, structural health, coverage, structured facts, explicit gaps, graph, optional
baseline, and optional GPU evidence.

## Evidence flow

```text
representative update
  ├─ TraceSession envelope
  ├─ one measured semantic region
  └─ matching NVTX labels (optional)
            │
            ▼
 application.jsonl (trace/6) ── validate ── graph/3
            │                                │
 official nsys stats CSV ── normalize ───────┤
            │                                ▼
            └──────────────────── evidence/1 packet
                                      ├─ bounded JSON / Markdown for agents
                                      └─ viewer/4 HTML for humans
```

Analysis never proceeds through an invalid hierarchy. Optional evidence is represented by coverage
and reasons, never silent empty arrays. Tensor footprint and observed memory lifetime are distinct.

## Product targets

1. Zero overhead outside one selected invocation.
2. One caller-owned measured region with forward/backward/optimizer semantics.
3. Required provenance and typed timing mode before comparisons.
4. Structural trust before graph construction or findings.
5. Bounded agent queries and durable JSON/Markdown evidence.
6. Baseline deltas by semantic path, memory, and gradient facts.
7. One accessible offline UI for application and optional GPU evidence.
8. Stable normalization from official Nsight CSV, retaining raw `.nsys-rep`.

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/6` | Execution evidence stream |
| `candle-graph/graph/3` | Validated semantic graph |
| `candle-graph/evidence/1` | Unified agent/human packet |
| `candle-graph/comparison/1` | Explicit baseline comparison |
| `candle-graph/viewer/4` | Embedded offline UI payload |

## Related project

Tofy is the primary consumer. It captures update 1 by default, publishes an atomic per-update
bundle, emits the same semantic labels to candle-graph and NVTX, and exposes the packet to repair
agents and humans.
