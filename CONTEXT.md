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
baseline runs and five compatible independent candidate runs, retaining every raw sample.

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
 application.jsonl (trace/7) ── validate ── graph/4
            │                                │
 official nsys stats CSV ── normalize ───────┤
            │                                ▼
            └──────────────────── evidence/2 packet
                                      ├─ bounded JSON / Markdown for agents
                                      ├─ comparison/2 across repeated runs
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

## Schemas

| Schema | Role |
| --- | --- |
| `candle-graph/trace/7` | Execution evidence stream and terminal outcome |
| `candle-graph/graph/4` | Validated call/data graph with separate timing planes |
| `candle-graph/evidence/2` | Capability-qualified agent/human packet |
| `candle-graph/comparison/2` | Replicated outer-wall comparison |
| `candle-graph/viewer/5` | Embedded offline UI payload |
| `candle-graph/nsight-capture/1` | Nsight raw/normalized artifact provenance |
| `candle-graph/bundle/1` | Content-addressed atomic evidence bundle |
| `candle-graph/bundle-verification/1` | Deep verification receipt bound to one manifest digest |
