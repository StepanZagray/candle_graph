# Documentation

Start with the page that matches the job:

| Goal | Read |
| --- | --- |
| Add capture code to an application | [Runtime evidence guide](runtime-analysis-guide.md) |
| Understand commands and JSON output size | [CLI reference](cli-reference.md) |
| Choose Cargo features | [Cargo features](features.md) |
| Integrate or change the offline UI | [HTML visualizer](visualizer.md) |
| Consume a wire format | [Schemas and compatibility](schemas.md) |
| Understand product terms and boundaries | [Product context](../CONTEXT.md) |

## The workflow in one picture

```text
selected update or inference call
             │
             ▼
 application.jsonl (trace/10)
             │
       validate + qualify  ◀──── optional official Nsight CSV + capture manifest
             │
             ▼
       evidence/4 packet
        ├─ overview/summary/query JSON
        ├─ report.md
        ├─ viewer/5 HTML
        └─ bundle/1 manifest + publication/1 receipt
                         │
                         ├─ comparison/6 across repeated verified bundles
                         └─ campaign-status/1 + series/1 across a campaign
```

The execution graph is a derived view, not the source of truth. Structural validation and
capability qualification happen before graph-dependent findings or queries are produced.

## Three rules that prevent most mistakes

1. Capture one representative invocation and put exactly one caller-owned measured region inside
   the session envelope.
2. Keep host time, device intervals, Nsight time, logical storage, and physical memory as separate
 evidence planes.
3. Prefer a verified bundle over a raw trace after publication, and treat the bundle directory as
   immutable.
