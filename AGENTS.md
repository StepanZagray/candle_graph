## Scope

- These rules apply to this repository.
- Read this file before making edits.
- Read [`CONTEXT.md`](CONTEXT.md): **trace-only** TensorFlow Profiler-style graphs for Candle.

## Chat

- If the user asks for explanation only, do not modify files.
- Prefer concrete, implementation-level explanations.

## Edits

- Use repository-native tooling and normal editor operations for file changes.
- Keep changes minimal and targeted; avoid unrelated refactors.
- Do not preserve backward compatibility unless the user explicitly asks for it.
- After code edits, run `cargo test` and `cargo clippy` when possible; resolve errors and warnings.

## Agent skills

Agent skills are installed globally under `~/.agents/skills/`. Invoke them with `/skill-name` in Cursor chat.
Use the global `/research` skill for the local-memory research workflow.

### Issue tracker

GitHub Issues on `StepanZagray/candle_graph` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` at the repo root and ADRs under `docs/adr/`. See `docs/agents/domain.md`.

### Cargo features

| Feature | Use |
| --- | --- |
| `visualizer` (default) | Standalone `viewer.html` evidence viewer |
| `candle` | Candle tensor and storage capture helpers |
| `all` | `visualizer` plus `candle` |

Details: [`docs/features.md`](docs/features.md), [`docs/runtime-analysis-guide.md`](docs/runtime-analysis-guide.md).
Prefer bounded CLI queries (`summary`, `query --kind …`) over loading full graph JSON.

### UI / UX (HTML visualizer)

Standalone HTML visualizer (`src/viewer/`, normally emitted as `viewer.html`). Stack: embedded CSS + vanilla JavaScript + dagre — not React. Read [`docs/visualizer.md`](docs/visualizer.md) before changing layout, tabs, or graph views.

| Skill | When to use |
|-------|-------------|
| `/ui-ux-pro-max` | Color, typography, spacing, accessibility, chart styling, interaction patterns. Use stack `html-tailwind` or general UX domains when querying. |
| `/web-design-guidelines` | Audit existing viewer HTML/CSS/JS against Vercel Web Interface Guidelines. |
| `/interface-design` | Refine the visualizer's layout hierarchy, controls, states, and design-system consistency. |
| `/prototype` | Quick throwaway UI variants before committing to the embedded viewer. |

Prefer improving the existing visualizer over introducing a frontend framework.

## Module map

| Module | Role |
| --- | --- |
| `trace/` | Parse/emit `candle-graph/trace/6` JSONL |
| `instrument/` | `TraceSession`, `SpanGuard`, probe API |
| `graph/` | `ExecutionGraph` from trace events |
| `evidence.rs` | Unified evidence and baseline comparison packets |
| `artifact.rs` | Atomic bundle publication and deep verification receipts |
| `nsight.rs` | Normalize supported official Nsight CSV reports |
| `cli/trace_cli.rs` | `import`, `view`, `summary`, `query`, `compare`, `report` |
| `viewer/` | `render_evidence_html` (`candle-graph/viewer/4`) |

There is no static Rust analysis layer in this crate.
