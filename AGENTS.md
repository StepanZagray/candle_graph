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

Matt Pocock engineering skills are installed under `.agents/skills/`. Invoke them with `/skill-name` in Cursor chat.

### Issue tracker

GitHub Issues on `StepanZagray/candle_graph` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` at the repo root and ADRs under `docs/adr/`. See `docs/agents/domain.md`.

### Cargo features

| Feature | Use |
| --- | --- |
| `visualizer` (default) | `model.html` trace viewer |
| `all` | Same as default today |

Details: [`docs/features.md`](docs/features.md), [`docs/runtime-analysis-guide.md`](docs/runtime-analysis-guide.md).
Prefer bounded CLI queries (`summary`, `query --kind …`) over loading full graph JSON.

### UI / UX (HTML visualizer)

Standalone HTML visualizer (`src/viewer/`, emitted as `model.html`). Stack: embedded CSS + vanilla JavaScript + dagre — not React. Read [`docs/visualizer.md`](docs/visualizer.md) before changing layout, tabs, or graph views.

| Skill | When to use |
|-------|-------------|
| `/ui-ux-pro-max` | Color, typography, spacing, accessibility, chart styling, interaction patterns. Use stack `html-tailwind` or general UX domains when querying. |
| `/web-design-guidelines` | Audit existing viewer HTML/CSS/JS against Vercel Web Interface Guidelines. |
| `/design-an-interface` | Explore radically different UI layouts or control surfaces (parallel design options). |
| `/prototype` | Quick throwaway UI variants before committing to the embedded viewer. |

Prefer improving the existing visualizer over introducing a frontend framework.

## Related projects

- Sibling crate used by [Tofy](../Tofy): path dependency for post-run profiling and HTML inspection.

## Module map

| Module | Role |
| --- | --- |
| `trace/` | Parse/emit `candle-graph/trace/4` JSONL |
| `instrument/` | `TraceSession`, `SpanGuard`, probe API |
| `graph/` | `ExecutionGraph` from trace events |
| `cli/trace_cli.rs` | `import`, `view`, `summary`, `query` |
| `viewer/` | `render_trace_html` (`candle-graph/viewer/2`) |

There is no static Rust analysis layer in this crate.
