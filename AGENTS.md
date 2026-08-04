## Scope

- These rules apply to this repository.
- Read this file before making edits.

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
| `static` (default) | Agent IR + queries — no HTML, no runtime merge |
| `runtime` | Profile traces, operation timings, gradient audit |
| `visualizer` | `model.html` for humans |

Details: [`docs/features.md`](docs/features.md). Prefer the query API over full IR for agents.

### UI / UX (HTML visualizer)

This repo ships a standalone HTML visualizer (`src/viewer/`, emitted as `model.html`). Stack: embedded CSS + vanilla JavaScript + dagre — not React. Read [`docs/visualizer.md`](docs/visualizer.md) before changing layout, tabs, or graph views.

| Skill | When to use |
|-------|-------------|
| `/ui-ux-pro-max` | Color, typography, spacing, accessibility, chart styling, interaction patterns. Use stack `html-tailwind` or general UX domains when querying. |
| `/web-design-guidelines` | Audit existing viewer HTML/CSS/JS against Vercel Web Interface Guidelines. |
| `/design-an-interface` | Explore radically different UI layouts or control surfaces (parallel design options). |
| `/prototype` | Quick throwaway UI variants before committing to the embedded viewer. |

Prefer improving the existing visualizer over introducing a frontend framework.

## Related projects

- Sibling crate used by [Tofy](../Tofy): path dependency for model analysis and audit workflows.
- See `docs/agent-query-api.md` for the bounded query API agents should prefer over loading full source trees.
