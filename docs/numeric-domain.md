# Numeric domain and saturation hazards

The analyzer does **not** hardcode “this loss is bad.” It stores float-range transfer
facts and, for audited candle-nn helpers, expandable **body recipes**. The same domain
pass judges local source and expanded library bodies, then classifies whether the hazard
can fail training or interfere with inference.

## What is decidable statically

A proof that a logit *will* saturate depends on weights and data. What *is* decidable
from the catalog is the **composition hazard**: a `log` / `div` / `recip` / `sqrt`
whose operand comes from an op whose float range attains the forbidden endpoint, with
no discharging guard between.

## Catalog vs verdict

| Layer | Contents |
|---|---|
| `OpEffect.domain` / `requires` | Transfer facts (`sigmoid → SaturatingUnit`, `log` requires `StrictlyPositive`, …) |
| `library_body(op, version)` | Expandable recipe (e.g. BCE = sigmoid → log → mul), version-gated to 0.11.0 |
| Domain pass | Emits `numeric-domain-violation` / `zero-times-infinity` |
| Impact pass | Labels `TrainingLossNaN` / `GradientPoison` / `InferenceOutputRisk` / `LocalOnly` |

Measured sigmoid f32 thresholds (candle `(exp(-v)+1).recip()`): upper `16.6355`,
lower `-88.7228`.

## Affine discharge

Only the epsilon-guard form `affine(mul, add)` with **`mul > 0` and `add > 0`** on a
non-negative / unit operand lifts to `StrictlyPositive`. Reflections such as
`affine(-1, 1)` (`1 - p`) keep a zero-attaining domain.

```rust
// flagged
candle_nn::ops::sigmoid(&x)?.log()?
// clean
candle_nn::ops::sigmoid(&x)?.affine(1.0, 1e-7)?.log()?
// library call: expanded, then flagged by the same rules
candle_nn::loss::binary_cross_entropy_with_logit(&logit, &target)?
```

## Impact → severity

| Impact | Meaning | Strict / check |
|---|---|---|
| `TrainingLossNaN` | Hazard reaches a loss sink | Error + Proven |
| `GradientPoison` | Hazard on trainable → loss path | Error + Proven |
| `InferenceOutputRisk` | Hazard reaches entry return | Error on forward entrypoints; else Warning |
| `LocalOnly` | Unstable locally, not shown to escape | Warning + Proven |

`--strict` / `cargo candle-graph check` still fail only on proven Errors.

## Not in this slice

- Runtime protocol `/2` value observations and step time-series
- Expanding arbitrary dependency crates beyond audited body recipes
