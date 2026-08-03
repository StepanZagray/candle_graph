# Runtime tensor and gradient protocol

Static analysis remains the source of model structure. A small or synthetic instrumented run can
add concrete tensor metadata and gradient observations using `candle-graph/runtime/1`. The trace
can be a JSON document or a stream of JSONL events.

## JSON document

```json
{
  "schema": "candle-graph/runtime/1",
  "run": {
    "entrypoint": "model::train_step",
    "profile": "debug",
    "cargo_features": ["cuda"],
    "cfg": ["feature=\"cuda\""],
    "analysis_id": "analysis:candle-graph%2Fbuild%2F1%3A0123456789abcdef",
    "build_id": "candle-graph/build/1:0123456789abcdef"
  },
  "tensors": [
    {
      "event_id": "encoder-output-1",
      "static_id": "tensor:function%3A...:output",
      "source": "model::train_step output",
      "shape": [2, 32, 768],
      "dtype": "BF16",
      "device": "cuda:0",
      "contiguous": true,
      "requires_grad": true
    }
  ],
  "operations": [],
  "gradients": [
    {
      "event_id": "adapter-weight-grad-1",
      "root": "train_vb",
      "key": "adapter.proj.weight",
      "state": "present",
      "norm": 0.03125
    }
  ]
}
```

`analysis_id` and `build_id` are optional. When omitted, older traces remain valid. When supplied,
importers must compare them against the active analysis/build identity and either reject the
trace or report an identity mismatch as Unknown evidence — never silently accept a mismatched run.

`static_id` is optional. When present, it should be copied from a `tensors` query or the model IR;
the importer uses it to refine that exact static tensor contract. Gradient correlation uses the
pair `(root, key)`, matching `builder_root` and `key` in a parameter query.

Gradient `state` is one of:

- `present`: a finite, non-zero gradient;
- `missing`: no gradient was produced and `norm` must be omitted;
- `zero`: the gradient exists but its norm is zero;
- `non_finite`: the gradient or its norm contains NaN/Inf.

Duplicate event IDs and internally invalid gradient facts (state/norm contradictions) are
rejected. Repeated tensor observations with the same `static_id`, and repeated gradient facts for
the same `(root, key)`, are audited for conflicts. Conflicting observations are reported with
confidence `unknown` and must not be refined as Proven — `agreed_tensor` / `gradient` return no
winner rather than silently overwriting.

## JSONL stream

The first event declares run metadata. Later events may be appended as the program executes:

```jsonl
{"kind":"meta","schema":"candle-graph/runtime/1","entrypoint":"tasks::bridge::train_step","profile":"debug","cargo_features":["cuda"],"cfg":["feature=\"cuda\""]}
{"kind":"tensor","event_id":"encoder-output-1","shape":[2,32,768],"dtype":"BF16","device":"cuda:0","contiguous":true,"requires_grad":true}
{"kind":"gradient","event_id":"adapter-weight-grad-1","root":"train_vb","key":"adapter.proj.weight","state":"present","norm":0.03125}
```

Import and query the merged result:

```bash
cargo run --release -- /path/to/model \
  --runtime-trace run.runtime.jsonl \
  --query runtime --format json
```

## Emit JSONL from a short rollout

The library includes a dependency-free streaming helper, so model crates can instrument a short
CPU or synthetic rollout without loading checkpoints into the analyzer. Copy a stable tensor id
from `--query tensors` and pass it as `static_id`:

```rust
use candle_graph::runtime::{
    GradientFact, GradientState, RunMetadata, RuntimeTraceWriter, TensorObservation,
};

let file = std::fs::File::create("run.runtime.jsonl")?;
let mut trace = RuntimeTraceWriter::new(file, RunMetadata {
    entrypoint: "model::train_step".into(),
    profile: "debug".into(),
    cargo_features: vec![],
    cfg: vec![],
    analysis_id: Some(analysis_id_from_summary),
    build_id: Some(build_id_from_cargo_query),
})?;

trace.tensor(TensorObservation {
    event_id: "step-0-output".into(),
    static_id: Some(static_tensor_id),
    source: Some("model::train_step output".into()),
    shape: output.dims().to_vec(),
    dtype: format!("{:?}", output.dtype()),
    device: "cpu".into(),
    contiguous: output.is_contiguous(),
    requires_grad: true,
    storage_id: None,
})?;
trace.gradient(GradientFact {
    event_id: "step-0-head-weight-grad".into(),
    root: "train_vb".into(),
    key: "head.weight".into(),
    state: GradientState::Present,
    norm: Some(gradient_norm),
})?;
trace.finish()?;
```

Then import the emitted trace:

```bash
cargo run --release -- /path/to/model \
  --runtime-trace run.runtime.jsonl --query runtime --format json
```

The trace records one observed execution. It refines concrete runtime properties but never turns a
conditional static architecture or training claim into a universal one.

## Schema v2 (`candle-graph/runtime/2`)

v2 adds optional `step: u64` on tensor, gradient, and value observations so a trace can represent a
time series (JSONL or a single multi-fact document). Importers treat v1 traces as v2 with all steps
omitted.

Additional record type in v2+ documents:

```json
{
  "event_id": "step-42-q-logit",
  "source": "losses.q",
  "step": 42,
  "min": -3.1,
  "max": 22.4,
  "abs_max": 22.4,
  "nonfinite_count": 0,
  "saturated_count": 0
}
```

Value observations support numeric-domain runtime confirmation; they are optional and must be
emitted by the probe — the analyzer does not invent them.

## Schema v3 (`candle-graph/runtime/3`)

v3 extends v2 for profiling:

- `run.phase`: `"train"` or `"infer"`
- `operations[].duration_ns`: wall time for one observed op
- `edge_timings[]`: `{ from_static_id, to_static_id, duration_ns, step? }`

Emit v3 JSONL with `ProfileSession` (`candle_graph::profile`) or
`RuntimeTraceWriter::new_with_schema(…, SCHEMA_V3, …)`.
Merge into the model IR with:

```bash
cargo candle-graph profile --runtime-trace forward-profile.jsonl --path /path/to/model
```

## End-to-end workflow

For the recommended order of operations (static gate → checkpoint audit → phase graphs → offline
profiler), worked examples, and limitations, see
[runtime-analysis-guide.md](runtime-analysis-guide.md).
