//! Integration tests for the runtime evidence / gradient-audit protocol.
//!
//! `runtime` is also available via `candle_graph::runtime`; path-include keeps these tests
//! focused on the transport schema even if wiring changes.

#[path = "../src/runtime.rs"]
mod runtime;

use runtime::{
    parse, parse_json, parse_jsonl, ExpectedIdentity, GradientFact, GradientState, IdentityField,
    ObservationConfidence, OperationObservation, ParamIdentity, RunMetadata, RuntimeTrace,
    RuntimeTraceWriter, TensorConflictKind, TensorObservation, SCHEMA,
};

fn sample_document_json() -> String {
    serde_json::json!({
        "schema": SCHEMA,
        "run": {
            "entrypoint": "train",
            "profile": "release",
            "cargo_features": ["cuda", "metal"],
            "cfg": ["debug_assertions"]
        },
        "tensors": [
            {
                "event_id": "t2",
                "static_id": "loss",
                "source": "model.rs:20",
                "shape": [1],
                "dtype": "F32",
                "device": "cpu",
                "contiguous": true,
                "requires_grad": true
            },
            {
                "event_id": "t1",
                "static_id": "w",
                "shape": [4, 4],
                "dtype": "BF16",
                "device": "cuda:0",
                "contiguous": true,
                "requires_grad": true,
                "storage_id": "s1"
            }
        ],
        "operations": [
            {
                "event_id": "o1",
                "op": "matmul",
                "inputs": ["s1", "s2"],
                "output": "s3"
            }
        ],
        "gradients": [
            {
                "event_id": "g2",
                "root": "vb",
                "key": "bias",
                "state": "missing"
            },
            {
                "event_id": "g1",
                "root": "vb",
                "key": "weight",
                "state": "present",
                "norm": 1.5
            }
        ]
    })
    .to_string()
}

#[test]
fn parse_json_document_normalizes_and_queries() {
    let trace = parse_json(&sample_document_json()).unwrap();
    assert_eq!(trace.schema, SCHEMA);
    assert_eq!(trace.run.entrypoint, "train");
    assert_eq!(trace.run.cargo_features, vec!["cuda", "metal"]);
    assert!(trace.run.analysis_id.is_none());
    assert!(trace.run.build_id.is_none());

    // Deterministic sort by event_id for tensors.
    assert_eq!(trace.tensors[0].event_id, "t1");
    assert_eq!(trace.tensors[1].event_id, "t2");

    // Gradients sorted by root then key.
    assert_eq!(trace.gradients[0].key, "bias");
    assert_eq!(trace.gradients[1].key, "weight");

    let w = trace.tensors_by_static_id("w");
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].dtype, "BF16");
    assert_eq!(trace.tensor_confidence("w"), ObservationConfidence::Proven);

    let g = trace.gradient("vb", "weight").unwrap();
    assert_eq!(g.state, GradientState::Present);
    assert_eq!(g.norm, Some(1.5));
    assert!(trace.gradients_for("vb", "bias")[0].state == GradientState::Missing);
}

#[test]
fn parse_jsonl_events_assemble_trace() {
    let jsonl = r#"
{"kind":"meta","schema":"candle-graph/runtime/1","entrypoint":"Model::forward","profile":"debug","cargo_features":["b"],"cfg":["a"]}
{"kind":"tensor","event_id":"t1","static_id":"x","shape":[2,3],"dtype":"F32","device":"cpu","contiguous":true,"requires_grad":false}
{"kind":"operation","event_id":"o1","op":"add","inputs":["a","b"],"output":"c"}
{"kind":"gradient","event_id":"g1","root":"vb","key":"w","state":"zero","norm":0.0}
"#;
    let trace = parse_jsonl(jsonl).unwrap();
    assert_eq!(trace.run.entrypoint, "Model::forward");
    // cargo_features / cfg sorted
    assert_eq!(trace.run.cargo_features, vec!["b"]);
    assert_eq!(trace.run.cfg, vec!["a"]);
    assert_eq!(trace.tensors.len(), 1);
    assert_eq!(trace.operations[0].op, "add");
    assert_eq!(
        trace.gradient("vb", "w").unwrap().state,
        GradientState::Zero
    );
}

#[test]
fn streaming_writer_emits_importable_static_ids_and_gradients() {
    let run = RunMetadata {
        entrypoint: "model::step".into(),
        profile: "debug".into(),
        cargo_features: vec!["cpu".into()],
        cfg: vec![],
        analysis_id: Some("analysis:candle-graph%2Fbuild%2F1%3Aabc".into()),
        build_id: Some("candle-graph/build/1:abc".into()),
    };
    let mut writer = RuntimeTraceWriter::new(Vec::new(), run).unwrap();
    writer
        .tensor(TensorObservation {
            event_id: "step-0-output".into(),
            static_id: Some("tensor:function%3Amodel%3Aoutput".into()),
            source: Some("model::step output".into()),
            shape: vec![2, 8],
            dtype: "F32".into(),
            device: "cpu".into(),
            contiguous: true,
            requires_grad: true,
            storage_id: None,
            step: None,
        })
        .unwrap();
    writer
        .operation(OperationObservation {
            event_id: "step-0-head".into(),
            op: "Linear::forward".into(),
            static_id: Some("operation:function%3Amodel%3Ahead".into()),
            source: Some("model::step head".into()),
            inputs: vec!["step-0-input".into()],
            output: Some("step-0-output".into()),
        })
        .unwrap();
    writer
        .gradient(GradientFact {
            event_id: "step-0-weight-grad".into(),
            root: "train_vb".into(),
            key: "head.weight".into(),
            state: GradientState::Present,
            norm: Some(0.25),
            step: None,
        })
        .unwrap();

    let bytes = writer.finish().unwrap();
    let trace = parse_jsonl(std::str::from_utf8(&bytes).unwrap()).unwrap();
    assert_eq!(
        trace.run.build_id.as_deref(),
        Some("candle-graph/build/1:abc")
    );
    assert_eq!(
        trace.tensors[0].static_id.as_deref(),
        Some("tensor:function%3Amodel%3Aoutput")
    );
    assert_eq!(
        trace.gradient("train_vb", "head.weight").unwrap().state,
        GradientState::Present
    );
}

#[test]
fn streaming_writer_rejects_duplicate_ids_and_invalid_gradients() {
    let run = RunMetadata {
        entrypoint: "model::step".into(),
        profile: "debug".into(),
        cargo_features: vec![],
        cfg: vec![],
        analysis_id: None,
        build_id: None,
    };
    let mut writer = RuntimeTraceWriter::new(Vec::new(), run).unwrap();
    let tensor = TensorObservation {
        event_id: "event-1".into(),
        static_id: None,
        source: None,
        shape: vec![1],
        dtype: "F32".into(),
        device: "cpu".into(),
        contiguous: true,
        requires_grad: false,
        storage_id: None,
        step: None,
    };
    writer.tensor(tensor.clone()).unwrap();
    assert!(writer
        .tensor(tensor)
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    assert!(writer
        .gradient(GradientFact {
            event_id: "event-2".into(),
            root: "vb".into(),
            key: "weight".into(),
            state: GradientState::Present,
            norm: Some(0.0),
            step: None,
        })
        .unwrap_err()
        .to_string()
        .contains("zero norm"));
}

#[test]
fn parse_auto_detects_json_and_jsonl() {
    let doc = parse(&sample_document_json()).unwrap();
    assert_eq!(doc.tensors.len(), 2);

    let jsonl = format!(
        "{}\n{}",
        r#"{"kind":"meta","schema":"candle-graph/runtime/1","entrypoint":"e","profile":"debug"}"#,
        r#"{"kind":"tensor","event_id":"t1","shape":[1],"dtype":"F32","device":"cpu","contiguous":true,"requires_grad":false}"#
    );
    let stream = parse(&jsonl).unwrap();
    assert_eq!(stream.tensors.len(), 1);
}

#[test]
fn duplicate_event_ids_are_rejected() {
    let bad = serde_json::json!({
        "schema": SCHEMA,
        "run": {"entrypoint": "train", "profile": "release"},
        "tensors": [{
            "event_id": "dup",
            "shape": [1],
            "dtype": "F32",
            "device": "cpu",
            "contiguous": true,
            "requires_grad": false
        }],
        "gradients": [{
            "event_id": "dup",
            "root": "vb",
            "key": "w",
            "state": "present",
            "norm": 1.0
        }]
    })
    .to_string();
    let err = parse_json(&bad).unwrap_err().to_string();
    assert!(
        err.contains("duplicate event_id"),
        "expected duplicate rejection, got: {err}"
    );
}

#[test]
fn invalid_gradient_facts_are_rejected() {
    let missing_with_norm = serde_json::json!({
        "schema": SCHEMA,
        "run": {"entrypoint": "e", "profile": "debug"},
        "gradients": [{
            "event_id": "g1",
            "root": "vb",
            "key": "w",
            "state": "missing",
            "norm": 1.0
        }]
    })
    .to_string();
    let err = parse_json(&missing_with_norm).unwrap_err().to_string();
    assert!(
        err.contains("invalid gradient fact") || err.contains("missing state"),
        "got: {err}"
    );

    let present_zero = serde_json::json!({
        "schema": SCHEMA,
        "run": {"entrypoint": "e", "profile": "debug"},
        "gradients": [{
            "event_id": "g1",
            "root": "vb",
            "key": "w",
            "state": "present",
            "norm": 0.0
        }]
    })
    .to_string();
    let err = parse_json(&present_zero).unwrap_err().to_string();
    assert!(
        err.contains("present state cannot have zero norm") || err.contains("invalid gradient"),
        "got: {err}"
    );

    let non_finite_finite = serde_json::json!({
        "schema": SCHEMA,
        "run": {"entrypoint": "e", "profile": "debug"},
        "gradients": [{
            "event_id": "g1",
            "root": "vb",
            "key": "w",
            "state": "non_finite",
            "norm": 3.0
        }]
    })
    .to_string();
    let err = parse_json(&non_finite_finite).unwrap_err().to_string();
    assert!(
        err.contains("non_finite") || err.contains("invalid gradient"),
        "got: {err}"
    );
}

#[test]
fn tensor_conflicts_and_gradient_audit() {
    let json = serde_json::json!({
        "schema": SCHEMA,
        "run": {
            "entrypoint": "train",
            "profile": "release",
            "cargo_features": [],
            "cfg": []
        },
        "tensors": [
            {
                "event_id": "t1",
                "static_id": "hidden",
                "shape": [8, 8],
                "dtype": "BF16",
                "device": "cuda:0",
                "contiguous": true,
                "requires_grad": true
            },
            {
                "event_id": "t2",
                "static_id": "hidden",
                "shape": [8, 16],
                "dtype": "F32",
                "device": "cpu",
                "contiguous": false,
                "requires_grad": true
            },
            {
                "event_id": "t3",
                "static_id": "ok",
                "shape": [1],
                "dtype": "F32",
                "device": "cpu",
                "contiguous": true,
                "requires_grad": false
            },
            {
                "event_id": "t4",
                "static_id": "ok",
                "shape": [1],
                "dtype": "F32",
                "device": "cpu",
                "contiguous": true,
                "requires_grad": false
            }
        ],
        "gradients": [
            {
                "event_id": "g1",
                "root": "vb",
                "key": "a.weight",
                "state": "missing"
            },
            {
                "event_id": "g2",
                "root": "vb",
                "key": "b.weight",
                "state": "non_finite"
            },
            {
                "event_id": "g3",
                "root": "other",
                "key": "c.weight",
                "state": "zero",
                "norm": 0.0
            },
            {
                "event_id": "g4",
                "root": "vb",
                "key": "d.weight",
                "state": "present",
                "norm": 2.0
            }
        ]
    })
    .to_string();

    let trace = parse_json(&json).unwrap();
    let audit = trace.audit();

    assert_eq!(
        audit.missing_gradients,
        vec![ParamIdentity {
            root: "vb".into(),
            key: "a.weight".into()
        }]
    );
    assert_eq!(
        audit.non_finite_gradients,
        vec![ParamIdentity {
            root: "vb".into(),
            key: "b.weight".into()
        }]
    );
    assert_eq!(
        audit.zero_gradients,
        vec![ParamIdentity {
            root: "other".into(),
            key: "c.weight".into()
        }]
    );

    assert_eq!(audit.tensor_conflicts.len(), 4);
    assert!(audit
        .tensor_conflicts
        .iter()
        .all(|c| { c.confidence == ObservationConfidence::Unknown }));
    assert!(audit
        .tensor_conflicts
        .iter()
        .any(|c| c.static_id == "hidden" && c.kind == TensorConflictKind::Dtype));
    assert!(audit
        .tensor_conflicts
        .iter()
        .any(|c| c.static_id == "hidden" && c.kind == TensorConflictKind::Device));
    assert!(audit
        .tensor_conflicts
        .iter()
        .any(|c| c.static_id == "hidden" && c.kind == TensorConflictKind::Layout));
    assert!(audit
        .tensor_conflicts
        .iter()
        .any(|c| c.static_id == "hidden" && c.kind == TensorConflictKind::Shape));
    assert!(!audit.tensor_conflicts.iter().any(|c| c.static_id == "ok"));
    assert!(trace.agreed_tensor("hidden").is_none());
    assert!(trace.agreed_tensor("ok").is_some());
    assert_eq!(
        trace.tensor_confidence("hidden"),
        ObservationConfidence::Unknown
    );
    assert!(!audit.is_clean());
    assert!(audit.has_unknown_evidence());
}

#[test]
fn contradictory_gradients_are_unknown_not_silently_chosen() {
    let json = serde_json::json!({
        "schema": SCHEMA,
        "run": {"entrypoint": "e", "profile": "debug"},
        "gradients": [
            {
                "event_id": "g1",
                "root": "vb",
                "key": "w",
                "state": "present",
                "norm": 1.0
            },
            {
                "event_id": "g2",
                "root": "vb",
                "key": "w",
                "state": "zero",
                "norm": 0.0
            }
        ]
    })
    .to_string();
    let trace = parse_json(&json).unwrap();
    assert!(
        trace.gradient("vb", "w").is_none(),
        "must not silently pick a contradictory gradient winner"
    );
    assert_eq!(
        trace.gradient_confidence("vb", "w"),
        ObservationConfidence::Unknown
    );
    let audit = trace.audit();
    assert_eq!(audit.gradient_conflicts.len(), 1);
    assert_eq!(
        audit.gradient_conflicts[0].identity,
        ParamIdentity {
            root: "vb".into(),
            key: "w".into()
        }
    );
    assert_eq!(
        audit.gradient_conflicts[0].confidence,
        ObservationConfidence::Unknown
    );
    assert!(audit.missing_gradients.is_empty());
    assert!(audit.zero_gradients.is_empty());
    assert!(audit.has_unknown_evidence());
}

#[test]
fn optional_identity_omitted_remains_compatible() {
    let trace = parse_json(&sample_document_json()).unwrap();
    let expected = ExpectedIdentity {
        analysis_id: Some("analysis:crate".into()),
        build_id: Some("pkg@1+feat".into()),
    };
    // Trace omitted identity fields → no comparison → compatible.
    assert!(trace.check_identity(&expected).is_empty());
    trace.require_identity(&expected).unwrap();
    let audit = trace.audit_with_identity(Some(&expected));
    assert!(audit.identity_mismatches.is_empty());
}

#[test]
fn supplied_identity_mismatch_is_reported_or_rejected() {
    let json = serde_json::json!({
        "schema": SCHEMA,
        "run": {
            "entrypoint": "train",
            "profile": "debug",
            "analysis_id": "analysis:other",
            "build_id": "pkg@1+cuda"
        }
    })
    .to_string();
    let trace = parse_json(&json).unwrap();
    assert_eq!(trace.run.analysis_id.as_deref(), Some("analysis:other"));
    assert_eq!(trace.run.build_id.as_deref(), Some("pkg@1+cuda"));

    let expected = ExpectedIdentity {
        analysis_id: Some("analysis:crate".into()),
        build_id: Some("pkg@1+cuda".into()),
    };
    let mismatches = trace.check_identity(&expected);
    assert_eq!(mismatches.len(), 1);
    assert_eq!(mismatches[0].field, IdentityField::AnalysisId);
    assert_eq!(mismatches[0].expected, "analysis:crate");
    assert_eq!(mismatches[0].observed, "analysis:other");

    let err = trace.require_identity(&expected).unwrap_err().to_string();
    assert!(
        err.contains("analysis_id") && err.contains("mismatch"),
        "got: {err}"
    );

    let audit = trace.audit_with_identity(Some(&expected));
    assert_eq!(audit.identity_mismatches.len(), 1);
    assert!(audit.has_unknown_evidence());

    // Matching identity is accepted.
    let ok = ExpectedIdentity {
        analysis_id: Some("analysis:other".into()),
        build_id: Some("pkg@1+cuda".into()),
    };
    trace.require_identity(&ok).unwrap();
}

#[test]
fn normalize_is_deterministic() {
    let mut a: RuntimeTrace = serde_json::from_str(&sample_document_json()).unwrap();
    let mut b: RuntimeTrace = serde_json::from_str(&sample_document_json()).unwrap();
    // Shuffle-like reverse then normalize both.
    a.tensors.reverse();
    a.gradients.reverse();
    b.tensors.reverse();
    b.gradients.reverse();
    a.normalize();
    b.normalize();
    assert_eq!(a, b);
    a.validate().unwrap();
}
