use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_graph::discover::{self, ScanOptions};
use candle_graph::model_ir::{BuilderRole, Confidence, ParameterRole, TensorRole};
use candle_graph::query::{self, QueryKind, QueryRequest};
use candle_graph::runtime::{self, ExpectedIdentity, ObservationConfidence, SCHEMA};
use candle_graph::verify;

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("candle-graph-model-ir-{unique}"));
        fs::create_dir_all(root.join("src")).unwrap();
        for package in ["candle-core", "candle-nn"] {
            let stub = root.join(package);
            fs::create_dir_all(stub.join("src")).unwrap();
            fs::write(
                stub.join("Cargo.toml"),
                format!(
                    "[package]\nname = \"{package}\"\nversion = \"0.11.0\"\nedition = \"2021\"\n"
                ),
            )
            .unwrap();
            fs::write(stub.join("src/lib.rs"), "").unwrap();
        }
        fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "synthetic-model"
version = "0.1.0"
edition = "2021"

[features]
default = ["cuda"]
cuda = []

[dependencies]
candle-core = { path = "candle-core" }
candle-nn = { path = "candle-nn" }
"#,
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), SOURCE).unwrap();
        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

const SOURCE: &str = r#"
pub struct VarBuilder;
pub struct VarMap;
pub struct Tensor;
pub struct Linear;
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;

impl VarBuilder {
    pub fn pp(&self, _name: &str) -> Self { Self }
    pub fn from_varmap(_varmap: &VarMap) -> Self { Self }
}
impl VarMap {
    pub fn all_vars(&self) -> Vec<Tensor> { vec![] }
}

pub mod model {
    use super::*;

    pub struct Encoder {
        projection: Linear,
    }
    impl Encoder {
        pub fn new(vb: VarBuilder) -> Result<Self> {
            let projection = candle_nn::linear(4, 8, vb.pp("projection"))?;
            Ok(Self { projection })
        }
        pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
            let (batch, tokens) = ids.dims2()?;
            ids.reshape((batch, tokens, 8))
        }
        pub fn apply_projection(&self, input: &Tensor) -> Result<Tensor> {
            self.projection.forward(input)
        }
    }

    pub struct Bridge {
        projection: Linear,
    }
    impl Bridge {
        pub fn new(train_vb: VarBuilder) -> Result<Self> {
            let projection = candle_nn::linear(8, 16, train_vb.pp("adapter"))?;
            Ok(Self { projection })
        }
        pub fn conditioning(&self, states: &Tensor) -> Result<Tensor> {
            let (batch, slots, hidden) = states.dims3()?;
            states.reshape((batch, slots, hidden))
        }
    }

    pub struct ArbitrarilyNamedModel {
        projection: Linear,
    }
    impl ArbitrarilyNamedModel {
        pub fn new(vb: VarBuilder) -> Result<Self> {
            let projection = candle_nn::linear(16, 16, vb.pp("projection"))?;
            Ok(Self { projection })
        }
        pub fn calculate(&self, input: &Tensor) -> Result<Tensor> {
            input.reshape((1, 16))
        }
    }
}

pub use model::{ArbitrarilyNamedModel, Bridge, Encoder};

// Neither spelling is semantic: the public tensor signature is a candidate boundary, while the
// function called `forward` has no tensor boundary.
pub fn unusual_public_boundary(input: &Tensor) -> Result<Tensor> {
    input.reshape((1, 4))
}
pub fn forward() {}

pub fn run_pipeline() {
    try_run_encoder();
    try_run_bridge();
}

pub fn try_run_encoder() {
    let train_varmap = VarMap;
    let vb = VarBuilder::from_varmap(&train_varmap);
    let _encoder = Encoder::new(vb);
    let vars = train_varmap
        .all_vars()
        .into_iter()
        .filter(|name| !name.contains("running_mean"))
        .collect::<Vec<_>>();
    AdamW::new(vars);
    save("artifacts/latent/model.safetensors");
}

pub fn try_run_bridge() {
    load("artifacts/latent/model.safetensors");
    save("artifacts/bridge/context.safetensors");
}
"#;

#[test]
fn source_names_and_literals_do_not_create_unproven_semantic_facts() {
    let fixture = Fixture::new();
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();

    assert_eq!(
        model
            .components
            .iter()
            .map(|component| component.name.as_str())
            .collect::<Vec<_>>(),
        ["ArbitrarilyNamedModel", "Bridge", "Encoder"]
    );
    assert!(model.functions.iter().any(|function| {
        function.qualified_name == "unusual_public_boundary" && function.is_entrypoint
    }));
    assert!(model
        .functions
        .iter()
        .any(|function| function.qualified_name == "forward" && !function.is_entrypoint));
    assert!(
        model.stages.is_empty(),
        "`run_pipeline` and `try_run_*` names do not prove stages"
    );
    assert!(
        model.artifacts.is_empty(),
        "a filename literal passed to an unresolved `save`/`load` does not prove an artifact"
    );
    assert!(
        model.optimizers.is_empty(),
        "`all_vars` and `AdamW` spellings do not prove optimizer identity or membership"
    );
    assert!(model.architecture_edges.is_empty());
    assert!(model
        .parameters
        .iter()
        .all(|parameter| parameter.role != ParameterRole::Optimized));
    assert!(model.findings.iter().any(|finding| {
        finding.rule == "compiler-semantic-evidence"
            && finding.confidence == candle_graph::model_ir::Confidence::Unknown
    }));
    assert!(model
        .tensors
        .iter()
        .any(|tensor| tensor.role == TensorRole::Input && tensor.shape.rank == Some(2)));
    assert!(model
        .cargo
        .as_ref()
        .unwrap()
        .active_features
        .contains(&"cuda".to_string()));
    assert!(model
        .components
        .iter()
        .flat_map(|component| &component.builders)
        .all(|builder| builder.role == BuilderRole::Unknown));

    let heuristic_model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            heuristic_architecture: true,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    assert!(!heuristic_model.stages.is_empty());
    assert!(!heuristic_model.artifacts.is_empty());
    assert!(!heuristic_model.optimizers.is_empty());
    assert!(heuristic_model
        .stages
        .iter()
        .flat_map(|stage| &stage.evidence)
        .all(|evidence| evidence.confidence == Confidence::Heuristic));
    assert!(heuristic_model
        .artifacts
        .iter()
        .flat_map(|artifact| &artifact.evidence)
        .all(|evidence| evidence.confidence == Confidence::Heuristic));
    assert!(heuristic_model
        .optimizers
        .iter()
        .flat_map(|optimizer| &optimizer.evidence)
        .all(|evidence| evidence.confidence == Confidence::Heuristic));
}

#[test]
fn analysis_identity_tracks_the_exact_cargo_configuration() {
    let fixture = Fixture::new();
    let with_defaults = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let mut without_defaults_options = ScanOptions {
        dataflow: false,
        ..ScanOptions::default()
    };
    without_defaults_options.cargo.no_default_features = true;
    let without_defaults = discover::analyze(fixture.path(), &without_defaults_options).unwrap();

    let with_defaults_build = &with_defaults.cargo.as_ref().unwrap().build_id;
    let without_defaults_build = &without_defaults.cargo.as_ref().unwrap().build_id;
    assert!(with_defaults_build.starts_with("candle-graph/build/1:"));
    assert_ne!(with_defaults_build, without_defaults_build);
    assert_ne!(with_defaults.analysis_id, without_defaults.analysis_id);
}

#[test]
fn query_api_is_bounded_and_uses_qualified_selectors() {
    let fixture = Fixture::new();
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();

    let mut request = QueryRequest::new(QueryKind::Entrypoints);
    request.selector = Some("model::Encoder::forward".to_string());
    request.limit = 1;
    let response = query::execute(&model, &request).unwrap();
    assert_eq!(response.total, 1);
    assert_eq!(response.returned, 1);

    let summary = query::execute(&model, &QueryRequest::new(QueryKind::Summary)).unwrap();
    assert_eq!(summary.returned, 1);
    assert_eq!(summary.items[0]["coverage"]["components"], 3);
}

#[test]
fn query_progressive_disclosure_omits_tensor_detail_until_narrowed() {
    let fixture = Fixture::new();
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();

    let summary = query::execute(&model, &QueryRequest::new(QueryKind::Summary)).unwrap();
    assert!(summary.items[0].get("drill_down").is_some());
    assert!(summary.items[0].get("tensors").is_none());
    assert!(summary.items[0].get("evidence").is_none());

    let architecture = query::execute(&model, &QueryRequest::new(QueryKind::Architecture)).unwrap();
    assert!(architecture.items[0]["entrypoints"].is_number());
    assert!(architecture.items[0].get("tensors").is_none());

    let functions = query::execute(&model, &QueryRequest::new(QueryKind::Functions)).unwrap();
    assert!(functions.total >= 1);
    for item in &functions.items {
        assert!(item["tensor_inputs"].is_number(), "{item}");
        assert!(item["drill_down"].is_array());
        assert!(item.get("evidence").is_none());
        assert!(item.get("shape").is_none());
    }

    let tensors = query::execute(&model, &QueryRequest::new(QueryKind::Tensors)).unwrap();
    assert!(tensors.total >= 1);
    for item in &tensors.items {
        assert!(item.get("evidence").is_none(), "listing must omit evidence");
        assert!(item.get("shape").is_none());
        assert!(item["shape_rank"].is_number() || item["shape_rank"].is_null());
        assert!(item["drill_down"].is_array());
    }

    let tensor_id = tensors.items[0]["id"].as_str().unwrap().to_string();
    let mut detail = QueryRequest::new(QueryKind::Tensor);
    detail.selector = Some(tensor_id.clone());
    let detailed = query::execute(&model, &detail).unwrap();
    assert_eq!(detailed.returned, 1);
    assert!(detailed.items[0]["evidence"].is_array());
    assert!(detailed.items[0]["shape"].is_object());

    let findings = query::execute(&model, &QueryRequest::new(QueryKind::Findings)).unwrap();
    assert!(findings.total >= 1);
    assert!(findings.items[0].get("evidence").is_none());
    assert!(findings.items[0]["drill_down"].is_array());
    let finding_id = findings.items[0]["id"].as_str().unwrap().to_string();
    let mut narrow = QueryRequest::new(QueryKind::Findings);
    narrow.selector = Some(finding_id);
    let detailed_finding = query::execute(&model, &narrow).unwrap();
    assert_eq!(detailed_finding.returned, 1);
    assert!(detailed_finding.items[0]["evidence"].is_array());

    let doctor = query::execute(&model, &QueryRequest::new(QueryKind::Doctor)).unwrap();
    assert!(doctor.items[0]["trust"].is_object());
    assert!(doctor.items[0]["finding_counts_by_rule"].is_object());
    assert!(doctor.items[0].get("tensors").is_none());
}

#[test]
fn query_offset_paginates_deterministically() {
    let fixture = Fixture::new();
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();

    let all = query::execute(&model, &QueryRequest::new(QueryKind::Functions)).unwrap();
    assert!(all.total >= 2, "fixture should expose multiple functions");

    let mut page = QueryRequest::new(QueryKind::Functions);
    page.limit = 1;
    page.offset = 0;
    let first = query::execute(&model, &page).unwrap();
    assert_eq!(first.returned, 1);
    assert_eq!(first.offset, 0);
    assert!(first.truncated);
    assert_eq!(first.items[0]["id"], all.items[0]["id"]);

    page.offset = 1;
    let second = query::execute(&model, &page).unwrap();
    assert_eq!(second.returned, 1);
    assert_eq!(second.offset, 1);
    assert_eq!(second.items[0]["id"], all.items[1]["id"]);
    assert_ne!(first.items[0]["id"], second.items[0]["id"]);
}

#[test]
fn cargo_scan_excludes_unselected_tests_examples_and_benches() {
    let fixture = Fixture::new();
    for (directory, name) in [
        ("tests", "test_decoy"),
        ("examples", "example_decoy"),
        ("benches", "bench_decoy"),
    ] {
        fs::create_dir_all(fixture.path().join(directory)).unwrap();
        fs::write(
            fixture.path().join(directory).join("decoy.rs"),
            format!("pub fn {name}(value: Tensor) -> Tensor {{ value }}"),
        )
        .unwrap();
    }

    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let names = model
        .functions
        .iter()
        .map(|function| function.name.as_str())
        .collect::<Vec<_>>();
    assert!(!names.contains(&"test_decoy"));
    assert!(!names.contains(&"example_decoy"));
    assert!(!names.contains(&"bench_decoy"));
}

#[test]
fn parameterized_module_forward_links_parameter_uses() {
    let fixture = Fixture::new();
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: true,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let encoder = &model
        .components
        .iter()
        .find(|component| component.name == "Encoder")
        .unwrap()
        .id;

    let projection = model
        .parameters
        .iter()
        .filter(|parameter| {
            &parameter.component == encoder && parameter.key.starts_with("projection.")
        })
        .collect::<Vec<_>>();
    assert!(!projection.is_empty());
    assert!(
        projection
            .iter()
            .all(|parameter| !parameter.uses.is_empty()),
        "known Linear::forward should link its implicit weight/bias reads"
    );
    assert!(model.coverage.linked_parameter_uses >= projection.len());

    let function_id = model
        .functions
        .iter()
        .find(|function| {
            function
                .qualified_name
                .ends_with("Encoder::apply_projection")
        })
        .unwrap()
        .id
        .0
        .clone();
    let mut operations_request = QueryRequest::new(QueryKind::Operations);
    operations_request.selector = Some(function_id);
    let operations = query::execute(&model, &operations_request).unwrap();
    assert!(!operations.items.is_empty());
    assert!(operations.items[0]["inputs"].is_number());
    assert!(operations.items[0].get("evidence").is_none());

    let mut detail = QueryRequest::new(QueryKind::Operation);
    detail.selector = Some(operations.items[0]["id"].as_str().unwrap().to_string());
    let operation = query::execute(&model, &detail).unwrap();
    assert!(operation.items[0]["evidence"].is_array());
}

#[test]
fn runtime_trace_refines_static_contracts_and_audits_gradients() {
    let fixture = Fixture::new();
    let static_model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let tensor = static_model.tensors.first().unwrap();
    let parameter = static_model.parameters.first().unwrap();
    let trace_path = fixture.path().join("runtime.json");
    let trace = serde_json::json!({
        "schema": "candle-graph/runtime/1",
        "run": {
            "entrypoint": "model::Encoder::forward",
            "profile": "debug",
            "cargo_features": ["cuda"],
            "cfg": ["feature=\"cuda\""]
        },
        "tensors": [{
            "event_id": "tensor-1",
            "static_id": tensor.id.0,
            "shape": [2, 32, 8],
            "dtype": "F32",
            "device": "cuda:0",
            "contiguous": true,
            "requires_grad": true
        }],
        "operations": [],
        "gradients": [{
            "event_id": "grad-1",
            "root": parameter.builder_root,
            "key": parameter.key,
            "state": "zero",
            "norm": 0.0
        }]
    });
    fs::write(&trace_path, serde_json::to_vec_pretty(&trace).unwrap()).unwrap();

    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            runtime_trace: Some(trace_path),
            ..ScanOptions::default()
        },
    )
    .unwrap();

    let observed = model
        .tensors
        .iter()
        .find(|candidate| candidate.id == tensor.id)
        .unwrap();
    assert_eq!(observed.shape.rank, Some(3));
    assert_eq!(observed.dtype, "F32");
    assert_eq!(model.runtime.as_ref().unwrap().zero_gradients, 1);
    assert!(model
        .findings
        .iter()
        .any(|finding| finding.rule == "runtime-gradient"));
}

#[test]
fn runtime_build_identity_mismatch_prevents_refinement() {
    let fixture = Fixture::new();
    let static_model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let tensor = static_model.tensors.first().unwrap();
    let trace_path = fixture.path().join("wrong-build.runtime.jsonl");
    let trace = format!(
        "{}\n{}\n",
        serde_json::json!({
            "kind": "meta",
            "schema": SCHEMA,
            "entrypoint": "model::Encoder::forward",
            "profile": "debug",
            "analysis_id": static_model.analysis_id.0,
            "build_id": "candle-graph/build/1:wrong"
        }),
        serde_json::json!({
            "kind": "tensor",
            "event_id": "wrong-build-tensor",
            "static_id": tensor.id.0,
            "shape": [999],
            "dtype": "F64",
            "device": "cpu",
            "contiguous": true,
            "requires_grad": false
        })
    );
    fs::write(&trace_path, trace).unwrap();

    let merged = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            runtime_trace: Some(trace_path),
            ..ScanOptions::default()
        },
    )
    .unwrap();
    assert_eq!(merged.runtime.as_ref().unwrap().identity_mismatches, 1);
    assert!(merged.findings.iter().any(|finding| {
        finding.rule == "runtime-identity" && finding.message.contains("build_id mismatch")
    }));
    assert_ne!(
        merged
            .tensors
            .iter()
            .find(|candidate| candidate.id == tensor.id)
            .unwrap()
            .dtype,
        "F64"
    );
}

#[test]
fn runtime_conflicts_and_identity_are_unknown_not_proven_winners() {
    let fixture = Fixture::new();
    let static_model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let tensor = static_model.tensors.first().unwrap();
    let parameter = static_model.parameters.first().unwrap();

    let conflicting = serde_json::json!({
        "schema": SCHEMA,
        "run": {
            "entrypoint": "model::Encoder::forward",
            "profile": "debug",
            "analysis_id": "analysis:wrong-crate",
            "build_id": "synthetic-model@0.1.0+cuda/debug"
        },
        "tensors": [
            {
                "event_id": "tensor-a",
                "static_id": tensor.id.0,
                "shape": [2, 32, 8],
                "dtype": "F32",
                "device": "cuda:0",
                "contiguous": true,
                "requires_grad": true
            },
            {
                "event_id": "tensor-b",
                "static_id": tensor.id.0,
                "shape": [2, 16, 8],
                "dtype": "BF16",
                "device": "cpu",
                "contiguous": false,
                "requires_grad": false
            }
        ],
        "gradients": [
            {
                "event_id": "grad-a",
                "root": parameter.builder_root,
                "key": parameter.key,
                "state": "present",
                "norm": 1.0
            },
            {
                "event_id": "grad-b",
                "root": parameter.builder_root,
                "key": parameter.key,
                "state": "zero",
                "norm": 0.0
            }
        ]
    })
    .to_string();

    let trace = runtime::parse_json(&conflicting).unwrap();
    assert!(
        trace.agreed_tensor(&tensor.id.0).is_none(),
        "contradictory tensor observations must not yield a silent winner"
    );
    assert_eq!(
        trace.tensor_confidence(&tensor.id.0),
        ObservationConfidence::Unknown
    );
    assert!(
        trace
            .gradient(&parameter.builder_root, &parameter.key)
            .is_none(),
        "contradictory gradient observations must not yield a silent winner"
    );
    assert_eq!(
        trace.gradient_confidence(&parameter.builder_root, &parameter.key),
        ObservationConfidence::Unknown
    );

    let expected = ExpectedIdentity {
        analysis_id: Some(static_model.analysis_id.0.clone()),
        build_id: Some("synthetic-model@0.1.0+cuda/debug".into()),
    };
    let audit = trace.audit_with_identity(Some(&expected));
    assert!(!audit.tensor_conflicts.is_empty());
    assert!(!audit.gradient_conflicts.is_empty());
    assert_eq!(audit.identity_mismatches.len(), 1);
    assert!(audit
        .tensor_conflicts
        .iter()
        .all(|c| c.confidence == ObservationConfidence::Unknown));
    assert!(audit
        .gradient_conflicts
        .iter()
        .all(|c| c.confidence == ObservationConfidence::Unknown));
    assert!(audit.has_unknown_evidence());
    assert!(trace.require_identity(&expected).is_err());

    // Omitted optional identity remains backward compatible with an expected identity.
    let legacy = serde_json::json!({
        "schema": SCHEMA,
        "run": {
            "entrypoint": "model::Encoder::forward",
            "profile": "debug"
        },
        "tensors": [{
            "event_id": "tensor-1",
            "static_id": tensor.id.0,
            "shape": [2, 32, 8],
            "dtype": "F32",
            "device": "cuda:0",
            "contiguous": true,
            "requires_grad": true
        }],
        "gradients": []
    })
    .to_string();
    let legacy_trace = runtime::parse_json(&legacy).unwrap();
    legacy_trace.require_identity(&expected).unwrap();
    assert_eq!(
        legacy_trace.tensor_confidence(&tensor.id.0),
        ObservationConfidence::Proven
    );
}

#[test]
fn safetensors_header_rejects_bad_dimensions_and_offsets() {
    let fixture = Fixture::new();
    let path = fixture.path().join("bad.safetensors");

    let oversized = serde_json::to_vec(&serde_json::json!({
        "layer.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 8]
        }
    }))
    .unwrap();
    let mut file = Vec::with_capacity(8 + oversized.len() + 8);
    file.extend_from_slice(&(oversized.len() as u64).to_le_bytes());
    file.extend_from_slice(&oversized);
    file.extend_from_slice(&[0u8; 8]);
    fs::write(&path, &file).unwrap();
    let err = format!("{:#}", verify::read_header(&path).unwrap_err());
    assert!(
        err.contains("data_offsets") || err.contains("requires"),
        "expected size mismatch, got: {err}"
    );

    let past_eof = serde_json::to_vec(&serde_json::json!({
        "layer.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    }))
    .unwrap();
    let mut short = Vec::with_capacity(8 + past_eof.len());
    short.extend_from_slice(&(past_eof.len() as u64).to_le_bytes());
    short.extend_from_slice(&past_eof);
    // No data payload — offsets claim 24 bytes.
    fs::write(&path, &short).unwrap();
    let err = format!("{:#}", verify::read_header(&path).unwrap_err());
    assert!(
        err.contains("exceeds data region") || err.contains("data_offsets"),
        "expected bounds rejection, got: {err}"
    );

    let valid = serde_json::to_vec(&serde_json::json!({
        "__metadata__": {"format": "pt"},
        "layer.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    }))
    .unwrap();
    let mut ok = Vec::with_capacity(8 + valid.len() + 24);
    ok.extend_from_slice(&(valid.len() as u64).to_le_bytes());
    ok.extend_from_slice(&valid);
    ok.extend_from_slice(&[0u8; 24]);
    fs::write(&path, &ok).unwrap();
    let header = verify::read_header(&path).unwrap();
    assert_eq!(header["layer.weight"].shape, vec![2, 3]);
    assert_eq!(header["layer.weight"].dtype, "F32");
}

#[test]
fn composition_modules_and_doctor_support_modular_agent_workflows() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("candle-graph-composition-{unique}"));
    fs::create_dir_all(root.join("src")).unwrap();
    for package in ["candle-core", "candle-nn"] {
        let stub = root.join(package);
        fs::create_dir_all(stub.join("src")).unwrap();
        fs::write(
            stub.join("Cargo.toml"),
            format!("[package]\nname = \"{package}\"\nversion = \"0.11.0\"\nedition = \"2021\"\n"),
        )
        .unwrap();
        fs::write(stub.join("src/lib.rs"), "").unwrap();
    }
    fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "composition-fixture"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { path = "candle-core" }
candle-nn = { path = "candle-nn" }
"#,
    )
    .unwrap();
    fs::write(
        root.join("src/lib.rs"),
        r#"
pub struct VarBuilder;
pub struct Tensor;
pub struct Linear;
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;
impl VarBuilder { pub fn pp(&self, _name: &str) -> Self { Self } }

pub mod model {
    use super::*;
    pub struct Encoder { head: Linear }
    impl Encoder {
        pub fn new(vb: VarBuilder) -> Result<Self> {
            Ok(Self { head: candle_nn::linear(4, 8, vb.pp("head"))? })
        }
        pub fn forward(&self, input: &Tensor) -> Result<Tensor> { Ok(input.clone()) }
    }
    pub struct Bridge { head: Linear }
    impl Bridge {
        pub fn new(vb: VarBuilder) -> Result<Self> {
            Ok(Self { head: candle_nn::linear(8, 8, vb.pp("head"))? })
        }
        pub fn forward(&self, input: &Tensor) -> Result<Tensor> { Ok(input.clone()) }
    }
    pub struct Stack { encoder: Encoder, bridge: Bridge }
    impl Stack {
        pub fn new(vb: VarBuilder) -> Result<Self> {
            Ok(Self {
                encoder: Encoder::new(vb.pp("encoder"))?,
                bridge: Bridge::new(vb.pp("bridge"))?,
            })
        }
        pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.bridge.forward(&self.encoder.forward(input)?)
        }
    }
}
pub use model::{Bridge, Encoder, Stack};
"#,
    )
    .unwrap();

    let model = discover::analyze(
        &root,
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let _ = fs::remove_dir_all(&root);

    assert_eq!(model.coverage.composition_edges, 2);
    let composition = query::execute(&model, &QueryRequest::new(QueryKind::Composition)).unwrap();
    assert_eq!(composition.total, 2);
    assert_eq!(composition.items[0]["confidence"], "heuristic");

    let mut modules = QueryRequest::new(QueryKind::Modules);
    modules.selector = Some("Stack".into());
    let modules = query::execute(&model, &modules).unwrap();
    assert!(modules.total >= 1);
    assert_eq!(modules.items[0]["component_name"], "Stack");

    let doctor = query::execute(&model, &QueryRequest::new(QueryKind::Doctor)).unwrap();
    assert!(doctor.items[0]["coverage_quality"]["component_entrypoints"].is_number());
    assert!(doctor.items[0]["trust"]["actionable_warnings"].is_boolean());

    let entrypoints = query::execute(&model, &QueryRequest::new(QueryKind::Entrypoints)).unwrap();
    assert!(entrypoints
        .items
        .iter()
        .any(|item| item["is_component_entrypoint"] == true));
}

#[test]
fn subprocess_pipeline_discovers_orchestrator_and_cli_flags() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("candle-graph-pipeline-{unique}"));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "pipeline-fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::write(
        root.join("src/lib.rs"),
        r#"
use std::process::Command;

pub struct Placeholder;

fn run_stage_command(stage: &str, args: &[String]) {
    let executable = std::env::current_exe().unwrap();
    let mut command = Command::new(&executable);
    command.args(args);
    let _ = (stage, command);
}

fn train_encoder() {
    run_stage_command(
        "latent",
        &vec!["app".to_string(), "--latent".to_string(), "--output".to_string()],
    );
}

pub fn run_pipeline() {
    prepare_data();
    train_encoder();
}

fn prepare_data() {}
"#,
    )
    .unwrap();

    let conservative = discover::analyze(
        &root,
        &ScanOptions {
            dataflow: false,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    assert!(
        conservative.stages.is_empty(),
        "subprocess and stage names must remain gated by --heuristic-architecture"
    );

    let model = discover::analyze(
        &root,
        &ScanOptions {
            dataflow: false,
            heuristic_architecture: true,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let _ = fs::remove_dir_all(&root);

    assert!(model.coverage.subprocess_stages >= 1);
    let pipeline = query::execute(&model, &QueryRequest::new(QueryKind::Pipeline)).unwrap();
    assert!(pipeline.items[0]["subprocess_stages"].as_u64().unwrap() >= 1);
    let stages = query::execute(&model, &QueryRequest::new(QueryKind::Stages)).unwrap();
    assert!(stages.total >= 2);
    let latent = stages
        .items
        .iter()
        .find(|stage| stage["subprocess_key"] == "latent")
        .expect("latent subprocess stage");
    assert_eq!(latent["dispatch"], "subprocess");
    assert_eq!(latent["subprocess_key"], "latent");
    assert_eq!(latent["launcher"], "run_stage_command");
    assert!(model
        .stages
        .iter()
        .flat_map(|stage| &stage.evidence)
        .all(|evidence| evidence.confidence == Confidence::Heuristic));
}

#[test]
fn expanded_bce_with_logits_emits_proven_training_impact_finding() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("candle-graph-bce-{unique}"));
    fs::create_dir_all(root.join("src")).unwrap();
    for package in ["candle-core", "candle-nn"] {
        let stub = root.join(package);
        fs::create_dir_all(stub.join("src")).unwrap();
        fs::write(
            stub.join("Cargo.toml"),
            format!("[package]\nname = \"{package}\"\nversion = \"0.11.0\"\nedition = \"2021\"\n"),
        )
        .unwrap();
        fs::write(stub.join("src/lib.rs"), "").unwrap();
    }
    fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "bce-demo"
version = "0.1.0"
edition = "2021"
[dependencies]
candle-core = { path = "candle-core" }
candle-nn = { path = "candle-nn" }
"#,
    )
    .unwrap();
    fs::write(
        root.join("src/lib.rs"),
        r#"
pub struct Tensor;
pub struct Marker;
pub type Result<T> = std::result::Result<T, ()>;
pub fn q_loss(logit: Tensor, target: Tensor) -> Result<Tensor> {
    candle_nn::loss::binary_cross_entropy_with_logit(&logit, &target)
}
"#,
    )
    .unwrap();

    let model = discover::analyze(
        &root,
        &ScanOptions {
            dataflow: true,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let _ = fs::remove_dir_all(&root);

    assert!(
        model.findings.iter().any(|finding| {
            matches!(
                finding.rule.as_str(),
                "numeric-domain-violation" | "zero-times-infinity"
            ) && finding.severity == candle_graph::model_ir::FindingSeverity::Error
                && finding.confidence == Confidence::Proven
                && finding.message.contains("training impact")
                && finding
                    .message
                    .contains("expanded from candle-nn-0.11.0/src/loss.rs")
        }),
        "expected expanded-body training-impact finding, got {:#?}",
        model
            .findings
            .iter()
            .map(|f| (&f.rule, &f.severity, &f.message))
            .collect::<Vec<_>>()
    );
}
