use std::sync::atomic::{AtomicU64, Ordering};

static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

use std::io::Cursor;

use candle_graph::{
    dataflow::{self, GradState},
    discover::{self, ScanOptions},
    phase::{entrypoint_phases, ExecutionPhase},
    profile::{ProfileConfig, ProfileSession},
    query::{self, QueryKind, QueryRequest},
    runtime::{self, EdgeTimingObservation, OperationObservation, RunMetadata, SCHEMA_V3},
};

#[test]
fn forward_entrypoint_gets_train_and_infer_graphs() {
    assert_eq!(
        entrypoint_phases("forward", "Model::forward", false),
        vec![ExecutionPhase::Train, ExecutionPhase::Infer]
    );
    assert_eq!(
        entrypoint_phases("eval_step", "model::eval_step", false),
        vec![ExecutionPhase::Infer]
    );
    assert_eq!(
        entrypoint_phases("leworld_loss", "train::leworld_loss", true),
        vec![ExecutionPhase::Train]
    );
}

#[test]
fn infer_phase_severs_trainable_grad_connectivity() {
    const SRC: &str = r#"
pub struct Tensor;
pub struct Var;
pub type Result<T> = std::result::Result<T, ()>;
impl Tensor {
    pub fn matmul(&self, _: &Tensor) -> Result<Tensor> { Ok(Tensor) }
    pub fn sum_all(&self) -> Result<Tensor> { Ok(Tensor) }
}
impl Var {
    pub fn as_tensor(&self) -> Tensor { Tensor }
}
pub fn forward(x: Var) -> Result<Tensor> {
    let t = x.as_tensor();
    let y = t.matmul(&t)?;
    y.sum_all()
}
"#;
    let fixture = write_temp_model(SRC);
    let krate = candle_graph::load::load(fixture.path()).unwrap();
    let train =
        dataflow::analyze_with_phase(&krate, "forward", None, ExecutionPhase::Train).unwrap();
    let infer =
        dataflow::analyze_with_phase(&krate, "forward", None, ExecutionPhase::Infer).unwrap();
    let train_active = train
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.grad,
                GradState::Differentiable | GradState::Trainable | GradState::LayoutDependent
            )
        })
        .count();
    let infer_active = infer
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.grad,
                GradState::Differentiable | GradState::Trainable | GradState::LayoutDependent
            )
        })
        .count();
    assert!(train_active >= infer_active);
    assert_eq!(infer_active, 0);
}

#[test]
fn profile_session_emits_timed_operations_and_edges() {
    let path = std::env::temp_dir().join(format!("candle-graph-profile-{}", std::process::id()));
    let mut session = ProfileSession::open(
        &path,
        ProfileConfig {
            analysis_id: Some("analysis:test".into()),
            ..ProfileConfig::train("Model::forward")
        },
    )
    .unwrap();
    let event = session
        .begin_operation("operation:train:Model::forward:0", "matmul", &[])
        .unwrap();
    std::thread::sleep(std::time::Duration::from_micros(100));
    let duration = session
        .end_operation(&event, Some("tensor-out".into()))
        .unwrap();
    assert!(duration > 0);
    session
        .record_edge_timing(
            "tensor:train:Model::forward:0",
            "operation:train:Model::forward:0",
            duration,
        )
        .unwrap();
    session.finish().unwrap();

    let bytes = std::fs::read_to_string(&path).unwrap();
    let trace = runtime::parse_jsonl(&bytes).unwrap();
    assert_eq!(trace.schema, SCHEMA_V3);
    assert_eq!(trace.run.phase.as_deref(), Some("train"));
    assert_eq!(trace.operations.len(), 1);
    assert!(trace.operations[0].duration_ns.unwrap() > 0);
    assert_eq!(trace.edge_timings.len(), 1);
    let _ = std::fs::remove_file(path);
}

#[test]
fn unified_scan_builds_distinct_phase_tensor_ids() {
    const SRC: &str = r#"
pub struct Tensor;
pub struct Var;
pub struct VarBuilder;
pub type Result<T> = std::result::Result<T, ()>;
impl Tensor {
    pub fn matmul(&self, _: &Tensor) -> Result<Tensor> { Ok(Tensor) }
}
pub struct Model;
impl Model {
    pub fn new(_: VarBuilder) -> Result<Self> { Ok(Model) }
    pub fn forward(&self, x: Tensor) -> Result<Tensor> { x.matmul(&x) }
}
"#;
    let fixture = write_temp_model(SRC);
    let model = discover::analyze(
        fixture.path(),
        &ScanOptions {
            dataflow: true,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    let train_tensors = model
        .tensors
        .iter()
        .filter(|tensor| tensor.execution_phase == Some(ExecutionPhase::Train))
        .count();
    let infer_tensors = model
        .tensors
        .iter()
        .filter(|tensor| tensor.execution_phase == Some(ExecutionPhase::Infer))
        .count();
    assert!(train_tensors > 0);
    assert!(infer_tensors > 0);

    let graph_train = query::execute(&model, &QueryRequest::new(QueryKind::GraphTrain)).unwrap();
    assert!(graph_train.items[0]["tensor_count"].as_u64().unwrap() > 0);
}

struct TempModel {
    root: std::path::PathBuf,
}

impl TempModel {
    fn path(&self) -> &std::path::Path {
        &self.root
    }
}

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn write_temp_model(source: &str) -> TempModel {
    let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("candle-graph-phase-{}-{id}", std::process::id()));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"phase-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(root.join("src/lib.rs"), source).unwrap();
    TempModel { root }
}

#[test]
fn runtime_v3_jsonl_roundtrip_includes_edge_timings() {
    let run = RunMetadata {
        entrypoint: "forward".into(),
        profile: "test".into(),
        cargo_features: vec![],
        cfg: vec![],
        analysis_id: None,
        build_id: None,
        phase: Some("infer".into()),
    };
    let mut writer =
        runtime::RuntimeTraceWriter::new_with_schema(Cursor::new(Vec::new()), SCHEMA_V3, run)
            .unwrap();
    writer
        .operation(OperationObservation {
            event_id: "op-1".into(),
            op: "matmul".into(),
            static_id: Some("operation:infer:forward:1".into()),
            source: None,
            inputs: vec![],
            output: None,
            step: Some(0),
            duration_ns: Some(42),
        })
        .unwrap();
    writer
        .edge_timing(EdgeTimingObservation {
            event_id: "edge-1".into(),
            from_static_id: "tensor:infer:forward:0".into(),
            to_static_id: "operation:infer:forward:1".into(),
            duration_ns: 42,
            step: Some(0),
        })
        .unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    let trace = runtime::parse_jsonl(std::str::from_utf8(&bytes).unwrap()).unwrap();
    assert_eq!(trace.operations[0].duration_ns, Some(42));
    assert_eq!(trace.edge_timings.len(), 1);
}
