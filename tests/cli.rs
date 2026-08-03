use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    path: std::path::PathBuf,
}

impl Fixture {
    fn new(source: &str) -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("candle-graph-cli-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("model.rs"), source).unwrap();
        Self { path }
    }

    /// Minimal Cargo package so `cargo candle-graph` can resolve features/cfg.
    fn cargo_package(source: &str) -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "candle-graph-cargo-cli-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(path.join("src")).unwrap();
        for package in ["candle-core", "candle-nn"] {
            let stub = path.join(package);
            std::fs::create_dir_all(stub.join("src")).unwrap();
            std::fs::write(
                stub.join("Cargo.toml"),
                format!(
                    "[package]\nname = \"{package}\"\nversion = \"0.11.0\"\nedition = \"2021\"\n"
                ),
            )
            .unwrap();
            std::fs::write(stub.join("src/lib.rs"), "").unwrap();
        }
        std::fs::write(
            path.join("Cargo.toml"),
            r#"
[package]
name = "cli-demo"
version = "0.1.0"
edition = "2021"

[features]
default = []
cuda = []

[dependencies]
candle-core = { path = "candle-core" }
candle-nn = { path = "candle-nn" }
"#,
        )
        .unwrap();
        std::fs::write(path.join("src/lib.rs"), source).unwrap();
        Self { path }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

const MODEL: &str = r#"
struct Root {
    head: Linear,
}

impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let head = nn::linear_no_bias(8, 8, vb.pp("head"))?;
        Ok(Self { head })
    }
}

fn alignment(positive: Tensor, logits: Tensor, labels: Tensor) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;
    let contrastive = candle_nn::loss::cross_entropy(&logits, &labels)?;
    positive.broadcast_add(&contrastive)
}
"#;

fn command(fixture: &Fixture) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_candle-graph"));
    command.arg(&fixture.path).args(["--root", "Root"]);
    command
}

#[test]
fn cli_html_json_and_canonical_baseline_work_together() {
    let fixture = Fixture::new(MODEL);
    let html = fixture.path.join("graph.html");
    let baseline = fixture.path.join("graph.baseline");

    let output = command(&fixture)
        .args([
            "--entry",
            "alignment",
            "--format",
            "html",
            "--output",
            html.to_str().unwrap(),
            "--update-baseline",
            baseline.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let html_text = std::fs::read_to_string(&html).unwrap();
    assert!(html_text.contains("data-viewer=\"candle-graph\""));
    assert!(html_text.contains("\"dtype_risks\""));
    let baseline_text = std::fs::read_to_string(&baseline).unwrap();
    assert!(baseline_text.contains("dtype-risk"));

    let output = command(&fixture)
        .args([
            "--entry",
            "alignment",
            "--format",
            "json",
            "--check",
            baseline.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["dataflow"]["coverage"]["dtype_risks"], 1);
    assert_eq!(json["dataflow"]["dtype_risks"][0]["op"], "broadcast_add");

    let output = command(&fixture)
        .args(["--entry", "alignment", "--strict"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("1 dtype risks"));

    std::fs::write(
        fixture.path.join("model.rs"),
        MODEL.replace("vb.pp(\"head\")", "vb.pp(\"renamed_head\")"),
    )
    .unwrap();
    let output = command(&fixture)
        .args(["--check", baseline.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("baseline mismatch"));
}

#[test]
fn cli_query_is_compact_by_default_with_limit_pagination() {
    let fixture = Fixture::new(MODEL);
    let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
        .arg(&fixture.path)
        .args(["--query", "summary", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["schema"], "candle-graph/query/1");
    assert!(summary["items"][0]["drill_down"].is_array());
    assert!(summary["items"][0].get("tensors").is_none());

    let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
        .arg(&fixture.path)
        .args(["--query", "functions", "--limit", "1", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let functions: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(functions["returned"], 1);
    assert!(functions["truncated"].as_bool().unwrap_or(false) || functions["total"] == 1);
    let item = &functions["items"][0];
    assert!(item["tensor_inputs"].is_number(), "{item}");
    assert!(item["drill_down"].is_array());
    assert!(item.get("evidence").is_none());
    if functions["total"].as_u64().unwrap_or(0) > 1 {
        let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
            .arg(&fixture.path)
            .args([
                "--query",
                "functions",
                "--limit",
                "1",
                "--offset",
                "1",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        let second: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(second["offset"], 1);
        assert_ne!(second["items"][0]["id"], functions["items"][0]["id"]);
    }

    let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
        .arg(&fixture.path)
        .args(["--query", "tensors", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tensors: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    if tensors["returned"].as_u64().unwrap_or(0) > 0 {
        let item = &tensors["items"][0];
        assert!(item.get("evidence").is_none(), "{item}");
        assert!(item.get("shape").is_none(), "{item}");
        assert!(item["drill_down"].is_array());
        let select = item["id"].as_str().unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_candle-graph"))
            .arg(&fixture.path)
            .args(["--query", "tensor", "--select", select, "--format", "json"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let detail: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(detail["returned"], 1);
        assert!(detail["items"][0]["evidence"].is_array());
        assert!(detail["items"][0]["shape"].is_object());
    }
}

const CARGO_MODEL: &str = r#"
pub struct VarBuilder;
impl VarBuilder {
    pub fn pp(&self, _name: &str) -> Self { Self }
}
pub struct Linear;
pub struct Tensor;
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod nn {
    use super::*;
    pub fn linear_no_bias(_in: usize, _out: usize, _vb: VarBuilder) -> Result<Linear> {
        Ok(Linear)
    }
}

pub struct Root {
    head: Linear,
}

impl Root {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let head = nn::linear_no_bias(8, 8, vb.pp("head"))?;
        Ok(Self { head })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        Ok(x)
    }
}
"#;

#[test]
fn cargo_candle_graph_check_emits_ir_and_notes_without_failing_on_information() {
    let fixture = Fixture::cargo_package(CARGO_MODEL);
    let output = Command::new(env!("CARGO_BIN_EXE_cargo-candle-graph"))
        .args(["check", fixture.path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let model: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(model["schema"], "candle-graph/model/1");
    assert!(!model["components"].as_array().unwrap().is_empty());
    assert!(model["functions"].is_array());
    assert!(model["findings"].is_array());
    assert!(model["cargo"].is_object());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("compiler-semantic-evidence") || stderr.contains("note:"),
        "{stderr}"
    );
    assert!(stderr.contains("0 error(s)"));
}

#[test]
fn cargo_candle_graph_query_and_feature_forwarding_work() {
    let fixture = Fixture::cargo_package(CARGO_MODEL);
    let output = Command::new(env!("CARGO_BIN_EXE_cargo-candle-graph"))
        .args([
            "query",
            "cargo",
            "--path",
            fixture.path.to_str().unwrap(),
            "--features",
            "cuda",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cargo: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cargo["schema"], "candle-graph/query/1");
    let item = &cargo["items"][0];
    let features = item
        .pointer("/context/active_features")
        .or_else(|| item.get("active_features"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(features.iter().any(|feature| feature == "cuda"), "{item}");
}

#[test]
fn cargo_candle_graph_model_baseline_round_trip() {
    let fixture = Fixture::cargo_package(CARGO_MODEL);
    let baseline = fixture.path.join("model.candle-graph-baseline");
    let output = Command::new(env!("CARGO_BIN_EXE_cargo-candle-graph"))
        .args([
            "check",
            fixture.path.to_str().unwrap(),
            "--update-baseline",
            baseline.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&baseline).unwrap();
    assert!(text.contains("candle-graph/model-baseline/1"));

    let output = Command::new(env!("CARGO_BIN_EXE_cargo-candle-graph"))
        .args([
            "check",
            fixture.path.to_str().unwrap(),
            "--check",
            baseline.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
