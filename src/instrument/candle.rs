//! Candle tensor helpers — PyTorch `record_shapes` / `profile_memory` ergonomics for probes.
//!
//! Enable with the `candle` feature: `candle-graph = { path = "..", features = ["candle"] }`.

use anyhow::Context as _;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

use crate::capability::{ExpectedGradient, GradientContract, GradientFamilyContract};
use crate::instrument::SpanId;
use crate::instrument::{OpRecord, TensorRecord, TraceSession};
use crate::phase::ExecutionStep;
use crate::trace::memory::{category_for_step, dense_tensor_bytes};
use crate::trace::{GradientState, MemoryCategory};

/// Owned metadata captured from a Candle tensor (safe to store between probe calls).
///
/// Backend identity (`storage_id`, `tensor_id`) is read-only: it always comes
/// from the observed tensor. Semantic naming goes through [`Self::with_label`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleCapture {
    storage_id: String,
    tensor_id: String,
    pub label: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub tensor_bytes: u64,
    pub requires_grad: bool,
    pub category: MemoryCategory,
}

/// Owned op observation including output tensor metadata and input byte sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleOpCapture {
    pub op_name: String,
    pub inputs: Vec<String>,
    pub output: CandleCapture,
    pub input_tensor_bytes: u64,
    pub duration_ns: u64,
    pub timestamp_ns: u64,
}

/// Stable string form of [`Tensor::id`] for trace dedup.
pub fn tensor_id(t: &Tensor) -> String {
    format!("{id:?}", id = t.id())
}

/// Process-local identity of the backend storage shared by aliased tensor views.
pub fn storage_id(t: &Tensor) -> String {
    let (storage, _) = t.storage_and_layout();
    format!("storage:{:p}", &*storage)
}

/// Device label for JSONL (`cpu`, `cuda:0`, `metal:0`, …).
pub fn device_label(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "cpu".into(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
    }
}

/// Lowercase dtype label (`f32`, `bf16`, …).
pub fn dtype_label(dtype: DType) -> String {
    format!("{dtype:?}").to_ascii_lowercase()
}

/// Dense tensor footprint. This does not claim the size of shared backing storage.
pub fn tensor_dense_bytes(t: &Tensor) -> u64 {
    dense_tensor_bytes(t.dims(), &dtype_label(t.dtype()))
        .expect("every Candle dtype has a known byte width")
}

impl CandleCapture {
    pub fn from_tensor(t: &Tensor, step: Option<ExecutionStep>) -> Self {
        let requires_grad = t.is_variable();
        let category = if requires_grad {
            MemoryCategory::Parameter
        } else {
            category_for_step(step, false)
        };
        Self {
            storage_id: storage_id(t),
            tensor_id: tensor_id(t),
            label: None,
            shape: t.dims().to_vec(),
            dtype: dtype_label(t.dtype()),
            device: device_label(t.device()),
            tensor_bytes: tensor_dense_bytes(t),
            requires_grad,
            category,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Process-local identity of the backend storage observed at capture time.
    pub fn storage_id(&self) -> &str {
        &self.storage_id
    }

    /// Backend tensor identity observed at capture time.
    pub fn tensor_id(&self) -> &str {
        &self.tensor_id
    }
}

impl CandleOpCapture {
    pub fn new(
        op_name: impl Into<String>,
        inputs: Vec<String>,
        output: &Tensor,
        input_bytes: u64,
        duration_ns: u64,
        timestamp_ns: u64,
        step: Option<ExecutionStep>,
    ) -> Self {
        Self {
            op_name: op_name.into(),
            inputs,
            output: CandleCapture::from_tensor(output, step),
            input_tensor_bytes: input_bytes,
            duration_ns,
            timestamp_ns,
        }
    }
}

/// Sum dense footprints for input tensors (PyTorch input-shape-style metadata).
pub fn inputs_dense_bytes(tensors: &[&Tensor]) -> u64 {
    tensors.iter().map(|t| tensor_dense_bytes(t)).sum()
}

/// Record a Candle tensor via [`TraceSession::record_tensor`].
pub fn record_tensor(
    session: &TraceSession,
    span_id: SpanId,
    cap: &CandleCapture,
) -> anyhow::Result<()> {
    session.record_tensor(
        span_id,
        TensorRecord {
            tensor_id: cap.tensor_id(),
            label: cap.label.as_deref(),
            shape: &cap.shape,
            dtype: &cap.dtype,
            device: &cap.device,
            requires_grad: cap.requires_grad,
            dense_bytes: Some(cap.tensor_bytes),
            category: cap.category,
        },
    )
}

/// Record a semantically labeled Candle tensor in one call.
pub fn record_tensor_with_label(
    session: &TraceSession,
    span_id: SpanId,
    label: impl Into<String>,
    tensor: &Tensor,
    step: Option<ExecutionStep>,
) -> anyhow::Result<()> {
    let cap = CandleCapture::from_tensor(tensor, step).with_label(label);
    record_tensor(session, span_id, &cap)
}

/// Exact-coverage gradient capture: one validated manifest drives both the
/// declared contract and the recorded event population.
#[derive(Debug)]
pub struct GradientCapturePlan {
    root: String,
    entries: Vec<(String, Var)>,
    contract: GradientContract,
}

impl GradientCapturePlan {
    /// Build a plan from named variables. Keys are sorted ascending and the
    /// exact gradient contract is validated (non-empty manifest, unique keys,
    /// family membership) with its SHA-256 digest bound at construction.
    pub fn from_named_vars(
        root: impl Into<String>,
        vars: impl IntoIterator<Item = (String, Var)>,
        assign_family: impl Fn(&str) -> String,
        families: Vec<GradientFamilyContract>,
    ) -> anyhow::Result<Self> {
        let root = root.into();
        let mut entries: Vec<(String, Var)> = vars.into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        let expected = entries
            .iter()
            .map(|(key, _)| ExpectedGradient::new(root.clone(), key.clone(), assign_family(key)))
            .collect();
        let contract = GradientContract::new(expected, families)
            .context("building exact gradient contract from named vars")?;
        Ok(Self {
            root,
            entries,
            contract,
        })
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    /// Validated exact gradient contract. Clone it into the capture contract's
    /// `gradient_contract` alongside `CoverageLevel::Complete`.
    pub fn contract(&self) -> &GradientContract {
        &self.contract
    }

    /// Record exactly one gradient event per manifest entry, in manifest order.
    pub fn record(&self, session: &TraceSession, grads: &GradStore) -> anyhow::Result<()> {
        for (key, var) in &self.entries {
            let (state, norm) = match grads.get(var.as_tensor()) {
                None => (GradientState::Missing, None),
                Some(grad) => {
                    let norm = gradient_l2_norm(grad)
                        .with_context(|| format!("computing gradient norm for {key:?}"))?;
                    if !norm.is_finite() {
                        (GradientState::NonFinite, None)
                    } else if norm == 0.0 {
                        (GradientState::Zero, Some(0.0))
                    } else {
                        (GradientState::Present, Some(norm))
                    }
                }
            };
            session
                .record_gradient(&self.root, key, state, norm)
                .with_context(|| format!("recording gradient event for {key:?}"))?;
        }
        Ok(())
    }
}

/// L2 norm computed on device with a single scalar readback.
fn gradient_l2_norm(grad: &Tensor) -> anyhow::Result<f64> {
    let norm = grad
        .detach()
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()?;
    Ok(f64::from(norm))
}

/// Record a timed operation from a [`CandleOpCapture`].
pub fn record_op(
    session: &TraceSession,
    span_id: SpanId,
    cap: &CandleOpCapture,
) -> anyhow::Result<()> {
    session.record_op(
        span_id,
        OpRecord {
            op_name: &cap.op_name,
            inputs: &cap.inputs,
            output: Some(cap.output.tensor_id()),
            shape: &cap.output.shape,
            dtype: &cap.output.dtype,
            device: &cap.output.device,
            duration_ns: cap.duration_ns,
            timestamp_ns: cap.timestamp_ns,
            output_dense_bytes: Some(cap.output.tensor_bytes),
            input_dense_bytes: cap.input_tensor_bytes,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn dense_tensor_footprint_matches_shape_dtype() {
        let t = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor_dense_bytes(&t), 4 * 8 * 4);
    }

    #[test]
    fn capture_from_tensor() {
        let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        let cap = CandleCapture::from_tensor(&t, Some(ExecutionStep::Forward));
        assert_eq!(cap.shape, vec![2, 3]);
        assert_eq!(cap.category, MemoryCategory::Activation);
    }

    fn temp_trace(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "candle-graph-candle-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn zero_var() -> Var {
        Var::zeros((2,), DType::F32, &Device::Cpu).unwrap()
    }

    fn params_family() -> Vec<GradientFamilyContract> {
        vec![GradientFamilyContract::data_conditional("params", 1)]
    }

    #[test]
    fn gradient_plan_sorts_keys_into_manifest_order() {
        let plan = GradientCapturePlan::from_named_vars(
            "varmap",
            vec![
                ("decoder.weight".to_string(), zero_var()),
                ("encoder.bias".to_string(), zero_var()),
                ("encoder.weight".to_string(), zero_var()),
            ],
            |_| "params".to_string(),
            params_family(),
        )
        .unwrap();

        assert_eq!(plan.root(), "varmap");
        let keys: Vec<&str> = plan
            .contract()
            .expected
            .iter()
            .map(|expected| expected.key.as_str())
            .collect();
        assert_eq!(keys, ["decoder.weight", "encoder.bias", "encoder.weight"]);
        assert!(plan
            .contract()
            .expected
            .iter()
            .all(|expected| expected.root == "varmap" && expected.family == "params"));
    }

    #[test]
    fn gradient_plan_rejects_duplicate_keys_and_empty_manifests() {
        let duplicate = GradientCapturePlan::from_named_vars(
            "varmap",
            vec![
                ("encoder.weight".to_string(), zero_var()),
                ("encoder.weight".to_string(), zero_var()),
            ],
            |_| "params".to_string(),
            params_family(),
        );
        let message = format!("{:#}", duplicate.unwrap_err());
        assert!(message.contains("more than once"), "got: {message}");

        let empty = GradientCapturePlan::from_named_vars(
            "varmap",
            Vec::<(String, Var)>::new(),
            |_| "params".to_string(),
            params_family(),
        );
        let message = format!("{:#}", empty.unwrap_err());
        assert!(message.contains("must not be empty"), "got: {message}");
    }

    #[test]
    fn gradient_plan_records_exact_manifest_population() {
        use crate::capability::{CaptureContract, CoverageLevel};
        use crate::instrument::ProfileRun;
        use crate::trace::parse_trace;

        let device = Device::Cpu;
        let var_a = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let var_b = Var::new(&[4.0f32, 5.0], &device).unwrap();
        let plan = GradientCapturePlan::from_named_vars(
            "varmap",
            vec![
                ("used.weight".to_string(), var_a.clone()),
                ("frozen.weight".to_string(), var_b),
            ],
            |_| "params".to_string(),
            params_family(),
        )
        .unwrap();

        let loss = (var_a.as_tensor() * 2.0).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let path = temp_trace("gradient-plan");
        let run =
            ProfileRun::training("train::update", 1, "cpu").capture_contract(CaptureContract {
                gradients: CoverageLevel::Complete,
                gradient_contract: Some(plan.contract().clone()),
                ..CaptureContract::default()
            });
        let session = TraceSession::open(&path, run).unwrap();
        plan.record(&session, &grads).unwrap();
        session.finish().unwrap();

        let doc = parse_trace(&path).unwrap();
        let manifest_keys: Vec<&str> = plan
            .contract()
            .expected
            .iter()
            .map(|expected| expected.key.as_str())
            .collect();
        let recorded_keys: Vec<&str> = doc
            .gradients
            .iter()
            .map(|event| event.key.as_str())
            .collect();
        assert_eq!(recorded_keys, manifest_keys);
        assert_eq!(recorded_keys, ["frozen.weight", "used.weight"]);

        let frozen = &doc.gradients[0];
        assert_eq!(frozen.state, GradientState::Missing);
        assert_eq!(frozen.norm, None);

        let used = &doc.gradients[1];
        assert_eq!(used.state, GradientState::Present);
        assert!(used.norm.is_some_and(|norm| norm > 0.0));
    }
}
