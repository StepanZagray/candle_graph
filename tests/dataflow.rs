//! Integration tests for expression-level dataflow / dtype / gradient analysis.
//!
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_graph::dataflow::{
    analyze, analyze_with_candle_version, EdgeKind, GradState, NodeKind, NumericImpact,
};
use candle_graph::load;
use candle_graph::op_semantics::AbstractDtype;

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(source: &str) -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "candle-graph-dataflow-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("model.rs"), source).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn bf16_positive_and_f32_cross_entropy_conflict_at_broadcast_add() {
    // Positive path stays BF16; contrastive cross_entropy path is F32; they meet at
    // broadcast_add, which requires matching dtypes.
    const SRC: &str = r#"
fn contrastive_loss(pred: Tensor, positive: Tensor, target: Tensor) -> Result<Tensor> {
    let pred_bf16 = pred.to_dtype(DType::BF16)?;
    let positive_bf16 = positive.to_dtype(DType::BF16)?;
    let pos_loss = pred_bf16.broadcast_mul(&positive_bf16)?.sum_all()?;

    let logits = pred.to_dtype(DType::F32)?;
    let contrastive = candle_nn::loss::cross_entropy(&logits, &target)?;

    pos_loss.broadcast_add(&contrastive)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "contrastive_loss").unwrap();

    assert!(
        !graph.dtype_conflicts().is_empty(),
        "expected dtype conflict(s), diagnostics={:?}, conflicts={:?}",
        graph.diagnostics,
        graph.dtype_conflicts
    );
    let conflict = graph
        .dtype_conflicts()
        .iter()
        .find(|c| c.op == "broadcast_add")
        .expect("broadcast_add conflict");
    assert!(
        matches!(
            (conflict.left, conflict.right),
            (AbstractDtype::Bf16, AbstractDtype::F32) | (AbstractDtype::F32, AbstractDtype::Bf16)
        ),
        "unexpected dtypes {:?}",
        (conflict.left, conflict.right)
    );
    assert!(
        graph.loss_nodes.iter().any(|id| {
            matches!(
                &graph.node(*id).kind,
                NodeKind::Call { callee } if callee == "cross_entropy"
            )
        }),
        "cross_entropy should be recorded as a loss sink"
    );
}

#[test]
fn unknown_positive_plus_f32_is_a_dtype_risk() {
    // This mirrors the pre-fix bridge shape: the positive loss comes from a helper whose
    // activation dtype is not statically known, while the contrastive logits are explicitly F32.
    // The analyzer must flag the join as a risk without overstating it as a proven conflict.
    const SRC: &str = r#"
fn fine_grained_token_slot_alignment(
    target: Tensor,
    mask: Tensor,
    conditioning: Tensor,
) -> Result<Tensor> {
    external_alignment(target, mask, conditioning)
}

fn conditioning_alignment_loss(
    target: Tensor,
    mask: Tensor,
    conditioning: Tensor,
) -> Result<Tensor> {
    let positive = fine_grained_token_slot_alignment(target, mask, conditioning)?;
    let logits = conditioning.to_dtype(DType::F32)?;
    let labels = make_targets();
    let forward = candle_nn::loss::cross_entropy(&logits, &labels)?;
    let backward = candle_nn::loss::cross_entropy(&logits.t()?, &labels)?;
    let contrastive = forward.broadcast_add(&backward)?.affine(0.5, 0.0)?;
    positive.broadcast_add(&contrastive)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "conditioning_alignment_loss").unwrap();

    assert!(graph.dtype_conflicts().is_empty());
    let risk = graph
        .dtype_risks()
        .iter()
        .find(|risk| risk.op == "broadcast_add")
        .expect("pre-fix final broadcast_add should be a dtype risk");
    assert_eq!(risk.known, AbstractDtype::F32);
}

#[test]
fn severed_gradient_via_rms_norm_no_bwd() {
    const SRC: &str = r#"
fn forward(xs: Tensor, weight: Tensor) -> Result<Tensor> {
    let ys = candle_nn::ops::rms_norm(&xs, &weight, 1e-5)?;
    ys.sum_all()
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze_with_candle_version(&krate, "forward", Some("0.11.0")).unwrap();

    let severing = graph.severing_edges();
    assert!(
        !severing.is_empty(),
        "expected severing edges from rms_norm; nodes={:?}",
        graph
            .nodes
            .iter()
            .map(|n| (&n.kind, n.grad))
            .collect::<Vec<_>>()
    );
    assert!(
        graph.nodes.iter().any(|n| {
            matches!(&n.kind, NodeKind::Call { callee } if callee == "rms_norm")
                && matches!(n.grad, GradState::Severed)
        }),
        "rms_norm node should be Severed"
    );
    assert!(
        graph.edges.iter().any(|e| {
            matches!(e.kind, EdgeKind::Severing)
                && matches!(
                    &graph.node(e.to).kind,
                    NodeKind::Call { callee } if callee == "rms_norm"
                )
        }),
        "severing edge should target rms_norm"
    );
}

#[test]
fn detach_severs_and_marks_explicit_var_dead() {
    const SRC: &str = r#"
fn loss(scale: candle_core::Var, target: Tensor) -> Result<Tensor> {
    let frozen = scale.detach();
    candle_nn::loss::mse(&frozen, &target)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "loss").unwrap();

    assert!(!graph.severing_edges().is_empty());
    let dead = graph.dead_params();
    assert!(
        !dead.is_empty(),
        "detached Var should be a dead trainable param; param_nodes={:?} loss={:?} grads={:?}",
        graph.param_nodes,
        graph.loss_nodes,
        graph
            .nodes
            .iter()
            .map(|n| (&n.kind, n.grad))
            .collect::<Vec<_>>()
    );
}

#[test]
fn tensor_parameter_name_does_not_claim_trainability() {
    const SRC: &str = r#"
fn loss(frozen_weight: Tensor, target: Tensor) -> Result<Tensor> {
    let detached = frozen_weight.detach();
    candle_nn::loss::mse(&detached, &target)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "loss").unwrap();

    assert!(
        graph.param_nodes.is_empty(),
        "a Tensor named weight is not proof of optimizer/trainable identity"
    );
    assert!(graph.dead_params().is_empty());
}

#[test]
fn interprocedural_inherent_method_and_branch() {
    const SRC: &str = r#"
struct Model;

impl Model {
    fn encode(&self, xs: Tensor) -> Result<Tensor> {
        xs.relu()
    }

    fn forward(&self, xs: Tensor, flag: bool) -> Result<Tensor> {
        let h = if flag {
            self.encode(xs)?
        } else {
            xs.gelu()?
        };
        h.sum_all()
    }
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "Model::forward").unwrap();

    assert!(
        graph
            .nodes
            .iter()
            .any(|n| { matches!(&n.kind, NodeKind::Call { callee } if callee.contains("encode")) }),
        "expected interprocedural encode call"
    );
    assert!(
        graph.nodes.iter().any(|n| matches!(n.kind, NodeKind::Phi)),
        "expected branch phi"
    );
    assert!(graph.entry_return.is_some());
}

#[test]
fn paths_to_reaches_loss_from_operand() {
    const SRC: &str = r#"
fn loss(pred: Tensor, target: Tensor) -> Result<Tensor> {
    let diff = pred.broadcast_sub(&target)?;
    diff.sqr()?.mean_all()
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "loss").unwrap();

    let pred = graph
        .nodes
        .iter()
        .find(|n| matches!(&n.kind, NodeKind::Param { name } if name == "pred"))
        .map(|n| n.id)
        .expect("pred param");
    let sink = graph
        .nodes
        .iter()
        .rev()
        .find(|n| matches!(&n.kind, NodeKind::Call { callee } if callee == "mean_all"))
        .map(|n| n.id)
        .expect("mean_all");
    let paths = graph.paths_to(pred, sink);
    assert!(!paths.is_empty(), "expected a path from pred to mean_all");
}

#[test]
fn to_dtype_and_same_dtype_matmul_ok() {
    const SRC: &str = r#"
fn project(xs: Tensor, weight: Tensor) -> Result<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    let weight = weight.to_dtype(DType::F32)?;
    xs.matmul(&weight)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "project").unwrap();
    assert!(
        graph.dtype_conflicts().is_empty(),
        "F32 matmul should not conflict: {:?}",
        graph.dtype_conflicts
    );
    assert!(graph.nodes.iter().any(|n| {
        matches!(&n.kind, NodeKind::Call { callee } if callee == "matmul")
            && matches!(n.dtype, AbstractDtype::F32)
    }));
}

#[test]
fn recursion_guard_on_mutual_calls() {
    const SRC: &str = r#"
fn a(xs: Tensor) -> Result<Tensor> {
    b(xs)
}
fn b(xs: Tensor) -> Result<Tensor> {
    a(xs)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "a").unwrap();
    assert!(
        graph
            .diagnostics
            .iter()
            .any(|d| d.message.contains("recursion guard")),
        "expected recursion guard diagnostic: {:?}",
        graph.diagnostics
    );
}

#[test]
fn trait_module_forward_is_loaded_and_analyzed() {
    const SRC: &str = r#"
struct Model;

impl candle_core::Module for Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.relu()
    }
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "Model::forward").unwrap();

    assert!(graph
        .nodes
        .iter()
        .any(|node| matches!(&node.kind, NodeKind::Call { callee } if callee == "relu")));
}

#[test]
fn receiver_field_type_resolves_duplicate_forward_methods() {
    const SRC: &str = r#"
struct Encoder;
struct Decoder;
struct Model { encoder: Encoder }

impl Encoder {
    fn forward(&self, xs: Tensor) -> Result<Tensor> { xs.relu() }
}
impl Decoder {
    fn forward(&self, xs: Tensor) -> Result<Tensor> { xs.silu() }
}
impl Model {
    fn forward(&self, xs: Tensor) -> Result<Tensor> {
        self.encoder.forward(xs)
    }
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "Model::forward").unwrap();

    assert!(graph
        .nodes
        .iter()
        .any(|node| matches!(&node.kind, NodeKind::Call { callee } if callee == "relu")));
    assert!(!graph
        .nodes
        .iter()
        .any(|node| matches!(&node.kind, NodeKind::Call { callee } if callee == "silu")));
}

#[test]
fn scalar_tensor_arithmetic_preserves_dtype_without_false_risk() {
    const SRC: &str = r#"
fn scale(xs: Tensor) -> Result<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    Ok(xs * 0.5)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "scale").unwrap();

    assert!(graph.dtype_risks().is_empty(), "{:?}", graph.dtype_risks());
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.kind, NodeKind::Call { callee } if callee == "mul")
            && node.dtype == AbstractDtype::F32
    }));
}

#[test]
fn names_ending_in_f32_are_not_dtype_evidence() {
    const SRC: &str = r#"
fn cast(xs: Tensor) -> Result<Tensor> {
    xs.to_dtype(custom::F32)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "cast").unwrap();
    let cast = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, NodeKind::Call { callee } if callee == "to_dtype"))
        .unwrap();
    assert_eq!(cast.dtype, AbstractDtype::Unknown);
}

#[test]
fn tensor_zeros_infers_literal_dtype() {
    let fixture = Fixture::new(
        r#"
fn explicit() -> Result<Tensor> {
    Tensor::zeros((2, 3), DType::F32, Device::Cpu)
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "explicit").unwrap();
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.kind, NodeKind::Call { callee } if callee == "Tensor::zeros")
            && node.dtype == AbstractDtype::F32
    }));
}

#[test]
fn tensor_dtype_method_preserves_receiver_dtype() {
    let fixture = Fixture::new(
        r#"
fn copy_dtype(x: Tensor) -> Result<Tensor> {
    let _d = x.dtype();
    Tensor::zeros((1,), x.dtype(), x.device())
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "copy_dtype").unwrap();
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.kind, NodeKind::Literal { text } if text == "dtype()")
            && node.dtype == AbstractDtype::Unknown
    }));
    let param = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, NodeKind::Param { name } if name == "x"))
        .unwrap();
    assert_eq!(param.dtype, AbstractDtype::Unknown);
}

#[test]
fn interprocedural_detach_cannot_be_bypassed_by_callsite_edges() {
    const SRC: &str = r#"
use candle_nn::loss::mse;

fn cut(weight: Var) -> Result<Tensor> {
    weight.detach()
}

fn loss(weight: Var, target: Tensor) -> Result<Tensor> {
    let prediction = cut(weight)?;
    mse(&prediction, &target)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze_with_candle_version(&krate, "loss", Some("0.11.0")).unwrap();

    assert_eq!(
        graph.loss_nodes.len(),
        1,
        "imported mse must resolve as a loss"
    );
    assert!(
        graph.dead_params().iter().any(
            |id| matches!(&graph.node(*id).kind, NodeKind::Param { name } if name == "weight")
        ),
        "detach inside a helper must keep the trainable argument disconnected from the loss"
    );
}

#[test]
fn tensor_node_inventory_excludes_self_and_scalar_literals() {
    const SRC: &str = r#"
struct Model;
impl Model {
    fn forward(&self, xs: Tensor) -> Result<Tensor> {
        let slots = 4;
        xs.narrow(1, 0, slots)
    }
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze(&krate, "Model::forward").unwrap();

    assert!(graph
        .tensor_nodes
        .iter()
        .all(|id| match &graph.node(*id).kind {
            NodeKind::Literal { .. } => false,
            NodeKind::Param { name } if name == "self" => false,
            _ => true,
        }));
}

#[test]
fn bce_with_logits_hazard_comes_from_body_expansion_not_a_hardcoded_flag() {
    const SRC: &str = r#"
fn q_loss(logit: Tensor, target: Tensor) -> Result<Tensor> {
    candle_nn::loss::binary_cross_entropy_with_logit(&logit, &target)
}
"#;
    let fixture = Fixture::new(SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze_with_candle_version(&krate, "q_loss", Some("0.11.0")).unwrap();
    assert!(
        graph.numeric_domain_violations.iter().any(|finding| {
            finding.proven
                && finding.library_cite.as_deref() == Some("candle-nn-0.11.0/src/loss.rs:64-74")
                && finding.impact == NumericImpact::TrainingLossNaN
        }),
        "expected expanded-body domain violation with training impact, got {:?}",
        graph.numeric_domain_violations
    );
    assert!(
        graph.zero_times_infinity.iter().any(|finding| {
            finding.library_cite.as_deref() == Some("candle-nn-0.11.0/src/loss.rs:64-74")
                && finding.impact == NumericImpact::TrainingLossNaN
        }),
        "expected expanded-body zero-times-infinity with training impact, got {:?}",
        graph.zero_times_infinity
    );
}

#[test]
fn sigmoid_log_is_numeric_domain_violation_but_affine_guard_is_safe() {
    const UNSAFE_SRC: &str = r#"
fn unsafe_direct(x: Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(&x)?.log()
}
"#;
    let fixture = Fixture::new(UNSAFE_SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze_with_candle_version(&krate, "unsafe_direct", Some("0.11.0")).unwrap();
    assert!(
        graph
            .numeric_domain_violations
            .iter()
            .any(|finding| finding.proven && finding.op == "log"),
        "expected proven numeric-domain-violation for sigmoid.log, got {:?}",
        graph.numeric_domain_violations
    );

    const SAFE_SRC: &str = r#"
fn safe_guard(x: Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(&x)?.affine(1.0, 1e-7)?.log()
}
"#;
    let fixture = Fixture::new(SAFE_SRC);
    let krate = load::load(fixture.path()).unwrap();
    let graph = analyze_with_candle_version(&krate, "safe_guard", Some("0.11.0")).unwrap();
    assert!(
        graph.numeric_domain_violations.is_empty(),
        "epsilon-guard affine must discharge StrictlyPositive, got {:?}",
        graph.numeric_domain_violations
    );
    assert!(
        graph.zero_times_infinity.is_empty(),
        "safe local composition must not emit zero-times-infinity: {:?}",
        graph.zero_times_infinity
    );
}
