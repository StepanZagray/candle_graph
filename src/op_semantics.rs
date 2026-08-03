//! Transfer rules for candle Tensor ops and known candle-nn helpers.
//!
//! Rules are deliberate and incomplete by design: anything not listed returns
//! [`OpEffect::Unknown`] rather than inventing behaviour.

use serde::Serialize;

/// The Candle release whose version-sensitive implementation details back this catalog.
pub const AUDITED_CANDLE_VERSION: &str = "0.11.0";

pub fn is_audited_candle_version(version: &str) -> bool {
    version == AUDITED_CANDLE_VERSION
}

/// Return the shared audited version only when candle-core and candle-nn resolve consistently.
///
/// A mismatched pair is not a sound basis for candle-nn rules because those implementations call
/// directly into candle-core's operation and autograd APIs.
pub fn matched_candle_version<'a>(
    candle_core_version: Option<&'a str>,
    candle_nn_version: Option<&'a str>,
) -> Option<&'a str> {
    match (candle_core_version, candle_nn_version) {
        (Some(core), Some(nn)) if core == nn && is_audited_candle_version(nn) => Some(nn),
        _ => None,
    }
}

/// Abstract dtype carried on expression nodes. Coarse by intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstractDtype {
    F64,
    F32,
    F16,
    Bf16,
    I16,
    I32,
    I64,
    U32,
    U8,
    F8E4M3,
    F6E2M3,
    F6E3M2,
    F4,
    F8E8M0,
    /// Could not be determined statically.
    Unknown,
}

impl AbstractDtype {
    pub fn parse(text: &str) -> Self {
        let t = text.trim();
        let t = t.rsplit("::").next().unwrap_or(t);
        match t {
            "F64" | "f64" => Self::F64,
            "F32" | "f32" => Self::F32,
            "F16" | "f16" => Self::F16,
            "BF16" | "Bf16" | "bf16" => Self::Bf16,
            "I16" | "i16" => Self::I16,
            "I32" | "i32" => Self::I32,
            "I64" | "i64" => Self::I64,
            "U32" | "u32" => Self::U32,
            "U8" | "u8" => Self::U8,
            "F8E4M3" => Self::F8E4M3,
            "F6E2M3" => Self::F6E2M3,
            "F6E3M2" => Self::F6E3M2,
            "F4" => Self::F4,
            "F8E8M0" => Self::F8E8M0,
            _ => Self::Unknown,
        }
    }

    pub fn is_known(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

impl std::fmt::Display for AbstractDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64 => write!(f, "F64"),
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::Bf16 => write!(f, "BF16"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::I64 => write!(f, "I64"),
            Self::U32 => write!(f, "U32"),
            Self::U8 => write!(f, "U8"),
            Self::F8E4M3 => write!(f, "F8E4M3"),
            Self::F6E2M3 => write!(f, "F6E2M3"),
            Self::F6E3M2 => write!(f, "F6E3M2"),
            Self::F4 => write!(f, "F4"),
            Self::F8E8M0 => write!(f, "F8E8M0"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// How an op affects gradient connectivity relative to its tensor inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GradFlow {
    /// Gradients flow through all tensor operands (ordinary differentiable op).
    Propagates,
    /// Explicitly cuts the autograd graph (`detach`, `apply_op*_no_bwd`, …).
    Severs,
    /// Gradient exists only under layout/contiguity assumptions the analyzer cannot prove.
    LayoutDependent,
    /// Not enough information.
    Unknown,
}

/// How the result dtype relates to operand dtypes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DtypeRule {
    /// Result dtype equals the (sole / first) tensor operand.
    Preserve,
    /// All tensor operands must share a dtype; result is that dtype. Mismatch is a conflict.
    SameAsInputs,
    /// Result dtype is taken from an explicit argument (e.g. `to_dtype(DType::F32)`).
    Explicit,
    /// Result has a fixed Candle dtype independent of its operands.
    Fixed(AbstractDtype),
    /// Result dtype cannot be stated.
    Unknown,
}

/// Float range of an op's result after rounding — not the mathematical range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericDomain {
    Real,
    NonNegative,
    /// Mathematically open; attains `0.0` and `1.0` exactly after f32 rounding.
    SaturatingUnit,
    /// Provably strictly inside its bounds (e.g. after an epsilon-guard `affine`).
    StrictlyPositive,
    Unknown,
}

/// Precondition an op places on its primary tensor operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainRequirement {
    StrictlyPositive,
    NonZero,
    NonNegative,
    None,
}

/// Measured f32 saturation thresholds for candle-nn 0.11.0 `sigmoid` =
/// `(exp(-v) + 1).recip()` (`candle-nn-0.11.0/src/ops.rs:59-60`).
pub const SIGMOID_F32_UPPER_SATURATION: f32 = 16.6355;
pub const SIGMOID_F32_LOWER_SATURATION: f32 = -88.7228;

/// One step in an audited library-op body. Step indices refer to earlier atoms in the same body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyAtom {
    /// Call-site argument (`0` = first tensor operand).
    Arg(usize),
    /// Identity view that records an API-level domain fact (e.g. BCE targets ∈ [0, 1]).
    Assume {
        src: u16,
        domain: NumericDomain,
    },
    Unary {
        op: &'static str,
        src: u16,
    },
    Binary {
        op: &'static str,
        left: u16,
        right: u16,
    },
    /// `tensor.affine(mul, add)` with proven literal coefficients.
    Affine {
        src: u16,
        mul: f64,
        add: f64,
    },
}

/// Expandable body of a candle-nn (or similar) helper, audited for one Candle release.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LibraryBody {
    pub cite: &'static str,
    pub steps: &'static [BodyAtom],
}

/// candle-nn 0.11.0 `binary_cross_entropy_with_logit` (`loss.rs:64-74`):
/// `sigmoid` → `log` / `(1-p).log` → multiply by target coefficients.
const BCE_WITH_LOGITS_011: LibraryBody = LibraryBody {
    cite: "candle-nn-0.11.0/src/loss.rs:64-74",
    steps: &[
        BodyAtom::Arg(0), // 0: logits
        BodyAtom::Arg(1), // 1: target
        // candle-nn documents targets as unit labels / probabilities.
        BodyAtom::Assume {
            src: 1,
            domain: NumericDomain::SaturatingUnit,
        }, // 2: t
        BodyAtom::Unary {
            op: "sigmoid",
            src: 0,
        }, // 3: p
        BodyAtom::Unary { op: "log", src: 3 }, // 4: log(p)
        BodyAtom::Binary {
            op: "mul",
            left: 2,
            right: 4,
        }, // 5: t * log(p)
        BodyAtom::Affine {
            src: 3,
            mul: -1.0,
            add: 1.0,
        }, // 6: 1 - p
        BodyAtom::Unary { op: "log", src: 6 }, // 7: log(1-p)
        BodyAtom::Affine {
            src: 2,
            mul: -1.0,
            add: 1.0,
        }, // 8: 1 - t
        BodyAtom::Binary {
            op: "mul",
            left: 8,
            right: 7,
        }, // 9: (1-t) * log(1-p) — 0 * -inf when saturated
        BodyAtom::Binary {
            op: "add",
            left: 5,
            right: 9,
        }, // 10
    ],
};

/// Look up an expandable body for `op` under an audited Candle version.
///
/// Bodies are transfer recipes, not precomputed “unsafe” verdicts. The dataflow domain pass
/// decides whether the expanded composition is hazardous.
pub fn library_body(op: &str, candle_nn_version: Option<&str>) -> Option<&'static LibraryBody> {
    if !candle_nn_version.is_some_and(is_audited_candle_version) {
        return None;
    }
    let name = op.rsplit("::").next().unwrap_or(op);
    match name {
        "binary_cross_entropy_with_logit" => Some(&BCE_WITH_LOGITS_011),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpEffect {
    pub name: String,
    pub dtype: DtypeRule,
    pub grad: GradFlow,
    /// True when this call is treated as a scalar loss sink for connectivity queries.
    pub is_loss: bool,
    /// Optional note carried into diagnostics (citation / reason).
    pub note: Option<&'static str>,
    /// Result domain after float rounding. `Unknown` when not audited.
    pub domain: NumericDomain,
    /// Requirement on the primary tensor operand.
    pub requires: DomainRequirement,
}

impl OpEffect {
    fn known(
        name: &str,
        dtype: DtypeRule,
        grad: GradFlow,
        is_loss: bool,
        note: Option<&'static str>,
    ) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            grad,
            is_loss,
            note,
            domain: NumericDomain::Real,
            requires: DomainRequirement::None,
        }
    }

    fn with_domain(mut self, domain: NumericDomain) -> Self {
        self.domain = domain;
        self
    }

    fn with_requires(mut self, requires: DomainRequirement) -> Self {
        self.requires = requires;
        self
    }

    pub fn unknown(name: &str) -> Self {
        Self {
            name: name.to_string(),
            dtype: DtypeRule::Unknown,
            grad: GradFlow::Unknown,
            is_loss: false,
            note: Some("no transfer rule; left Unknown"),
            domain: NumericDomain::Unknown,
            requires: DomainRequirement::None,
        }
    }

    pub fn domain_rule_label(&self) -> String {
        match self.domain {
            NumericDomain::Real => "real".into(),
            NumericDomain::NonNegative => "non_negative".into(),
            NumericDomain::SaturatingUnit => format!(
                "saturating_unit(f32_upper={SIGMOID_F32_UPPER_SATURATION},f32_lower={SIGMOID_F32_LOWER_SATURATION})"
            ),
            NumericDomain::StrictlyPositive => "strictly_positive".into(),
            NumericDomain::Unknown => "unknown".into(),
        }
    }
}

/// Strength of a domain-requirement failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainViolationConfidence {
    Proven,
    Unknown,
}

/// Whether `domain` fails to discharge `requires`.
///
/// Returns `Some(Proven)` for a catalog-known violation, `Some(Unknown)` when the producer
/// domain is unknown, and `None` when the requirement is met.
pub fn domain_violation(
    requires: DomainRequirement,
    domain: NumericDomain,
) -> Option<DomainViolationConfidence> {
    match requires {
        DomainRequirement::None => None,
        DomainRequirement::StrictlyPositive => match domain {
            NumericDomain::StrictlyPositive => None,
            NumericDomain::Unknown => Some(DomainViolationConfidence::Unknown),
            NumericDomain::Real | NumericDomain::NonNegative | NumericDomain::SaturatingUnit => {
                Some(DomainViolationConfidence::Proven)
            }
        },
        DomainRequirement::NonZero => match domain {
            NumericDomain::StrictlyPositive => None,
            NumericDomain::Unknown => Some(DomainViolationConfidence::Unknown),
            NumericDomain::Real | NumericDomain::NonNegative | NumericDomain::SaturatingUnit => {
                Some(DomainViolationConfidence::Proven)
            }
        },
        DomainRequirement::NonNegative => match domain {
            NumericDomain::NonNegative
            | NumericDomain::StrictlyPositive
            | NumericDomain::SaturatingUnit => None,
            NumericDomain::Unknown => Some(DomainViolationConfidence::Unknown),
            NumericDomain::Real => Some(DomainViolationConfidence::Proven),
        },
    }
}

/// Join domains through a dtype-preserving unary/binary where both sides contribute.
pub fn join_domain(left: NumericDomain, right: NumericDomain) -> NumericDomain {
    use NumericDomain::*;
    match (left, right) {
        (Unknown, _) | (_, Unknown) => Unknown,
        (StrictlyPositive, StrictlyPositive) => StrictlyPositive,
        (StrictlyPositive, NonNegative) | (NonNegative, StrictlyPositive) => NonNegative,
        (NonNegative, NonNegative) => NonNegative,
        (SaturatingUnit, SaturatingUnit) => SaturatingUnit,
        (SaturatingUnit, NonNegative) | (NonNegative, SaturatingUnit) => NonNegative,
        (SaturatingUnit, StrictlyPositive) | (StrictlyPositive, SaturatingUnit) => NonNegative,
        _ => Real,
    }
}

/// Transfer for `tensor.affine(mul, add)` with proven literal coefficients.
///
/// Only the epsilon-guard form `mul > 0 && add > 0` on a non-negative / unit operand
/// discharges `StrictlyPositive`. Reflections such as `affine(-1, 1)` (`1 - p`) keep a
/// zero-attaining domain.
pub fn affine_domain(operand: NumericDomain, mul: Option<f64>, add: Option<f64>) -> NumericDomain {
    match (mul, add) {
        (Some(m), Some(a)) if m > 0.0 && a > 0.0 => match operand {
            NumericDomain::SaturatingUnit
            | NumericDomain::NonNegative
            | NumericDomain::StrictlyPositive => NumericDomain::StrictlyPositive,
            NumericDomain::Unknown => NumericDomain::Unknown,
            NumericDomain::Real => NumericDomain::Real,
        },
        _ => match operand {
            // 1 - p over a unit interval still attains 0 and 1.
            NumericDomain::SaturatingUnit => NumericDomain::SaturatingUnit,
            NumericDomain::StrictlyPositive => NumericDomain::Real,
            NumericDomain::NonNegative => NumericDomain::Real,
            NumericDomain::Unknown => NumericDomain::Unknown,
            NumericDomain::Real => NumericDomain::Real,
        },
    }
}

pub fn domain_includes_zero(domain: NumericDomain) -> bool {
    matches!(
        domain,
        NumericDomain::NonNegative | NumericDomain::SaturatingUnit | NumericDomain::Real
    )
}

/// Last path segment of a call, e.g. `broadcast_add` or `cross_entropy`.
pub fn lookup(op: &str) -> OpEffect {
    // Without Cargo evidence the version-sensitive answer is unknown. Callers that resolved the
    // target crate must use `lookup_for`.
    lookup_for(op, None)
}

/// Look up an operation against an audited Candle version.
///
/// Generic Tensor algebra is stable across the supported catalog. Rules tied to Candle's
/// custom `*_no_bwd` implementations are only asserted for versions whose source was audited;
/// other versions stay `Unknown` rather than inheriting a potentially stale gradient claim.
pub fn lookup_for(op: &str, candle_nn_version: Option<&str>) -> OpEffect {
    let name = op.rsplit("::").next().unwrap_or(op);
    let audited_no_bwd = matches!(
        name,
        "softmax_last_dim"
            | "rms_norm"
            | "layer_norm"
            | "sdpa"
            | "rope"
            | "rope_i"
            | "rope_thd"
            | "flash_attn"
            | "flash_attn_varlen_cpu"
            | "flash_attn_varlen_unfused"
            | "run_flash_attn_cpu"
    );
    if audited_no_bwd && !candle_nn_version.is_some_and(is_audited_candle_version) {
        return OpEffect::unknown(name);
    }
    match name {
        // --- explicit dtype ---
        "to_dtype" => OpEffect::known(
            name,
            DtypeRule::Explicit,
            GradFlow::Propagates,
            false,
            Some("candle Tensor::to_dtype"),
        ),

        // --- same-dtype binaries (conflict when operands disagree) ---
        "add" | "sub" | "mul" | "maximum" | "minimum" => OpEffect::known(
            name,
            DtypeRule::SameAsInputs,
            GradFlow::Propagates,
            false,
            Some("candle binary op; operands must share dtype"),
        ),
        "div" => OpEffect::known(
            name,
            DtypeRule::SameAsInputs,
            GradFlow::Propagates,
            false,
            Some("candle binary op; operands must share dtype"),
        )
        .with_requires(DomainRequirement::NonZero),
        "broadcast_add" | "broadcast_sub" | "broadcast_mul" | "broadcast_maximum"
        | "broadcast_minimum" | "broadcast_pow" => OpEffect::known(
            name,
            DtypeRule::SameAsInputs,
            GradFlow::Propagates,
            false,
            Some("candle broadcast binary; operands must share dtype"),
        ),
        "broadcast_div" => OpEffect::known(
            name,
            DtypeRule::SameAsInputs,
            GradFlow::Propagates,
            false,
            Some("candle broadcast binary; operands must share dtype"),
        )
        .with_requires(DomainRequirement::NonZero),
        "matmul" | "broadcast_matmul" => OpEffect::known(
            name,
            DtypeRule::SameAsInputs,
            GradFlow::Propagates,
            false,
            Some("candle matmul; operands must share dtype"),
        ),

        // --- dtype-preserving unary math ---
        "neg" | "sin" | "cos" | "tanh" | "gelu" | "silu" | "erf" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary"),
        ),
        "abs" | "sqr" | "relu" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary"),
        )
        .with_domain(NumericDomain::NonNegative),
        "exp" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary; f32 underflows to exactly 0.0"),
        )
        .with_domain(NumericDomain::NonNegative),
        "sqrt" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary"),
        )
        .with_domain(NumericDomain::NonNegative)
        .with_requires(DomainRequirement::NonNegative),
        "log" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary; undefined at 0"),
        )
        .with_requires(DomainRequirement::StrictlyPositive),
        "recip" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving unary; undefined at 0"),
        )
        .with_requires(DomainRequirement::NonZero),
        "floor" | "round" | "sign" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle operation is created without a backward rule"),
        ),
        "ceil" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Unknown,
            false,
            Some("candle backward reports this operation as unsupported"),
        ),
        "cmp" | "eq" | "ne" | "lt" | "gt" | "ge" | "le" => OpEffect::known(
            name,
            DtypeRule::Fixed(AbstractDtype::U8),
            GradFlow::Severs,
            false,
            Some("candle-core 0.11.0 comparisons return U8 and do not propagate gradients"),
        ),
        "argmin" | "argmin_keepdim" | "argmax" | "argmax_keepdim" => OpEffect::known(
            name,
            DtypeRule::Fixed(AbstractDtype::U32),
            GradFlow::Severs,
            false,
            Some("candle-core 0.11.0 index reductions return U32 without a backward op"),
        ),

        // --- dtype-preserving reductions / shape views with known backprop ---
        "sum" | "sum_keepdim" | "sum_all" | "mean" | "mean_keepdim" | "mean_all" | "max"
        | "max_keepdim" | "min" | "min_keepdim" | "log_sum_exp" | "powf" | "elu" | "clamp" => {
            OpEffect::known(
                name,
                DtypeRule::Preserve,
                GradFlow::Propagates,
                false,
                Some("dtype-preserving reduction/affine"),
            )
        }
        // Domain is a transfer over (mul, add) literals — see `affine_domain`.
        "affine" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("dtype-preserving affine; domain transfers through positive add"),
        )
        .with_domain(NumericDomain::Unknown),

        // --- layout / view ops with explicit Candle backward rules ---
        "reshape" | "flatten_all" | "flatten_to" | "flatten_from" | "squeeze" | "unsqueeze"
        | "transpose" | "permute" | "narrow" | "contiguous" | "broadcast_as" | "broadcast_left"
        | "expand" | "t" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("layout/view operation with an explicit backward rule"),
        ),

        // --- explicit sever ---
        "detach" | "as_detached_tensor" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("explicit detach"),
        ),

        // --- candle-nn losses (expandable bodies via `library_body`, not precomputed verdicts) ---
        "mse" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            true,
            Some("candle_nn::loss::mse"),
        ),
        "nll" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            true,
            Some("candle_nn::loss::nll expects log-probabilities"),
        ),
        "cross_entropy" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            true,
            Some("candle_nn::loss::cross_entropy via log_softmax + nll"),
        ),
        "huber" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            true,
            Some("candle_nn::loss::huber"),
        ),
        "binary_cross_entropy_with_logit" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            true,
            Some(
                "candle_nn::loss::binary_cross_entropy_with_logit; body expanded when \
                 Candle 0.11.0 is resolved — see library_body()",
            ),
        ),

        // --- candle-nn ops implemented via apply_op*_no_bwd (candle-nn 0.11.0) ---
        "softmax_last_dim" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle_nn::ops::softmax_last_dim uses apply_op1_no_bwd"),
        )
        .with_domain(NumericDomain::SaturatingUnit),
        "rms_norm" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle_nn::ops::rms_norm uses apply_op2_no_bwd"),
        ),
        "layer_norm" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle_nn::ops::layer_norm uses apply_op3_no_bwd"),
        ),
        "sdpa" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle_nn::ops::sdpa uses apply_op3_no_bwd"),
        ),
        "rope" | "rope_i" | "rope_thd" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle_nn::rotary_emb::* uses apply_op3_no_bwd"),
        ),
        "flash_attn"
        | "flash_attn_varlen_cpu"
        | "flash_attn_varlen_unfused"
        | "run_flash_attn_cpu" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Severs,
            false,
            Some("candle-nn 0.11.0 CPU/varlen attention returns a detached output"),
        ),

        // --- candle-nn ops that DO have differentiable slow paths ---
        "rms_norm_slow" | "layer_norm_slow" | "rope_slow" | "rope_i_slow" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("candle_nn differentiable helper"),
        ),
        "sigmoid" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some(
                "candle_nn sigmoid saturates to exactly 0/1 in f32 \
                 (candle-nn-0.11.0/src/ops.rs:59-60)",
            ),
        )
        .with_domain(NumericDomain::SaturatingUnit),
        "softmax" | "log_softmax" => OpEffect::known(
            name,
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("candle_nn softmax family; probabilities attain 0/1 after f32 rounding"),
        )
        .with_domain(NumericDomain::SaturatingUnit),

        // device moves preserve dtype and grad
        "to_device" | "clone" => {
            OpEffect::known(name, DtypeRule::Preserve, GradFlow::Propagates, false, None)
        }

        _ => OpEffect::unknown(name),
    }
}

/// Exact method semantics where the bare method name is insufficient.
///
/// In candle-nn 0.11.0 `RmsNorm::forward` chooses a no-backward custom kernel for contiguous
/// inputs and a differentiable implementation otherwise, whereas `forward_diff` always selects
/// the differentiable implementation. Treating both as a bare `forward` would lose the crucial
/// distinction.
pub fn lookup_method(
    receiver_type: &str,
    method: &str,
    candle_nn_version: Option<&str>,
) -> OpEffect {
    let receiver = receiver_type
        .split('<')
        .next()
        .unwrap_or(receiver_type)
        .rsplit("::")
        .next()
        .unwrap_or(receiver_type);
    match (receiver, method, candle_nn_version) {
        ("RmsNorm", "forward_diff", Some(AUDITED_CANDLE_VERSION)) => OpEffect::known(
            "RmsNorm::forward_diff",
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("audited candle-nn RmsNorm::forward_diff uses differentiable LayerNorm::forward"),
        ),
        ("RmsNorm", "forward", Some(AUDITED_CANDLE_VERSION)) => OpEffect::known(
            "RmsNorm::forward",
            DtypeRule::Preserve,
            GradFlow::LayoutDependent,
            false,
            Some("audited candle-nn RmsNorm::forward uses a no-bwd kernel for contiguous input"),
        ),
        (
            "Linear" | "Embedding" | "Conv1d" | "Conv2d" | "ConvTranspose1d" | "ConvTranspose2d"
            | "PReLU" | "GroupNorm" | "BatchNorm",
            "forward" | "forward_t",
            Some(AUDITED_CANDLE_VERSION),
        ) => OpEffect::known(
            &format!("{receiver}::{method}"),
            DtypeRule::Preserve,
            GradFlow::Propagates,
            false,
            Some("known candle-nn parameterized module forward"),
        ),
        _ => OpEffect::unknown(&format!("{receiver}::{method}")),
    }
}

/// Resolve an operation label that may already contain a receiver type.
///
/// Dataflow nodes preserve labels such as `RmsNorm::forward`; this helper keeps their exact
/// method rule instead of collapsing them back to the ambiguous bare name `forward`.
pub fn lookup_resolved(op: &str, candle_nn_version: Option<&str>) -> OpEffect {
    if let Some((receiver, method)) = op.rsplit_once("::") {
        let method_effect = lookup_method(receiver, method, candle_nn_version);
        if !matches!(method_effect.dtype, DtypeRule::Unknown)
            || !matches!(method_effect.grad, GradFlow::Unknown)
        {
            return method_effect;
        }
    }
    lookup_for(op, candle_nn_version)
}

/// Whether `op` is a same-dtype binary that should emit a conflict diagnostic on mismatch.
#[allow(dead_code)]
pub fn requires_same_dtype(op: &str) -> bool {
    matches!(lookup(op).dtype, DtypeRule::SameAsInputs)
}

/// Whether `op` is a known candle-nn helper that severs backward.
#[allow(dead_code)]
pub fn is_no_backward(op: &str) -> bool {
    matches!(lookup(op).grad, GradFlow::Severs)
}
