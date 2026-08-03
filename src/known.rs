//! Ground truth for the candle-nn constructors that register parameters.
//!
//! Every entry here was read out of candle-nn 0.11.0 source rather than recalled, because the
//! whole output's correctness rests on these names matching what `VarBuilder::get` actually
//! stores. Citations are to
//! `~/.cargo/registry/src/*/candle-nn-0.11.0/src/`.

use serde::Serialize;

/// What a leaf tensor is for. Kept coarse on purpose: it drives display grouping and the
/// "is this trainable" question, not arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ParamKind {
    Weight,
    Bias,
    RunningMean,
    RunningVar,
    /// Registered through a raw `vb.get(..)` / `vb.get_with_hints(..)` call in user code.
    Raw,
}

/// One leaf tensor a constructor registers, relative to the `VarBuilder` it was handed.
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    pub name: &'static str,
    pub kind: ParamKind,
    /// False when the tensor is only registered for some configurations (e.g. `layer_norm`'s
    /// bias, which exists only when `affine`). Callers must mark these conditional unless they
    /// can resolve the config.
    pub unconditional: bool,
    /// True when the tensor *name* is formatted from the constructor's config argument rather
    /// than being a fixed literal. `LSTM::new` builds `weight_ih_l{layer_idx}{direction}`
    /// (rnn.rs:139-147), so [`Leaf::name`] is only the correct key for the default config.
    /// Callers that cannot resolve the config must emit a name family, not this literal.
    pub config_named: bool,
}

const fn leaf(name: &'static str, kind: ParamKind) -> Leaf {
    Leaf {
        name,
        kind,
        unconditional: true,
        config_named: false,
    }
}

const fn cond(name: &'static str, kind: ParamKind) -> Leaf {
    Leaf {
        name,
        kind,
        unconditional: false,
        config_named: false,
    }
}

/// A leaf whose name is derived from the config argument; `name` is the default-config spelling.
const fn named_by_config(name: &'static str, kind: ParamKind, unconditional: bool) -> Leaf {
    Leaf {
        name,
        kind,
        unconditional,
        config_named: true,
    }
}

/// A candle-nn constructor whose parameter layout we know exactly.
#[derive(Debug, Clone, Copy)]
pub struct Constructor {
    /// Last path segment of the call, e.g. `linear` for `candle_nn::linear` or `nn::linear`.
    pub func: &'static str,
    /// Zero-based index of the `VarBuilder` argument.
    pub vb_arg: usize,
    pub leaves: &'static [Leaf],
    /// Source citation, carried into diagnostics so a reader can check us.
    pub cite: &'static str,
}

/// `linear` and friends: `weight` always, `bias` only in the biased variants.
/// linear.rs:84-95 (`linear`), :97-101 (`linear_no_bias`), :103 (`linear_b`).
const LINEAR: &[Leaf] = &[
    leaf("weight", ParamKind::Weight),
    leaf("bias", ParamKind::Bias),
];
const LINEAR_NO_BIAS: &[Leaf] = &[leaf("weight", ParamKind::Weight)];
/// `linear_b(.., bias: bool, ..)` — the bias is config-dependent.
const LINEAR_B: &[Leaf] = &[
    leaf("weight", ParamKind::Weight),
    cond("bias", ParamKind::Bias),
];

/// embedding.rs:39-49 — the tensor is named "weight" despite the binding being `embeddings`.
const EMBEDDING: &[Leaf] = &[leaf("weight", ParamKind::Weight)];

/// layer_norm.rs:146-163. `weight` is unconditional; `bias` exists only when `config.affine`.
/// Note `impl From<f64> for LayerNormConfig` (:52-59) sets `affine: true`, so the common
/// `layer_norm(dim, 1e-5, vb)` form does register a bias.
const LAYER_NORM: &[Leaf] = &[
    leaf("weight", ParamKind::Weight),
    cond("bias", ParamKind::Bias),
];
/// layer_norm.rs:166 — explicitly no bias.
const LAYER_NORM_NO_BIAS: &[Leaf] = &[leaf("weight", ParamKind::Weight)];
/// layer_norm.rs:212-219 builds a `LayerNormConfig { affine: false, .. }`, so weight only.
const RMS_NORM: &[Leaf] = &[leaf("weight", ParamKind::Weight)];

/// group_norm.rs:76-84 — affine weight and bias are always registered.
const GROUP_NORM: &[Leaf] = &[
    leaf("weight", ParamKind::Weight),
    leaf("bias", ParamKind::Bias),
];

/// activation.rs:104-108 — PReLU always stores one scalar or one value per channel as `weight`.
const PRELU: &[Leaf] = &[leaf("weight", ParamKind::Weight)];

/// conv.rs:307-327 (`conv1d`), :386-410 (`conv2d`), and the transpose/no-bias variants.
const CONV: &[Leaf] = &[
    leaf("weight", ParamKind::Weight),
    leaf("bias", ParamKind::Bias),
];
const CONV_NO_BIAS: &[Leaf] = &[leaf("weight", ParamKind::Weight)];

/// rnn.rs:304-343 (`GRU::new`). Names are hard-coded `_l0` literals — `GRUConfig` has no
/// layer/direction fields (rnn.rs:257-262) — so the keys are always exactly these. Both biases
/// are conditional: `GRUConfig::default()` (rnn.rs:264) supplies them, `default_no_bias()`
/// (rnn.rs:276) sets both inits to `None` and registers nothing.
const GRU: &[Leaf] = &[
    leaf("weight_ih_l0", ParamKind::Weight),
    leaf("weight_hh_l0", ParamKind::Weight),
    cond("bias_ih_l0", ParamKind::Bias),
    cond("bias_hh_l0", ParamKind::Bias),
];

/// rnn.rs:134-187 (`LSTM::new`). Unlike GRU, every name is built with `format!` from the config:
/// `weight_ih_l{layer_idx}{direction}` (rnn.rs:145-147), where `direction` is `""` for
/// `Direction::Forward` and `"_reverse"` for `Backward` (rnn.rs:141-144). The literals below are
/// therefore only the *default-config* spelling and are marked `config_named` so an unresolved
/// config produces a name family instead of a wrong key.
const LSTM: &[Leaf] = &[
    named_by_config("weight_ih_l0", ParamKind::Weight, true),
    named_by_config("weight_hh_l0", ParamKind::Weight, true),
    named_by_config("bias_ih_l0", ParamKind::Bias, false),
    named_by_config("bias_hh_l0", ParamKind::Bias, false),
];

/// batch_norm.rs:301-317. Running stats are always registered; affine params are conditional.
/// These tensors are commonly excluded from optimizer variable lists, so flagging them
/// distinctly is worth the extra kinds.
const BATCH_NORM: &[Leaf] = &[
    leaf("running_mean", ParamKind::RunningMean),
    leaf("running_var", ParamKind::RunningVar),
    cond("weight", ParamKind::Weight),
    cond("bias", ParamKind::Bias),
];

pub const CONSTRUCTORS: &[Constructor] = &[
    Constructor {
        func: "linear",
        vb_arg: 2,
        leaves: LINEAR,
        cite: "linear.rs:84",
    },
    Constructor {
        func: "linear_no_bias",
        vb_arg: 2,
        leaves: LINEAR_NO_BIAS,
        cite: "linear.rs:97",
    },
    Constructor {
        func: "linear_b",
        vb_arg: 3,
        leaves: LINEAR_B,
        cite: "linear.rs:103",
    },
    Constructor {
        func: "embedding",
        vb_arg: 2,
        leaves: EMBEDDING,
        cite: "embedding.rs:39",
    },
    Constructor {
        func: "layer_norm",
        vb_arg: 2,
        leaves: LAYER_NORM,
        cite: "layer_norm.rs:146",
    },
    Constructor {
        func: "layer_norm_no_bias",
        vb_arg: 2,
        leaves: LAYER_NORM_NO_BIAS,
        cite: "layer_norm.rs:166",
    },
    Constructor {
        func: "rms_norm",
        vb_arg: 2,
        leaves: RMS_NORM,
        cite: "layer_norm.rs:212",
    },
    Constructor {
        func: "group_norm",
        vb_arg: 3,
        leaves: GROUP_NORM,
        cite: "group_norm.rs:76",
    },
    Constructor {
        func: "prelu",
        vb_arg: 1,
        leaves: PRELU,
        cite: "activation.rs:104",
    },
    Constructor {
        func: "conv1d",
        vb_arg: 4,
        leaves: CONV,
        cite: "conv.rs:307",
    },
    Constructor {
        func: "conv1d_no_bias",
        vb_arg: 4,
        leaves: CONV_NO_BIAS,
        cite: "conv.rs:329",
    },
    Constructor {
        func: "conv2d",
        vb_arg: 4,
        leaves: CONV,
        cite: "conv.rs:386",
    },
    Constructor {
        func: "conv2d_no_bias",
        vb_arg: 4,
        leaves: CONV_NO_BIAS,
        cite: "conv.rs:413",
    },
    Constructor {
        func: "conv_transpose1d",
        vb_arg: 4,
        leaves: CONV,
        cite: "conv.rs:345",
    },
    Constructor {
        func: "conv_transpose1d_no_bias",
        vb_arg: 4,
        leaves: CONV_NO_BIAS,
        cite: "conv.rs:366",
    },
    Constructor {
        func: "conv_transpose2d",
        vb_arg: 4,
        leaves: CONV,
        cite: "conv.rs:434",
    },
    Constructor {
        func: "conv_transpose2d_no_bias",
        vb_arg: 4,
        leaves: CONV_NO_BIAS,
        cite: "conv.rs:455",
    },
    Constructor {
        func: "batch_norm",
        vb_arg: 2,
        leaves: BATCH_NORM,
        cite: "batch_norm.rs:301",
    },
    Constructor {
        func: "gru",
        vb_arg: 3,
        leaves: GRU,
        cite: "rnn.rs:345",
    },
    Constructor {
        func: "lstm",
        vb_arg: 3,
        leaves: LSTM,
        cite: "rnn.rs:189",
    },
];

pub fn lookup(func: &str) -> Option<&'static Constructor> {
    CONSTRUCTORS.iter().find(|c| c.func == func)
}

/// `VarBuilder` methods that register a tensor directly, with the index of the name argument.
/// var_builder.rs:203 (`get_with_hints`), :213 (`get`), :218 (`get_unchecked`).
pub fn raw_get_name_arg(method: &str) -> Option<usize> {
    match method {
        "get" => Some(1),
        "get_with_hints" => Some(1),
        "get_with_hints_dtype" => Some(1),
        "get_unchecked" => Some(0),
        "get_unchecked_dtype" => Some(0),
        _ => None,
    }
}

/// `VarBuilder` methods that return a *re-prefixed* builder rather than a tensor.
/// var_builder.rs:162 (`pp`), :150 (`push_prefix`), :139 (`set_prefix`), :129 (`root`).
pub fn prefix_method(method: &str) -> Option<PrefixOp> {
    match method {
        "pp" | "push_prefix" => Some(PrefixOp::Push),
        "set_prefix" => Some(PrefixOp::Replace),
        "root" => Some(PrefixOp::Root),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixOp {
    Push,
    Replace,
    Root,
}
