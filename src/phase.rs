//! Train vs inference execution phase for static graphs and runtime traces.

use serde::{Deserialize, Serialize};

/// Whether analysis or profiling targets training (autograd) or inference (no-grad).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Train,
    Infer,
}

impl ExecutionPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Infer => "infer",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.replace('-', "_").to_ascii_lowercase().as_str() {
            "train" | "training" => Some(Self::Train),
            "infer" | "inference" | "eval" | "evaluate" => Some(Self::Infer),
            _ => None,
        }
    }
}

/// Classify which static graphs to build for an entrypoint.
pub fn entrypoint_phases(name: &str, qualified_name: &str, is_loss: bool) -> Vec<ExecutionPhase> {
    if is_loss {
        return vec![ExecutionPhase::Train];
    }
    let lower = format!("{name} {qualified_name}").to_ascii_lowercase();
    let infer_hint = lower.contains("eval")
        || lower.contains("infer")
        || lower.contains("predict")
        || lower.contains("inference");
    let train_hint = lower.contains("train")
        || lower.contains("loss")
        || lower.contains("backward")
        || lower.contains("optim");
    if name == "forward" || name == "forward_t" {
        return vec![ExecutionPhase::Train, ExecutionPhase::Infer];
    }
    if infer_hint && train_hint {
        return vec![ExecutionPhase::Train, ExecutionPhase::Infer];
    }
    if infer_hint {
        return vec![ExecutionPhase::Infer];
    }
    if train_hint {
        return vec![ExecutionPhase::Train];
    }
    vec![ExecutionPhase::Train]
}
