//! Train vs inference execution phase for trace metadata.

use serde::{Deserialize, Serialize};

/// Whether profiling targets training (autograd) or inference (no-grad).
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

/// Training-step slice inside a probe run (PyTorch profiler timeline phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStep {
    Forward,
    Backward,
    Optimizer,
}

impl ExecutionStep {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Optimizer => "optimizer",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.replace('-', "_").to_ascii_lowercase().as_str() {
            "forward" | "fwd" => Some(Self::Forward),
            "backward" | "backward_pass" | "bwd" | "backprop" => Some(Self::Backward),
            "optimizer" | "optim" | "step" => Some(Self::Optimizer),
            _ => None,
        }
    }
}
