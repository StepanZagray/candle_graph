//! Typed coverage and capability states used across capture, analysis, and comparison.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageLevel {
    #[default]
    None,
    Partial,
    Complete,
}

impl CoverageLevel {
    pub fn with_observations(self, count: usize) -> Self {
        if count == 0 {
            self
        } else {
            self.max(Self::Partial)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementScope {
    #[default]
    Unknown,
    ProfiledWork,
    ProductionEquivalent,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureContract {
    pub measurement_scope: MeasurementScope,
    pub operations: CoverageLevel,
    #[serde(default)]
    pub tensors: CoverageLevel,
    #[serde(default)]
    pub gradients: CoverageLevel,
    pub logical_memory: CoverageLevel,
    pub physical_memory: CoverageLevel,
    pub device_timing: CoverageLevel,
    #[serde(default)]
    pub required_semantic_labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    Unavailable,
    Partial,
    Complete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityState {
    pub level: CapabilityLevel,
    pub source: String,
    pub reason: String,
}

impl Default for CapabilityState {
    fn default() -> Self {
        Self::unavailable("capability was not recorded by this evidence schema")
    }
}

impl CapabilityState {
    pub fn invalid(source: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            level: CapabilityLevel::Invalid,
            source: source.into(),
            reason: reason.into(),
        }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            level: CapabilityLevel::Unavailable,
            source: "none".into(),
            reason: reason.into(),
        }
    }

    pub fn from_coverage(
        coverage: CoverageLevel,
        source: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            level: match coverage {
                CoverageLevel::None => CapabilityLevel::Unavailable,
                CoverageLevel::Partial => CapabilityLevel::Partial,
                CoverageLevel::Complete => CapabilityLevel::Complete,
            },
            source: source.into(),
            reason: reason.into(),
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(
            self.level,
            CapabilityLevel::Partial | CapabilityLevel::Complete
        )
    }

    pub fn is_complete(&self) -> bool {
        self.level == CapabilityLevel::Complete
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCapabilities {
    pub structural_trace: CapabilityState,
    pub outer_wall_time: CapabilityState,
    pub nested_host_time: CapabilityState,
    pub nested_device_time: CapabilityState,
    pub operation_coverage: CapabilityState,
    #[serde(default)]
    pub tensor_coverage: CapabilityState,
    #[serde(default)]
    pub gradient_coverage: CapabilityState,
    pub logical_memory_coverage: CapabilityState,
    pub physical_memory_coverage: CapabilityState,
    pub gpu_correlation: CapabilityState,
    pub provenance_binding: CapabilityState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    StructuralTrace,
    OuterWallTime,
    NestedHostTime,
    NestedDeviceTime,
    Operations,
    Tensors,
    Gradients,
    LogicalMemory,
    PhysicalMemory,
    GpuCorrelation,
    ProvenanceBinding,
}

impl EvidenceCapabilities {
    pub fn get(&self, kind: CapabilityKind) -> &CapabilityState {
        match kind {
            CapabilityKind::StructuralTrace => &self.structural_trace,
            CapabilityKind::OuterWallTime => &self.outer_wall_time,
            CapabilityKind::NestedHostTime => &self.nested_host_time,
            CapabilityKind::NestedDeviceTime => &self.nested_device_time,
            CapabilityKind::Operations => &self.operation_coverage,
            CapabilityKind::Tensors => &self.tensor_coverage,
            CapabilityKind::Gradients => &self.gradient_coverage,
            CapabilityKind::LogicalMemory => &self.logical_memory_coverage,
            CapabilityKind::PhysicalMemory => &self.physical_memory_coverage,
            CapabilityKind::GpuCorrelation => &self.gpu_correlation,
            CapabilityKind::ProvenanceBinding => &self.provenance_binding,
        }
    }
}
