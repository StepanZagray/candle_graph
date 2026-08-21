//! Typed coverage and capability states used across capture, analysis, and comparison.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Domain separator for ordered gradient-manifest digests.
pub const GRADIENT_MANIFEST_SCHEMA: &str = "candle-graph/gradient-manifest/1";

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

/// One parameter expected in a complete gradient capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedGradient {
    pub root: String,
    pub key: String,
    pub family: String,
}

impl ExpectedGradient {
    pub fn new(root: impl Into<String>, key: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            key: key.into(),
            family: family.into(),
        }
    }
}

/// Expected runtime state of one caller-defined parameter family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientFamilyExpectation {
    Active,
    Inactive,
    /// The caller's data determines whether this family is missing, attached
    /// with an exact zero, or present. If any members are present, at least
    /// `min_present` must be present; zero alone is not a structural failure.
    DataConditional,
}

/// Family-level expectation applied after exact parameter-manifest validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradientFamilyContract {
    pub family: String,
    pub expectation: GradientFamilyExpectation,
    pub min_present: usize,
}

impl GradientFamilyContract {
    pub fn active(family: impl Into<String>, min_present: usize) -> Self {
        Self {
            family: family.into(),
            expectation: GradientFamilyExpectation::Active,
            min_present,
        }
    }

    pub fn inactive(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            expectation: GradientFamilyExpectation::Inactive,
            min_present: 0,
        }
    }

    pub fn data_conditional(family: impl Into<String>, min_present: usize) -> Self {
        Self {
            family: family.into(),
            expectation: GradientFamilyExpectation::DataConditional,
            min_present,
        }
    }
}

/// Exact, digest-bound gradient parameter manifest and family expectations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradientContract {
    pub manifest_sha256: String,
    pub expected: Vec<ExpectedGradient>,
    pub families: Vec<GradientFamilyContract>,
}

impl GradientContract {
    pub fn new(
        expected: Vec<ExpectedGradient>,
        families: Vec<GradientFamilyContract>,
    ) -> Result<Self> {
        let contract = Self {
            manifest_sha256: gradient_manifest_sha256(&expected),
            expected,
            families,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Validate a constructed or deserialized contract before trusting it.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.expected.is_empty(),
            "gradient manifest must not be empty"
        );
        ensure!(
            !self.families.is_empty(),
            "gradient families must not be empty"
        );
        ensure!(
            self.manifest_sha256 == gradient_manifest_sha256(&self.expected),
            "gradient manifest SHA-256 does not match its ordered parameter manifest"
        );

        let mut keys = BTreeSet::new();
        let mut members = BTreeMap::<&str, usize>::new();
        for parameter in &self.expected {
            ensure!(
                !parameter.root.trim().is_empty()
                    && !parameter.key.trim().is_empty()
                    && !parameter.family.trim().is_empty(),
                "gradient manifest root, key, and family must not be empty"
            );
            ensure!(
                keys.insert((parameter.root.as_str(), parameter.key.as_str())),
                "gradient manifest declares ({:?}, {:?}) more than once",
                parameter.root,
                parameter.key
            );
            *members.entry(parameter.family.as_str()).or_default() += 1;
        }

        let mut family_names = BTreeSet::new();
        for family in &self.families {
            ensure!(
                !family.family.trim().is_empty(),
                "gradient family name must not be empty"
            );
            ensure!(
                family_names.insert(family.family.as_str()),
                "gradient family {:?} is declared more than once",
                family.family
            );
            let member_count = members.get(family.family.as_str()).copied().unwrap_or(0);
            ensure!(
                member_count > 0,
                "gradient family {:?} has no manifest members",
                family.family
            );
            match family.expectation {
                GradientFamilyExpectation::Inactive => ensure!(
                    family.min_present == 0,
                    "inactive gradient family {:?} must have min_present=0",
                    family.family
                ),
                GradientFamilyExpectation::Active
                | GradientFamilyExpectation::DataConditional => ensure!(
                    family.min_present > 0 && family.min_present <= member_count,
                    "active or data-conditional gradient family {:?} needs min_present in 1..={member_count}",
                    family.family
                ),
            }
        }
        for family in members.keys() {
            ensure!(
                family_names.contains(family),
                "gradient manifest family {family:?} has no family contract"
            );
        }
        Ok(())
    }
}

fn gradient_manifest_sha256(expected: &[ExpectedGradient]) -> String {
    let mut digest = Sha256::new();
    digest.update(GRADIENT_MANIFEST_SCHEMA.as_bytes());
    digest.update([0]);
    for parameter in expected {
        for value in [&parameter.root, &parameter.key, &parameter.family] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureContract {
    pub measurement_scope: MeasurementScope,
    pub operations: CoverageLevel,
    #[serde(default)]
    pub tensors: CoverageLevel,
    #[serde(default)]
    pub gradients: CoverageLevel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_contract: Option<GradientContract>,
    pub logical_memory: CoverageLevel,
    pub physical_memory: CoverageLevel,
    pub device_timing: CoverageLevel,
    #[serde(default)]
    pub required_semantic_labels: Vec<String>,
}

impl CaptureContract {
    /// Validate relationships that must hold before a producer starts capture.
    pub fn validate(&self) -> Result<()> {
        match (self.gradients, self.gradient_contract.as_ref()) {
            (CoverageLevel::Complete, Some(contract)) => contract.validate(),
            (CoverageLevel::Complete, None) => {
                anyhow::bail!("complete gradient coverage requires an exact gradient contract")
            }
            (_, Some(_)) => anyhow::bail!(
                "an exact gradient contract requires complete declared gradient coverage"
            ),
            (_, None) => Ok(()),
        }
    }
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
