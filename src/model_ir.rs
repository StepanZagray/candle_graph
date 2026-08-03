//! Unified, agent-oriented representation of a Candle model crate.
//!
//! The structure and expression analyzers intentionally keep their own compact arenas while
//! running. `ModelIr` is the durable interchange layer that joins those arenas with Cargo,
//! pipeline, optimizer, artifact, tensor-contract, and runtime evidence.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Versioned schema emitted by scans and consumed by the query/runtime layers.
pub const MODEL_IR_SCHEMA: &str = "candle-graph/model/1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StableId(pub String);

impl StableId {
    pub fn new(kind: &str, parts: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let mut value = String::from(kind);
        for part in parts {
            value.push(':');
            escape_id_part(part.as_ref(), &mut value);
        }
        Self(value)
    }
}

impl std::fmt::Display for StableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Source,
    Cargo,
    Checkpoint,
    Runtime,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Proven,
    Conditional,
    Heuristic,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub kind: EvidenceKind,
    pub confidence: Confidence,
    pub source: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Crate,
    Restricted,
    Private,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuilderRole {
    Trainable,
    Frozen,
    State,
    Conditional,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterRole {
    Optimized,
    Frozen,
    RunningState,
    Excluded,
    Conditional,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Input,
    Output,
    Parameter,
    Activation,
    Target,
    Mask,
    Loss,
    Cache,
    State,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceFact {
    Cpu,
    Cuda { ordinal: Option<u32> },
    Metal,
    SameAs(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutFact {
    Contiguous,
    NonContiguous,
    Strided,
    SameAs(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dimension {
    /// Stable semantic label when one is known (`batch`, `tokens`, `hidden`, ...).
    pub name: Option<String>,
    /// Literal or symbolic Rust expression.
    pub expr: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShapeFact {
    pub rank: Option<usize>,
    pub dimensions: Vec<Dimension>,
    pub source_expr: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub id: StableId,
    pub name: String,
    pub role: TensorRole,
    pub owner_function: StableId,
    pub parameter: Option<StableId>,
    pub shape: ShapeFact,
    pub dtype: String,
    pub device: DeviceFact,
    pub layout: LayoutFact,
    pub requires_grad: Option<bool>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuilderNamespace {
    pub name: String,
    pub role: BuilderRole,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Component {
    pub id: StableId,
    pub name: String,
    pub qualified_name: String,
    pub source: String,
    pub constructor: StableId,
    pub builders: Vec<BuilderNamespace>,
    pub modules: Vec<StableId>,
    pub parameters: Vec<StableId>,
    pub entrypoints: Vec<StableId>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureEdge {
    pub id: StableId,
    pub from: StableId,
    pub to: StableId,
    pub via_function: StableId,
    pub source: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Module {
    pub id: StableId,
    pub component: StableId,
    pub parent: Option<StableId>,
    pub type_name: String,
    pub qualified_type: Option<String>,
    pub field: Option<String>,
    pub builder_root: String,
    pub prefix: String,
    pub repeat: Option<String>,
    pub source: String,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    pub id: StableId,
    pub component: StableId,
    pub module: StableId,
    pub key: String,
    pub builder_root: String,
    pub role: ParameterRole,
    pub kind: String,
    pub symbolic_shape: Option<String>,
    pub checkpoint_shape: Option<Vec<usize>>,
    pub checkpoint_dtype: Option<String>,
    pub source: String,
    pub uses: Vec<StableId>,
    pub optimizer_memberships: Vec<StableId>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Function {
    pub id: StableId,
    pub name: String,
    pub qualified_name: String,
    pub owner_type: Option<String>,
    pub visibility: Visibility,
    pub parameters: Vec<FunctionParameter>,
    pub return_type: Option<String>,
    /// Source-level `#[cfg(...)]` predicates inherited by this definition.
    pub cfg_predicates: Vec<String>,
    /// Whether those predicates match the selected Cargo feature/target context.
    /// `None` means no Cargo context was available or a predicate was unsupported.
    pub cfg_active: Option<bool>,
    pub source: String,
    pub calls: Vec<StableId>,
    pub tensor_inputs: Vec<StableId>,
    pub tensor_outputs: Vec<StableId>,
    pub is_entrypoint: bool,
    pub is_loss: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionParameter {
    pub name: String,
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub id: StableId,
    pub function: StableId,
    pub name: String,
    pub qualified_name: Option<String>,
    pub inputs: Vec<StableId>,
    pub output: StableId,
    pub source: String,
    pub dtype_rule: String,
    pub gradient_rule: String,
    pub device_rule: String,
    pub shape_rule: String,
    /// Float-range transfer after rounding (`real`, `non_negative`, `saturating_unit`, …).
    #[serde(default)]
    pub domain_rule: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Prepare,
    Train,
    Evaluate,
    Export,
    Probe,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StageDispatchKind {
    #[default]
    Unknown,
    Inline,
    Subprocess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineStage {
    pub id: StableId,
    pub name: String,
    pub kind: StageKind,
    pub function: StableId,
    pub order: Option<usize>,
    pub components: Vec<StableId>,
    pub consumes: Vec<StableId>,
    pub produces: Vec<StableId>,
    pub depends_on: Vec<StableId>,
    pub source: String,
    pub evidence: Vec<Evidence>,
    #[serde(default)]
    pub dispatch: StageDispatchKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subprocess_key: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cli_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launcher: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orchestrator: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Checkpoint,
    OptimizerState,
    Dataset,
    Vocabulary,
    Cache,
    EvaluationReport,
    Configuration,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub id: StableId,
    pub name: String,
    pub kind: ArtifactKind,
    pub path_expr: String,
    pub produced_by: Vec<StableId>,
    pub consumed_by: Vec<StableId>,
    pub source: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuilderSourceKind {
    VarMap,
    MmapSafetensors,
    FromTensors,
    BufferedSafetensors,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssemblySite {
    pub id: StableId,
    pub function: StableId,
    pub function_name: String,
    pub component: StableId,
    pub component_name: String,
    pub builder_root: String,
    pub prefix_chain: Vec<String>,
    pub varmap: Option<String>,
    pub source_kind: BuilderSourceKind,
    pub role: BuilderRole,
    pub checkpoint_load: Option<String>,
    pub source: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerMembership {
    pub id: StableId,
    pub stage: StableId,
    pub optimizer: String,
    pub varmap: String,
    pub components: Vec<StableId>,
    pub builder_roots: Vec<String>,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub conditional: Option<String>,
    pub source: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Error,
    Warning,
    Information,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    pub id: StableId,
    pub rule: String,
    pub severity: FindingSeverity,
    pub confidence: Confidence,
    pub message: String,
    pub source: Option<String>,
    pub related: Vec<StableId>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCoverage {
    pub components: usize,
    pub architecture_edges: usize,
    pub modules: usize,
    pub parameters: usize,
    pub functions: usize,
    pub entrypoints: usize,
    pub component_entrypoints: usize,
    pub composition_edges: usize,
    pub assembly_sites: usize,
    pub subprocess_stages: usize,
    pub tensors: usize,
    pub operations: usize,
    pub pipeline_stages: usize,
    pub artifacts: usize,
    pub optimizer_memberships: usize,
    pub linked_parameter_uses: usize,
    pub tensors_with_shape: usize,
    pub tensors_with_dtype: usize,
    pub tensors_with_device: usize,
    pub runtime_observations: usize,
    pub diagnostics: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CargoSummary {
    /// Deterministic identity of the exact Cargo configuration used for this scan.
    #[serde(default)]
    pub build_id: String,
    pub workspace_root: String,
    pub manifest_path: String,
    pub package_name: String,
    pub package_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_target: Option<String>,
    pub active_features: Vec<String>,
    pub active_cfg: Vec<String>,
    pub candle_packages: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSummary {
    pub trace_schema: String,
    pub entrypoint: Option<String>,
    pub profile: Option<String>,
    pub tensor_observations: usize,
    #[serde(default)]
    pub operation_observations: usize,
    pub gradient_observations: usize,
    pub missing_gradients: usize,
    pub zero_gradients: usize,
    pub non_finite_gradients: usize,
    #[serde(default)]
    pub tensor_conflicts: usize,
    #[serde(default)]
    pub gradient_conflicts: usize,
    #[serde(default)]
    pub identity_mismatches: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_non_finite_step: Option<u64>,
    #[serde(default)]
    pub saturating_activations: usize,
    #[serde(default)]
    pub value_observations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIr {
    pub schema: String,
    pub analysis_id: StableId,
    pub cargo: Option<CargoSummary>,
    pub coverage: ModelCoverage,
    pub components: Vec<Component>,
    pub architecture_edges: Vec<ArchitectureEdge>,
    pub modules: Vec<Module>,
    pub parameters: Vec<Parameter>,
    pub functions: Vec<Function>,
    pub tensors: Vec<TensorContract>,
    pub operations: Vec<Operation>,
    pub stages: Vec<PipelineStage>,
    pub artifacts: Vec<Artifact>,
    pub optimizers: Vec<OptimizerMembership>,
    pub assembly_sites: Vec<AssemblySite>,
    pub findings: Vec<Finding>,
    pub runtime: Option<RuntimeSummary>,
}

impl ModelIr {
    pub fn empty(analysis_id: StableId) -> Self {
        Self {
            schema: MODEL_IR_SCHEMA.to_string(),
            analysis_id,
            cargo: None,
            coverage: ModelCoverage::default(),
            components: Vec::new(),
            architecture_edges: Vec::new(),
            modules: Vec::new(),
            parameters: Vec::new(),
            functions: Vec::new(),
            tensors: Vec::new(),
            operations: Vec::new(),
            stages: Vec::new(),
            artifacts: Vec::new(),
            optimizers: Vec::new(),
            assembly_sites: Vec::new(),
            findings: Vec::new(),
            runtime: None,
        }
    }

    pub fn normalize(&mut self) {
        self.components.sort_by(|a, b| a.id.cmp(&b.id));
        self.architecture_edges.sort_by(|a, b| a.id.cmp(&b.id));
        self.modules.sort_by(|a, b| a.id.cmp(&b.id));
        self.parameters.sort_by(|a, b| a.id.cmp(&b.id));
        self.functions.sort_by(|a, b| a.id.cmp(&b.id));
        self.tensors.sort_by(|a, b| a.id.cmp(&b.id));
        self.operations.sort_by(|a, b| a.id.cmp(&b.id));
        self.stages.sort_by(|a, b| {
            (a.order.unwrap_or(usize::MAX), &a.id).cmp(&(b.order.unwrap_or(usize::MAX), &b.id))
        });
        self.artifacts.sort_by(|a, b| a.id.cmp(&b.id));
        self.optimizers.sort_by(|a, b| a.id.cmp(&b.id));
        self.findings.sort_by(|a, b| a.id.cmp(&b.id));
        self.components.dedup_by(|a, b| a.id == b.id);
        self.architecture_edges.dedup_by(|a, b| a.id == b.id);
        self.modules.dedup_by(|a, b| a.id == b.id);
        self.parameters.dedup_by(|a, b| a.id == b.id);
        self.functions.dedup_by(|a, b| a.id == b.id);
        self.tensors.dedup_by(|a, b| a.id == b.id);
        self.operations.dedup_by(|a, b| a.id == b.id);
        self.artifacts.dedup_by(|a, b| a.id == b.id);
        self.optimizers.dedup_by(|a, b| a.id == b.id);
        let mut merged_findings: Vec<Finding> = Vec::with_capacity(self.findings.len());
        for finding in self.findings.drain(..) {
            if let Some(existing) = merged_findings
                .last_mut()
                .filter(|existing| existing.id == finding.id)
            {
                existing.related.extend(finding.related);
                existing.evidence.extend(finding.evidence);
            } else {
                merged_findings.push(finding);
            }
        }
        self.findings = merged_findings;
        for component in &mut self.components {
            component.modules.sort();
            component.modules.dedup();
            component.parameters.sort();
            component.parameters.dedup();
            component.entrypoints.sort();
            component.entrypoints.dedup();
            component.builders.sort_by(|a, b| a.name.cmp(&b.name));
        }
        for function in &mut self.functions {
            function.calls.sort();
            function.calls.dedup();
            function.tensor_inputs.sort();
            function.tensor_inputs.dedup();
            function.tensor_outputs.sort();
            function.tensor_outputs.dedup();
        }
        for parameter in &mut self.parameters {
            parameter.uses.sort();
            parameter.uses.dedup();
            parameter.optimizer_memberships.sort();
            parameter.optimizer_memberships.dedup();
        }
        for finding in &mut self.findings {
            finding.related.sort();
            finding.related.dedup();
            finding
                .evidence
                .sort_by(|a, b| (&a.source, &a.detail).cmp(&(&b.source, &b.detail)));
            finding
                .evidence
                .dedup_by(|a, b| a.source == b.source && a.detail == b.detail);
        }
        self.refresh_coverage();
    }

    pub fn refresh_coverage(&mut self) {
        self.coverage.components = self.components.len();
        self.coverage.architecture_edges = self.architecture_edges.len();
        self.coverage.modules = self.modules.len();
        self.coverage.parameters = self.parameters.len();
        self.coverage.functions = self.functions.len();
        self.coverage.entrypoints = self.functions.iter().filter(|f| f.is_entrypoint).count();
        let component_types: BTreeMap<String, ()> = self
            .components
            .iter()
            .flat_map(|component| [component.name.clone(), component.qualified_name.clone()])
            .map(|name| (name, ()))
            .collect();
        self.coverage.component_entrypoints = self
            .functions
            .iter()
            .filter(|function| {
                function.is_entrypoint
                    && function
                        .owner_type
                        .as_ref()
                        .is_some_and(|owner| component_types.contains_key(owner))
            })
            .count();
        self.coverage.composition_edges = self
            .architecture_edges
            .iter()
            .filter(|edge| edge.id.0.starts_with("composition-edge:"))
            .count();
        self.coverage.assembly_sites = self.assembly_sites.len();
        self.coverage.subprocess_stages = self
            .stages
            .iter()
            .filter(|stage| stage.dispatch == StageDispatchKind::Subprocess)
            .count();
        self.coverage.tensors = self.tensors.len();
        self.coverage.operations = self.operations.len();
        self.coverage.pipeline_stages = self.stages.len();
        self.coverage.artifacts = self.artifacts.len();
        self.coverage.optimizer_memberships = self.optimizers.len();
        self.coverage.linked_parameter_uses =
            self.parameters.iter().map(|p| p.uses.len()).sum::<usize>();
        self.coverage.tensors_with_shape = self
            .tensors
            .iter()
            .filter(|t| t.shape.rank.is_some() || !t.shape.dimensions.is_empty())
            .count();
        self.coverage.tensors_with_dtype =
            self.tensors.iter().filter(|t| t.dtype != "Unknown").count();
        self.coverage.tensors_with_device = self
            .tensors
            .iter()
            .filter(|t| !matches!(t.device, DeviceFact::Unknown))
            .count();
        self.coverage.diagnostics = self.findings.len();
    }
}

fn escape_id_part(part: &str, out: &mut String) {
    for byte in part.bytes() {
        match byte {
            b'%' | b':' | b'/' | b'\\' | b' ' => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
            _ => out.push(byte as char),
        }
    }
}
