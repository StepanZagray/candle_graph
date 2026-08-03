//! Bounded runtime evidence and gradient-audit protocol.
//!
//! Transport/import layer for target-side probe emissions. Schema:
//! `candle-graph/runtime/1`. No Candle dependency.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::io::Write;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};


/// Schema identifier written into every document and expected on parse.
pub const SCHEMA: &str = "candle-graph/runtime/1";
pub const SCHEMA_V2: &str = "candle-graph/runtime/2";
pub const SCHEMA_V3: &str = "candle-graph/runtime/3";

/// Build / process metadata for a single probe run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Analyzed or probed entrypoint, e.g. `train` or `Model::forward`.
    pub entrypoint: String,
    /// Cargo profile, e.g. `debug` or `release`.
    pub profile: String,
    /// Enabled Cargo features at probe time.
    #[serde(default)]
    pub cargo_features: Vec<String>,
    /// Relevant `cfg` flags observed by the probe.
    #[serde(default)]
    pub cfg: Vec<String>,
    /// Optional analysis identity this trace was captured against (e.g. model IR `analysis_id`).
    /// Omitted for backward compatibility; when present, importers must check it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_id: Option<String>,
    /// Optional build identity (package/features/profile fingerprint) for the probed binary.
    /// Omitted for backward compatibility; when present, importers must check it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_id: Option<String>,
    /// Train vs inference phase for this trace (`train` / `infer`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// Expected analysis/build identity supplied by an importer when correlating a trace.
///
/// Only fields that are `Some` on **both** this value and the trace metadata are compared.
/// Omitted optional identity on either side is compatible (backward compatible).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedIdentity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_id: Option<String>,
}

/// Which optional identity field disagreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityField {
    AnalysisId,
    BuildId,
}

impl fmt::Display for IdentityField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnalysisId => write!(f, "analysis_id"),
            Self::BuildId => write!(f, "build_id"),
        }
    }
}

/// Mismatch between trace metadata and an importer-supplied expected identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityMismatch {
    pub field: IdentityField,
    pub expected: String,
    pub observed: String,
}

/// Confidence for a refined runtime observation.
///
/// Importers must not treat [`ObservationConfidence::Unknown`] conflicts as proven facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationConfidence {
    /// Observations for this identity agree.
    Proven,
    /// Observations disagree, identity does not match, or evidence is otherwise insufficient.
    Unknown,
}

/// One tensor snapshot emitted by the probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorObservation {
    /// Stable event identity within a trace; used for duplicate detection.
    pub event_id: String,
    /// Optional stable static identity shared across related observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_id: Option<String>,
    /// Optional source location or label from the probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional training step index when the trace is a time series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    /// Whether storage is contiguous (layout fact).
    pub contiguous: bool,
    pub requires_grad: bool,
    /// Optional opaque storage identity from the runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_id: Option<String>,
}

/// One operation observation emitted by the probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationObservation {
    pub event_id: String,
    /// Operation name / kind, e.g. `matmul`, `add`.
    pub op: String,
    /// Optional stable static identity for correlating with static analysis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_id: Option<String>,
    /// Optional source location or label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Input tensor event or storage ids observed at the call.
    #[serde(default)]
    pub inputs: Vec<String>,
    /// Output tensor event or storage id, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Optional training step index when the trace is a time series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    /// Wall time spent in this operation (nanoseconds), when profiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ns: Option<u64>,
}

/// Observed data-flow edge timing between two static ids (profiler output).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeTimingObservation {
    pub event_id: String,
    pub from_static_id: String,
    pub to_static_id: String,
    pub duration_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
}

/// Builder-root + parameter-key identity for gradient facts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParamIdentity {
    /// Builder root namespace (e.g. `vb`), matching static analysis roots.
    pub root: String,
    /// Parameter key under that root.
    pub key: String,
}

/// Observed gradient presence / quality for one parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientState {
    Present,
    Missing,
    Zero,
    NonFinite,
}

impl fmt::Display for GradientState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Present => write!(f, "present"),
            Self::Missing => write!(f, "missing"),
            Self::Zero => write!(f, "zero"),
            Self::NonFinite => write!(f, "non_finite"),
        }
    }
}

/// Per-parameter gradient fact keyed by builder root + parameter key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientFact {
    pub event_id: String,
    pub root: String,
    pub key: String,
    pub state: GradientState,
    /// Optional training step index when the trace is a time series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    /// Optional L2 (or probe-defined) norm; must be consistent with [`GradientState`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<f64>,
}

impl GradientFact {
    pub fn identity(&self) -> ParamIdentity {
        ParamIdentity {
            root: self.root.clone(),
            key: self.key.clone(),
        }
    }

    /// Returns an error when state and optional norm contradict each other.
    pub fn validate(&self) -> Result<()> {
        match self.state {
            GradientState::Missing => {
                if self.norm.is_some() {
                    bail!(
                        "invalid gradient fact {}:{}: missing state must not carry a norm",
                        self.root,
                        self.key
                    );
                }
            }
            GradientState::Zero => {
                if let Some(n) = self.norm {
                    if n != 0.0 || !n.is_finite() {
                        bail!(
                            "invalid gradient fact {}:{}: zero state requires norm 0.0 when set, got {n}",
                            self.root,
                            self.key
                        );
                    }
                }
            }
            GradientState::Present => {
                if let Some(n) = self.norm {
                    if !n.is_finite() {
                        bail!(
                            "invalid gradient fact {}:{}: present state cannot have non-finite norm",
                            self.root,
                            self.key
                        );
                    }
                    if n == 0.0 {
                        bail!(
                            "invalid gradient fact {}:{}: present state cannot have zero norm (use zero)",
                            self.root,
                            self.key
                        );
                    }
                }
            }
            GradientState::NonFinite => {
                if let Some(n) = self.norm {
                    if n.is_finite() {
                        bail!(
                            "invalid gradient fact {}:{}: non_finite state cannot have finite norm {n}",
                            self.root,
                            self.key
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Forward value-range observation for numeric-domain runtime audits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueObservation {
    pub event_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    pub min: f64,
    pub max: f64,
    pub abs_max: f64,
    #[serde(default)]
    pub nonfinite_count: u64,
    #[serde(default)]
    pub saturated_count: u64,
}

/// Full runtime trace document (`candle-graph/runtime/1` or `/2`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTrace {
    pub schema: String,
    pub run: RunMetadata,
    #[serde(default)]
    pub tensors: Vec<TensorObservation>,
    #[serde(default)]
    pub operations: Vec<OperationObservation>,
    #[serde(default)]
    pub gradients: Vec<GradientFact>,
    #[serde(default)]
    pub values: Vec<ValueObservation>,
    #[serde(default)]
    pub edge_timings: Vec<EdgeTimingObservation>,
}

/// JSONL / streaming event records that assemble into a [`RuntimeTrace`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEvent {
    /// Declares schema + run metadata (at most one per stream).
    Meta {
        schema: String,
        #[serde(flatten)]
        run: RunMetadata,
    },
    Tensor(TensorObservation),
    Operation(OperationObservation),
    Gradient(GradientFact),
    Value(ValueObservation),
    EdgeTiming(EdgeTimingObservation),
}

/// Streaming JSONL emitter for instrumented CPU or synthetic rollouts.
///
/// The writer emits metadata immediately, rejects duplicate event ids before writing an event,
/// and uses the same validation rules as the importer. Tensor callers should populate
/// [`TensorObservation::static_id`] with an id copied from the model IR or a `tensor` query.
pub struct RuntimeTraceWriter<W: Write> {
    writer: W,
    seen_event_ids: HashSet<String>,
}

impl<W: Write> RuntimeTraceWriter<W> {
    /// Start a JSONL stream by writing its required metadata event (schema v1).
    pub fn new(writer: W, run: RunMetadata) -> Result<Self> {
        Self::new_with_schema(writer, SCHEMA, run)
    }

    /// Start a JSONL stream with an explicit schema (`/1`, `/2`, or `/3`).
    pub fn new_with_schema(writer: W, schema: &str, run: RunMetadata) -> Result<Self> {
        let mut output = Self {
            writer,
            seen_event_ids: HashSet::new(),
        };
        output.write_event(&RuntimeEvent::Meta {
            schema: schema.to_string(),
            run,
        })?;
        Ok(output)
    }

    /// Emit one tensor observation. `static_id` is the static/runtime correlation key.
    pub fn tensor(&mut self, observation: TensorObservation) -> Result<()> {
        self.reserve_event_id(&observation.event_id)?;
        self.write_event(&RuntimeEvent::Tensor(observation))
    }

    /// Emit one operation observation.
    pub fn operation(&mut self, observation: OperationObservation) -> Result<()> {
        self.reserve_event_id(&observation.event_id)?;
        self.write_event(&RuntimeEvent::Operation(observation))
    }

    /// Emit one parameter-gradient fact after checking state/norm consistency.
    pub fn gradient(&mut self, fact: GradientFact) -> Result<()> {
        fact.validate()?;
        self.reserve_event_id(&fact.event_id)?;
        self.write_event(&RuntimeEvent::Gradient(fact))
    }

    /// Emit one forward value-range observation.
    pub fn value(&mut self, observation: ValueObservation) -> Result<()> {
        self.reserve_event_id(&observation.event_id)?;
        self.write_event(&RuntimeEvent::Value(observation))
    }

    /// Emit one data-flow edge timing observation (runtime v3).
    pub fn edge_timing(&mut self, observation: EdgeTimingObservation) -> Result<()> {
        self.reserve_event_id(&observation.event_id)?;
        self.write_event(&RuntimeEvent::EdgeTiming(observation))
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .context("flushing runtime JSONL trace")
    }

    /// Flush the stream and return the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        self.writer
            .flush()
            .context("flushing runtime JSONL trace")?;
        Ok(self.writer)
    }

    fn reserve_event_id(&mut self, event_id: &str) -> Result<()> {
        if event_id.is_empty() {
            bail!("empty event_id is not allowed");
        }
        if !self.seen_event_ids.insert(event_id.to_string()) {
            bail!("duplicate event_id `{event_id}`");
        }
        Ok(())
    }

    fn write_event(&mut self, event: &RuntimeEvent) -> Result<()> {
        let mut line = serde_json::to_vec(event).context("serializing runtime JSONL event")?;
        line.push(b'\n');
        self.writer
            .write_all(&line)
            .context("writing runtime JSONL event")
    }
}

/// Which tensor attribute disagreed across observations sharing a `static_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorConflictKind {
    Dtype,
    Device,
    Shape,
    Layout,
    RequiresGrad,
}

fn unknown_confidence() -> ObservationConfidence {
    ObservationConfidence::Unknown
}

/// Conflict among repeated tensor observations for one static id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorConflict {
    pub static_id: String,
    pub kind: TensorConflictKind,
    /// Distinct observed values, sorted.
    pub values: Vec<String>,
    /// Always [`ObservationConfidence::Unknown`]: do not refine as proven.
    #[serde(default = "unknown_confidence")]
    pub confidence: ObservationConfidence,
}

/// Conflict among gradient facts for one parameter identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradientConflict {
    pub identity: ParamIdentity,
    /// Distinct observed states, sorted.
    pub states: Vec<String>,
    /// Event ids that participated, sorted.
    pub event_ids: Vec<String>,
    /// Always [`ObservationConfidence::Unknown`]: do not pick a winning state.
    #[serde(default = "unknown_confidence")]
    pub confidence: ObservationConfidence,
}

/// Compact audit summary over a normalized trace.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeAudit {
    pub missing_gradients: Vec<ParamIdentity>,
    pub non_finite_gradients: Vec<ParamIdentity>,
    pub zero_gradients: Vec<ParamIdentity>,
    #[serde(default)]
    pub tensor_conflicts: Vec<TensorConflict>,
    #[serde(default)]
    pub gradient_conflicts: Vec<GradientConflict>,
    #[serde(default)]
    pub identity_mismatches: Vec<IdentityMismatch>,
    /// First step at which any parameter gradient was `non_finite`, when steps are present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_non_finite_step: Option<u64>,
    /// Parameter identities whose observed forward `abs_max` crossed a saturation threshold.
    #[serde(default)]
    pub saturating_activations: Vec<ParamIdentity>,
}

impl RuntimeAudit {
    pub fn is_clean(&self) -> bool {
        self.missing_gradients.is_empty()
            && self.non_finite_gradients.is_empty()
            && self.zero_gradients.is_empty()
            && self.tensor_conflicts.is_empty()
            && self.gradient_conflicts.is_empty()
            && self.identity_mismatches.is_empty()
    }

    /// True when any conflict or identity mismatch requires Unknown (not Proven) treatment.
    pub fn has_unknown_evidence(&self) -> bool {
        !self.tensor_conflicts.is_empty()
            || !self.gradient_conflicts.is_empty()
            || !self.identity_mismatches.is_empty()
    }
}

impl RuntimeTrace {
    /// Sort collections deterministically for stable diffs and audits.
    pub fn normalize(&mut self) {
        self.run.cargo_features.sort();
        self.run.cargo_features.dedup();
        self.run.cfg.sort();
        self.run.cfg.dedup();

        self.tensors.sort_by(|a, b| {
            a.event_id
                .cmp(&b.event_id)
                .then_with(|| a.static_id.cmp(&b.static_id))
                .then_with(|| a.source.cmp(&b.source))
        });
        self.operations.sort_by(|a, b| {
            a.event_id
                .cmp(&b.event_id)
                .then_with(|| a.op.cmp(&b.op))
                .then_with(|| a.static_id.cmp(&b.static_id))
        });
        self.gradients.sort_by(|a, b| {
            a.root
                .cmp(&b.root)
                .then_with(|| a.key.cmp(&b.key))
                .then_with(|| a.step.cmp(&b.step))
                .then_with(|| a.event_id.cmp(&b.event_id))
        });
        self.values.sort_by(|a, b| {
            a.event_id
                .cmp(&b.event_id)
                .then_with(|| a.step.cmp(&b.step))
                .then_with(|| a.source.cmp(&b.source))
        });
        self.edge_timings.sort_by(|a, b| {
            a.from_static_id
                .cmp(&b.from_static_id)
                .then_with(|| a.to_static_id.cmp(&b.to_static_id))
                .then_with(|| a.step.cmp(&b.step))
                .then_with(|| a.event_id.cmp(&b.event_id))
        });
    }

    /// Reject duplicate event ids and invalid gradient facts.
    ///
    /// Contradictory tensor/gradient *observations* are not rejected here; they are reported by
    /// [`Self::audit`] as Unknown conflicts so importers never silently overwrite a winner.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA && self.schema != SCHEMA_V2 && self.schema != SCHEMA_V3 {
            bail!(
                "unsupported runtime schema {:?}; expected {:?}, {:?}, or {:?}",
                self.schema,
                SCHEMA,
                SCHEMA_V2,
                SCHEMA_V3
            );
        }

        let mut seen = HashSet::new();
        let mut push_id = |id: &str| -> Result<()> {
            if id.is_empty() {
                bail!("empty event_id is not allowed");
            }
            if !seen.insert(id.to_string()) {
                bail!("duplicate event_id `{id}`");
            }
            Ok(())
        };

        for t in &self.tensors {
            push_id(&t.event_id)?;
        }
        for op in &self.operations {
            push_id(&op.event_id)?;
        }
        for g in &self.gradients {
            push_id(&g.event_id)?;
            g.validate()?;
        }
        for v in &self.values {
            push_id(&v.event_id)?;
        }
        for edge in &self.edge_timings {
            push_id(&edge.event_id)?;
        }
        Ok(())
    }

    /// Normalize then validate; returns a ready-to-query trace.
    pub fn finalize(mut self) -> Result<Self> {
        self.normalize();
        self.validate()?;
        Ok(self)
    }

    /// Compare optional trace identity fields against an importer-supplied expectation.
    ///
    /// Fields omitted on either side are skipped (backward compatible). When both sides supply a
    /// field and they differ, a mismatch is returned — callers must reject or report it as Unknown.
    pub fn check_identity(&self, expected: &ExpectedIdentity) -> Vec<IdentityMismatch> {
        let mut out = Vec::new();
        if let (Some(expected_id), Some(observed)) = (
            expected.analysis_id.as_deref(),
            self.run.analysis_id.as_deref(),
        ) {
            if expected_id != observed {
                out.push(IdentityMismatch {
                    field: IdentityField::AnalysisId,
                    expected: expected_id.to_string(),
                    observed: observed.to_string(),
                });
            }
        }
        if let (Some(expected_id), Some(observed)) =
            (expected.build_id.as_deref(), self.run.build_id.as_deref())
        {
            if expected_id != observed {
                out.push(IdentityMismatch {
                    field: IdentityField::BuildId,
                    expected: expected_id.to_string(),
                    observed: observed.to_string(),
                });
            }
        }
        out
    }

    /// Reject when supplied identity disagrees with `expected`.
    ///
    /// Compatible when either side omits a field.
    pub fn require_identity(&self, expected: &ExpectedIdentity) -> Result<()> {
        let mismatches = self.check_identity(expected);
        if let Some(first) = mismatches.first() {
            bail!(
                "runtime {} mismatch: expected {:?}, observed {:?}",
                first.field,
                first.expected,
                first.observed
            );
        }
        Ok(())
    }

    /// All tensor observations carrying `static_id`.
    pub fn tensors_by_static_id(&self, static_id: &str) -> Vec<&TensorObservation> {
        self.tensors
            .iter()
            .filter(|t| t.static_id.as_deref() == Some(static_id))
            .collect()
    }

    /// Agreed tensor observation for `static_id`, if any.
    ///
    /// Returns `None` when there are no observations or when shape/dtype/device conflict — never
    /// silently picks a contradictory winner.
    pub fn agreed_tensor(&self, static_id: &str) -> Option<&TensorObservation> {
        let obs = self.tensors_by_static_id(static_id);
        if obs.is_empty() {
            return None;
        }
        let first = obs[0];
        for other in &obs[1..] {
            if other.shape != first.shape
                || other.dtype != first.dtype
                || other.device != first.device
                || other.contiguous != first.contiguous
                || other.requires_grad != first.requires_grad
            {
                return None;
            }
        }
        Some(first)
    }

    /// Confidence for refining a static tensor from this trace.
    pub fn tensor_confidence(&self, static_id: &str) -> ObservationConfidence {
        match self.agreed_tensor(static_id) {
            Some(_) => ObservationConfidence::Proven,
            None => ObservationConfidence::Unknown,
        }
    }

    /// Gradient facts for a builder-root + parameter-key identity.
    pub fn gradients_for(&self, root: &str, key: &str) -> Vec<&GradientFact> {
        self.gradients
            .iter()
            .filter(|g| g.root == root && g.key == key)
            .collect()
    }

    /// Agreed gradient fact for the identity, if any.
    ///
    /// Returns `None` when facts are missing or contradictory (state/norm disagree). Does not
    /// silently overwrite by returning the first observation.
    pub fn gradient(&self, root: &str, key: &str) -> Option<&GradientFact> {
        let facts = self.gradients_for(root, key);
        agreed_gradient(&facts)
    }

    /// Confidence for refining a parameter gradient from this trace.
    pub fn gradient_confidence(&self, root: &str, key: &str) -> ObservationConfidence {
        match self.gradient(root, key) {
            Some(_) => ObservationConfidence::Proven,
            None => ObservationConfidence::Unknown,
        }
    }

    /// Compact audit: gradient quality, tensor/gradient conflicts, optional identity mismatches.
    pub fn audit(&self) -> RuntimeAudit {
        self.audit_with_identity(None)
    }

    /// Like [`Self::audit`], also reporting identity mismatches when `expected` is supplied.
    pub fn audit_with_identity(&self, expected: Option<&ExpectedIdentity>) -> RuntimeAudit {
        let gradient_conflicts = self.collect_gradient_conflicts();
        let conflicted: HashSet<ParamIdentity> = gradient_conflicts
            .iter()
            .map(|c| c.identity.clone())
            .collect();

        let mut missing = BTreeSet::new();
        let mut non_finite = BTreeSet::new();
        let mut zero = BTreeSet::new();

        // Only classify agreed (non-conflicted) gradient identities.
        let mut by_param: BTreeMap<ParamIdentity, Vec<&GradientFact>> = BTreeMap::new();
        for g in &self.gradients {
            by_param.entry(g.identity()).or_default().push(g);
        }
        for (id, facts) in by_param {
            if conflicted.contains(&id) {
                continue;
            }
            let Some(g) = latest_agreed_gradient(&facts).or_else(|| agreed_gradient(&facts)) else {
                continue;
            };
            match g.state {
                GradientState::Missing => {
                    missing.insert(id);
                }
                GradientState::NonFinite => {
                    non_finite.insert(id);
                }
                GradientState::Zero => {
                    zero.insert(id);
                }
                GradientState::Present => {}
            }
        }

        let mut by_static: BTreeMap<&str, Vec<&TensorObservation>> = BTreeMap::new();
        for t in &self.tensors {
            if let Some(sid) = t.static_id.as_deref() {
                by_static.entry(sid).or_default().push(t);
            }
        }

        let mut conflicts = Vec::new();
        for (static_id, obs) in by_static {
            if obs.len() < 2 {
                continue;
            }
            push_conflict(
                &mut conflicts,
                static_id,
                TensorConflictKind::Dtype,
                |t| t.dtype.clone(),
                &obs,
            );
            push_conflict(
                &mut conflicts,
                static_id,
                TensorConflictKind::Device,
                |t| t.device.clone(),
                &obs,
            );
            push_conflict(
                &mut conflicts,
                static_id,
                TensorConflictKind::Shape,
                |t| format_shape(&t.shape),
                &obs,
            );
            push_conflict(
                &mut conflicts,
                static_id,
                TensorConflictKind::Layout,
                |t| t.contiguous.to_string(),
                &obs,
            );
            push_conflict(
                &mut conflicts,
                static_id,
                TensorConflictKind::RequiresGrad,
                |t| t.requires_grad.to_string(),
                &obs,
            );
        }
        conflicts.sort_by(|a, b| {
            a.static_id
                .cmp(&b.static_id)
                .then_with(|| a.kind.cmp(&b.kind))
        });

        let identity_mismatches = expected
            .map(|expected| self.check_identity(expected))
            .unwrap_or_default();

        let first_non_finite_step = self
            .gradients
            .iter()
            .filter(|g| matches!(g.state, GradientState::NonFinite))
            .filter_map(|g| g.step)
            .min();

        let mut saturating = BTreeSet::new();
        for value in &self.values {
            if value.saturated_count == 0 {
                continue;
            }
            if let Some(source) = &value.source {
                saturating.insert(ParamIdentity {
                    root: "value".into(),
                    key: source.clone(),
                });
            }
        }

        RuntimeAudit {
            missing_gradients: missing.into_iter().collect(),
            non_finite_gradients: non_finite.into_iter().collect(),
            zero_gradients: zero.into_iter().collect(),
            tensor_conflicts: conflicts,
            gradient_conflicts,
            identity_mismatches,
            first_non_finite_step,
            saturating_activations: saturating.into_iter().collect(),
        }
    }

    fn collect_gradient_conflicts(&self) -> Vec<GradientConflict> {
        let mut by_param: BTreeMap<ParamIdentity, Vec<&GradientFact>> = BTreeMap::new();
        for g in &self.gradients {
            by_param.entry(g.identity()).or_default().push(g);
        }
        let mut out = Vec::new();
        for (identity, facts) in by_param {
            if facts.len() < 2 {
                continue;
            }
            // Distinct steps with one observation each are a time series, not a conflict.
            if facts.iter().all(|g| g.step.is_some()) {
                let steps: BTreeSet<_> = facts.iter().filter_map(|g| g.step).collect();
                if steps.len() == facts.len() {
                    continue;
                }
            }
            if latest_agreed_gradient(&facts).is_some() || agreed_gradient(&facts).is_some() {
                continue;
            }
            let mut states: BTreeSet<String> = BTreeSet::new();
            let mut event_ids: BTreeSet<String> = BTreeSet::new();
            for g in &facts {
                states.insert(g.state.to_string());
                event_ids.insert(g.event_id.clone());
            }
            // Distinct norms with same state also conflict (e.g. two present norms).
            let mut norms: BTreeSet<String> = BTreeSet::new();
            for g in &facts {
                norms.insert(match g.norm {
                    Some(n) => format!("{n}"),
                    None => "none".to_string(),
                });
            }
            if states.len() <= 1 && norms.len() <= 1 {
                continue;
            }
            out.push(GradientConflict {
                identity,
                states: states.into_iter().collect(),
                event_ids: event_ids.into_iter().collect(),
                confidence: ObservationConfidence::Unknown,
            });
        }
        out.sort_by(|a, b| a.identity.cmp(&b.identity));
        out
    }
}

/// Returns the sole agreed fact when all entries share state and norm; otherwise `None`.
fn agreed_gradient<'a>(facts: &[&'a GradientFact]) -> Option<&'a GradientFact> {
    let first = facts.first().copied()?;
    for other in facts.iter().skip(1) {
        if other.state != first.state {
            return None;
        }
        match (first.norm, other.norm) {
            (None, None) => {}
            (Some(a), Some(b)) if float_eq(a, b) => {}
            (None, Some(_)) | (Some(_), None) => return None,
            (Some(_), Some(_)) => return None,
        }
    }
    Some(first)
}

/// When every fact carries a step, return the latest-step observation (time-series rollup).
fn latest_agreed_gradient<'a>(facts: &[&'a GradientFact]) -> Option<&'a GradientFact> {
    if facts.is_empty() || !facts.iter().all(|g| g.step.is_some()) {
        return None;
    }
    facts
        .iter()
        .copied()
        .max_by_key(|g| g.step.unwrap_or(0))
}

fn float_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    a == b
}

fn format_shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn push_conflict(
    out: &mut Vec<TensorConflict>,
    static_id: &str,
    kind: TensorConflictKind,
    project: impl Fn(&TensorObservation) -> String,
    obs: &[&TensorObservation],
) {
    let mut values: BTreeSet<String> = BTreeSet::new();
    for t in obs {
        values.insert(project(t));
    }
    if values.len() > 1 {
        out.push(TensorConflict {
            static_id: static_id.to_string(),
            kind,
            values: values.into_iter().collect(),
            confidence: ObservationConfidence::Unknown,
        });
    }
}

/// Parse a single JSON document into a validated, normalized trace.
pub fn parse_json(input: &str) -> Result<RuntimeTrace> {
    let trace: RuntimeTrace =
        serde_json::from_str(input).context("failed to parse runtime JSON document")?;
    trace.finalize()
}

/// Parse JSONL event records into a validated, normalized trace.
pub fn parse_jsonl(input: &str) -> Result<RuntimeTrace> {
    let mut schema: Option<String> = None;
    let mut run: Option<RunMetadata> = None;
    let mut tensors = Vec::new();
    let mut operations = Vec::new();
    let mut gradients = Vec::new();
    let mut values = Vec::new();
    let mut edge_timings = Vec::new();

    for (line_no, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: RuntimeEvent = serde_json::from_str(line)
            .with_context(|| format!("failed to parse runtime JSONL line {}", line_no + 1))?;
        match event {
            RuntimeEvent::Meta {
                schema: s,
                run: meta,
            } => {
                if schema.is_some() || run.is_some() {
                    bail!(
                        "duplicate meta event on JSONL line {}; only one meta record is allowed",
                        line_no + 1
                    );
                }
                schema = Some(s);
                run = Some(meta);
            }
            RuntimeEvent::Tensor(t) => tensors.push(t),
            RuntimeEvent::Operation(op) => operations.push(op),
            RuntimeEvent::Gradient(g) => gradients.push(g),
            RuntimeEvent::Value(v) => values.push(v),
            RuntimeEvent::EdgeTiming(edge) => edge_timings.push(edge),
        }
    }

    let schema = schema.unwrap_or_else(|| SCHEMA.to_string());
    let run = run.context("JSONL stream is missing a meta event with run metadata")?;

    RuntimeTrace {
        schema,
        run,
        tensors,
        operations,
        gradients,
        values,
        edge_timings,
    }
    .finalize()
}

/// Auto-detect a single JSON object versus JSONL event records.
pub fn parse(input: &str) -> Result<RuntimeTrace> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("empty runtime evidence input");
    }
    // A document starts with `{` and is a single JSON value; JSONL is line-oriented events.
    if trimmed.starts_with('{') && !trimmed.contains('\n') {
        return parse_json(trimmed);
    }
    if trimmed.starts_with('{') {
        // Multi-line JSON document vs JSONL: try document first when the whole buffer is one value.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.is_object() && value.get("schema").is_some() && value.get("run").is_some() {
                let trace: RuntimeTrace = serde_json::from_value(value)
                    .context("failed to parse runtime JSON document")?;
                return trace.finalize();
            }
        }
    }
    parse_jsonl(input)
}
