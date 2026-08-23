use candle_graph::nsight::NsightEvidence;
use candle_graph::trace::{
    GradientEvent, GradientState, RunOutcome, SpanRecord, TerminalEvent, TimingMode, TraceRunMeta,
    SCHEMA as TRACE_SCHEMA,
};
use candle_graph::{
    CaptureContract, CoverageLevel, EvidencePacket, ExpectedGradient, GradientContract,
    GradientFamilyContract, MeasurementScope, SpanKind, TraceDocument,
};

fn gradient(root: &str, key: &str, state: GradientState, norm: Option<f64>) -> GradientEvent {
    GradientEvent {
        event_id: format!("gradient-{key}"),
        root: root.into(),
        key: key.into(),
        state,
        norm,
    }
}

#[test]
fn public_schema_constants_identify_the_breaking_contract_revision() {
    assert_eq!(candle_graph::TRACE_SCHEMA, "candle-graph/trace/9");
    assert_eq!(candle_graph::EVIDENCE_SCHEMA, "candle-graph/evidence/3");
    assert_eq!(candle_graph::COMPARISON_SCHEMA, "candle-graph/comparison/4");
    assert_eq!(candle_graph::BUNDLE_SCHEMA, "candle-graph/bundle/1");
    assert_eq!(
        candle_graph::GRADIENT_MANIFEST_SCHEMA,
        "candle-graph/gradient-manifest/1"
    );
}

fn contract() -> GradientContract {
    GradientContract::new(
        vec![
            ExpectedGradient::new("parameters", "trunk.weight", "trunk"),
            ExpectedGradient::new("parameters", "frozen.weight", "frozen"),
            ExpectedGradient::new("parameters", "optional.weight", "optional"),
        ],
        vec![
            GradientFamilyContract::active("trunk", 1),
            GradientFamilyContract::inactive("frozen"),
            GradientFamilyContract::data_conditional("optional", 1),
        ],
    )
    .unwrap()
}

#[test]
fn ordered_manifest_digest_has_a_stable_golden_value() {
    assert_eq!(
        contract().manifest_sha256,
        "sha256:d7f0b4aa8a4072b9824018c8d3eff5757286f1348366c15ef8780be38945fcc6"
    );
}

fn document(gradients: Vec<GradientEvent>) -> TraceDocument {
    document_with_contract(gradients, contract())
}

fn document_with_contract(
    gradients: Vec<GradientEvent>,
    gradient_contract: GradientContract,
) -> TraceDocument {
    TraceDocument {
        schema: TRACE_SCHEMA.into(),
        run: TraceRunMeta {
            run_id: "gradient-contract-run".into(),
            correlation_id: "gradient/contract/run".into(),
            entrypoint: "train::step".into(),
            phase: candle_graph::ExecutionPhase::Train,
            timestamp: "2026-08-21T00:00:00Z".into(),
            capture_step: 1,
            warmup_steps: 0,
            device: "cpu".into(),
            measured_region_device_synchronized: false,
            timing_mode: TimingMode::Host,
            capture_contract: CaptureContract {
                measurement_scope: MeasurementScope::ProductionEquivalent,
                gradients: CoverageLevel::Complete,
                gradient_contract: Some(gradient_contract),
                ..CaptureContract::default()
            },
            comparison_identity: None,
            tags: Default::default(),
            candle_version: None,
        },
        spans: vec![SpanRecord {
            id: "root".into(),
            parent_id: None,
            name: "train::step".into(),
            kind: SpanKind::Function,
            measured: true,
            start_ns: 0,
            closed: true,
            duration_ns: 10,
            step: None,
        }],
        ops: vec![],
        tensors: vec![],
        memory: vec![],
        device_memory: vec![],
        device_intervals: vec![],
        gradients,
        edges: vec![],
        terminal: TerminalEvent {
            outcome: RunOutcome::Complete,
            timestamp_ns: 10,
            reason: None,
        },
    }
}

fn conditional_matrix_codes(states: [(GradientState, Option<f64>); 3]) -> Vec<String> {
    let conditional_contract = GradientContract::new(
        vec![
            ExpectedGradient::new("parameters", "trunk.weight", "trunk"),
            ExpectedGradient::new("parameters", "optional.a", "optional"),
            ExpectedGradient::new("parameters", "optional.b", "optional"),
            ExpectedGradient::new("parameters", "optional.c", "optional"),
        ],
        vec![
            GradientFamilyContract::active("trunk", 1),
            GradientFamilyContract::data_conditional("optional", 2),
        ],
    )
    .unwrap();
    let mut gradients = vec![gradient(
        "parameters",
        "trunk.weight",
        GradientState::Present,
        Some(1.0),
    )];
    gradients.extend(
        ["optional.a", "optional.b", "optional.c"]
            .into_iter()
            .zip(states)
            .map(|(key, (state, norm))| gradient("parameters", key, state, norm)),
    );
    EvidencePacket::from_document(
        document_with_contract(gradients, conditional_contract),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap()
    .health
    .issues
    .into_iter()
    .map(|issue| issue.code)
    .collect()
}

fn valid_gradients() -> Vec<GradientEvent> {
    vec![
        gradient(
            "parameters",
            "trunk.weight",
            GradientState::Present,
            Some(2.0),
        ),
        gradient("parameters", "frozen.weight", GradientState::Missing, None),
        gradient(
            "parameters",
            "optional.weight",
            GradientState::Missing,
            None,
        ),
    ]
}

fn issue_codes(gradients: Vec<GradientEvent>) -> Vec<String> {
    EvidencePacket::from_document(
        document(gradients),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap()
    .health
    .issues
    .into_iter()
    .map(|issue| issue.code)
    .collect()
}

#[test]
fn exact_manifest_and_family_expectations_make_gradient_coverage_complete() {
    let packet = EvidencePacket::from_document(
        document(valid_gradients()),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();

    assert!(packet.health.structurally_valid);
    assert!(packet.capabilities.gradient_coverage.is_complete());
    assert_eq!(
        packet.capabilities.gradient_coverage.source,
        "exact gradient contract"
    );
    assert!(packet
        .capabilities
        .gradient_coverage
        .reason
        .contains("3 manifest entries and 3 family expectations validated"));
    assert!(packet
        .facts
        .iter()
        .any(|fact| fact.code == "gradient_manifest_sha256"));
}

#[test]
fn complete_gradient_coverage_rejects_a_missing_manifest_key() {
    let packet = EvidencePacket::from_document(
        document(vec![
            gradient(
                "parameters",
                "trunk.weight",
                GradientState::Present,
                Some(2.0),
            ),
            gradient("parameters", "frozen.weight", GradientState::Missing, None),
        ]),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();

    assert!(!packet.health.structurally_valid);
    assert_eq!(
        packet.capabilities.gradient_coverage.level,
        candle_graph::capability::CapabilityLevel::Invalid
    );
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_manifest_missing_key"));
}

#[test]
fn gradient_state_requires_a_consistent_norm() {
    let packet = EvidencePacket::from_document(
        document(vec![
            gradient("parameters", "trunk.weight", GradientState::Present, None),
            gradient("parameters", "frozen.weight", GradientState::Missing, None),
            gradient(
                "parameters",
                "optional.weight",
                GradientState::Missing,
                None,
            ),
        ]),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();

    assert!(!packet.health.structurally_valid);
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_state_norm_inconsistent"));
}

#[test]
fn active_family_rejects_an_all_zero_capture() {
    let packet = EvidencePacket::from_document(
        document(vec![
            gradient("parameters", "trunk.weight", GradientState::Zero, Some(0.0)),
            gradient("parameters", "frozen.weight", GradientState::Missing, None),
            gradient(
                "parameters",
                "optional.weight",
                GradientState::Missing,
                None,
            ),
        ]),
        NsightEvidence::unavailable("not captured"),
    )
    .unwrap();

    assert!(!packet.health.structurally_valid);
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_active_family_below_minimum"));
}

#[test]
fn inactive_and_data_conditional_families_enforce_their_contracts() {
    let inactive_codes = issue_codes(vec![
        gradient(
            "parameters",
            "trunk.weight",
            GradientState::Present,
            Some(2.0),
        ),
        gradient(
            "parameters",
            "frozen.weight",
            GradientState::Present,
            Some(1.0),
        ),
        gradient(
            "parameters",
            "optional.weight",
            GradientState::Missing,
            None,
        ),
    ]);
    assert!(inactive_codes
        .iter()
        .any(|code| code == "gradient_inactive_family_leakage"));

    let conditional_codes = issue_codes(vec![
        gradient(
            "parameters",
            "trunk.weight",
            GradientState::Present,
            Some(2.0),
        ),
        gradient("parameters", "frozen.weight", GradientState::Missing, None),
        gradient(
            "parameters",
            "optional.weight",
            GradientState::Zero,
            Some(0.0),
        ),
    ]);
    assert!(!conditional_codes
        .iter()
        .any(|code| code == "gradient_conditional_family_below_minimum"));
}

#[test]
fn multi_member_data_conditional_matrix_is_fail_closed() {
    let missing = (GradientState::Missing, None);
    let zero = (GradientState::Zero, Some(0.0));
    let present = (GradientState::Present, Some(1.0));
    let non_finite = (GradientState::NonFinite, None);

    for states in [[missing; 3], [zero; 3], [present, present, missing]] {
        assert!(!conditional_matrix_codes(states)
            .iter()
            .any(|code| code == "gradient_conditional_family_below_minimum"));
    }
    assert!(conditional_matrix_codes([present, zero, missing])
        .iter()
        .any(|code| code == "gradient_conditional_family_below_minimum"));
    assert!(conditional_matrix_codes([non_finite, missing, missing])
        .iter()
        .any(|code| code == "gradient_family_non_finite"));
}

#[test]
fn exact_manifest_rejects_duplicate_undeclared_and_ambiguous_events() {
    let mut duplicate = valid_gradients();
    duplicate.push(gradient(
        "parameters",
        "trunk.weight",
        GradientState::Present,
        Some(2.0),
    ));
    let duplicate_codes = issue_codes(duplicate);
    assert!(duplicate_codes
        .iter()
        .any(|code| code == "gradient_manifest_duplicate_key"));
    assert!(duplicate_codes
        .iter()
        .any(|code| code == "duplicate_gradient_event_id"));

    let mut undeclared = valid_gradients();
    undeclared.push(gradient(
        "parameters",
        "extra.weight",
        GradientState::Present,
        Some(1.0),
    ));
    assert!(issue_codes(undeclared)
        .iter()
        .any(|code| code == "gradient_manifest_undeclared_key"));

    let mut empty_event_id = valid_gradients();
    empty_event_id[0].event_id.clear();
    assert!(issue_codes(empty_event_id)
        .iter()
        .any(|code| code == "empty_gradient_event_id"));
}

#[test]
fn every_gradient_state_has_an_unambiguous_norm_encoding() {
    for (state, norm) in [
        (GradientState::Present, Some(0.0)),
        (GradientState::Present, Some(f64::INFINITY)),
        (GradientState::Zero, None),
        (GradientState::Zero, Some(-0.0)),
        (GradientState::Missing, Some(0.0)),
        (GradientState::NonFinite, Some(f64::NAN)),
    ] {
        let mut gradients = valid_gradients();
        gradients[0].state = state;
        gradients[0].norm = norm;
        assert!(
            issue_codes(gradients)
                .iter()
                .any(|code| code == "gradient_state_norm_inconsistent"),
            "state {state:?} with norm {norm:?} was accepted"
        );
    }
}

#[test]
fn contract_digest_and_family_classification_are_validated_after_deserialization() {
    let mut tampered = contract();
    tampered.expected[0].key = "different.weight".into();
    assert!(tampered.validate().is_err());

    let mut unclassified = contract();
    unclassified
        .families
        .retain(|family| family.family != "trunk");
    assert!(unclassified
        .validate()
        .unwrap_err()
        .to_string()
        .contains("has no family contract"));
}

#[test]
fn capture_contract_rejects_gradient_contract_coverage_mismatches() {
    let missing = CaptureContract {
        gradients: CoverageLevel::Complete,
        ..CaptureContract::default()
    };
    assert!(missing.validate().is_err());

    let undeclared = CaptureContract {
        gradients: CoverageLevel::Partial,
        gradient_contract: Some(contract()),
        ..CaptureContract::default()
    };
    assert!(undeclared.validate().is_err());
}

#[test]
fn capture_contract_rejects_empty_and_duplicate_required_semantic_labels() {
    let empty = CaptureContract {
        required_semantic_labels: vec!["  ".into()],
        ..CaptureContract::default()
    };
    assert!(empty.validate().is_err());

    let duplicate = CaptureContract {
        required_semantic_labels: vec!["train/forward".into(), "train/forward".into()],
        ..CaptureContract::default()
    };
    assert!(duplicate.validate().is_err());
}

#[test]
fn capture_contract_requires_an_exact_gpu_and_cpu_semantic_partition() {
    let valid = CaptureContract {
        required_semantic_labels: vec!["train/forward".into(), "train/prepare".into()],
        gpu_expected_semantic_labels: vec!["train/forward".into()],
        cpu_only_semantic_labels: vec!["train/prepare".into()],
        ..CaptureContract::default()
    };
    valid.validate().unwrap();
    assert_eq!(
        valid.resolved_gpu_expected_semantic_labels(),
        vec!["train/forward".to_string()]
    );

    let unclassified = CaptureContract {
        cpu_only_semantic_labels: Vec::new(),
        gpu_expected_semantic_labels: vec!["train/forward".into()],
        ..valid.clone()
    };
    assert!(unclassified.validate().is_err());

    let overlapping = CaptureContract {
        cpu_only_semantic_labels: vec!["train/forward".into(), "train/prepare".into()],
        ..valid
    };
    assert!(overlapping.validate().is_err());
}

#[test]
fn complete_coverage_without_an_exact_contract_is_invalid() {
    let mut document = document(valid_gradients());
    document.run.capture_contract.gradient_contract = None;
    let packet =
        EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
            .unwrap();

    assert!(!packet.health.structurally_valid);
    assert_eq!(
        packet.capabilities.gradient_coverage.level,
        candle_graph::capability::CapabilityLevel::Invalid
    );
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_contract_missing"));
}

#[test]
fn partial_gradient_events_still_obey_event_level_invariants() {
    let mut document = document(vec![gradient(
        "parameters",
        "trunk.weight",
        GradientState::Present,
        None,
    )]);
    document.run.capture_contract.gradients = CoverageLevel::Partial;
    document.run.capture_contract.gradient_contract = None;
    let packet =
        EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
            .unwrap();

    assert!(!packet.health.structurally_valid);
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_state_norm_inconsistent"));
}

#[test]
fn interrupted_exact_capture_reports_missing_manifest_events_as_partial() {
    let mut document = document(Vec::new());
    document.terminal.outcome = RunOutcome::Failed;
    document.terminal.reason = Some("interrupted before gradient inspection".into());
    let packet =
        EvidencePacket::from_document(document, NsightEvidence::unavailable("not captured"))
            .unwrap();

    assert!(packet.health.structurally_valid);
    assert!(!packet.health.capture_complete);
    assert_eq!(
        packet.capabilities.gradient_coverage.level,
        candle_graph::capability::CapabilityLevel::Partial
    );
    assert!(packet
        .health
        .issues
        .iter()
        .any(|issue| issue.code == "gradient_manifest_missing_key"));
}
