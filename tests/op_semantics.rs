use candle_graph::op_semantics::{
    self, AbstractDtype, DomainRequirement, DtypeRule, GradFlow, NumericDomain,
};

#[test]
fn no_backward_rules_are_version_gated() {
    assert_eq!(
        op_semantics::lookup_for("candle_nn::ops::rms_norm", Some("0.10.2")).grad,
        GradFlow::Unknown
    );
    assert_eq!(
        op_semantics::lookup_for("candle_nn::ops::rms_norm", Some("0.11.0")).grad,
        GradFlow::Severs
    );
    assert_eq!(
        op_semantics::lookup_for("candle_nn::ops::rms_norm", None).grad,
        GradFlow::Unknown
    );
}

#[test]
fn candle_011_cpu_attention_is_known_to_sever_autograd() {
    for op in [
        "flash_attn",
        "flash_attn_varlen_cpu",
        "flash_attn_varlen_unfused",
        "run_flash_attn_cpu",
    ] {
        assert_eq!(
            op_semantics::lookup_for(op, Some("0.11.0")).grad,
            GradFlow::Severs
        );
        assert_eq!(
            op_semantics::lookup_for(op, Some("0.10.2")).grad,
            GradFlow::Unknown
        );
    }
}

#[test]
fn rms_norm_forward_variants_are_not_conflated() {
    assert_eq!(
        op_semantics::lookup_method("candle_nn::RmsNorm", "forward_diff", Some("0.10.2")).grad,
        GradFlow::Unknown
    );
    assert_eq!(
        op_semantics::lookup_method("candle_nn::RmsNorm", "forward", Some("0.10.2")).grad,
        GradFlow::Unknown
    );
    assert_eq!(
        op_semantics::lookup_method("candle_nn::RmsNorm", "forward_diff", Some("0.11.0")).grad,
        GradFlow::Propagates
    );
}

#[test]
fn non_differentiable_and_view_operations_have_precise_gradient_rules() {
    for op in ["floor", "round", "sign"] {
        assert_eq!(
            op_semantics::lookup_for(op, Some("0.11.0")).grad,
            GradFlow::Severs
        );
    }
    for op in ["reshape", "transpose", "permute", "narrow"] {
        assert_eq!(
            op_semantics::lookup_for(op, Some("0.11.0")).grad,
            GradFlow::Propagates
        );
    }
}

#[test]
fn resolved_operation_labels_preserve_receiver_semantics() {
    let effect = op_semantics::lookup_resolved("RmsNorm::forward", Some("0.11.0"));
    assert_eq!(effect.grad, GradFlow::LayoutDependent);

    let effect = op_semantics::lookup_resolved("RmsNorm::forward_diff", Some("0.11.0"));
    assert_eq!(effect.grad, GradFlow::Propagates);
}

#[test]
fn candle_011_dtype_catalog_matches_core_dtypes_and_index_outputs() {
    assert_eq!(AbstractDtype::parse("Bool"), AbstractDtype::Unknown);
    assert_eq!(
        op_semantics::lookup_for("Tensor::eq", Some("0.11.0")).dtype,
        DtypeRule::Fixed(AbstractDtype::U8)
    );
    assert_eq!(
        op_semantics::lookup_for("Tensor::argmax", Some("0.11.0")).dtype,
        DtypeRule::Fixed(AbstractDtype::U32)
    );
}

#[test]
fn candle_packages_must_match_the_single_audited_release() {
    assert_eq!(
        op_semantics::matched_candle_version(Some("0.11.0"), Some("0.11.0")),
        Some("0.11.0")
    );
    assert_eq!(
        op_semantics::matched_candle_version(Some("0.10.2"), Some("0.10.2")),
        None
    );
    assert_eq!(
        op_semantics::matched_candle_version(Some("0.11.0"), Some("0.10.2")),
        None
    );
}

#[test]
fn bce_with_logits_exposes_expandable_body_only_for_audited_011() {
    assert!(
        op_semantics::library_body("binary_cross_entropy_with_logit", Some("0.11.0")).is_some()
    );
    assert!(
        op_semantics::library_body("binary_cross_entropy_with_logit", Some("0.10.2")).is_none()
    );
    assert!(op_semantics::library_body("binary_cross_entropy_with_logit", None).is_none());
    // Catalog must not carry a precomputed unsafe verdict.
    let effect = op_semantics::lookup_for("binary_cross_entropy_with_logit", Some("0.11.0"));
    assert!(effect.is_loss);
}

#[test]
fn sigmoid_and_log_carry_domain_transfer_rules() {
    let sigmoid = op_semantics::lookup_for("sigmoid", Some("0.11.0"));
    assert_eq!(sigmoid.domain, NumericDomain::SaturatingUnit);
    let log = op_semantics::lookup_for("log", Some("0.11.0"));
    assert_eq!(log.requires, DomainRequirement::StrictlyPositive);
    assert!(op_semantics::domain_violation(log.requires, sigmoid.domain).is_some());
    assert!(
        op_semantics::domain_violation(log.requires, NumericDomain::StrictlyPositive).is_none()
    );
}

#[test]
fn affine_epsilon_guard_discharges_but_reflection_does_not() {
    assert_eq!(
        op_semantics::affine_domain(NumericDomain::SaturatingUnit, Some(1.0), Some(1e-7)),
        NumericDomain::StrictlyPositive
    );
    assert_eq!(
        op_semantics::affine_domain(NumericDomain::SaturatingUnit, Some(-1.0), Some(1.0)),
        NumericDomain::SaturatingUnit
    );
}
