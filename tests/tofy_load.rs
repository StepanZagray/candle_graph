use std::path::Path;

use candle_graph::cargo_context::CargoOptions;
use candle_graph::discover::{analyze, ScanOptions};

const TOFY: &str = "/home/stepan/Coding/Personal/Tofy";

#[test]
fn candle_graph_feature_names_are_not_forwarded_to_target_crate() {
    let tofy = Path::new(TOFY);
    if !tofy.join("Cargo.toml").is_file() {
        return;
    }
    for features in [
        vec!["static".into()],
        vec!["visualizer".into()],
        vec!["all".into()],
        vec!["static".into(), "visualizer".into(), "all".into()],
    ] {
        let model = analyze(
            tofy,
            &ScanOptions {
                cargo: CargoOptions {
                    features,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("candle-graph feature flags must be ignored: {err:#}"));
        assert!(
            !model.components.is_empty(),
            "expected components after stripping candle-graph features"
        );
    }
}

#[test]
fn target_crate_features_still_forwarded() {
    let mut options = CargoOptions {
        features: vec!["static".into(), "visualizer".into(), "cuda".into()],
        ..Default::default()
    };
    let stripped = options.strip_candle_graph_features();
    assert_eq!(stripped, vec!["static", "visualizer"]);
    assert_eq!(options.features, vec!["cuda"]);
}
