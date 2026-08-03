//! Integration tests for bounded Cargo/config discovery.
//!
//! The module is not yet wired into `lib.rs`, so it is path-included here.

#[path = "../src/cargo_context.rs"]
mod cargo_context;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use cargo_context::{discover, CargoOptions};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "candle-graph-cargo-ctx-{}-{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src").join("nested")).unwrap();
        fs::create_dir_all(root.join("candle-stub").join("src")).unwrap();

        fs::write(
            root.join("candle-stub").join("Cargo.toml"),
            r#"
[package]
name = "candle-core"
version = "9.9.9"
edition = "2021"
"#,
        )
        .unwrap();
        fs::write(
            root.join("candle-stub").join("src").join("lib.rs"),
            "pub fn stub() {}\n",
        )
        .unwrap();

        fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "fixture-crate"
version = "0.3.1"
edition = "2021"

[features]
default = ["alpha"]
alpha = []
beta = []

[dependencies]
candle-alias = { package = "candle-core", path = "candle-stub" }
"#,
        )
        .unwrap();
        fs::write(root.join("src").join("lib.rs"), "pub mod nested;\n").unwrap();
        fs::write(
            root.join("src").join("nested").join("mod.rs"),
            "pub fn leaf() {}\n",
        )
        .unwrap();

        Self { root }
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn discovers_manifest_from_crate_root_and_nested_source_dir() {
    let fixture = Fixture::new();
    let opts = CargoOptions::default();

    let from_root = discover(fixture.root(), &opts).expect("discover from crate root");
    let from_nested = discover(fixture.root().join("src").join("nested"), &opts)
        .expect("discover from nested dir");

    assert_eq!(from_root.package_name, "fixture-crate");
    assert_eq!(from_root.package_version, "0.3.1");
    assert_eq!(from_root.manifest_path, from_nested.manifest_path);
    assert_eq!(from_root.workspace_root, from_nested.workspace_root);
    assert_eq!(from_root.package_id, from_nested.package_id);

    let manifest = fixture.root().join("Cargo.toml");
    let want = manifest.canonicalize().unwrap();
    assert_eq!(
        from_root.manifest_path.canonicalize().unwrap(),
        want,
        "manifest_path should resolve to fixture Cargo.toml"
    );
    assert_eq!(
        from_root.workspace_root.canonicalize().unwrap(),
        fixture.root().canonicalize().unwrap()
    );
    assert!(from_root.target_directory.is_absolute());
    assert!(
        from_root.targets.iter().any(|t| t.name == "fixture_crate"
            || t.name == "fixture-crate"
            || t.kind.iter().any(|k| k == "lib")),
        "expected a lib target, got {:?}",
        from_root.targets
    );

    assert_eq!(
        from_root
            .candle_versions
            .get("candle-core")
            .map(String::as_str),
        Some("9.9.9")
    );
    assert_eq!(
        from_root
            .dependency_aliases
            .get("candle_alias")
            .map(String::as_str),
        Some("candle-core")
    );
    assert_eq!(
        from_root.selected_source_roots(None).unwrap(),
        vec![fixture.root().join("src/lib.rs")]
    );
    assert!(from_root
        .selected_source_roots(Some("missing-target"))
        .unwrap_err()
        .to_string()
        .contains("available targets"));
}

#[test]
fn resolves_default_and_explicit_features() {
    let fixture = Fixture::new();

    let default_ctx = discover(fixture.root(), &CargoOptions::default()).unwrap();
    assert!(
        default_ctx.active_features.iter().any(|f| f == "default"),
        "default features should include `default`: {:?}",
        default_ctx.active_features
    );
    assert!(
        default_ctx.active_features.iter().any(|f| f == "alpha"),
        "default features should activate alpha: {:?}",
        default_ctx.active_features
    );
    assert!(
        !default_ctx.active_features.iter().any(|f| f == "beta"),
        "default features should not activate beta: {:?}",
        default_ctx.active_features
    );

    let explicit = CargoOptions {
        features: vec!["beta".into()],
        all_features: false,
        no_default_features: true,
        target: None,
        package_target: None,
    };
    let explicit_ctx = discover(fixture.root(), &explicit).unwrap();
    assert_eq!(
        explicit_ctx.active_features,
        vec!["beta".to_string()],
        "explicit beta with no-default should activate only beta"
    );

    let all = CargoOptions {
        features: Vec::new(),
        all_features: true,
        no_default_features: false,
        target: None,
        package_target: None,
    };
    let all_ctx = discover(fixture.root(), &all).unwrap();
    for want in ["alpha", "beta", "default"] {
        assert!(
            all_ctx.active_features.iter().any(|f| f == want),
            "all-features missing `{want}`: {:?}",
            all_ctx.active_features
        );
    }

    assert!(
        is_sorted(&default_ctx.active_features),
        "active_features must be sorted"
    );
    assert!(is_sorted(&all_ctx.active_features));
}

#[test]
fn cfgs_are_sorted_and_include_feature_cfgs() {
    let fixture = Fixture::new();
    let opts = CargoOptions {
        features: vec!["beta".into()],
        all_features: false,
        no_default_features: true,
        target: None,
        package_target: None,
    };
    let ctx = discover(fixture.root(), &opts).unwrap();

    assert!(is_sorted(&ctx.cfgs), "cfgs must be sorted: {:?}", ctx.cfgs);
    assert!(
        ctx.cfgs.iter().any(|c| c == "feature=\"beta\""),
        "expected feature cfg for beta: {:?}",
        ctx.cfgs
    );
    assert!(
        ctx.cfgs
            .iter()
            .any(|c| c.starts_with("target_os=") || c == "unix" || c == "windows"),
        "expected rustc host cfg values: {:?}",
        ctx.cfgs
    );

    let mut deduped = ctx.cfgs.clone();
    deduped.dedup();
    assert_eq!(ctx.cfgs, deduped, "cfgs must be unique");
}

#[test]
fn missing_cargo_toml_returns_clear_error() {
    let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "candle-graph-cargo-ctx-missing-{}-{unique}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let err = discover(&dir, &CargoOptions::default()).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Cargo.toml not found") || msg.contains("failed to locate Cargo.toml"),
        "unexpected error: {msg}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn evaluates_nested_cfg_predicates_against_selected_context() {
    let active = vec![
        "feature=\"cuda\"".to_string(),
        "target_arch=\"x86_64\"".to_string(),
        "unix".to_string(),
    ];
    assert_eq!(
        cargo_context::cfg_predicates_active(
            &["all(feature = \"cuda\", any(unix, windows))".to_string()],
            &active,
        ),
        Some(true)
    );
    assert_eq!(
        cargo_context::cfg_predicates_active(&["not(feature = \"cuda\")".to_string()], &active,),
        Some(false)
    );
    assert_eq!(
        cargo_context::cfg_predicates_active(&["any()".to_string()], &active),
        Some(false)
    );
    assert_eq!(
        cargo_context::cfg_predicates_active(&["custom(predicate)".to_string()], &active),
        None
    );
}

fn is_sorted(items: &[String]) -> bool {
    items.windows(2).all(|w| w[0] <= w[1])
}
