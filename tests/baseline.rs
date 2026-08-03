//! Integration tests for deterministic structure baselines.
//!
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::BTreeSet, fs};

use candle_graph::baseline::{
    check, compare, load, parse, render, update, Baseline, BaselineError, SCHEMA,
};
use candle_graph::ir::{Acquisition, Certainty, Key, KeySeg, Repeat, SrcSpan, Structure};
use candle_graph::known::ParamKind;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp_dir(label: &str) -> std::path::PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "candle-graph-baseline-{}-{}-{n}",
        label,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn key(segs: &[&str]) -> Key {
    Key {
        segs: segs
            .iter()
            .map(|s| {
                if s.starts_with('{') && s.ends_with('}') {
                    KeySeg::Dynamic {
                        expr: s[1..s.len() - 1].to_string(),
                    }
                } else {
                    KeySeg::Literal(s.to_string())
                }
            })
            .collect(),
    }
}

/// Small two-module structure with a repeated child and one parameter.
fn sample_structure() -> Structure {
    let mut s = Structure::default();
    let root_def = s.add_def("Root".into(), Some("Root::new".into()), SrcSpan::UNKNOWN);
    let block_def = s.add_def("Block".into(), Some("Block::new".into()), SrcSpan::UNKNOWN);

    let root = s.add_instance(
        root_def,
        None,
        None,
        Key::default(),
        "base_vb".into(),
        false,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );
    s.root = Some(root);

    let block = s.add_instance(
        block_def,
        Some(root),
        Some("blocks".into()),
        key(&["layers", "{index}"]),
        "base_vb".into(),
        false,
        Some(Repeat {
            var: "index".into(),
            bound: "cfg.num_layers".into(),
        }),
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );

    let site = s.add_site(
        block_def,
        Acquisition::Constructor {
            func: "linear".into(),
            cite: "linear.rs:84",
        },
        key(&["weight"]),
        ParamKind::Weight,
        Some("[dim, dim]".into()),
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );
    s.add_param(
        site,
        block,
        key(&["layers", "{index}", "weight"]),
        "base_vb".into(),
        Certainty::Certain,
    );

    // Second builder root + conditional param for richer coverage.
    let adapter_def = s.add_def("Adapter".into(), None, SrcSpan::UNKNOWN);
    let adapter = s.add_instance(
        adapter_def,
        Some(root),
        Some("adapter".into()),
        key(&["adapter"]),
        "train_vb".into(),
        false,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Conditional("cfg.enabled".into()),
    );
    let adapter_site = s.add_site(
        adapter_def,
        Acquisition::RawGet {
            method: "get".into(),
        },
        key(&["weight"]),
        ParamKind::Raw,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Conditional("cfg.enabled".into()),
    );
    s.add_param(
        adapter_site,
        adapter,
        key(&["adapter", "weight"]),
        "train_vb".into(),
        Certainty::Conditional("cfg.enabled".into()),
    );

    s
}

#[test]
fn render_is_deterministic_and_omits_arena_ids() {
    let structure = sample_structure();
    let a = render(&structure, &[] as &[&str]);
    let b = render(&structure, &[] as &[&str]);
    assert_eq!(a, b);

    assert!(a.starts_with(&format!("# {SCHEMA}")), "{a}");
    assert!(!a.contains("ModuleInstanceId"), "{a}");
    assert!(!a.contains("ParamId"), "{a}");
    // No raw arena integers as identity fields.
    assert!(!a.contains("\tid="), "{a}");

    // Stable ordering: modules by path; params by (root, key) — not raw line lexicographic order.
    let module_lines: Vec<_> = a.lines().filter(|l| l.starts_with("module\t")).collect();
    let mut sorted_modules = module_lines.clone();
    sorted_modules.sort();
    assert_eq!(module_lines, sorted_modules);

    let param_roots_keys: Vec<_> = Baseline::from_structure(&structure, &[] as &[&str])
        .params
        .iter()
        .map(|p| (p.root.clone(), p.key.clone()))
        .collect();
    let mut expected_order = param_roots_keys.clone();
    expected_order.sort();
    assert_eq!(param_roots_keys, expected_order);
    assert!(
        a.find("root=base_vb\tkind=weight").unwrap() < a.find("root=train_vb\tkind=raw").unwrap()
    );

    assert!(a.contains("root=base_vb"), "{a}");
    assert!(a.contains("root=train_vb"), "{a}");
    assert!(a.contains("repeat=index over cfg.num_layers"), "{a}");
    assert!(a.contains("kind=weight"), "{a}");
    assert!(a.contains("kind=raw"), "{a}");
    assert!(a.contains("source=constructor:linear"), "{a}");
    assert!(a.contains("source=raw_get:get"), "{a}");
    assert!(a.contains("certainty=conditional:cfg.enabled"), "{a}");
    assert!(a.contains("path=Root/blocks:Block"), "{a}");
}

#[test]
fn anonymous_sibling_modules_use_prefixes_to_keep_paths_unique() {
    let mut structure = Structure::default();
    let root_def = structure.add_def("Root".into(), Some("Root::new".into()), SrcSpan::UNKNOWN);
    let lora_def = structure.add_def("Lora".into(), Some("Lora::new".into()), SrcSpan::UNKNOWN);
    let root = structure.add_instance(
        root_def,
        None,
        None,
        Key::default(),
        "vb".into(),
        false,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );
    structure.root = Some(root);
    for prefix in ["layers.0.q", "layers.0.v"] {
        structure.add_instance(
            lora_def,
            Some(root),
            None,
            key(&prefix.split('.').collect::<Vec<_>>()),
            "vb".into(),
            false,
            None,
            SrcSpan::UNKNOWN,
            Certainty::Certain,
        );
    }

    let baseline = Baseline::from_structure(&structure, &[] as &[&str]);
    let paths: BTreeSet<_> = baseline
        .modules
        .iter()
        .map(|module| module.path.as_str())
        .collect();
    assert_eq!(paths.len(), 3);
    assert!(paths.contains("Root/Lora@layers.0.q"));
    assert!(paths.contains("Root/Lora@layers.0.v"));
}

#[test]
fn dataflow_hook_appends_normalized_sorted_lines() {
    let structure = sample_structure();
    let text = render(
        &structure,
        &["  loss->adapter.weight  ", "grad:base", "grad:base", ""],
    );
    let df: Vec<_> = text
        .lines()
        .filter(|l| l.starts_with("dataflow\t"))
        .collect();
    assert_eq!(
        df,
        vec!["dataflow\tgrad:base", "dataflow\tloss->adapter.weight"]
    );

    let parsed = parse(&text).unwrap();
    assert_eq!(
        parsed.dataflow,
        vec!["grad:base".to_string(), "loss->adapter.weight".to_string()]
    );
}

#[test]
fn round_trip_parse_preserves_semantics() {
    let structure = sample_structure();
    let text = render(&structure, &["df:a", "df:b"]);
    let parsed = parse(&text).unwrap();
    assert_eq!(parsed.render(), text);

    let again = Baseline::from_structure(&structure, &["df:b", "df:a"]);
    assert!(compare(&parsed, &again).is_empty());
}

#[test]
fn semantic_differences_report_added_removed_changed() {
    let expected = Baseline::from_structure(&sample_structure(), &["keep", "gone"]);

    let mut actual = Baseline::from_structure(&sample_structure(), &["keep", "new"]);
    // Change adapter param certainty (conditional -> certain).
    let adapter = actual
        .params
        .iter_mut()
        .find(|p| p.root == "train_vb")
        .unwrap();
    adapter.certainty = "certain".into();
    // Drop the base weight param.
    actual.params.retain(|p| p.root != "base_vb");
    // Change repeated module bound.
    let block = actual
        .modules
        .iter_mut()
        .find(|m| m.path.contains("blocks"))
        .unwrap();
    block.repeat.as_mut().unwrap().bound = "other".into();

    let diff = compare(&actual, &expected);
    assert!(!diff.is_empty());

    let added = diff.added.join("\n");
    let removed = diff.removed.join("\n");
    let changed = diff.to_string();

    assert!(
        added.contains("dataflow\tnew"),
        "added should mention new dataflow: {added}"
    );
    assert!(
        removed.contains("dataflow\tgone"),
        "removed should mention gone dataflow: {removed}"
    );
    assert!(
        removed.contains("key=layers.{index}.weight"),
        "removed should include base weight param: {removed}"
    );
    assert!(
        changed.contains("### changed"),
        "diff display should have changed section: {changed}"
    );
    assert!(
        changed.contains("module") && changed.contains("repeat=index over other"),
        "changed module should show new repeat: {changed}"
    );
    assert!(
        changed.contains("param") && changed.contains("certainty=certain"),
        "changed param certainty should appear: {changed}"
    );
}

#[test]
fn missing_baseline_is_a_clear_error() {
    let dir = temp_dir("missing");
    let path = dir.join("does-not-exist.baseline");
    let err = load(&path).unwrap_err();
    match &err {
        BaselineError::Missing(p) => assert_eq!(p, &path),
        other => panic!("expected Missing, got {other}"),
    }
    assert!(err.to_string().contains("baseline not found"));

    let structure = sample_structure();
    let check_err = check(&structure, &path, &[] as &[&str]).unwrap_err();
    assert!(matches!(check_err, BaselineError::Missing(_)));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn invalid_baseline_is_a_clear_error() {
    let err = parse("not a baseline\n").unwrap_err();
    match err {
        BaselineError::Invalid { message, .. } => {
            assert!(message.contains("expected header"), "{message}");
        }
        other => panic!("expected Invalid, got {other}"),
    }

    let dir = temp_dir("invalid");
    let path = dir.join("bad.baseline");
    fs::write(&path, "# candle-graph/baseline/1\nmodule\tnope\n").unwrap();
    let err = load(&path).unwrap_err();
    match err {
        BaselineError::Invalid {
            path: Some(p),
            message,
        } => {
            assert_eq!(p, path);
            assert!(
                message.contains("key=value") || message.contains("missing field"),
                "{message}"
            );
        }
        other => panic!("expected Invalid with path, got {other}"),
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn check_passes_on_matching_baseline_and_fails_on_drift() {
    let dir = temp_dir("check");
    let path = dir.join("model.baseline");
    let structure = sample_structure();
    update(&structure, &path, &["df:1"]).unwrap();

    check(&structure, &path, &["df:1"]).unwrap();

    let err = check(&structure, &path, &["df:2"]).unwrap_err();
    match err {
        BaselineError::Mismatch(diff) => {
            assert!(!diff.is_empty());
            let text = diff.to_string();
            assert!(
                text.contains("### added") || text.contains("### removed"),
                "{text}"
            );
        }
        other => panic!("expected Mismatch, got {other}"),
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn update_is_atomic_and_replaces_contents() {
    let dir = temp_dir("update");
    let path = dir.join("model.baseline");
    let structure = sample_structure();

    update(&structure, &path, &[] as &[&str]).unwrap();
    assert!(path.is_file());
    // No leftover temp beside the target.
    let leftovers: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    let first = fs::read_to_string(&path).unwrap();
    assert!(first.contains("path=Root"));

    // Mutate structure and update again.
    let mut next = sample_structure();
    next.params[0].certainty = Certainty::Unknown("revisit".into());
    update(&next, &path, &["extra"]).unwrap();

    let second = fs::read_to_string(&path).unwrap();
    assert_ne!(first, second);
    assert!(second.contains("certainty=unknown:revisit"));
    assert!(second.contains("dataflow\textra"));
    assert!(!path.with_extension("baseline.tmp").exists());

    let leftovers: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn insertion_order_does_not_affect_canonical_text() {
    let a = sample_structure();

    // Build the same logical graph with defs/instances/params inserted in a different order.
    let mut b = Structure::default();
    let block_def = b.add_def("Block".into(), Some("Block::new".into()), SrcSpan::UNKNOWN);
    let adapter_def = b.add_def("Adapter".into(), None, SrcSpan::UNKNOWN);
    let root_def = b.add_def("Root".into(), Some("Root::new".into()), SrcSpan::UNKNOWN);

    let root = b.add_instance(
        root_def,
        None,
        None,
        Key::default(),
        "base_vb".into(),
        false,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );
    b.root = Some(root);

    let adapter = b.add_instance(
        adapter_def,
        Some(root),
        Some("adapter".into()),
        key(&["adapter"]),
        "train_vb".into(),
        false,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Conditional("cfg.enabled".into()),
    );
    let block = b.add_instance(
        block_def,
        Some(root),
        Some("blocks".into()),
        key(&["layers", "{index}"]),
        "base_vb".into(),
        false,
        Some(Repeat {
            var: "index".into(),
            bound: "cfg.num_layers".into(),
        }),
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );

    let adapter_site = b.add_site(
        adapter_def,
        Acquisition::RawGet {
            method: "get".into(),
        },
        key(&["weight"]),
        ParamKind::Raw,
        None,
        SrcSpan::UNKNOWN,
        Certainty::Conditional("cfg.enabled".into()),
    );
    b.add_param(
        adapter_site,
        adapter,
        key(&["adapter", "weight"]),
        "train_vb".into(),
        Certainty::Conditional("cfg.enabled".into()),
    );

    let site = b.add_site(
        block_def,
        Acquisition::Constructor {
            func: "linear".into(),
            cite: "linear.rs:84",
        },
        key(&["weight"]),
        ParamKind::Weight,
        Some("[dim, dim]".into()),
        SrcSpan::UNKNOWN,
        Certainty::Certain,
    );
    b.add_param(
        site,
        block,
        key(&["layers", "{index}", "weight"]),
        "base_vb".into(),
        Certainty::Certain,
    );

    assert_eq!(
        render(&a, &["z", "a"]),
        render(&b, &["a", "z"]),
        "canonical text must ignore arena insertion order"
    );
}
