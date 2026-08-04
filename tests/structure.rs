use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_graph::{
    extract::Extractor,
    ir::{Key, KeySeg},
    load, verify,
};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(source: &str) -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("candle-graph-test-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("model.rs"), source).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

const MODEL: &str = r#"
struct Root {
    blocks: Vec<Block>,
}

struct Block {
    norm: BatchNorm,
}

impl Root {
    fn new(base_vb: VarBuilder<'_>, train_vb: VarBuilder<'_>) -> Result<Self> {
        assert!(true);
        if false {
            anyhow::bail!("not reached");
        }

        let mut blocks = Vec::new();
        for index in 0..depth {
            blocks.push(Block::new(
                base_vb.pp(format!("pre_local_block_{index}")),
            )?);
        }

        let _adapter = enabled
            .then(|| nn::linear_no_bias(dim, dim, train_vb.pp("adapter")))
            .transpose()?;
        let _head = build_head(base_vb.pp("head"))?;
        Ok(Self { blocks })
    }
}

fn build_head(vb: VarBuilder<'_>) -> Result<Linear> {
    nn::linear_no_bias(dim, dim, vb)
}

impl Block {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let norm = nn::batch_norm(dim, 1e-5, vb.pp("norm").to_dtype(DType::F32))?;
        Ok(Self { norm })
    }
}
"#;

#[test]
fn extracts_formatted_prefixes_dtype_builders_and_builder_roots() {
    let fixture = Fixture::new(MODEL);
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();

    let keys: Vec<_> = structure
        .params
        .iter()
        .map(|param| (param.root.as_str(), param.key.to_string()))
        .collect();

    assert!(keys.contains(&(
        "base_vb",
        "pre_local_block_{index}.norm.running_mean".to_string()
    )));
    assert!(keys.contains(&(
        "base_vb",
        "pre_local_block_{index}.norm.running_var".to_string()
    )));
    assert!(keys.contains(&("train_vb", "adapter.weight".to_string())));
    assert!(keys.contains(&("base_vb", "head.weight".to_string())));
    assert!(
        structure.diagnostics.is_empty(),
        "{:?}",
        structure.diagnostics
    );
}

#[test]
fn unified_model_preserves_multi_root_identity() {
    let fixture = Fixture::new(MODEL);
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();

    let adapter = structure
        .params
        .iter()
        .find(|param| param.key.to_string() == "adapter.weight")
        .unwrap();
    assert_eq!(adapter.root, "train_vb");
}

#[test]
fn checkpoint_verification_separates_roots_and_conditional_absence() {
    use candle_graph::ir::Certainty;
    use candle_graph::model_ir::{
        Confidence, Evidence, EvidenceKind, ModelIr, Parameter, ParameterRole, StableId,
    };

    let fixture = Fixture::new(MODEL);
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();
    let mut model = ModelIr::empty(StableId::new("analysis", ["test"]));
    model.parameters = structure
        .params
        .iter()
        .map(|param| {
            let role = match &param.certainty {
                Certainty::Certain => ParameterRole::Optimized,
                Certainty::Conditional(_) => ParameterRole::Conditional,
                Certainty::Unknown(_) => ParameterRole::Unknown,
            };
            Parameter {
                id: StableId::new(
                    "parameter",
                    ["test", param.root.as_str(), param.key.to_string().as_str()],
                ),
                component: StableId::new("component", ["Root"]),
                module: StableId::new("module", ["Root"]),
                key: param.key.to_string(),
                builder_root: param.root.clone(),
                role,
                kind: String::new(),
                symbolic_shape: None,
                checkpoint_shape: None,
                checkpoint_dtype: None,
                source: String::new(),
                uses: Vec::new(),
                optimizer_memberships: Vec::new(),
                evidence: vec![Evidence {
                    kind: EvidenceKind::Source,
                    confidence: match &param.certainty {
                        Certainty::Certain => Confidence::Proven,
                        Certainty::Conditional(_) => Confidence::Heuristic,
                        Certainty::Unknown(_) => Confidence::Unknown,
                    },
                    source: None,
                    detail: String::new(),
                }],
            }
        })
        .collect();

    let header = verify::Header::from([
        (
            "pre_local_block_0.norm.running_mean".to_string(),
            verify::TensorInfo {
                shape: vec![8],
                dtype: "F32".to_string(),
            },
        ),
        (
            "pre_local_block_0.norm.running_var".to_string(),
            verify::TensorInfo {
                shape: vec![8],
                dtype: "F32".to_string(),
            },
        ),
    ]);
    let result = verify::verify_model(&mut model, &header, "base_vb");

    assert_eq!(result.root, "base_vb");
    assert_eq!(result.unclaimed, Vec::<String>::new());
    assert!(result.skipped_other_root > 0);
    assert!(result
        .missing_conditional
        .iter()
        .all(|key| key.contains(".norm.weight") || key.contains(".norm.bias")));
}

#[test]
fn reads_only_the_safetensors_header_format() {
    let fixture = Fixture::new("");
    let path = fixture.path().join("tiny.safetensors");
    let json = serde_json::to_vec(&serde_json::json!({
        "__metadata__": {"format": "pt"},
        "layer.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    }))
    .unwrap();
    let mut file = Vec::with_capacity(8 + json.len() + 24);
    file.extend_from_slice(&(json.len() as u64).to_le_bytes());
    file.extend_from_slice(&json);
    file.extend_from_slice(&[0u8; 24]);
    std::fs::write(&path, file).unwrap();

    let header = verify::read_header(&path).unwrap();
    assert_eq!(header.len(), 1);
    assert_eq!(header["layer.weight"].shape, vec![2, 3]);
    assert_eq!(header["layer.weight"].dtype, "F32");
}

#[test]
fn known_constructor_never_falls_back_to_the_wrong_builder_argument() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let _ = nn::linear(dim, vb.pp("wrong"), missing_builder)?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();

    assert!(structure.params.is_empty());
    assert!(structure
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("could not be resolved")));
}

#[test]
fn candle_011_catalog_includes_group_norm_and_prelu() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let _ = nn::group_norm(groups, channels, eps, vb.pp("group"))?;
        let _ = nn::prelu(Some(channels), vb.pp("activation"))?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::for_candle_version(&krate, Some("0.11.0"))
        .run("Root", None)
        .unwrap();
    let keys: Vec<_> = structure
        .params
        .iter()
        .map(|parameter| parameter.key.to_string())
        .collect();

    assert!(keys.contains(&"group.weight".to_string()));
    assert!(keys.contains(&"group.bias".to_string()));
    assert!(keys.contains(&"activation.weight".to_string()));
}

#[test]
fn stale_candle_constructor_catalog_is_not_applied_to_other_versions() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let _ = nn::linear(dim, dim, vb.pp("projection"))?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::for_candle_version(&krate, Some("0.10.2"))
        .run("Root", None)
        .unwrap();

    assert!(structure.params.is_empty());
    assert!(structure.diagnostics.iter().any(|diagnostic| {
        diagnostic.message.contains("unresolved constructor")
            && diagnostic.message.contains("VarBuilder")
    }));
}

#[test]
fn unresolved_builder_consuming_calls_are_reported_not_silently_dropped() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let _ = external_crate::CustomLayer::build(vb.pp("custom"))?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();

    assert!(structure.params.is_empty());
    assert!(structure.diagnostics.iter().any(|diagnostic| {
        diagnostic.message.contains("unresolved constructor")
            && diagnostic.message.contains("VarBuilder")
    }));
}

#[test]
fn unrelated_type_containing_varbuilder_is_not_an_entry_builder() {
    let fixture = Fixture::new(
        r#"
struct Root;
struct MyVarBuilderConfig;
impl Root {
    fn new(config: MyVarBuilderConfig) -> Result<Self> { Ok(Self) }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let error = Extractor::new(&krate).run("Root", None).unwrap_err();
    assert!(error
        .to_string()
        .contains("no constructor taking a VarBuilder"));
}

#[test]
fn loop_and_while_constructor_bodies_are_not_skipped() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        while enabled {
            let _ = nn::linear_no_bias(dim, dim, vb.pp("while_layer"))?;
        }
        loop {
            let _ = nn::embedding(vocab, dim, vb.pp("loop_layer"))?;
            break;
        }
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();
    let keys: Vec<_> = structure
        .params
        .iter()
        .map(|param| param.key.to_string())
        .collect();
    assert!(keys.contains(&"while_layer.weight".to_string()));
    assert!(keys.contains(&"loop_layer.weight".to_string()));
}

#[test]
fn bare_colliding_roots_fail_instead_of_using_last_definition() {
    let fixture = Fixture::new("");
    std::fs::write(
        fixture.path().join("a.rs"),
        "pub struct Root; impl Root { fn new(vb: VarBuilder<'_>) -> Self { Self } }",
    )
    .unwrap();
    std::fs::write(
        fixture.path().join("z.rs"),
        "pub struct Root; impl Root { fn new(vb: VarBuilder<'_>) -> Self { Self } }",
    )
    .unwrap();
    let krate = load::load(fixture.path()).unwrap();

    let error = Extractor::new(&krate).run("Root", None).unwrap_err();
    assert!(error.to_string().contains("ambiguous"));
}

#[test]
fn template_keys_preserve_literal_prefixes_when_matching_checkpoints() {
    let key = Key::default()
        .push(KeySeg::Template {
            text: "pre_local_block_{index}".to_string(),
        })
        .push_literal("weight");

    assert!(key.matches("pre_local_block_12.weight"));
    assert!(!key.matches("unrelated_12.weight"));
    assert!(!key.matches("pre_local.weight"));
}

#[test]
fn constructor_shapes_are_specific_to_each_parameter_leaf() {
    let fixture = Fixture::new(
        r#"
struct Root;
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let _projection = nn::linear(input_dim, output_dim, vb.pp("projection"))?;
        let _tokens = nn::embedding(vocab_size, hidden_dim, vb.pp("tokens"))?;
        let _conv = nn::conv2d(in_channels, out_channels, kernel, cfg, vb.pp("conv"))?;
        let _deconv =
            nn::conv_transpose2d(in_channels, out_channels, kernel, cfg, vb.pp("deconv"))?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();
    let shape = |key: &str| {
        let parameter = structure
            .params
            .iter()
            .find(|parameter| parameter.key.to_string() == key)
            .unwrap();
        structure.site(parameter.site).shape.clone()
    };

    assert_eq!(
        shape("projection.weight").as_deref(),
        Some("(output_dim, input_dim)")
    );
    assert_eq!(shape("projection.bias").as_deref(), Some("output_dim"));
    assert_eq!(
        shape("tokens.weight").as_deref(),
        Some("(vocab_size, hidden_dim)")
    );
    assert_eq!(
        shape("conv.weight").as_deref(),
        Some("(out_channels, in_channels / groups(cfg), kernel, kernel)")
    );
    assert_eq!(
        shape("deconv.weight").as_deref(),
        Some("(in_channels, out_channels, kernel, kernel)")
    );
}

#[test]
fn same_type_constructor_helpers_do_not_create_nested_self_modules() {
    let fixture = Fixture::new(
        r#"
struct Root { projection: Linear }
impl Root {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        Self::new_impl(vb)
    }
    fn new_impl(vb: VarBuilder<'_>) -> Result<Self> {
        let projection = nn::linear(4, 8, vb.pp("projection"))?;
        Ok(Self { projection })
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    let structure = Extractor::new(&krate).run("Root", None).unwrap();

    assert_eq!(structure.instances.len(), 1);
    assert!(
        structure
            .params
            .iter()
            .any(|parameter| parameter.key.to_string() == "projection.weight"),
        "{:#?}",
        structure.diagnostics
    );
}

#[test]
fn imported_builder_alias_and_nonstandard_constructor_are_supported() {
    let fixture = Fixture::new(
        r#"
use candle_nn::VarBuilder as VB;
use candle_nn as nn;

struct Root;
impl Root {
    fn build(vb: VB<'_>) -> Result<Self> {
        let _projection = nn::linear(4, 8, vb.pp("projection"))?;
        Ok(Self)
    }
}
"#,
    );
    let krate = load::load(fixture.path()).unwrap();
    assert_eq!(
        krate.resolve_unambiguous_import_path(&["nn".to_string(), "linear".to_string()]),
        ["candle_nn".to_string(), "linear".to_string()]
    );
    let constructor = &krate.method_candidates("Root", "build")[0];
    assert_eq!(constructor.vb_params, [0]);
    let structure = Extractor::new(&krate).run("Root", None).unwrap();
    assert!(
        structure
            .params
            .iter()
            .any(|parameter| parameter.key.to_string() == "projection.weight"),
        "diagnostics={:#?}",
        structure.diagnostics
    );
}
