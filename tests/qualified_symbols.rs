use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_graph::load;

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "candle-graph-qualified-test-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn write(&self, relative: &str, source: &str) {
        let path = self.path.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
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

const MODULE_A: &str = r#"
pub struct Widget {
    value: Vec<Option<Tensor>>,
}

impl Widget {
    pub(crate) fn build(
        &self,
        input: Option<&Tensor>,
        vb: VarBuilder<'_>,
    ) -> anyhow::Result<Self> {
        todo!()
    }
}

pub fn run(input: &Tensor) -> Result<Tensor> {
    todo!()
}

pub mod inner {
    pub struct Nested;

    pub fn nested_run(value: usize) -> usize {
        value
    }
}
"#;

const MODULE_Z: &str = r#"
pub struct Widget;

impl Widget {
    pub fn build(vb: VarBuilder<'_>) -> Self {
        todo!()
    }
}

pub fn run(input: Tensor) -> Tensor {
    input
}
"#;

#[test]
fn qualified_lookup_retains_colliding_bare_names_and_signatures() {
    let fixture = Fixture::new();
    fixture.write("a.rs", MODULE_A);
    fixture.write("z/mod.rs", MODULE_Z);

    let krate = load::load(fixture.path()).unwrap();

    let structs = krate.struct_candidates("Widget");
    assert_eq!(
        structs
            .iter()
            .map(|item| item.qualified_name.as_str())
            .collect::<Vec<_>>(),
        ["a::Widget", "z::Widget"]
    );
    assert_eq!(krate.struct_candidates("a::Widget")[0].module_path, "a");
    assert_eq!(krate.qualified_structs["a::Widget"].visibility, "pub");
    assert_eq!(
        krate.qualified_structs["a::Widget"].fields[0].ty.text,
        "Vec<Option<Tensor>>"
    );

    let functions = krate.function_candidates("run");
    assert_eq!(
        functions
            .iter()
            .map(|item| item.qualified_name.as_str())
            .collect::<Vec<_>>(),
        ["a::run", "z::run"]
    );
    let a_run = &krate.qualified_functions["a::run"];
    assert_eq!(a_run.param_types, ["& Tensor"]);
    assert_eq!(a_run.return_type, "Result<Tensor >");
    assert_eq!(a_run.visibility, "pub");

    let methods = krate.method_candidates("Widget", "build");
    assert_eq!(
        methods
            .iter()
            .map(|item| item.qualified_name.as_str())
            .collect::<Vec<_>>(),
        ["a::Widget::build", "z::Widget::build"]
    );
    let a_build = &krate.qualified_methods[&("a::Widget".to_string(), "build".to_string())];
    assert_eq!(a_build.module_path, "a");
    assert_eq!(a_build.qualified_type_name, "a::Widget");
    assert_eq!(a_build.qualified_name, "a::Widget::build");
    assert_eq!(a_build.visibility, "pub(crate)");
    assert_eq!(a_build.params, ["self", "input", "vb"]);
    assert_eq!(
        a_build.param_types,
        ["& self", "Option<& Tensor >", "VarBuilder<'_ >"]
    );
    assert_eq!(a_build.return_type, "anyhow::Result<Self >");
    assert_eq!(a_build.vb_params, [2]);

    assert_eq!(
        krate.function_candidates("a::inner::nested_run")[0].qualified_name,
        "a::inner::nested_run"
    );
    assert_eq!(
        krate.struct_candidates("a::inner::Nested")[0].module_path,
        "a::inner"
    );
}

#[test]
fn public_reexports_record_leaf_aliases_and_qualified_targets() {
    let fixture = Fixture::new();
    fixture.write(
        "lib.rs",
        r#"
pub use crate::a::Widget as PublicWidget;
pub use z::{Widget as ZWidget, run as zrun};

mod a;
mod z;
"#,
    );
    fixture.write("a.rs", MODULE_A);
    fixture.write("z.rs", MODULE_Z);

    let krate = load::load(fixture.path()).unwrap();

    let public_widget = krate.reexport_candidates("PublicWidget");
    assert_eq!(public_widget.len(), 1);
    assert_eq!(public_widget[0].qualified_name, "PublicWidget");
    assert_eq!(public_widget[0].target, "a::Widget");
    assert_eq!(public_widget[0].module_path, "");

    assert_eq!(krate.reexport_candidates("ZWidget")[0].target, "z::Widget");
    assert_eq!(krate.reexport_candidates("zrun")[0].target, "z::run");
    assert_eq!(
        krate.reexport_candidates("PublicWidget"),
        krate.reexport_candidates("PublicWidget")
    );
}

#[test]
fn legacy_bare_maps_remain_deterministic_last_definition_wins() {
    let fixture = Fixture::new();
    fixture.write("a.rs", MODULE_A);
    fixture.write("z.rs", MODULE_Z);

    for _ in 0..3 {
        let krate = load::load(fixture.path()).unwrap();
        assert_eq!(krate.structs["Widget"].qualified_name, "z::Widget");
        assert_eq!(krate.functions["run"].qualified_name, "z::run");
        assert_eq!(
            krate.methods[&("Widget".to_string(), "build".to_string())].qualified_name,
            "z::Widget::build"
        );
        assert_eq!(krate.struct_candidates("Widget").len(), 2);
        assert_eq!(krate.function_candidates("run").len(), 2);
        assert_eq!(krate.method_candidates("Widget", "build").len(), 2);
    }
}

#[test]
fn candidate_apis_retain_cfg_alternatives_with_the_same_qualified_name() {
    let fixture = Fixture::new();
    fixture.write(
        "choice.rs",
        r#"
#[cfg(feature = "one")]
pub struct Choice;
#[cfg(not(feature = "one"))]
pub struct Choice;

#[cfg(feature = "one")]
pub fn select(value: F32) -> F32 { value }
#[cfg(not(feature = "one"))]
pub fn select(value: BF16) -> BF16 { value }
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let choices = krate.struct_candidates("choice::Choice");
    assert_eq!(choices.len(), 2);
    assert_eq!(
        choices
            .iter()
            .map(|choice| choice.cfg_predicates.clone())
            .collect::<Vec<_>>(),
        [
            vec!["feature = \"one\"".to_string()],
            vec!["not (feature = \"one\")".to_string()],
        ]
    );
    let functions = krate.function_candidates("choice::select");
    assert_eq!(functions.len(), 2);
    assert!(functions
        .iter()
        .all(|function| !function.cfg_predicates.is_empty()));
    assert_eq!(
        krate.qualified_structs["choice::Choice"].qualified_name,
        "choice::Choice"
    );
    assert_eq!(
        krate.qualified_functions["choice::select"].qualified_name,
        "choice::select"
    );
}
