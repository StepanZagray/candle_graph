use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_graph::contracts;
use candle_graph::load;
use candle_graph::model_ir::{DeviceFact, LayoutFact, TensorRole};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "candle-graph-contracts-test-{}-{unique}",
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

fn tensor<'a>(
    function: &'a contracts::FunctionContracts,
    name: &str,
) -> &'a candle_graph::model_ir::TensorContract {
    function
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name} in {:?}", function.tensors))
}

#[test]
fn infers_symbolic_dims_dtype_device_and_contiguous_layout() {
    let fixture = Fixture::new();
    fixture.write(
        "model/adapter.rs",
        r#"
pub fn adapt(input: &Tensor, device: &Device) -> Result<Tensor> {
    let (batch, tokens, hidden) = input.dims3()?;
    let view = input.narrow(1, 0, tokens)?.transpose(1, 2)?.contiguous()?;
    let output_shape = (batch, hidden, tokens);
    let output = view
        .reshape(output_shape)?
        .to_dtype(DType::BF16)?
        .to_device(device)?;
    Ok(output)
}
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let functions = contracts::functions_named(&krate, "model::adapter::adapt");
    assert_eq!(functions.len(), 1);
    let function = &functions[0];

    let input = tensor(function, "input");
    assert_eq!(input.role, TensorRole::Input);
    assert_eq!(input.shape.rank, Some(3));
    assert_eq!(
        input
            .shape
            .dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>(),
        ["batch", "tokens", "hidden"]
    );

    let output = tensor(function, "output");
    assert_eq!(output.shape.rank, Some(3));
    assert_eq!(
        output
            .shape
            .dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>(),
        ["batch", "hidden", "tokens"]
    );
    assert_eq!(output.dtype, "BF16");
    assert_eq!(output.device, DeviceFact::SameAs("device".to_string()));
    assert_eq!(output.layout, LayoutFact::Contiguous);

    let returned = tensor(function, "return");
    assert_eq!(returned.role, TensorRole::Output);
    assert_eq!(returned.shape, output.shape);
    assert!(returned
        .evidence
        .iter()
        .any(|evidence| evidence.source.as_deref() == Some("model/adapter.rs:10")));
}

#[test]
fn tensor_constructor_infers_shape_and_dtype_but_to_vec_is_not_a_tensor() {
    let fixture = Fixture::new();
    fixture.write(
        "batch.rs",
        r#"
pub fn make_batch(device: &Device, batch: usize, seq: usize) -> Result<(Tensor, Vec<u32>)> {
    let ids = Tensor::from_vec(vec![1u32, 2u32], (batch, seq), device)?;
    let host = ids.to_vec2::<u32>()?;
    Ok((ids, host))
}
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let function = &contracts::functions_named(&krate, "batch::make_batch")[0];
    let ids = tensor(function, "ids");

    assert_eq!(ids.shape.rank, Some(2));
    assert_eq!(
        ids.shape
            .dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>(),
        ["batch", "seq"]
    );
    assert_eq!(ids.dtype, "U32");
    assert_eq!(ids.device, DeviceFact::SameAs("device".to_string()));
    assert_eq!(ids.layout, LayoutFact::Contiguous);
    assert_eq!(ids.requires_grad, Some(false));
    assert!(!function.tensors.iter().any(|tensor| tensor.name == "host"));
    assert_eq!(
        function
            .tensors
            .iter()
            .filter(|tensor| tensor.role == TensorRole::Output)
            .count(),
        1
    );
}

#[test]
fn permute_and_narrow_preserve_symbolic_rank_and_detach_fact() {
    let fixture = Fixture::new();
    fixture.write(
        "layout.rs",
        r#"
pub fn reorder(x: Tensor) -> Tensor {
    let (batch, slots, hidden) = x.dims3().unwrap();
    x.permute([0, 2, 1])
        .unwrap()
        .narrow(2, 0, slots)
        .unwrap()
        .detach()
}
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let function = &contracts::functions_named(&krate, "layout::reorder")[0];
    let returned = tensor(function, "return");

    assert_eq!(
        returned
            .shape
            .dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>(),
        ["batch", "hidden", "slots"]
    );
    assert_eq!(returned.layout, LayoutFact::Strided);
    assert_eq!(returned.requires_grad, Some(false));
}

#[test]
fn slice_destructuring_of_dims_establishes_rank_without_guessing() {
    let fixture = Fixture::new();
    fixture.write(
        "dims.rs",
        r#"
pub fn inspect(x: &Tensor) -> Tensor {
    let [batch, tokens] = x.dims() else {
        panic!("expected a matrix")
    };
    x.reshape((batch, tokens)).unwrap()
}
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let function = &contracts::functions_named(&krate, "dims::inspect")[0];
    let input = tensor(function, "x");
    assert_eq!(input.shape.rank, Some(2));
    assert_eq!(
        input
            .shape
            .dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>(),
        ["batch", "tokens"]
    );
}

#[test]
fn qualified_method_queries_keep_bare_owner_collisions() {
    let fixture = Fixture::new();
    fixture.write(
        "a.rs",
        r#"
pub struct Model;
impl Model {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, tokens) = input.dims2()?;
        input.reshape((batch, tokens))
    }
}
"#,
    );
    fixture.write(
        "b.rs",
        r#"
pub struct Model;
impl Model {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, slots, hidden) = input.dims3()?;
        input.reshape((batch, slots, hidden))
    }
}
"#,
    );

    let krate = load::load(fixture.path()).unwrap();
    let collided = contracts::methods_named(&krate, "Model", "forward");
    assert_eq!(
        collided
            .iter()
            .map(|function| function.qualified_name.as_str())
            .collect::<Vec<_>>(),
        ["a::Model::forward", "b::Model::forward"]
    );
    let exact = contracts::methods_named(&krate, "b::Model", "forward");
    assert_eq!(exact.len(), 1);
    assert_eq!(tensor(&exact[0], "input").shape.rank, Some(3));
}
