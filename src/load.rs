//! Crate loading and the symbol table.
//!
//! We deliberately do *not* attempt real type inference. The measured property that makes this
//! sound for candle model code is that such code is monomorphic and uses inherent `fn` methods
//! on concrete structs — no trait objects, no generic modules. So a crate-local
//! `struct -> field -> type` table plus `type -> methods` is enough to resolve
//! `SelfAttention::new(cfg, vb.pp("self_attn"))` to a body.
//!
//! Where that assumption breaks we record a diagnostic instead of guessing.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::ir::SrcSpan;

pub struct SourceFile {
    pub path: PathBuf,
    /// Path relative to the scanned root, used in all human-facing output.
    pub rel: String,
}

#[derive(Debug, Clone)]
pub struct LoadDiagnostic {
    pub path: String,
    pub message: String,
}

/// A parsed field type, decomposed just enough to see through the containers candle model code
/// actually uses.
#[derive(Debug, Clone)]
pub struct TypeRef {
    /// Full source text, e.g. `Vec<Layer>`.
    pub text: String,
    /// Innermost named type after peeling containers, e.g. `Layer`.
    pub base: String,
    pub container: Container,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Plain,
    Vec,
    Option,
    Box,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub ty: TypeRef,
    pub span: SrcSpan,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    /// Rust module containing the definition (`""` means the crate root).
    pub module_path: String,
    /// Module-qualified identity, for example `model::encoder::Encoder`.
    pub qualified_name: String,
    /// Source spelling of the visibility. Private items use the empty string.
    pub visibility: String,
    /// `#[cfg(...)]` predicates inherited from inline modules and attached to this definition.
    pub cfg_predicates: Vec<String>,
    pub fields: Vec<FieldDef>,
    pub span: SrcSpan,
}

/// An inherent method on a concrete type.
#[derive(Clone)]
pub struct ImplFn {
    pub type_name: String,
    pub fn_name: String,
    /// Implemented trait for trait methods (for example `candle_core::Module`).
    /// `None` denotes an inherent method.
    pub trait_name: Option<String>,
    /// Rust module containing the definition (`""` means the crate root).
    pub module_path: String,
    /// Qualified owning type for methods. Empty for free functions.
    pub qualified_type_name: String,
    /// Fully-qualified function identity.
    pub qualified_name: String,
    /// Source spelling of the visibility. Private items use the empty string.
    pub visibility: String,
    /// `#[cfg(...)]` predicates inherited from modules/impls and attached to this function.
    pub cfg_predicates: Vec<String>,
    /// Parameter names in order; used to find which argument is the `VarBuilder`.
    pub params: Vec<String>,
    /// Parameter type source text in the same order as `params`.
    ///
    /// Receivers are represented by their source spelling (`self`, `&self`, etc.).
    pub param_types: Vec<String>,
    /// Explicit return type source text, or `()` when omitted.
    pub return_type: String,
    /// Indices of every parameter whose type mentions `VarBuilder`.
    ///
    /// Plural because models routinely take more than one: a constructor may take a frozen
    /// `base_vb` (mmapped weights) and a trainable `train_vb` (a `VarMap`). Which
    /// builder a tensor came from *is* the frozen/trainable distinction, so collapsing them
    /// would discard the thing the gradient analysis will need most.
    pub vb_params: Vec<usize>,
    pub block: syn::Block,
    pub span: SrcSpan,
}

/// A public `use` leaf. Keeping the syntactic target is useful even when that target cannot be
/// resolved without Cargo/rustc name resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicReexport {
    /// Name made public by the re-export (the alias after `as`, when present).
    pub name: String,
    /// Qualified name of the public binding.
    pub qualified_name: String,
    /// Best-effort module-qualified target path.
    pub target: String,
    pub module_path: String,
    pub span: SrcSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportBinding {
    pub module_path: String,
    pub alias: String,
    pub target: String,
}

#[derive(Default)]
pub struct Crate {
    pub files: Vec<SourceFile>,
    /// Filesystem and parse failures retained as structured incomplete-analysis evidence.
    pub diagnostics: Vec<LoadDiagnostic>,
    /// Legacy bare-name map. When names collide, deterministic source traversal preserves the
    /// historical last-definition-wins behavior. Use `struct_candidates` for new analysis.
    pub structs: HashMap<String, StructDef>,
    /// `(type_name, fn_name) -> method`.
    pub methods: HashMap<(String, String), ImplFn>,
    /// Free functions by name, for `fn build_encoder(vb: VarBuilder) -> ..` helpers.
    pub functions: HashMap<String, ImplFn>,
    /// Exact qualified identities. These maps make unambiguous lookups cheap.
    pub qualified_structs: HashMap<String, StructDef>,
    pub qualified_methods: HashMap<(String, String), ImplFn>,
    pub qualified_functions: HashMap<String, ImplFn>,
    pub public_reexports: Vec<PublicReexport>,
    pub imports: Vec<ImportBinding>,
    /// Rust crate identifier to Cargo package name, including dependency renames.
    pub dependency_aliases: HashMap<String, String>,
    // Complete definition lists back the collision-safe candidate APIs. In particular, they also
    // retain cfg-gated duplicate qualified definitions instead of silently overwriting them.
    all_structs: Vec<StructDef>,
    all_methods: Vec<ImplFn>,
    all_functions: Vec<ImplFn>,
}

impl Crate {
    /// Every struct definition, including colliding and cfg-gated alternatives.
    pub fn all_structs(&self) -> impl Iterator<Item = &StructDef> {
        self.all_structs.iter()
    }

    /// Every inherent method, including colliding and cfg-gated alternatives.
    pub fn all_methods(&self) -> impl Iterator<Item = &ImplFn> {
        self.all_methods.iter()
    }

    /// Every free function, including colliding and cfg-gated alternatives.
    pub fn all_functions(&self) -> impl Iterator<Item = &ImplFn> {
        self.all_functions.iter()
    }

    pub fn field_type(&self, struct_name: &str, field: &str) -> Option<&TypeRef> {
        self.structs
            .get(struct_name)?
            .fields
            .iter()
            .find(|f| f.name == field)
            .map(|f| &f.ty)
    }

    pub fn file_label(&self, span: SrcSpan) -> String {
        match self.files.get(span.file) {
            Some(f) => format!("{}:{}", f.rel, span.line),
            None => format!("<unknown>:{}", span.line),
        }
    }

    /// All structs matching either a bare or exact qualified name, sorted by qualified identity
    /// and source location. Unlike the legacy map, collisions are never discarded.
    pub fn struct_candidates(&self, name: &str) -> Vec<&StructDef> {
        let qualified = name.contains("::");
        let mut found: Vec<_> = self
            .all_structs
            .iter()
            .filter(|def| {
                if qualified {
                    def.qualified_name == name
                } else {
                    def.name == name
                }
            })
            .collect();
        sort_struct_candidates(&mut found);
        found
    }

    /// All free functions matching either a bare or exact qualified name.
    pub fn function_candidates(&self, name: &str) -> Vec<&ImplFn> {
        let qualified = name.contains("::");
        let mut found: Vec<_> = self
            .all_functions
            .iter()
            .filter(|func| {
                if qualified {
                    func.qualified_name == name
                } else {
                    func.fn_name == name
                }
            })
            .collect();
        sort_fn_candidates(&mut found);
        found
    }

    /// All inherent methods matching a bare or qualified owner and a function name.
    pub fn method_candidates(&self, type_name: &str, fn_name: &str) -> Vec<&ImplFn> {
        let qualified = type_name.contains("::");
        let mut found: Vec<_> = self
            .all_methods
            .iter()
            .filter(|func| {
                func.fn_name == fn_name
                    && if qualified {
                        func.qualified_type_name == type_name
                    } else {
                        func.type_name == type_name
                    }
            })
            .collect();
        // Rust method syntax selects an inherent method ahead of trait methods. Preserve trait
        // methods when no inherent definition exists so the ubiquitous `impl Module for Model`
        // entrypoint remains analyzable.
        if found.iter().any(|func| func.trait_name.is_none()) {
            found.retain(|func| func.trait_name.is_none());
        }
        sort_fn_candidates(&mut found);
        found
    }

    /// Public re-export leaves matching a bare binding or exact qualified binding.
    pub fn reexport_candidates(&self, name: &str) -> Vec<&PublicReexport> {
        let qualified = name.contains("::");
        let mut found: Vec<_> = self
            .public_reexports
            .iter()
            .filter(|item| {
                if qualified {
                    item.qualified_name == name
                } else {
                    item.name == name
                }
            })
            .collect();
        found.sort_by(|a, b| {
            (&a.qualified_name, a.span.file, a.span.line, a.span.col).cmp(&(
                &b.qualified_name,
                b.span.file,
                b.span.line,
                b.span.col,
            ))
        });
        found
    }

    /// Resolve a source path through an explicit `use` binding in the containing module.
    pub fn resolve_import_path(&self, module_path: &str, segments: &[String]) -> Vec<String> {
        let Some(first) = segments.first() else {
            return Vec::new();
        };
        let Some(binding) = self
            .imports
            .iter()
            .find(|binding| binding.module_path == module_path && binding.alias == *first)
        else {
            return self.resolve_dependency_alias(segments);
        };
        let mut resolved = binding
            .target
            .split("::")
            .map(str::to_string)
            .collect::<Vec<_>>();
        resolved.extend(segments.iter().skip(1).cloned());
        self.resolve_dependency_alias(&resolved)
    }

    /// Resolve an alias when every binding of that spelling agrees on the same external target.
    /// This is used by legacy structure extraction where the current inline module is not carried
    /// through the compact arena.
    pub fn resolve_unambiguous_import_path(&self, segments: &[String]) -> Vec<String> {
        let Some(first) = segments.first() else {
            return Vec::new();
        };
        let mut targets = self
            .imports
            .iter()
            .filter(|binding| binding.alias == *first)
            .map(|binding| binding.target.as_str())
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        let [target] = targets.as_slice() else {
            return self.resolve_dependency_alias(segments);
        };
        let mut resolved = target.split("::").map(str::to_string).collect::<Vec<_>>();
        resolved.extend(segments.iter().skip(1).cloned());
        self.resolve_dependency_alias(&resolved)
    }

    pub fn set_dependency_aliases(&mut self, aliases: impl IntoIterator<Item = (String, String)>) {
        self.dependency_aliases = aliases.into_iter().collect();
        for binding in &mut self.imports {
            let parts = binding.target.split("::").collect::<Vec<_>>();
            let dependency = parts
                .first()
                .and_then(|name| self.dependency_aliases.get(*name))
                .map(|package| (0usize, package))
                .or_else(|| {
                    parts
                        .last()
                        .and_then(|name| self.dependency_aliases.get(*name))
                        .map(|package| (parts.len().saturating_sub(1), package))
                });
            if let Some((index, package)) = dependency {
                let mut resolved = if index == parts.len().saturating_sub(1) {
                    Vec::new()
                } else {
                    parts[..index]
                        .iter()
                        .map(|part| (*part).to_string())
                        .collect()
                };
                resolved.push(package.replace('-', "_"));
                resolved.extend(parts.iter().skip(index + 1).map(|part| (*part).to_string()));
                binding.target = resolved.join("::");
            }
        }
    }

    fn resolve_dependency_alias(&self, segments: &[String]) -> Vec<String> {
        let Some(first) = segments.first() else {
            return Vec::new();
        };
        let Some(package) = self.dependency_aliases.get(first) else {
            return segments.to_vec();
        };
        let mut resolved = vec![package.replace('-', "_")];
        resolved.extend(segments.iter().skip(1).cloned());
        resolved
    }
}

fn sort_struct_candidates(found: &mut Vec<&StructDef>) {
    found.sort_by(|a, b| {
        (&a.qualified_name, a.span.file, a.span.line, a.span.col).cmp(&(
            &b.qualified_name,
            b.span.file,
            b.span.line,
            b.span.col,
        ))
    });
}

fn sort_fn_candidates(found: &mut Vec<&ImplFn>) {
    found.sort_by(|a, b| {
        (&a.qualified_name, a.span.file, a.span.line, a.span.col).cmp(&(
            &b.qualified_name,
            b.span.file,
            b.span.line,
            b.span.col,
        ))
    });
}

/// Parse every `.rs` file under `root`, skipping `target/` and hidden directories.
pub fn load(root: &Path) -> Result<Crate> {
    let mut krate = Crate::default();

    for entry in WalkDir::new(root)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(matches!(
                name.as_ref(),
                "target" | "artifacts" | "runs" | "node_modules"
            ) || name.starts_with('.'))
        })
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                krate.diagnostics.push(LoadDiagnostic {
                    path: error
                        .path()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_else(|| root.to_string_lossy().into_owned()),
                    message: format!("filesystem traversal failed: {error}"),
                });
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }

        let path = entry.path().to_path_buf();
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;

        // A file that does not parse is reported and skipped rather than aborting the run: a
        // partial map of a large crate is more useful than no map.
        let ast = match syn::parse_file(&text) {
            Ok(ast) => ast,
            Err(err) => {
                krate.diagnostics.push(LoadDiagnostic {
                    path: rel,
                    message: format!("parse error: {err}"),
                });
                continue;
            }
        };

        let file_index = krate.files.len();
        krate.files.push(SourceFile { path, rel });
        let module_path = file_module_path(&krate.files[file_index].rel);
        let inherited_cfg = cfg_predicates(&ast.attrs);
        collect_items(
            &mut krate,
            file_index,
            &module_path,
            &inherited_cfg,
            &ast.items,
        );
    }

    Ok(krate)
}

/// Load only Rust modules reachable from the selected Cargo crate roots.
///
/// This deliberately excludes sibling tests, examples, benches, and inactive alternative source
/// trees unless the selected Cargo target names one of them as its root.
pub fn load_from_roots(root: &Path, crate_roots: &[PathBuf]) -> Result<Crate> {
    let mut krate = Crate::default();
    let mut visited = HashSet::new();
    let normalized_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    for crate_root in crate_roots {
        load_module_file(
            &mut krate,
            &normalized_root,
            crate_root,
            "",
            &[],
            &mut visited,
        )?;
    }
    Ok(krate)
}

fn load_module_file(
    krate: &mut Crate,
    scan_root: &Path,
    path: &Path,
    module_path: &str,
    inherited_cfg: &[String],
    visited: &mut HashSet<(PathBuf, String)>,
) -> Result<()> {
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !visited.insert((normalized, module_path.to_string())) {
        return Ok(());
    }

    let rel = path
        .strip_prefix(scan_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            krate.diagnostics.push(LoadDiagnostic {
                path: rel,
                message: format!("read error: {error}"),
            });
            return Ok(());
        }
    };
    let ast = match syn::parse_file(&text) {
        Ok(ast) => ast,
        Err(error) => {
            krate.diagnostics.push(LoadDiagnostic {
                path: rel,
                message: format!("parse error: {error}"),
            });
            return Ok(());
        }
    };

    let file_index = krate.files.len();
    krate.files.push(SourceFile {
        path: path.to_path_buf(),
        rel,
    });
    let file_cfg = combined_cfg(inherited_cfg, &ast.attrs);
    collect_items(krate, file_index, module_path, &file_cfg, &ast.items);

    let module_dir = module_directory(path);
    load_external_modules(
        krate,
        scan_root,
        path,
        &module_dir,
        module_path,
        &file_cfg,
        &ast.items,
        visited,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_external_modules(
    krate: &mut Crate,
    scan_root: &Path,
    declaring_file: &Path,
    module_dir: &Path,
    module_path: &str,
    inherited_cfg: &[String],
    items: &[syn::Item],
    visited: &mut HashSet<(PathBuf, String)>,
) -> Result<()> {
    for item in items {
        let syn::Item::Mod(module) = item else {
            continue;
        };
        let child_name = module.ident.to_string();
        let child_module = join_qualified(module_path, &child_name);
        let child_cfg = combined_cfg(inherited_cfg, &module.attrs);
        if let Some((_, inner)) = &module.content {
            load_external_modules(
                krate,
                scan_root,
                declaring_file,
                &module_dir.join(&child_name),
                &child_module,
                &child_cfg,
                inner,
                visited,
            )?;
            continue;
        }

        let explicit_path = module.attrs.iter().find_map(|attr| {
            if !attr.path().is_ident("path") {
                return None;
            }
            let syn::Meta::NameValue(value) = &attr.meta else {
                return None;
            };
            let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(value),
                ..
            }) = &value.value
            else {
                return None;
            };
            Some(value.value())
        });
        let candidates = if let Some(explicit) = explicit_path {
            vec![declaring_file
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join(explicit)]
        } else {
            vec![
                module_dir.join(format!("{child_name}.rs")),
                module_dir.join(&child_name).join("mod.rs"),
            ]
        };
        match candidates.iter().find(|candidate| candidate.is_file()) {
            Some(source) => {
                load_module_file(krate, scan_root, source, &child_module, &child_cfg, visited)?
            }
            None => krate.diagnostics.push(LoadDiagnostic {
                path: declaring_file
                    .strip_prefix(scan_root)
                    .unwrap_or(declaring_file)
                    .to_string_lossy()
                    .into_owned(),
                message: format!(
                    "module `{child_module}` was declared but no source file was found"
                ),
            }),
        }
    }
    Ok(())
}

fn module_directory(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    match path.file_name().and_then(|name| name.to_str()) {
        Some("lib.rs" | "main.rs" | "mod.rs") => parent.to_path_buf(),
        _ => parent.join(
            path.file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or_default(),
        ),
    }
}

fn collect_items(
    krate: &mut Crate,
    file: usize,
    module_path: &str,
    inherited_cfg: &[String],
    items: &[syn::Item],
) {
    // Imports are module-scoped regardless of source order. Collect them before definitions so
    // signature aliases such as `use candle_nn::VarBuilder as VB` resolve everywhere.
    for item in items {
        if let syn::Item::Use(item_use) = item {
            collect_use(krate, file, module_path, item_use, is_public(&item_use.vis));
        }
    }
    for item in items {
        match item {
            syn::Item::Struct(s) => {
                let def = struct_def(file, module_path, inherited_cfg, s);
                krate.structs.insert(def.name.clone(), def.clone());
                krate
                    .qualified_structs
                    .insert(def.qualified_name.clone(), def.clone());
                krate.all_structs.push(def);
            }
            syn::Item::Impl(imp) => collect_impl(krate, file, module_path, inherited_cfg, imp),
            syn::Item::Fn(f) => {
                let predicates = combined_cfg(inherited_cfg, &f.attrs);
                let mut func = impl_fn(
                    file,
                    module_path,
                    String::new(),
                    String::new(),
                    None,
                    &f.vis,
                    &f.sig,
                    &f.block,
                    predicates,
                );
                augment_builder_aliases(krate, module_path, &mut func);
                krate.functions.insert(func.fn_name.clone(), func.clone());
                krate
                    .qualified_functions
                    .insert(func.qualified_name.clone(), func.clone());
                krate.all_functions.push(func);
            }
            syn::Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    let child_module = join_qualified(module_path, &m.ident.to_string());
                    let child_cfg = combined_cfg(inherited_cfg, &m.attrs);
                    collect_items(krate, file, &child_module, &child_cfg, inner);
                }
            }
            syn::Item::Use(_) => {}
            _ => {}
        }
    }
}

fn collect_impl(
    krate: &mut Crate,
    file: usize,
    module_path: &str,
    inherited_cfg: &[String],
    imp: &syn::ItemImpl,
) {
    let Some(type_name) = type_base_name(&imp.self_ty) else {
        return;
    };
    let trait_name = imp.trait_.as_ref().map(|(_, path, _)| type_text(path));
    let qualified_type_name = qualify_type(module_path, &imp.self_ty)
        .unwrap_or_else(|| join_qualified(module_path, &type_name));
    let impl_cfg = combined_cfg(inherited_cfg, &imp.attrs);
    for item in &imp.items {
        if let syn::ImplItem::Fn(f) = item {
            let predicates = combined_cfg(&impl_cfg, &f.attrs);
            let mut func = impl_fn(
                file,
                module_path,
                type_name.clone(),
                qualified_type_name.clone(),
                trait_name.clone(),
                &f.vis,
                &f.sig,
                &f.block,
                predicates,
            );
            augment_builder_aliases(krate, module_path, &mut func);
            krate
                .methods
                .insert((type_name.clone(), func.fn_name.clone()), func.clone());
            krate.qualified_methods.insert(
                (qualified_type_name.clone(), func.fn_name.clone()),
                func.clone(),
            );
            krate.all_methods.push(func);
        }
    }
}

fn augment_builder_aliases(krate: &Crate, module_path: &str, function: &mut ImplFn) {
    for (index, type_text) in function.param_types.iter().enumerate() {
        if function.vb_params.contains(&index) {
            continue;
        }
        let Some(builder_alias) = syn::parse_str::<syn::Type>(type_text)
            .ok()
            .is_some_and(|ty| contains_resolved_builder_type(&ty, krate, module_path))
            .then_some(index)
        else {
            continue;
        };
        function.vb_params.push(builder_alias);
    }
    function.vb_params.sort_unstable();
    function.vb_params.dedup();
}

fn contains_resolved_builder_type(ty: &syn::Type, krate: &Crate, module_path: &str) -> bool {
    match ty {
        syn::Type::Path(path) => path.path.segments.iter().any(|segment| {
            let source = vec![segment.ident.to_string()];
            let resolved = krate.resolve_import_path(module_path, &source);
            matches!(
                resolved.last().map(String::as_str),
                Some("VarBuilder" | "VarBuilderArgs")
            ) || match &segment.arguments {
                syn::PathArguments::AngleBracketed(arguments) => {
                    arguments.args.iter().any(|argument| {
                        matches!(
                            argument,
                            syn::GenericArgument::Type(inner)
                                if contains_resolved_builder_type(inner, krate, module_path)
                        )
                    })
                }
                syn::PathArguments::Parenthesized(arguments) => arguments
                    .inputs
                    .iter()
                    .any(|inner| contains_resolved_builder_type(inner, krate, module_path)),
                syn::PathArguments::None => false,
            }
        }),
        syn::Type::Reference(reference) => {
            contains_resolved_builder_type(&reference.elem, krate, module_path)
        }
        syn::Type::Paren(paren) => contains_resolved_builder_type(&paren.elem, krate, module_path),
        syn::Type::Group(group) => contains_resolved_builder_type(&group.elem, krate, module_path),
        syn::Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .any(|inner| contains_resolved_builder_type(inner, krate, module_path)),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn impl_fn(
    file: usize,
    module_path: &str,
    type_name: String,
    qualified_type_name: String,
    trait_name: Option<String>,
    visibility: &syn::Visibility,
    sig: &syn::Signature,
    block: &syn::Block,
    cfg_predicates: Vec<String>,
) -> ImplFn {
    let mut params = Vec::new();
    let mut param_types = Vec::new();
    let mut vb_params = Vec::new();

    for (index, input) in sig.inputs.iter().enumerate() {
        match input {
            syn::FnArg::Receiver(receiver) => {
                params.push("self".to_string());
                param_types.push(type_text(receiver));
            }
            syn::FnArg::Typed(pat) => {
                let name = match &*pat.pat {
                    syn::Pat::Ident(id) => id.ident.to_string(),
                    other => type_text(other),
                };
                // Matches `VarBuilder`, `Option<VarBuilder>` and references without accepting
                // unrelated types such as `MyVarBuilderConfig`.
                if contains_named_type(&pat.ty, "VarBuilder")
                    || contains_named_type(&pat.ty, "VarBuilderArgs")
                {
                    vb_params.push(index);
                }
                params.push(name);
                param_types.push(type_text(&pat.ty));
            }
        }
    }

    let fn_name = sig.ident.to_string();
    let qualified_name = if qualified_type_name.is_empty() {
        join_qualified(module_path, &fn_name)
    } else {
        join_qualified(&qualified_type_name, &fn_name)
    };
    ImplFn {
        type_name,
        fn_name,
        trait_name,
        module_path: module_path.to_string(),
        qualified_type_name,
        qualified_name,
        visibility: visibility_text(visibility),
        cfg_predicates,
        params,
        param_types,
        return_type: match &sig.output {
            syn::ReturnType::Default => "()".to_string(),
            syn::ReturnType::Type(_, ty) => type_text(ty),
        },
        vb_params,
        block: block.clone(),
        span: span_of(file, sig.ident.span()),
    }
}

fn struct_def(
    file: usize,
    module_path: &str,
    inherited_cfg: &[String],
    s: &syn::ItemStruct,
) -> StructDef {
    let mut fields = Vec::new();
    if let syn::Fields::Named(named) = &s.fields {
        for f in &named.named {
            let Some(ident) = &f.ident else { continue };
            fields.push(FieldDef {
                name: ident.to_string(),
                ty: type_ref(&f.ty),
                span: span_of(file, ident.span()),
            });
        }
    }
    StructDef {
        name: s.ident.to_string(),
        module_path: module_path.to_string(),
        qualified_name: join_qualified(module_path, &s.ident.to_string()),
        visibility: visibility_text(&s.vis),
        cfg_predicates: combined_cfg(inherited_cfg, &s.attrs),
        fields,
        span: span_of(file, s.ident.span()),
    }
}

fn combined_cfg(inherited: &[String], attrs: &[syn::Attribute]) -> Vec<String> {
    let mut predicates = inherited.to_vec();
    predicates.extend(cfg_predicates(attrs));
    predicates.sort();
    predicates.dedup();
    predicates
}

fn cfg_predicates(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter_map(|attribute| match &attribute.meta {
            syn::Meta::List(list) if list.path.is_ident("cfg") => Some(list.tokens.to_string()),
            _ => None,
        })
        .collect()
}

/// Derive a Rust module path from a path relative to the scan root.
///
/// `foo.rs` and `foo/mod.rs` both map to `foo`; `lib.rs` and `main.rs` map to the crate root.
/// When a whole conventional Cargo crate is scanned, the leading `src/` directory is not a
/// module.
fn file_module_path(rel: &str) -> String {
    let normalized = rel.replace('\\', "/");
    let mut parts: Vec<&str> = normalized
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    let Some(file) = parts.pop() else {
        return String::new();
    };
    if parts.first() == Some(&"src") {
        parts.remove(0);
    }
    let stem = file.strip_suffix(".rs").unwrap_or(file);
    if !matches!(stem, "mod" | "lib" | "main") {
        parts.push(stem);
    }
    parts.join("::")
}

fn join_qualified(module_path: &str, leaf: &str) -> String {
    if module_path.is_empty() {
        leaf.to_string()
    } else if leaf.is_empty() {
        module_path.to_string()
    } else {
        format!("{module_path}::{leaf}")
    }
}

fn visibility_text(vis: &syn::Visibility) -> String {
    type_text(vis).replace("pub (", "pub(").replace(" )", ")")
}

fn is_public(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

fn qualify_type(module_path: &str, ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return match ty {
            syn::Type::Reference(reference) => qualify_type(module_path, &reference.elem),
            syn::Type::Paren(paren) => qualify_type(module_path, &paren.elem),
            _ => None,
        };
    };
    let parts: Vec<String> = path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    qualify_path_parts(module_path, &parts)
}

fn qualify_path_parts(module_path: &str, parts: &[String]) -> Option<String> {
    let (first, rest) = parts.split_first()?;
    match first.as_str() {
        "crate" => Some(rest.join("::")),
        "self" => Some(join_qualified(module_path, &rest.join("::"))),
        "super" => {
            let mut base: Vec<&str> = module_path
                .split("::")
                .filter(|part| !part.is_empty())
                .collect();
            let mut remaining = rest;
            while remaining.first().map(String::as_str) == Some("super") {
                base.pop();
                remaining = &remaining[1..];
            }
            base.pop();
            let suffix = remaining.join("::");
            Some(join_qualified(&base.join("::"), &suffix))
        }
        _ if matches!(
            first.as_str(),
            "candle"
                | "candle_core"
                | "candle_nn"
                | "candle_transformers"
                | "std"
                | "core"
                | "alloc"
        ) =>
        {
            Some(parts.join("::"))
        }
        _ if parts.len() == 1 => Some(join_qualified(module_path, first)),
        _ => Some(parts.join("::")),
    }
}

fn collect_use(
    krate: &mut Crate,
    file: usize,
    module_path: &str,
    item_use: &syn::ItemUse,
    public: bool,
) {
    let mut leaves = Vec::new();
    flatten_use_tree(Vec::new(), &item_use.tree, &mut leaves);
    for (path, alias, span) in leaves {
        let Some(source_name) = path.last() else {
            continue;
        };
        let name = alias.unwrap_or_else(|| source_name.clone());
        let target = qualify_path_parts(module_path, &path).unwrap_or_else(|| path.join("::"));
        krate.imports.push(ImportBinding {
            module_path: module_path.to_string(),
            alias: name.clone(),
            target: target.clone(),
        });
        if public {
            krate.public_reexports.push(PublicReexport {
                qualified_name: join_qualified(module_path, &name),
                name,
                target,
                module_path: module_path.to_string(),
                span: span_of(file, span),
            });
        }
    }
}

fn flatten_use_tree(
    prefix: Vec<String>,
    tree: &syn::UseTree,
    leaves: &mut Vec<(Vec<String>, Option<String>, proc_macro2::Span)>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix;
            next.push(path.ident.to_string());
            flatten_use_tree(next, &path.tree, leaves);
        }
        syn::UseTree::Name(name) => {
            if name.ident == "self" {
                if !prefix.is_empty() {
                    leaves.push((prefix, None, name.ident.span()));
                }
            } else {
                let mut path = prefix;
                path.push(name.ident.to_string());
                leaves.push((path, None, name.ident.span()));
            }
        }
        syn::UseTree::Rename(rename) => {
            let mut path = prefix;
            if rename.ident != "self" {
                path.push(rename.ident.to_string());
            }
            leaves.push((path, Some(rename.rename.to_string()), rename.rename.span()));
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(prefix.clone(), item, leaves);
            }
        }
        // A glob has no statically enumerable leaf.
        syn::UseTree::Glob(_) => {}
    }
}

/// Peel `Vec<T>`, `Option<T>` and `Box<T>` down to the named type inside.
pub fn type_ref(ty: &syn::Type) -> TypeRef {
    let text = type_text(ty);
    let (base, container) = peel(ty);
    TypeRef {
        text,
        base,
        container,
    }
}

fn peel(ty: &syn::Type) -> (String, Container) {
    let syn::Type::Path(p) = ty else {
        return (type_base_name(ty).unwrap_or_default(), Container::Plain);
    };
    let Some(last) = p.path.segments.last() else {
        return (String::new(), Container::Plain);
    };
    let container = match last.ident.to_string().as_str() {
        "Vec" => Container::Vec,
        "Option" => Container::Option,
        "Box" => Container::Box,
        _ => Container::Plain,
    };
    if container == Container::Plain {
        return (last.ident.to_string(), Container::Plain);
    }
    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
        for arg in &args.args {
            if let syn::GenericArgument::Type(inner) = arg {
                let (base, _) = peel(inner);
                return (base, container);
            }
        }
    }
    (last.ident.to_string(), container)
}

/// Last path segment of a type, e.g. `nn::Linear` -> `Linear`.
pub fn type_base_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => type_base_name(&r.elem),
        syn::Type::Paren(p) => type_base_name(&p.elem),
        _ => None,
    }
}

fn contains_named_type(ty: &syn::Type, wanted: &str) -> bool {
    match ty {
        syn::Type::Path(path) => path.path.segments.iter().any(|segment| {
            segment.ident == wanted
                || match &segment.arguments {
                    syn::PathArguments::AngleBracketed(arguments) => {
                        arguments.args.iter().any(|argument| match argument {
                            syn::GenericArgument::Type(inner) => contains_named_type(inner, wanted),
                            _ => false,
                        })
                    }
                    syn::PathArguments::Parenthesized(arguments) => {
                        arguments
                            .inputs
                            .iter()
                            .any(|inner| contains_named_type(inner, wanted))
                            || match &arguments.output {
                                syn::ReturnType::Default => false,
                                syn::ReturnType::Type(_, output) => {
                                    contains_named_type(output, wanted)
                                }
                            }
                    }
                    syn::PathArguments::None => false,
                }
        }),
        syn::Type::Reference(reference) => contains_named_type(&reference.elem, wanted),
        syn::Type::Paren(paren) => contains_named_type(&paren.elem, wanted),
        syn::Type::Group(group) => contains_named_type(&group.elem, wanted),
        syn::Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .any(|inner| contains_named_type(inner, wanted)),
        _ => false,
    }
}

pub fn type_text<T: quote::ToTokens>(t: &T) -> String {
    let mut text = quote::quote!(#t).to_string();
    // `quote` inserts spaces around punctuation; tighten the common cases so paths read
    // naturally in output.
    for (from, to) in [(" :: ", "::"), (" < ", "<"), (" > ", ">"), (" , ", ", ")] {
        text = text.replace(from, to);
    }
    text
}

pub fn span_of(file: usize, span: proc_macro2::Span) -> SrcSpan {
    let start = span.start();
    SrcSpan {
        file,
        line: start.line,
        col: start.column,
    }
}
