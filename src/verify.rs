//! Cross-check the static parameter set against a real checkpoint.
//!
//! The checkpoint is *evidence*, not ground truth. A tensor missing from a safetensors file may
//! mean the analyzer invented it, or that the checkpoint is stale, or that the parameter is
//! genuinely conditional and this configuration did not create it. The report states what was
//! observed and leaves the conclusion to the reader.
//!
//! Reading the header directly rather than depending on `safetensors` keeps this decoupled from
//! candle's version: the layout is a little-endian `u64` byte count followed by that many bytes
//! of JSON.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::ir::Key;
use crate::model_ir::{Confidence, ModelIr, Parameter, ParameterRole};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
}

pub type Header = BTreeMap<String, TensorInfo>;
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Read tensor names, shapes and dtypes from a safetensors file without loading any data.
///
/// Validates header bounds, per-tensor `shape`/`dtype`/`data_offsets`, element-byte product
/// against declared offsets, and that offsets fall within the file's data region.
pub fn read_header(path: &Path) -> Result<Header> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len();

    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("{} is too short to be a safetensors file", path.display()))?;
    let header_len_u64 = u64::from_le_bytes(prefix);
    if header_len_u64 > MAX_HEADER_BYTES {
        bail!(
            "{} declares a {} byte header, exceeding the {} byte safety limit",
            path.display(),
            header_len_u64,
            MAX_HEADER_BYTES
        );
    }
    let header_end = 8u64
        .checked_add(header_len_u64)
        .filter(|end| *end <= file_len)
        .with_context(|| format!("{} declares a header longer than the file", path.display()))?;
    let header_len = usize::try_from(header_len_u64)
        .with_context(|| format!("safetensors header of {} is too large", path.display()))?;

    let mut bytes = vec![0u8; header_len];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading safetensors header of {}", path.display()))?;

    debug_assert_eq!(header_end, 8 + header_len as u64);
    let json: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing safetensors header of {}", path.display()))?;
    let obj = json
        .as_object()
        .with_context(|| format!("safetensors header of {} is not an object", path.display()))?;

    let data_len = file_len - header_end;
    let mut header = Header::new();
    let mut occupied: Vec<(u64, u64, String)> = Vec::new();

    for (name, value) in obj {
        if name == "__metadata__" {
            continue;
        }
        let info = parse_tensor_entry(name, value, data_len)
            .with_context(|| format!("tensor `{name}` in {}", path.display()))?;
        occupied.push((info.offset_start, info.offset_end, name.clone()));
        header.insert(
            name.clone(),
            TensorInfo {
                shape: info.shape,
                dtype: info.dtype,
            },
        );
    }

    occupied.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    for window in occupied.windows(2) {
        let (a_start, a_end, a_name) = &window[0];
        let (b_start, _b_end, b_name) = &window[1];
        if *b_start < *a_end {
            bail!(
                "overlapping data_offsets in {}: `{}` [{a_start},{a_end}) overlaps `{b_name}` starting at {b_start}",
                path.display(),
                a_name
            );
        }
    }

    Ok(header)
}

struct ParsedTensor {
    shape: Vec<usize>,
    dtype: String,
    offset_start: u64,
    offset_end: u64,
}

fn parse_tensor_entry(
    name: &str,
    value: &serde_json::Value,
    data_len: u64,
) -> Result<ParsedTensor> {
    let obj = value
        .as_object()
        .with_context(|| format!("entry `{name}` is not an object"))?;

    let shape_value = obj
        .get("shape")
        .with_context(|| format!("tensor `{name}` is missing shape"))?;
    let shape_arr = shape_value
        .as_array()
        .with_context(|| format!("tensor `{name}` shape must be an array"))?;
    let mut shape = Vec::with_capacity(shape_arr.len());
    for (i, dim) in shape_arr.iter().enumerate() {
        let n = dim.as_u64().with_context(|| {
            format!("tensor `{name}` shape[{i}] must be a non-negative integer")
        })?;
        let n = usize::try_from(n)
            .with_context(|| format!("tensor `{name}` shape[{i}] = {n} is too large"))?;
        shape.push(n);
    }

    let dtype = obj
        .get("dtype")
        .and_then(|d| d.as_str())
        .with_context(|| format!("tensor `{name}` is missing a string dtype"))?
        .to_string();
    let element_size = dtype_nbytes(&dtype)
        .with_context(|| format!("tensor `{name}` has unsupported or unknown dtype `{dtype}`"))?;

    let offsets_value = obj
        .get("data_offsets")
        .with_context(|| format!("tensor `{name}` is missing data_offsets"))?;
    let offsets = offsets_value
        .as_array()
        .with_context(|| format!("tensor `{name}` data_offsets must be an array"))?;
    if offsets.len() != 2 {
        bail!("tensor `{name}` data_offsets must have exactly two entries");
    }
    let offset_start = offsets[0].as_u64().with_context(|| {
        format!("tensor `{name}` data_offsets[0] must be a non-negative integer")
    })?;
    let offset_end = offsets[1].as_u64().with_context(|| {
        format!("tensor `{name}` data_offsets[1] must be a non-negative integer")
    })?;
    if offset_end < offset_start {
        bail!("tensor `{name}` data_offsets end {offset_end} is before start {offset_start}");
    }
    if offset_end > data_len {
        bail!(
            "tensor `{name}` data_offsets end {offset_end} exceeds data region length {data_len}"
        );
    }

    let declared = offset_end - offset_start;
    let expected = tensor_nbytes(&shape, element_size)
        .with_context(|| format!("tensor `{name}` byte size overflows"))?;
    if declared != expected {
        bail!(
            "tensor `{name}` data_offsets span {declared} bytes but shape {:?} dtype {dtype} requires {expected}",
            shape
        );
    }

    Ok(ParsedTensor {
        shape,
        dtype,
        offset_start,
        offset_end,
    })
}

fn dtype_nbytes(dtype: &str) -> Result<u64> {
    let n = match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1u64,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => bail!("unsupported safetensors dtype `{dtype}`"),
    };
    Ok(n)
}

fn tensor_nbytes(shape: &[usize], element_size: u64) -> Result<u64> {
    let mut n = element_size;
    for &dim in shape {
        let dim = u64::try_from(dim).context("shape dimension does not fit u64")?;
        n = n
            .checked_mul(dim)
            .context("shape × dtype byte size overflows u64")?;
    }
    Ok(n)
}

#[derive(Debug, Default, Serialize)]
pub struct VerifyReport {
    /// `VarBuilder` root this checkpoint was compared against.
    pub root: String,
    /// Analyzer parameters matched to at least one tensor.
    pub matched: usize,
    /// Parameters the analyzer marked *certain* that the checkpoint does not contain. These are
    /// the actionable ones: either the analyzer is wrong or the checkpoint is.
    pub missing_certain: Vec<String>,
    /// Parameters the analyzer already flagged conditional that the checkpoint does not
    /// contain. Expected — this is the analyzer and the checkpoint agreeing.
    pub missing_conditional: Vec<String>,
    /// Checkpoint tensors no analyzer parameter claims.
    pub unclaimed: Vec<String>,
    /// Parameters belonging to a different builder root, not comparable against this file.
    pub skipped_other_root: usize,
    pub checkpoint_tensors: usize,
}

/// Match every parameter under `root` against the header, annotating parameters in place.
///
/// Scoping to one builder root matters: a model may draw from several `VarBuilder`s — frozen
/// mmapped base weights alongside a trainable `VarMap` — and each has its own checkpoint file.
pub fn verify_model(model: &mut ModelIr, header: &Header, root: &str) -> VerifyReport {
    let mut report = VerifyReport {
        checkpoint_tensors: header.len(),
        root: root.to_string(),
        ..Default::default()
    };
    let mut claimed: Vec<bool> = vec![false; header.len()];
    let names: Vec<&String> = header.keys().collect();

    for parameter in &mut model.parameters {
        if parameter.builder_root != root {
            report.skipped_other_root += 1;
            continue;
        }
        let key = Key::from_dotted(&parameter.key);

        if key.is_template() {
            let hits: Vec<usize> = names
                .iter()
                .enumerate()
                .filter(|(_, name)| key.matches(name))
                .map(|(i, _)| i)
                .collect();
            if hits.is_empty() {
                record_missing(&mut report, parameter, parameter.key.clone());
                parameter.checkpoint_shape = None;
                parameter.checkpoint_dtype = None;
            } else {
                for hit in &hits {
                    claimed[*hit] = true;
                }
                let info = &header[names[hits[0]]];
                parameter.checkpoint_shape = Some(info.shape.clone());
                parameter.checkpoint_dtype = Some(info.dtype.clone());
                report.matched += 1;
            }
            continue;
        }

        let exact = parameter.key.clone();
        match names.iter().position(|n| **n == exact) {
            Some(hit) => {
                claimed[hit] = true;
                let info = &header[names[hit]];
                parameter.checkpoint_shape = Some(info.shape.clone());
                parameter.checkpoint_dtype = Some(info.dtype.clone());
                report.matched += 1;
            }
            None => {
                record_missing(&mut report, parameter, exact);
                parameter.checkpoint_shape = None;
                parameter.checkpoint_dtype = None;
            }
        }
    }

    for (index, claimed) in claimed.iter().enumerate() {
        if !claimed {
            report.unclaimed.push(names[index].to_string());
        }
    }
    report.missing_certain.sort();
    report.missing_conditional.sort();
    report.unclaimed.sort();
    report
}

fn record_missing(report: &mut VerifyReport, parameter: &Parameter, key: String) {
    if missing_is_certain(parameter) {
        report.missing_certain.push(key);
    } else {
        report.missing_conditional.push(key);
    }
}

fn missing_is_certain(parameter: &Parameter) -> bool {
    match parameter.role {
        ParameterRole::Conditional | ParameterRole::Unknown | ParameterRole::Excluded => false,
        ParameterRole::Optimized | ParameterRole::Frozen | ParameterRole::RunningState => parameter
            .evidence
            .iter()
            .any(|evidence| evidence.confidence == Confidence::Proven),
    }
}
