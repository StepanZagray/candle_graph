//! Atomic publication of self-contained, content-addressed evidence bundles.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evidence::build_evidence;

pub const SCHEMA: &str = "candle-graph/bundle/1";
pub const VERIFICATION_SCHEMA: &str = "candle-graph/bundle-verification/1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub schema: String,
    pub run_id: String,
    pub files: Vec<BundleFile>,
}

/// Deterministic proof that every file in one evidence bundle matched its manifest when checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleVerificationReceipt {
    pub schema: String,
    pub bundle_schema: String,
    pub run_id: String,
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub files_verified: usize,
    pub bytes_verified: u64,
}

/// Deeply verify a published bundle. Declared files must be regular files with matching sizes and
/// digests, and every regular file below the root must be declared except the manifest itself.
pub fn verify_bundle(root: &Path) -> Result<BundleVerificationReceipt> {
    if !root.is_dir() {
        bail!("bundle directory does not exist: {}", root.display());
    }
    let manifest_path = root.join("bundle.json");
    let manifest_metadata = fs::symlink_metadata(&manifest_path)
        .with_context(|| format!("read bundle manifest {}", manifest_path.display()))?;
    if !manifest_metadata.file_type().is_file() {
        bail!(
            "bundle manifest is not a regular file: {}",
            manifest_path.display()
        );
    }
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read bundle manifest {}", manifest_path.display()))?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse bundle manifest {}", manifest_path.display()))?;
    if manifest.schema != SCHEMA {
        bail!(
            "unsupported bundle schema {:?}; expected {SCHEMA:?}",
            manifest.schema
        );
    }

    let mut declared = std::collections::BTreeMap::new();
    for file in &manifest.files {
        if !is_safe_relative_path(&file.path) || file.path == "bundle.json" {
            bail!("unsafe or reserved bundle path {:?}", file.path);
        }
        if declared.insert(file.path.as_str(), file).is_some() {
            bail!("bundle manifest declares {:?} more than once", file.path);
        }
        if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("bundle manifest has an invalid SHA-256 for {:?}", file.path);
        }
    }

    let mut observed = Vec::new();
    collect_bundle_files(root, root, &mut observed)?;
    observed.sort();
    for relative in &observed {
        if relative == Path::new("bundle.json") {
            continue;
        }
        let normalized = relative.to_string_lossy().replace('\\', "/");
        if !declared.contains_key(normalized.as_str()) {
            bail!("undeclared regular file in bundle: {normalized}");
        }
    }
    if observed
        .iter()
        .filter(|path| path.as_path() != Path::new("bundle.json"))
        .count()
        != declared.len()
    {
        let missing = declared
            .keys()
            .find(|path| !observed.iter().any(|item| item == Path::new(path)))
            .copied()
            .unwrap_or("<unknown>");
        bail!("declared bundle file is missing: {missing}");
    }

    let mut bytes_verified = 0u64;
    for (relative, expected) in &declared {
        let path = root.join(relative);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("read declared bundle file {}", path.display()))?;
        if !metadata.file_type().is_file() {
            bail!("declared bundle path is not a regular file: {relative}");
        }
        let actual = describe_file(root, &path)?;
        if actual.bytes != expected.bytes {
            bail!(
                "bundle file {relative:?} size mismatch: expected {}, observed {}",
                expected.bytes,
                actual.bytes
            );
        }
        if !actual.sha256.eq_ignore_ascii_case(&expected.sha256) {
            bail!("bundle file {relative:?} SHA-256 mismatch");
        }
        bytes_verified = bytes_verified
            .checked_add(actual.bytes)
            .context("verified bundle byte count overflowed u64")?;
    }

    Ok(BundleVerificationReceipt {
        schema: VERIFICATION_SCHEMA.into(),
        bundle_schema: manifest.schema,
        run_id: manifest.run_id,
        manifest_path: "bundle.json".into(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        files_verified: declared.len(),
        bytes_verified,
    })
}

/// Publish all evidence into a new directory. The final path appears only after every artifact,
/// hash, and manifest has been written successfully.
pub fn publish_bundle(
    destination: &Path,
    trace: &Path,
    nsight_dir: Option<&Path>,
) -> Result<BundleManifest> {
    if destination.exists() {
        bail!(
            "bundle destination already exists: {}",
            destination.display()
        );
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create bundle parent {}", parent.display()))?;
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .context("bundle destination needs a file name")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&temporary)
        .with_context(|| format!("create temporary bundle {}", temporary.display()))?;

    let result = write_bundle(&temporary, trace, nsight_dir).and_then(|manifest| {
        verify_bundle(&temporary).context("verify staged evidence bundle")?;
        sync_directory(&temporary)?;
        fs::rename(&temporary, destination)
            .with_context(|| format!("atomically publish bundle {}", destination.display()))?;
        sync_directory(parent)?;
        Ok(manifest)
    });
    if result.is_err() && temporary.exists() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn write_bundle(root: &Path, trace: &Path, nsight_dir: Option<&Path>) -> Result<BundleManifest> {
    let mut files = Vec::new();
    copy_and_record(trace, root, Path::new("trace.jsonl"), &mut files)?;

    if let Some(nsight) = nsight_dir {
        fs::create_dir(root.join("nsight"))?;
        let mut sources = fs::read_dir(nsight)
            .with_context(|| format!("read Nsight directory {}", nsight.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        sources.sort_by_key(|entry| entry.file_name());
        for entry in sources {
            let source = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("inspect Nsight artifact {}", source.display()))?;
            if file_type.is_symlink() {
                bail!(
                    "symbolic links are not allowed in Nsight inputs: {}",
                    source.display()
                );
            }
            if file_type.is_dir() {
                bail!(
                    "directories are not allowed in Nsight inputs: {}",
                    source.display()
                );
            }
            if !file_type.is_file() {
                bail!(
                    "special files are not allowed in Nsight inputs: {}",
                    source.display()
                );
            }
            let file_name = source
                .file_name()
                .context("Nsight artifact missing file name")?;
            copy_and_record(
                &source,
                root,
                &Path::new("nsight").join(file_name),
                &mut files,
            )?;
        }
    }
    let staged_trace = root.join("trace.jsonl");
    let staged_nsight = nsight_dir.map(|_| root.join("nsight"));
    let evidence = build_evidence(&staged_trace, staged_nsight.as_deref())?;
    write_and_record(
        root,
        Path::new("evidence.json"),
        &(serde_json::to_string_pretty(&evidence)? + "\n"),
        &mut files,
    )?;
    write_and_record(
        root,
        Path::new("report.md"),
        &evidence.markdown(),
        &mut files,
    )?;
    #[cfg(feature = "visualizer")]
    write_and_record(
        root,
        Path::new("viewer.html"),
        &crate::viewer::render_evidence_html(&evidence),
        &mut files,
    )?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let manifest = BundleManifest {
        schema: SCHEMA.into(),
        run_id: evidence.provenance.run_id.clone(),
        files,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)? + "\n";
    let path = root.join("bundle.json");
    let mut file = File::create(&path)?;
    file.write_all(manifest_json.as_bytes())?;
    file.sync_all()?;
    Ok(manifest)
}

fn copy_and_record(
    source: &Path,
    root: &Path,
    relative: &Path,
    files: &mut Vec<BundleFile>,
) -> Result<()> {
    let target = root.join(relative);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, &target).with_context(|| format!("copy {} into bundle", source.display()))?;
    File::open(&target)?.sync_all()?;
    files.push(describe_file(root, &target)?);
    Ok(())
}

fn write_and_record(
    root: &Path,
    relative: &Path,
    contents: &str,
    files: &mut Vec<BundleFile>,
) -> Result<()> {
    let path = root.join(relative);
    let mut file = File::create(&path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    files.push(describe_file(root, &path)?);
    Ok(())
}

fn describe_file(root: &Path, path: &Path) -> Result<BundleFile> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok(BundleFile {
        path: path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/"),
        bytes,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn collect_bundle_files(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("read bundle directory {}", directory.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            bail!(
                "symbolic links are not allowed in bundles: {}",
                path.display()
            );
        }
        if file_type.is_dir() {
            collect_bundle_files(root, &path, files)?;
        } else if file_type.is_file() {
            files.push(path.strip_prefix(root)?.to_path_buf());
        } else {
            bail!(
                "special files are not allowed in bundles: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn is_safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !value.contains('\\')
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?
        .sync_all()
        .with_context(|| format!("sync directory {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CaptureContract;
    use crate::trace::{
        write_jsonl, RunOutcome, SpanKind, SpanRecord, TerminalEvent, TimingMode, TraceDocument,
        TraceRunMeta, SCHEMA as TRACE_SCHEMA,
    };

    #[test]
    fn publishes_complete_content_addressed_directory() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "candle-graph-bundle-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let trace = root.join("input.jsonl");
        let nsight = root.join("nsight-input");
        let destination = root.join("bundle");
        fs::create_dir(&nsight).unwrap();
        fs::write(nsight.join("capture.nsys-rep"), b"retained raw capture").unwrap();
        let document = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "bundle-run".into(),
                correlation_id: "bundle/run".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "demo".into(),
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
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 10,
                reason: None,
            },
        };
        write_jsonl(&trace, &document.to_events()).unwrap();
        let manifest = publish_bundle(&destination, &trace, Some(&nsight)).unwrap();
        assert!(destination.join("bundle.json").is_file());
        assert!(destination.join("evidence.json").is_file());
        assert!(destination.join("report.md").is_file());
        assert_eq!(
            fs::read(destination.join("nsight/capture.nsys-rep")).unwrap(),
            b"retained raw capture"
        );
        assert!(manifest
            .files
            .iter()
            .any(|file| file.path == "nsight/capture.nsys-rep"));
        assert!(manifest.files.iter().all(|file| file.sha256.len() == 64));
        let receipt = verify_bundle(&destination).unwrap();
        assert_eq!(receipt.run_id, "bundle-run");
        assert_eq!(receipt.manifest_sha256.len(), 64);
        assert_eq!(receipt.files_verified, manifest.files.len());
        let bundled_evidence: crate::evidence::EvidencePacket =
            serde_json::from_slice(&fs::read(destination.join("evidence.json")).unwrap()).unwrap();
        assert_eq!(bundled_evidence.provenance.run_id, manifest.run_id);
        assert!(publish_bundle(&destination, &trace, Some(&nsight)).is_err());

        fs::write(
            destination.join("nsight/capture.nsys-rep"),
            b"tampered raw capture",
        )
        .unwrap();
        assert!(verify_bundle(&destination)
            .unwrap_err()
            .to_string()
            .contains("mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publication_rejects_non_regular_nsight_inputs() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "candle-graph-nsight-input-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let trace = root.join("input.jsonl");
        let document = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "nsight-input-run".into(),
                correlation_id: "nsight/input/run".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "demo".into(),
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
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 10,
                reason: None,
            },
        };
        write_jsonl(&trace, &document.to_events()).unwrap();

        let directory_input = root.join("directory-input");
        fs::create_dir_all(directory_input.join("nested")).unwrap();
        let error = publish_bundle(
            &root.join("directory-bundle"),
            &trace,
            Some(&directory_input),
        )
        .unwrap_err();
        assert!(error.to_string().contains("directories are not allowed"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            use std::os::unix::net::UnixListener;

            let symlink_input = root.join("symlink-input");
            fs::create_dir(&symlink_input).unwrap();
            fs::write(root.join("regular-input"), b"input").unwrap();
            symlink(root.join("regular-input"), symlink_input.join("linked")).unwrap();
            let error = publish_bundle(&root.join("symlink-bundle"), &trace, Some(&symlink_input))
                .unwrap_err();
            assert!(error.to_string().contains("symbolic links are not allowed"));

            let special_input = root.join("special-input");
            fs::create_dir(&special_input).unwrap();
            let _socket = UnixListener::bind(special_input.join("socket")).unwrap();
            let error = publish_bundle(&root.join("special-bundle"), &trace, Some(&special_input))
                .unwrap_err();
            assert!(error.to_string().contains("special files are not allowed"));
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verification_rejects_tampered_deleted_and_injected_files() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "candle-graph-bundle-tamper-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let trace = root.join("input.jsonl");
        let destination = root.join("bundle");
        let document = TraceDocument {
            schema: TRACE_SCHEMA.into(),
            run: TraceRunMeta {
                run_id: "tamper-run".into(),
                correlation_id: "tamper/run".into(),
                entrypoint: "demo".into(),
                phase: crate::ExecutionPhase::Infer,
                timestamp: "2026-08-19T00:00:00Z".into(),
                capture_step: 1,
                warmup_steps: 0,
                device: "cpu".into(),
                measured_region_device_synchronized: false,
                timing_mode: TimingMode::Host,
                capture_contract: CaptureContract::default(),
                comparison_identity: None,
                tags: Default::default(),
                candle_version: None,
            },
            spans: vec![SpanRecord {
                id: "root".into(),
                parent_id: None,
                name: "demo".into(),
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
            gradients: vec![],
            edges: vec![],
            terminal: TerminalEvent {
                outcome: RunOutcome::Complete,
                timestamp_ns: 10,
                reason: None,
            },
        };
        write_jsonl(&trace, &document.to_events()).unwrap();
        publish_bundle(&destination, &trace, None).unwrap();

        let evidence_path = destination.join("evidence.json");
        let original_evidence = fs::read(&evidence_path).unwrap();
        fs::write(&evidence_path, b"tampered").unwrap();
        assert!(verify_bundle(&destination)
            .unwrap_err()
            .to_string()
            .contains("mismatch"));
        fs::write(&evidence_path, original_evidence).unwrap();

        let report_path = destination.join("report.md");
        let original_report = fs::read(&report_path).unwrap();
        fs::remove_file(&report_path).unwrap();
        assert!(verify_bundle(&destination)
            .unwrap_err()
            .to_string()
            .contains("missing"));
        fs::write(&report_path, original_report).unwrap();

        fs::write(destination.join("injected.txt"), b"not declared").unwrap();
        assert!(verify_bundle(&destination)
            .unwrap_err()
            .to_string()
            .contains("undeclared"));
        fs::remove_dir_all(root).unwrap();
    }
}
