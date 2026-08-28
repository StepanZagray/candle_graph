//! Canonical capture-to-bundle publication with idempotent crash reconciliation.
//!
//! [`CaptureRun`] owns the whole path from "start capturing one planned profile
//! run" to "a deeply verified, atomically published evidence bundle exists at
//! the destination": staging-trace placement, session lifetime, evidence and
//! viewer derivation, manifest hashing, fsync, atomic rename, and a typed
//! [`PublicationReceipt`]. Applications never write evidence files or rename
//! directories themselves.
//!
//! Publication is idempotent across crashes: retrying the same planned capture
//! against a destination that already holds a verified bundle for the same run
//! coordinates returns [`PublicationStatus::AlreadyPublished`] with a fresh
//! verification receipt, while any other pre-existing destination fails closed
//! as a conflict.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifact::{publish_bundle, verify_bundle, BundleVerificationReceipt};
use crate::instrument::{ProfileRun, TraceSession};
use crate::trace::{parse_trace, TraceRunMeta};

pub const PUBLICATION_SCHEMA: &str = "candle-graph/publication/1";

/// Whether this receipt covers a fresh publication or a reconciled retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationStatus {
    Published,
    AlreadyPublished,
}

/// Typed proof that one planned capture is durably published and verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationReceipt {
    pub schema: String,
    pub status: PublicationStatus,
    pub bundle_path: PathBuf,
    pub run_id: String,
    pub verification: BundleVerificationReceipt,
}

/// Outcome of [`CaptureRun::begin`]: either the planned capture is already
/// durably published (crash-retry reconciliation) or an active capture run.
#[allow(clippy::large_enum_variant)]
pub enum CaptureBegin {
    AlreadyPublished(Box<PublicationReceipt>),
    Active(CaptureRun),
}

impl std::fmt::Debug for CaptureBegin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyPublished(receipt) => {
                f.debug_tuple("AlreadyPublished").field(receipt).finish()
            }
            Self::Active(run) => f
                .debug_struct("Active")
                .field("destination", &run.destination)
                .field("staging_trace", &run.staging_trace)
                .finish(),
        }
    }
}

/// One planned capture from session open through atomic bundle publication.
pub struct CaptureRun {
    session: Option<TraceSession>,
    run: ProfileRun,
    destination: PathBuf,
    staging_trace: PathBuf,
    nsight_dir: Option<PathBuf>,
}

impl CaptureRun {
    /// Reconcile the destination and, when the planned capture is not yet
    /// published, open a staged trace session next to it.
    pub fn begin(destination: impl Into<PathBuf>, run: ProfileRun) -> Result<CaptureBegin> {
        let destination = destination.into();
        for ancestor in destination.ancestors().skip(1) {
            ensure!(
                !ancestor.join("bundle.json").is_file(),
                "refusing to publish bundle {} inside existing bundle {}",
                destination.display(),
                ancestor.display()
            );
        }
        if let Some(receipt) = reconcile_published_bundle(&destination, &run)? {
            return Ok(CaptureBegin::AlreadyPublished(Box::new(receipt)));
        }
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
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
        let staging_trace = parent.join(format!(
            ".{name}.trace-{}-{nonce}.jsonl",
            std::process::id()
        ));
        let session = TraceSession::open(&staging_trace, run.clone())?;
        Ok(CaptureBegin::Active(Self {
            session: Some(session),
            run,
            destination,
            staging_trace,
            nsight_dir: None,
        }))
    }

    /// The live trace session for spans, ops, stats, scalars, and gradients.
    pub fn session(&self) -> &TraceSession {
        self.session
            .as_ref()
            .expect("an active capture run owns its trace session")
    }

    /// Retain a directory of official Nsight artifacts inside the bundle.
    pub fn with_nsight_dir(mut self, nsight_dir: impl Into<PathBuf>) -> Self {
        self.nsight_dir = Some(nsight_dir.into());
        self
    }

    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// The staged trace path. It exists until publication succeeds; on failure
    /// it is retained for diagnosis.
    pub fn staging_trace(&self) -> &Path {
        &self.staging_trace
    }

    /// Finish the session as complete and atomically publish the bundle.
    pub fn publish(mut self) -> Result<PublicationReceipt> {
        let session = self
            .session
            .take()
            .expect("an active capture run owns its trace session");
        let trace = session.finish()?;
        self.finalize(&trace)
    }

    /// Finish the session as an explicit failed capture and still publish the
    /// diagnosable bundle, so crashed campaign steps stay queryable.
    pub fn publish_failed(mut self, reason: impl Into<String>) -> Result<PublicationReceipt> {
        let session = self
            .session
            .take()
            .expect("an active capture run owns its trace session");
        let trace = session.finish_failed(reason)?;
        self.finalize(&trace)
    }

    fn finalize(&mut self, trace: &Path) -> Result<PublicationReceipt> {
        // The destination may have appeared since `begin` (concurrent retry).
        if let Some(receipt) = reconcile_published_bundle(&self.destination, &self.run)? {
            let _ = fs::remove_file(trace);
            return Ok(receipt);
        }
        publish_bundle(&self.destination, trace, self.nsight_dir.as_deref()).with_context(
            || {
                format!(
                    "publish capture bundle {} (staged trace retained at {})",
                    self.destination.display(),
                    trace.display()
                )
            },
        )?;
        let verification = verify_bundle(&self.destination).with_context(|| {
            format!(
                "verify published capture bundle {}",
                self.destination.display()
            )
        })?;
        let _ = fs::remove_file(trace);
        Ok(PublicationReceipt {
            schema: PUBLICATION_SCHEMA.into(),
            status: PublicationStatus::Published,
            bundle_path: self.destination.clone(),
            run_id: verification.run_id.clone(),
            verification,
        })
    }
}

/// If the destination already holds a deeply verified bundle for the same
/// planned capture, return its receipt; a missing destination returns `None`;
/// anything else fails closed as a conflict.
///
/// Retry identity is the planned-run coordinates — entrypoint, correlation ID,
/// phase, capture step, and device — not byte equality, because a retried
/// capture legitimately re-records timings under a new run ID.
pub fn reconcile_published_bundle(
    destination: &Path,
    run: &ProfileRun,
) -> Result<Option<PublicationReceipt>> {
    if !destination.exists() {
        return Ok(None);
    }
    let verification = verify_bundle(destination).with_context(|| {
        format!(
            "existing destination {} is not a verifiable evidence bundle; refusing to reuse or overwrite it",
            destination.display()
        )
    })?;
    let trace_path = destination.join("trace.jsonl");
    let document = parse_trace(&trace_path)
        .with_context(|| format!("parse published bundle trace {}", trace_path.display()))?;
    ensure!(
        verification.run_id == document.run.run_id,
        "published bundle manifest run ID {:?} does not match its trace run ID {:?}",
        verification.run_id,
        document.run.run_id
    );
    ensure_same_planned_capture(&document.run, run, destination)?;
    Ok(Some(PublicationReceipt {
        schema: PUBLICATION_SCHEMA.into(),
        status: PublicationStatus::AlreadyPublished,
        bundle_path: destination.to_path_buf(),
        run_id: verification.run_id.clone(),
        verification,
    }))
}

fn ensure_same_planned_capture(
    published: &TraceRunMeta,
    run: &ProfileRun,
    destination: &Path,
) -> Result<()> {
    let mismatches = [
        ("entrypoint", published.entrypoint != run.entrypoint),
        (
            "correlation_id",
            published.correlation_id != run.correlation_id,
        ),
        ("phase", published.phase != run.phase),
        ("capture_step", published.capture_step != run.capture_step),
        ("device", published.device != run.device),
    ]
    .into_iter()
    .filter_map(|(field, differs)| differs.then_some(field))
    .collect::<Vec<_>>();
    if !mismatches.is_empty() {
        bail!(
            "bundle destination {} already holds a different planned capture (conflicting {})",
            destination.display(),
            mismatches.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::trace_cli::load_evidence;
    use crate::trace::RunOutcome;

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "candle-graph-publication-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn active(begin: CaptureBegin) -> CaptureRun {
        match begin {
            CaptureBegin::Active(run) => run,
            CaptureBegin::AlreadyPublished(receipt) => {
                panic!("expected an active capture run, got {:?}", receipt.status)
            }
        }
    }

    #[test]
    fn capture_run_publishes_verified_bundle_and_cleans_staging() {
        let root = temp_root("publish");
        let destination = root.join("profiles/update-2");
        let run = ProfileRun::training("train::update", 2, "cpu");
        let capture = active(CaptureRun::begin(&destination, run.clone()).unwrap());
        let staging = capture.staging_trace().to_path_buf();
        {
            let session = capture.session();
            let measured = session.begin_measurement("update");
            session
                .record_scalar(measured.id(), "loss/total", 0.75)
                .unwrap();
        }
        let receipt = capture.publish().unwrap();
        assert_eq!(receipt.status, PublicationStatus::Published);
        assert_eq!(receipt.schema, PUBLICATION_SCHEMA);
        assert_eq!(receipt.run_id, receipt.verification.run_id);
        assert!(destination.join("bundle.json").is_file());
        assert!(!staging.exists(), "staging trace must be removed");

        let evidence = load_evidence(&destination).unwrap();
        assert_eq!(evidence.provenance.run_id, receipt.run_id);
        assert_eq!(evidence.tensor_stats.len(), 1);

        // Crash-retry: the same planned capture reconciles without recapturing.
        let retry = CaptureRun::begin(&destination, run).unwrap();
        match retry {
            CaptureBegin::AlreadyPublished(reconciled) => {
                assert_eq!(reconciled.status, PublicationStatus::AlreadyPublished);
                assert_eq!(reconciled.run_id, receipt.run_id);
                assert_eq!(
                    reconciled.verification.manifest_sha256,
                    receipt.verification.manifest_sha256
                );
            }
            CaptureBegin::Active(_) => panic!("expected reconciliation"),
        }

        // A different planned capture at the same destination is a conflict.
        let conflict = CaptureRun::begin(
            &destination,
            ProfileRun::training("train::update", 3, "cpu"),
        )
        .unwrap_err();
        assert!(conflict.to_string().contains("conflicting"));
        assert!(conflict.to_string().contains("capture_step"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn non_bundle_destination_fails_closed() {
        let root = temp_root("conflict");
        let destination = root.join("profiles/update-2");
        fs::create_dir_all(&destination).unwrap();
        fs::write(destination.join("application.jsonl"), b"not a bundle").unwrap();
        let error = CaptureRun::begin(
            &destination,
            ProfileRun::training("train::update", 2, "cpu"),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("not a verifiable evidence bundle"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_captures_publish_diagnosable_verified_bundles() {
        let root = temp_root("failed");
        let destination = root.join("update-7");
        let run = ProfileRun::training("train::update", 7, "cpu");
        let capture = active(CaptureRun::begin(&destination, run.clone()).unwrap());
        let receipt = capture.publish_failed("loss became non-finite").unwrap();
        assert_eq!(receipt.status, PublicationStatus::Published);
        verify_bundle(&destination).unwrap();
        let document = parse_trace(destination.join("trace.jsonl")).unwrap();
        assert_eq!(document.terminal.outcome, RunOutcome::Failed);
        assert_eq!(
            document.terminal.reason.as_deref(),
            Some("loss became non-finite")
        );

        // Failed publications also reconcile instead of recapturing.
        match CaptureRun::begin(&destination, run).unwrap() {
            CaptureBegin::AlreadyPublished(reconciled) => {
                assert_eq!(reconciled.run_id, receipt.run_id);
            }
            CaptureBegin::Active(_) => panic!("expected reconciliation"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nested_bundle_destinations_are_rejected() {
        let root = temp_root("nested");
        let outer = root.join("outer");
        let capture = active(
            CaptureRun::begin(&outer, ProfileRun::training("train::update", 1, "cpu")).unwrap(),
        );
        drop(capture.session().begin_measurement("update"));
        capture.publish().unwrap();
        let error = CaptureRun::begin(
            outer.join("inner"),
            ProfileRun::training("train::update", 2, "cpu"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("inside existing bundle"));
        fs::remove_dir_all(root).unwrap();
    }
}
