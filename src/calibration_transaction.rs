//! Transactional, fail-closed calibration lifecycle for M-SLIM profiles.
//!
//! The candidate is evaluated in shadow before it can replace the active
//! profile. Entering canary performs a checkpointed atomic activation; promotion
//! always requires an explicit call. Any failed canary gate restores and verifies
//! the checkpoint before the candidate is quarantined.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const RECEIPT_SCHEMA: &str = "mslim.calibration_transaction.receipt/v2";
const LEGACY_RECEIPT_SCHEMA: &str = "mslim.calibration_transaction.receipt/v1";
const PROFILE_SCHEMA: &str = "mslim.calibration_profile/v1";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub type CalibrationResult<T> = Result<T, CalibrationError>;

#[derive(Debug)]
pub enum CalibrationError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
    Integrity(String),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "calibration I/O error: {error}"),
            Self::Json(error) => write!(f, "calibration JSON error: {error}"),
            Self::Invalid(message) => write!(f, "invalid calibration transaction: {message}"),
            Self::Integrity(message) => write!(f, "calibration integrity failure: {message}"),
        }
    }
}

impl std::error::Error for CalibrationError {}

impl From<std::io::Error> for CalibrationError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for CalibrationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationState {
    Proposed,
    Shadow,
    Canary,
    Promoted,
    RolledBack,
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProfile {
    pub schema: String,
    pub profile_id: String,
    pub model_identity: String,
    pub backend_identity: String,
    pub parameters: BTreeMap<String, Value>,
}

impl CalibrationProfile {
    pub fn new(
        profile_id: impl Into<String>,
        model_identity: impl Into<String>,
        backend_identity: impl Into<String>,
        parameters: BTreeMap<String, Value>,
    ) -> CalibrationResult<Self> {
        let profile = Self {
            schema: PROFILE_SCHEMA.into(),
            profile_id: profile_id.into(),
            model_identity: model_identity.into(),
            backend_identity: backend_identity.into(),
            parameters,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> CalibrationResult<()> {
        if self.schema != PROFILE_SCHEMA {
            return Err(CalibrationError::Invalid(
                "unsupported profile schema".into(),
            ));
        }
        if self.profile_id.trim().is_empty()
            || self.model_identity.trim().is_empty()
            || self.backend_identity.trim().is_empty()
        {
            return Err(CalibrationError::Invalid(
                "profile, model and backend identities must be non-empty".into(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> CalibrationResult<String> {
        digest_serializable(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationObservation {
    pub profile_id: String,
    pub workload_id: String,
    pub seed: u64,
    pub sample_count: u64,
    pub parity_ok: bool,
    pub maximum_exactness_error: f64,
    pub maximum_certificate_error: f64,
    pub throughput_tokens_per_second: f64,
    pub latency_p95_ms: f64,
    pub peak_memory_bytes: u64,
    pub bytes_read_per_token: u64,
}

impl CalibrationObservation {
    pub fn validate(&self) -> CalibrationResult<()> {
        if self.profile_id.trim().is_empty() || self.workload_id.trim().is_empty() {
            return Err(CalibrationError::Invalid(
                "observation identity is empty".into(),
            ));
        }
        if self.sample_count == 0 {
            return Err(CalibrationError::Invalid(
                "observation has no samples".into(),
            ));
        }
        for (name, value) in [
            ("maximum_exactness_error", self.maximum_exactness_error),
            ("maximum_certificate_error", self.maximum_certificate_error),
            (
                "throughput_tokens_per_second",
                self.throughput_tokens_per_second,
            ),
            ("latency_p95_ms", self.latency_p95_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CalibrationError::Invalid(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationGates {
    pub maximum_exactness_error: f64,
    pub maximum_certificate_error: f64,
    pub minimum_throughput_ratio: f64,
    pub maximum_latency_ratio: f64,
    pub maximum_memory_ratio: f64,
    pub minimum_samples: u64,
}

impl Default for CalibrationGates {
    fn default() -> Self {
        Self {
            maximum_exactness_error: 0.0,
            maximum_certificate_error: 0.0,
            minimum_throughput_ratio: 1.0,
            maximum_latency_ratio: 1.0,
            maximum_memory_ratio: 1.0,
            minimum_samples: 3,
        }
    }
}

impl CalibrationGates {
    pub fn validate(&self) -> CalibrationResult<()> {
        if self.minimum_samples == 0 {
            return Err(CalibrationError::Invalid(
                "minimum_samples must be positive".into(),
            ));
        }
        for (name, value) in [
            ("maximum_exactness_error", self.maximum_exactness_error),
            ("maximum_certificate_error", self.maximum_certificate_error),
            ("minimum_throughput_ratio", self.minimum_throughput_ratio),
            ("maximum_latency_ratio", self.maximum_latency_ratio),
            ("maximum_memory_ratio", self.maximum_memory_ratio),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CalibrationError::Invalid(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateReport {
    pub accepted: bool,
    pub correctness_passed: bool,
    pub performance_passed: bool,
    pub failures: Vec<String>,
}

pub fn evaluate_candidate(
    baseline: &CalibrationObservation,
    candidate: &CalibrationObservation,
    gates: &CalibrationGates,
) -> CalibrationResult<GateReport> {
    baseline.validate()?;
    candidate.validate()?;
    gates.validate()?;
    if baseline.workload_id != candidate.workload_id || baseline.seed != candidate.seed {
        return Err(CalibrationError::Invalid(
            "baseline and candidate must use the same workload and seed".into(),
        ));
    }

    if baseline.sample_count < gates.minimum_samples
        || !baseline.parity_ok
        || baseline.maximum_exactness_error > gates.maximum_exactness_error
        || baseline.maximum_certificate_error > gates.maximum_certificate_error
        || baseline.throughput_tokens_per_second <= 0.0
        || baseline.latency_p95_ms <= 0.0
        || baseline.peak_memory_bytes == 0
        || candidate.peak_memory_bytes == 0
    {
        return Err(CalibrationError::Invalid(
            "baseline or memory evidence is insufficient".into(),
        ));
    }
    // Correctness is deliberately evaluated before any performance property.
    let mut failures = Vec::new();
    if candidate.sample_count < gates.minimum_samples {
        failures.push("insufficient_candidate_samples".into());
    }
    if !candidate.parity_ok {
        failures.push("parity_failed".into());
    }
    if candidate.maximum_exactness_error > gates.maximum_exactness_error {
        failures.push("exactness_error_exceeded".into());
    }
    if candidate.maximum_certificate_error > gates.maximum_certificate_error {
        failures.push("certificate_error_exceeded".into());
    }
    let correctness_passed = failures.is_empty();
    if !correctness_passed {
        return Ok(GateReport {
            accepted: false,
            correctness_passed: false,
            performance_passed: false,
            failures,
        });
    }

    let throughput_floor = baseline.throughput_tokens_per_second * gates.minimum_throughput_ratio;
    if candidate.throughput_tokens_per_second < throughput_floor {
        failures.push("throughput_gate_failed".into());
    }
    if candidate.latency_p95_ms > baseline.latency_p95_ms * gates.maximum_latency_ratio {
        failures.push("latency_gate_failed".into());
    }
    let memory_limit = (baseline.peak_memory_bytes as f64) * gates.maximum_memory_ratio;
    if (candidate.peak_memory_bytes as f64) > memory_limit {
        failures.push("memory_gate_failed".into());
    }
    let performance_passed = failures.is_empty();
    Ok(GateReport {
        accepted: correctness_passed && performance_passed,
        correctness_passed,
        performance_passed,
        failures,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProposal {
    pub proposal_id: String,
    pub baseline_profile_digest: String,
    pub candidate: CalibrationProfile,
    pub gates: CalibrationGates,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProfileCheckpoint {
    pub profile_path: String,
    pub checkpoint_path: String,
    pub original_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationTransactionReceipt {
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub product: String,
    #[serde(default)]
    pub technical_namespace: String,
    pub schema: String,
    pub sequence: u64,
    pub timestamp_unix_ms: u128,
    pub proposal_id: String,
    pub state: CalibrationState,
    pub event: String,
    pub details: Map<String, Value>,
    pub previous_hash: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize)]
struct ReceiptPayload<'a> {
    project: &'a str,
    product: &'a str,
    technical_namespace: &'a str,
    schema: &'a str,
    sequence: u64,
    timestamp_unix_ms: u128,
    proposal_id: &'a str,
    state: CalibrationState,
    event: &'a str,
    details: &'a Map<String, Value>,
    previous_hash: &'a str,
}

/// Exact payload used by historical v1 chains.  It intentionally has no
/// identity fields; changing this layout would make already-written hashes
/// unverifiable.
#[derive(Debug, Clone, Serialize)]
struct LegacyReceiptPayload<'a> {
    schema: &'a str,
    sequence: u64,
    timestamp_unix_ms: u128,
    proposal_id: &'a str,
    state: CalibrationState,
    event: &'a str,
    details: &'a Map<String, Value>,
    previous_hash: &'a str,
}

#[derive(Debug)]
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: PathBuf) -> CalibrationResult<Self> {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                file.sync_all()?;
                Ok(Self { path })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Err(
                CalibrationError::Invalid("calibration transaction is locked".into()),
            ),
            Err(error) => Err(error.into()),
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug)]
pub struct CalibrationTransaction {
    root: PathBuf,
    profile_path: PathBuf,
    receipt_path: PathBuf,
    proposal: CalibrationProposal,
    state: CalibrationState,
    checkpoint: Option<ProfileCheckpoint>,
    baseline_profile_id: String,
    shadow_passed: bool,
    canary_passed: bool,
    proposal_recorded: bool,
    // Held for the whole transaction, also across different transaction roots.
    _profile_lease: FileLock,
}

impl CalibrationTransaction {
    pub fn new(
        root: impl AsRef<Path>,
        profile_path: impl AsRef<Path>,
        proposal: CalibrationProposal,
    ) -> CalibrationResult<Self> {
        proposal.candidate.validate()?;
        proposal.gates.validate()?;
        if proposal.proposal_id.trim().is_empty() {
            return Err(CalibrationError::Invalid("proposal_id is empty".into()));
        }
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let profile_path = profile_path.as_ref().to_path_buf();
        if !profile_path.is_absolute() {
            return Err(CalibrationError::Invalid(
                "profile path must be absolute".into(),
            ));
        }
        let _profile_lease = FileLock::acquire(profile_path.with_extension("transaction.lock"))?;
        if pending_profile_path(&profile_path).exists() {
            return Err(CalibrationError::Integrity(
                "pending activation requires recovery".into(),
            ));
        }
        let baseline_bytes = fs::read(&profile_path)?;
        if digest_bytes(&baseline_bytes) != proposal.baseline_profile_digest {
            return Err(CalibrationError::Integrity(
                "baseline changed before proposal".into(),
            ));
        }
        let baseline: CalibrationProfile = serde_json::from_slice(&baseline_bytes)?;
        baseline.validate()?;
        if baseline.model_identity != proposal.candidate.model_identity
            || baseline.backend_identity != proposal.candidate.backend_identity
        {
            return Err(CalibrationError::Invalid(
                "candidate cannot change model or backend".into(),
            ));
        }
        let receipt_path = root.join("calibration-receipts.jsonl");
        let existing_receipts = verify_receipt_chain(&receipt_path)?;
        if existing_receipts
            .iter()
            .any(|receipt| receipt.schema == LEGACY_RECEIPT_SCHEMA)
        {
            return Err(CalibrationError::Invalid(
                "legacy v1 receipt chain is verified but read-only; start a new v2 transaction root"
                    .into(),
            ));
        }
        if existing_receipts
            .iter()
            .any(|r| r.proposal_id == proposal.proposal_id)
        {
            return Err(CalibrationError::Invalid(
                "proposal_id has already been used".into(),
            ));
        }
        Ok(Self {
            root,
            profile_path,
            receipt_path,
            proposal,
            state: CalibrationState::Proposed,
            checkpoint: None,
            baseline_profile_id: baseline.profile_id,
            shadow_passed: false,
            canary_passed: false,
            proposal_recorded: false,
            _profile_lease,
        })
    }

    pub fn state(&self) -> CalibrationState {
        self.state
    }

    pub fn receipt_path(&self) -> &Path {
        &self.receipt_path
    }

    pub fn propose(&mut self) -> CalibrationResult<CalibrationTransactionReceipt> {
        if self.proposal_recorded {
            return self.idempotent_receipt("proposal_recorded");
        }
        self.require_state(CalibrationState::Proposed)?;
        let receipt = self.append_receipt(
            "proposal_recorded",
            Map::from_iter([
                ("proposal".into(), serde_json::to_value(&self.proposal)?),
                ("profile_path".into(), json!(self.profile_path)),
            ]),
        )?;
        self.proposal_recorded = true;
        Ok(receipt)
    }

    pub fn begin_shadow(&mut self) -> CalibrationResult<CalibrationTransactionReceipt> {
        self.require_state(CalibrationState::Proposed)?;
        if !self.proposal_recorded {
            return Err(CalibrationError::Invalid(
                "proposal must be recorded before shadow".into(),
            ));
        }
        self.state = CalibrationState::Shadow;
        self.append_receipt("shadow_started", Map::new())
    }

    pub fn complete_shadow(
        &mut self,
        baseline: &CalibrationObservation,
        candidate: &CalibrationObservation,
    ) -> CalibrationResult<GateReport> {
        self.require_state(CalibrationState::Shadow)?;
        self.validate_observation_identities(baseline, candidate)?;
        self.shadow_passed = false;
        let report = evaluate_candidate(baseline, candidate, &self.proposal.gates)?;
        let mut details = Map::new();
        details.insert("gate_report".into(), serde_json::to_value(&report)?);
        if report.accepted {
            details.insert("baseline".into(), serde_json::to_value(baseline)?);
            details.insert("candidate".into(), serde_json::to_value(candidate)?);
            self.append_receipt("shadow_passed", details)?;
            self.shadow_passed = true;
        } else {
            self.state = CalibrationState::Quarantined;
            self.append_receipt("shadow_failed_quarantined", details)?;
        }
        Ok(report)
    }

    /// Activates the candidate for canary traffic. This never promotes it.
    pub fn begin_canary(&mut self) -> CalibrationResult<CalibrationTransactionReceipt> {
        self.require_state(CalibrationState::Shadow)?;
        if !self.shadow_passed {
            return Err(CalibrationError::Invalid(
                "canary requires a passing shadow evaluation".into(),
            ));
        }
        let _lock = FileLock::acquire(self.root.join(".calibration.lock"))?;
        let active = fs::read(&self.profile_path)?;
        let active_digest = digest_bytes(&active);
        if active_digest != self.proposal.baseline_profile_digest {
            return Err(CalibrationError::Integrity(
                "active profile no longer matches proposal baseline".into(),
            ));
        }
        let checkpoint_path = self.root.join(format!(
            "{}-profile.checkpoint",
            digest_bytes(self.proposal.proposal_id.as_bytes())
        ));
        atomic_write(&checkpoint_path, &active)?;
        let candidate = serde_json::to_vec_pretty(&self.proposal.candidate)?;
        // Write-ahead marker blocks production readers after an interrupted canary.
        atomic_write(
            &pending_profile_path(&self.profile_path),
            &serde_json::to_vec_pretty(&json!({
                "schema": "mslim.pending_calibration/v1",
                "proposal_id": self.proposal.proposal_id,
                "checkpoint_path": checkpoint_path,
                "original_digest": active_digest,
                "candidate_digest": digest_bytes(&candidate)
            }))?,
        )?;
        atomic_write(&self.profile_path, &candidate)?;
        let written = fs::read(&self.profile_path)?;
        if digest_bytes(&written) != digest_bytes(&candidate) {
            atomic_write(&self.profile_path, &active)?;
            return Err(CalibrationError::Integrity(
                "candidate profile activation could not be verified".into(),
            ));
        }
        self.checkpoint = Some(ProfileCheckpoint {
            profile_path: self.profile_path.to_string_lossy().into_owned(),
            checkpoint_path: checkpoint_path.to_string_lossy().into_owned(),
            original_digest: active_digest,
        });
        self.state = CalibrationState::Canary;
        let mut details = Map::new();
        details.insert("candidate_digest".into(), json!(digest_bytes(&candidate)));
        self.append_receipt("canary_started", details)
    }

    pub fn complete_canary(
        &mut self,
        baseline: &CalibrationObservation,
        candidate: &CalibrationObservation,
    ) -> CalibrationResult<GateReport> {
        self.require_state(CalibrationState::Canary)?;
        self.validate_observation_identities(baseline, candidate)?;
        self.canary_passed = false;
        let report = evaluate_candidate(baseline, candidate, &self.proposal.gates)?;
        let mut details = Map::new();
        details.insert("gate_report".into(), serde_json::to_value(&report)?);
        if report.accepted {
            // Remain in Canary. Promotion is a separate, explicit authority action.
            details.insert("baseline".into(), serde_json::to_value(baseline)?);
            details.insert("candidate".into(), serde_json::to_value(candidate)?);
            self.append_receipt("canary_passed_awaiting_promotion", details)?;
            self.canary_passed = true;
        } else {
            self.rollback_internal("canary_gate_failed", details)?;
        }
        Ok(report)
    }

    pub fn promote(&mut self, authority: &str) -> CalibrationResult<CalibrationTransactionReceipt> {
        self.require_state(CalibrationState::Canary)?;
        if authority.trim().is_empty() {
            return Err(CalibrationError::Invalid(
                "promotion authority is empty".into(),
            ));
        }
        if !self.canary_passed {
            return Err(CalibrationError::Invalid(
                "promotion requires a passing canary in this transaction".into(),
            ));
        }
        let expected = digest_bytes(&serde_json::to_vec_pretty(&self.proposal.candidate)?);
        if digest_bytes(&fs::read(&self.profile_path)?) != expected {
            return Err(CalibrationError::Integrity(
                "active candidate changed before promotion".into(),
            ));
        }
        self.state = CalibrationState::Promoted;
        let details = Map::from_iter([
            ("authority".into(), json!(authority)),
            ("candidate_digest".into(), json!(expected)),
        ]);
        let receipt = match self.append_receipt("promotion_committed", details) {
            Ok(receipt) => receipt,
            Err(error) => {
                self.state = CalibrationState::Canary;
                return Err(error);
            }
        };
        fs::remove_file(pending_profile_path(&self.profile_path))?;
        Ok(receipt)
    }

    pub fn rollback(&mut self, reason: &str) -> CalibrationResult<CalibrationTransactionReceipt> {
        if self.state == CalibrationState::RolledBack {
            return self.idempotent_receipt("rollback_verified");
        }
        if self.state != CalibrationState::Canary && self.state != CalibrationState::Promoted {
            return Err(CalibrationError::Invalid(
                "rollback requires canary or promoted state".into(),
            ));
        }
        self.rollback_internal(reason, Map::new())
    }

    fn rollback_internal(
        &mut self,
        reason: &str,
        mut details: Map<String, Value>,
    ) -> CalibrationResult<CalibrationTransactionReceipt> {
        if reason.trim().is_empty() {
            return Err(CalibrationError::Invalid("rollback reason is empty".into()));
        }
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| CalibrationError::Integrity("rollback checkpoint is missing".into()))?
            .clone();
        let _lock = FileLock::acquire(self.root.join(".calibration.lock"))?;
        let original = fs::read(&checkpoint.checkpoint_path)?;
        if digest_bytes(&original) != checkpoint.original_digest {
            self.state = CalibrationState::Quarantined;
            let _ = self.append_receipt(
                "rollback_checkpoint_corrupt_quarantined",
                Map::from_iter([("reason".into(), json!(reason))]),
            );
            return Err(CalibrationError::Integrity(
                "rollback checkpoint digest mismatch".into(),
            ));
        }
        let expected_candidate =
            digest_bytes(&serde_json::to_vec_pretty(&self.proposal.candidate)?);
        if digest_bytes(&fs::read(&self.profile_path)?) != expected_candidate {
            return Err(CalibrationError::Integrity(
                "active profile changed outside transaction; refusing overwrite".into(),
            ));
        }
        atomic_write(&self.profile_path, &original)?;
        let restored = fs::read(&self.profile_path)?;
        if digest_bytes(&restored) != checkpoint.original_digest {
            self.state = CalibrationState::Quarantined;
            let _ = self.append_receipt(
                "rollback_verification_failed_quarantined",
                Map::from_iter([("reason".into(), json!(reason))]),
            );
            return Err(CalibrationError::Integrity(
                "restored profile digest mismatch".into(),
            ));
        }
        self.state = CalibrationState::RolledBack;
        details.insert("reason".into(), json!(reason));
        details.insert("restored_digest".into(), json!(checkpoint.original_digest));
        let receipt = self.append_receipt("rollback_verified", details)?;
        let pending = pending_profile_path(&self.profile_path);
        if pending.exists() {
            fs::remove_file(pending)?;
        }
        Ok(receipt)
    }

    fn validate_observation_identities(
        &self,
        baseline: &CalibrationObservation,
        candidate: &CalibrationObservation,
    ) -> CalibrationResult<()> {
        if baseline.profile_id != self.baseline_profile_id
            || candidate.profile_id != self.proposal.candidate.profile_id
        {
            return Err(CalibrationError::Invalid(
                "measurement belongs to another profile".into(),
            ));
        }
        Ok(())
    }

    fn require_state(&self, expected: CalibrationState) -> CalibrationResult<()> {
        if self.state != expected {
            return Err(CalibrationError::Invalid(format!(
                "expected state {expected:?}, found {:?}",
                self.state
            )));
        }
        Ok(())
    }

    fn append_receipt(
        &self,
        event: &str,
        details: Map<String, Value>,
    ) -> CalibrationResult<CalibrationTransactionReceipt> {
        let _receipt_lock = FileLock::acquire(self.root.join(".calibration-receipts.lock"))?;
        let receipts = verify_receipt_chain(&self.receipt_path)?;
        let sequence = receipts.len() as u64;
        let previous_hash = receipts
            .last()
            .map(|row| row.hash.clone())
            .unwrap_or_default();
        let timestamp_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CalibrationError::Invalid("system clock precedes epoch".into()))?
            .as_millis();
        let payload = ReceiptPayload {
            project: crate::project_identity::PROJECT,
            product: crate::project_identity::PRODUCT,
            technical_namespace: crate::project_identity::TECHNICAL_NAMESPACE,
            schema: RECEIPT_SCHEMA,
            sequence,
            timestamp_unix_ms,
            proposal_id: &self.proposal.proposal_id,
            state: self.state,
            event,
            details: &details,
            previous_hash: &previous_hash,
        };
        let hash = digest_serializable(&payload)?;
        let receipt = CalibrationTransactionReceipt {
            project: crate::project_identity::PROJECT.into(),
            product: crate::project_identity::PRODUCT.into(),
            technical_namespace: crate::project_identity::TECHNICAL_NAMESPACE.into(),
            schema: RECEIPT_SCHEMA.into(),
            sequence,
            timestamp_unix_ms,
            proposal_id: self.proposal.proposal_id.clone(),
            state: self.state,
            event: event.into(),
            details,
            previous_hash,
            hash,
        };
        append_json_line(&self.receipt_path, &receipt)?;
        Ok(receipt)
    }

    fn idempotent_receipt(&self, event: &str) -> CalibrationResult<CalibrationTransactionReceipt> {
        verify_receipt_chain(&self.receipt_path)?
            .into_iter()
            .rev()
            .find(|receipt| {
                receipt.proposal_id == self.proposal.proposal_id && receipt.event == event
            })
            .ok_or_else(|| {
                CalibrationError::Invalid(format!("idempotent receipt {event} does not exist"))
            })
    }
}

pub fn verify_receipt_chain(
    path: impl AsRef<Path>,
) -> CalibrationResult<Vec<CalibrationTransactionReceipt>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path)?;
    let mut receipts = Vec::new();
    let mut expected_previous = String::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            return Err(CalibrationError::Integrity(format!(
                "empty receipt at line {}",
                index + 1
            )));
        }
        let receipt: CalibrationTransactionReceipt = serde_json::from_str(&line)?;
        if !matches!(
            receipt.schema.as_str(),
            RECEIPT_SCHEMA | LEGACY_RECEIPT_SCHEMA
        ) || receipt.sequence != index as u64
            || receipt.previous_hash != expected_previous
        {
            return Err(CalibrationError::Integrity(format!(
                "receipt chain link invalid at line {}",
                index + 1
            )));
        }
        let expected_hash = if receipt.schema == LEGACY_RECEIPT_SCHEMA {
            if !receipt.project.is_empty()
                || !receipt.product.is_empty()
                || !receipt.technical_namespace.is_empty()
            {
                return Err(CalibrationError::Integrity(format!(
                    "legacy receipt contains v2 identity fields at line {}",
                    index + 1
                )));
            }
            digest_serializable(&LegacyReceiptPayload {
                schema: &receipt.schema,
                sequence: receipt.sequence,
                timestamp_unix_ms: receipt.timestamp_unix_ms,
                proposal_id: &receipt.proposal_id,
                state: receipt.state,
                event: &receipt.event,
                details: &receipt.details,
                previous_hash: &receipt.previous_hash,
            })?
        } else {
            if receipt.project != crate::project_identity::PROJECT
                || receipt.product != crate::project_identity::PRODUCT
                || receipt.technical_namespace != crate::project_identity::TECHNICAL_NAMESPACE
            {
                return Err(CalibrationError::Integrity(format!(
                    "receipt project identity invalid at line {}",
                    index + 1
                )));
            }
            digest_serializable(&ReceiptPayload {
                project: &receipt.project,
                product: &receipt.product,
                technical_namespace: &receipt.technical_namespace,
                schema: &receipt.schema,
                sequence: receipt.sequence,
                timestamp_unix_ms: receipt.timestamp_unix_ms,
                proposal_id: &receipt.proposal_id,
                state: receipt.state,
                event: &receipt.event,
                details: &receipt.details,
                previous_hash: &receipt.previous_hash,
            })?
        };
        if receipt.hash != expected_hash {
            return Err(CalibrationError::Integrity(format!(
                "receipt hash invalid at line {}",
                index + 1
            )));
        }
        expected_previous = receipt.hash.clone();
        receipts.push(receipt);
    }
    Ok(receipts)
}

fn append_json_line(path: &Path, value: &impl Serialize) -> CalibrationResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> CalibrationResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| CalibrationError::Invalid("atomic write target has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("profile"),
        std::process::id(),
        sequence
    ));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn digest_serializable(value: &impl Serialize) -> CalibrationResult<String> {
    Ok(digest_bytes(&serde_json::to_vec(value)?))
}

fn digest_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// A pending marker is a denial of production activation, not an authorization.
pub fn pending_profile_path(profile_path: &Path) -> PathBuf {
    profile_path.with_extension("pending.json")
}

impl Drop for CalibrationTransaction {
    fn drop(&mut self) {
        if self.state == CalibrationState::Canary {
            // An I/O failure leaves the write-ahead marker in place: readers fail closed.
            let _ = self.rollback("transaction_dropped_without_promotion");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn profile(id: &str, value: u64) -> CalibrationProfile {
        CalibrationProfile::new(
            id,
            "model",
            "portable",
            BTreeMap::from([("page_tokens".into(), json!(value))]),
        )
        .unwrap()
    }

    fn observation(id: &str, throughput: f64, parity: bool) -> CalibrationObservation {
        CalibrationObservation {
            profile_id: id.into(),
            workload_id: "decode-32k".into(),
            seed: 7,
            sample_count: 8,
            parity_ok: parity,
            maximum_exactness_error: 0.0,
            maximum_certificate_error: 0.0,
            throughput_tokens_per_second: throughput,
            latency_p95_ms: 10.0,
            peak_memory_bytes: 100,
            bytes_read_per_token: 1000,
        }
    }

    fn transaction() -> (TempDir, PathBuf, CalibrationTransaction) {
        let temp = tempfile::tempdir().unwrap();
        let active_path = temp.path().join("active.json");
        let active = profile("active", 16);
        atomic_write(&active_path, &serde_json::to_vec_pretty(&active).unwrap()).unwrap();
        let proposal = CalibrationProposal {
            proposal_id: "candidate-1".into(),
            baseline_profile_digest: digest_bytes(&fs::read(&active_path).unwrap()),
            candidate: profile("candidate", 64),
            gates: CalibrationGates::default(),
        };
        let tx =
            CalibrationTransaction::new(temp.path().join("state"), active_path.clone(), proposal)
                .unwrap();
        (temp, active_path, tx)
    }

    #[test]
    fn failed_canary_forces_verified_rollback() {
        let (_temp, active_path, mut tx) = transaction();
        let original = fs::read(&active_path).unwrap();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        assert!(
            tx.complete_shadow(
                &observation("active", 10.0, true),
                &observation("candidate", 12.0, true),
            )
            .unwrap()
            .accepted
        );
        tx.begin_canary().unwrap();
        let report = tx
            .complete_canary(
                &observation("active", 10.0, true),
                &observation("candidate", 20.0, false),
            )
            .unwrap();
        assert!(!report.accepted);
        assert_eq!(tx.state(), CalibrationState::RolledBack);
        assert_eq!(fs::read(active_path).unwrap(), original);
    }

    #[test]
    fn explicit_rollback_is_idempotent() {
        let (_temp, active_path, mut tx) = transaction();
        let original = fs::read(&active_path).unwrap();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        tx.complete_shadow(
            &observation("active", 10.0, true),
            &observation("candidate", 11.0, true),
        )
        .unwrap();
        tx.begin_canary().unwrap();
        let first = tx.rollback("operator_cancelled").unwrap();
        let second = tx.rollback("operator_cancelled").unwrap();
        assert_eq!(first.hash, second.hash);
        assert_eq!(fs::read(active_path).unwrap(), original);
        let receipts = verify_receipt_chain(tx.receipt_path()).unwrap();
        assert_eq!(
            receipts
                .iter()
                .filter(|row| row.event == "rollback_verified")
                .count(),
            1
        );
    }

    #[test]
    fn forged_receipt_is_rejected() {
        let (_temp, _active_path, mut tx) = transaction();
        tx.propose().unwrap();
        let path = tx.receipt_path().to_path_buf();
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, text.replace("proposal_recorded", "proposal_forged")).unwrap();
        assert!(matches!(
            verify_receipt_chain(path),
            Err(CalibrationError::Integrity(_))
        ));
    }

    fn collect_receipt_chains(root: &Path, output: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_receipt_chains(&path, output);
            } else if path.file_name().and_then(|name| name.to_str())
                == Some("calibration-receipts.jsonl")
            {
                output.push(path);
            }
        }
    }

    #[test]
    fn installed_legacy_chains_remain_verifiable_and_read_only() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("deploy/local-production/runtime/receipts");
        let mut paths = Vec::new();
        collect_receipt_chains(&root, &mut paths);
        for path in paths {
            let receipts = verify_receipt_chain(&path)
                .unwrap_or_else(|error| panic!("{} is not verifiable: {error}", path.display()));
            if receipts
                .iter()
                .any(|receipt| receipt.schema == LEGACY_RECEIPT_SCHEMA)
            {
                assert!(
                    receipts
                        .iter()
                        .all(|receipt| receipt.schema == LEGACY_RECEIPT_SCHEMA)
                );
            }
        }
    }

    #[test]
    fn candidate_never_auto_promotes() {
        let (_temp, _active_path, mut tx) = transaction();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        tx.complete_shadow(
            &observation("active", 10.0, true),
            &observation("candidate", 12.0, true),
        )
        .unwrap();
        tx.begin_canary().unwrap();
        tx.complete_canary(
            &observation("active", 10.0, true),
            &observation("candidate", 12.0, true),
        )
        .unwrap();
        assert_eq!(tx.state(), CalibrationState::Canary);
        tx.promote("operator:test").unwrap();
        assert_eq!(tx.state(), CalibrationState::Promoted);
    }
    #[test]
    fn unmeasured_shadow_cannot_activate() {
        let (_temp, path, mut tx) = transaction();
        let original = fs::read(&path).unwrap();
        assert!(tx.begin_shadow().is_err());
        let first = tx.propose().unwrap();
        assert_eq!(first.hash, tx.propose().unwrap().hash);
        tx.begin_shadow().unwrap();
        assert!(tx.begin_canary().is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn foreign_measurements_and_uncertified_baseline_are_rejected() {
        let (_temp, _path, mut tx) = transaction();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        assert!(
            tx.complete_shadow(
                &observation("active", 10.0, true),
                &observation("other", 20.0, true)
            )
            .is_err()
        );
        assert!(
            tx.complete_shadow(
                &observation("active", 10.0, false),
                &observation("candidate", 20.0, true)
            )
            .is_err()
        );
        assert!(tx.begin_canary().is_err());
    }

    #[test]
    fn dropping_unpromoted_canary_restores_file_and_clears_pending_marker() {
        let (_temp, path, mut tx) = transaction();
        let original = fs::read(&path).unwrap();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        tx.complete_shadow(
            &observation("active", 10.0, true),
            &observation("candidate", 20.0, true),
        )
        .unwrap();
        tx.begin_canary().unwrap();
        assert!(pending_profile_path(&path).exists());
        assert!(tx.promote("test").is_err());
        drop(tx);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(!pending_profile_path(&path).exists());
    }

    #[test]
    fn external_changes_are_not_overwritten_by_rollback() {
        let (_temp, path, mut tx) = transaction();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        tx.complete_shadow(
            &observation("active", 10.0, true),
            &observation("candidate", 20.0, true),
        )
        .unwrap();
        tx.begin_canary().unwrap();
        fs::write(&path, b"external modification").unwrap();
        assert!(tx.rollback("test").is_err());
        drop(tx);
        assert_eq!(fs::read(&path).unwrap(), b"external modification");
        assert!(pending_profile_path(&path).exists());
    }

    #[test]
    fn same_profile_cannot_have_two_simultaneous_transactions() {
        let (temp, path, tx) = transaction();
        let second = CalibrationTransaction::new(
            temp.path().join("other"),
            path.clone(),
            tx.proposal.clone(),
        );
        assert!(second.is_err());
        assert!(path.with_extension("transaction.lock").exists());
    }

    #[test]
    fn promotion_requires_unchanged_candidate_file() {
        let (_temp, path, mut tx) = transaction();
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        tx.complete_shadow(
            &observation("active", 10.0, true),
            &observation("candidate", 20.0, true),
        )
        .unwrap();
        tx.begin_canary().unwrap();
        tx.complete_canary(
            &observation("active", 10.0, true),
            &observation("candidate", 20.0, true),
        )
        .unwrap();
        fs::write(path, b"changed").unwrap();
        assert!(tx.promote("operator").is_err());
    }
    #[test]
    fn measured_decimal_times_survive_receipt_replay() {
        let (_dir, path, mut tx) = transaction();
        let mut base = observation("active", 4.954511420406614, true);
        base.latency_p95_ms = 4883.16112;
        let mut candidate = observation("candidate", 15.74695350653615, true);
        candidate.latency_p95_ms = 1464.1582419999997;
        tx.propose().unwrap();
        tx.begin_shadow().unwrap();
        assert!(tx.complete_shadow(&base, &candidate).unwrap().accepted);
        tx.begin_canary().unwrap();
        assert!(tx.complete_canary(&base, &candidate).unwrap().accepted);
        tx.promote("decimal-regression").unwrap();
        let receipts = verify_receipt_chain(tx.receipt_path()).unwrap();
        let measured = receipts
            .iter()
            .find(|r| r.event == "canary_passed_awaiting_promotion")
            .unwrap();
        assert_eq!(
            measured.details["candidate"]["latency_p95_ms"]
                .as_f64()
                .unwrap()
                .to_bits(),
            candidate.latency_p95_ms.to_bits()
        );
        drop(tx);
        assert!(!pending_profile_path(&path).exists());
    }

    #[test]
    fn finite_json_floats_keep_their_bits() {
        let mut bits = 0x937a_fc05_631b_890d_u64;
        for _ in 0..4096 {
            bits = bits
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = f64::from_bits(bits);
            if !value.is_finite() {
                continue;
            }
            let encoded = serde_json::to_vec(&value).unwrap();
            let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(
                value.to_bits(),
                decoded.to_bits(),
                "{}",
                String::from_utf8_lossy(&encoded)
            );
        }
    }
}
