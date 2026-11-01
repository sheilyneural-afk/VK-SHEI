//! Narrow prefill profiles layered on CalibrationProfile/CalibrationTransaction.
//! No second registry or activation state machine. Only batch size, weight
//! reuse and a bit-exact attention kernel configuration are tunable; identity,
//! fidelity and measured coverage remain immutable.

use crate::calibration_transaction::{
    CalibrationProfile, pending_profile_path, verify_receipt_chain,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, sync::OnceLock};

pub fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillSettings {
    pub batch_tokens: Option<usize>,
    pub weight_reuse_tokens: usize,
    #[serde(default)]
    pub attention: crate::attention_kernel::AttentionKernelConfig,
}

impl PrefillSettings {
    pub fn validate(&self) -> Result<()> {
        self.attention.validate_activation(false)?;
        ensure!(
            self.batch_tokens.is_none_or(|n| (1..=512).contains(&n)),
            "prefill batch must be 1..=512"
        );
        ensure!(
            [1, 8, 16, 32].contains(&self.weight_reuse_tokens),
            "prefill reuse must be 1,8,16,32"
        );
        Ok(())
    }
    pub fn id(&self) -> String {
        let base = format!(
            "batch-{}-reuse-{}",
            self.batch_tokens
                .map_or("sequential".into(), |n| n.to_string()),
            self.weight_reuse_tokens
        );
        if self.attention == Default::default() {
            base
        } else {
            format!("{base}-{}", self.attention.id())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillEnvironment {
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub backend: String,
    pub hardware_digest: String,
    pub execution_digest: String,
    pub implementation_digest: String,
}

impl PrefillEnvironment {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.backend.is_empty(), "missing prefill backend");
        for digest in [
            &self.model_digest,
            &self.tokenizer_digest,
            &self.hardware_digest,
            &self.execution_digest,
            &self.implementation_digest,
        ] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid prefill identity digest"
            );
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256(&serde_json::to_vec(self)?))
    }
}

/// Source/build identity is computed once, outside the inference hot path.
/// A new binary/profile must be recalibrated when relevant implementation changes.
pub fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut hash = Sha256::new();
        for source in [
            include_str!("gpu_llm/mod.rs"),
            include_str!("gpu_llm/kernels.rs"),
            include_str!("gpu_llm/gpu_resources.rs"),
            include_str!("gpu_llm/prefill.rs"),
            include_str!("gpu_llm/tiled_attention.rs"),
            include_str!("attention_kernel.rs"),
            include_str!("../assets/tiled_attention.cl"),
            include_str!("../assets/gpu_llm.cl"),
            include_str!("gpu_region_store.rs"),
            include_str!("gpu_micro_regions.rs"),
            include_str!("adaptive_extent.rs"),
            include_str!("attention_certificate.rs"),
            include_str!("cold_store.rs"),
            include_str!("calibration_profile.rs"),
            include_str!("calibration_transaction.rs"),
            include_str!("calibration_predictor.rs"),
            include_str!("prefill_calibration.rs"),
            include_str!("../Cargo.lock"),
            include_str!("../Cargo.toml"),
        ] {
            hash.update((source.len() as u64).to_le_bytes());
            hash.update(source.as_bytes());
        }
        hash.update(env!("CARGO_PKG_VERSION").as_bytes());
        hash.update(std::env::consts::ARCH.as_bytes());
        hash.update(std::env::consts::OS.as_bytes());
        hash.update([
            cfg!(debug_assertions) as u8,
            cfg!(feature = "hip-backend") as u8,
        ]);
        hex::encode(hash.finalize())
    })
}

pub fn host_identity() -> Value {
    // Do not hash free RAM, wall clocks, CPU MHz or other changing observations.
    let host = crate::adapter_catalog::host_capabilities();
    let cpu = fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("model name") || l.starts_with("Hardware"))
            .map(str::to_owned)
    });
    json!({"architecture": host.architecture, "os": std::env::consts::OS,
        "parallelism": host.cpu_parallelism, "memory_total_bytes": host.memory_total_bytes,
        "simd": host.simd, "cpu": cpu,
        "kernel": fs::read_to_string("/proc/sys/kernel/osrelease").ok()})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillProfileParameters {
    pub environment: PrefillEnvironment,
    pub settings: PrefillSettings,
    pub validated_context_tokens: Vec<usize>,
    pub prompt_suite_sha256: String,
}

impl PrefillProfileParameters {
    pub fn validate(&self) -> Result<()> {
        self.environment.validate()?;
        self.settings.validate()?;
        ensure!(
            (1..=16).contains(&self.validated_context_tokens.len()),
            "invalid profile coverage"
        );
        ensure!(
            self.validated_context_tokens
                .iter()
                .all(|n| (1..=32768).contains(n))
                && self
                    .validated_context_tokens
                    .windows(2)
                    .all(|w| w[0] < w[1]),
            "profile coverage must be positive, sorted and unique"
        );
        ensure!(
            self.prompt_suite_sha256.len() == 64
                && self
                    .prompt_suite_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "invalid prompt suite digest"
        );
        Ok(())
    }
    pub fn into_profile(self) -> Result<CalibrationProfile> {
        self.validate()?;
        let profile = CalibrationProfile::new(
            self.settings.id(),
            self.environment.model_digest.clone(),
            self.environment.backend.clone(),
            BTreeMap::from([("prefill".into(), serde_json::to_value(self)?)]),
        )?;
        Ok(profile)
    }
}

pub fn parameters(
    profile: &CalibrationProfile,
    environment: &PrefillEnvironment,
) -> Result<PrefillProfileParameters> {
    profile.validate()?;
    ensure!(
        profile.parameters.len() == 1 && profile.parameters.contains_key("prefill"),
        "profile contains parameters outside the prefill allowlist"
    );
    let params: PrefillProfileParameters =
        serde_json::from_value(profile.parameters["prefill"].clone())?;
    params.validate()?;
    ensure!(
        &params.environment == environment,
        "profile belongs to another model, host, implementation or execution configuration"
    );
    ensure!(
        profile.model_identity == environment.model_digest
            && profile.backend_identity == environment.backend,
        "profile model/backend identity mismatch"
    );
    ensure!(
        profile.profile_id == params.settings.id(),
        "profile ID does not match settings"
    );
    Ok(params)
}

/// Explicit opt-in only. A pending or unpromoted file cannot configure production.
/// SHA chains detect corruption; they are NOT signatures against a malicious owner.
pub fn read_promoted_profile(
    path: &Path,
    environment: &PrefillEnvironment,
) -> Result<PrefillProfileParameters> {
    ensure!(
        !pending_profile_path(path).exists(),
        "prefill profile has an unfinished canary; recovery required"
    );
    ensure!(
        !path.with_extension("transaction.lock").exists(),
        "prefill profile is being modified"
    );
    let metadata = fs::metadata(path)?;
    ensure!(metadata.len() <= 64 * 1024, "prefill profile is too large");
    let bytes = fs::read(path)?;
    let profile: CalibrationProfile = serde_json::from_slice(&bytes)?;
    let params = parameters(&profile, environment)?;
    let parent = path.parent().context("prefill profile has no parent")?;
    let receipts = verify_receipt_chain(parent.join("calibration-receipts.jsonl"))?;
    let digest = sha256(&bytes);
    let activation = receipts
        .iter()
        .rev()
        .find(|receipt| {
            matches!(
                receipt.event.as_str(),
                "promotion_committed" | "rollback_verified"
            )
        })
        .context("prefill profile has no activation evidence")?;
    let authorized_digest = match activation.event.as_str() {
        "promotion_committed" => activation
            .details
            .get("candidate_digest")
            .and_then(Value::as_str),
        _ => activation
            .details
            .get("restored_digest")
            .and_then(Value::as_str),
    };
    ensure!(
        authorized_digest == Some(digest.as_str()),
        "prefill profile is not the authorized activation"
    );
    // A rollback restores the previous state, so it keeps a previously promoted
    // profile loadable. It must not turn a profile that was never promoted
    // (for example a hand-written baseline) into an authorized one.
    if activation.event == "rollback_verified" {
        ensure!(
            receipts.iter().any(|receipt| {
                receipt.event == "promotion_committed"
                    && receipt
                        .details
                        .get("candidate_digest")
                        .and_then(Value::as_str)
                        == Some(digest.as_str())
            }),
            "prefill profile was restored by a rollback but was never promoted"
        );
    }
    // Recheck after reading the chain to reject a concurrently started activation.
    ensure!(
        !pending_profile_path(path).exists()
            && !path.with_extension("transaction.lock").exists()
            && fs::read(path)? == bytes,
        "prefill profile changed while reading"
    );
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn env() -> PrefillEnvironment {
        PrefillEnvironment {
            model_digest: "a".repeat(64),
            tokenizer_digest: "b".repeat(64),
            backend: "test".into(),
            hardware_digest: "c".repeat(64),
            execution_digest: "d".repeat(64),
            implementation_digest: "e".repeat(64),
        }
    }
    fn profile() -> CalibrationProfile {
        PrefillProfileParameters {
            environment: env(),
            settings: PrefillSettings {
                batch_tokens: Some(64),
                weight_reuse_tokens: 8,
                attention: Default::default(),
            },
            validated_context_tokens: vec![16, 32, 64],
            prompt_suite_sha256: "f".repeat(64),
        }
        .into_profile()
        .unwrap()
    }
    #[test]
    fn only_prefill_settings_can_be_carried() {
        let mut p = profile();
        assert!(parameters(&p, &env()).is_ok());
        p.parameters.insert("lod_epsilon".into(), json!(0.1));
        assert!(parameters(&p, &env()).is_err());
    }
    #[test]
    fn stale_environment_and_unknown_fields_are_rejected() {
        let mut e = env();
        e.hardware_digest = "f".repeat(64);
        assert!(parameters(&profile(), &e).is_err());
        let mut p = profile();
        p.parameters.get_mut("prefill").unwrap()["settings"]["precision"] = json!("q4");
        assert!(parameters(&p, &env()).is_err());
    }
    #[test]
    fn unpromoted_and_pending_profiles_cannot_activate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("active.json");
        fs::write(&path, serde_json::to_vec_pretty(&profile()).unwrap()).unwrap();
        assert!(read_promoted_profile(&path, &env()).is_err());
        fs::write(pending_profile_path(&path), b"{}").unwrap();
        assert!(
            read_promoted_profile(&path, &env())
                .unwrap_err()
                .to_string()
                .contains("unfinished")
        );
    }
    #[test]
    fn profile_dimensions_and_reuse_are_bounded() {
        assert!(
            PrefillSettings {
                batch_tokens: Some(513),
                weight_reuse_tokens: 8,
                attention: Default::default(),
            }
            .validate()
            .is_err()
        );
        assert!(
            PrefillSettings {
                batch_tokens: Some(64),
                weight_reuse_tokens: 7,
                attention: Default::default(),
            }
            .validate()
            .is_err()
        );
    }
}
