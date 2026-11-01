//! Generate a project-local execution profile from measured host capabilities.
//! This never activates an unverified model and never persists identity data.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Serialize)]
struct ExecutionProfile {
    schema: &'static str,
    generated_unix_seconds: u64,
    host: mslim::adapter_catalog::HostCapabilities,
    cpu_threads_for_runtime: Option<usize>,
    system_memory_reserve_bytes: Option<u64>,
    model_memory_budget_bytes: Option<u64>,
    kv_hot_budget_bytes: Option<u64>,
    cold_context_budget_bytes: Option<u64>,
    calibration_status: &'static str,
    accelerator_memory_budgets: Vec<AcceleratorBudget>,
    validated_bundles: Vec<ValidatedBundle>,
    activation: &'static str,
}

#[derive(Debug, Serialize)]
struct AcceleratorBudget {
    backend: String,
    name: String,
    reported_bytes: Option<u64>,
    safe_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ValidatedBundle {
    digest: String,
    path: String,
    adapter: String,
    status: &'static str,
    calibration: &'static str,
    calibrated_runtime_configuration: Option<Value>,
    calibration_environment: Option<Value>,
    validation_matrix_sha256: Option<String>,
}

struct CalibrationEvidence {
    status: &'static str,
    frozen_runtime_configuration: Option<Value>,
    calibration_environment: Option<Value>,
    validation_matrix_sha256: Option<String>,
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let host = mslim::adapter_catalog::host_capabilities();
    // Thread count is a measured calibration result, not a fixed subtraction
    // from the detected host parallelism.
    let cpu_threads_for_runtime = None;
    // These budgets must come from an uncached calibration run. Do not turn
    // arbitrary percentages into an apparent production guarantee.
    let system_memory_reserve_bytes = None;
    let model_memory_budget_bytes = None;
    let kv_hot_budget_bytes = None;
    let cold_context_budget_bytes = None;
    let accelerator_memory_budgets = host
        .accelerators
        .iter()
        .filter(|accelerator| accelerator.available)
        .map(|accelerator| AcceleratorBudget {
            backend: accelerator.backend.clone(),
            name: accelerator.name.clone(),
            reported_bytes: accelerator.memory_bytes,
            safe_bytes: accelerator
                .memory_bytes
                .map(|bytes| bytes.saturating_mul(3) / 4),
        })
        .collect();
    let validated_bundles = validated_bundles(&root, &host)?;
    let generated_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("reloj del sistema anterior a Unix epoch")?
        .as_secs();
    let calibration_ready = validated_bundles
        .iter()
        .all(|bundle| bundle.calibration == "available")
        && !validated_bundles.is_empty();
    let profile = ExecutionProfile {
        schema: "mslim.execution_profile/v1",
        generated_unix_seconds,
        host,
        cpu_threads_for_runtime,
        system_memory_reserve_bytes,
        model_memory_budget_bytes,
        kv_hot_budget_bytes,
        cold_context_budget_bytes,
        calibration_status: if calibration_ready {
            "available-for-recorded-runtime-configuration"
        } else {
            "required-before-configuration-readback"
        },
        accelerator_memory_budgets,
        activation: if calibration_ready {
            "requires-worker-configuration-readback-match"
        } else {
            "requires-calibration-and-worker-configuration-readback"
        },
        validated_bundles,
    };
    let destination = root.join("deploy/local-production/runtime/profiles/effective.json");
    fs::create_dir_all(
        destination
            .parent()
            .context("perfil sin directorio padre")?,
    )?;
    let temporary = tempfile::NamedTempFile::new_in(destination.parent().unwrap())?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&profile)?)?;
    temporary
        .persist(&destination)
        .map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    Ok(())
}

fn validated_bundles(
    root: &Path,
    host: &mslim::adapter_catalog::HostCapabilities,
) -> Result<Vec<ValidatedBundle>> {
    let models = root.join("deploy/local-production/runtime/models");
    let validation = root.join("deploy/local-production/runtime/validation");
    let calibration = root.join("deploy/local-production/runtime/calibration");
    let mut bundles = Vec::new();
    if !models.is_dir() {
        return Ok(bundles);
    }
    for entry in fs::read_dir(&models)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let digest = entry.file_name().to_string_lossy().into_owned();
        let receipt = validation.join(format!("{digest}.json"));
        let bytes = match fs::read(receipt) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let value: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get("project").and_then(serde_json::Value::as_str) != Some("KV-SHEI")
            || value.get("product").and_then(serde_json::Value::as_str) != Some("M-SLIM")
            || value
                .get("technical_namespace")
                .and_then(serde_json::Value::as_str)
                != Some("mslim")
            || value.get("schema").and_then(serde_json::Value::as_str)
                != Some(mslim::project_identity::VALIDATION_RECEIPT_SCHEMA)
            || value.get("status").and_then(serde_json::Value::as_str)
                != Some("validated-for-local-execution")
            || !mslim::build_info::is_current_clean_source_revision(value.get("source_revision"))
        {
            continue;
        }
        let Some(adapter) = value.get("adapter").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if !host.adapters.iter().any(|status| {
            status.descriptor.id == adapter
                && status.availability == mslim_adapter_api::AdapterAvailability::RuntimeAvailable
        }) {
            continue;
        }
        let calibration = calibration_evidence(&calibration, &entry.file_name());
        bundles.push(ValidatedBundle {
            digest,
            path: format!("runtime/models/{}", entry.file_name().to_string_lossy()),
            adapter: adapter.to_owned(),
            status: "validated-for-local-execution",
            calibration: calibration.status,
            calibrated_runtime_configuration: calibration.frozen_runtime_configuration,
            calibration_environment: calibration.calibration_environment,
            validation_matrix_sha256: calibration.validation_matrix_sha256,
        });
    }
    bundles.sort_by(|left, right| left.digest.cmp(&right.digest));
    Ok(bundles)
}

fn calibration_evidence(calibration_root: &Path, digest: &std::ffi::OsStr) -> CalibrationEvidence {
    let path = calibration_root.join(format!("{}.json", digest.to_string_lossy()));
    let Ok(bytes) = fs::read(path) else {
        return CalibrationEvidence {
            status: "missing",
            frozen_runtime_configuration: None,
            calibration_environment: None,
            validation_matrix_sha256: None,
        };
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return CalibrationEvidence {
            status: "invalid",
            frozen_runtime_configuration: None,
            calibration_environment: None,
            validation_matrix_sha256: None,
        };
    };
    let expected_bundle = digest.to_string_lossy();
    if mslim::calibration_contract::validate_calibration_receipt(&value, &expected_bundle).is_ok() {
        CalibrationEvidence {
            status: "available",
            frozen_runtime_configuration: value.get("frozen_runtime_configuration").cloned(),
            calibration_environment: value.get("calibration_environment").cloned(),
            validation_matrix_sha256: value
                .get("validation_matrix_sha256")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }
    } else {
        CalibrationEvidence {
            status: "invalid",
            frozen_runtime_configuration: None,
            calibration_environment: None,
            validation_matrix_sha256: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::calibration_evidence;
    use serde_json::json;
    use std::fs;

    fn canonical_receipt(bundle: &str) -> serde_json::Value {
        json!({
            "project": "KV-SHEI",
            "product": "M-SLIM",
            "technical_namespace": "mslim",
            "schema": mslim::project_identity::CALIBRATION_RECEIPT_SCHEMA,
            "source_revision": {
                "commit": mslim::build_info::SOURCE_COMMIT,
                "tracked_files_modified_at_build": false
            },
            "bundle": bundle,
            "adapter": "accelerator-runtime",
            "identity_digest": "b".repeat(64),
            "implementation_digest": mslim::calibration_profile::implementation_digest(),
            "calibration_environment": {
                "schema": mslim::calibration_contract::CALIBRATION_ENVIRONMENT_SCHEMA,
                "backend": "accelerator-runtime",
                "device_key": "d".repeat(64)
            },
            "started_unix_seconds": 1,
            "warmup_runs": 1,
            "measured_runs": 3,
            "decode_tokens_per_run": 8,
            "frozen_runtime_configuration": {
                "kv_mode": "exact-coldstore",
                "weight_mode": "resident",
                "max_context_tokens": 8192,
                "lod_epsilon": 0.0,
                "stage_tokens": 64,
                "prefill_batch_tokens": 128,
                "prefill_weight_reuse_tokens": 8,
                "hot_v_tokens": 512,
                "prompt_replay_tokens": 0,
                "shared_prefix": false,
                "micro_hot_v_tokens": 0,
                "deferred_cold_commit": false,
                "exact_hot_fused_attention": true,
                "decode_unroll4": false,
                "decode_fused_kernels": true,
                "prefill_fuse_projections": true,
                "prefill_profile_id": null
            },
            "prefill_execution_contract": {
                "kind": mslim::calibration_contract::TOKEN_ONLY_PREFILL_KIND,
                "adapter_path": mslim::calibration_contract::ACCELERATOR_TOKEN_ONLY_PREFILL_PATH,
                "diagnostic_full_vocabulary_logits_captured": false
            },
            "decode_execution_contract": {
                "kind": mslim::calibration_contract::TOKEN_ONLY_DECODE_KIND,
                "adapter_path": mslim::calibration_contract::ACCELERATOR_TOKEN_ONLY_PATH,
                "diagnostic_full_vocabulary_logits_captured": false
            },
            "context_samples": [{
                "prompt_tokens": 1,
                "prefill_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                "prefill_tokens_per_second_median": 1000000.0,
                "decode_tokens": 8,
                "decode_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                "decode_tokens_per_second_median": 8000000.0,
                "status": "measured"
            }],
            "snapshot_contract": {
                "kind": "physical-coldstore-handle",
                "physical_kv_persisted": true,
                "restore_latency_measured_as_physical_kv_hydration": false
            },
            "validation_matrix_gate": "same-backend-and-kv-integrity-passed",
            "validation_matrix_sha256": "f".repeat(64),
            "status": "measured-sustained-execution"
        })
    }

    #[test]
    fn autoconfigure_reuses_only_canonical_matching_calibration() {
        let root = tempfile::tempdir().unwrap();
        let bundle = "a".repeat(64);
        let path = root.path().join(format!("{bundle}.json"));
        fs::write(
            &path,
            serde_json::to_vec(&canonical_receipt(&bundle)).unwrap(),
        )
        .unwrap();
        let evidence = calibration_evidence(root.path(), std::ffi::OsStr::new(&bundle));
        assert_eq!(
            evidence.status,
            if mslim::build_info::current_build_is_clean() {
                "available"
            } else {
                "invalid"
            }
        );
        assert_eq!(
            evidence.frozen_runtime_configuration.is_some(),
            mslim::build_info::current_build_is_clean()
        );
        let mut legacy = canonical_receipt(&bundle);
        legacy["schema"] = json!("kvslim.calibration_receipt/v2");
        fs::write(path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(
            calibration_evidence(root.path(), std::ffi::OsStr::new(&bundle)).status,
            "invalid"
        );
    }
}
