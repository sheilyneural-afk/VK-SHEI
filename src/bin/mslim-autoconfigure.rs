//! Generate a project-local execution profile from measured host capabilities.
//! This never activates an unverified model and never persists identity data.

use anyhow::{Context, Result};
use serde::Serialize;
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
            "available-and-reusable-until-identity-changes"
        } else {
            "required-before-automatic-activation"
        },
        accelerator_memory_budgets,
        activation: if calibration_ready {
            "automatic-activation-allowed-with-cached-certification"
        } else {
            "requires-calibration-before-automatic-activation"
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
        bundles.push(ValidatedBundle {
            digest,
            path: format!("runtime/models/{}", entry.file_name().to_string_lossy()),
            adapter: adapter.to_owned(),
            status: "validated-for-local-execution",
            calibration: calibration_receipt_is_reusable(&calibration, &entry.file_name()),
        });
    }
    bundles.sort_by(|left, right| left.digest.cmp(&right.digest));
    Ok(bundles)
}

fn calibration_receipt_is_reusable(
    calibration_root: &Path,
    digest: &std::ffi::OsStr,
) -> &'static str {
    let path = calibration_root.join(format!("{}.json", digest.to_string_lossy()));
    let Ok(bytes) = fs::read(path) else {
        return "missing";
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "invalid";
    };
    let expected_bundle = digest.to_string_lossy();
    if mslim::calibration_contract::validate_calibration_receipt(&value, &expected_bundle).is_ok() {
        "available"
    } else {
        "invalid"
    }
}

#[cfg(test)]
mod tests {
    use super::calibration_receipt_is_reusable;
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
            "started_unix_seconds": 1,
            "warmup_runs": 1,
            "measured_runs": 3,
            "decode_tokens_per_run": 8,
            "context_samples": [{
                "prompt_tokens": 1,
                "prefill_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                "decode_tokens": 8,
                "decode_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                "status": "measured"
            }],
            "snapshot_contract": {
                "kind": "physical-coldstore-handle",
                "physical_kv_persisted": true,
                "restore_latency_measured_as_physical_kv_hydration": false
            },
            "validation_matrix_gate": "same-backend-and-kv-integrity-passed",
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
        assert_eq!(
            calibration_receipt_is_reusable(root.path(), std::ffi::OsStr::new(&bundle)),
            if mslim::build_info::current_build_is_clean() {
                "available"
            } else {
                "invalid"
            }
        );
        let mut legacy = canonical_receipt(&bundle);
        legacy["schema"] = json!("kvslim.calibration_receipt/v2");
        fs::write(path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(
            calibration_receipt_is_reusable(root.path(), std::ffi::OsStr::new(&bundle)),
            "invalid"
        );
    }
}
