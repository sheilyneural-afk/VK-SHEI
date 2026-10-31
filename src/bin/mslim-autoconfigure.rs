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
        if value.get("schema").and_then(serde_json::Value::as_str)
            != Some("mslim.validation_receipt/v1")
            || value.get("status").and_then(serde_json::Value::as_str)
                != Some("validated-for-local-execution")
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
    if value.get("schema").and_then(serde_json::Value::as_str)
        == Some("mslim.calibration_receipt/v1")
        && value.get("status").and_then(serde_json::Value::as_str)
            == Some("measured-basic-execution")
        && value
            .get("identity_digest")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|digest| !digest.is_empty())
    {
        "available"
    } else {
        "invalid"
    }
}
