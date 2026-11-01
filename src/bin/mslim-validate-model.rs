//! Real local model/adaptor gate: inspect, load, prefill, decode and KV replay.

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

fn main() -> Result<()> {
    let bundle_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("uso: mslim-validate-model <runtime/models/<digest>>")?,
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundle_path = if bundle_path.is_absolute() {
        bundle_path
    } else {
        root.join(bundle_path)
    };
    let config_path = bundle_path.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path)?)
        .with_context(|| format!("config.json inválido: {}", config_path.display()))?;
    let bundle = mslim::model_bundle::ModelBundle::load(&bundle_path)?;
    enforce_profile_budget(&root, &bundle.root)?;
    let selection = mslim::adapter_catalog::select_available_for_model(&bundle.root, None, None)?;
    let coldstore = if selection.adapter_id == "accelerator-runtime" {
        let temporary_root = root.join("deploy/local-production/runtime/tmp");
        fs::create_dir_all(&temporary_root)?;
        let coldstore = tempfile::Builder::new()
            .prefix("validation-coldstore-")
            .tempdir_in(&temporary_root)?;
        config["mslim_kv_mode"] = Value::String("exact-coldstore".into());
        config["mslim_lod_epsilon"] = Value::from(0.0);
        config["mslim_cold_directory"] =
            Value::String(coldstore.path().to_string_lossy().into_owned());
        Some(coldstore)
    } else {
        None
    };
    let adapter = mslim::adapter_catalog::create_adapter(&selection.adapter_id)?;
    let started = Instant::now();
    let conformance = mslim::conformance::run_adapter_conformance(
        adapter,
        &bundle.root,
        &config,
        &bundle.probe_token_ids()?,
    )?;
    let bundle_digest = bundle
        .root
        .file_name()
        .and_then(|name| name.to_str())
        .context("el bundle debe estar bajo un directorio con digest")?;
    ensure!(
        bundle_digest.len() == 64 && bundle_digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "digest de bundle inválido"
    );
    let validation_root = root.join("deploy/local-production/runtime/validation");
    fs::create_dir_all(&validation_root)?;
    let receipt = serde_json::json!({
        "project": mslim::project_identity::PROJECT,
        "product": mslim::project_identity::PRODUCT,
        "technical_namespace": mslim::project_identity::TECHNICAL_NAMESPACE,
        "schema": mslim::project_identity::VALIDATION_RECEIPT_SCHEMA,
        "source_revision": mslim::build_info::source_revision(),
        "bundle_sha256": bundle_digest,
        "adapter": selection.adapter_id,
        "checks": ["inspect", "load", "prefill", "decode", "kv_snapshot", "kv_restore"],
        "conformance": conformance.clone(),
        "runtime_configuration": {
            "kv_mode": if selection.adapter_id == "accelerator-runtime" {"exact-coldstore"} else {"adapter-default"},
            "epsilon": 0.0,
            "coldstore_scope": if coldstore.is_some() {"temporary-validation-namespace"} else {"not-required"}
        },
        "elapsed_millis": started.elapsed().as_millis(),
        "status": "validated-for-local-execution"
    });
    let receipt_path = validation_root.join(format!("{bundle_digest}.json"));
    let temporary_receipt = tempfile::NamedTempFile::new_in(&validation_root)?;
    fs::write(
        temporary_receipt.path(),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    temporary_receipt
        .persist(&receipt_path)
        .map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&conformance)?);
    Ok(())
}

fn enforce_profile_budget(root: &Path, bundle: &Path) -> Result<()> {
    let profile_path = root.join("deploy/local-production/runtime/profiles/effective.json");
    let Ok(bytes) = fs::read(&profile_path) else {
        return Ok(());
    };
    let profile: Value = serde_json::from_slice(&bytes).context("perfil de ejecución inválido")?;
    let Some(budget) = profile
        .get("model_memory_budget_bytes")
        .and_then(Value::as_u64)
    else {
        return Ok(());
    };
    let weight_bytes = fs::read_dir(bundle)?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    ensure!(
        weight_bytes <= budget,
        "los pesos del bundle ({weight_bytes} bytes) superan el presupuesto del perfil ({budget} bytes)"
    );
    Ok(())
}
