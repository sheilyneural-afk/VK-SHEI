//! Real local model/adaptor gate: inspect, load, prefill, decode and KV replay.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
struct ValidationReport {
    bundle: String,
    adapter: String,
    inspect: bool,
    load: bool,
    prefill: bool,
    decode: bool,
    kv_snapshot: bool,
    kv_restore: bool,
    status: &'static str,
}

fn main() -> Result<()> {
    let bundle = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("uso: kvslim-validate-model <runtime/models/<digest>>")?,
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundle = if bundle.is_absolute() {
        bundle
    } else {
        root.join(bundle)
    };
    let config_path = bundle.join("config.json");
    let config: Value = serde_json::from_slice(&fs::read(&config_path)?)
        .with_context(|| format!("config.json inválido: {}", config_path.display()))?;
    enforce_profile_budget(&root, &bundle)?;
    let selection = kvslim::adapter_catalog::select_available_for_model(&bundle, None, None)?;
    let mut adapter = kvslim::adapter_catalog::create_adapter(&selection.adapter_id)?;
    adapter.inspect(&bundle, &config)?;
    adapter.load(&bundle, &config)?;
    let prefill = adapter.prefill(&[0])?;
    ensure!(
        prefill.is_object(),
        "prefill no devolvió un resultado estructurado"
    );
    adapter.decode(0)?;
    let snapshot = adapter.snapshot_kv()?;
    ensure!(!snapshot.is_empty(), "snapshot KV vacío");
    adapter.restore_kv(&snapshot)?;
    let bundle_digest = bundle
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
        "schema": "kvslim.validation_receipt/v1",
        "bundle_sha256": bundle_digest,
        "adapter": selection.adapter_id,
        "checks": ["inspect", "load", "prefill", "decode", "kv_snapshot", "kv_restore"],
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
    let report = ValidationReport {
        bundle: bundle.display().to_string(),
        adapter: selection.adapter_id,
        inspect: true,
        load: true,
        prefill: true,
        decode: true,
        kv_snapshot: true,
        kv_restore: true,
        status: "validated-for-local-execution",
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
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
