//! Inventario local de bundles de modelos y recomendación fail-closed.
//! Solo inspecciona runtime/models dentro del proyecto.

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
struct ModelEntry {
    bundle: String,
    path: String,
    inspectable: bool,
    bundle_integrity: String,
    model_type: Option<String>,
    architectures: Vec<String>,
    runtime_status: Vec<String>,
    executable_adapters: Vec<String>,
    validation_status: String,
    recommendation: String,
    error: Option<String>,
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = root.join("deploy/local-production/runtime/models");
    let mut entries = Vec::new();
    if models.is_dir() {
        for item in fs::read_dir(&models).with_context(|| format!("leer {}", models.display()))? {
            let item = item?;
            if !item.file_type()?.is_dir() {
                continue;
            }
            let bundle_path = item.path();
            let bundle = item.file_name().to_string_lossy().into_owned();
            let integrity = bundle_integrity(&bundle_path, &bundle);
            let validation_status = validation_status(&root, &bundle);
            match mslim::model_config::ModelSpec::inspect(&bundle_path) {
                Ok(spec) => {
                    let executable_adapters = if integrity == "verified" {
                        match mslim::adapter_catalog::select_available_for_model(
                            &bundle_path,
                            None,
                            None,
                        ) {
                            Ok(selection) => vec![selection.adapter_id],
                            Err(_) => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };
                    let recommendation = if executable_adapters.is_empty() {
                        "inspeccionable; no hay un runtime validado para ejecutarlo ahora"
                    } else {
                        "ejecutable con el adaptador validado indicado"
                    };
                    entries.push(ModelEntry {
                        bundle,
                        path: format!("runtime/models/{}", item.file_name().to_string_lossy()),
                        inspectable: true,
                        bundle_integrity: integrity.clone(),
                        model_type: Some(spec.model_type),
                        architectures: spec.architectures,
                        runtime_status: mslim::adapter_catalog::adapter_status()
                            .into_iter()
                            .map(|status| {
                                format!("{}: {:?}", status.descriptor.id, status.availability)
                            })
                            .collect(),
                        executable_adapters,
                        validation_status: validation_status.clone(),
                        recommendation: if integrity == "verified" {
                            recommendation.into()
                        } else {
                            "solo inspección; importar de nuevo para generar un bundle verificado"
                                .into()
                        },
                        error: None,
                    });
                }
                Err(error) => entries.push(ModelEntry {
                    bundle,
                    path: format!("runtime/models/{}", item.file_name().to_string_lossy()),
                    inspectable: false,
                    bundle_integrity: integrity,
                    model_type: None,
                    architectures: Vec::new(),
                    runtime_status: Vec::new(),
                    executable_adapters: Vec::new(),
                    validation_status,
                    recommendation: "rechazado para ejecución; corregir el bundle o su manifiesto"
                        .into(),
                    error: Some(error.to_string()),
                }),
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&entries)?);
    Ok(())
}

fn validation_status(root: &Path, bundle: &str) -> String {
    let receipt = root
        .join("deploy/local-production/runtime/validation")
        .join(format!("{bundle}.json"));
    let Ok(bytes) = fs::read(receipt) else {
        return "not-validated".into();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "invalid-receipt".into();
    };
    if value.get("project").and_then(serde_json::Value::as_str) == Some("KV-SHEI")
        && value.get("product").and_then(serde_json::Value::as_str) == Some("M-SLIM")
        && value
            .get("technical_namespace")
            .and_then(serde_json::Value::as_str)
            == Some("mslim")
        && value.get("schema").and_then(serde_json::Value::as_str)
            == Some(mslim::project_identity::VALIDATION_RECEIPT_SCHEMA)
        && value
            .get("bundle_sha256")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|digest| digest.eq_ignore_ascii_case(bundle))
        && value.get("status").and_then(serde_json::Value::as_str)
            == Some("validated-for-local-execution")
    {
        "validated-for-local-execution".into()
    } else {
        "invalid-receipt".into()
    }
}

fn bundle_integrity(path: &std::path::Path, directory_name: &str) -> String {
    if directory_name.len() != 64 || !directory_name.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return "invalid-directory-digest".into();
    }
    if mslim::model_bundle::validated_bundle_manifest(path).is_err() {
        return "manifest-mismatch".into();
    }
    let mut files = Vec::new();
    if collect_bundle_files(path, path, &mut files).is_err() {
        return "unreadable-bundle".into();
    }
    files.sort();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for relative in files {
        let name = relative.to_string_lossy().replace('\\', "/");
        let mut file = match fs::File::open(path.join(&relative)) {
            Ok(file) => file,
            Err(_) => return "unreadable-bundle".into(),
        };
        hasher.update(name.as_bytes());
        hasher.update([0]);
        loop {
            let read = match file.read(&mut buffer) {
                Ok(read) => read,
                Err(_) => return "unreadable-bundle".into(),
            };
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        hasher.update([0]);
    }
    if hex::encode(hasher.finalize()).eq_ignore_ascii_case(directory_name) {
        "verified".into()
    } else {
        "content-mismatch".into()
    }
}

fn collect_bundle_files(
    root: &Path,
    current: &Path,
    files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(std::io::Error::other("bundle contains symlink"));
        }
        if file_type.is_dir() {
            collect_bundle_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(std::io::Error::other)?
                .to_owned();
            if relative != Path::new("bundle-manifest.json") {
                files.push(relative);
            }
        }
    }
    Ok(())
}
