//! Imports a user-owned model bundle into the project-local model store.

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
struct BundleManifest {
    schema: &'static str,
    bundle_sha256: String,
    files: Vec<String>,
}

fn collect_files(root: &Path, current: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            bail!("el bundle contiene un enlace simbólico: {}", path.display());
        }
        if file_type.is_dir() {
            collect_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path.strip_prefix(root)?.to_owned();
            if relative != Path::new("bundle-manifest.json") {
                files.push(relative);
            }
        } else {
            bail!("tipo de archivo no soportado: {}", path.display());
        }
    }
    Ok(())
}

fn digest_bundle(source: &Path, files: &[PathBuf]) -> Result<String> {
    let mut hasher = Sha256::new();
    for relative in files {
        let relative_text = relative.to_string_lossy().replace('\\', "/");
        hasher.update(relative_text.as_bytes());
        hasher.update([0]);
        hasher.update(fs::read(source.join(relative))?);
        hasher.update([0]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn copy_bundle(source: &Path, destination: &Path, files: &[PathBuf]) -> Result<()> {
    for relative in files {
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source.join(relative), target)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let source = PathBuf::from(env::args().nth(1).unwrap_or_else(|| {
        eprintln!("uso: mslim-import-model <directorio-del-modelo>");
        std::process::exit(2);
    }));
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models_root = root.join("deploy/local-production/runtime/models");
    let runtime_root = root.join("deploy/local-production/runtime");
    let source = fs::canonicalize(&source).with_context(|| format!("leer {}", source.display()))?;
    ensure!(source.is_dir(), "el origen no es un directorio");
    if source.starts_with(&models_root) {
        bail!("el origen ya está dentro del almacén de modelos");
    }
    mslim::model_config::ModelSpec::inspect(&source)
        .context("el bundle no contiene una configuración de modelo válida")?;
    let mut files = Vec::new();
    collect_files(&source, &source, &mut files)?;
    files.sort();
    ensure!(!files.is_empty(), "el bundle está vacío");
    let digest = digest_bundle(&source, &files)?;
    fs::create_dir_all(&models_root)?;
    fs::create_dir_all(runtime_root.join("tmp"))?;
    let temporary = tempfile::Builder::new()
        .prefix("model-import-")
        .tempdir_in(runtime_root.join("tmp"))?;
    copy_bundle(&source, temporary.path(), &files)?;
    let manifest = BundleManifest {
        schema: mslim::model_bundle::MODEL_BUNDLE_MANIFEST_SCHEMA,
        bundle_sha256: digest.clone(),
        files: files
            .iter()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .collect(),
    };
    let mut manifest_file = fs::File::create(temporary.path().join("bundle-manifest.json"))?;
    manifest_file.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    manifest_file.write_all(b"\n")?;
    let destination = models_root.join(&digest);
    if destination.exists() {
        println!("ya existe el bundle verificado: runtime/models/{digest}");
        return Ok(());
    }
    let temporary_path = temporary.keep();
    fs::rename(&temporary_path, &destination)
        .with_context(|| format!("activar runtime/models/{digest}"))?;
    println!("modelo importado: runtime/models/{digest}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic_and_manifest_is_not_authoritative() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("nested")).unwrap();
        fs::write(root.path().join("config.json"), b"config").unwrap();
        fs::write(root.path().join("nested/weights.bin"), b"weights").unwrap();
        fs::write(root.path().join("bundle-manifest.json"), b"old").unwrap();
        let mut files = Vec::new();
        collect_files(root.path(), root.path(), &mut files).unwrap();
        files.sort();
        let first = digest_bundle(root.path(), &files).unwrap();
        fs::write(root.path().join("bundle-manifest.json"), b"new").unwrap();
        let second = digest_bundle(root.path(), &files).unwrap();
        assert_eq!(first, second);
        fs::write(root.path().join("nested/weights.bin"), b"changed").unwrap();
        let third = digest_bundle(root.path(), &files).unwrap();
        assert_ne!(first, third);
    }
}
