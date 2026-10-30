//! Installs only artifacts declared by a KVSLIM release manifest.
//! Network access is opt-in and the core remains network-free by default.

use anyhow::{Context, Result, bail, ensure};
use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: String,
    artifacts: Vec<Artifact>,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    id: String,
    kind: String,
    source: String,
    destination: String,
    version: String,
    sha256: String,
    required: bool,
    license_confirmation_required: bool,
}

fn usage() -> ! {
    eprintln!("uso: kvslim-install-release <manifest.json> [--accept-licenses]");
    std::process::exit(2);
}

fn safe_destination(runtime: &Path, relative: &str) -> Result<PathBuf> {
    ensure!(!relative.is_empty() && !relative.starts_with('/') && !relative.contains(".."));
    let path = runtime.join(relative);
    ensure!(path.starts_with(runtime), "destino fuera de runtime");
    Ok(path)
}

fn download_verified(client: &Client, artifact: &Artifact, destination: &Path) -> Result<()> {
    ensure!(
        artifact.source.starts_with("https://"),
        "{} no usa HTTPS",
        artifact.id
    );
    ensure!(
        artifact.sha256.len() == 64 && artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "SHA-256 inválido para {}",
        artifact.id
    );
    let response = client.get(&artifact.source).send()?.error_for_status()?;
    let runtime = destination.parent().context("destino sin directorio")?;
    fs::create_dir_all(runtime)?;
    let temporary = tempfile::NamedTempFile::new_in(runtime)?;
    let mut output = temporary.reopen()?;
    let mut response = response;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
    }
    output.flush()?;
    let actual = hex::encode(hasher.finalize());
    ensure!(
        actual.eq_ignore_ascii_case(&artifact.sha256),
        "hash no coincide para {}: esperado {}, obtenido {}",
        artifact.id,
        artifact.sha256,
        actual
    );
    drop(output);
    temporary
        .persist(destination)
        .with_context(|| format!("activar {}", destination.display()))?;
    println!(
        "instalado {} v{} ({})",
        artifact.id, artifact.version, artifact.kind
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let manifest_path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let accept_licenses = args.any(|arg| arg == "--accept-licenses");
    let manifest: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    ensure!(
        manifest.schema == "kvslim.local_release/v1",
        "schema de release no soportado"
    );
    let project_root = manifest_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("el manifiesto debe estar dentro de deploy/local-production")?;
    let runtime = project_root.join("deploy/local-production/runtime");
    ensure!(
        runtime.starts_with(project_root),
        "runtime fuera del proyecto"
    );
    let client = Client::builder().https_only(true).build()?;

    for artifact in &manifest.artifacts {
        if artifact.license_confirmation_required && !accept_licenses {
            if artifact.required {
                bail!("{} requiere --accept-licenses", artifact.id);
            }
            println!("omitido {}: requiere confirmación de licencia", artifact.id);
            continue;
        }
        let destination = safe_destination(&runtime, &artifact.destination)?;
        if destination.exists() {
            let mut file = fs::File::open(&destination)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
            if hex::encode(hasher.finalize()).eq_ignore_ascii_case(&artifact.sha256) {
                println!("ya verificado {}", artifact.id);
                continue;
            }
            fs::remove_file(&destination)?;
        }
        if let Err(error) = download_verified(&client, artifact, &destination) {
            if artifact.required {
                return Err(error);
            }
            eprintln!("artefacto opcional {} no instalado: {error:#}", artifact.id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destinations_cannot_escape_runtime() {
        let root = tempfile::tempdir().unwrap();
        assert!(safe_destination(root.path(), "models/bundle/file").is_ok());
        assert!(safe_destination(root.path(), "../outside").is_err());
        assert!(safe_destination(root.path(), "/absolute").is_err());
    }

    #[test]
    fn manifest_artifact_requires_https_and_sha256_shape() {
        let artifact = Artifact {
            id: "test".into(),
            kind: "runtime".into(),
            source: "http://invalid".into(),
            destination: "runtimes/test.bin".into(),
            version: "1".into(),
            sha256: "00".into(),
            required: true,
            license_confirmation_required: false,
        };
        let client = Client::builder().build().unwrap();
        let root = tempfile::tempdir().unwrap();
        assert!(download_verified(&client, &artifact, &root.path().join("test.bin")).is_err());
    }
}
