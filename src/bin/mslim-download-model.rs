//! Download a model repository into the project-local, content-addressed store.
//! This is explicit: it never downloads or executes arbitrary files.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

const API: &str = "https://huggingface.co/api";
const DEFAULT_MAX_BYTES: u64 = 20 * 1024 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct Repository {
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    siblings: Vec<FileEntry>,
}

#[derive(Debug, Deserialize)]
struct FileEntry {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .context("uso: mslim-download-model [--auto [tarea]|<org/model>] [revision] [max-bytes]")?;
    if first == "--auto" {
        let query = args.next().unwrap_or_else(|| "local-inference".into());
        let max_bytes = args
            .next()
            .map(|value| value.parse())
            .transpose()
            .context("max-bytes inválido")?
            .unwrap_or(DEFAULT_MAX_BYTES);
        let host = mslim::adapter_catalog::host_capabilities();
        let search_query = hardware_search_query(&host, &query);
        let candidates = mslim::model_catalog::huggingface::search_with_details(&search_query, 50)?;
        let recommendation = mslim::model_resolver::rank_candidates(&host, candidates)
            .into_iter()
            .find(|item| {
                item.state == mslim::model_resolver::RecommendationState::RequiresLocalValidation
            })
            .context("no hay un candidato con adaptador local validado")?;
        eprintln!(
            "candidato seleccionado automáticamente: {}",
            recommendation.candidate.model_id
        );
        return download(
            &recommendation.candidate.model_id,
            recommendation
                .candidate
                .revision
                .as_deref()
                .unwrap_or("main"),
            max_bytes,
        );
    }
    let model_id = first;
    let revision = args.next().unwrap_or_else(|| "main".into());
    let max_bytes = args
        .next()
        .map(|value| value.parse())
        .transpose()
        .context("max-bytes inválido")?
        .unwrap_or(DEFAULT_MAX_BYTES);
    download(&model_id, &revision, max_bytes)
}

fn hardware_search_query(host: &mslim::adapter_catalog::HostCapabilities, task: &str) -> String {
    let model_types = host
        .adapters
        .iter()
        .filter(|status| {
            status.availability == mslim_adapter_api::AdapterAvailability::RuntimeAvailable
        })
        .flat_map(|status| status.descriptor.supported_model_types.iter())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    let task_term = if task != "local-inference" { task } else { "" };
    if model_types.is_empty() {
        format!("instruct {task_term}").trim().to_owned()
    } else {
        format!("instruct {task_term} {model_types}")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn download(model_id: &str, revision: &str, max_bytes: u64) -> Result<()> {
    validate_id(model_id)?;
    validate_id(revision)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = root.join("deploy/local-production/runtime");
    let models = runtime.join("models");
    let tmp_root = runtime.join("tmp");
    fs::create_dir_all(&models)?;
    fs::create_dir_all(&tmp_root)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30 * 60))
        .user_agent("M-SLIM-model-downloader/0.4")
        .build()?;
    let mut request = client.get(format!("{API}/models/{model_id}?full=true&blobs=true"));
    if let Some(token) = local_token(&runtime) {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .context("consultar repositorio de Hugging Face")?;
    ensure!(
        response.status().is_success(),
        "Hugging Face respondió HTTP {}",
        response.status()
    );
    let repository: Repository = response
        .json()
        .context("decodificar metadatos del modelo")?;
    let gated = repository.private
        || repository.gated == serde_json::Value::Bool(true)
        || repository.gated == serde_json::Value::String("manual".into());
    if gated && !has_local_token(&runtime) {
        ensure!(
            false,
            "repositorio privado o gated: configure un token de lectura"
        );
    }
    let files = repository
        .siblings
        .into_iter()
        .filter(|file| allowed_file(&file.rfilename))
        .collect::<Vec<_>>();
    ensure!(
        files.iter().any(|file| file.rfilename == "config.json"),
        "el repositorio no contiene config.json"
    );
    ensure!(
        files.iter().any(|file| is_weight(&file.rfilename)),
        "no se encontró un archivo de pesos admitido"
    );
    let total = files
        .iter()
        .filter_map(|file| file.size)
        .try_fold(0u64, u64::checked_add)
        .context("tamaño declarado fuera de rango")?;
    ensure!(
        total <= max_bytes,
        "el modelo declara {total} bytes y supera el límite de {max_bytes}"
    );
    let temporary = tempfile::Builder::new()
        .prefix("model-download-")
        .tempdir_in(&tmp_root)?;
    for file in &files {
        let destination = safe_join(temporary.path(), &file.rfilename)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut request = client.get(format!(
            "https://huggingface.co/{model_id}/resolve/{revision}/{}?download=true",
            file.rfilename
        ));
        if let Some(token) = local_token(&runtime) {
            request = request.bearer_auth(token);
        }
        let bytes = download_file_with_retries(&client, request, &file.rfilename)?;
        if let Some(size) = file.size {
            ensure!(
                bytes.len() as u64 == size,
                "tamaño inesperado para {}",
                file.rfilename
            );
        }
        fs::File::create(&destination)?.write_all(&bytes)?;
    }
    let mut entries = Vec::new();
    collect_files(temporary.path(), temporary.path(), &mut entries)?;
    entries.sort();
    let digest = digest_bundle(temporary.path(), &entries)?;
    let manifest = serde_json::json!({"schema":"mslim.model_bundle/v1","bundle_sha256":digest,"files":entries.iter().map(|p| p.to_string_lossy().replace('\\', "/")).collect::<Vec<_>>(),"source":{"provider":"huggingface","model_id":model_id,"revision":revision}});
    fs::write(
        temporary.path().join("bundle-manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let destination = models.join(&digest);
    if destination.exists() {
        println!("ya existe runtime/models/{digest}");
        return Ok(());
    }
    let temporary_path = temporary.keep();
    fs::rename(temporary_path, &destination)
        .with_context(|| format!("activar runtime/models/{digest}"))?;
    println!("modelo descargado y guardado en runtime/models/{digest}");
    Ok(())
}

fn download_file_with_retries(
    client: &reqwest::blocking::Client,
    request: reqwest::blocking::RequestBuilder,
    filename: &str,
) -> Result<Vec<u8>> {
    let url = request
        .try_clone()
        .context("no se pudo preparar la descarga repetible")?
        .build()
        .context("preparar descarga")?
        .url()
        .clone();
    let bearer = request
        .try_clone()
        .context("no se pudo conservar la credencial de descarga")?
        .build()
        .context("preparar credencial")?
        .headers()
        .get(reqwest::header::AUTHORIZATION)
        .cloned();
    let mut last_error = None;
    for attempt in 1..=3 {
        let mut retry = client.get(url.clone());
        if let Some(value) = bearer.clone() {
            retry = retry.header(reqwest::header::AUTHORIZATION, value);
        }
        match retry
            .send()
            .and_then(|response| response.error_for_status())
        {
            Ok(response) => match response.bytes() {
                Ok(bytes) => return Ok(bytes.to_vec()),
                Err(error) => last_error = Some(anyhow::anyhow!(error)),
            },
            Err(error) => last_error = Some(anyhow::anyhow!(error)),
        }
        eprintln!("reintento {attempt}/3 para {filename}");
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("descarga fallida")))
        .with_context(|| format!("leer {filename}"))
}

fn validate_id(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && !value.contains("..")
            && !value.starts_with('/')
            && !value.chars().any(char::is_control),
        "identificador inválido"
    );
    Ok(())
}

fn local_token(runtime: &Path) -> Option<String> {
    fs::read_to_string(runtime.join("secrets/huggingface.token"))
        .ok()
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
        .or_else(|| {
            std::env::var("HF_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())
        })
}

fn has_local_token(runtime: &Path) -> bool {
    local_token(runtime).is_some()
}
fn is_weight(path: &str) -> bool {
    [".safetensors", ".bin", ".gguf", ".ggml", ".onnx"]
        .iter()
        .any(|suffix| path.ends_with(suffix))
}
fn allowed_file(path: &str) -> bool {
    !path.starts_with('.')
        && !path.contains("..")
        && (path == "config.json"
            || path.starts_with("tokenizer")
            || path.starts_with("special_tokens")
            || path.ends_with(".json")
            || path.ends_with(".model")
            || path.ends_with(".tiktoken")
            || is_weight(path)
            || path == "vocab.json"
            || path == "merges.txt")
}
fn safe_join(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    ensure!(
        !path.is_absolute()
            && !path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
        "ruta de archivo remota inválida"
    );
    Ok(root.join(path))
}
fn collect_files(root: &Path, current: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, files)?;
        } else if entry.file_type()?.is_file() {
            files.push(path.strip_prefix(root)?.to_owned());
        }
    }
    Ok(())
}
fn digest_bundle(root: &Path, files: &[PathBuf]) -> Result<String> {
    let mut hasher = Sha256::new();
    for relative in files {
        let name = relative.to_string_lossy().replace('\\', "/");
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(fs::read(root.join(relative))?);
        hasher.update([0]);
    }
    Ok(hex::encode(hasher.finalize()))
}
