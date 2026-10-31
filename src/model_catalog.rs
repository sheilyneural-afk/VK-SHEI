//! Optional model discovery. Providers describe candidates; only the local
//! importer and bundle verifier can make a candidate executable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCandidate {
    pub provider: String,
    pub model_id: String,
    pub revision: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub license: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub last_modified: Option<String>,
    pub source_url: String,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub estimated_weight_bytes: Option<u64>,
    pub gated: bool,
    pub private: bool,
}

#[cfg(feature = "model-catalog")]
pub mod huggingface {
    use super::ModelCandidate;
    use anyhow::{Context, Result, bail, ensure};
    use reqwest::blocking::Client;
    use serde::Deserialize;
    use std::time::Duration;

    const API_BASE: &str = "https://huggingface.co/api";

    #[derive(Debug, Deserialize)]
    struct Item {
        id: String,
        #[serde(default)]
        pipeline_tag: Option<String>,
        #[serde(default)]
        library_name: Option<String>,
        #[serde(default)]
        likes: Option<u64>,
        #[serde(default)]
        downloads: Option<u64>,
        #[serde(default)]
        last_modified: Option<String>,
        #[serde(default)]
        sha: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        siblings: Vec<Sibling>,
        #[serde(default)]
        config: Option<serde_json::Value>,
        #[serde(default)]
        gated: serde_json::Value,
        #[serde(default)]
        private: bool,
    }

    #[derive(Debug, Deserialize)]
    struct Sibling {
        #[serde(default)]
        rfilename: String,
        #[serde(default)]
        size: Option<u64>,
    }

    fn client() -> Result<Client> {
        Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("M-SLIM-model-catalog/0.4")
            .build()
            .context("crear cliente de catálogo")
    }

    /// Reads the project-local token only when the caller explicitly needs it.
    /// The value is never returned in an error or diagnostic.
    fn project_token() -> Option<String> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("deploy/local-production/runtime/secrets/huggingface.token");
        std::fs::read_to_string(path)
            .ok()
            .map(|token| token.trim().to_owned())
            .filter(|token| !token.is_empty())
            .or_else(|| {
                std::env::var("HF_TOKEN")
                    .ok()
                    .filter(|token| !token.trim().is_empty())
            })
    }

    fn request_with_optional_token(
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match project_token() {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    pub fn search(query: &str, limit: u16) -> Result<Vec<ModelCandidate>> {
        let query = query.trim();
        ensure!(!query.is_empty(), "la búsqueda no puede estar vacía");
        ensure!((1..=100).contains(&limit), "limit debe estar entre 1 y 100");
        let response =
            request_with_optional_token(client()?.get(format!("{API_BASE}/models")).query(&[
                ("search", query),
                ("limit", &limit.to_string()),
                ("full", "true"),
            ]))
            .send()
            .context("consultar Hugging Face")?;
        let status = response.status();
        if !status.is_success() {
            bail!("Hugging Face respondió HTTP {status}");
        }
        let items: Vec<Item> = response.json().context("decodificar catálogo")?;
        Ok(items.into_iter().map(to_candidate).collect())
    }

    pub fn search_with_details(query: &str, limit: u16) -> Result<Vec<ModelCandidate>> {
        search(query, limit)?
            .into_iter()
            .map(|candidate| {
                let detailed = inspect(&candidate.model_id)?;
                Ok(ModelCandidate {
                    provider: candidate.provider,
                    model_id: candidate.model_id,
                    revision: detailed.revision.or(candidate.revision),
                    pipeline_tag: detailed.pipeline_tag.or(candidate.pipeline_tag),
                    library_name: detailed.library_name.or(candidate.library_name),
                    license: detailed.license.or(candidate.license),
                    downloads: detailed.downloads.or(candidate.downloads),
                    likes: detailed.likes.or(candidate.likes),
                    last_modified: detailed.last_modified.or(candidate.last_modified),
                    source_url: candidate.source_url,
                    model_type: detailed.model_type.or(candidate.model_type),
                    architectures: if detailed.architectures.is_empty() {
                        candidate.architectures
                    } else {
                        detailed.architectures
                    },
                    estimated_weight_bytes: detailed
                        .estimated_weight_bytes
                        .or(candidate.estimated_weight_bytes),
                    gated: detailed.gated,
                    private: detailed.private,
                })
            })
            .collect()
    }

    pub fn inspect(model_id: &str) -> Result<ModelCandidate> {
        let model_id = model_id.trim();
        ensure!(
            !model_id.is_empty() && !model_id.contains("..") && !model_id.starts_with('/'),
            "identificador inválido"
        );
        let response = request_with_optional_token(
            client()?.get(format!("{API_BASE}/models/{model_id}?full=true&blobs=true")),
        )
        .send()
        .context("consultar Hugging Face")?;
        let status = response.status();
        if !status.is_success() {
            bail!("Hugging Face respondió HTTP {status}");
        }
        Ok(to_candidate(response.json().context("decodificar modelo")?))
    }

    pub fn validate_token(token: &str) -> Result<bool> {
        let token = token.trim();
        ensure!(!token.is_empty(), "el token está vacío");
        let response = client()?
            .get(format!("{API_BASE}/whoami-v2"))
            .bearer_auth(token)
            .send()
            .context("validar credencial de Hugging Face")?;
        Ok(response.status().is_success())
    }

    pub fn validate_project_token() -> Result<bool> {
        Ok(project_token().is_some_and(|token| validate_token(&token).is_ok_and(|valid| valid)))
    }

    fn to_candidate(item: Item) -> ModelCandidate {
        ModelCandidate {
            provider: "huggingface".into(),
            model_id: item.id.clone(),
            revision: item.sha,
            pipeline_tag: item.pipeline_tag,
            library_name: item.library_name,
            license: item
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:"))
                .map(str::to_owned),
            downloads: item.downloads,
            likes: item.likes,
            last_modified: item.last_modified,
            source_url: format!("https://huggingface.co/{}", item.id),
            model_type: item
                .config
                .as_ref()
                .and_then(|config| config.get("model_type"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            architectures: item
                .config
                .as_ref()
                .and_then(|config| config.get("architectures"))
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            estimated_weight_bytes: {
                let bytes: u64 = item
                    .siblings
                    .iter()
                    .filter(|file| is_weight_file(&file.rfilename))
                    .filter_map(|file| file.size)
                    .sum();
                (bytes > 0).then_some(bytes)
            },
            gated: item.gated == serde_json::Value::Bool(true)
                || item.gated == serde_json::Value::String("manual".into()),
            private: item.private,
        }
    }

    fn is_weight_file(path: &str) -> bool {
        [".safetensors", ".bin", ".gguf", ".ggml", ".onnx"]
            .iter()
            .any(|suffix| path.ends_with(suffix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_contract_is_provider_neutral() {
        let candidate = ModelCandidate {
            provider: "example".into(),
            model_id: "org/model".into(),
            revision: None,
            pipeline_tag: Some("text-generation".into()),
            library_name: None,
            license: Some("apache-2.0".into()),
            downloads: None,
            likes: None,
            last_modified: None,
            source_url: "https://example.invalid/org/model".into(),
            model_type: Some("example".into()),
            architectures: Vec::new(),
            estimated_weight_bytes: None,
            gated: false,
            private: false,
        };
        let encoded = serde_json::to_string(&candidate).unwrap();
        assert!(encoded.contains("org/model"));
        assert!(encoded.contains("text-generation"));
    }
}
