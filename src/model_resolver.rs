//! Provider-neutral candidate ranking.
//!
//! This layer is intentionally a recommender, not an execution authority. A
//! candidate must still be imported and exercised by a real adapter before it
//! can become active.

use crate::adapter_catalog::HostCapabilities;
use crate::model_catalog::ModelCandidate;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum RecommendationState {
    Candidate,
    NoLocalInferenceAdapter,
    IncompatibleWithLocalAdapters,
    RequiresLocalValidation,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelRecommendation {
    pub candidate: ModelCandidate,
    pub score: i32,
    pub state: RecommendationState,
    pub reasons: Vec<String>,
    pub exclusions: Vec<String>,
}

/// Rank catalog candidates against the capabilities currently observed by the
/// local process. This never downloads, loads or activates a model.
pub fn rank_candidates(
    host: &HostCapabilities,
    candidates: impl IntoIterator<Item = ModelCandidate>,
) -> Vec<ModelRecommendation> {
    let has_adapter = host.inference_adapter_available;
    let mut ranked = candidates
        .into_iter()
        .map(|candidate| rank_one(host, has_adapter, candidate))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.candidate.model_id.cmp(&right.candidate.model_id))
    });
    ranked
}

fn rank_one(
    host: &HostCapabilities,
    has_adapter: bool,
    candidate: ModelCandidate,
) -> ModelRecommendation {
    let mut score = 0;
    let mut reasons = Vec::new();
    let mut exclusions = Vec::new();
    let has_catalog_token = std::env::var("HF_TOKEN").is_ok()
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("deploy/local-production/runtime/secrets/huggingface.token")
            .is_file();
    if (candidate.gated || candidate.private) && !has_catalog_token {
        score -= 1000;
        exclusions.push("requiere autenticación de Hugging Face".into());
    }
    if candidate.pipeline_tag.as_deref().is_some_and(|tag| {
        matches!(
            tag,
            "text-generation" | "text2text-generation" | "conversational"
        )
    }) {
        score += 40;
        reasons.push("tarea compatible con generación de texto".into());
    } else {
        exclusions.push("la tarea del modelo no está confirmada para generación de texto".into());
    }
    if candidate
        .library_name
        .as_deref()
        .is_some_and(|library| matches!(library, "transformers" | "llama.cpp" | "ggml" | "mlx"))
    {
        score += 15;
        reasons.push("biblioteca con formato conocido por el ecosistema local".into());
    }
    let supported_by_declared_adapter = host.adapters.iter().any(|status| {
        status.availability == kvslim_adapter_api::AdapterAvailability::RuntimeAvailable
            && candidate.model_type.as_deref().is_some_and(|model_type| {
                status.descriptor.supported_model_types.is_empty()
                    || status
                        .descriptor
                        .supported_model_types
                        .iter()
                        .any(|supported| supported == model_type)
            })
    });
    if candidate.model_type.is_none() {
        score -= 1000;
        exclusions.push("la arquitectura del modelo no está confirmada por el catálogo".into());
    } else if !supported_by_declared_adapter {
        score -= 1000;
        exclusions.push("ningún adaptador disponible declara esta arquitectura".into());
    } else {
        score += 35;
        reasons.push("arquitectura declarada por un adaptador disponible".into());
    }
    if candidate.license.is_some() {
        score += 5;
        reasons.push("licencia declarada en el catálogo".into());
    } else {
        exclusions.push("licencia no declarada; requiere revisión".into());
    }
    if candidate.downloads.unwrap_or_default() > 100_000 {
        score += 5;
        reasons.push("uso comunitario significativo".into());
    }
    if host.cpu_available {
        reasons.push(format!(
            "CPU local disponible con {} hilos",
            host.cpu_parallelism
        ));
    }
    for accelerator in &host.accelerators {
        if accelerator.available {
            reasons.push(format!(
                "acelerador local: {} {} ({} MiB)",
                accelerator.backend,
                accelerator.name,
                accelerator
                    .memory_bytes
                    .map(|bytes| bytes / (1024 * 1024))
                    .unwrap_or_default()
            ));
        }
    }
    if let Some(available) = host.memory_available_bytes {
        reasons.push(format!(
            "memoria disponible local: {} MiB",
            available / (1024 * 1024)
        ));
        if let Some(weights) = candidate.estimated_weight_bytes {
            let budget = available.saturating_mul(70) / 100;
            if weights > budget {
                score -= 1000;
                exclusions.push(format!(
                    "los pesos ({:.1} GiB) superan el 70% de la memoria disponible",
                    weights as f64 / (1024.0 * 1024.0 * 1024.0)
                ));
            } else {
                let capacity_bonus = ((weights as f64).log2() as i32 - 28).clamp(0, 80);
                score += 20 + capacity_bonus;
                reasons.push("los pesos caben en el presupuesto local conservador".into());
                reasons.push(format!(
                    "capacidad del modelo priorizada dentro del presupuesto (+{capacity_bonus})"
                ));
            }
        } else {
            exclusions.push(
                "tamaño de pesos no disponible; requiere inspección antes de descargar".into(),
            );
        }
    } else {
        exclusions.push("memoria disponible no expuesta por el sistema operativo".into());
    }
    if has_adapter {
        reasons.push("existe al menos un adaptador de inferencia sondeado".into());
    } else {
        exclusions.push("no hay adaptador de inferencia local sondeado".into());
    }
    let memory_ok = host
        .memory_available_bytes
        .zip(candidate.estimated_weight_bytes)
        .is_none_or(|(available, weights)| weights <= available.saturating_mul(70) / 100);
    let auth_ok = !(candidate.gated || candidate.private) || has_catalog_token;
    let state = if !has_adapter {
        RecommendationState::NoLocalInferenceAdapter
    } else if !auth_ok
        || candidate.model_type.is_none()
        || !supported_by_declared_adapter
        || !memory_ok
        || candidate.estimated_weight_bytes.is_none()
    {
        RecommendationState::IncompatibleWithLocalAdapters
    } else {
        RecommendationState::RequiresLocalValidation
    };
    ModelRecommendation {
        candidate,
        score,
        state,
        reasons,
        exclusions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(adapter: bool) -> HostCapabilities {
        HostCapabilities {
            architecture: "test",
            cpu_available: true,
            cpu_parallelism: 4,
            memory_total_bytes: Some(16 * 1024 * 1024 * 1024),
            memory_available_bytes: Some(8 * 1024 * 1024 * 1024),
            accelerators: Vec::new(),
            simd: Vec::new(),
            context_kv_core_available: true,
            inference_adapter_available: adapter,
            adapters: if adapter {
                vec![kvslim_adapter_api::AdapterStatus {
                    descriptor: kvslim_adapter_api::AdapterDescriptor {
                        id: "test".into(),
                        backend: "test".into(),
                        compute: kvslim_adapter_api::ComputeKind::Any,
                        priority: 1,
                        activation: kvslim_adapter_api::ActivationPolicy::Explicit,
                        supported_model_types: vec!["llama".into()],
                    },
                    availability: kvslim_adapter_api::AdapterAvailability::RuntimeAvailable,
                    reason: "test".into(),
                }]
            } else {
                Vec::new()
            },
        }
    }

    fn candidate(id: &str, pipeline: Option<&str>) -> ModelCandidate {
        ModelCandidate {
            provider: "test".into(),
            model_id: id.into(),
            revision: None,
            pipeline_tag: pipeline.map(str::to_owned),
            library_name: Some("transformers".into()),
            license: Some("apache-2.0".into()),
            downloads: Some(1_000_000),
            likes: None,
            last_modified: None,
            source_url: format!("https://example.invalid/{id}"),
            model_type: Some("llama".into()),
            architectures: Vec::new(),
            estimated_weight_bytes: Some(1024),
            gated: false,
            private: false,
        }
    }

    #[test]
    fn ranks_text_generation_and_never_claims_activation() {
        let ranked = rank_candidates(
            &host(true),
            [
                candidate("wrong", Some("image-classification")),
                candidate("right", Some("text-generation")),
            ],
        );
        assert_eq!(ranked[0].candidate.model_id, "right");
        assert_eq!(
            ranked[0].state,
            RecommendationState::RequiresLocalValidation
        );
    }

    #[test]
    fn no_adapter_is_explicitly_reported() {
        let ranked = rank_candidates(&host(false), [candidate("model", Some("text-generation"))]);
        assert_eq!(
            ranked[0].state,
            RecommendationState::NoLocalInferenceAdapter
        );
        assert!(
            ranked[0]
                .exclusions
                .iter()
                .any(|item| item.contains("adaptador"))
        );
    }
}
