//! Hugging Face config.json inspector independent of model weights and LibTorch.
//! Validation here is metadata validation, NOT proof of end-to-end inference.

use crate::portable::{AttentionGeometry, FloatFormat};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingSpec {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_dir: PathBuf,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub declared_dtype: Option<String>,
    pub storage_format: Option<FloatFormat>,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub attention: AttentionGeometry,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<RopeScalingSpec>,
    pub rms_norm_eps: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub weight_shards: Vec<String>,
    pub tokenizer_json_present: bool,
    /// End-of-sequence ids declared by the bundle, if present.
    pub eos_token_ids: Vec<u32>,
}

/// Resolve the checkpoint's authoritative generation stop IDs. Hugging Face
/// generation settings take precedence over the architectural config because
/// chat checkpoints can declare an additional end-of-turn token there.
pub fn resolve_eos_token_ids(
    model_dir: impl AsRef<Path>,
    vocab_size: Option<usize>,
) -> Result<Vec<u32>> {
    let model_dir = model_dir.as_ref();
    let generation = model_dir.join("generation_config.json");
    let source = if generation.is_file() {
        generation
    } else {
        model_dir.join("config.json")
    };
    let bytes = fs::read(&source)
        .with_context(|| format!("no se pudo leer la configuración EOS {}", source.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("configuración EOS inválida: {}", source.display()))?;
    let Some(raw) = value.get("eos_token_id") else {
        ensure!(
            source.file_name().and_then(|name| name.to_str()) != Some("generation_config.json"),
            "generation_config.json necesita eos_token_id"
        );
        return Ok(Vec::new());
    };
    let candidates = match raw {
        Value::Number(_) => vec![raw],
        Value::Array(values) => {
            ensure!(!values.is_empty(), "eos_token_id no puede estar vacío");
            values.iter().collect()
        }
        _ => anyhow::bail!("eos_token_id debe ser entero o lista de enteros"),
    };
    let mut resolved = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let id = candidate
            .as_u64()
            .context("eos_token_id debe contener enteros")?;
        if let Some(vocab) = vocab_size {
            ensure!(
                id < vocab as u64,
                "eos_token_id {id} excede vocab_size {vocab}"
            );
        }
        let id = u32::try_from(id).context("eos_token_id fuera de rango")?;
        if !resolved.contains(&id) {
            resolved.push(id);
        }
    }
    Ok(resolved)
}

impl ModelSpec {
    pub fn inspect(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let source = model_dir.join("config.json");
        let bytes =
            fs::read(&source).with_context(|| format!("no se pudo leer {}", source.display()))?;
        let config: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("config.json inválido: {}", source.display()))?;

        let positive = |key: &str| -> Result<usize> {
            let value = config
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("config.json necesita '{key}' entero positivo"))?;
            ensure!(value > 0, "{key} debe ser positivo");
            usize::try_from(value).context("dimensión fuera del rango de esta plataforma")
        };
        let optional_positive = |key: &str| -> Result<Option<usize>> {
            if config.get(key).is_none_or(Value::is_null) {
                Ok(None)
            } else {
                let value = positive(key)?;
                Ok(Some(value))
            }
        };

        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .context("falta model_type en config.json")?
            .to_owned();
        let architectures = config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|xs| {
                xs.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let hidden_size = positive("hidden_size")?;
        let num_hidden_layers = positive("num_hidden_layers")?;
        let query_heads = positive("num_attention_heads")?;
        let kv_heads = optional_positive("num_key_value_heads")?.unwrap_or(query_heads);
        let head_dim = match optional_positive("head_dim")? {
            Some(dim) => dim,
            None => {
                ensure!(
                    hidden_size.is_multiple_of(query_heads),
                    "hidden_size no divisible por num_attention_heads y falta head_dim"
                );
                hidden_size / query_heads
            }
        };
        let attention = AttentionGeometry::new(query_heads, kv_heads, head_dim)
            .map_err(|e| anyhow::anyhow!(e))?;
        let declared_dtype = config
            .get("torch_dtype")
            .or_else(|| config.get("dtype"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let storage_format = declared_dtype.as_deref().and_then(Self::parse_float_dtype);
        let intermediate_size = optional_positive("intermediate_size")?;
        let vocab_size = optional_positive("vocab_size")?;
        let max_position_embeddings = optional_positive("max_position_embeddings")?;
        let rope_theta = config.get("rope_theta").and_then(Value::as_f64);
        let rope_scaling = match config.get("rope_scaling") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let rope_type = value
                    .get("rope_type")
                    .or_else(|| value.get("type"))
                    .and_then(Value::as_str)
                    .context("rope_scaling necesita rope_type")?;
                if rope_type != "llama3" {
                    None
                } else {
                    let finite_positive = |key: &str| -> Result<f64> {
                        let number = value
                            .get(key)
                            .and_then(Value::as_f64)
                            .with_context(|| format!("rope_scaling necesita {key}"))?;
                        ensure!(number.is_finite() && number > 0.0, "{key} inválido");
                        Ok(number)
                    };
                    let original = value
                        .get("original_max_position_embeddings")
                        .and_then(Value::as_u64)
                        .context("rope_scaling necesita original_max_position_embeddings")?;
                    ensure!(original > 0, "original_max_position_embeddings inválido");
                    let scaling = RopeScalingSpec {
                        factor: finite_positive("factor")?,
                        low_freq_factor: finite_positive("low_freq_factor")?,
                        high_freq_factor: finite_positive("high_freq_factor")?,
                        original_max_position_embeddings: usize::try_from(original)
                            .context("original_max_position_embeddings fuera de rango")?,
                    };
                    ensure!(
                        scaling.high_freq_factor > scaling.low_freq_factor,
                        "high_freq_factor debe superar low_freq_factor"
                    );
                    Some(scaling)
                }
            }
        };
        let rms_norm_eps = config.get("rms_norm_eps").and_then(Value::as_f64);
        for (name, number) in [("rope_theta", rope_theta), ("rms_norm_eps", rms_norm_eps)] {
            if let Some(value) = number {
                ensure!(value.is_finite() && value > 0.0, "{name} inválido");
            }
        }
        let tie_word_embeddings = config.get("tie_word_embeddings").and_then(Value::as_bool);
        let tokenizer_json_present = model_dir.join("tokenizer.json").is_file();
        let mut weight_shards = fs::read_dir(model_dir)
            .with_context(|| format!("no se pudo listar {}", model_dir.display()))?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                (path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors"))
                    .then(|| path.file_name().and_then(|s| s.to_str()).map(str::to_owned))
                    .flatten()
            })
            .collect::<Vec<_>>();
        weight_shards.sort();
        let eos_token_ids = resolve_eos_token_ids(model_dir, vocab_size)?;

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            model_type,
            architectures,
            declared_dtype,
            storage_format,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            attention,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scaling,
            rms_norm_eps,
            tie_word_embeddings,
            weight_shards,
            tokenizer_json_present,
            eos_token_ids,
        })
    }

    pub fn parse_float_dtype(dtype: &str) -> Option<FloatFormat> {
        match dtype.to_ascii_lowercase().as_str() {
            "bfloat16" | "bf16" | "torch.bfloat16" => Some(FloatFormat::Bf16),
            "float16" | "half" | "fp16" | "f16" | "torch.float16" => Some(FloatFormat::F16),
            "float32" | "float" | "fp32" | "f32" | "torch.float32" => Some(FloatFormat::F32),
            _ => None,
        }
    }

    pub fn ensure_cpu_runtime_compatible(&self) -> Result<()> {
        ensure!(self.tokenizer_json_present, "falta tokenizer.json");
        ensure!(!self.weight_shards.is_empty(), "no hay shards de pesos");
        ensure!(self.intermediate_size.is_some(), "falta intermediate_size");
        ensure!(self.vocab_size.is_some(), "falta vocab_size");
        for shard in &self.weight_shards {
            ensure!(
                self.model_dir.join(shard).is_file(),
                "falta el shard {shard}"
            );
        }
        Ok(())
    }

    pub fn declared_context_limit(&self) -> Option<usize> {
        self.max_position_embeddings
    }

    pub fn support_summary(&self) -> Value {
        serde_json::json!({
            "model_type": self.model_type,
            "declared_dtype": self.declared_dtype,
            "portable_byte_storage": self.storage_format.is_some(),
            "model_metadata_inspection": true,
            "decoder_geometry_declared": self.intermediate_size.is_some()
                && self.vocab_size.is_some(),
            "rope_scaling": self.rope_scaling,
            "cpu_runtime_ready_for_loading": self.ensure_cpu_runtime_compatible().is_ok(),
            "rust_cpu_reference_attention": true,
            "native_cpu_runtime_compiled": cfg!(feature = "native-cpu-backend"),
            "native_cpu_bundle_conformance": false,
            "torch_inference_backend_compiled": cfg!(feature = "torch-backend"),
            "gpu_reference_attention_compiled": cfg!(feature = "accelerator-backend"),
            "gpu_kernels_implemented": cfg!(feature = "accelerator-backend"),
            "accelerator_full_model_inference": false,
            "remote_api_controls_external_weights_or_kv": false,
            "declared_max_position_embeddings_not_certified": self.max_position_embeddings
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(json_config: Value) -> (tempfile::TempDir, ModelSpec) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&json_config).unwrap(),
        )
        .unwrap();
        let spec = ModelSpec::inspect(dir.path()).unwrap();
        (dir, spec)
    }

    #[test]
    fn accepts_generic_gqa_metadata_without_backend_claim() {
        let (_tmp, spec) = model(json!({
            "model_type": "qwen2",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "torch_dtype": "float16",
            "max_position_embeddings": 32768
        }));
        assert_eq!(spec.attention, AttentionGeometry::new(12, 2, 128).unwrap());
        assert_eq!(spec.storage_format, Some(FloatFormat::F16));
        assert!(spec.ensure_cpu_runtime_compatible().is_err());
    }

    #[test]
    fn accepts_declared_geometry_but_requires_real_bundle_files() {
        let (tmp, spec) = model(json!({
            "model_type": "llama",
            "hidden_size": 1536,
            "intermediate_size": 4608,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 130560,
            "rope_theta": 5000000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072
        }));
        assert!(spec.ensure_cpu_runtime_compatible().is_err());
        fs::write(tmp.path().join("model-00001.safetensors"), []).unwrap();
        fs::write(tmp.path().join("tokenizer.json"), []).unwrap();
        assert!(
            ModelSpec::inspect(tmp.path())
                .unwrap()
                .ensure_cpu_runtime_compatible()
                .is_ok()
        );
    }

    #[test]
    fn rejects_malformed_rope_scaling() {
        let dir = tempfile::tempdir().unwrap();
        let data = json!({
            "model_type":"llama", "hidden_size":128, "num_hidden_layers":1,
            "num_attention_heads":1, "rope_theta":500000.0,
            "rope_scaling": {
                "rope_type":"llama3", "factor":8.0,
                "low_freq_factor":4.0, "high_freq_factor":1.0,
                "original_max_position_embeddings":8192
            }
        });
        fs::write(dir.path().join("config.json"), data.to_string()).unwrap();
        assert!(ModelSpec::inspect(dir.path()).is_err());
    }

    #[test]
    fn missing_or_invalid_head_dims_fails() {
        let dir = tempfile::tempdir().unwrap();
        let data = json!({"model_type":"llama","hidden_size":1537,"num_hidden_layers":1,
            "num_attention_heads":12,"num_key_value_heads":5});
        fs::write(dir.path().join("config.json"), data.to_string()).unwrap();
        assert!(ModelSpec::inspect(dir.path()).is_err());
    }

    #[test]
    fn recognizes_float_formats_without_claiming_fp8() {
        assert_eq!(
            ModelSpec::parse_float_dtype("torch.float16"),
            Some(FloatFormat::F16)
        );
        assert_eq!(
            ModelSpec::parse_float_dtype("BF16"),
            Some(FloatFormat::Bf16)
        );
        assert_eq!(
            ModelSpec::parse_float_dtype("float32"),
            Some(FloatFormat::F32)
        );
        assert_eq!(ModelSpec::parse_float_dtype("fp8"), None);
    }

    #[test]
    fn generation_config_is_authoritative_for_chat_stop_ids() {
        let config = json!({
            "model_type":"llama", "hidden_size":128, "num_hidden_layers":1,
            "num_attention_heads":1, "vocab_size":128256, "eos_token_id":128001
        });
        let (dir, initial) = model(config);
        assert_eq!(initial.eos_token_ids, vec![128001]);
        fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id":[128001,128009,128009]}"#,
        )
        .unwrap();
        assert_eq!(
            ModelSpec::inspect(dir.path()).unwrap().eos_token_ids,
            vec![128001, 128009]
        );
    }

    #[test]
    fn generation_stop_ids_fail_closed_when_malformed_or_outside_vocab() {
        let config = json!({
            "model_type":"llama", "hidden_size":128, "num_hidden_layers":1,
            "num_attention_heads":1, "vocab_size":128256, "eos_token_id":128001
        });
        let (dir, _) = model(config);
        for invalid in [
            r#"{"eos_token_id":[]}"#,
            r#"{"eos_token_id":"128009"}"#,
            r#"{"eos_token_id":[128256]}"#,
            r#"{}"#,
        ] {
            fs::write(dir.path().join("generation_config.json"), invalid).unwrap();
            assert!(
                ModelSpec::inspect(dir.path()).is_err(),
                "accepted {invalid}"
            );
        }
    }
}
