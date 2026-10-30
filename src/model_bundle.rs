//! Validated model bundle shared by every concrete runtime.
//!
//! The bundle loader performs only format and identity checks. It does not
//! claim that a model can execute on a given backend; that decision remains
//! with the adapter after it has inspected the bundle.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::model_config::ModelSpec;

#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub root: PathBuf,
    pub spec: ModelSpec,
    pub chat_template: Option<String>,
    pub tensor_names: BTreeSet<String>,
}

impl ModelBundle {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let spec = ModelSpec::inspect(root)?;
        let tokenizer_path = root.join("tokenizer.json");
        ensure!(tokenizer_path.is_file(), "falta tokenizer.json");
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow::anyhow!("tokenizer.json inválido: {error}"))?;

        let chat_template = read_chat_template(root)?;
        let mut tensor_names = BTreeSet::new();
        for shard in &spec.weight_shards {
            let bytes = fs::read(root.join(shard)).with_context(|| {
                format!("no se pudo leer el shard {}", root.join(shard).display())
            })?;
            let tensors = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("safetensors inválido: {shard}"))?;
            tensor_names.extend(tensors.names().into_iter().cloned());
        }
        ensure!(!tensor_names.is_empty(), "el bundle no contiene tensores");

        Ok(Self {
            root: root.to_owned(),
            spec,
            chat_template,
            tensor_names,
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names.contains(name)
    }

    /// Resolves the conventional decoder-only tensor contract from the
    /// bundle inventory. It deliberately uses names and declared dimensions,
    /// never a model name or vendor identifier.
    pub fn validate_decoder_layout(&self) -> Result<()> {
        let required = ["model.embed_tokens.weight", "model.norm.weight"];
        for name in required {
            ensure!(
                self.has_tensor(name),
                "decoder bundle missing tensor {name}"
            );
        }
        if self.spec.tie_word_embeddings != Some(true) {
            ensure!(
                self.has_tensor("lm_head.weight"),
                "decoder bundle missing tensor lm_head.weight"
            );
        }
        for layer in 0..self.spec.num_hidden_layers {
            for suffix in [
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                let name = format!("model.layers.{layer}.{suffix}");
                ensure!(
                    self.has_tensor(&name),
                    "decoder bundle missing tensor {name}"
                );
            }
        }
        Ok(())
    }

    pub fn probe_token_ids(&self) -> Result<Vec<u32>> {
        let tokenizer = Tokenizer::from_file(self.root.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("tokenizer.json inválido: {error}"))?;
        let encoded = tokenizer
            .encode("KVSLIM conformance probe", true)
            .map_err(|error| anyhow::anyhow!("no se pudo tokenizar la prueba: {error}"))?;
        let ids = encoded.get_ids().to_vec();
        ensure!(
            !ids.is_empty(),
            "la prueba de tokenización no produjo tokens"
        );
        Ok(ids)
    }
}

fn read_chat_template(root: &Path) -> Result<Option<String>> {
    let path = root.join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("tokenizer_config.json inválido: {}", path.display()))?;
    Ok(value
        .get("chat_template")
        .and_then(Value::as_str)
        .map(str::to_owned))
}
