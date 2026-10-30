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
