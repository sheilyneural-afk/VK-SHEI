//! Validated model bundle shared by every concrete runtime.
//!
//! The bundle loader performs only format and identity checks. It does not
//! claim that a model can execute on a given backend; that decision remains
//! with the adapter after it has inspected the bundle.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use memmap2::MmapOptions;
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
    pub tensor_shapes: BTreeMap<String, Vec<usize>>,
    /// Physical dtype read from each safetensors directory. This is authoritative;
    /// `config.json::torch_dtype` is only a model declaration.
    pub tensor_dtypes: BTreeMap<String, String>,
    /// Exact payload bytes per tensor, excluding safetensors metadata.
    pub tensor_payload_bytes: BTreeMap<String, usize>,
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
        let mut tensor_shapes = BTreeMap::new();
        let mut tensor_dtypes = BTreeMap::new();
        let mut tensor_payload_bytes = BTreeMap::new();
        for shard in &spec.weight_shards {
            let path = root.join(shard);
            let file = File::open(&path)
                .with_context(|| format!("no se pudo abrir el shard {}", path.display()))?;
            // SAFETY: immutable read-only mapping, alive for the complete tensor scan.
            // This avoids copying multi-gigabyte model shards merely to inspect metadata.
            let bytes = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("no se pudo mapear el shard {}", path.display()))?;
            let tensors = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("safetensors inválido: {shard}"))?;
            for name in tensors.names() {
                let tensor = tensors.tensor(name)?;
                ensure!(
                    tensor_names.insert(name.clone()),
                    "tensor duplicado entre shards: {name}"
                );
                tensor_shapes.insert(name.clone(), tensor.shape().to_vec());
                tensor_dtypes.insert(name.clone(), format!("{:?}", tensor.dtype()));
                tensor_payload_bytes.insert(name.clone(), tensor.data().len());
            }
        }
        ensure!(!tensor_names.is_empty(), "el bundle no contiene tensores");

        Ok(Self {
            root: root.to_owned(),
            spec,
            chat_template,
            tensor_names,
            tensor_shapes,
            tensor_dtypes,
            tensor_payload_bytes,
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names.contains(name)
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensor_shapes.get(name).map(Vec::as_slice)
    }

    pub fn tensor_dtype(&self, name: &str) -> Option<&str> {
        self.tensor_dtypes.get(name).map(String::as_str)
    }

    pub fn physical_dtype_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for dtype in self.tensor_dtypes.values() {
            *counts.entry(dtype.clone()).or_insert(0) += 1;
        }
        counts
    }

    pub fn total_tensor_payload_bytes(&self) -> usize {
        self.tensor_payload_bytes.values().copied().sum()
    }

    /// Rejects a backend/model mismatch using physical tensor directories,
    /// rather than trusting the optional dtype declaration in config.json.
    pub fn ensure_all_tensors_dtype(&self, expected: &str) -> Result<()> {
        let mismatches = dtype_mismatches(&self.tensor_dtypes, expected, 8);
        ensure!(
            mismatches.is_empty(),
            "backend requires physical {expected} tensors; incompatible tensors: {}",
            mismatches.join(", ")
        );
        Ok(())
    }

    pub fn validate_decoder_layout(&self) -> Result<()> {
        crate::decoder_layout::resolve(self)?;
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

fn dtype_mismatches(
    inventory: &BTreeMap<String, String>,
    expected: &str,
    limit: usize,
) -> Vec<String> {
    inventory
        .iter()
        .filter(|(_, dtype)| dtype.as_str() != expected)
        .map(|(name, dtype)| format!("{name}:{dtype}"))
        .take(limit)
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_dtype_inventory_is_fail_closed_and_bounded() {
        let inventory = BTreeMap::from([
            ("a".to_owned(), "BF16".to_owned()),
            ("b".to_owned(), "F16".to_owned()),
            ("c".to_owned(), "U8".to_owned()),
        ]);
        assert_eq!(
            dtype_mismatches(&inventory, "BF16", 1),
            vec!["b:F16".to_owned()]
        );
        assert_eq!(dtype_mismatches(&inventory, "BF16", 8).len(), 2);
        assert!(dtype_mismatches(&inventory, "F32", 0).is_empty());
    }
}
