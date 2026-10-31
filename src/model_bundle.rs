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
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::model_config::ModelSpec;

pub const MODEL_BUNDLE_MANIFEST_SCHEMA: &str = "mslim.model_bundle/v1";

#[derive(Debug, Clone, Deserialize)]
pub struct ModelBundleManifest {
    pub schema: String,
    pub bundle_sha256: String,
    pub files: Vec<String>,
}

/// Validate the installed bundle identity without re-hashing multi-gigabyte
/// weights. Full content hashing remains the responsibility of import and the
/// explicit inventory command.
pub fn validated_bundle_manifest(root: impl AsRef<Path>) -> Result<ModelBundleManifest> {
    let root = root.as_ref();
    let directory_digest = root
        .file_name()
        .and_then(|name| name.to_str())
        .context("bundle directory has no UTF-8 SHA-256 identity")?;
    ensure!(
        is_sha256(directory_digest),
        "bundle directory is not named by a SHA-256 digest"
    );
    let path = root.join("bundle-manifest.json");
    let manifest: ModelBundleManifest = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("missing bundle manifest: {}", path.display()))?,
    )
    .with_context(|| format!("invalid bundle manifest: {}", path.display()))?;
    ensure!(
        manifest.schema == MODEL_BUNDLE_MANIFEST_SCHEMA,
        "unsupported bundle manifest schema"
    );
    ensure!(
        manifest
            .bundle_sha256
            .eq_ignore_ascii_case(directory_digest),
        "bundle manifest digest does not match its directory"
    );

    let listed = manifest
        .files
        .iter()
        .map(|name| safe_manifest_path(name))
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        listed.len() == manifest.files.len(),
        "bundle manifest contains duplicate files"
    );
    let mut installed = BTreeSet::new();
    collect_installed_files(root, root, &mut installed)?;
    ensure!(
        listed == installed,
        "bundle manifest file inventory does not match installed files"
    );
    Ok(manifest)
}

pub fn validated_bundle_digest(root: impl AsRef<Path>) -> Result<String> {
    Ok(validated_bundle_manifest(root)?.bundle_sha256)
}

fn safe_manifest_path(value: &str) -> Result<PathBuf> {
    use std::path::Component;

    let path = Path::new(value);
    ensure!(
        !value.is_empty() && !path.is_absolute(),
        "invalid bundle manifest path"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "bundle manifest path escapes its root"
    );
    ensure!(
        path != Path::new("bundle-manifest.json"),
        "bundle manifest cannot list itself"
    );
    Ok(path.to_owned())
}

fn collect_installed_files(
    root: &Path,
    current: &Path,
    files: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        ensure!(!file_type.is_symlink(), "bundle contains a symbolic link");
        if file_type.is_dir() {
            collect_installed_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path.strip_prefix(root)?.to_owned();
            if relative != Path::new("bundle-manifest.json") {
                files.insert(relative);
            }
        } else {
            anyhow::bail!("bundle contains an unsupported file type");
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

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
        validated_bundle_manifest(root)?;
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
            .encode("M-SLIM conformance probe", true)
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

    #[test]
    fn manifest_rejects_legacy_schema_and_inventory_drift() {
        let parent = tempfile::tempdir().unwrap();
        let digest = "a".repeat(64);
        let root = parent.path().join(&digest);
        fs::create_dir(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        let manifest = |schema: &str, files: Vec<&str>| {
            serde_json::json!({
                "schema": schema,
                "bundle_sha256": digest,
                "files": files
            })
        };
        fs::write(
            root.join("bundle-manifest.json"),
            serde_json::to_vec(&manifest("kvslim.model_bundle/v1", vec!["config.json"])).unwrap(),
        )
        .unwrap();
        assert!(validated_bundle_manifest(&root).is_err());

        fs::write(
            root.join("bundle-manifest.json"),
            serde_json::to_vec(&manifest(MODEL_BUNDLE_MANIFEST_SCHEMA, vec!["config.json"]))
                .unwrap(),
        )
        .unwrap();
        assert!(validated_bundle_manifest(&root).is_ok());

        fs::write(root.join("unexpected.json"), b"{}").unwrap();
        assert!(validated_bundle_manifest(&root).is_err());
    }
}
