//! Weight file loading for the LibTorch backend.
//!
//! Reads and maps BF16 safetensors shards. Hardware-agnostic: no device
//! selection, no model loading, no inference — just byte→tensor mapping.

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use tch::{Kind, Tensor};

pub struct WeightFile {
    path: PathBuf,
    mapping: Mmap,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).with_context(|| format!("no se pudo abrir {}", path.display()))?;
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("no se pudo mapear {}", path.display()))?;
        SafeTensors::deserialize(&mapping)
            .with_context(|| format!("safetensors inválido: {}", path.display()))?;
        Ok(Self { path, mapping })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let archive = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("safetensors inválido: {}", self.path.display()))?;
        let view = archive
            .tensor(name)
            .with_context(|| format!("peso ausente: {name}"))?;
        let kind = match view.dtype() {
            Dtype::BF16 => Kind::BFloat16,
            Dtype::F16 => Kind::Half,
            Dtype::F32 => Kind::Float,
            other => bail!("dtype de peso no soportado para {name}: {other:?}"),
        };
        let shape = view.shape().iter().map(|&v| v as i64).collect::<Vec<_>>();
        let tensor = Tensor::f_from_data_size(view.data(), &shape, kind)
            .with_context(|| format!("no se pudo materializar {name}"))?;
        // `f_from_data_size` copies the bytes into a libtorch allocation. The
        // mapped pages it just read are no longer needed by this tensor, but
        // they keep counting as process RSS while the shard mapping lives,
        // which doubled the load peak (~30 GiB for ~15 GiB of BF16 weights).
        self.release_mapped(view.data());
        Ok(tensor)
    }

    /// Drops the resident pages backing `data` (a slice of this read-only
    /// mapping) once they have been copied. The mapping stays valid: a later
    /// access simply faults the same bytes back in from the file.
    fn release_mapped(&self, data: &[u8]) {
        #[cfg(unix)]
        {
            let base = self.mapping.as_ptr() as usize;
            let start = data.as_ptr() as usize;
            if data.is_empty() || start < base || start + data.len() > base + self.mapping.len() {
                return;
            }
            // SAFETY: the mapping is private and read-only, so MADV_DONTNEED
            // cannot discard modified data; it only unmaps clean file pages,
            // which are re-read from the unchanged file on the next access.
            // No borrowed slice of these pages outlives this call (the tensor
            // owns its own copy). memmap2 aligns the range to page bounds.
            // Failure is harmless (only the RSS optimisation is lost).
            let _ = unsafe {
                self.mapping.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    start - base,
                    data.len(),
                )
            };
        }
        #[cfg(not(unix))]
        let _ = data;
    }

    pub fn require_bf16(&self, name: &str, expected: &[i64]) -> Result<Tensor> {
        let tensor = self.tensor(name)?;
        ensure!(tensor.kind() == Kind::BFloat16, "{name} no es BF16");
        ensure!(
            tensor.size() == expected,
            "geometría incorrecta para {name}"
        );
        Ok(tensor.to_device(Device::Cpu))
    }

    fn tensor_names(&self) -> Result<Vec<String>> {
        let archive = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("safetensors inválido: {}", self.path.display()))?;
        Ok(archive.names().into_iter().cloned().collect())
    }
}

pub fn model_weight_file(model_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let model_dir = model_dir.as_ref();
    let direct = model_dir.join("model.safetensors");
    if direct.is_file() {
        return Ok(direct);
    }
    let mut shards = std::fs::read_dir(model_dir)
        .with_context(|| format!("no se pudo listar {}", model_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        })
        .collect::<Vec<_>>();
    shards.sort();
    ensure!(
        shards.len() == 1,
        "esta fase requiere exactamente un shard safetensors"
    );
    Ok(shards.remove(0))
}

pub struct WeightArchive {
    shards: Vec<WeightFile>,
    owners: HashMap<String, usize>,
}

impl WeightArchive {
    pub fn open_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut paths = Vec::new();
        let direct = model_dir.join("model.safetensors");
        if direct.is_file() {
            paths.push(direct);
        } else {
            paths = std::fs::read_dir(model_dir)
                .with_context(|| format!("no se pudo listar {}", model_dir.display()))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("model-") && name.ends_with(".safetensors")
                        })
                })
                .collect();
            paths.sort();
        }
        ensure!(
            !paths.is_empty(),
            "no hay shards safetensors en {}",
            model_dir.display()
        );
        let shards = paths
            .into_iter()
            .map(WeightFile::open)
            .collect::<Result<Vec<_>>>()?;
        let mut owners = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            for name in shard.tensor_names()? {
                owners.entry(name).or_insert(index);
            }
        }
        Ok(Self { shards, owners })
    }

    pub fn require_bf16(&self, name: &str, expected: &[i64]) -> Result<Tensor> {
        let index = self
            .owners
            .get(name)
            .copied()
            .with_context(|| format!("peso ausente: {name}"))?;
        self.shards[index].require_bf16(name, expected)
    }
}
