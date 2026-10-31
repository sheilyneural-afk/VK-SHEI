//! OpenCL kernel handles and GPU-agnostic structs for the LLM backend.
//!
//! Contains `RopeKernelConfig` (rotary embedding params), lightweight buffer
//! wrapper structs, checkpoint shard loading, and `LlmKernels` (the kernel
//! handle registry). No `unsafe` code.

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, MmapOptions};
use opencl3::{kernel::Kernel, memory::Buffer, program::Program};
use safetensors::{SafeTensors, tensor::TensorView};
use std::{
    collections::HashMap,
    fs::File,
    path::Path,
};

use crate::model_config::ModelSpec;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeKernelConfig {
    pub(crate) theta: f32,
    pub(crate) llama3_scaling: u32,
    pub(crate) factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) original_max_position: f32,
}

impl RopeKernelConfig {
    pub(crate) fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let theta = spec.rope_theta.context("missing RoPE theta")? as f32;
        ensure!(theta.is_finite() && theta > 0.0, "invalid RoPE theta");
        if let Some(scaling) = &spec.rope_scaling {
            Ok(Self {
                theta,
                llama3_scaling: 1,
                factor: scaling.factor as f32,
                low_freq_factor: scaling.low_freq_factor as f32,
                high_freq_factor: scaling.high_freq_factor as f32,
                original_max_position: scaling.original_max_position_embeddings as f32,
            })
        } else {
            Ok(Self {
                theta,
                llama3_scaling: 0,
                factor: 1.0,
                low_freq_factor: 1.0,
                high_freq_factor: 2.0,
                original_max_position: 1.0,
            })
        }
    }

    #[cfg(test)]
    pub(crate) fn inverse_frequency(self, frequency_dim: usize, head_dim: usize) -> f32 {
        let mut inverse = (-(frequency_dim as f32) * self.theta.ln() / head_dim as f32).exp();
        if self.llama3_scaling != 0 {
            let wavelength = std::f32::consts::TAU / inverse;
            let low_wavelength = self.original_max_position / self.low_freq_factor;
            let high_wavelength = self.original_max_position / self.high_freq_factor;
            if wavelength > low_wavelength {
                inverse /= self.factor;
            } else if wavelength >= high_wavelength {
                let smooth = (self.original_max_position / wavelength - self.low_freq_factor)
                    / (self.high_freq_factor - self.low_freq_factor);
                inverse = (1.0 - smooth) * (inverse / self.factor) + smooth * inverse;
            }
        }
        inverse
    }
}

pub(crate) struct GpuWeight {
    pub(crate) data: Buffer<u8>,
    #[cfg(feature = "hip-backend")]
    pub(crate) hip: Option<HipWeight>,
}

pub(crate) struct GpuLayer {
    pub(crate) input_norm: GpuWeight,
    pub(crate) q: GpuWeight,
    pub(crate) k: GpuWeight,
    pub(crate) v: GpuWeight,
    pub(crate) o: GpuWeight,
    pub(crate) post_norm: GpuWeight,
    pub(crate) gate: GpuWeight,
    pub(crate) up: GpuWeight,
    pub(crate) down: GpuWeight,
}

/// Read-only collection of every safetensors shard in one checkpoint.
/// Tensor ownership is discovered from the files themselves instead of
/// assuming that all weights live in one archive. This is deliberately kept
/// independent of Hugging Face's optional index JSON: the BF16 payload is the
/// authority and duplicate tensor names fail closed.
pub(crate) struct CheckpointShards {
    pub(crate) mappings: Vec<memmap2::Mmap>,
    pub(crate) tensor_to_shard: HashMap<String, usize>,
    pub(crate) payload_bytes: usize,
}

impl CheckpointShards {
    pub(crate) fn open(model_dir: &Path, shard_names: &[String]) -> Result<Self> {
        ensure!(
            !shard_names.is_empty(),
            "checkpoint has no safetensors shards"
        );
        let mut mappings = Vec::with_capacity(shard_names.len());
        let mut tensor_to_shard = HashMap::new();
        let mut payload_bytes = 0usize;
        for (shard_index, shard_name) in shard_names.iter().enumerate() {
            let path = model_dir.join(shard_name);
            let file = File::open(&path)
                .with_context(|| format!("open checkpoint shard {}", path.display()))?;
            // SAFETY: Immutable, read-only mapping retained by this owner.
            let mapping = unsafe { MmapOptions::new().map(&file)? };
            // Sequential read-ahead improves streamed layer loading without
            // allocating a second copy of the shard in process memory.
            mapping.advise(Advice::Sequential)?;
            let archive = SafeTensors::deserialize(&mapping)
                .with_context(|| format!("parse checkpoint shard {}", path.display()))?;
            for name in archive.names() {
                let view = archive.tensor(name).with_context(|| {
                    format!("read tensor directory {name} in {}", path.display())
                })?;
                payload_bytes = payload_bytes
                    .checked_add(view.data().len())
                    .context("checkpoint BF16 payload byte count overflow")?;
                ensure!(
                    tensor_to_shard
                        .insert(name.to_owned(), shard_index)
                        .is_none(),
                    "duplicate checkpoint tensor across shards: {name}"
                );
            }
            mappings.push(mapping);
        }
        Ok(Self {
            mappings,
            tensor_to_shard,
            payload_bytes,
        })
    }

    pub(crate) fn with_tensor<R>(
        &self,
        name: &str,
        use_tensor: impl FnOnce(TensorView<'_>) -> Result<R>,
    ) -> Result<R> {
        let shard = *self
            .tensor_to_shard
            .get(name)
            .with_context(|| format!("required original weight not found: {name}"))?;
        let archive = SafeTensors::deserialize(
            self.mappings
                .get(shard)
                .context("checkpoint tensor shard index invalid")?,
        )?;
        let view = archive
            .tensor(name)
            .with_context(|| format!("checkpoint tensor directory lost {name}"))?;
        use_tensor(view)
    }

    pub(crate) fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    #[cfg(all(test, feature = "physical-conformance"))]
    pub(crate) fn tensor_count(&self) -> usize {
        self.tensor_to_shard.len()
    }
}

pub(crate) struct GpuKv {
    pub(crate) key: Buffer<u8>,
    // Dense GPU: full BF16 value cache. LOD GPU: None, BF16 V stays
    // in the ORIGINAL CPU ColdStore and only a bounded stage enters GPU.
    pub(crate) value: Option<Buffer<u8>>,
}

pub(crate) struct LodStage {
    pub(crate) values: Buffer<u8>,
    pub(crate) indices: Buffer<u32>,
    pub(crate) accumulator: Buffer<f32>,
    pub(crate) selected_mass: Buffer<f32>,
}

/// Bounded HOT device tier for exact BF16 V in canonical GPU LOD mode.
/// ColdStore remains the canonical verified authority, always appended.
/// If the exact query history exceeds capacity, no partial approximation
/// is attempted: full verified ColdStore staging resumes immediately.
/// Only legal with epsilon=0, where the exact certificate selects all V.
// GPU BF16 Q/K/V and the device-resident certified index/result
// for the optional micro fused attention kernel.
pub(crate) type MicroHotBuffers<'a> = (
    &'a Buffer<u8>,
    &'a Buffer<u8>,
    &'a Buffer<u8>,
    &'a Buffer<u32>,
    &'a Buffer<u8>,
);

pub(crate) struct MicroSelectedDevice {
    pub(crate) indices: Buffer<u32>,
    pub(crate) last_positions: Vec<u32>,
    pub(crate) upload_calls: usize,
    pub(crate) reuse_hits: usize,
}

pub(crate) struct GpuHotValues {
    pub(crate) values: Vec<Buffer<u8>>,
    // Only configured by --gpu-micro-hot-v-tokens in explicitly positive
    // epsilon mode. This is a BF16 device cache, NOT a replacement for
    // the exact original CPU ColdStore.
    pub(crate) micro_indices: Option<MicroSelectedDevice>,
    pub(crate) capacity_tokens: usize,
    pub(crate) valid_prefix_tokens: usize,
    pub(crate) attention_calls: usize,
    pub(crate) cold_fallback_calls: usize,
    pub(crate) prefetched_bytes: usize,
    pub(crate) micro_sparse_calls: usize,
    pub(crate) micro_full_exact_fallback_calls: usize,
}

/// Optional same-prompt prefill replay for the original LOD GPU route.
/// All BF16 bytes are copied directly from the PREVIOUS physical GPU
/// computation while the original ColdStore is being sealed. On a
/// hit, the next request revalidates SHA-256 and recreates the SAME
/// canonical CPU ColdStore+4-bit axis before any generated token.
/// Nothing is quantized, omitted, or approximated, and cache misses
/// always compute the actual BF16 prefill normally.
/// The cache is bounded and RAM-only (one exact prompt, no disk).
pub(crate) struct VerifiedGpuPromptReplay {
    pub(crate) ids: Vec<u32>,
    pub(crate) kv: Vec<(Vec<u8>, Vec<u8>)>,
    pub(crate) last_logits_bf16: Vec<u8>,
    pub(crate) argmax: u32,
    pub(crate) sha256: [u8; 32],
    pub(crate) bytes: usize,
}

impl VerifiedGpuPromptReplay {
    pub(crate) fn calculate_digest(
        ids: &[u32],
        kv: &[(Vec<u8>, Vec<u8>)],
        logits: &[u8],
        argmax: u32,
    ) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"mslim-gpu-exact-prompt-replay-v1");
        hash.update((ids.len() as u64).to_le_bytes());
        for id in ids {
            hash.update(id.to_le_bytes());
        }
        hash.update((kv.len() as u64).to_le_bytes());
        for (key, value) in kv {
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key);
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value);
        }
        hash.update((logits.len() as u64).to_le_bytes());
        hash.update(logits);
        hash.update(argmax.to_le_bytes());
        hash.finalize().into()
    }

    pub(crate) fn verify(&self) -> bool {
        self.sha256
            == Self::calculate_digest(&self.ids, &self.kv, &self.last_logits_bf16, self.argmax)
    }
}

pub(crate) struct Workspace {
    pub(crate) hidden0: Buffer<u8>,
    pub(crate) hidden1: Buffer<u8>,
    pub(crate) norm1: Buffer<u8>,
    pub(crate) norm2: Buffer<u8>,
    pub(crate) query: Buffer<u8>,
    pub(crate) query_rope: Buffer<u8>,
    pub(crate) key: Buffer<u8>,
    pub(crate) key_rope: Buffer<u8>,
    pub(crate) value: Buffer<u8>,
    pub(crate) attention: Buffer<u8>,
    pub(crate) attn_o: Buffer<u8>,
    pub(crate) gate: Buffer<u8>,
    pub(crate) up: Buffer<u8>,
    pub(crate) activation: Buffer<u8>,
    pub(crate) down: Buffer<u8>,
    pub(crate) logits: Buffer<u8>,
    pub(crate) token_id: Buffer<u32>,
    pub(crate) scores: Buffer<f32>,
    pub(crate) kv: Vec<GpuKv>,
    pub(crate) lod_stage: Option<LodStage>,
}

pub(crate) struct LlmKernels {
    pub(crate) embed: Kernel,
    pub(crate) gemv: Kernel,
    pub(crate) gemv_unroll4: Kernel,
    pub(crate) fused_qkv: Kernel,
    pub(crate) fused_swiglu: Kernel,
    pub(crate) norm: Kernel,
    pub(crate) rope: Kernel,
    pub(crate) qk: Kernel,
    pub(crate) micro_clear: Kernel,
    pub(crate) micro_sparse_qk: Kernel,
    pub(crate) micro_hot_fused: Kernel,
    pub(crate) exact_hot_fused: Kernel,
    pub(crate) softmax: Kernel,
    pub(crate) pv: Kernel,
    pub(crate) lod_pv_stage: Kernel,
    pub(crate) lod_pv_finish: Kernel,
    pub(crate) add: Kernel,
    pub(crate) swiglu: Kernel,
    pub(crate) argmax: Kernel,
}

impl LlmKernels {
    pub(crate) fn load(program: &Program) -> Result<Self> {
        let get = |name| {
            Kernel::create(program, name).with_context(|| format!("GPU kernel missing: {name}"))
        };
        Ok(Self {
            embed: get("llm_embed")?,
            gemv: get("llm_gemv_bf16")?,
            gemv_unroll4: get("llm_gemv_bf16_unroll4")?,
            fused_qkv: get("llm_fused_qkv_bf16")?,
            fused_swiglu: get("llm_fused_swiglu_bf16")?,
            norm: get("llm_rmsnorm")?,
            rope: get("llm_rope")?,
            qk: get("llm_attention_qk")?,
            micro_clear: get("llm_micro_clear_scores")?,
            micro_sparse_qk: get("llm_micro_sparse_qk")?,
            micro_hot_fused: get("llm_micro_hot_fused_attention")?,
            exact_hot_fused: get("llm_exact_hot_fused_attention")?,
            softmax: get("llm_attention_softmax")?,
            pv: get("llm_attention_pv")?,
            lod_pv_stage: get("llm_lod_pv_stage")?,
            lod_pv_finish: get("llm_lod_pv_finish")?,
            add: get("llm_add")?,
            swiglu: get("llm_swiglu")?,
            argmax: get("llm_argmax")?,
        })
    }
}

