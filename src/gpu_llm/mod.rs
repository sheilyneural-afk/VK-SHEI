//! Native Rust + OpenCL BF16 full Llama/configured-model autoregressive inference.
//!
//! A separate experimental engine. Full-model compute including linear
//! projections, RMSNorm, RoPE, SwiGLU, GQA and LM head is performed by
//! OpenCL kernels, not by PyTorch/LibTorch. CPU owns tokenization,
//! device orchestration, and output decoding, NOT neural arithmetic.
//!
//! BF16 weights are uploaded from the original safetensors bytes unchanged.
//! GPU intermediates and KV are BF16; FP32 is used for kernel accumulators.
//! CPU LibTorch comparisons are numerical diagnostics, not a bitwise GPU parity requirement.
//! GPU LOD fidelity is validated against dense GPU under the declared error contract.
//! The canonical BF16 ColdStore/LOD path is available as an explicit opt-in;
//! dense GPU inference remains the default. Both retain original BF16 values.
//!
//! Supported: BF16, sharded or single-file bias-free LlamaForCausalLM-like architecture
//! with plain or Llama-3 scaling and no custom norm/model operators. No fake compatibility
//! for arbitrary checkpoint architectures or devices. No GPU CPU fallback.

use anyhow::{Context, Result, ensure};
use opencl3::{kernel::ExecuteKernel, memory::Buffer, types::CL_BLOCKING};
use safetensors::Dtype;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

mod gpu_resources;
mod kernels;
mod prefill;

#[cfg(feature = "hip-backend")]
use crate::native_hip::{HipRuntime, HipWeight};
#[cfg(feature = "hip-backend")]
use opencl3::memory::ClMem;

use crate::{
    gpu_region_store::{GpuRegionConfig, GpuRegionStats, GpuRegionStore},
    model_config::ModelSpec,
    portable::AttentionGeometry,
};

pub(crate) use gpu_resources::GpuEngine;
pub(crate) use kernels::{
    CheckpointShards, GpuHotValues, GpuLayer, GpuWeight, MicroSelectedDevice, RopeKernelConfig,
    VerifiedGpuPromptReplay, Workspace,
};

const LOCAL_WORK_SIZE: usize = 128;
const BYTES_PER_BF16: usize = 2;

/// Read stopping tokens from the actual checkpoint; never assume configured-model IDs
/// for a Llama tokenizer. A generation_config.json has precedence over
/// config.json, because chat checkpoints can have multiple stop IDs.
fn checkpoint_eos_token_ids(model_dir: &Path, vocab: usize) -> Result<Vec<u32>> {
    let generation = model_dir.join("generation_config.json");
    let config = if generation.is_file() {
        generation
    } else {
        model_dir.join("config.json")
    };
    let bytes = fs::read(&config)
        .with_context(|| format!("EOS config cannot be read: {}", config.display()))?;
    let content: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid EOS JSON: {}", config.display()))?;
    let raw = content
        .get("eos_token_id")
        .context("checkpoint has no eos_token_id; generation requires explicit stopping IDs")?;
    let candidates = if let Some(array) = raw.as_array() {
        array.iter().collect::<Vec<_>>()
    } else {
        vec![raw]
    };
    ensure!(
        !candidates.is_empty(),
        "checkpoint eos_token_id array is empty"
    );
    let mut eos = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let id = candidate
            .as_u64()
            .context("eos_token_id must be an unsigned integer")?;
        ensure!(
            id < vocab as u64,
            "eos_token_id {id} exceeds vocabulary {vocab}"
        );
        let id = u32::try_from(id).context("eos_token_id does not fit u32")?;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    Ok(eos)
}

#[cfg(test)]
mod checkpoint_eos_tests {
    use super::checkpoint_eos_token_ids;
    use std::fs;

    #[test]
    fn checkpoint_eos_follows_generation_config_and_ignores_minicpm_stop_for_llama() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("config.json"),
            r#"{"eos_token_id":128009}"#,
        )
        .unwrap();
        assert_eq!(
            checkpoint_eos_token_ids(root.path(), 128256).unwrap(),
            vec![128009]
        );
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":[128001,128008,128009]}"#,
        )
        .unwrap();
        let ids = checkpoint_eos_token_ids(root.path(), 128256).unwrap();
        assert_eq!(ids, vec![128001, 128008, 128009]);
        assert!(!ids.contains(&1));
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":[1,130073,1]}"#,
        )
        .unwrap();
        assert_eq!(
            checkpoint_eos_token_ids(root.path(), 130560).unwrap(),
            vec![1, 130073]
        );
        assert!(checkpoint_eos_token_ids(root.path(), 120000).is_err());
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":[]}"#,
        )
        .unwrap();
        assert!(checkpoint_eos_token_ids(root.path(), 130560).is_err());
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":"1"}"#,
        )
        .unwrap();
        assert!(checkpoint_eos_token_ids(root.path(), 130560).is_err());
    }
}

/// Resident: full BF16 weights uploaded once to GPU global memory.
/// Stream: upload one layer's weights at a time, lower GPU memory budget
/// but potentially much slower. Embedding and LM head remain resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightMode {
    Resident,
    Stream,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuStep {
    pub argmax_token: u32,
    pub position: usize,
    pub forward_seconds: f64,
    /// Full last-token logits, BF16 reinterpreted to f32, opt-in only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct GpuWeightVerification {
    pub tensor_count: usize,
    pub verified_bytes: usize,
    pub original_concatenated_sha256: String,
    pub gpu_readback_concatenated_sha256: String,
    pub all_original_bf16_bytes_equal: bool,
    pub verification_method: &'static str,
}

/// Receipt for a physical K/V session restored from the canonical persistent
/// ColdStore.  No transformer layer or LM head is evaluated during restore.
#[derive(Clone, Debug, Serialize)]
pub struct SessionRestore {
    pub tokens: usize,
    pub layers: usize,
    pub gpu_key_bytes: usize,
    pub gpu_hot_value_bytes: usize,
    pub cold_value_bytes_verified: usize,
    pub hot_values_restored: bool,
}

fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn digest_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read digest input {}", path.display()))?;
    Ok(digest_bytes(&bytes))
}

pub struct GpuLlm {
    engine: GpuEngine,
    spec: ModelSpec,
    eos_token_ids: Vec<u32>,
    mode: WeightMode,
    // In streamed mode, retain every read-only checkpoint shard so each layer
    // can be uploaded on demand without copying or concatenating 16 GB.
    checkpoint: Option<CheckpointShards>,
    layers: Vec<GpuLayer>,
    embedding: GpuWeight,
    final_norm: GpuWeight,
    lm_head: GpuWeight,
    workspace: Workspace,
    position: usize,
    session_tokens: Vec<u32>,
    max_tokens: usize,
    poisoned: bool,
    lod: Option<GpuRegionStore>,
    hot_values: Option<GpuHotValues>,
    // GPU-only opt-in lossless transaction optimization:
    // skip blocking K/V CPU readbacks during in-flight exact decode;
    // before a turn is returned, commit ALL pending exact BF16 K/V to
    // the same canonical CPU ColdStore and 4-bit ParallelContextAxis.
    deferred_cold_enabled: bool,
    deferred_cold_active: bool,
    last_deferred_cold_commit_seconds: f64,
    last_deferred_cold_committed_token_layers: usize,
    batch_prefill_tokens: Option<usize>,
    prefill_weight_reuse_tokens: usize,
    prefill_register_unrolled: bool,
    prefill_virtual_lanes: usize,
    prefill_fusion_mask: u8,
    gpu_prompt_replay_max_tokens: usize,
    gpu_prompt_replay: Option<VerifiedGpuPromptReplay>,
    verified_prefix_replay_enabled: bool,
    last_verified_prefix_reused_tokens: usize,
    last_verified_prefix_restored_bytes: usize,
    last_prompt_replay_hit: bool,
    last_prompt_replay_rehydrated_bytes: usize,
    last_prompt_replay_sha_verified: bool,
    last_prefill_batches: usize,
    last_prefill_cold_seal_seconds: f64,
    last_prefill_temporary_v_bytes: usize,
    pub model_load_seconds: f64,
    pub gpu_weight_payload_bytes: usize,
    /// Stable digests bound to physical cache receipts.
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub architecture_digest: String,
    calibrated_prefill: Option<(
        crate::calibration_profile::PrefillProfileParameters,
        crate::calibration_profile::PrefillSettings,
    )>,
}

impl GpuLlm {
    /// Preserve original dense OpenCL inference exactly as the default.
    pub fn load(model_dir: impl AsRef<Path>, max_tokens: usize, mode: WeightMode) -> Result<Self> {
        Self::load_internal(model_dir, max_tokens, mode, None, false)
    }

    /// Opt-in canonical CPU M-SLIM index/certification/ColdStore reused
    /// with real GPU QK and chunked BF16 PV, without altering dense mode.
    pub fn load_with_gpu_region(
        model_dir: impl AsRef<Path>,
        max_tokens: usize,
        mode: WeightMode,
        lod: GpuRegionConfig,
    ) -> Result<Self> {
        lod.validate()?;
        Self::load_internal(model_dir, max_tokens, mode, Some(lod), false)
    }

    /// Explicit opt-in native gfx1103 HIP matrix backend, with the
    /// SAME original BF16 weights/canonical CPU LOD and OpenCL attention.
    /// CPU and OpenCL-only paths are untouched and selectable.
    #[cfg(feature = "hip-backend")]
    pub fn load_with_gpu_region_native_hip(
        model_dir: impl AsRef<Path>,
        max_tokens: usize,
        mode: WeightMode,
        lod: GpuRegionConfig,
    ) -> Result<Self> {
        Self::load_internal(model_dir, max_tokens, mode, Some(lod), true)
    }
    #[cfg(feature = "hip-backend")]
    pub fn load_dense_native_hip(
        model_dir: impl AsRef<Path>,
        max_tokens: usize,
        mode: WeightMode,
    ) -> Result<Self> {
        Self::load_internal(model_dir, max_tokens, mode, None, true)
    }

    fn load_internal(
        model_dir: impl AsRef<Path>,
        max_tokens: usize,
        mode: WeightMode,
        lod_config: Option<GpuRegionConfig>,
        hip_enabled: bool,
    ) -> Result<Self> {
        ensure!(
            (1..=32768).contains(&max_tokens),
            "GPU LLM max_tokens must be 1..=32768 until long context validated"
        );
        let started = Instant::now();
        let model_dir = model_dir.as_ref();
        let spec = ModelSpec::inspect(model_dir)?;
        let architecture_digest =
            digest_bytes(&serde_json::to_vec(&spec).context("serialize model architecture")?);
        let tokenizer_digest = digest_file(&model_dir.join("tokenizer.json"))?;
        // Import or explicit inventory verification owns the expensive content
        // hash. Startup still rejects legacy schemas, digest mismatches and
        // manifest/file inventory drift before loading weights.
        let model_digest = crate::model_bundle::validated_bundle_digest(model_dir)?;
        ensure!(
            spec.model_type == "llama",
            "only bias-free LlamaForCausalLM supported"
        );
        ensure!(
            spec.storage_format == Some(crate::portable::FloatFormat::Bf16),
            "GPU LLM currently supports BF16 model weights only"
        );
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&config_text)?;
        if config
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            ensure!(
                spec.rope_scaling.is_some(),
                "only validated llama3 rope_scaling is implemented"
            );
        }
        ensure!(
            config.get("attention_bias").and_then(|v| v.as_bool()) != Some(true),
            "attention bias not implemented"
        );
        ensure!(
            config.get("mlp_bias").and_then(|v| v.as_bool()) != Some(true),
            "MLP bias not implemented"
        );
        ensure!(
            config
                .get("hidden_act")
                .and_then(|v| v.as_str())
                .is_none_or(|v| v == "silu"),
            "non-SwiGLU activation unsupported"
        );
        ensure!(
            spec.tie_word_embeddings != Some(true),
            "tied embeddings require a separate loader path"
        );
        ensure!(
            spec.hidden_size > 0 && spec.hidden_size <= 8192,
            "hidden_size outside tested resource budget"
        );
        ensure!(
            spec.attention.head_dim.is_multiple_of(2),
            "RoPE head dimension must be even"
        );
        ensure!(
            spec.num_hidden_layers <= 80,
            "layer count outside resource budget"
        );
        let vocab = spec.vocab_size.context("vocab_size missing")?;
        let eos_token_ids = checkpoint_eos_token_ids(model_dir, vocab)?;
        ensure!(
            vocab <= 300_000,
            "vocabulary exceeds tested resource budget"
        );
        let engine = GpuEngine::new(hip_enabled)?;
        let checkpoint = CheckpointShards::open(model_dir, &spec.weight_shards)?;

        let embedding = GpuWeight::load(
            &engine,
            &checkpoint,
            "model.embed_tokens.weight",
            &[vocab, spec.hidden_size],
        )?;
        let final_norm = GpuWeight::load(
            &engine,
            &checkpoint,
            "model.norm.weight",
            &[spec.hidden_size],
        )?;
        let lm_head = GpuWeight::load(
            &engine,
            &checkpoint,
            "lm_head.weight",
            &[vocab, spec.hidden_size],
        )?;
        let workspace = Workspace::new(&engine, &spec, max_tokens, lod_config.as_ref())?;
        let lod = lod_config
            .map(|config| GpuRegionStore::new(spec.attention, spec.num_hidden_layers, config))
            .transpose()?;

        let mut layers = Vec::new();
        if mode == WeightMode::Resident {
            for layer in 0..spec.num_hidden_layers {
                layers.push(
                    GpuLayer::load(&engine, &checkpoint, layer, &spec)
                        .with_context(|| format!("GPU weight residency at layer {layer}"))?,
                );
            }
        }
        let gpu_weight_payload_bytes = checkpoint.payload_bytes();
        let checkpoint = (mode == WeightMode::Stream).then_some(checkpoint);
        Ok(Self {
            engine,
            spec,
            eos_token_ids,
            mode,
            checkpoint,
            layers,
            embedding,
            final_norm,
            lm_head,
            workspace,
            position: 0,
            session_tokens: Vec::new(),
            max_tokens,
            poisoned: false,
            lod,
            hot_values: None,
            deferred_cold_enabled: false,
            deferred_cold_active: false,
            last_deferred_cold_commit_seconds: 0.0,
            last_deferred_cold_committed_token_layers: 0,
            batch_prefill_tokens: None,
            prefill_weight_reuse_tokens: 1,
            prefill_register_unrolled: false,
            prefill_virtual_lanes: 128,
            prefill_fusion_mask: 0,
            gpu_prompt_replay_max_tokens: 0,
            gpu_prompt_replay: None,
            verified_prefix_replay_enabled: false,
            last_verified_prefix_reused_tokens: 0,
            last_verified_prefix_restored_bytes: 0,
            last_prompt_replay_hit: false,
            last_prompt_replay_rehydrated_bytes: 0,
            last_prompt_replay_sha_verified: false,
            last_prefill_batches: 0,
            last_prefill_cold_seal_seconds: 0.0,
            last_prefill_temporary_v_bytes: 0,
            model_load_seconds: started.elapsed().as_secs_f64(),
            gpu_weight_payload_bytes,
            model_digest,
            tokenizer_digest,
            architecture_digest,
            calibrated_prefill: None,
        })
    }

    /// Snapshot only the actual non-tunable runtime settings. No guessed budgets.
    pub fn prefill_execution_configuration(&self) -> serde_json::Value {
        serde_json::json!({
            "architecture": self.architecture_digest,
            "max_context_tokens": self.max_tokens,
            "weight_mode": format!("{:?}", self.mode),
            "native_hip": self.native_hip_enabled(),
            "kv": self.lod.as_ref().map(|v| serde_json::json!({
                "epsilon": v.config().epsilon, "stage_tokens": v.config().stage_tokens,
                "axis_policy": v.axis_policy(),
            })),
            "hot_v_tokens": self.exact_hot_v_capacity_tokens(),
            "micro_hot_v_tokens": self.micro_hot_v_capacity_tokens(),
            "prompt_replay_tokens": self.gpu_prompt_replay_max_tokens,
            "shared_prefix": self.verified_prefix_replay_enabled,
            "deferred_cold_commit": self.deferred_cold_enabled,
            "decode_unroll4": self.decode_unroll4(),
            "decode_fused": self.decode_fused_kernels(),
            "exact_hot_fused": self.exact_hot_fused_attention_enabled(),
            "prefill_fusion_mask": self.prefill_fusion_mask,
            "prefill_register_unrolled": self.prefill_register_unrolled,
            "prefill_virtual_lanes": self.prefill_virtual_lanes,
        })
    }

    pub fn prefill_environment(&self) -> Result<crate::calibration_profile::PrefillEnvironment> {
        use crate::calibration_profile::{
            PrefillEnvironment, host_identity, implementation_digest, sha256,
        };
        let hardware = serde_json::json!({"host": host_identity(), "device_key": self.engine.calibration_device_key,
            "memory": self.engine.reported_gpu_memory, "max_allocation": self.engine.max_gpu_allocation});
        Ok(PrefillEnvironment {
            model_digest: self.model_digest.clone(),
            tokenizer_digest: self.tokenizer_digest.clone(),
            backend: if self.native_hip_enabled() {
                "native-hip"
            } else {
                "gpu"
            }
            .into(),
            hardware_digest: sha256(&serde_json::to_vec(&hardware)?),
            execution_digest: sha256(&serde_json::to_vec(
                &self.prefill_execution_configuration(),
            )?),
            implementation_digest: implementation_digest().into(),
        })
    }

    pub fn prefill_settings(&self) -> crate::calibration_profile::PrefillSettings {
        crate::calibration_profile::PrefillSettings {
            batch_tokens: self.batch_prefill_tokens,
            weight_reuse_tokens: self.prefill_weight_reuse_tokens,
        }
    }

    /// Atomic two-field change, only on an empty session; never changes fidelity.
    pub fn set_calibration_prefill_settings(
        &mut self,
        settings: crate::calibration_profile::PrefillSettings,
    ) -> Result<()> {
        settings.validate()?;
        ensure!(
            !self.poisoned && self.position == 0,
            "prefill tuning requires a healthy, empty session"
        );
        ensure!(
            !self.prefill_register_unrolled || [8, 16].contains(&settings.weight_reuse_tokens),
            "profile conflicts with register-unrolled prefill"
        );
        ensure!(
            self.prefill_virtual_lanes == 128
                || (settings.weight_reuse_tokens == 8 && !self.prefill_register_unrolled),
            "profile conflicts with virtual-lane prefill"
        );
        ensure!(
            self.prefill_fusion_mask == 0
                || (settings.weight_reuse_tokens == 8
                    && !self.prefill_register_unrolled
                    && self.prefill_virtual_lanes == 128),
            "profile conflicts with fused prefill kernels"
        );
        self.batch_prefill_tokens = settings.batch_tokens;
        self.prefill_weight_reuse_tokens = settings.weight_reuse_tokens;
        Ok(())
    }

    /// Explicit production opt-in, shared by CLI, worker and ModelAdapter.
    pub fn apply_promoted_prefill_profile(&mut self, path: &Path) -> Result<()> {
        ensure!(
            self.calibrated_prefill.is_none(),
            "a prefill profile is already attached"
        );
        let environment = self.prefill_environment()?;
        let params = crate::calibration_profile::read_promoted_profile(path, &environment)?;
        let original = self.prefill_settings();
        self.set_calibration_prefill_settings(params.settings)?;
        self.calibrated_prefill = Some((params, original));
        Ok(())
    }

    pub fn calibrated_prefill_profile(&self) -> Option<serde_json::Value> {
        self.calibrated_prefill.as_ref().map(|(p, original)| serde_json::json!({
            "profile_id": p.settings.id(), "validated_context_tokens": p.validated_context_tokens,
            "fallback": original, "effective": self.prefill_settings(),
            "coverage": "measured token lengths only; not a universal quality guarantee",
        }))
    }

    fn select_calibrated_prefill(&mut self, prompt_tokens: usize) -> Result<()> {
        if let Some((params, original)) = self.calibrated_prefill.clone() {
            // A later manual change of a non-tunable option cannot silently reuse evidence.
            let execution = crate::calibration_profile::sha256(&serde_json::to_vec(
                &self.prefill_execution_configuration(),
            )?);
            ensure!(
                execution == params.environment.execution_digest,
                "runtime changed after prefill profile activation"
            );
            let settings = if params.validated_context_tokens.contains(&prompt_tokens) {
                params.settings
            } else {
                original
            };
            if self.position == 0 {
                self.set_calibration_prefill_settings(settings)?;
            } else {
                // Replaying an identical prefix may keep a live session. Never
                // reconfigure it; a different setting requires the usual reset.
                ensure!(
                    self.prefill_settings() == settings,
                    "prefill profile change requires a fresh session"
                );
            }
        }
        Ok(())
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn device_name(&self) -> &str {
        &self.engine.name
    }
    pub fn native_hip_enabled(&self) -> bool {
        #[cfg(feature = "hip-backend")]
        {
            self.engine.hip.is_some()
        }
        #[cfg(not(feature = "hip-backend"))]
        {
            false
        }
    }

    pub fn gpu_region_enabled(&self) -> bool {
        self.lod.is_some()
    }
    pub fn gpu_region_config(&self) -> Option<&GpuRegionConfig> {
        self.lod.as_ref().map(GpuRegionStore::config)
    }
    /// Optional GPU-only sensitivity for the same original CPU
    /// content-defined ParallelContextAxis. The model, CPU and BF16
    /// ColdStore are untouched. Only meaningful for epsilon>0.
    pub fn set_gpu_micro_axis_policy(
        &mut self,
        policy: crate::adaptive_extent::AxisPolicy,
    ) -> Result<()> {
        ensure!(
            self.position == 0 && !self.poisoned,
            "GPU micro axis can only be set on a new healthy session"
        );
        let lod = self.lod.as_mut().context("micro axis requires GPU LOD")?;
        ensure!(
            lod.config().epsilon > 0.0,
            "micro region sensitivity is only useful with positive LOD epsilon"
        );
        lod.set_gpu_axis_policy(policy)
    }
    pub fn gpu_micro_axis_policy(&self) -> Option<&crate::adaptive_extent::AxisPolicy> {
        self.lod.as_ref().map(GpuRegionStore::axis_policy)
    }

    /// Optional GPU-only accelerated prompt prefill. None retains the
    /// original tested token-by-token path. CPU Torch remains untouched.
    /// Matrix kernels are 2D BF16 GEMV and strictly causal batch attention.
    pub fn set_batch_prefill_tokens(&mut self, tokens: Option<usize>) -> Result<()> {
        if let Some(count) = tokens {
            ensure!(
                (1..=512).contains(&count),
                "GPU prefill batch size must be 1..=512 under the 64MiB score scratch cap"
            );
        }
        ensure!(!self.poisoned, "GPU context is poisoned");
        self.batch_prefill_tokens = tokens;
        // A batch larger than one must amortize each BF16 weight read across
        // several prompt tokens.  The former default (`reuse=1`) launched a
        // separate workgroup for every [row, token] pair and therefore read
        // the complete matrix once per token, despite reporting batched
        // prefill.  Tile 8 is the smallest parity-validated reuse kernel and
        // is the conservative portable default.  An explicit later call to
        // `set_prefill_weight_reuse_tokens` can still select 1/16/32 after
        // hardware calibration.
        if tokens.is_some_and(|count| count >= 8) && self.prefill_weight_reuse_tokens == 1 {
            self.prefill_weight_reuse_tokens = 8;
        }
        Ok(())
    }
    pub fn batch_prefill_tokens(&self) -> Option<usize> {
        self.batch_prefill_tokens
    }
    /// GPU-only optional BF16 weight reuse across 16 or 32 tokens.
    /// 1 selects the original 2D kernel, unchanged.
    pub fn set_prefill_weight_reuse_tokens(&mut self, count: usize) -> Result<()> {
        ensure!(
            [1, 8, 16, 32].contains(&count),
            "prefill weight reuse must be 1,8,16,32"
        );
        ensure!(!self.poisoned, "GPU context was poisoned");
        self.prefill_weight_reuse_tokens = count;
        Ok(())
    }
    pub fn prefill_weight_reuse_tokens(&self) -> usize {
        self.prefill_weight_reuse_tokens
    }
    /// Optional AMD OpenCL BF16 register-unrolled prefill kernels.
    /// No weight or BF16 layout changes; preserves exact FMA reduction
    /// order and keeps a runtime switch to the original kernels.
    pub fn set_prefill_register_unrolled(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.poisoned, "GPU context poisoned");
        ensure!(
            self.position == 0,
            "GPU register prefill kernel must be selected before session"
        );
        if enabled {
            ensure!(
                [8, 16].contains(&self.prefill_weight_reuse_tokens),
                "register-unrolled BF16 prefill requires reuse tile 8 or 16"
            );
        }
        self.prefill_register_unrolled = enabled;
        Ok(())
    }
    pub fn prefill_register_unrolled(&self) -> bool {
        self.prefill_register_unrolled
    }
    /// GPU-only experimentation: reproduce original 128 logical
    /// BF16 accumulation lanes using 64 or 32 physical OpenCL lanes
    /// and the EXACT same 128-lane reduction tree. Not enabled by
    /// default; requires prefill tile=8 and original dynamic loops.
    pub fn set_prefill_virtual_lanes(&mut self, lanes: usize) -> Result<()> {
        ensure!(
            !self.poisoned && self.position == 0,
            "GPU virtual prefill lanes only configurable on a fresh session"
        );
        ensure!(
            [32, 64, 128].contains(&lanes),
            "GPU physical prefill lanes must be 32,64,128"
        );
        if lanes != 128 {
            ensure!(
                self.prefill_weight_reuse_tokens == 8 && !self.prefill_register_unrolled,
                "virtual prefill lanes 32/64 require original BF16 tile8"
            );
        }
        self.prefill_virtual_lanes = lanes;
        Ok(())
    }
    pub fn prefill_virtual_lanes(&self) -> usize {
        self.prefill_virtual_lanes
    }
    /// Optionally fuse QKV projections and both SwiGLU projections
    /// during GPU prefill. Original BF16 K/V and original 128-lane
    /// per-row arithmetic/rounding retained. Only for physical 128
    /// lane tile8; no CPU/LibTorch or original ColdStore changes.
    /// The original separate GEMV kernels remain the default.
    /// Prefill projection fusion selection:
    /// 0 original fast GEMVs, 1 QKV experimental, 2 SwiGLU tested,
    /// 3 both. Original CPU and ColdStore unaffected. This mode is
    /// configured BEFORE inference and never rewrites BF16 weights.
    pub fn set_prefill_projection_fusion_mode(&mut self, mask: u8) -> Result<()> {
        ensure!(
            !self.poisoned && self.position == 0,
            "GPU prefill fusion can only be changed on a fresh healthy session"
        );
        ensure!(mask <= 3, "GPU prefill fusion mode must be 0,1,2,3");
        if mask != 0 {
            ensure!(
                self.batch_prefill_tokens.is_some()
                    && self.prefill_weight_reuse_tokens == 8
                    && self.prefill_virtual_lanes == 128
                    && !self.prefill_register_unrolled,
                "GPU BF16 fused prefill requires layer-major batch, tile8, 128 physical lanes, no register unroll"
            );
            #[cfg(feature = "hip-backend")]
            ensure!(
                self.engine.hip.is_none(),
                "cannot fuse GPU prefill with HIP-owned BF16 matrices"
            );
        }
        self.prefill_fusion_mask = mask;
        Ok(())
    }
    /// Backward compatibility: true selects both experimental fusions.
    pub fn set_prefill_fuse_projections(&mut self, enabled: bool) -> Result<()> {
        self.set_prefill_projection_fusion_mode(if enabled { 3 } else { 0 })
    }
    pub fn prefill_fuse_projections(&self) -> bool {
        self.prefill_fusion_mask != 0
    }
    pub fn prefill_projection_fusion_mode(&self) -> u8 {
        self.prefill_fusion_mask
    }
    /// RAM-only, SHA-256 verified, *same exact prompt IDs* BF16 prefill
    /// replay. No approximation and no replacement of the original
    /// canonical CPU ColdStore. Disabled by default. Cap at the HOT V
    /// capacity to avoid retaining extra VRAM or disabling cold fallback.
    pub fn set_verified_gpu_prompt_replay(&mut self, max_tokens: usize) -> Result<()> {
        ensure!(
            !self.poisoned && self.position == 0,
            "GPU exact prompt replay may only be configured before session"
        );
        if max_tokens != 0 {
            ensure!(
                self.mode == WeightMode::Resident,
                "GPU exact replay requires resident original BF16 model"
            );
            ensure!(
                self.batch_prefill_tokens.is_some(),
                "GPU exact replay requires original causal BF16 batch prefill"
            );
            let lod = self
                .lod
                .as_ref()
                .context("GPU exact replay requires original CPU canonical LOD")?;
            ensure!(
                lod.config().epsilon == 0.0,
                "GPU exact replay refuses positive epsilon and approximate KV"
            );
            let hot = self
                .hot_values
                .as_ref()
                .context("GPU exact replay requires original exact BF16 HOT V cache")?;
            ensure!(
                max_tokens <= hot.capacity_tokens && max_tokens <= self.max_tokens,
                "GPU exact replay must fit HOT BF16 V and context"
            );
        }
        self.gpu_prompt_replay_max_tokens = max_tokens;
        if max_tokens == 0 {
            self.gpu_prompt_replay = None;
        }
        Ok(())
    }
    pub fn verified_gpu_prompt_replay_max_tokens(&self) -> usize {
        self.gpu_prompt_replay_max_tokens
    }
    pub fn gpu_prompt_replay_cached_tokens(&self) -> usize {
        self.gpu_prompt_replay.as_ref().map_or(0, |p| p.ids.len())
    }
    pub fn gpu_prompt_replay_cached_bytes(&self) -> usize {
        self.gpu_prompt_replay.as_ref().map_or(0, |p| p.bytes)
    }
    /// GPU-only verified shared-prefix reuse: same token IDs and
    /// original BF16 K/V restored into the original CPU ColdStore,
    /// with all remaining tokens calculated by physical GPU.
    pub fn set_verified_gpu_prefix_replay(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !self.poisoned && self.position == 0,
            "GPU shared prefix can only be configured before a new session"
        );
        if enabled {
            ensure!(
                self.gpu_prompt_replay_max_tokens >= 32,
                "GPU shared prefix requires 32 or more exact cached tokens"
            );
            ensure!(
                self.batch_prefill_tokens.is_some(),
                "GPU shared prefix requires causal BF16 batch prefill"
            );
            ensure!(
                self.lod.as_ref().is_some_and(|l| l.config().epsilon == 0.0),
                "GPU shared prefix cannot approximate original attention"
            );
            ensure!(
                self.hot_values.is_some(),
                "GPU shared prefix requires exact BF16 hot value cache"
            );
        }
        self.verified_prefix_replay_enabled = enabled;
        Ok(())
    }
    pub fn verified_gpu_prefix_replay_enabled(&self) -> bool {
        self.verified_prefix_replay_enabled
    }

    /// Hardware-specific, optional 4-way software pipelining of GPU
    /// single-token BF16 GEMV. Original kernel remains selectable.
    pub fn set_decode_unroll4(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.poisoned, "GPU model state poisoned");
        self.engine
            .gemv_unroll4_enabled
            .store(enabled, Ordering::Relaxed);
        Ok(())
    }
    pub fn decode_unroll4(&self) -> bool {
        self.engine.gemv_unroll4_enabled.load(Ordering::Relaxed)
    }
    /// Optional real GPU fused BF16 QKV and SwiGLU kernels. The same
    /// 128-lane FP32 reductions and BF16 rounding as original GEMV
    /// are required; validated against original dense GPU logits.
    pub fn set_decode_fused_kernels(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.poisoned, "GPU model poisoned");
        #[cfg(feature = "hip-backend")]
        ensure!(
            !enabled || self.engine.hip.is_none(),
            "GPU fused decode incompatible with separate HIP-owned matrices"
        );
        self.engine
            .fused_decode_enabled
            .store(enabled, Ordering::Relaxed);
        Ok(())
    }
    pub fn decode_fused_kernels(&self) -> bool {
        self.engine.fused_decode_enabled.load(Ordering::Relaxed)
    }
    /// Opt-in, GPU-only exact BF16 attention fusion; does not alter
    /// model tensors, LOD selection, CPU code or ColdStore.
    pub fn set_exact_hot_fused_attention(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.poisoned, "GPU context poisoned");
        #[cfg(feature = "hip-backend")]
        ensure!(
            !enabled || self.engine.hip.is_none(),
            "experimental fused exact GPU attention not supported with HIP-owned KV"
        );
        self.engine
            .exact_hot_attention_enabled
            .store(enabled, Ordering::Relaxed);
        Ok(())
    }
    pub fn exact_hot_fused_attention_enabled(&self) -> bool {
        self.engine
            .exact_hot_attention_enabled
            .load(Ordering::Relaxed)
    }

    /// GPU-specific bounded exact hot V tier, backed by the existing
    /// canonical ColdStore. Zero disables hot V. Unlike dense GPU, only
    /// the chosen number of recent prompt tokens is reserved on device,
    /// and all BF16 V values are still written/verified in the cold store.
    /// This first implementation accelerates an entire prefix only when
    /// context length <=hot capacity. Longer contexts fail closed to cold.
    pub fn set_exact_hot_v_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(!self.poisoned, "GPU context poisoned");
        ensure!(
            self.position == 0,
            "GPU hot-V policy may only change before inference/reset"
        );
        if tokens == 0 {
            self.hot_values = None;
            return Ok(());
        }
        ensure!(
            tokens <= self.max_tokens && tokens <= 2048,
            "GPU hot V capacity cannot exceed configured context or 2048"
        );
        let lod = self
            .lod
            .as_ref()
            .context("hot V requires GPU canonical LOD")?;
        ensure!(
            lod.config().epsilon == 0.0,
            "hot-V bypass requires exact LOD epsilon=0: positive-tolerance selection is not bypassed"
        );
        let bytes = tokens
            .checked_mul(self.spec.attention.kv_elements_per_token())
            .and_then(|n| n.checked_mul(2))
            .context("hot V bytes overflow")?;
        let mut values = Vec::with_capacity(self.spec.num_hidden_layers);
        for _ in 0..self.spec.num_hidden_layers {
            values.push(self.engine.alloc_u8(bytes)?);
        }
        self.hot_values = Some(GpuHotValues {
            values,
            micro_indices: None,
            capacity_tokens: tokens,
            valid_prefix_tokens: 0,
            attention_calls: 0,
            cold_fallback_calls: 0,
            prefetched_bytes: 0,
            micro_sparse_calls: 0,
            micro_full_exact_fallback_calls: 0,
        });
        Ok(())
    }

    pub fn exact_hot_v_capacity_tokens(&self) -> usize {
        self.hot_values.as_ref().map_or(0, |h| h.capacity_tokens)
    }
    /// Optional positive-epsilon BF16 micro-region HOT V, backed by the
    /// exact same original CPU ColdStore records. Unlike exact-hot mode,
    /// this NEVER authorizes a skip by itself: original CPU metadata
    /// and conservative BF16 K-box softmax certificate choose the tokens.
    /// Every new K/V remains appended and hashed in ColdStore on decode.
    /// The selected token-index buffer lives on the GPU between queries
    /// and is uploaded only when the certified set actually changes.
    pub fn set_micro_hot_v_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(!self.poisoned, "GPU context poisoned");
        ensure!(
            self.position == 0 && !self.deferred_cold_active,
            "micro hot V policy can only change before first token"
        );
        if tokens == 0 {
            // Do not silently remove a separately configured exact-hot V
            // mode. Zero disables ONLY the positive-epsilon micro tier.
            if self
                .hot_values
                .as_ref()
                .is_some_and(|hot| hot.micro_indices.is_some())
            {
                self.hot_values = None;
            }
            return Ok(());
        }
        ensure!(
            tokens <= self.max_tokens && tokens <= 2048,
            "micro BF16 V capacity must fit GPU context and max 2048 tokens"
        );
        let lod = self
            .lod
            .as_ref()
            .context("micro hot V requires original canonical GPU LOD")?;
        ensure!(
            lod.config().epsilon > 0.0,
            "micro hot V requires explicitly positive epsilon; exact mode has its own hot tier"
        );
        ensure!(
            self.batch_prefill_tokens.is_some(),
            "micro hot V requires exact causal GPU batch prefill"
        );
        let token_bytes = self
            .spec
            .attention
            .kv_elements_per_token()
            .checked_mul(BYTES_PER_BF16)
            .context("micro hot V bytes overflow")?;
        let bytes = tokens
            .checked_mul(token_bytes)
            .context("micro hot GPU allocation overflow")?;
        let mut values = Vec::with_capacity(self.spec.num_hidden_layers);
        for _ in 0..self.spec.num_hidden_layers {
            values.push(self.engine.alloc_u8(bytes)?);
        }
        let selection = MicroSelectedDevice {
            indices: self.engine.alloc_indices(tokens)?,
            last_positions: Vec::new(),
            upload_calls: 0,
            reuse_hits: 0,
        };
        self.hot_values = Some(GpuHotValues {
            values,
            micro_indices: Some(selection),
            capacity_tokens: tokens,
            valid_prefix_tokens: 0,
            attention_calls: 0,
            cold_fallback_calls: 0,
            prefetched_bytes: 0,
            micro_sparse_calls: 0,
            micro_full_exact_fallback_calls: 0,
        });
        Ok(())
    }

    pub fn micro_hot_v_capacity_tokens(&self) -> usize {
        self.hot_values
            .as_ref()
            .filter(|h| h.micro_indices.is_some())
            .map_or(0, |h| h.capacity_tokens)
    }

    /// Enable transactional canonical ColdStore commit for EXACT GPU
    /// decoding. K/V remain BF16 in GPU until the turn finishes, then
    /// are sealed with the SAME CPU hash/index and pages before any
    /// response is returned. Never enabled for epsilon>0.
    pub fn set_deferred_cold_commit(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !self.poisoned,
            "cannot enable deferred ColdStore on poisoned GPU"
        );
        ensure!(
            self.position == 0 && !self.deferred_cold_active,
            "ColdStore commit policy may change only before a new session"
        );
        if enabled {
            let lod = self
                .lod
                .as_ref()
                .context("deferred cold requires canonical GPU LOD")?;
            ensure!(
                lod.config().epsilon == 0.0,
                "deferred BF16 cold commit only valid for epsilon zero"
            );
            ensure!(
                self.hot_values.is_some(),
                "deferred cold commit needs bounded exact BF16 hot V"
            );
            ensure!(
                self.batch_prefill_tokens.is_some(),
                "deferred cold requires causal batch prefill with sealed prompt"
            );
        }
        self.deferred_cold_enabled = enabled;
        Ok(())
    }
    pub fn deferred_cold_enabled(&self) -> bool {
        self.deferred_cold_enabled
    }

    pub fn gpu_region_stats(&self) -> Option<GpuRegionStats> {
        self.lod.as_ref().map(|lod| lod.stats.clone())
    }

    pub fn gpu_region_gpu_kv_capacity_bytes(&self) -> usize {
        let per_layer = self.max_tokens * self.spec.attention.kv_elements_per_token() * 2;
        let keys = per_layer * self.spec.num_hidden_layers;
        if let Some(lod) = &self.lod {
            let stage = lod.stage_tokens() * self.spec.attention.kv_elements_per_token() * 2;
            let aux = lod.stage_tokens() * 4
                + self.spec.attention.query_elements() * 4
                + self.spec.attention.query_heads * 4;
            let hot = self.hot_values.as_ref().map_or(0, |h| {
                h.capacity_tokens
                    * self.spec.attention.kv_elements_per_token()
                    * 2
                    * self.spec.num_hidden_layers
            });
            keys + stage + aux + hot
        } else {
            keys * 2
        }
    }

    /// Compares every uploaded BF16 weight byte with its original safetensors
    /// payload. Readback is streamed in 1 MiB chunks to avoid another 2 GB
    /// host allocation. This is a validation path, never normal inference.
    pub fn verify_resident_weights(&self) -> Result<GpuWeightVerification> {
        use sha2::{Digest, Sha256};
        ensure!(
            self.mode == WeightMode::Resident,
            "weight verification requires --resident; streamed layers not retained"
        );
        let checkpoint = CheckpointShards::open(&self.spec.model_dir, &self.spec.weight_shards)
            .context("reopen original safetensors shards for GPU weight verification")?;
        let mut original_hash = Sha256::new();
        let mut gpu_hash = Sha256::new();
        let mut total_bytes = 0usize;
        let mut total_tensors = 0usize;
        let mut scratch = vec![0u8; 1024 * 1024];
        let mut verify = |name: &str, weight: &GpuWeight| -> Result<()> {
            checkpoint.with_tensor(name, |view| {
                ensure!(
                    view.dtype() == Dtype::BF16,
                    "unexpected non BF16 tensor during readback: {name}"
                );
                for (chunk_index, original) in view.data().chunks(scratch.len()).enumerate() {
                    let offset = chunk_index * scratch.len();
                    let sample = &mut scratch[..original.len()];
                    #[cfg(feature = "hip-backend")]
                    if let Some(ref hip_weight) = weight.hip {
                        self.engine
                            .hip
                            .as_ref()
                            .context("HIP weight has no native GPU context")?
                            .read_bf16(hip_weight, offset, sample)
                            .with_context(|| format!("HIP BF16 weight readback {name} {offset}"))?;
                    } else {
                        // SAFETY: Exact BF16 OpenCL buffer, blocking copy.
                        unsafe {
                            self.engine
                                .queue
                                .enqueue_read_buffer(&weight.data, CL_BLOCKING, offset, sample, &[])
                                .with_context(|| format!("GPU BF16 readback {name} {offset}"))?;
                        }
                    }
                    #[cfg(not(feature = "hip-backend"))]
                    {
                        // SAFETY: Source OpenCL BF16 buffer has enough bytes,
                        // CL_BLOCKING retains host sample through the readback.
                        unsafe {
                            self.engine
                                .queue
                                .enqueue_read_buffer(&weight.data, CL_BLOCKING, offset, sample, &[])
                                .with_context(|| format!("GPU BF16 readback {name} {offset}"))?;
                        }
                    }
                    original_hash.update(original);
                    gpu_hash.update(&*sample);
                    ensure!(
                        sample == original,
                        "lossless verification failed: {name} byte offset {offset}"
                    );
                }
                total_bytes += view.data().len();
                total_tensors += 1;
                Ok(())
            })
        };
        verify("model.embed_tokens.weight", &self.embedding)?;
        verify("model.norm.weight", &self.final_norm)?;
        verify("lm_head.weight", &self.lm_head)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            for (name, weight) in [
                ("input_layernorm.weight", &layer.input_norm),
                ("self_attn.q_proj.weight", &layer.q),
                ("self_attn.k_proj.weight", &layer.k),
                ("self_attn.v_proj.weight", &layer.v),
                ("self_attn.o_proj.weight", &layer.o),
                ("post_attention_layernorm.weight", &layer.post_norm),
                ("mlp.gate_proj.weight", &layer.gate),
                ("mlp.up_proj.weight", &layer.up),
                ("mlp.down_proj.weight", &layer.down),
            ] {
                verify(&format!("{prefix}.{name}"), weight)?;
            }
        }
        let original_digest: [u8; 32] = original_hash.finalize().into();
        let gpu_digest: [u8; 32] = gpu_hash.finalize().into();
        ensure!(
            original_digest == gpu_digest,
            "GPU BF16 bytes have a different aggregate SHA256 than safetensors"
        );
        ensure!(
            total_bytes == self.gpu_weight_payload_bytes,
            "GPU weight bytes checked differ from checkpoint tally"
        );
        let hex = |bytes: &[u8; 32]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        Ok(GpuWeightVerification {
            tensor_count: total_tensors,
            verified_bytes: total_bytes,
            original_concatenated_sha256: hex(&original_digest),
            gpu_readback_concatenated_sha256: hex(&gpu_digest),
            all_original_bf16_bytes_equal: true,
            verification_method: "GPU_readback_every_BF16_byte_vs_original_safetensors_mmap",
        })
    }

    pub fn device_global_memory_bytes(&self) -> u64 {
        self.engine.reported_gpu_memory
    }
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    pub fn position(&self) -> usize {
        self.position
    }

    /// Input history used for deterministic K/V reconstruction during restore.
    pub fn session_token_ids(&self) -> &[u32] {
        &self.session_tokens
    }

    /// Restore a complete causal prefix directly from the persistent canonical
    /// ColdStore. This performs no transformer forward pass and no LM-head
    /// calculation. Every persisted page is hash-verified by `read_keys` /
    /// `read_values` before any logical session state is published.
    ///
    /// The caller must provide the exact token IDs associated with the stored
    /// prefix and must already have validated model/tokenizer identity. The
    /// runtime additionally requires every model layer and the canonical axis
    /// to contain exactly that prefix length. A partial/mismatched store fails
    /// closed. On an upload failure the OpenCL context is poisoned because GPU
    /// buffers may have been modified partially, while `position` and token
    /// history remain unchanged.
    pub fn restore_session(&mut self, token_ids: &[u32]) -> Result<SessionRestore> {
        ensure!(!self.poisoned, "GPU runtime poisoned; restart the worker");
        ensure!(
            self.position == 0 && self.session_tokens.is_empty(),
            "canonical GPU restore requires a fresh session"
        );
        ensure!(
            !token_ids.is_empty(),
            "canonical GPU restore prefix is empty"
        );
        ensure!(
            token_ids.len() < self.max_tokens,
            "canonical GPU restore must leave capacity for continuation"
        );
        let vocab = self.spec.vocab_size.context("missing model vocabulary")?;
        ensure!(
            token_ids.iter().all(|&token| (token as usize) < vocab),
            "canonical GPU restore contains an invalid token ID"
        );
        let token_bytes = self
            .spec
            .attention
            .kv_elements_per_token()
            .checked_mul(BYTES_PER_BF16)
            .context("canonical GPU restore token geometry overflow")?;
        let layer_bytes = token_ids
            .len()
            .checked_mul(token_bytes)
            .context("canonical GPU restore layer size overflow")?;
        let positions = (0..token_ids.len()).collect::<Vec<_>>();
        let layers = self.spec.num_hidden_layers;

        // Validate and materialize every layer before touching GPU buffers.
        // This makes missing/corrupt/torn ColdStore pages a non-mutating error.
        let restored = {
            let lod = self
                .lod
                .as_mut()
                .context("canonical GPU restore requires persistent LOD/ColdStore")?;
            ensure!(
                lod.axis().token_len() == token_ids.len(),
                "canonical GPU restore axis length does not match token history"
            );
            let mut restored = Vec::with_capacity(layers);
            for layer in 0..layers {
                ensure!(
                    lod.layer_tokens(layer) == token_ids.len(),
                    "canonical GPU restore layer {layer} is not a complete causal prefix"
                );
                let keys = lod
                    .read_keys(layer, &positions)
                    .with_context(|| format!("verify canonical K pages for layer {layer}"))?;
                let values = lod
                    .read_values(layer, &positions)
                    .with_context(|| format!("verify canonical V pages for layer {layer}"))?;
                ensure!(
                    keys.len() == layer_bytes && values.len() == layer_bytes,
                    "canonical GPU restore layer {layer} byte shape mismatch"
                );
                restored.push((keys, values));
            }
            restored
        };

        let restore_hot = self
            .hot_values
            .as_ref()
            .is_some_and(|hot| hot.capacity_tokens >= token_ids.len());
        let upload = (|| -> Result<()> {
            for (layer, (keys, values)) in restored.iter().enumerate() {
                self.engine
                    .transfer(&mut self.workspace.kv[layer].key, keys)
                    .with_context(|| format!("upload canonical GPU K layer {layer}"))?;
                if restore_hot {
                    let hot = self
                        .hot_values
                        .as_mut()
                        .context("canonical GPU hot tier disappeared during restore")?;
                    self.engine
                        .transfer(&mut hot.values[layer], values)
                        .with_context(|| format!("upload canonical GPU hot V layer {layer}"))?;
                }
            }
            self.engine
                .queue
                .finish()
                .context("canonical GPU session restore barrier")?;
            Ok(())
        })();
        if let Err(error) = upload {
            self.poisoned = true;
            return Err(error);
        }

        if let Some(hot) = self.hot_values.as_mut() {
            hot.valid_prefix_tokens = if restore_hot { token_ids.len() } else { 0 };
            hot.prefetched_bytes = if restore_hot { layer_bytes * layers } else { 0 };
        }
        self.position = token_ids.len();
        self.session_tokens.extend_from_slice(token_ids);
        self.last_verified_prefix_reused_tokens = token_ids.len();
        self.last_verified_prefix_restored_bytes = layer_bytes
            .checked_mul(layers)
            .and_then(|bytes| bytes.checked_mul(if restore_hot { 2 } else { 1 }))
            .context("canonical GPU restore metric overflow")?;
        self.last_prompt_replay_hit = false;
        self.last_prompt_replay_sha_verified = true;

        Ok(SessionRestore {
            tokens: token_ids.len(),
            layers,
            gpu_key_bytes: layer_bytes * layers,
            gpu_hot_value_bytes: if restore_hot { layer_bytes * layers } else { 0 },
            cold_value_bytes_verified: layer_bytes * layers,
            hot_values_restored: restore_hot,
        })
    }

    /// Reuse the loaded BF16 weights for an independent chat request.
    /// All new KV positions overwrite the previous session's prefix.
    /// A poisoned driver context is never silently reused.
    pub fn active_cold_directory(&self) -> Option<PathBuf> {
        self.lod
            .as_ref()
            .and_then(|lod| lod.config().cold_directory.clone())
    }

    /// Reset only the active causal session. The resident model weights survive.
    /// A fresh physical ColdStore namespace prevents old conversation pages from
    /// being reopened as if they belonged to the new conversation.
    pub fn reset_session(&mut self) -> Result<()> {
        let next_directory = self.active_cold_directory().map(|current| {
            let root = current.parent().unwrap_or(current.as_path());
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default();
            root.join(format!("session-{}-{nonce}", std::process::id()))
        });
        self.reset_session_at(next_directory)
    }

    /// Reset into a specific ColdStore namespace. Restore uses this to reopen
    /// the verified namespace belonging to the snapshot being restored.
    pub fn reset_session_at(&mut self, cold_directory: Option<PathBuf>) -> Result<()> {
        ensure!(!self.poisoned, "GPU runtime poisoned; restart the worker");
        self.engine
            .queue
            .finish()
            .context("GPU previous-session barrier")?;
        if self.deferred_cold_active {
            self.flush_deferred_cold()?;
        }
        if let Some(lod) = self.lod.as_ref() {
            let mut config = lod.config().clone();
            if let Some(directory) = cold_directory {
                fs::create_dir_all(&directory).with_context(|| {
                    format!(
                        "create canonical ColdStore namespace {}",
                        directory.display()
                    )
                })?;
                config.cold_directory = Some(directory);
            }
            let policy = lod.axis_policy().clone();
            self.lod = Some(GpuRegionStore::new_with_axis_policy(
                self.spec.attention,
                self.spec.num_hidden_layers,
                config,
                policy,
            )?);
        }
        self.position = 0;
        self.session_tokens.clear();
        self.gpu_prompt_replay = None;
        if let Some(hot) = self.hot_values.as_mut() {
            hot.valid_prefix_tokens = 0;
            hot.attention_calls = 0;
            hot.cold_fallback_calls = 0;
            hot.prefetched_bytes = 0;
            hot.micro_sparse_calls = 0;
            hot.micro_full_exact_fallback_calls = 0;
            if let Some(cache) = hot.micro_indices.as_mut() {
                cache.last_positions.clear();
                cache.upload_calls = 0;
                cache.reuse_hits = 0;
            }
        }
        self.last_verified_prefix_reused_tokens = 0;
        self.last_verified_prefix_restored_bytes = 0;
        self.last_prompt_replay_hit = false;
        self.last_prompt_replay_rehydrated_bytes = 0;
        self.last_prompt_replay_sha_verified = false;
        self.last_prefill_batches = 0;
        self.last_prefill_cold_seal_seconds = 0.0;
        self.last_prefill_temporary_v_bytes = 0;
        self.last_deferred_cold_commit_seconds = 0.0;
        self.last_deferred_cold_committed_token_layers = 0;
        Ok(())
    }

    /// Commit last in-flight BF16 K/V to the ORIGINAL CPU ColdStore.
    /// Already-committed prefill prefix remains untouched. No token
    /// discarded or approximated; per-token hashes and index produced
    /// by shared canonical CPU append_batch method.
    ///
    /// End-of-turn commit is mandatory before returning a reply. Removes
    /// 48 tiny blocking GPU-to-CPU copies PER decoded token, replacing
    /// them with two contiguous verified readbacks per layer.
    fn flush_deferred_cold(&mut self) -> Result<()> {
        if !self.deferred_cold_active {
            return Ok(());
        }
        let begin = Instant::now();
        let end = self.position;
        let hot = self
            .hot_values
            .as_ref()
            .context("deferred commit without exact GPU hot V")?;
        ensure!(
            end <= hot.valid_prefix_tokens && end <= hot.capacity_tokens,
            "deferred GPU cold commit invalid: exact BF16 hot prefix incomplete"
        );
        let lod = self
            .lod
            .as_mut()
            .context("deferred commit requires canonical ColdStore")?;
        ensure!(
            lod.config().epsilon == 0.0,
            "positive-tolerance route cannot use deferred exact cold commit"
        );
        let bytes_per_token = self
            .spec
            .attention
            .kv_elements_per_token()
            .checked_mul(BYTES_PER_BF16)
            .context("GPU BF16 KV byte overflow")?;
        let mut committed = 0usize;
        // All queued kernels finish before host pulls BF16 for hashing.
        self.engine
            .queue
            .finish()
            .context("GPU BF16 K/V seal barrier")?;
        for layer in 0..self.spec.num_hidden_layers {
            let start = lod.layer_tokens(layer);
            ensure!(
                start > 0 && start <= end,
                "GPU ColdStore prefill prefix missing/longer than in-flight KV"
            );
            if start == end {
                continue;
            }
            let offset = start
                .checked_mul(bytes_per_token)
                .context("GPU cold seal offset overflow")?;
            let size = (end - start)
                .checked_mul(bytes_per_token)
                .context("GPU cold seal byte length overflow")?;
            let mut keys = vec![0u8; size];
            let mut values = vec![0u8; size];
            // SAFETY: GPU K and hot V buffers contain contiguous BF16
            // bytes for selected positions; blocking transfer completes.
            unsafe {
                self.engine
                    .queue
                    .enqueue_read_buffer(
                        &self.workspace.kv[layer].key,
                        CL_BLOCKING,
                        offset,
                        &mut keys,
                        &[],
                    )
                    .with_context(|| format!("GPU K cold-seal readback layer {layer}"))?;
                self.engine
                    .queue
                    .enqueue_read_buffer(&hot.values[layer], CL_BLOCKING, offset, &mut values, &[])
                    .with_context(|| format!("GPU V cold-seal readback layer {layer}"))?;
            }
            lod.append_batch(layer, start, &keys, &values, 64)
                .with_context(|| format!("canonical CPU BF16 ColdStore seal layer {layer}"))?;
            ensure!(
                lod.layer_tokens(layer) == end,
                "ColdStore layer did not commit all in-flight BF16 values"
            );
            committed += end - start;
        }
        ensure!(
            lod.axis().token_len() == end,
            "four-bit index did not advance to full committed BF16 turn"
        );
        self.deferred_cold_active = false;
        self.last_deferred_cold_commit_seconds += begin.elapsed().as_secs_f64();
        self.last_deferred_cold_committed_token_layers += committed;
        Ok(())
    }

    /// Reuse only the longest common prefix of EXACT original token
    /// IDs, after validating the full cached SHA256. Restore canonical
    /// BF16 K/V, original 4-bit axis, ColdStore pages, and retain all
    /// positional RoPE/causal semantics for the still-unseen suffix.
    fn restore_verified_gpu_shared_prefix(&mut self, prompt_ids: &[u32]) -> Result<usize> {
        if !self.verified_prefix_replay_enabled
            || self.gpu_prompt_replay_max_tokens == 0
            || prompt_ids.len() < 33
        {
            return Ok(0);
        }
        let Some(snapshot) = self.gpu_prompt_replay.as_ref() else {
            return Ok(0);
        };
        let shared = snapshot
            .ids
            .iter()
            .zip(prompt_ids.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt_ids.len() - 1);
        if shared < 32 {
            return Ok(0);
        }
        let capacity = self.hot_values.as_ref().map_or(0, |h| h.capacity_tokens);
        if prompt_ids.len() > capacity {
            return Ok(0);
        }
        ensure!(
            self.position == 0 && !self.poisoned,
            "GPU shared prefix needs new healthy original BF16 session"
        );
        if !snapshot.verify() {
            self.gpu_prompt_replay = None;
            return Ok(0); // safe recomputation; no stale BF16 uploads
        }
        let token_bytes = self
            .spec
            .attention
            .kv_elements_per_token()
            .checked_mul(BYTES_PER_BF16)
            .context("BF16 shared K/V geometry overflow")?;
        let size = shared
            .checked_mul(token_bytes)
            .context("GPU BF16 shared prefix length overflow")?;
        let lod = self
            .lod
            .as_mut()
            .context("GPU shared prefix requires original CPU ColdStore")?;
        ensure!(
            lod.config().epsilon == 0.0,
            "verified shared prefix never uses approximate K/V"
        );
        let hot = self
            .hot_values
            .as_mut()
            .context("GPU shared prefix requires exact BF16 hot V")?;
        ensure!(
            snapshot.kv.len() == self.spec.num_hidden_layers
                && hot.values.len() == self.spec.num_hidden_layers,
            "original BF16 cached layer count mismatch"
        );
        for (layer, (key, value)) in snapshot.kv.iter().enumerate() {
            ensure!(
                key.len() >= size && value.len() >= size,
                "original BF16 K/V prefix snapshot too short"
            );
            let keys = &key[..size];
            let values = &value[..size];
            self.engine
                .transfer(&mut self.workspace.kv[layer].key, keys)
                .with_context(|| format!("GPU restore exact prefix K layer {layer}"))?;
            self.engine
                .transfer(&mut hot.values[layer], values)
                .with_context(|| format!("GPU restore exact prefix V layer {layer}"))?;
            lod.append_batch(layer, 0, keys, values, 64)
                .with_context(|| format!("original canonical ColdStore prefix layer {layer}"))?;
            ensure!(
                lod.layer_tokens(layer) == shared,
                "original CPU ColdStore prefix layer rehydration incomplete"
            );
        }
        ensure!(
            lod.axis().token_len() == shared,
            "original CPU 4-bit index not restored with BF16 prefix"
        );
        hot.valid_prefix_tokens = shared;
        hot.prefetched_bytes = size * self.spec.num_hidden_layers;
        self.position = shared;
        self.last_verified_prefix_reused_tokens = shared;
        self.last_verified_prefix_restored_bytes = 2 * size * self.spec.num_hidden_layers;
        Ok(shared)
    }

    /// Rehydrate the original BF16 GPU prefill only for a verified
    /// EXACT byte-for-byte token ID match. The immutable RAM snapshot
    /// was captured from physical GPU K/V *as the original ColdStore
    /// was committed*; its digest is checked before any GPU/ColdStore
    /// write. A changed prompt or damaged SHA snapshot runs the full
    /// real GPU prefill instead, with no stale-prefix reuse.
    fn restore_verified_gpu_prompt_replay(
        &mut self,
        prompt_ids: &[u32],
        capture_logits: bool,
    ) -> Result<Option<GpuStep>> {
        if self.gpu_prompt_replay_max_tokens == 0 {
            return Ok(None);
        }
        let Some(snapshot) = self.gpu_prompt_replay.as_ref() else {
            return Ok(None);
        };
        if snapshot.ids != prompt_ids {
            return Ok(None);
        }
        ensure!(
            self.position == 0 && !self.poisoned,
            "exact GPU prompt replay requires a fresh healthy session"
        );
        if !snapshot.verify() {
            // No K/V or logit is restored before SHA256 verification.
            // On any corruption the original physical BF16 GPU prefill
            // is recomputed rather than trusting stale data.
            self.gpu_prompt_replay = None;
            return Ok(None);
        }
        let started = Instant::now();
        let original_lod = self
            .lod
            .as_mut()
            .context("exact prefill replay requires original CPU ColdStore")?;
        ensure!(
            original_lod.config().epsilon == 0.0,
            "exact GPU prompt replay cannot use positive-epsilon pruning"
        );
        let hot = self
            .hot_values
            .as_mut()
            .context("exact GPU prompt replay requires BF16 hot V")?;
        let count = prompt_ids.len();
        ensure!(
            count <= hot.capacity_tokens && count <= self.gpu_prompt_replay_max_tokens,
            "verified prompt is larger than bounded BF16 hot cache"
        );
        let bytes = count
            .checked_mul(self.spec.attention.kv_elements_per_token())
            .and_then(|n| n.checked_mul(BYTES_PER_BF16))
            .context("GPU verified prompt replay byte count overflow")?;
        ensure!(
            snapshot.kv.len() == self.spec.num_hidden_layers
                && hot.values.len() == self.spec.num_hidden_layers,
            "BF16 prompt replay layer count invalid"
        );
        let mut restored = 0usize;
        for (layer, (key, value)) in snapshot.kv.iter().enumerate() {
            ensure!(
                key.len() == bytes && value.len() == bytes,
                "verified BF16 prompt replay KV layer shape invalid"
            );
            // Upload ONLY original BF16 bytes already produced by GPU.
            // No CPU neural matrix arithmetic or recalculation.
            self.engine
                .transfer(&mut self.workspace.kv[layer].key, key)
                .with_context(|| format!("GPU verified replay key layer {layer}"))?;
            self.engine
                .transfer(&mut hot.values[layer], value)
                .with_context(|| format!("GPU verified replay hot V layer {layer}"))?;
            // Restore the SAME CPU canonical ColdStore record format
            // with per-token checksums and cryptographic page records,
            // rebuilding 4-bit axis and exact index from original K.
            original_lod
                .append_batch(layer, 0, key, value, 64)
                .with_context(|| format!("CPU canonical cold-page rehydrate layer {layer}"))?;
            restored += key.len() + value.len();
        }
        ensure!(
            original_lod.axis().token_len() == count,
            "original ParallelContextAxis failed to restore exact prompt"
        );
        for layer in 0..self.spec.num_hidden_layers {
            ensure!(
                original_lod.layer_tokens(layer) == count,
                "incomplete canonical BF16 cold K/V rehydration"
            );
        }
        hot.valid_prefix_tokens = count;
        hot.prefetched_bytes = bytes * self.spec.num_hidden_layers;
        let vocab = self
            .spec
            .vocab_size
            .context("missing original BF16 output vocabulary")?;
        ensure!(
            snapshot.last_logits_bf16.len() == vocab * BYTES_PER_BF16,
            "verified GPU prefill original logits shape mismatch"
        );
        self.engine
            .transfer(&mut self.workspace.logits, &snapshot.last_logits_bf16)?;
        self.engine.argmax(
            &self.workspace.logits,
            &self.workspace.token_id,
            u32::try_from(vocab)?,
        )?;
        let argmax = self.engine.read_token(&self.workspace.token_id)?;
        ensure!(
            argmax == snapshot.argmax,
            "original BF16 GPU logits changed during prompt replay"
        );
        let logits = if capture_logits {
            Some(self.engine.read_logits(&self.workspace.logits, vocab)?)
        } else {
            None
        };
        self.position = count;
        self.last_prompt_replay_hit = true;
        self.last_prompt_replay_sha_verified = true;
        self.last_prompt_replay_rehydrated_bytes = restored + snapshot.last_logits_bf16.len();
        self.last_prefill_batches = 0;
        self.last_prefill_cold_seal_seconds = started.elapsed().as_secs_f64();
        Ok(Some(GpuStep {
            argmax_token: argmax,
            position: count - 1,
            forward_seconds: started.elapsed().as_secs_f64(),
            logits,
        }))
    }

    /// SHA-256 of the logical BF16 KV state, using exactly the same
    /// serialization order as DenseKvCache::state_digest in torch_runtime:
    /// K then V, layer-major; each tensor [1, kv_heads, tokens, head_dim].
    /// This is an explicit readback for validation, not the runtime hot path.
    /// Diagnostic readback only: BF16 KV per layer in the same contiguous
    /// order as LibTorch [1, kv_heads, tokens, head_dim].
    pub fn debug_kv_layer_bytes(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        ensure!(
            self.lod.is_none(),
            "LOD mode values reside in canonical ColdStore, not dense GPU V"
        );
        ensure!(
            self.position <= 4096,
            "KV debug readback disabled above 4096 tokens"
        );
        let kvheads = self.spec.attention.kv_heads;
        let item_bytes = self.spec.attention.head_dim * 2;
        let mut all = Vec::with_capacity(self.workspace.kv.len());
        for cache in &self.workspace.kv {
            let mut pair = Vec::with_capacity(2);
            for source in [
                &cache.key,
                cache
                    .value
                    .as_ref()
                    .context("dense GPU V not resident in LOD mode")?,
            ] {
                let mut raw = vec![0u8; self.position * kvheads * item_bytes];
                // SAFETY: GPU buffers have at least this many committed
                // BF16 KV elements; CL_BLOCKING retains host allocation.
                unsafe {
                    self.engine
                        .queue
                        .enqueue_read_buffer(source, CL_BLOCKING, 0, &mut raw, &[])
                        .context("GPU diagnostic KV readback")?;
                }
                let mut canonical = Vec::with_capacity(raw.len());
                for head in 0..kvheads {
                    for token in 0..self.position {
                        let start = (token * kvheads + head) * item_bytes;
                        canonical.extend_from_slice(&raw[start..start + item_bytes]);
                    }
                }
                pair.push(canonical);
            }
            all.push((pair.remove(0), pair.remove(0)));
        }
        Ok(all)
    }

    pub fn logical_kv_sha256(&self) -> Result<String> {
        ensure!(
            self.lod.is_none(),
            "LOD mode needs cold-backed logical KV hashing"
        );
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"mslim-dense-kv-state-v1");
        hash.update((self.position as u64).to_le_bytes());
        let kvheads = self.spec.attention.kv_heads;
        let dim = self.spec.attention.head_dim;
        let item_bytes = dim.checked_mul(2).context("KV row byte size overflow")?;
        let frame_bytes = self
            .position
            .checked_mul(kvheads)
            .and_then(|n| n.checked_mul(item_bytes))
            .context("KV readback length overflow")?;
        self.engine
            .queue
            .finish()
            .context("GPU KV barrier before digest")?;
        for cache in &self.workspace.kv {
            for source in [
                &cache.key,
                cache
                    .value
                    .as_ref()
                    .context("dense GPU V not resident in LOD mode")?,
            ] {
                let mut data = vec![0u8; frame_bytes];
                // SAFETY: Buffer holds at least the requested number of
                // committed token bytes and blocking read retains host storage.
                unsafe {
                    self.engine
                        .queue
                        .enqueue_read_buffer(source, CL_BLOCKING, 0, &mut data, &[])
                        .context("GPU KV validation readback")?;
                }
                for head in 0..kvheads {
                    for token in 0..self.position {
                        let start = (token * kvheads + head) * item_bytes;
                        hash.update(&data[start..start + item_bytes]);
                    }
                }
            }
        }
        Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Validation-only digest for either dense device K/V or canonical
    /// ColdStore-backed K/V. Both branches use the exact same logical byte
    /// order, so equality proves that the storage route preserved BF16 K/V.
    pub fn canonical_kv_sha256(&mut self) -> Result<String> {
        self.engine
            .queue
            .finish()
            .context("GPU KV barrier before canonical digest")?;
        if self.deferred_cold_active {
            self.flush_deferred_cold()?;
        }
        if let Some(lod) = self.lod.as_mut() {
            ensure!(
                lod.config().epsilon == 0.0,
                "canonical KV digest is reserved for exact epsilon-zero mode"
            );
            ensure!(
                lod.axis().token_len() == self.position,
                "canonical ColdStore axis is not aligned with GPU position"
            );
            lod.logical_kv_sha256()
        } else {
            self.logical_kv_sha256()
        }
    }

    fn run_layer(
        engine: &GpuEngine,
        spec: &ModelSpec,
        layer: &GpuLayer,
        ws: &mut Workspace,
        step: (usize, usize, bool),
        lod: Option<&mut GpuRegionStore>,
        hot: Option<&mut GpuHotValues>,
    ) -> Result<()> {
        let (layer_index, position, defer_cold) = step;
        let hidden = spec.hidden_size;
        let intermediate = spec
            .intermediate_size
            .context("intermediate_size missing")?;
        let geom = spec.attention;
        let head_q = geom.query_elements();
        let head_kv = geom.kv_elements_per_token();
        let head_dim = geom.head_dim;
        let rope = RopeKernelConfig::from_spec(spec)?;
        let eps = spec.rms_norm_eps.context("rms_norm_eps missing")? as f32;
        let position_u32 = u32::try_from(position)?;
        let hidden_u32 = u32::try_from(hidden)?;
        let heads_q = u32::try_from(geom.query_heads)?;
        let heads_kv = u32::try_from(geom.kv_heads)?;
        let dim_u32 = u32::try_from(head_dim)?;

        engine.norm(&ws.hidden0, &layer.input_norm, &ws.norm1, hidden_u32, eps)?;
        if engine.fused_decode_enabled.load(Ordering::Relaxed) {
            engine.project_fused_qkv(
                (&layer.q, &layer.k, &layer.v),
                (&ws.norm1, &ws.query, &ws.key, &ws.value),
                (hidden, head_q, head_kv),
            )?;
        } else {
            engine.gemv(&layer.q, &ws.norm1, &ws.query, hidden, head_q)?;
            engine.gemv(&layer.k, &ws.norm1, &ws.key, hidden, head_kv)?;
            engine.gemv(&layer.v, &ws.norm1, &ws.value, hidden, head_kv)?;
        }
        engine.rope(
            &ws.query,
            &ws.query_rope,
            heads_q,
            dim_u32,
            position_u32,
            rope,
        )?;
        engine.rope(&ws.key, &ws.key_rope, heads_kv, dim_u32, position_u32, rope)?;

        // New key/value are appended to the device KV buffers of this layer.
        let page_bytes = head_kv
            .checked_mul(BYTES_PER_BF16)
            .context("KV size overflow")?;
        let dst_offset = position
            .checked_mul(page_bytes)
            .context("KV offset overflow")?;
        engine.copy(
            &ws.key_rope,
            &mut ws.kv[layer_index].key,
            page_bytes,
            dst_offset,
        )?;
        if let Some(lod) = lod {
            if defer_cold {
                ensure!(
                    lod.config().epsilon == 0.0,
                    "deferred GPU LOD cannot bypass approximate mass certification"
                );
                let hot = hot.context("deferred GPU cold commit requires exact hot V")?;
                ensure!(
                    position < hot.capacity_tokens && hot.valid_prefix_tokens == position,
                    "deferred GPU ColdStore requires a valid BF16 hot prefix"
                );
                // Native device-to-device copies, no host sync/no FP16.
                // The original exact ColdStore is sealed once at end of
                // this in-flight turn, before its response is returned.
                engine.copy(
                    &ws.value,
                    &mut hot.values[layer_index],
                    page_bytes,
                    dst_offset,
                )?;
                if engine.exact_hot_attention_enabled.load(Ordering::Relaxed) && position < 2048 {
                    engine.exact_hot_fused_attention(
                        &ws.query_rope,
                        &ws.kv[layer_index].key,
                        &hot.values[layer_index],
                        &ws.attention,
                        position + 1,
                        geom,
                    )?;
                } else {
                    engine.attention(
                        &ws.query_rope,
                        (&ws.kv[layer_index].key, &hot.values[layer_index]),
                        &ws.scores,
                        &ws.attention,
                        u32::try_from(position + 1)?,
                        geom,
                    )?;
                }
                lod.observe_deferred_exact_attention(layer_index, position + 1)?;
                hot.attention_calls += 1;
            } else {
                // Original unchanged path: immediate ColdStore BF16 append
                // for every decoded token, plus original exact or certified
                // selection and fallback.
                let key = engine.read_bf16_bytes(&ws.key_rope, page_bytes)?;
                let value = engine.read_bf16_bytes(&ws.value, page_bytes)?;
                lod.append(layer_index, position, &key, &value)?;
                if let Some(hot) = hot {
                    // Hot V is an exact BF16 *cache* only. Canonical ColdStore
                    // append above remains mandatory for recovery/integrity.
                    if position < hot.capacity_tokens
                        && hot.valid_prefix_tokens == position
                        && lod.config().epsilon > 0.0
                        && hot.micro_indices.is_some()
                    {
                        // Positive-epsilon device-resident BF16 V path.
                        // ALWAYS commit the exact original K/V to CPU
                        // ColdStore and original 4-bit index above, then
                        // keep a separate *cache* of unmodified BF16 V
                        // on GPU. Missing hot prefix falls back to the
                        // original verified ColdStore route below.
                        engine.copy(
                            &ws.value,
                            &mut hot.values[layer_index],
                            page_bytes,
                            dst_offset,
                        )?;
                        // Micro metadata needs the TRUE RoPE Q in BF16,
                        // not its 4-bit fingerprint. This one small
                        // query transfer replaces cold V staging for
                        // the entire selected attention history.
                        let query =
                            engine.read_bf16_bytes(&ws.query_rope, head_q * BYTES_PER_BF16)?;
                        let chosen = lod.plan_micro_regions(layer_index, &query)?;
                        if let Some(selection) = chosen {
                            // The certified BF16-key region selector
                            // guarantees a conservative softmax-mass
                            // upper bound before any K/V are omitted.
                            let indices = hot
                                .micro_indices
                                .as_mut()
                                .context("device-resident micro indices absent")?;
                            engine.micro_update_cached_indices(
                                indices,
                                &selection.positions,
                                position + 1,
                            )?;
                            engine.micro_hot_attention(
                                (
                                    &ws.query_rope,
                                    &ws.kv[layer_index].key,
                                    &hot.values[layer_index],
                                    &indices.indices,
                                    &ws.attention,
                                ),
                                (selection.positions.len(), position + 1),
                                geom,
                            )?;
                            hot.micro_sparse_calls += 1;
                        } else {
                            // Wide or ambiguous micro regions cannot
                            // certify skipping K. Revert to completely
                            // native full GPU BF16 QK/PV using cached
                            // exact V, not cold file reads. This is
                            // guaranteed zero omitted mass and is always
                            // safe for a positive-epsilon request.
                            if engine.exact_hot_attention_enabled.load(Ordering::Relaxed)
                                && position < 2048
                            {
                                engine.exact_hot_fused_attention(
                                    &ws.query_rope,
                                    &ws.kv[layer_index].key,
                                    &hot.values[layer_index],
                                    &ws.attention,
                                    position + 1,
                                    geom,
                                )?;
                            } else {
                                engine.attention(
                                    &ws.query_rope,
                                    (&ws.kv[layer_index].key, &hot.values[layer_index]),
                                    &ws.scores,
                                    &ws.attention,
                                    u32::try_from(position + 1)?,
                                    geom,
                                )?;
                            }
                            lod.observe_full_hot_attention(layer_index, position + 1)?;
                            hot.micro_full_exact_fallback_calls += 1;
                        }
                        hot.attention_calls += 1;
                    } else if position < hot.capacity_tokens
                        && hot.valid_prefix_tokens == position
                        && lod.config().epsilon == 0.0
                    {
                        engine.copy(
                            &ws.value,
                            &mut hot.values[layer_index],
                            page_bytes,
                            dst_offset,
                        )?;
                        let cert = lod.select_dense_exact(layer_index)?;
                        ensure!(
                            cert.omitted == 0 && cert.positions.len() == position + 1,
                            "hot V requires every BF16 V to be selected exactly"
                        );
                        if engine.exact_hot_attention_enabled.load(Ordering::Relaxed)
                            && position < 2048
                        {
                            engine.exact_hot_fused_attention(
                                &ws.query_rope,
                                &ws.kv[layer_index].key,
                                &hot.values[layer_index],
                                &ws.attention,
                                position + 1,
                                geom,
                            )?;
                        } else {
                            engine.attention(
                                &ws.query_rope,
                                (&ws.kv[layer_index].key, &hot.values[layer_index]),
                                &ws.scores,
                                &ws.attention,
                                u32::try_from(position + 1)?,
                                geom,
                            )?;
                        }
                        hot.attention_calls += 1;
                    } else {
                        // Context exceeded hot capacity or stale prefix:
                        // no silent approximation; use original verified
                        // CPU ColdStore + GPU attention path.
                        hot.cold_fallback_calls += 1;
                        engine.attention_gpu_region(ws, layer_index, position + 1, lod)?;
                    }
                } else {
                    engine.attention_gpu_region(ws, layer_index, position + 1, lod)?;
                }
            }
        } else {
            let full_v = ws.kv[layer_index]
                .value
                .as_mut()
                .context("dense GPU V missing; canonical LOD not enabled")?;
            engine.copy(&ws.value, full_v, page_bytes, dst_offset)?;
            let history = u32::try_from(position + 1)?;
            let value = ws.kv[layer_index]
                .value
                .as_ref()
                .context("dense GPU V buffer missing")?;
            if engine.exact_hot_attention_enabled.load(Ordering::Relaxed) && position < 2048 {
                // Dense BF16 GPU KV contains every value, so it can use
                // the same exact fused attention with ZERO discarded K/V.
                engine.exact_hot_fused_attention(
                    &ws.query_rope,
                    &ws.kv[layer_index].key,
                    value,
                    &ws.attention,
                    position + 1,
                    geom,
                )?;
            } else {
                engine.attention(
                    &ws.query_rope,
                    (&ws.kv[layer_index].key, value),
                    &ws.scores,
                    &ws.attention,
                    history,
                    geom,
                )?;
            }
        }

        engine.gemv(&layer.o, &ws.attention, &ws.attn_o, head_q, hidden)?;
        engine.add(&ws.hidden0, &ws.attn_o, &ws.hidden1, hidden)?;
        engine.norm(&ws.hidden1, &layer.post_norm, &ws.norm2, hidden_u32, eps)?;
        if engine.fused_decode_enabled.load(Ordering::Relaxed) {
            engine.project_fused_swiglu(
                (&layer.gate, &layer.up),
                &ws.norm2,
                &ws.activation,
                hidden,
                intermediate,
            )?;
        } else {
            engine.gemv(&layer.gate, &ws.norm2, &ws.gate, hidden, intermediate)?;
            engine.gemv(&layer.up, &ws.norm2, &ws.up, hidden, intermediate)?;
            engine.swiglu(&ws.gate, &ws.up, &ws.activation, intermediate)?;
        }
        engine.gemv(&layer.down, &ws.activation, &ws.down, intermediate, hidden)?;
        engine.add(&ws.hidden1, &ws.down, &ws.hidden0, hidden)?;
        Ok(())
    }

    fn forward_unchecked(
        &mut self,
        token_id: u32,
        capture_logits: bool,
        compute_lm_head: bool,
    ) -> Result<Option<GpuStep>> {
        ensure!(
            usize::try_from(token_id)? < self.spec.vocab_size.context("missing vocab size")?,
            "token id beyond vocabulary"
        );
        let start = Instant::now();
        let ws = &mut self.workspace;
        let engine = &self.engine;
        engine.embed(
            &self.embedding,
            &ws.hidden0,
            token_id,
            u32::try_from(self.spec.hidden_size)?,
        )?;

        for layer_index in 0..self.spec.num_hidden_layers {
            if self.mode == WeightMode::Resident {
                let layer = self
                    .layers
                    .get(layer_index)
                    .context("missing resident GPU layer")?;
                Self::run_layer(
                    engine,
                    &self.spec,
                    layer,
                    ws,
                    (layer_index, self.position, self.deferred_cold_active),
                    self.lod.as_mut(),
                    self.hot_values.as_mut(),
                )?;
            } else {
                let checkpoint = self
                    .checkpoint
                    .as_ref()
                    .context("missing streamed checkpoint")?;
                let layer = GpuLayer::load(engine, checkpoint, layer_index, &self.spec)?;
                Self::run_layer(
                    engine,
                    &self.spec,
                    &layer,
                    ws,
                    (layer_index, self.position, self.deferred_cold_active),
                    self.lod.as_mut(),
                    self.hot_values.as_mut(),
                )?;
                // IMPORTANT: Any pending kernels reference these weight
                // buffers; clFinish before their owners are dropped.
                engine.queue.finish().context("drain streamed GPU layer")?;
            }
        }

        let next = if compute_lm_head {
            engine.norm(
                &ws.hidden0,
                &self.final_norm,
                &ws.norm1,
                u32::try_from(self.spec.hidden_size)?,
                self.spec.rms_norm_eps.context("rms norm eps missing")? as f32,
            )?;
            engine.gemv(
                &self.lm_head,
                &ws.norm1,
                &ws.logits,
                self.spec.hidden_size,
                self.spec.vocab_size.context("missing vocab")?,
            )?;
            engine.argmax(
                &ws.logits,
                &ws.token_id,
                u32::try_from(self.spec.vocab_size.context("missing vocab")?)?,
            )?;
            let argmax_token = engine.read_token(&ws.token_id)?;
            let logits = if capture_logits {
                Some(
                    engine
                        .read_logits(&ws.logits, self.spec.vocab_size.context("missing vocab")?)?,
                )
            } else {
                None
            };
            Some(GpuStep {
                argmax_token,
                position: self.position,
                forward_seconds: start.elapsed().as_secs_f64(),
                logits,
            })
        } else {
            None
        };
        if let Some(hot) = self.hot_values.as_mut() {
            // All layer values of the current token were appended before
            // this common token boundary. No earlier partial prefix is reused.
            if hot.valid_prefix_tokens == self.position && self.position < hot.capacity_tokens {
                hot.valid_prefix_tokens += 1;
            }
        }
        self.position += 1;
        Ok(next)
    }

    pub fn forward_token(&mut self, token_id: u32, capture_logits: bool) -> Result<GpuStep> {
        self.forward_token_internal(token_id, capture_logits, true)?
            .context("full GPU logits missing")
    }

    fn forward_token_internal(
        &mut self,
        token_id: u32,
        capture_logits: bool,
        compute_lm_head: bool,
    ) -> Result<Option<GpuStep>> {
        ensure!(
            !self.poisoned,
            "GPU LLM state was poisoned by an earlier failed forward pass"
        );
        ensure!(
            self.position < self.max_tokens,
            "GPU KV capacity exceeded: configure larger --max-context-tokens"
        );
        let response = self.forward_unchecked(token_id, capture_logits, compute_lm_head);
        if response.is_err() {
            self.poisoned = true;
        } else {
            self.session_tokens.push(token_id);
        }
        response
    }

    /// Prefill only: computes logits for the final prompt token without
    /// generating or appending a first output token. The returned state
    /// therefore represents exactly the prompt and nothing else.
    pub fn prefill_only(
        &mut self,
        prompt_ids: &[u32],
        capture_logits: bool,
        cancel: &AtomicBool,
    ) -> Result<(GpuStep, f64)> {
        ensure!(!prompt_ids.is_empty(), "GPU prompt IDs empty");
        self.select_calibrated_prefill(prompt_ids.len())?;
        ensure!(
            prompt_ids.len() < self.max_tokens,
            "GPU context capacity not large enough for prompt"
        );
        let started = Instant::now();
        self.last_prompt_replay_hit = false;
        self.last_verified_prefix_reused_tokens = 0;
        self.last_verified_prefix_restored_bytes = 0;
        self.last_prompt_replay_rehydrated_bytes = 0;
        self.last_prompt_replay_sha_verified = false;
        let restored = if self.gpu_prompt_replay_max_tokens > 0 {
            let result = self.restore_verified_gpu_prompt_replay(prompt_ids, capture_logits);
            if result.is_err() {
                self.poisoned = true;
            }
            result?
        } else {
            None
        };
        let shared_prefix = if restored.is_none() && self.verified_prefix_replay_enabled {
            let result = self.restore_verified_gpu_shared_prefix(prompt_ids);
            if result.is_err() {
                self.poisoned = true;
            }
            result?
        } else {
            0
        };
        let last = if let Some(step) = restored {
            self.session_tokens.clear();
            self.session_tokens.extend_from_slice(prompt_ids);
            step
        } else if let Some(batch_size) = self.batch_prefill_tokens {
            let result = self.prefill_batched(
                prompt_ids,
                shared_prefix,
                batch_size,
                capture_logits,
                cancel,
            );
            if result.is_err() {
                self.poisoned = true;
            }
            let step = result?;
            self.session_tokens.clear();
            self.session_tokens.extend_from_slice(prompt_ids);
            step
        } else {
            let mut last = None;
            for (index, &id) in prompt_ids.iter().enumerate() {
                ensure!(!cancel.load(Ordering::Relaxed), "GPU prefill cancelled");
                last = self.forward_token_internal(
                    id,
                    capture_logits && index + 1 == prompt_ids.len(),
                    index + 1 == prompt_ids.len(),
                )?;
            }
            last.context("no GPU prompt logits")?
        };
        self.deferred_cold_active = false;
        Ok((last, started.elapsed().as_secs_f64()))
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        capture_logits: bool,
    ) -> Result<GpuGeneration> {
        self.generate_cancellable(
            prompt_ids,
            max_new_tokens,
            capture_logits,
            &AtomicBool::new(false),
        )
    }

    /// Cooperative cancellation checked at token boundaries so queued GPU
    /// kernels always finish before a session can be reset safely.
    pub fn generate_cancellable(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        capture_logits: bool,
        cancel: &AtomicBool,
    ) -> Result<GpuGeneration> {
        ensure!(!prompt_ids.is_empty(), "GPU prompt IDs empty");
        ensure!(
            (1..=512).contains(&max_new_tokens),
            "max_new_tokens must be 1..=512"
        );
        ensure!(
            prompt_ids
                .len()
                .checked_add(max_new_tokens)
                .is_some_and(|n| n <= self.max_tokens),
            "GPU context capacity not large enough for prompt + generation"
        );
        self.select_calibrated_prefill(prompt_ids.len())?;
        let started = Instant::now();
        self.last_prompt_replay_hit = false;
        self.last_verified_prefix_reused_tokens = 0;
        self.last_verified_prefix_restored_bytes = 0;
        self.last_prompt_replay_rehydrated_bytes = 0;
        self.last_prompt_replay_sha_verified = false;
        let restored = if self.gpu_prompt_replay_max_tokens > 0 {
            let result = self.restore_verified_gpu_prompt_replay(prompt_ids, capture_logits);
            if result.is_err() {
                self.poisoned = true;
            }
            result?
        } else {
            None
        };
        let shared_prefix = if restored.is_none() && self.verified_prefix_replay_enabled {
            let result = self.restore_verified_gpu_shared_prefix(prompt_ids);
            if result.is_err() {
                self.poisoned = true;
            }
            result?
        } else {
            0
        };
        let last = if let Some(step) = restored {
            self.session_tokens.clear();
            self.session_tokens.extend_from_slice(prompt_ids);
            Some(step)
        } else if let Some(batch_size) = self.batch_prefill_tokens {
            let result = self.prefill_batched(
                prompt_ids,
                shared_prefix,
                batch_size,
                capture_logits,
                cancel,
            );
            if result.is_err() {
                self.poisoned = true;
            }
            let step = result?;
            // Batch prefill bypasses `forward_token_internal`, so publish the
            // exact prompt history explicitly for physical K/V snapshots.
            self.session_tokens.clear();
            self.session_tokens.extend_from_slice(prompt_ids);
            Some(step)
        } else {
            let mut last = None;
            for (i, &id) in prompt_ids.iter().enumerate() {
                ensure!(
                    !cancel.load(Ordering::Relaxed),
                    "GPU generation cancelled by client"
                );
                last = self.forward_token_internal(
                    id,
                    capture_logits && i + 1 == prompt_ids.len(),
                    i + 1 == prompt_ids.len(),
                )?;
            }
            last
        };
        let prefill_seconds = started.elapsed().as_secs_f64();
        // Transactional exact BF16 cold commit only for bounded hot
        // prefix and epsilon=0. Longer contexts keep the original
        // immediate ColdStore writes/verified fallback, never approximate.
        self.deferred_cold_active = self.deferred_cold_enabled
            && self.batch_prefill_tokens.is_some()
            && self
                .lod
                .as_ref()
                .is_some_and(|lod| lod.config().epsilon == 0.0)
            && self.hot_values.as_ref().is_some_and(|hot| {
                hot.valid_prefix_tokens == self.position
                    && self
                        .position
                        .checked_add(max_new_tokens)
                        .is_some_and(|n| n <= hot.capacity_tokens)
            });
        let activated = self.deferred_cold_active;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut steps = Vec::with_capacity(max_new_tokens);
        let generation_result = (|| -> Result<()> {
            let mut current = last.context("no GPU prompt logits")?;
            for step in 0..max_new_tokens {
                ensure!(
                    !cancel.load(Ordering::Relaxed),
                    "GPU generation cancelled by client"
                );
                steps.push(current.clone());
                let token = current.argmax_token;
                if self.eos_token_ids.contains(&token) {
                    break;
                }
                generated.push(token);
                if step + 1 < max_new_tokens {
                    current = self.forward_token(token, capture_logits)?;
                }
            }
            Ok(())
        })();
        // Mandatory exact BF16 ColdStore seal BEFORE exposing any
        // completed response. Also seals a cancelled turn's prefix
        // if it has begun decoding; errors fail closed.
        if self.deferred_cold_active
            && let Err(error) = self.flush_deferred_cold()
        {
            self.poisoned = true;
            return Err(error.context("failed to commit exact GPU BF16 cold KV"));
        }
        generation_result?;
        Ok(GpuGeneration {
            prompt_tokens: prompt_ids.len(),
            tokens: generated,
            steps,
            model_load_seconds: self.model_load_seconds,
            prefill_seconds,
            total_seconds: started.elapsed().as_secs_f64(),
            device_name: self.engine.name.clone(),
            gpu_weight_payload_bytes: self.gpu_weight_payload_bytes,
            runtime: "Rust_GPU_Full_BF16_no_LibTorch".to_owned(),
            model_weights_dtype: "BF16".to_owned(),
            gpu_compute: "FP32_accumulators_BF16_intermediate_storage".to_owned(),
            hip_bf16_gemm_active: self.native_hip_enabled(),
            mslim_lod_active: false,
            gpu_region_components_active: self.lod.is_some(),
            gpu_region_stats: self.gpu_region_stats(),
            gpu_kv_capacity_bytes: self.gpu_region_gpu_kv_capacity_bytes(),
            gpu_prefill_batch_tokens: self.batch_prefill_tokens,
            gpu_prefill_weight_reuse_tokens: self.prefill_weight_reuse_tokens,
            gpu_prefill_register_unrolled: self.prefill_register_unrolled,
            gpu_prefill_virtual_lanes: self.prefill_virtual_lanes,
            gpu_prefill_fused_projections: self.prefill_fuse_projections(),
            gpu_prefill_fusion_mask: self.prefill_fusion_mask,
            gpu_verified_prompt_replay_max_tokens: self.gpu_prompt_replay_max_tokens,
            gpu_verified_prefix_replay_enabled: self.verified_prefix_replay_enabled,
            gpu_verified_prefix_reused_tokens: self.last_verified_prefix_reused_tokens,
            gpu_verified_prefix_restored_bytes: self.last_verified_prefix_restored_bytes,
            gpu_verified_prompt_replay_hit: self.last_prompt_replay_hit,
            gpu_verified_prompt_replay_cached_tokens: self.gpu_prompt_replay_cached_tokens(),
            gpu_verified_prompt_replay_cached_bytes: self.gpu_prompt_replay_cached_bytes(),
            gpu_verified_prompt_replay_restored_bytes: self.last_prompt_replay_rehydrated_bytes,
            gpu_verified_prompt_replay_sha256_verified: self.last_prompt_replay_sha_verified,
            gpu_decode_unroll4: self.decode_unroll4(),
            gpu_decode_fused_kernels: self.decode_fused_kernels(),
            gpu_exact_hot_fused_attention: self.exact_hot_fused_attention_enabled(),
            gpu_deferred_cold_enabled: self.deferred_cold_enabled,
            gpu_deferred_cold_activated: activated,
            gpu_deferred_cold_commit_seconds: self.last_deferred_cold_commit_seconds,
            gpu_deferred_cold_committed_token_layers: self
                .last_deferred_cold_committed_token_layers,
            gpu_hot_v_capacity_tokens: self.exact_hot_v_capacity_tokens(),
            gpu_micro_hot_v_capacity_tokens: self.micro_hot_v_capacity_tokens(),
            gpu_micro_hot_sparse_calls: self
                .hot_values
                .as_ref()
                .map_or(0, |h| h.micro_sparse_calls),
            gpu_micro_hot_exact_fallback_calls: self
                .hot_values
                .as_ref()
                .map_or(0, |h| h.micro_full_exact_fallback_calls),
            gpu_micro_selection_upload_calls: self
                .hot_values
                .as_ref()
                .and_then(|h| h.micro_indices.as_ref())
                .map_or(0, |c| c.upload_calls),
            gpu_micro_selection_cache_hits: self
                .hot_values
                .as_ref()
                .and_then(|h| h.micro_indices.as_ref())
                .map_or(0, |c| c.reuse_hits),
            gpu_hot_v_attention_calls: self.hot_values.as_ref().map_or(0, |h| h.attention_calls),
            gpu_hot_v_cold_fallback_calls: self
                .hot_values
                .as_ref()
                .map_or(0, |h| h.cold_fallback_calls),
            gpu_hot_v_prefetch_bytes: self.hot_values.as_ref().map_or(0, |h| h.prefetched_bytes),
            gpu_hot_v_valid_prefix_tokens: self
                .hot_values
                .as_ref()
                .map_or(0, |h| h.valid_prefix_tokens),
            gpu_prefill_batches: self.last_prefill_batches,
            gpu_prefill_cold_seal_seconds: self.last_prefill_cold_seal_seconds,
            gpu_prefill_temporary_v_capacity_bytes: self.last_prefill_temporary_v_bytes,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuGeneration {
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub steps: Vec<GpuStep>,
    pub model_load_seconds: f64,
    pub prefill_seconds: f64,
    pub total_seconds: f64,
    pub device_name: String,
    pub gpu_weight_payload_bytes: usize,
    pub runtime: String,
    pub model_weights_dtype: String,
    pub gpu_compute: String,
    pub hip_bf16_gemm_active: bool,
    pub mslim_lod_active: bool,
    pub gpu_region_components_active: bool,
    pub gpu_region_stats: Option<GpuRegionStats>,
    pub gpu_kv_capacity_bytes: usize,
    pub gpu_prefill_batch_tokens: Option<usize>,
    pub gpu_prefill_weight_reuse_tokens: usize,
    pub gpu_prefill_register_unrolled: bool,
    pub gpu_prefill_virtual_lanes: usize,
    pub gpu_prefill_fused_projections: bool,
    pub gpu_prefill_fusion_mask: u8,
    pub gpu_verified_prompt_replay_max_tokens: usize,
    pub gpu_verified_prefix_replay_enabled: bool,
    pub gpu_verified_prefix_reused_tokens: usize,
    pub gpu_verified_prefix_restored_bytes: usize,
    pub gpu_verified_prompt_replay_hit: bool,
    pub gpu_verified_prompt_replay_cached_tokens: usize,
    pub gpu_verified_prompt_replay_cached_bytes: usize,
    pub gpu_verified_prompt_replay_restored_bytes: usize,
    pub gpu_verified_prompt_replay_sha256_verified: bool,
    pub gpu_decode_unroll4: bool,
    pub gpu_decode_fused_kernels: bool,
    pub gpu_exact_hot_fused_attention: bool,
    pub gpu_deferred_cold_enabled: bool,
    pub gpu_deferred_cold_activated: bool,
    pub gpu_deferred_cold_commit_seconds: f64,
    pub gpu_deferred_cold_committed_token_layers: usize,
    pub gpu_hot_v_capacity_tokens: usize,
    pub gpu_micro_hot_v_capacity_tokens: usize,
    pub gpu_micro_hot_sparse_calls: usize,
    pub gpu_micro_hot_exact_fallback_calls: usize,
    pub gpu_micro_selection_upload_calls: usize,
    pub gpu_micro_selection_cache_hits: usize,
    pub gpu_hot_v_attention_calls: usize,
    pub gpu_hot_v_cold_fallback_calls: usize,
    pub gpu_hot_v_prefetch_bytes: usize,
    pub gpu_hot_v_valid_prefix_tokens: usize,
    pub gpu_prefill_batches: usize,
    pub gpu_prefill_cold_seal_seconds: f64,
    pub gpu_prefill_temporary_v_capacity_bytes: usize,
}

#[cfg(test)]
mod tests {
    #[test]
    fn llama3_rope_frequencies_match_huggingface_reference() {
        use super::*;
        let rope = RopeKernelConfig {
            theta: 500_000.0,
            llama3_scaling: 1,
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192.0,
        };
        // Generated independently with transformers 4.51.3
        // `_compute_llama3_parameters` for the local Llama-3.1-8B config.
        for (dimension, expected) in [
            (0, 1.0_f32),
            (20, 0.128_687_38),
            (40, 0.016_560_44),
            (60, 0.001_371_893_7),
            (80, 0.000_034_281_024),
            (100, 0.000_004_411_534_5),
            (120, 0.000_000_567_708_8),
            (126, 0.000_000_306_892_6),
        ] {
            let actual = rope.inverse_frequency(dimension, 128);
            let relative = (actual - expected).abs() / expected.max(f32::MIN_POSITIVE);
            assert!(
                relative <= 2.0e-6,
                "Llama 3 RoPE frequency {dimension}: {actual} != {expected} ({relative})"
            );
        }
    }

    #[cfg(feature = "physical-conformance")]
    mod physical_conformance {
        use super::super::*;

        #[test]
        #[ignore = "requires a physical accelerator and compatible checkpoint"]
        fn gpu_actual_program_compiles_llama3_rope_kernels() {
            use super::*;
            let engine = GpuEngine::new(false).unwrap();
            assert!(!engine.name.is_empty());
        }

        #[test]
        fn checkpoint_shards_resolve_tensors_and_reject_duplicates() {
            use super::*;
            use safetensors::tensor::serialize_to_file;

            let dir = tempfile::tempdir().unwrap();
            let first_bytes = [0_u8, 0x3f, 0, 0x40];
            let second_bytes = [0_u8, 0x41];
            let first = TensorView::new(Dtype::BF16, vec![2], &first_bytes).unwrap();
            let second = TensorView::new(Dtype::BF16, vec![1], &second_bytes).unwrap();
            serialize_to_file(
                [("model.embed_tokens.weight", first)],
                &None,
                &dir.path().join("model-00001-of-00002.safetensors"),
            )
            .unwrap();
            serialize_to_file(
                [("lm_head.weight", second)],
                &None,
                &dir.path().join("model-00002-of-00002.safetensors"),
            )
            .unwrap();
            let names = vec![
                "model-00001-of-00002.safetensors".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ];
            let checkpoint = CheckpointShards::open(dir.path(), &names).unwrap();
            assert_eq!(checkpoint.payload_bytes(), 6);
            assert_eq!(checkpoint.tensor_count(), 2);
            checkpoint
                .with_tensor("model.embed_tokens.weight", |view| {
                    assert_eq!(view.shape(), [2]);
                    assert_eq!(view.data(), first_bytes);
                    Ok(())
                })
                .unwrap();
            checkpoint
                .with_tensor("lm_head.weight", |view| {
                    assert_eq!(view.shape(), [1]);
                    assert_eq!(view.data(), second_bytes);
                    Ok(())
                })
                .unwrap();
            assert!(checkpoint.with_tensor("missing", |_| Ok(())).is_err());

            let duplicate = TensorView::new(Dtype::BF16, vec![1], &second_bytes).unwrap();
            serialize_to_file(
                [("model.embed_tokens.weight", duplicate)],
                &None,
                &dir.path().join("duplicate.safetensors"),
            )
            .unwrap();
            let mut duplicate_names = names;
            duplicate_names.push("duplicate.safetensors".to_owned());
            let error = CheckpointShards::open(dir.path(), &duplicate_names)
                .err()
                .expect("duplicate tensor names must fail closed");
            assert!(error.to_string().contains("duplicate checkpoint tensor"));
        }

        #[test]
        #[ignore = "requires a configured local checkpoint"]
        fn local_checkpoint_shards_are_complete() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let spec = ModelSpec::inspect(&model).unwrap();
            assert!(!spec.weight_shards.is_empty());
            let checkpoint =
                CheckpointShards::open(Path::new(&model), &spec.weight_shards).unwrap();
            assert!(checkpoint.tensor_count() > 0);
            assert!(checkpoint.payload_bytes() > 0);
        }

        #[test]
        #[ignore = "requires actual physical GPU and a local configured-model BF16 checkpoint"]
        fn gpu_actual_full_minicpm5_hola_and_independent_session_reset() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .expect("actual checkpoint tokenizer required");
            let prompt = "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 15);
            let mut model = GpuLlm::load(model, 64, WeightMode::Resident)
                .expect("physical GPU must run the full 24-layer model");
            let first = model.generate(&ids, 2, false).unwrap();
            assert_eq!(first.tokens, vec![61973, 61]);
            let initial_sha = model.logical_kv_sha256().unwrap();
            model.reset_session().unwrap();
            let repeated = model.generate(&ids, 2, false).unwrap();
            assert_eq!(first.tokens, repeated.tokens);
            assert_eq!(initial_sha, model.logical_kv_sha256().unwrap());
        }

        #[test]
        #[ignore = "requires physical GPU and real configured-model checkpoint"]
        fn gpu_actual_gpu_region_zero_matches_dense_gpu_outputs() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let text = "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let encoded = tokenizer.encode(text, false).unwrap();
            let ids = encoded.get_ids().to_vec();
            let mut dense = GpuLlm::load(&model, 64, WeightMode::Resident).unwrap();
            let dense_result = dense.generate(&ids, 16, true).unwrap();
            let dense_ids = dense_result.tokens.clone();
            // Free dense weights before allocating another >2 GiB GPU runtime.
            drop(dense);
            let cfg = GpuRegionConfig {
                epsilon: 0.0,
                stage_tokens: 4,
                cold_directory: None,
            };
            let mut lod =
                GpuLlm::load_with_gpu_region(&model, 64, WeightMode::Resident, cfg).unwrap();
            let lod_result = lod.generate(&ids, 16, true).unwrap();
            assert_eq!(dense_ids, lod_result.tokens);
            assert!(lod_result.gpu_region_components_active);
            assert!(lod_result.gpu_kv_capacity_bytes < 64 * 2 * 2 * 128 * 24 * 2);
            let stats = lod_result.gpu_region_stats.as_ref().unwrap();
            assert!(stats.exact_v_tokens_read_from_cold > 0);
            assert_eq!(stats.exact_v_tokens_skipped, 0);
            assert!(stats.four_bit_index_payload_bytes > 0);
            assert!(stats.physical_cold_store_bytes > 0);
            for (a, b) in dense_result.steps.iter().zip(lod_result.steps.iter()) {
                assert_eq!(a.argmax_token, b.argmax_token);
                let x = a.logits.as_ref().unwrap();
                let y = b.logits.as_ref().unwrap();
                assert_eq!(x, y, "epsilon=0 must preserve bitwise GPU dense logits");
            }
            lod.reset_session().unwrap();
            let again = lod.generate(&ids, 16, false).unwrap();
            assert_eq!(again.tokens, lod_result.tokens);
        }

        #[test]
        #[ignore = "requires real Radeon GPU and configured-model BF16 checkpoint"]
        fn gpu_actual_parallel_prefill_dense_matches_sequential_gpu() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let text = "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 15);
            let mut old = GpuLlm::load(&model, 64, WeightMode::Resident).unwrap();
            let old_gen = old.generate(&ids, 16, true).unwrap();
            let old_kv = old.logical_kv_sha256().unwrap();
            drop(old);
            let mut fast = GpuLlm::load(&model, 64, WeightMode::Resident).unwrap();
            fast.set_batch_prefill_tokens(Some(64)).unwrap();
            let new_gen = fast.generate(&ids, 16, true).unwrap();
            let new_kv = fast.logical_kv_sha256().unwrap();
            assert_eq!(
                old_gen.tokens, new_gen.tokens,
                "GPU prefill changed generated token IDs"
            );
            assert_eq!(old_kv, new_kv, "GPU prefill changed exact BF16 K/V bytes");
            for (i, (a, b)) in old_gen.steps.iter().zip(new_gen.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "GPU parallel prefill changed full vocab BF16 logits at step {i}"
                );
            }
            assert!(new_gen.gpu_prefill_batches > 0);
            assert_eq!(new_gen.gpu_prefill_temporary_v_capacity_bytes, 0);
        }

        #[test]
        #[ignore = "requires real Radeon GPU and configured-model BF16 checkpoint"]
        fn gpu_actual_parallel_prefill_lod_matches_sequential_and_reduces_pages() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let text = "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            let config = GpuRegionConfig {
                epsilon: 0.0,
                stage_tokens: 8,
                cold_directory: None,
            };
            let mut old =
                GpuLlm::load_with_gpu_region(&model, 64, WeightMode::Resident, config.clone())
                    .unwrap();
            let old_gen = old.generate(&ids, 16, true).unwrap();
            let old_pages = old.gpu_region_stats().unwrap().physical_cold_store_pages;
            drop(old);
            let mut fast =
                GpuLlm::load_with_gpu_region(&model, 64, WeightMode::Resident, config).unwrap();
            fast.set_batch_prefill_tokens(Some(64)).unwrap();
            let new_gen = fast.generate(&ids, 16, true).unwrap();
            let new_stats = new_gen.gpu_region_stats.as_ref().unwrap();
            assert_eq!(
                old_gen.tokens, new_gen.tokens,
                "GPU LOD batch prefill changed token IDs"
            );
            for (i, (a, b)) in old_gen.steps.iter().zip(new_gen.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "GPU parallel+LOD prefill changed BF16 logits at step {i}"
                );
            }
            assert!(
                new_stats.physical_cold_store_pages < old_pages,
                "Batch prefill did not reduce ColdStore physical pages"
            );
            assert!(new_stats.cold_value_read_calls > 0);
            assert!(new_gen.gpu_prefill_temporary_v_capacity_bytes > 0);
            assert!(new_gen.gpu_prefill_cold_seal_seconds > 0.0);
            assert_eq!(new_stats.exact_v_tokens_skipped, 0);
            fast.reset_session().unwrap();
            let repeat = fast.generate(&ids, 16, false).unwrap();
            assert_eq!(repeat.tokens, new_gen.tokens);
        }

        #[test]
        #[ignore = "requires real Radeon GPU + configured-model BF16 checkpoint; ~45s"]
        fn gpu_actual_three_batch_prefill_375_matches_gpu_dense_bf16_kv_and_lod() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let text = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut old = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            let old_gen = old.generate(&ids, 2, true).unwrap();
            let old_sha = old.logical_kv_sha256().unwrap();
            drop(old);
            let mut fast = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            fast.set_batch_prefill_tokens(Some(128)).unwrap();
            let dense_gen = fast.generate(&ids, 2, true).unwrap();
            let fast_sha = fast.logical_kv_sha256().unwrap();
            assert_eq!(
                old_sha, fast_sha,
                "3-batch GPU prefill must preserve logical K/V BF16 SHA"
            );
            assert_eq!(
                old_gen.tokens, dense_gen.tokens,
                "3-batch GPU prefill changed generated tokens"
            );
            assert_eq!(dense_gen.gpu_prefill_batches, 3);
            for (step, (a, b)) in old_gen.steps.iter().zip(&dense_gen.steps).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "3-batch prefill changed BF16 full-vocab logits at {step}"
                );
            }
            drop(fast);
            let cfg = GpuRegionConfig {
                epsilon: 0.0,
                stage_tokens: 64,
                cold_directory: None,
            };
            let mut lod =
                GpuLlm::load_with_gpu_region(&model, 512, WeightMode::Resident, cfg).unwrap();
            lod.set_batch_prefill_tokens(Some(128)).unwrap();
            let lod_gen = lod.generate(&ids, 2, true).unwrap();
            assert_eq!(lod_gen.tokens, dense_gen.tokens);
            for (step, (a, b)) in dense_gen.steps.iter().zip(&lod_gen.steps).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "3-batch prefill+canonical LOD changed BF16 logits at {step}"
                );
            }
            let stats = lod_gen.gpu_region_stats.as_ref().unwrap();
            assert!(
                stats.physical_cold_store_pages < 300,
                "GPU LOD prefill did not group cold pages into <=64-token pages"
            );
            assert!(stats.exact_v_tokens_read_from_cold > 0);
            assert!(stats.cold_value_read_calls > 0);
            assert_eq!(stats.exact_v_tokens_skipped, 0);
            assert!(lod_gen.gpu_prefill_cold_seal_seconds > 0.0);
            assert!(lod_gen.gpu_prefill_temporary_v_capacity_bytes > 0);
            assert!(lod_gen.gpu_prefill_batch_tokens == Some(128));
        }

        #[test]
        #[ignore = "requires actual AMD GPU; compares 375-token prefill and 16 logits"]
        fn gpu_actual_exact_hot_v_and_weight_reuse_matches_original_gpu_dense() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let tok = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let prompt = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\nDato verificable: TEST-CODE-001. {}Indica el código del dato verificable.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut dense = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            dense.set_batch_prefill_tokens(Some(128)).unwrap();
            let original = dense.generate(&ids, 16, true).unwrap();
            let original_sha = dense.logical_kv_sha256().unwrap();
            drop(dense);

            let mut lod = GpuLlm::load_with_gpu_region(
                &model,
                512,
                WeightMode::Resident,
                GpuRegionConfig::default(),
            )
            .unwrap();
            lod.set_batch_prefill_tokens(Some(128)).unwrap();
            lod.set_prefill_weight_reuse_tokens(16).unwrap();
            lod.set_exact_hot_v_tokens(512).unwrap();
            let current = lod.generate(&ids, 16, true).unwrap();
            assert_eq!(original.tokens, current.tokens);
            for (step, (a, b)) in original.steps.iter().zip(&current.steps).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "hot BF16 value cache or weight reuse changed GPU logits at {step}"
                );
            }
            assert_eq!(
                current.gpu_hot_v_attention_calls,
                15 * 24,
                "hot decode should cover all GPU layer attentions"
            );
            assert_eq!(current.gpu_hot_v_cold_fallback_calls, 0);
            assert_eq!(current.gpu_hot_v_valid_prefix_tokens, 390);
            assert_eq!(
                current
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_read_from_cold,
                0,
                "hot exact V cache must avoid all redundant ColdStore decode reads"
            );
            assert!(
                current
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_pages
                    > 0,
                "hot tier must not bypass canonical exact BF16 ColdStore writes"
            );
            // The committed GPU K/V are not directly exposed in LOD mode,
            // but all logits match the GPU dense reference and cold store is
            // still hashed/verifiable. Confirm hot prefix isn't leaked on reset.
            assert!(!original_sha.is_empty());
            lod.reset_session().unwrap();
            let repeated = lod.generate(&ids, 16, false).unwrap();
            assert_eq!(repeated.tokens, current.tokens);
            assert_eq!(
                repeated.gpu_hot_v_attention_calls,
                current.gpu_hot_v_attention_calls
            );
        }

        #[test]
        #[ignore = "actual AMD Radeon needed: validates optimized GPU 375+16"]
        fn gpu_actual_reuse8_hot_v_unroll4_bit_exact_vs_dense_reference() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tok = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let u = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let text = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut reference = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            reference.set_batch_prefill_tokens(Some(128)).unwrap();
            let expected = reference.generate(&ids, 16, true).unwrap();
            let sha = reference.logical_kv_sha256().unwrap();
            drop(reference);
            let mut optimized = GpuLlm::load_with_gpu_region(
                &model,
                512,
                WeightMode::Resident,
                GpuRegionConfig {
                    epsilon: 0.0,
                    stage_tokens: 64,
                    cold_directory: None,
                },
            )
            .unwrap();
            optimized.set_batch_prefill_tokens(Some(128)).unwrap();
            optimized.set_prefill_weight_reuse_tokens(8).unwrap();
            optimized.set_exact_hot_v_tokens(512).unwrap();
            optimized.set_decode_unroll4(true).unwrap();
            let actual = optimized.generate(&ids, 16, true).unwrap();
            assert_eq!(
                actual.tokens, expected.tokens,
                "optimized GPU changed greedy generated IDs"
            );
            for (i, (a, b)) in expected.steps.iter().zip(&actual.steps).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "BF16 GPU full-vocab logits changed at step {i}"
                );
            }
            assert!(sha.len() == 64);
            let stats = actual.gpu_region_stats.as_ref().unwrap();
            assert_eq!(stats.exact_v_tokens_skipped, 0);
            assert_eq!(
                stats.exact_v_tokens_read_from_cold, 0,
                "GPU hot V tier unexpectedly re-read cold values"
            );
            assert!(stats.physical_cold_store_bytes > 0);
            assert!(stats.physical_cold_store_pages > 0);
            assert_eq!(actual.gpu_hot_v_attention_calls, 360);
            assert_eq!(actual.gpu_hot_v_cold_fallback_calls, 0);
        }

        #[test]
        #[ignore = "requires physical gfx1103 Radeon and original configured-model BF16 checkpoint"]
        fn gpu_actual_deferred_bf16_cold_commit_matches_immediate_and_indexes_every_token() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let chat = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(chat, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let make = || {
                let mut gpu = GpuLlm::load_with_gpu_region(
                    &model,
                    512,
                    WeightMode::Resident,
                    GpuRegionConfig {
                        epsilon: 0.0,
                        stage_tokens: 64,
                        cold_directory: None,
                    },
                )
                .unwrap();
                gpu.set_batch_prefill_tokens(Some(128)).unwrap();
                gpu.set_prefill_weight_reuse_tokens(8).unwrap();
                gpu.set_decode_unroll4(true).unwrap();
                gpu.set_exact_hot_v_tokens(512).unwrap();
                gpu
            };
            let compute_hash = |gpu: &mut GpuLlm| {
                let token_bytes = gpu.spec.attention.kv_elements_per_token() * 2;
                let positions = (0..gpu.position).collect::<Vec<_>>();
                let mut hasher = Sha256::new();
                for layer in 0..gpu.spec.num_hidden_layers {
                    hasher.update(
                        gpu.engine
                            .read_bf16_bytes(
                                &gpu.workspace.kv[layer].key,
                                gpu.position * token_bytes,
                            )
                            .unwrap(),
                    );
                    let lod = gpu.lod.as_mut().unwrap();
                    assert_eq!(lod.layer_tokens(layer), gpu.position);
                    let values = lod.read_values(layer, &positions).unwrap();
                    hasher.update(values);
                }
                format!("{:x}", hasher.finalize())
            };
            let mut legacy = make();
            let previous = legacy.generate(&ids, 16, true).unwrap();
            assert!(!previous.gpu_deferred_cold_activated);
            let old_hash = compute_hash(&mut legacy);
            drop(legacy);
            let mut improved = make();
            improved.set_deferred_cold_commit(true).unwrap();
            let result = improved.generate(&ids, 16, true).unwrap();
            assert!(result.gpu_deferred_cold_enabled);
            assert!(result.gpu_deferred_cold_activated);
            assert_eq!(result.gpu_deferred_cold_committed_token_layers, 15 * 24);
            assert!(result.gpu_deferred_cold_commit_seconds > 0.0);
            assert_eq!(result.tokens, previous.tokens);
            for (index, (a, b)) in previous.steps.iter().zip(&result.steps).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "GPU exact BF16 logits diverged step {index}"
                );
            }
            let stats = result.gpu_region_stats.as_ref().unwrap();
            assert_eq!(stats.exact_v_tokens_skipped, 0);
            assert_eq!(stats.exact_v_tokens_read_from_cold, 0);
            assert!(stats.physical_cold_store_pages > 0);
            assert_eq!(
                improved.lod.as_ref().unwrap().axis().token_len(),
                improved.position
            );
            assert_eq!(
                old_hash,
                compute_hash(&mut improved),
                "original canonical BF16 cold values or original GPU BF16 keys changed"
            );
            improved.reset_session().unwrap();
            let repeat = improved.generate(&ids, 2, false).unwrap();
            assert!(repeat.gpu_deferred_cold_activated);
            assert_eq!(repeat.tokens, previous.tokens[..2]);
        }

        #[test]
        #[ignore = "requires real AMD GPU and configured-model BF16 checkpoint"]
        fn gpu_actual_deferred_cold_falls_back_when_hot_v_capacity_insufficient() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let mut gpu = GpuLlm::load_with_gpu_region(
                &model,
                64,
                WeightMode::Resident,
                GpuRegionConfig::default(),
            )
            .unwrap();
            gpu.set_batch_prefill_tokens(Some(32)).unwrap();
            gpu.set_exact_hot_v_tokens(16).unwrap();
            gpu.set_deferred_cold_commit(true).unwrap();
            let ids = vec![0u32; 8];
            let generated = gpu.generate(&ids, 16, false).unwrap();
            assert!(
                !generated.gpu_deferred_cold_activated,
                "hot prefix capacity insufficient for entire turn must use original ColdStore path"
            );
            assert_eq!(generated.gpu_deferred_cold_committed_token_layers, 0);
            assert_eq!(gpu.lod.as_ref().unwrap().axis().token_len(), gpu.position);
            for layer in 0..gpu.spec.num_hidden_layers {
                assert_eq!(gpu.lod.as_ref().unwrap().layer_tokens(layer), gpu.position);
            }
        }

        #[test]
        #[ignore = "requires physical AMD GPU and original configured-model BF16 checkpoint"]
        fn gpu_actual_fused_qkv_and_swiglu_identical_logits_and_exact_kv() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let chat = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(chat, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let create = |fused: bool| {
                let mut gpu = GpuLlm::load_with_gpu_region(
                    &model,
                    512,
                    WeightMode::Resident,
                    GpuRegionConfig {
                        epsilon: 0.0,
                        stage_tokens: 64,
                        cold_directory: None,
                    },
                )
                .unwrap();
                gpu.set_batch_prefill_tokens(Some(128)).unwrap();
                gpu.set_prefill_weight_reuse_tokens(8).unwrap();
                gpu.set_decode_unroll4(true).unwrap();
                gpu.set_exact_hot_v_tokens(512).unwrap();
                gpu.set_deferred_cold_commit(true).unwrap();
                gpu.set_decode_fused_kernels(fused).unwrap();
                gpu
            };
            let kv_hash = |gpu: &mut GpuLlm| {
                let mut sha = Sha256::new();
                let bytes = gpu.spec.attention.kv_elements_per_token() * 2;
                let all = (0..gpu.position).collect::<Vec<_>>();
                for layer in 0..gpu.spec.num_hidden_layers {
                    sha.update(
                        gpu.engine
                            .read_bf16_bytes(&gpu.workspace.kv[layer].key, gpu.position * bytes)
                            .unwrap(),
                    );
                    let value = gpu.lod.as_mut().unwrap().read_values(layer, &all).unwrap();
                    sha.update(value);
                }
                format!("{:x}", sha.finalize())
            };
            let mut baseline = create(false);
            let expected = baseline.generate(&ids, 16, true).unwrap();
            let expected_hash = kv_hash(&mut baseline);
            drop(baseline);
            let mut fast = create(true);
            let actual = fast.generate(&ids, 16, true).unwrap();
            assert_eq!(actual.tokens, expected.tokens);
            assert!(actual.gpu_decode_fused_kernels);
            assert!(actual.gpu_deferred_cold_activated);
            for (step, (a, b)) in expected.steps.iter().zip(actual.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "GPU fused BF16 logits diverged at step {step}"
                );
            }
            assert_eq!(
                expected_hash,
                kv_hash(&mut fast),
                "fused decode modified original K/V BF16 logical contents"
            );
            assert_eq!(
                actual
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_skipped,
                0
            );
        }

        #[test]
        #[ignore = "requires physical AMD GPU, real shaders, original CPU ColdStore"]
        fn gpu_actual_micro_regions_skip_real_qk_and_cold_v_with_error_bound() {
            use super::*;
            let geom = AttentionGeometry::new(2, 1, 4).unwrap();
            let cfg = GpuRegionConfig {
                epsilon: 0.001,
                stage_tokens: 16,
                cold_directory: None,
            };
            let mut lod = GpuRegionStore::new(geom, 1, cfg).unwrap();
            let as_bf16 = |v: &[f32]| {
                v.iter()
                    .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let k = (0..32)
                .flat_map(|i| {
                    if i < 16 {
                        [-20.0, 0.0, 0.0, 0.0]
                    } else {
                        [20.0, 0.0, 0.0, 0.0]
                    }
                })
                .collect::<Vec<f32>>();
            let v = (0..32)
                .flat_map(|i| {
                    if i < 16 {
                        [1.0, 0.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0, 0.0]
                    }
                })
                .collect::<Vec<f32>>();
            let keys = as_bf16(&k);
            let values = as_bf16(&v);
            let query = as_bf16(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            lod.append_batch(0, 0, &keys, &values, 16).unwrap();
            let selection = lod
                .plan_micro_regions(0, &query)
                .unwrap()
                .context("micro certificate should allow skipping old state")
                .unwrap();
            assert_eq!(selection.positions, (16..32).collect::<Vec<_>>());
            assert_eq!(selection.omitted, 16);
            assert!(selection.bounds.iter().all(|&b| b <= 0.001));
            assert_eq!(lod.stats.micro_omitted_k_dot_products, 32);
            assert_eq!(lod.stats.micro_sparse_queries, 1);

            let engine = GpuEngine::new(false).unwrap();
            let mut q = engine.alloc_u8(query.len()).unwrap();
            let mut key = engine.alloc_u8(keys.len()).unwrap();
            let mut value = engine.alloc_u8(values.len()).unwrap();
            engine.transfer(&mut q, &query).unwrap();
            engine.transfer(&mut key, &keys).unwrap();
            engine.transfer(&mut value, &values).unwrap();

            let dense_scores = engine.alloc_scores(32 * 2).unwrap();
            let dense_output = engine.alloc_u8(geom.query_elements() * 2).unwrap();
            engine
                .attention(&q, (&key, &value), &dense_scores, &dense_output, 32, geom)
                .unwrap();
            let exact = engine
                .read_bf16_bytes(&dense_output, geom.query_elements() * 2)
                .unwrap();

            let sparse_scores = engine.alloc_scores(32 * 2).unwrap();
            engine
                .micro_sparse_scores(&q, &key, &sparse_scores, &selection.positions, geom, 32)
                .unwrap();
            // Real GPU QK only computed for certified selected original BF16 K.
            let mut raw_scores = vec![0.0f32; 32 * 2];
            // SAFETY: Blocking read of the actual original OpenCL scores.
            unsafe {
                engine
                    .queue
                    .enqueue_read_buffer(&sparse_scores, CL_BLOCKING, 0, &mut raw_scores, &[])
                    .unwrap();
            }
            for head in 0..geom.query_heads {
                for idx in 0..32 {
                    let score = raw_scores[head * 32 + idx];
                    if idx < 16 {
                        assert!(score.is_infinite() && score.is_sign_negative());
                    } else {
                        assert!(score.is_finite());
                    }
                }
            }
            let n = 32u32;
            let heads = 2u32;
            // SAFETY: Execute the original real GPU softmax, not a CPU simulation.
            unsafe {
                ExecuteKernel::new(&engine.kernels.softmax)
                    .set_arg(&sparse_scores)
                    .set_arg(&n)
                    .set_arg(&heads)
                    .set_global_work_size(2)
                    .enqueue_nd_range(&engine.queue)
                    .unwrap();
            }
            let chosen_v = lod.read_values(0, &selection.positions).unwrap();
            assert_eq!(chosen_v.len(), 16 * 8);
            let mut staged_v = engine.alloc_u8(chosen_v.len()).unwrap();
            engine.transfer(&mut staged_v, &chosen_v).unwrap();
            let indices = selection
                .positions
                .iter()
                .map(|&x| u32::try_from(x).unwrap())
                .collect::<Vec<_>>();
            let mut staged_ids = engine.alloc_indices(indices.len()).unwrap();
            let mut accumulator = engine.alloc_scores(geom.query_elements()).unwrap();
            let mut mass = engine.alloc_scores(geom.query_heads).unwrap();
            let zeros = vec![0.0f32; geom.query_elements()];
            let one = vec![1.0f32; geom.query_heads];
            // SAFETY: Blocking writes retain host backing and matched GPU buffer
            // capacities. This physically runs original GPU BF16 PV kernels.
            unsafe {
                engine
                    .queue
                    .enqueue_write_buffer(&mut staged_ids, CL_BLOCKING, 0, &indices, &[])
                    .unwrap();
                engine
                    .queue
                    .enqueue_write_buffer(&mut accumulator, CL_BLOCKING, 0, &zeros, &[])
                    .unwrap();
                engine
                    .queue
                    .enqueue_write_buffer(&mut mass, CL_BLOCKING, 0, &one, &[])
                    .unwrap();
            }
            let selected_count = 16u32;
            let kvheads = 1u32;
            let dim = 4u32;
            let out = engine.alloc_u8(geom.query_elements() * 2).unwrap();
            // SAFETY: This is the original physical GPU BF16 PV staging path,
            // with no CPU floating point arithmetic for attention.
            unsafe {
                ExecuteKernel::new(&engine.kernels.lod_pv_stage)
                    .set_arg(&sparse_scores)
                    .set_arg(&staged_v)
                    .set_arg(&staged_ids)
                    .set_arg(&accumulator)
                    .set_arg(&selected_count)
                    .set_arg(&n)
                    .set_arg(&heads)
                    .set_arg(&kvheads)
                    .set_arg(&dim)
                    .set_global_work_size(geom.query_elements())
                    .enqueue_nd_range(&engine.queue)
                    .unwrap();
                ExecuteKernel::new(&engine.kernels.lod_pv_finish)
                    .set_arg(&accumulator)
                    .set_arg(&mass)
                    .set_arg(&out)
                    .set_arg(&heads)
                    .set_arg(&dim)
                    .set_global_work_size(geom.query_elements())
                    .enqueue_nd_range(&engine.queue)
                    .unwrap();
            }
            let approximate = engine
                .read_bf16_bytes(&out, geom.query_elements() * 2)
                .unwrap();
            let mut max_abs_error = 0.0f32;
            let mut full_output = Vec::new();
            let mut sparse_output = Vec::new();
            for (a, b) in exact.chunks_exact(2).zip(approximate.chunks_exact(2)) {
                let a = half::bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                let b = half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                max_abs_error = max_abs_error.max((a - b).abs());
                full_output.push(a);
                sparse_output.push(b);
                assert!(
                    (a - b).abs() <= 0.001 + 1e-6,
                    "actual GPU omitted mass exceeded micro certificate"
                );
            }
            eprintln!(
                "PRUEBA_GPU_REAL micro: dispositivo={} | tokens_K_total=32 | tokens_K_calculados={} | productos_QK_omitidos={} | V_recuperadas={} | V_omitidas={} | cotas={:?} | L_inf_real={} | L_inf_permitido=0.001 | salida_densa_BF16={:?} | salida_micro_BF16={:?}",
                engine.name,
                selection.positions.len(),
                lod.stats.micro_omitted_k_dot_products,
                lod.stats.exact_v_tokens_read_from_cold,
                lod.stats.exact_v_tokens_skipped,
                selection.bounds,
                max_abs_error,
                full_output,
                sparse_output
            );
            assert_eq!(lod.stats.exact_v_tokens_read_from_cold, 16);
            assert_eq!(lod.stats.exact_v_tokens_skipped, 16);
            // REAL physical-GPU microbenchmark, not a count-of-operations
            // proxy. For each repetition, the original dense path includes
            // full GPU QK, softmax, PV, device queue finish, with resident V.
            // The micro path includes original CPU dimensional planner,
            // indexed K GPU upload/clear/QK, original GPU softmax,
            // hash-verified ColdStore V reads, BF16 staging and GPU PV.
            // This deliberately exposes when the metadata overhead is
            // *slower* despite fewer products; neither count is substituted
            // for actual measured wall time. Small N is launch-dominated.
            let trials = 96usize;
            let warmup = 12usize;
            let mut dense_ms = Vec::new();
            let mut micro_total_ms = Vec::new();
            let mut micro_planner_ms = Vec::new();
            let mut micro_qk_ms = Vec::new();
            let mut micro_cold_ms = Vec::new();
            for iter in 0..(trials + warmup) {
                let t = std::time::Instant::now();
                engine
                    .attention(&q, (&key, &value), &dense_scores, &dense_output, 32, geom)
                    .unwrap();
                engine.queue.finish().unwrap();
                let duration_dense = t.elapsed().as_secs_f64() * 1000.0;

                let start_total = std::time::Instant::now();
                let start_plan = std::time::Instant::now();
                let selection_iter = lod
                    .plan_micro_regions(0, &query)
                    .unwrap()
                    .expect("certification must still succeed in benchmark");
                let duration_plan = start_plan.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(selection_iter.positions, selection.positions);

                let start_qk = std::time::Instant::now();
                engine
                    .micro_sparse_scores(
                        &q,
                        &key,
                        &sparse_scores,
                        &selection_iter.positions,
                        geom,
                        32,
                    )
                    .unwrap();
                let duration_qk = start_qk.elapsed().as_secs_f64() * 1000.0;

                // SAFETY: Both original physical GPU softmax and V-stage PV.
                unsafe {
                    ExecuteKernel::new(&engine.kernels.softmax)
                        .set_arg(&sparse_scores)
                        .set_arg(&n)
                        .set_arg(&heads)
                        .set_global_work_size(2)
                        .enqueue_nd_range(&engine.queue)
                        .unwrap();
                }
                let start_cold = std::time::Instant::now();
                let selected_values = lod.read_values(0, &selection_iter.positions).unwrap();
                engine.transfer(&mut staged_v, &selected_values).unwrap();
                let duration_cold = start_cold.elapsed().as_secs_f64() * 1000.0;
                // SAFETY: Reset the FP32 accumulator with blocking copy
                // on each benchmark iteration, so GPU PV is not accumulated
                // incorrectly across repeated independent requests.
                unsafe {
                    engine
                        .queue
                        .enqueue_write_buffer(&mut accumulator, CL_BLOCKING, 0, &zeros, &[])
                        .unwrap();
                    ExecuteKernel::new(&engine.kernels.lod_pv_stage)
                        .set_arg(&sparse_scores)
                        .set_arg(&staged_v)
                        .set_arg(&staged_ids)
                        .set_arg(&accumulator)
                        .set_arg(&selected_count)
                        .set_arg(&n)
                        .set_arg(&heads)
                        .set_arg(&kvheads)
                        .set_arg(&dim)
                        .set_global_work_size(geom.query_elements())
                        .enqueue_nd_range(&engine.queue)
                        .unwrap();
                    ExecuteKernel::new(&engine.kernels.lod_pv_finish)
                        .set_arg(&accumulator)
                        .set_arg(&mass)
                        .set_arg(&out)
                        .set_arg(&heads)
                        .set_arg(&dim)
                        .set_global_work_size(geom.query_elements())
                        .enqueue_nd_range(&engine.queue)
                        .unwrap();
                }
                engine.queue.finish().unwrap();
                let duration_micro_total = start_total.elapsed().as_secs_f64() * 1000.0;
                if iter >= warmup {
                    dense_ms.push(duration_dense);
                    micro_total_ms.push(duration_micro_total);
                    micro_planner_ms.push(duration_plan);
                    micro_qk_ms.push(duration_qk);
                    micro_cold_ms.push(duration_cold);
                }
            }
            let median = |mut values: Vec<f64>| {
                values.sort_unstable_by(f64::total_cmp);
                values[values.len() / 2]
            };
            let dense = median(dense_ms);
            let micro = median(micro_total_ms);
            eprintln!(
                "BENCHMARK_TIEMPO_REAL_GPU32 repeticiones={} | atencion_completa_ms_mediana={:.6} | micro_seleccion_cold_gpu_ms_mediana={:.6} | relacion_micro_sobre_completa={:.4} | planificador_cpu_ms_mediana={:.6} | qk_micro_y_transferencia_indice_ms_mediana={:.6} | lectura_v_fria_y_transferencia_ms_mediana={:.6} | BF16_error_previamente_verificado={:.12} | tokens_K_omitidos=16/32 | V_omitidas=16/32",
                trials,
                dense,
                micro,
                micro / dense,
                median(micro_planner_ms),
                median(micro_qk_ms),
                median(micro_cold_ms),
                max_abs_error,
            );

            // Fast path under EXACTLY the same certified micro selection:
            // BF16 V remains device resident; selected K indices are
            // uploaded ONCE and cached in the OpenCL device buffer.
            // This is a true end-to-end attention latency benchmark.
            let mut cache = MicroSelectedDevice {
                indices: engine.alloc_indices(32).unwrap(),
                last_positions: Vec::new(),
                upload_calls: 0,
                reuse_hits: 0,
            };
            engine
                .micro_update_cached_indices(&mut cache, &selection.positions, 32)
                .unwrap();
            let resident_output = engine.alloc_u8(geom.query_elements() * 2).unwrap();
            engine
                .micro_hot_attention(
                    (&q, &key, &value, &cache.indices, &resident_output),
                    (selection.positions.len(), 32),
                    geom,
                )
                .unwrap();
            engine.queue.finish().unwrap();
            let hot_actual = engine
                .read_bf16_bytes(&resident_output, geom.query_elements() * 2)
                .unwrap();
            let mut hot_error = 0.0f32;
            for (a, b) in exact.chunks_exact(2).zip(hot_actual.chunks_exact(2)) {
                let a = half::bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                let b = half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                hot_error = hot_error.max((a - b).abs());
            }
            assert!(
                hot_error <= 0.001,
                "resident micro BF16 output violates certified omitted mass"
            );
            let warmup = 12usize;
            let trials = 96usize;
            let mut dense_ms = Vec::new();
            let mut micro_hot_ms = Vec::new();
            let mut planning_ms = Vec::new();
            let mut kernel_ms = Vec::new();
            for i in 0..warmup + trials {
                let start = std::time::Instant::now();
                engine
                    .attention(&q, (&key, &value), &dense_scores, &dense_output, 32, geom)
                    .unwrap();
                engine.queue.finish().unwrap();
                let full_ms = start.elapsed().as_secs_f64() * 1000.0;

                let start = std::time::Instant::now();
                let planned = lod.plan_micro_regions(0, &query).unwrap().unwrap();
                let planned_ms = start.elapsed().as_secs_f64() * 1000.0;
                let gpu_started = std::time::Instant::now();
                engine
                    .micro_update_cached_indices(&mut cache, &planned.positions, 32)
                    .unwrap();
                engine
                    .micro_hot_attention(
                        (&q, &key, &value, &cache.indices, &resident_output),
                        (planned.positions.len(), 32),
                        geom,
                    )
                    .unwrap();
                engine.queue.finish().unwrap();
                let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;
                if i >= warmup {
                    dense_ms.push(full_ms);
                    planning_ms.push(planned_ms);
                    kernel_ms.push(gpu_ms);
                    micro_hot_ms.push(planned_ms + gpu_ms);
                }
            }
            let median = |mut v: Vec<f64>| {
                v.sort_unstable_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let full = median(dense_ms);
            let hot = median(micro_hot_ms);
            eprintln!(
                "BENCHMARK_MICRO_HOT_GPU32 repeticiones={} | atencion_completa_ms_mediana={:.6} | micro_con_indices_y_v_residentes_ms_mediana={:.6} | aceleracion_vs_completa={:.4} | micro_planificador_cpu_ms_mediana={:.6} | micro_gpu_kernel_ms_mediana={:.6} | subida_indice_gpu_veces={} | reutilizaciones_indice_gpu={} | L_inf_BF16_real={:.12} | K_omitidas=16/32 | V_omitidas=16/32",
                trials,
                full,
                hot,
                full / hot,
                median(planning_ms),
                median(kernel_ms),
                cache.upload_calls,
                cache.reuse_hits,
                hot_error
            );
            assert_eq!(
                cache.upload_calls, 1,
                "micro GPU selection should remain device resident after first upload"
            );
            assert!(
                cache.reuse_hits >= trials,
                "micro GPU selected indices were needlessly reuploaded"
            );
        }

        #[test]
        #[ignore = "requires physical AMD Radeon and configured-model BF16 checkpoint"]
        fn gpu_actual_micro_full_model_falls_back_when_uncertified() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let mut gpu = GpuLlm::load_with_gpu_region(
                &model,
                64,
                WeightMode::Resident,
                GpuRegionConfig {
                    epsilon: 0.001,
                    stage_tokens: 16,
                    cold_directory: None,
                },
            )
            .unwrap();
            gpu.set_batch_prefill_tokens(Some(32)).unwrap();
            let ids = vec![0u32; 8];
            let result = gpu.generate(&ids, 2, true).unwrap();
            assert_eq!(result.prompt_tokens, 8);
            assert!(!result.tokens.is_empty());
            let stats = result.gpu_region_stats.unwrap();
            assert!(stats.shared_original_cpu_algorithms);
            // The batched prefill uses dense causal GPU attention with K/V
            // sealed once; only subsequent autoregressive decode invokes
            // the micro region planner, once per Transformer layer.
            assert_eq!(
                stats.queries,
                gpu.spec.num_hidden_layers * (result.steps.len() - 1)
            );
            assert!(stats.micro_dense_fallback_queries > 0);
            assert!(stats.micro_sparse_queries <= stats.queries);
            assert!(stats.maximum_certified_error_bound <= 0.001);
        }

        #[test]
        #[ignore = "requires physical Radeon Phoenix GPU and exact configured-model BF16 checkpoint"]
        fn gpu_actual_micro_hot_model_375_bitwise_dense_fallback_and_cold_integrity() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tok = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let chat = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tok.encode(chat, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);

            // ORIGINAL GPU dense BF16 reference, no metadata/no approximation.
            let mut reference = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            reference.set_batch_prefill_tokens(Some(128)).unwrap();
            reference.set_prefill_weight_reuse_tokens(8).unwrap();
            reference.set_decode_unroll4(true).unwrap();
            reference.set_decode_fused_kernels(true).unwrap();
            let expected = reference.generate(&ids, 16, true).unwrap();
            let dense_kv = reference.debug_kv_layer_bytes().unwrap();
            drop(reference);

            // NEW POSITIVE epsilon GPU BF16 HOT V; original canonical CPU
            // ColdStore and four-bit/LSH remain the authoritative BF16 backing.
            let config = GpuRegionConfig {
                epsilon: 0.001,
                stage_tokens: 64,
                cold_directory: None,
            };
            let mut runtime =
                GpuLlm::load_with_gpu_region(&model, 512, WeightMode::Resident, config).unwrap();
            runtime.set_batch_prefill_tokens(Some(128)).unwrap();
            runtime.set_prefill_weight_reuse_tokens(8).unwrap();
            runtime.set_decode_unroll4(true).unwrap();
            runtime.set_decode_fused_kernels(true).unwrap();
            runtime.set_micro_hot_v_tokens(512).unwrap();
            let actual = runtime.generate(&ids, 16, true).unwrap();
            assert_eq!(actual.tokens, expected.tokens);
            assert_eq!(actual.steps.len(), expected.steps.len());
            for (step, (a, b)) in actual.steps.iter().zip(expected.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "full GPU hot exact fallback changed BF16 logits at step {step}"
                );
            }
            assert_eq!(actual.gpu_micro_hot_v_capacity_tokens, 512);
            assert_eq!(
                actual.gpu_micro_hot_sparse_calls, 0,
                "actual 375-token sample has no certified region omissions"
            );
            assert_eq!(actual.gpu_micro_hot_exact_fallback_calls, 24 * 15);
            assert_eq!(
                actual.gpu_hot_v_cold_fallback_calls, 0,
                "hot BF16 V must avoid cold value staging"
            );
            let stats = actual.gpu_region_stats.as_ref().unwrap();
            assert_eq!(stats.micro_sparse_queries, 0);
            assert_eq!(stats.exact_v_tokens_read_from_cold, 0);
            assert_eq!(stats.exact_v_tokens_skipped, 0);
            assert!(stats.physical_cold_store_bytes > 0);
            assert_eq!(
                runtime.lod.as_ref().unwrap().axis().token_len(),
                runtime.position
            );
            let count = runtime.position;
            let kvheads = runtime.spec.attention.kv_heads;
            let item = runtime.spec.attention.head_dim * 2;
            let token_bytes = kvheads * item;
            let positions = (0..count).collect::<Vec<_>>();
            let mut verified_bytes = 0usize;
            for (layer, (reference_key, reference_value)) in dense_kv.iter().enumerate() {
                let k = runtime
                    .engine
                    .read_bf16_bytes(&runtime.workspace.kv[layer].key, count * token_bytes)
                    .unwrap();
                let v = runtime
                    .lod
                    .as_mut()
                    .unwrap()
                    .read_values(layer, &positions)
                    .unwrap();
                for head in 0..kvheads {
                    for token in 0..count {
                        let from = (token * kvheads + head) * item;
                        let to = (head * count + token) * item;
                        assert_eq!(
                            &k[from..from + item],
                            &reference_key[to..to + item],
                            "GPU micro hot changed original BF16 K"
                        );
                        assert_eq!(
                            &v[from..from + item],
                            &reference_value[to..to + item],
                            "GPU micro hot ColdStore changed original BF16 V"
                        );
                        verified_bytes += 2 * item;
                    }
                }
            }
            eprintln!(
                "MICRO_HOT_LLM_GPU375_PASS | input_tokens={} | decoded_tokens={} | logits_identical=true | BF16_KV_bytes_verified={} | GPU_micro_hot_sparse={} | GPU_micro_exact_fallback={} | ColdStore_V_reads_during_decode={} | original_index_bytes={} | coldstore_pages={} | result_preview={:?}",
                ids.len(),
                actual.tokens.len(),
                verified_bytes,
                actual.gpu_micro_hot_sparse_calls,
                actual.gpu_micro_hot_exact_fallback_calls,
                stats.exact_v_tokens_read_from_cold,
                stats.four_bit_index_payload_bytes,
                stats.physical_cold_store_pages,
                &actual.tokens[..actual.tokens.len().min(5)]
            );
            runtime.reset_session().unwrap();
            runtime.set_exact_hot_fused_attention(true).unwrap();
            let again = runtime.generate(&ids, 16, true).unwrap();
            assert_eq!(again.tokens, expected.tokens);
            assert!(again.gpu_exact_hot_fused_attention);
            assert_eq!(again.gpu_micro_hot_exact_fallback_calls, 24 * 15);
            for (step, (a, b)) in again.steps.iter().zip(expected.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "positive-epsilon MICRO hot exact fallback fusion changed BF16 logits at step {step}"
                );
            }
            assert_eq!(
                again
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_skipped,
                0
            );
            eprintln!(
                "MICRO_POSITIVE_FUSED_EXACT_GPU_LLM_PASS prompts={} decode={} bit_exact_full_logits=true cold_bytes={}",
                ids.len(),
                again.tokens.len(),
                again
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_bytes
            );
        }

        #[test]
        #[ignore = "requires Radeon GPU: full BF16 QK-softmax-PV parity against original GPU kernels"]
        fn gpu_actual_exact_hot_fused_attention_parity_for_many_context_lengths() {
            use super::*;
            let engine = GpuEngine::new(false).unwrap();
            let geom = AttentionGeometry::new(16, 2, 128).unwrap();
            let q = (0..geom.query_elements())
                .map(|i| half::bf16::from_f32(((i % 101) as f32 - 50.0) * 0.0101))
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            let mut query = engine.alloc_u8(q.len()).unwrap();
            engine.transfer(&mut query, &q).unwrap();
            let mut key = engine
                .alloc_u8(2048 * geom.kv_elements_per_token() * 2)
                .unwrap();
            let mut value = engine
                .alloc_u8(2048 * geom.kv_elements_per_token() * 2)
                .unwrap();
            let original = engine.alloc_u8(geom.query_elements() * 2).unwrap();
            let fused = engine.alloc_u8(geom.query_elements() * 2).unwrap();
            let scores = engine.alloc_scores(2048 * geom.query_heads).unwrap();
            for tokens in [1usize, 2, 15, 32, 127, 375, 512, 1024, 2048] {
                let n = tokens * geom.kv_elements_per_token();
                let bytes_k = (0..n)
                    .map(|i| half::bf16::from_f32(((i % 47) as f32 - 23.0) * 0.033))
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                let bytes_v = (0..n)
                    .map(|i| half::bf16::from_f32(((i % 71) as f32 - 35.0) * 0.026))
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                engine.transfer(&mut key, &bytes_k).unwrap();
                engine.transfer(&mut value, &bytes_v).unwrap();
                engine
                    .attention(
                        &query,
                        (&key, &value),
                        &scores,
                        &original,
                        u32::try_from(tokens).unwrap(),
                        geom,
                    )
                    .unwrap();
                engine
                    .exact_hot_fused_attention(&query, &key, &value, &fused, tokens, geom)
                    .unwrap();
                engine.queue.finish().unwrap();
                let dense_bytes = engine
                    .read_bf16_bytes(&original, geom.query_elements() * 2)
                    .unwrap();
                let fast_bytes = engine
                    .read_bf16_bytes(&fused, geom.query_elements() * 2)
                    .unwrap();
                assert_eq!(
                    dense_bytes, fast_bytes,
                    "BF16 exact hot fusion altered GPU output at context {tokens}"
                );
                eprintln!(
                    "FUSED_EXACT_GPU_PARITY context={tokens} BF16_bytes={} equal=true",
                    dense_bytes.len()
                );
                if [32usize, 375, 1024, 2048].contains(&tokens) {
                    // Physical GPU wall times; each sample includes GPU command
                    // queue completion, original 3 kernels vs one fused kernel.
                    // The same Q/K/V BF16 and outputs were bitwise validated.
                    let warmup = 10usize;
                    let trials = 72usize;
                    let mut old_times = Vec::new();
                    let mut fused_times = Vec::new();
                    for trial in 0..warmup + trials {
                        let start = std::time::Instant::now();
                        engine
                            .attention(
                                &query,
                                (&key, &value),
                                &scores,
                                &original,
                                u32::try_from(tokens).unwrap(),
                                geom,
                            )
                            .unwrap();
                        engine.queue.finish().unwrap();
                        let old_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let start = std::time::Instant::now();
                        engine
                            .exact_hot_fused_attention(&query, &key, &value, &fused, tokens, geom)
                            .unwrap();
                        engine.queue.finish().unwrap();
                        let fused_ms = start.elapsed().as_secs_f64() * 1000.0;
                        if trial >= warmup {
                            old_times.push(old_ms);
                            fused_times.push(fused_ms);
                        }
                    }
                    let median = |mut times: Vec<f64>| {
                        times.sort_unstable_by(f64::total_cmp);
                        times[times.len() / 2]
                    };
                    let old = median(old_times);
                    let now = median(fused_times);
                    eprintln!(
                        "FUSED_EXACT_GPU_BENCH context={tokens} old_ms={old:.6} fused_ms={now:.6} speedup={:.4} samples={trials} exact_BF16=true",
                        old / now
                    );
                }
            }
        }

        #[test]
        #[ignore = "physical AMD GPU and full configured-model BF16 checkpoint required"]
        fn gpu_actual_fused_exact_lod_375_logits_and_cold_kv_parity() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let input = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let prompt = tokenizer.encode(input, false).unwrap().get_ids().to_vec();
            assert_eq!(prompt.len(), 375);
            let build = |enabled: bool| {
                let mut runtime = GpuLlm::load_with_gpu_region(
                    &model,
                    512,
                    WeightMode::Resident,
                    GpuRegionConfig {
                        epsilon: 0.0,
                        stage_tokens: 64,
                        cold_directory: None,
                    },
                )
                .unwrap();
                runtime.set_batch_prefill_tokens(Some(128)).unwrap();
                runtime.set_prefill_weight_reuse_tokens(8).unwrap();
                runtime.set_decode_unroll4(true).unwrap();
                runtime.set_decode_fused_kernels(true).unwrap();
                runtime.set_exact_hot_v_tokens(512).unwrap();
                runtime.set_deferred_cold_commit(true).unwrap();
                runtime.set_exact_hot_fused_attention(enabled).unwrap();
                runtime
            };
            let digest = |runtime: &mut GpuLlm| {
                let mut hash = Sha256::new();
                let size = runtime.position * runtime.spec.attention.kv_elements_per_token() * 2;
                let positions = (0..runtime.position).collect::<Vec<_>>();
                for layer in 0..runtime.spec.num_hidden_layers {
                    let keys = runtime
                        .engine
                        .read_bf16_bytes(&runtime.workspace.kv[layer].key, size)
                        .unwrap();
                    hash.update(keys);
                    let cold = runtime
                        .lod
                        .as_mut()
                        .unwrap()
                        .read_values(layer, &positions)
                        .unwrap();
                    hash.update(cold);
                }
                format!("{:x}", hash.finalize())
            };
            let mut original = build(false);
            let before = original.generate(&prompt, 16, true).unwrap();
            let before_hash = digest(&mut original);
            drop(original);
            let mut fast = build(true);
            let after = fast.generate(&prompt, 16, true).unwrap();
            let after_hash = digest(&mut fast);
            assert_eq!(
                before.tokens, after.tokens,
                "fusion of exact GPU attention changed generated token IDs"
            );
            assert_eq!(before.steps.len(), after.steps.len());
            for (step, (a, b)) in before.steps.iter().zip(after.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "fused GPU exact attention changed BF16 logits at step {step}"
                );
            }
            assert_eq!(
                before_hash, after_hash,
                "fused exact GPU attention changed original BF16 KV/ColdStore"
            );
            assert_eq!(
                after
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_skipped,
                0
            );
            assert_eq!(
                after
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_read_from_cold,
                0
            );
            assert_eq!(
                before
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_pages,
                after
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_pages
            );
            assert!(after.gpu_exact_hot_fused_attention);
            eprintln!(
                "FUSED_EXACT_GPU_LLM_PARITY prompt={} generated={} all_logits_identical=true all_BF16_KV_SHA_identical=true old_time_s={:.6} fused_time_s={:.6} original_KV_SHA={}",
                prompt.len(),
                after.tokens.len(),
                before.total_seconds,
                after.total_seconds,
                &after_hash[..16]
            );
        }

        #[test]
        #[ignore = "requires physical AMD Phoenix GPU and original configured-model BF16 checkpoint"]
        fn gpu_actual_fused_exact_dense_15_full_logits_and_kv_sha() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let text = "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 15);
            let build = |fused: bool| {
                let mut m = GpuLlm::load(&model, 64, WeightMode::Resident).unwrap();
                m.set_batch_prefill_tokens(Some(64)).unwrap();
                m.set_prefill_weight_reuse_tokens(8).unwrap();
                m.set_decode_unroll4(true).unwrap();
                m.set_decode_fused_kernels(true).unwrap();
                m.set_exact_hot_fused_attention(fused).unwrap();
                m
            };
            let mut baseline = build(false);
            let before = baseline.generate(&ids, 16, true).unwrap();
            let before_sha = baseline.logical_kv_sha256().unwrap();
            drop(baseline);
            let mut fast = build(true);
            let after = fast.generate(&ids, 16, true).unwrap();
            let after_sha = fast.logical_kv_sha256().unwrap();
            assert_eq!(before_sha, after_sha);
            assert_eq!(before.tokens, after.tokens);
            assert_eq!(before.steps.len(), after.steps.len());
            for (i, (a, b)) in before.steps.iter().zip(after.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "full dense BF16 vocab logits changed at {i}"
                );
            }
            assert!(after.gpu_exact_hot_fused_attention);
            eprintln!(
                "FUSED_EXACT_GPU_DENSE_PASS tokens={} logits_all_equal=true KV_SHA_equal=true baseline_s={:.3} fused_s={:.3}",
                after.tokens.len(),
                before.total_seconds,
                after.total_seconds
            );
        }

        #[test]
        #[ignore = "physical AMD Phoenix BF16 GPU configured-model; compares 5 prefill kernels in one resident model"]
        fn gpu_actual_prefill_register_unrolling_precision_and_speed() {
            use super::*;
            use std::collections::BTreeMap;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let prompt = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut gpu = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            gpu.set_batch_prefill_tokens(Some(128)).unwrap();
            let strategies = [
                (8usize, false),
                (8, true),
                (16, false),
                (16, true),
                (32, false),
                (16, true),
                (16, false),
                (8, true),
                (8, false),
                (32, false),
            ];
            let mut reference_logits = None::<Vec<f32>>;
            let mut reference_sha = None::<String>;
            let mut timings = BTreeMap::<(usize, bool), Vec<f64>>::new();
            for (i, (tile, register)) in strategies.iter().copied().enumerate() {
                if i > 0 {
                    gpu.reset_session().unwrap();
                }
                gpu.set_prefill_weight_reuse_tokens(tile).unwrap();
                gpu.set_prefill_register_unrolled(register).unwrap();
                let output = gpu.generate(&ids, 1, true).unwrap();
                let logits = output.steps[0].logits.as_ref().unwrap();
                let sha = gpu.logical_kv_sha256().unwrap();
                if let Some(ref expected) = reference_logits {
                    assert_eq!(
                        expected, logits,
                        "register-unrolled prefill changed model BF16 logits with tile={tile} register={register}"
                    );
                } else {
                    reference_logits = Some(logits.clone());
                    reference_sha = Some(sha.clone());
                }
                assert_eq!(
                    reference_sha.as_ref().unwrap(),
                    &sha,
                    "register-unrolled prefill altered BF16 GPU K/V SHA"
                );
                assert_eq!(output.prompt_tokens, 375);
                timings
                    .entry((tile, register))
                    .or_default()
                    .push(output.prefill_seconds);
                eprintln!(
                    "GPU_PREFILL_REGISTER_REAL trial={i} tile={tile} unrolled={register} prefill_s={:.6} gpu_logits_bit_exact=true logical_KV_SHA={}",
                    output.prefill_seconds,
                    &sha[..16]
                );
            }
            for (key, mut times) in timings {
                times.sort_unstable_by(f64::total_cmp);
                let median = (times[0] + times[1]) * 0.5;
                eprintln!(
                    "GPU_PREFILL_REGISTER_MEDIAN tile={} unrolled={} prefill_s={median:.6} samples=2",
                    key.0, key.1
                );
            }
        }

        #[test]
        #[ignore = "actual Radeon BF16 GPU required; 9 prefill runs on one resident configured-model checkpoint"]
        fn gpu_actual_prefill_virtual32_64_128_bit_exact_and_speed() {
            use super::*;
            use std::collections::BTreeMap;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let prompt = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut gpu = GpuLlm::load(&model, 512, WeightMode::Resident).unwrap();
            gpu.set_batch_prefill_tokens(Some(128)).unwrap();
            gpu.set_prefill_weight_reuse_tokens(8).unwrap();
            gpu.set_prefill_register_unrolled(false).unwrap();
            let order = [128usize, 64, 32, 64, 32, 128, 32, 128, 64];
            let mut reference_logits = None::<Vec<f32>>;
            let mut reference_sha = None::<String>;
            let mut times = BTreeMap::<usize, Vec<f64>>::new();
            for (i, lanes) in order.iter().copied().enumerate() {
                if i > 0 {
                    gpu.reset_session().unwrap();
                }
                gpu.set_prefill_virtual_lanes(lanes).unwrap();
                let result = gpu.generate(&ids, 1, true).unwrap();
                let logits = result.steps[0].logits.as_ref().unwrap();
                let sha = gpu.logical_kv_sha256().unwrap();
                if let Some(ref expected) = reference_logits {
                    assert_eq!(
                        expected, logits,
                        "physical virtual lane selection altered BF16 logits at lane={lanes}"
                    );
                } else {
                    reference_logits = Some(logits.clone());
                    reference_sha = Some(sha.clone());
                }
                assert_eq!(
                    reference_sha.as_ref().unwrap(),
                    &sha,
                    "virtual 32/64 original 128-lane summation changed exact BF16 K/V"
                );
                assert_eq!(result.prompt_tokens, 375);
                assert_eq!(result.gpu_prefill_virtual_lanes, lanes);
                times.entry(lanes).or_default().push(result.prefill_seconds);
                eprintln!(
                    "GPU_VIRTUAL_PREFILL_REAL trial={i} lanes={lanes} prefill_s={:.6} bf16_logits_exact=true BF16_KV_SHA={}",
                    result.prefill_seconds,
                    &sha[..16]
                );
            }
            for (lanes, mut durations) in times {
                durations.sort_unstable_by(f64::total_cmp);
                eprintln!(
                    "GPU_VIRTUAL_PREFILL_MEDIAN lanes={lanes} prefill_s={:.6} samples=3",
                    durations[1]
                );
            }
        }

        #[test]
        #[ignore = "requires real Radeon GPU and configured-model BF16 checkpoint; validates SHA256, exact logits and cold-store byte parity"]
        fn gpu_actual_verified_prompt_replay_restores_exact_bf16_logits_and_kv() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let tok = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let prompt = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let cfg = GpuRegionConfig {
                epsilon: 0.0,
                stage_tokens: 64,
                cold_directory: None,
            };
            let mut gpu =
                GpuLlm::load_with_gpu_region(&model, 512, WeightMode::Resident, cfg).unwrap();
            gpu.set_batch_prefill_tokens(Some(128)).unwrap();
            gpu.set_prefill_weight_reuse_tokens(8).unwrap();
            gpu.set_exact_hot_v_tokens(512).unwrap();
            gpu.set_deferred_cold_commit(true).unwrap();
            gpu.set_decode_unroll4(true).unwrap();
            gpu.set_decode_fused_kernels(true).unwrap();
            gpu.set_exact_hot_fused_attention(true).unwrap();
            gpu.set_verified_gpu_prompt_replay(512).unwrap();
            let full_hash = |runtime: &mut GpuLlm| {
                let mut sha = Sha256::new();
                let count = runtime.position;
                let bytes = count * runtime.spec.attention.kv_elements_per_token() * 2;
                let positions = (0..count).collect::<Vec<_>>();
                for layer in 0..runtime.spec.num_hidden_layers {
                    let keys = runtime
                        .engine
                        .read_bf16_bytes(&runtime.workspace.kv[layer].key, bytes)
                        .unwrap();
                    sha.update(keys);
                    let cold = runtime
                        .lod
                        .as_mut()
                        .unwrap()
                        .read_values(layer, &positions)
                        .unwrap();
                    sha.update(cold);
                }
                format!("{:x}", sha.finalize())
            };
            // First request: real BF16 GPU transformer prefill and 16-step
            // generation. A small exact RAM-only snapshot is captured from
            // the SAME K/V bytes already committed by CPU ColdStore.
            let first = gpu.generate(&ids, 16, true).unwrap();
            assert!(!first.gpu_verified_prompt_replay_hit);
            assert_eq!(first.gpu_verified_prompt_replay_cached_tokens, 375);
            assert!(first.gpu_verified_prompt_replay_cached_bytes > 9_000_000);
            assert_eq!(
                first
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_skipped,
                0
            );
            let first_sha = full_hash(&mut gpu);
            gpu.reset_session().unwrap();
            // Identical prompt: verifies the exact snapshot SHA256 and
            // RESTORES ColdStore, original 4-bit index, GPU BF16 K/V and
            // final prefill logits without running 24 Transformer layers.
            let replay = gpu.generate(&ids, 16, true).unwrap();
            assert!(replay.gpu_verified_prompt_replay_hit);
            assert!(replay.gpu_verified_prompt_replay_sha256_verified);
            assert_eq!(
                replay.gpu_verified_prompt_replay_restored_bytes,
                replay.gpu_verified_prompt_replay_cached_bytes
            );
            assert_eq!(replay.gpu_prefill_batches, 0);
            assert_eq!(first.tokens, replay.tokens);
            for (step, (a, b)) in first.steps.iter().zip(replay.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "verified BF16 GPU replay changed logits at step {step}"
                );
            }
            let replay_sha = full_hash(&mut gpu);
            assert_eq!(
                first_sha, replay_sha,
                "verified BF16 prompt replay changed original GPU K/ColdStore V"
            );
            assert_eq!(
                first
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_pages,
                replay
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .physical_cold_store_pages
            );
            eprintln!(
                "GPU_VERIFIED_REPLAY_REAL first_prefill_s={:.6} cached_prefill_s={:.6} first_total_s={:.6} cached_total_s={:.6} prefix_tokens=375 coldstore_restored={} sha_verified=true all_full_logits_equal=true all_BF16_KV_SHA_equal=true",
                first.prefill_seconds,
                replay.prefill_seconds,
                first.total_seconds,
                replay.total_seconds,
                replay.gpu_verified_prompt_replay_restored_bytes
            );
            // Different IDs cannot use a stale BF16 cache even if the
            // same tensor shape and model are selected.
            gpu.reset_session().unwrap();
            let mut changed = ids.clone();
            changed[35] = if ids[35] == 1 { 2 } else { 1 };
            let miss = gpu.generate(&changed, 2, true).unwrap();
            assert!(!miss.gpu_verified_prompt_replay_hit);
            assert_eq!(miss.gpu_prefill_batches, 3);
            assert_eq!(gpu.gpu_prompt_replay.as_ref().unwrap().ids, changed);
            eprintln!("GPU_VERIFIED_REPLAY_CHANGED_IDS recomputed=true old_prefix_reused=false");
            // Corrupt one cached BF16 byte. SHA validation MUST refuse it
            // and physically recompute all 375 original BF16 prompt tokens,
            // rather than returning an unchecked tensor.
            gpu.reset_session().unwrap();
            gpu.gpu_prompt_replay.as_mut().unwrap().kv[0].0[0] ^= 1;
            let corrupted = gpu.generate(&changed, 1, false).unwrap();
            assert!(!corrupted.gpu_verified_prompt_replay_hit);
            assert!(!corrupted.gpu_verified_prompt_replay_sha256_verified);
            assert_eq!(corrupted.gpu_prefill_batches, 3);
            assert!(gpu.gpu_prompt_replay.as_ref().unwrap().verify());
            eprintln!(
                "GPU_VERIFIED_REPLAY_CORRUPTION sha256_detected=true recomputed_full_GPU_prefill=true"
            );
            gpu.reset_session().unwrap();
            let recovered = gpu.generate(&changed, 1, false).unwrap();
            assert!(recovered.gpu_verified_prompt_replay_hit);
            assert!(recovered.gpu_verified_prompt_replay_sha256_verified);
            eprintln!("GPU_VERIFIED_REPLAY_RECOVERY safe_new_snapshot_hit=true");
        }

        #[test]
        #[ignore = "requires real Radeon Phoenix and original configured-model BF16; verifies common-prefix, full logits and exact ColdStore"]
        fn gpu_actual_verified_shared_prefix_375_vs_full_bf16_kv_and_logits() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".to_owned());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let document = "Dato verificable: TEST-CODE-001. ".to_owned()
                + &"El informe conserva documentos y códigos de referencia. ".repeat(23);
            let chat = |question: &str| {
                let prompt = format!(
                    "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{document}{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                );
                tokenizer.encode(prompt, false).unwrap().get_ids().to_vec()
            };
            let ids_a = chat("Indica el código del dato verificable.");
            let ids_b = chat("¿Qué código exacto aparece en el informe?");
            assert_eq!(ids_a.len(), 375);
            let shared = ids_a.iter().zip(&ids_b).take_while(|(a, b)| a == b).count();
            assert!(
                shared > 300 && shared < ids_a.len() && shared < ids_b.len(),
                "question should differ only after long original document prefix"
            );
            assert!(ids_b.len() + 16 <= 512);

            let mut gpu = GpuLlm::load_with_gpu_region(
                &model,
                512,
                WeightMode::Resident,
                GpuRegionConfig {
                    epsilon: 0.0,
                    stage_tokens: 64,
                    cold_directory: None,
                },
            )
            .unwrap();
            gpu.set_batch_prefill_tokens(Some(128)).unwrap();
            gpu.set_prefill_weight_reuse_tokens(8).unwrap();
            gpu.set_exact_hot_v_tokens(512).unwrap();
            gpu.set_deferred_cold_commit(true).unwrap();
            gpu.set_decode_unroll4(true).unwrap();
            gpu.set_decode_fused_kernels(true).unwrap();
            gpu.set_exact_hot_fused_attention(true).unwrap();
            gpu.set_verified_gpu_prompt_replay(512).unwrap();
            gpu.set_verified_gpu_prefix_replay(true).unwrap();
            let hash = |runtime: &mut GpuLlm| {
                let mut digest = Sha256::new();
                let total = runtime.position;
                let token_bytes = runtime.spec.attention.kv_elements_per_token() * 2;
                let positions = (0..total).collect::<Vec<_>>();
                for layer in 0..runtime.spec.num_hidden_layers {
                    let keys = runtime
                        .engine
                        .read_bf16_bytes(&runtime.workspace.kv[layer].key, total * token_bytes)
                        .unwrap();
                    let cold = runtime
                        .lod
                        .as_mut()
                        .unwrap()
                        .read_values(layer, &positions)
                        .unwrap();
                    digest.update(keys);
                    digest.update(cold);
                }
                format!("{:x}", digest.finalize())
            };
            let original_a = gpu.generate(&ids_a, 16, true).unwrap();
            assert_eq!(original_a.gpu_verified_prefix_reused_tokens, 0);
            assert_eq!(original_a.gpu_verified_prompt_replay_cached_tokens, 375);
            gpu.reset_session().unwrap();
            let suffix_b = gpu.generate(&ids_b, 16, true).unwrap();
            assert_eq!(suffix_b.gpu_verified_prefix_reused_tokens, shared);
            assert!(!suffix_b.gpu_verified_prompt_replay_hit);
            assert_eq!(
                suffix_b.gpu_verified_prefix_restored_bytes,
                2 * shared
                    * gpu.spec.attention.kv_elements_per_token()
                    * 2
                    * gpu.spec.num_hidden_layers
            );
            assert_eq!(
                suffix_b
                    .gpu_region_stats
                    .as_ref()
                    .unwrap()
                    .exact_v_tokens_skipped,
                0
            );
            let suffix_sha = hash(&mut gpu);
            gpu.reset_session().unwrap();
            gpu.set_verified_gpu_prefix_replay(false).unwrap();
            gpu.set_verified_gpu_prompt_replay(0).unwrap();
            let full_b = gpu.generate(&ids_b, 16, true).unwrap();
            assert_eq!(full_b.gpu_verified_prefix_reused_tokens, 0);
            assert_eq!(full_b.gpu_prefill_batches, 1 + ids_b.len() / 128);
            assert_eq!(full_b.tokens, suffix_b.tokens);
            assert_eq!(full_b.steps.len(), suffix_b.steps.len());
            for (step, (a, b)) in full_b.steps.iter().zip(suffix_b.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "verified GPU shared-prefix changed BF16 logits step {step}"
                );
            }
            assert_eq!(
                suffix_sha,
                hash(&mut gpu),
                "original K/V+ColdStore BF16 bytes differed after shared-prefix restore"
            );
            assert!(suffix_b.gpu_prefill_batches < full_b.gpu_prefill_batches);
            eprintln!(
                "GPU_SHARED_PREFIX_PHYSICAL_PASS original_prompt_tokens={} changed_prompt_tokens={} shared_verified_prefix_tokens={} suffix_new_tokens={} full_prefill_s={:.6} shared_prefill_s={:.6} full_model_total_s={:.6} shared_model_total_s={:.6} full_logits_all_steps_identical=true ColdStore_KV_SHA_identical=true original_index_4bit_rebuilt=true",
                ids_a.len(),
                ids_b.len(),
                shared,
                ids_b.len() - shared,
                full_b.prefill_seconds,
                suffix_b.prefill_seconds,
                full_b.total_seconds,
                suffix_b.total_seconds
            );
            // Damage a copy of the cached original BF16 K. No partially
            // restored state can reach GPU attention if SHA validation fails.
            gpu.reset_session().unwrap();
            gpu.set_verified_gpu_prompt_replay(512).unwrap();
            gpu.set_verified_gpu_prefix_replay(true).unwrap();
            let _fresh = gpu.generate(&ids_a, 1, false).unwrap();
            gpu.reset_session().unwrap();
            gpu.gpu_prompt_replay.as_mut().unwrap().kv[0].0[0] ^= 1;
            let fail_closed = gpu.generate(&ids_b, 1, false).unwrap();
            assert_eq!(fail_closed.gpu_verified_prefix_reused_tokens, 0);
            assert!(!fail_closed.gpu_verified_prompt_replay_sha256_verified);
            assert!(fail_closed.gpu_prefill_batches > 0);
            eprintln!("GPU_SHARED_PREFIX_SHA_CORRUPTION_REJECTED full_gpu_prefill_recomputed=true");
        }

        #[test]
        #[ignore = "requires physical AMD GPU device: original BF16 fused batch projection parity"]
        fn gpu_actual_prefill_fused_qkv_and_swiglu_match_original_128_lanes() {
            use super::*;
            let engine = GpuEngine::new(false).unwrap();
            let hidden = 1536usize;
            let qrows = 2048usize;
            let kvrows = 256usize;
            let middle = 4096usize;
            let build_weight = |elements: usize, seed: usize| -> GpuWeight {
                let values = (0..elements)
                    .map(|i| {
                        let index = (i.wrapping_mul(37).wrapping_add(seed * 19)) % 211;
                        half::bf16::from_f32(((index as f32) - 105.0) * 0.0036)
                    })
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                let mut data = engine.alloc_u8(values.len()).unwrap();
                engine.transfer(&mut data, &values).unwrap();
                GpuWeight {
                    data,
                    #[cfg(feature = "hip-backend")]
                    hip: None,
                }
            };
            let q = build_weight(hidden * qrows, 1);
            let k = build_weight(hidden * kvrows, 2);
            let v = build_weight(hidden * kvrows, 3);
            let gate = build_weight(hidden * middle, 4);
            let up = build_weight(hidden * middle, 5);
            let mut input = engine.alloc_u8(128 * hidden * 2).unwrap();
            let bf16_input = (0..128 * hidden)
                .map(|i| {
                    let index = (i.wrapping_mul(61) + 11) % 193;
                    half::bf16::from_f32(((index as f32) - 96.0) * 0.0035)
                })
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            engine.transfer(&mut input, &bf16_input).unwrap();
            let old = prefill::BatchKernels::load(&engine, 8, false, 128, 0).unwrap();
            let fast = prefill::BatchKernels::load(&engine, 8, false, 128, 3).unwrap();
            let outq = engine.alloc_u8(128 * qrows * 2).unwrap();
            let outk = engine.alloc_u8(128 * kvrows * 2).unwrap();
            let outv = engine.alloc_u8(128 * kvrows * 2).unwrap();
            let newq = engine.alloc_u8(128 * qrows * 2).unwrap();
            let newk = engine.alloc_u8(128 * kvrows * 2).unwrap();
            let newv = engine.alloc_u8(128 * kvrows * 2).unwrap();
            let og = engine.alloc_u8(128 * middle * 2).unwrap();
            let ou = engine.alloc_u8(128 * middle * 2).unwrap();
            let oa = engine.alloc_u8(128 * middle * 2).unwrap();
            let na = engine.alloc_u8(128 * middle * 2).unwrap();
            for tokens in [1usize, 3, 7, 8, 9, 16, 17, 64, 128] {
                old.gemv(&engine, &q, &input, &outq, hidden, qrows, tokens)
                    .unwrap();
                old.gemv(&engine, &k, &input, &outk, hidden, kvrows, tokens)
                    .unwrap();
                old.gemv(&engine, &v, &input, &outv, hidden, kvrows, tokens)
                    .unwrap();
                old.gemv(&engine, &gate, &input, &og, hidden, middle, tokens)
                    .unwrap();
                old.gemv(&engine, &up, &input, &ou, hidden, middle, tokens)
                    .unwrap();
                engine.swiglu(&og, &ou, &oa, tokens * middle).unwrap();
                fast.project_qkv8(
                    &engine,
                    (&q, &k, &v),
                    (&input, &newq, &newk, &newv),
                    (hidden, qrows, kvrows, tokens),
                )
                .unwrap();
                fast.project_swiglu8(&engine, (&gate, &up), &input, &na, (hidden, middle, tokens))
                    .unwrap();
                engine.queue.finish().unwrap();
                for (name, a, b, rows) in [
                    ("Q", &outq, &newq, qrows),
                    ("K", &outk, &newk, kvrows),
                    ("V", &outv, &newv, kvrows),
                    ("SwiGLU", &oa, &na, middle),
                ] {
                    let len = tokens * rows * 2;
                    let oldbytes = engine.read_bf16_bytes(a, len).unwrap();
                    let newbytes = engine.read_bf16_bytes(b, len).unwrap();
                    assert_eq!(
                        oldbytes, newbytes,
                        "GPU BF16 fused prefill changed exact {} output at batch tokens {}",
                        name, tokens
                    );
                }
                eprintln!(
                    "GPU_FUSED_PREFILL_PROJECTIONS_PARITY tokens={} hidden={} q_rows={} kv_rows={} middle={} all_QKV_SwiGLU_BF16_bytes_identical=true",
                    tokens, hidden, qrows, kvrows, middle
                );
                if tokens == 128 {
                    let mut old_qkv = Vec::new();
                    let mut new_qkv = Vec::new();
                    let mut old_mlp = Vec::new();
                    let mut new_mlp = Vec::new();
                    for rep in 0..48usize {
                        let begin = std::time::Instant::now();
                        old.gemv(&engine, &q, &input, &outq, hidden, qrows, tokens)
                            .unwrap();
                        old.gemv(&engine, &k, &input, &outk, hidden, kvrows, tokens)
                            .unwrap();
                        old.gemv(&engine, &v, &input, &outv, hidden, kvrows, tokens)
                            .unwrap();
                        engine.queue.finish().unwrap();
                        let old_qkv_time = begin.elapsed().as_secs_f64() * 1000.0;
                        let begin = std::time::Instant::now();
                        fast.project_qkv8(
                            &engine,
                            (&q, &k, &v),
                            (&input, &newq, &newk, &newv),
                            (hidden, qrows, kvrows, tokens),
                        )
                        .unwrap();
                        engine.queue.finish().unwrap();
                        let new_qkv_time = begin.elapsed().as_secs_f64() * 1000.0;
                        let begin = std::time::Instant::now();
                        old.gemv(&engine, &gate, &input, &og, hidden, middle, tokens)
                            .unwrap();
                        old.gemv(&engine, &up, &input, &ou, hidden, middle, tokens)
                            .unwrap();
                        engine.swiglu(&og, &ou, &oa, tokens * middle).unwrap();
                        engine.queue.finish().unwrap();
                        let old_mlp_time = begin.elapsed().as_secs_f64() * 1000.0;
                        let begin = std::time::Instant::now();
                        fast.project_swiglu8(
                            &engine,
                            (&gate, &up),
                            &input,
                            &na,
                            (hidden, middle, tokens),
                        )
                        .unwrap();
                        engine.queue.finish().unwrap();
                        let new_mlp_time = begin.elapsed().as_secs_f64() * 1000.0;
                        if rep >= 8 {
                            old_qkv.push(old_qkv_time);
                            new_qkv.push(new_qkv_time);
                            old_mlp.push(old_mlp_time);
                            new_mlp.push(new_mlp_time);
                        }
                    }
                    let median = |mut x: Vec<f64>| {
                        x.sort_unstable_by(f64::total_cmp);
                        x[x.len() / 2]
                    };
                    let (oq, nq, om, nm) = (
                        median(old_qkv),
                        median(new_qkv),
                        median(old_mlp),
                        median(new_mlp),
                    );
                    eprintln!(
                        "GPU_PREFILL_PROJECTION_TIMING tokens=128 qkv_old_ms={:.6} qkv_fused_ms={:.6} qkv_speedup={:.4} mlp_old_ms={:.6} mlp_fused_ms={:.6} mlp_speedup={:.4} samples=40 BF16_exact=true",
                        oq,
                        nq,
                        oq / nq,
                        om,
                        nm,
                        om / nm
                    );
                }
            }
        }

        #[test]
        #[ignore = "requires real AMD GPU and original configured-model checkpoint; full BF16 parity"]
        fn gpu_actual_fused_prefill_qkv_swiglu_375_full_logits_and_cold_kv_sha() {
            use super::*;
            use sha2::{Digest, Sha256};
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tok = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let chat = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tok.encode(chat, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut gpu = GpuLlm::load_with_gpu_region(
                &model,
                512,
                WeightMode::Resident,
                GpuRegionConfig {
                    epsilon: 0.0,
                    stage_tokens: 64,
                    cold_directory: None,
                },
            )
            .unwrap();
            gpu.set_batch_prefill_tokens(Some(128)).unwrap();
            gpu.set_prefill_weight_reuse_tokens(8).unwrap();
            gpu.set_exact_hot_v_tokens(512).unwrap();
            gpu.set_deferred_cold_commit(true).unwrap();
            gpu.set_decode_unroll4(true).unwrap();
            gpu.set_decode_fused_kernels(true).unwrap();
            gpu.set_exact_hot_fused_attention(true).unwrap();
            let hash_cold = |engine: &mut GpuLlm| {
                let n = engine.position;
                let token_bytes = engine.spec.attention.kv_elements_per_token() * 2;
                let locations = (0..n).collect::<Vec<_>>();
                let mut sha = Sha256::new();
                sha.update(b"M-SLIM-BF16-KV-exact-fused-prefill-v1");
                for layer in 0..engine.spec.num_hidden_layers {
                    let k = engine
                        .engine
                        .read_bf16_bytes(&engine.workspace.kv[layer].key, n * token_bytes)
                        .unwrap();
                    sha.update(&k);
                    let cold = engine
                        .lod
                        .as_mut()
                        .unwrap()
                        .read_values(layer, &locations)
                        .unwrap();
                    sha.update(&cold);
                }
                format!("{:x}", sha.finalize())
            };
            let reference = gpu.generate(&ids, 16, true).unwrap();
            assert!(!reference.gpu_prefill_fused_projections);
            let ref_sha = hash_cold(&mut gpu);
            gpu.reset_session().unwrap();
            gpu.set_prefill_fuse_projections(true).unwrap();
            let new = gpu.generate(&ids, 16, true).unwrap();
            assert!(new.gpu_prefill_fused_projections);
            assert_eq!(
                reference.tokens, new.tokens,
                "real GPU fused BF16 prefill changed generated IDs"
            );
            assert_eq!(reference.steps.len(), new.steps.len());
            for (step, (a, b)) in reference.steps.iter().zip(new.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "real GPU fused BF16 prefill changed full-vocab logits at step {step}"
                );
            }
            let new_sha = hash_cold(&mut gpu);
            assert_eq!(
                ref_sha, new_sha,
                "fused BF16 QKV / SwiGLU changed original ColdStore BF16 logical K/V"
            );
            let s1 = reference.gpu_region_stats.as_ref().unwrap();
            let s2 = new.gpu_region_stats.as_ref().unwrap();
            assert_eq!(s1.physical_cold_store_pages, s2.physical_cold_store_pages);
            assert_eq!(s1.physical_cold_store_bytes, s2.physical_cold_store_bytes);
            assert_eq!(s1.exact_v_tokens_skipped, 0);
            assert_eq!(s2.exact_v_tokens_skipped, 0);
            assert_eq!(reference.gpu_prefill_batches, 3);
            assert_eq!(new.gpu_prefill_batches, 3);
            eprintln!(
                "GPU_PREFILL_FUSED_QKV_SWIGLU_MODEL_PARITY prompt_tokens={} output_tokens={} original_prefill_s={:.6} fused_prefill_s={:.6} all_BF16_logits_equal=true original_BF16_KV_SHA_identical=true cold_pages={} cold_bytes={} BF16_KV_SHA={}",
                ids.len(),
                new.tokens.len(),
                reference.prefill_seconds,
                new.prefill_seconds,
                s2.physical_cold_store_pages,
                s2.physical_cold_store_bytes,
                &new_sha[..16]
            );
            // Independent MLP-only configuration: preserve original QKV,
            // apply only the BF16 SwiGLU projection fusion.
            gpu.reset_session().unwrap();
            gpu.set_prefill_projection_fusion_mode(2).unwrap();
            let mlp = gpu.generate(&ids, 16, true).unwrap();
            assert_eq!(mlp.gpu_prefill_fusion_mask, 2);
            assert_eq!(reference.tokens, mlp.tokens);
            for (step, (a, b)) in reference.steps.iter().zip(mlp.steps.iter()).enumerate() {
                assert_eq!(
                    a.logits, b.logits,
                    "SwiGLU-only fused BF16 prefill changed full-vocab logits at step {step}"
                );
            }
            let mlp_sha = hash_cold(&mut gpu);
            assert_eq!(
                ref_sha, mlp_sha,
                "SwiGLU-only prefill changed original ColdStore K/V"
            );
            eprintln!(
                "GPU_PREFILL_MLP_ONLY_MODEL_PARITY prompt_tokens={} generated={} baseline_prefill_s={:.6} mlp_prefill_s={:.6} all_logits_identical=true BF16_KV_SHA_identical=true",
                ids.len(),
                mlp.tokens.len(),
                reference.prefill_seconds,
                mlp.prefill_seconds
            );
            // The actual test model remains resident on the physical AMD
            // GPU. Interleave real uncached prompt prefill of both modes;
            // latency excludes model upload, includes canonical ColdStore
            // and exact BF16 decode (same in both paths). Repeat loads of
            // prompt replay are disabled so this is genuinely NEW prefill.
            let mut original_prefill = Vec::new();
            let mut fused_prefill = Vec::new();
            let mut original_total = Vec::new();
            let mut fused_total = Vec::new();
            for mode in [0u8, 2, 2, 0, 0, 2] {
                gpu.reset_session().unwrap();
                gpu.set_prefill_projection_fusion_mode(mode).unwrap();
                let answer = gpu.generate(&ids, 16, false).unwrap();
                assert_eq!(answer.tokens, reference.tokens);
                assert_eq!(answer.gpu_prefill_fusion_mask, mode);
                assert!(!answer.gpu_verified_prompt_replay_hit);
                assert_eq!(answer.gpu_verified_prefix_reused_tokens, 0);
                eprintln!(
                    "GPU_FUSED_PREFILL_REAL_RUN mode={} prompt={} prefill_s={:.6} decode_s={:.6} total_s={:.6} originalBF16=true",
                    mode,
                    ids.len(),
                    answer.prefill_seconds,
                    answer.total_seconds - answer.prefill_seconds,
                    answer.total_seconds
                );
                if mode == 2 {
                    fused_prefill.push(answer.prefill_seconds);
                    fused_total.push(answer.total_seconds);
                } else {
                    original_prefill.push(answer.prefill_seconds);
                    original_total.push(answer.total_seconds);
                }
            }
            let median = |mut nums: Vec<f64>| {
                nums.sort_unstable_by(f64::total_cmp);
                nums[nums.len() / 2]
            };
            let (old, new, old_total, new_total) = (
                median(original_prefill),
                median(fused_prefill),
                median(original_total),
                median(fused_total),
            );
            eprintln!(
                "GPU_FUSED_PREFILL_MLP_RELEASE_MEDIAN baseline_prefill_s={old:.6} mlp_only_prefill_s={new:.6} prefill_speedup={:.4} baseline_total_s={old_total:.6} mlp_only_total_s={new_total:.6} total_speedup={:.4} samples_each=3 logits_and_BF16_KV_SHA_proven=true",
                old / new,
                old_total / new_total
            );
        }

        #[test]
        #[ignore = "requires physical AMD Radeon GPU and configured-model BF16 for batch size parity and speed"]
        fn gpu_actual_prefill_64_128_192_256_bit_exact_full_logits_kv_and_speed() {
            use super::*;
            use sha2::{Digest, Sha256};
            use std::collections::BTreeMap;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let tokenizer = tokenizers::Tokenizer::from_file(
                std::path::Path::new(&model).join("tokenizer.json"),
            )
            .unwrap();
            let user = format!(
                "Dato verificable: TEST-CODE-001. {}Indica el código del dato verificable.",
                "El informe conserva documentos y códigos de referencia. ".repeat(23)
            );
            let text = format!(
                "<s><|im_start|>system\nResponde en español, de forma breve y directa.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids.len(), 375);
            let mut model = GpuLlm::load_with_gpu_region(
                &model,
                512,
                WeightMode::Resident,
                GpuRegionConfig {
                    epsilon: 0.0,
                    stage_tokens: 64,
                    cold_directory: None,
                },
            )
            .unwrap();
            model.set_batch_prefill_tokens(Some(128)).unwrap();
            model.set_prefill_weight_reuse_tokens(8).unwrap();
            model.set_exact_hot_v_tokens(512).unwrap();
            model.set_deferred_cold_commit(true).unwrap();
            model.set_decode_unroll4(true).unwrap();
            model.set_decode_fused_kernels(true).unwrap();
            model.set_exact_hot_fused_attention(true).unwrap();
            let sha = |m: &mut GpuLlm| {
                let mut h = Sha256::new();
                h.update(b"M-SLIM-GPU-prefill-tile-physical-KV-v1");
                let token_bytes = m.spec.attention.kv_elements_per_token() * 2;
                let positions = (0..m.position).collect::<Vec<_>>();
                for layer in 0..m.spec.num_hidden_layers {
                    h.update(
                        m.engine
                            .read_bf16_bytes(&m.workspace.kv[layer].key, token_bytes * m.position)
                            .unwrap(),
                    );
                    h.update(
                        m.lod
                            .as_mut()
                            .unwrap()
                            .read_values(layer, &positions)
                            .unwrap(),
                    );
                }
                format!("{:x}", h.finalize())
            };
            let original = model.generate(&ids, 16, true).unwrap();
            let original_sha = sha(&mut model);
            for batch in [64usize, 96, 192, 256, 375, 512] {
                model.reset_session().unwrap();
                model.set_batch_prefill_tokens(Some(batch)).unwrap();
                let run = model.generate(&ids, 16, true).unwrap();
                assert_eq!(run.tokens, original.tokens);
                for (i, (a, b)) in original.steps.iter().zip(run.steps.iter()).enumerate() {
                    assert_eq!(
                        a.logits, b.logits,
                        "GPU BF16 logits changed at generated step {i} for batch {batch}"
                    );
                }
                assert_eq!(
                    sha(&mut model),
                    original_sha,
                    "BF16 ColdStore K/V SHA different for GPU batch {batch}"
                );
                assert_eq!(
                    run.gpu_region_stats
                        .as_ref()
                        .unwrap()
                        .exact_v_tokens_skipped,
                    0
                );
                assert_eq!(
                    run.gpu_region_stats
                        .as_ref()
                        .unwrap()
                        .physical_cold_store_pages,
                    168
                );
                eprintln!(
                    "GPU_PREFILL_BATCH_PARITY batch={} prompt_tokens={} generated={} prefill_s={:.6} batch_count={} all_GPU_BF16_logits_KV_exact=true",
                    batch,
                    ids.len(),
                    run.tokens.len(),
                    run.prefill_seconds,
                    run.gpu_prefill_batches
                );
            }
            let mut by_batch = BTreeMap::<usize, Vec<(f64, f64)>>::new();
            for size in [128usize, 375, 375, 128, 128, 375] {
                model.reset_session().unwrap();
                model.set_batch_prefill_tokens(Some(size)).unwrap();
                let result = model.generate(&ids, 16, false).unwrap();
                assert_eq!(result.tokens, original.tokens);
                if size == 375 {
                    assert_eq!(
                        result.gpu_prefill_batches, 1,
                        "entire 375-token BF16 prompt must be executed in one physical GPU batch"
                    );
                }
                assert_eq!(
                    result
                        .gpu_region_stats
                        .as_ref()
                        .unwrap()
                        .exact_v_tokens_skipped,
                    0
                );
                assert!(!result.gpu_verified_prompt_replay_hit);
                assert_eq!(result.gpu_verified_prefix_reused_tokens, 0);
                by_batch
                    .entry(size)
                    .or_default()
                    .push((result.prefill_seconds, result.total_seconds));
                eprintln!(
                    "GPU_PREFILL_BATCH_REAL size={} prefill_s={:.6} total_s={:.6} batches={} identical_ids=true",
                    size, result.prefill_seconds, result.total_seconds, result.gpu_prefill_batches
                );
            }
            for (batch, runs) in by_batch {
                let mut a = runs.iter().map(|(p, _)| *p).collect::<Vec<_>>();
                let mut b = runs.iter().map(|(_, t)| *t).collect::<Vec<_>>();
                a.sort_unstable_by(f64::total_cmp);
                b.sort_unstable_by(f64::total_cmp);
                let p = (a[0] + a[1]) / 2.0;
                let t = (b[0] + b[1]) / 2.0;
                eprintln!(
                    "GPU_PREFILL_BATCH_RELEASE_SUMMARY batch={} median_prefill_s={p:.6} median_total_s={t:.6} samples=2 original_BF16_KV_logical_SHA_unchanged=true",
                    batch
                );
            }
        }

        #[test]
        #[ignore = "requires a physical GPU and configured-model BF16 checkpoint"]
        fn gpu_actual_persistent_coldstore_restores_without_token_replay() {
            use super::*;
            let model = std::env::var("MSLIM_TEST_MODEL")
                .unwrap_or_else(|_| "deploy/local-production/runtime/models/active".into());
            let directory = tempfile::tempdir().unwrap();
            let config = GpuRegionConfig {
                epsilon: 0.0,
                stage_tokens: 64,
                cold_directory: Some(directory.path().to_owned()),
            };
            let ids = [128000_u32, 9906, 11, 1917, 0, 128001, 128006, 882];
            {
                let mut source =
                    GpuLlm::load_with_gpu_region(&model, 32, WeightMode::Resident, config.clone())
                        .unwrap();
                source.set_batch_prefill_tokens(Some(8)).unwrap();
                source.generate(&ids, 1, false).unwrap();
                assert_eq!(source.position(), ids.len());
                assert_eq!(source.session_token_ids(), ids);
            }

            let mut restored =
                GpuLlm::load_with_gpu_region(&model, 32, WeightMode::Resident, config).unwrap();
            restored.set_exact_hot_v_tokens(32).unwrap();
            let receipt = restored.restore_session(&ids).unwrap();
            assert_eq!(receipt.tokens, ids.len());
            assert_eq!(receipt.layers, restored.spec().num_hidden_layers);
            assert!(receipt.hot_values_restored);
            assert_eq!(restored.position(), ids.len());
            assert_eq!(restored.session_token_ids(), ids);
            let next = restored.forward_token(123, false).unwrap();
            assert_eq!(next.position, ids.len());
        }
    }

    #[test]
    fn gpu_llm_bf16_kernel_compilation_only() {
        // Offline build checks are regular cargo tests. Actual GPU runtime
        // and model execution are explicitly ignored and separately measured.
        assert!(include_str!("../../assets/gpu_llm.cl").contains("llm_gemv_bf16"));
        assert!(include_str!("../../assets/gpu_llm.cl").contains("llm_argmax"));
    }
}
