//! Concrete accelerator implementation behind the backend-neutral adapter contract.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use mslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use serde_json::{Value, json};

use crate::gpu_region_store::GpuRegionConfig;
use crate::model_bundle::ModelBundle;
use crate::opencl_llm::{OpenClLlm, WeightMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenClKvMode {
    Dense,
    ExactColdStore,
}

impl OpenClKvMode {
    fn parse(config: &Value) -> Result<Self> {
        match config
            .get("mslim_kv_mode")
            .and_then(Value::as_str)
            .unwrap_or("exact-coldstore")
        {
            "dense" => Ok(Self::Dense),
            "exact-coldstore" => Ok(Self::ExactColdStore),
            other => anyhow::bail!(
                "unsupported mslim_kv_mode: {other}; expected dense or exact-coldstore"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::ExactColdStore => "exact-coldstore",
        }
    }
}

pub struct OpenClModelAdapter {
    capabilities: AdapterCapabilities,
    model_dir: Option<PathBuf>,
    runtime: Option<OpenClLlm>,
    last_logits: Option<Vec<f32>>,
    cached_prompt: Vec<u32>,
    kv_mode: OpenClKvMode,
}

impl Default for OpenClModelAdapter {
    fn default() -> Self {
        Self {
            capabilities: AdapterCapabilities {
                adapter_id: "accelerator-runtime".into(),
                backend: "accelerator".into(),
                supports_prefill: true,
                supports_decode: true,
                supports_kv_restore: false,
                supports_streaming: false,
            },
            model_dir: None,
            runtime: None,
            last_logits: None,
            cached_prompt: Vec::new(),
            kv_mode: OpenClKvMode::ExactColdStore,
        }
    }
}

impl OpenClModelAdapter {
    pub(crate) fn calibration_runtime_mut(&mut self) -> Result<&mut OpenClLlm> {
        self.runtime.as_mut().context("accelerator is not loaded")
    }
}

impl ModelAdapter for OpenClModelAdapter {
    fn capabilities(&self) -> &AdapterCapabilities {
        &self.capabilities
    }

    fn inspect(&self, model_dir: &Path, config: &Value) -> Result<()> {
        let bundle = ModelBundle::load(model_dir)?;
        let spec = &bundle.spec;
        ensure!(
            spec.intermediate_size.is_some(),
            "model lacks intermediate_size"
        );
        ensure!(spec.vocab_size.is_some(), "model lacks vocab_size");
        bundle.validate_decoder_layout()?;
        let context = config
            .get("max_context_tokens")
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .context("model configuration must declare a positive context limit")?;
        ensure!(context > 0, "model context limit must be positive");
        Ok(())
    }

    fn load(&mut self, model_dir: &Path, config: &Value) -> Result<()> {
        self.inspect(model_dir, config)?;
        let max_tokens = config
            .get("max_context_tokens")
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .context("model configuration must declare a context limit")?
            as usize;
        let kv_mode = OpenClKvMode::parse(config)?;
        let epsilon = config
            .get("mslim_lod_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        ensure!(
            kv_mode == OpenClKvMode::ExactColdStore || epsilon == 0.0,
            "dense OpenCL mode cannot declare a positive LOD epsilon"
        );
        let stage_tokens = config
            .get("mslim_stage_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("mslim_stage_tokens fuera de rango")?
            .unwrap_or(64);
        let cold_directory_root = config
            .get("mslim_cold_directory")
            .or_else(|| config.get("mslim_cold_dir"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let cold_directory = if kv_mode == OpenClKvMode::ExactColdStore
            && let Some(root) = cold_directory_root
        {
            fs::create_dir_all(&root).context("create canonical ColdStore root")?;
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("ColdStore clock")?
                .as_nanos();
            let active = root.join(format!("session-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&active).context("create active canonical ColdStore namespace")?;
            Some(active)
        } else {
            None
        };
        let physical_restore = cold_directory.is_some();
        let lod = GpuRegionConfig {
            epsilon,
            stage_tokens,
            cold_directory,
        };
        let mode = match config
            .get("mslim_weight_mode")
            .and_then(Value::as_str)
            .unwrap_or("stream")
        {
            "resident" => WeightMode::Resident,
            "stream" => WeightMode::Stream,
            other => anyhow::bail!("unsupported mslim_weight_mode: {other}"),
        };
        let mut runtime = match kv_mode {
            OpenClKvMode::Dense => OpenClLlm::load(model_dir, max_tokens, mode)
                .context("load dense accelerator model adapter")?,
            OpenClKvMode::ExactColdStore => {
                OpenClLlm::load_with_gpu_region(model_dir, max_tokens, mode, lod)
                    .context("load exact canonical accelerator model adapter")?
            }
        };
        let prefill_batch = config
            .get("mslim_prefill_batch_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("mslim_prefill_batch_tokens fuera de rango")?
            .unwrap_or_else(|| max_tokens.clamp(1, 128));
        runtime.set_batch_prefill_tokens(Some(prefill_batch))?;
        if let Some(value) = config.get("mslim_prefill_weight_reuse_tokens") {
            let count = value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .context("invalid mslim_prefill_weight_reuse_tokens")?;
            runtime.set_prefill_weight_reuse_tokens(count)?;
        }

        if kv_mode == OpenClKvMode::ExactColdStore {
            let hot_tokens = config
                .get("mslim_hot_v_tokens")
                .and_then(Value::as_u64)
                .map(usize::try_from)
                .transpose()
                .context("mslim_hot_v_tokens fuera de rango")?
                .unwrap_or_else(|| max_tokens.min(512));
            runtime.set_exact_hot_v_tokens(hot_tokens)?;
            let replay_tokens = config
                .get("mslim_prompt_replay_tokens")
                .and_then(Value::as_u64)
                .map(usize::try_from)
                .transpose()
                .context("mslim_prompt_replay_tokens fuera de rango")?
                .unwrap_or(if mode == WeightMode::Resident {
                    hot_tokens
                } else {
                    0
                })
                .min(hot_tokens);
            runtime.set_verified_gpu_prompt_replay(replay_tokens)?;
            let shared_prefix = config
                .get("mslim_shared_prefix")
                .and_then(Value::as_bool)
                .unwrap_or(true)
                && replay_tokens >= 32;
            runtime.set_verified_gpu_prefix_replay(shared_prefix)?;
            runtime.set_deferred_cold_commit(
                config
                    .get("mslim_deferred_cold_commit")
                    .and_then(Value::as_bool)
                    .unwrap_or(mode == WeightMode::Resident),
            )?;
        }
        runtime.set_exact_hot_fused_attention(
            config
                .get("mslim_exact_hot_fused_attention")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        )?;
        runtime.set_decode_unroll4(
            config
                .get("mslim_decode_unroll4")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )?;
        runtime.set_decode_fused_kernels(
            config
                .get("mslim_decode_fused_kernels")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        )?;
        runtime.set_prefill_fuse_projections(
            config
                .get("mslim_prefill_fuse_projections")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        )?;
        if let Some(value) = config.get("mslim_prefill_profile") {
            let path = value
                .as_str()
                .context("mslim_prefill_profile must be a path")?;
            runtime.apply_promoted_prefill_profile(Path::new(path))?;
        }
        self.capabilities.supports_kv_restore = kv_mode == OpenClKvMode::Dense || physical_restore;
        self.kv_mode = kv_mode;
        self.model_dir = Some(model_dir.to_owned());
        self.runtime = Some(runtime);
        self.last_logits = None;
        self.cached_prompt.clear();
        Ok(())
    }

    fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        ensure!(!input.is_empty(), "accelerator prefill vacío");
        let runtime = self
            .runtime
            .as_mut()
            .context("local accelerator runtime is not loaded")?;

        if self.cached_prompt == input && runtime.position() >= input.len() {
            return Ok(json!({
                "tokens": input.len(),
                "position": runtime.position(),
                "mode": "verified-prefix-reuse",
                "reused_tokens": input.len(),
            }));
        }

        if runtime.position() != 0
            && runtime.position() == self.cached_prompt.len()
            && input.starts_with(&self.cached_prompt)
        {
            let suffix = &input[self.cached_prompt.len()..];
            let mut last_logits = None;
            for (index, &token) in suffix.iter().enumerate() {
                let step = runtime.forward_token(token, index + 1 == suffix.len())?;
                if step.logits.is_some() {
                    last_logits = step.logits;
                }
            }
            self.last_logits = last_logits;
            self.cached_prompt = input.to_vec();
            return Ok(json!({
                "tokens": input.len(),
                "position": runtime.position(),
                "mode": "verified-shared-prefix",
                "reused_tokens": input.len() - suffix.len(),
            }));
        }

        if runtime.position() != 0 {
            runtime
                .reset_session()
                .context("reset accelerator causal session before unrelated prompt")?;
        }
        let (step, prefill_seconds) = runtime
            .prefill_only(input, true, &std::sync::atomic::AtomicBool::new(false))
            .context("prefill prompt without decoding")?;
        self.last_logits = step.logits;
        self.cached_prompt = input.to_vec();
        Ok(json!({
            "tokens": input.len(),
            "position": runtime.position(),
            "mode": "batched-causal-prefill-only",
            "prefill_seconds": prefill_seconds,
        }))
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_mut()
            .context("local accelerator adapter is not loaded")?;
        let step = runtime.forward_token(token_id, true)?;
        self.last_logits = step.logits;
        // The runtime appends the token supplied to forward_token; the returned
        // argmax is only the prediction for the following position.
        self.cached_prompt.push(token_id);
        Ok(step.argmax_token)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.runtime
            .as_mut()
            .context("local accelerator adapter is not loaded")?
            .reset_session()
            .context("reset accelerator causal session")?;
        self.cached_prompt.clear();
        self.last_logits = None;
        Ok(())
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.last_logits.clone())
    }

    fn kv_integrity_report(&mut self) -> Result<Option<Value>> {
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        Ok(Some(json!({
            "schema": "mslim.logical-kv-integrity/v1",
            "authority": if self.kv_mode == OpenClKvMode::Dense {
                "dense-device-bf16-kv"
            } else {
                "canonical-bf16-coldstore"
            },
            "serialization": "k-then-v-layer-major-head-token-dim",
            "mode": self.kv_mode.label(),
            "tokens": runtime.position(),
            "layers": runtime.spec().num_hidden_layers,
            "logical_kv_sha256": runtime.canonical_kv_sha256()?,
        })))
    }

    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        if self.kv_mode == OpenClKvMode::Dense {
            let runtime = self
                .runtime
                .as_ref()
                .context("OpenCL adapter is not loaded")?;
            return serde_json::to_vec(&json!({
                "schema": "mslim.runtime.replay-snapshot/v1",
                "model_dir": self.model_dir,
                "tokens": runtime.session_token_ids(),
                "authority": "logical-token-replay",
                "kv_mode": self.kv_mode.label(),
            }))
            .context("serialize dense OpenCL logical replay snapshot");
        }
        ensure!(
            self.capabilities.supports_kv_restore,
            "physical OpenCL K/V snapshots require mslim_cold_directory"
        );
        let runtime = self
            .runtime
            .as_ref()
            .context("OpenCL adapter is not loaded")?;
        serde_json::to_vec(&json!({
            "schema": "mslim.runtime.physical-coldstore-snapshot/v1",
            "model_dir": self.model_dir,
            "cold_directory": runtime.active_cold_directory(),
            "tokens": runtime.session_token_ids(),
            "authority": "canonical-bf16-coldstore",
            "kv_mode": self.kv_mode.label(),
        }))
        .context("serialize physical OpenCL ColdStore snapshot handle")
    }

    fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        if self.kv_mode == OpenClKvMode::Dense {
            let envelope: Value = serde_json::from_slice(snapshot)
                .context("parse dense OpenCL logical replay snapshot")?;
            ensure!(
                envelope.get("schema").and_then(Value::as_str)
                    == Some("mslim.runtime.replay-snapshot/v1"),
                "unsupported dense OpenCL snapshot schema"
            );
            let expected_dir = envelope
                .get("model_dir")
                .and_then(Value::as_str)
                .context("dense OpenCL snapshot missing model identity")?;
            ensure!(
                self.model_dir.as_deref() == Some(Path::new(expected_dir)),
                "dense OpenCL snapshot model identity mismatch"
            );
            let tokens = envelope
                .get("tokens")
                .and_then(Value::as_array)
                .context("dense OpenCL snapshot missing token history")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|token| u32::try_from(token).ok())
                        .context("invalid token in dense OpenCL snapshot")
                })
                .collect::<Result<Vec<_>>>()?;
            let runtime = self
                .runtime
                .as_mut()
                .context("OpenCL adapter is not loaded")?;
            runtime.reset_session()?;
            for &token in &tokens {
                runtime.forward_token(token, false)?;
            }
            self.last_logits = None;
            self.cached_prompt = tokens;
            return Ok(());
        }
        ensure!(
            self.capabilities.supports_kv_restore,
            "physical OpenCL K/V restore requires mslim_cold_directory"
        );
        let envelope: Value = serde_json::from_slice(snapshot)
            .context("parse physical OpenCL ColdStore snapshot handle")?;
        ensure!(
            envelope.get("schema").and_then(Value::as_str)
                == Some("mslim.runtime.physical-coldstore-snapshot/v1"),
            "unsupported physical OpenCL snapshot schema"
        );
        ensure!(
            envelope.get("authority").and_then(Value::as_str) == Some("canonical-bf16-coldstore"),
            "snapshot does not reference canonical BF16 ColdStore authority"
        );
        let expected_dir = envelope
            .get("model_dir")
            .and_then(Value::as_str)
            .context("physical OpenCL snapshot missing model identity")?;
        let current_dir = self
            .model_dir
            .as_deref()
            .context("OpenCL adapter is not loaded")?;
        ensure!(
            Path::new(expected_dir) == current_dir,
            "physical OpenCL snapshot model identity mismatch"
        );
        let tokens = envelope
            .get("tokens")
            .and_then(Value::as_array)
            .context("physical OpenCL snapshot missing token history")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .context("invalid token in physical OpenCL snapshot")
            })
            .collect::<Result<Vec<_>>>()?;
        let cold_directory = envelope
            .get("cold_directory")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .context("physical OpenCL snapshot missing ColdStore namespace")?;
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        runtime.reset_session_at(Some(cold_directory))?;
        if !tokens.is_empty() {
            let receipt = runtime.restore_session(&tokens)?;
            ensure!(
                receipt.tokens == tokens.len(),
                "physical OpenCL restore returned an incomplete prefix"
            );
        }
        self.last_logits = None;
        self.cached_prompt = tokens;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_adapter_does_not_claim_physical_restore() {
        let adapter = OpenClModelAdapter::default();
        assert!(!adapter.capabilities().supports_kv_restore);
        assert!(!adapter.capabilities().supports_streaming);
        assert!(adapter.snapshot_kv().is_err());
    }

    #[test]
    fn kv_mode_is_explicit_and_fail_closed() {
        assert_eq!(
            OpenClKvMode::parse(&json!({"mslim_kv_mode": "dense"})).unwrap(),
            OpenClKvMode::Dense
        );
        assert_eq!(
            OpenClKvMode::parse(&json!({"mslim_kv_mode": "exact-coldstore"})).unwrap(),
            OpenClKvMode::ExactColdStore
        );
        assert!(OpenClKvMode::parse(&json!({"mslim_kv_mode": "automatic"})).is_err());
    }
}
