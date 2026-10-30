//! Concrete accelerator implementation behind the universal adapter boundary.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use kvslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use serde_json::{Value, json};

use crate::gpu_canonical_lod::CanonicalGpuLodConfig;
use crate::model_bundle::ModelBundle;
use crate::opencl_llm::{OpenClLlm, WeightMode};

pub struct OpenClModelAdapter {
    capabilities: AdapterCapabilities,
    model_dir: Option<PathBuf>,
    runtime: Option<OpenClLlm>,
    last_logits: Option<Vec<f32>>,
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
        }
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
        let epsilon = config
            .get("kvslim_lod_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let stage_tokens = config
            .get("kvslim_stage_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("kvslim_stage_tokens fuera de rango")?
            .unwrap_or(64);
        let cold_directory = config
            .get("kvslim_cold_directory")
            .or_else(|| config.get("kvslim_cold_dir"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let physical_restore = cold_directory.is_some();
        let lod = CanonicalGpuLodConfig {
            epsilon,
            stage_tokens,
            cold_directory,
        };
        let mut runtime =
            OpenClLlm::load_with_canonical_lod(model_dir, max_tokens, WeightMode::Resident, lod)
                .context("load exact canonical accelerator model adapter")?;
        let prefill_batch = config
            .get("kvslim_prefill_batch_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("kvslim_prefill_batch_tokens fuera de rango")?
            .unwrap_or_else(|| max_tokens.min(128).max(1));
        runtime.set_batch_prefill_tokens(Some(prefill_batch))?;

        let hot_tokens = config
            .get("kvslim_hot_v_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("kvslim_hot_v_tokens fuera de rango")?
            .unwrap_or_else(|| max_tokens.min(512));
        runtime.set_exact_hot_v_tokens(hot_tokens)?;
        let replay_tokens = config
            .get("kvslim_prompt_replay_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("kvslim_prompt_replay_tokens fuera de rango")?
            .unwrap_or(hot_tokens)
            .min(hot_tokens);
        runtime.set_verified_gpu_prompt_replay(replay_tokens)?;
        let shared_prefix = config
            .get("kvslim_shared_prefix")
            .and_then(Value::as_bool)
            .unwrap_or(true)
            && replay_tokens >= 32;
        runtime.set_verified_gpu_prefix_replay(shared_prefix)?;
        runtime.set_deferred_cold_commit(
            config
                .get("kvslim_deferred_cold_commit")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        )?;
        runtime.set_exact_hot_fused_attention(
            config
                .get("kvslim_exact_hot_fused_attention")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        )?;
        runtime.set_decode_unroll4(
            config
                .get("kvslim_decode_unroll4")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )?;
        runtime.set_decode_fused_kernels(
            config
                .get("kvslim_decode_fused_kernels")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )?;
        self.capabilities.supports_kv_restore = physical_restore;
        self.model_dir = Some(model_dir.to_owned());
        self.runtime = Some(runtime);
        self.last_logits = None;
        Ok(())
    }

    fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        let runtime = self
            .runtime
            .as_mut()
            .context("local accelerator adapter is not loaded")?;
        ensure!(!input.is_empty(), "accelerator prefill vacío");
        if runtime.position() == 0 {
            let generation = runtime.generate(input, 1, true)?;
            self.last_logits = generation.steps.last().and_then(|step| step.logits.clone());
            Ok(json!({
                "tokens": input.len(),
                "position": runtime.position(),
                "mode": "batched-causal",
                "batch_tokens": generation.gpu_prefill_batch_tokens,
                "prefill_seconds": generation.prefill_seconds,
            }))
        } else {
            for (index, &token) in input.iter().enumerate() {
                let step = runtime.forward_token(token, index + 1 == input.len())?;
                if let Some(step) = step.logits {
                    self.last_logits = Some(step);
                }
            }
            Ok(json!({
                "tokens": input.len(),
                "position": runtime.position(),
                "mode": "incremental-dense",
            }))
        }
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_mut()
            .context("local accelerator adapter is not loaded")?;
        let step = runtime.forward_token(token_id, true)?;
        self.last_logits = step.logits;
        Ok(step.argmax_token)
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.last_logits.clone())
    }

    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        ensure!(
            self.capabilities.supports_kv_restore,
            "physical OpenCL K/V snapshots require kvslim_cold_directory"
        );
        let runtime = self
            .runtime
            .as_ref()
            .context("OpenCL adapter is not loaded")?;
        serde_json::to_vec(&json!({
            "schema": "kvslim.runtime.physical-coldstore-snapshot/v1",
            "model_dir": self.model_dir,
            "tokens": runtime.session_token_ids(),
            "authority": "canonical-bf16-coldstore",
        }))
        .context("serialize physical OpenCL ColdStore snapshot handle")
    }

    fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        ensure!(
            self.capabilities.supports_kv_restore,
            "physical OpenCL K/V restore requires kvslim_cold_directory"
        );
        let envelope: Value = serde_json::from_slice(snapshot)
            .context("parse physical OpenCL ColdStore snapshot handle")?;
        ensure!(
            envelope.get("schema").and_then(Value::as_str)
                == Some("kvslim.runtime.physical-coldstore-snapshot/v1"),
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
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        if runtime.position() != 0 {
            runtime.reset_session()?;
        }
        if !tokens.is_empty() {
            let receipt = runtime.restore_canonical_session(&tokens)?;
            ensure!(
                receipt.tokens == tokens.len(),
                "physical OpenCL restore returned an incomplete prefix"
            );
        }
        self.last_logits = None;
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
}
