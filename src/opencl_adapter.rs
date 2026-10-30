//! Concrete accelerator implementation behind the universal adapter boundary.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use kvslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use serde_json::{Value, json};

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
                supports_kv_restore: true,
                supports_streaming: true,
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
        let runtime = OpenClLlm::load(model_dir, max_tokens, WeightMode::Resident)
            .context("load OpenCL model adapter")?;
        self.model_dir = Some(model_dir.to_owned());
        self.runtime = Some(runtime);
        self.last_logits = None;
        Ok(())
    }

    fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        for (index, &token) in input.iter().enumerate() {
            let step = runtime.forward_token(token, index + 1 == input.len())?;
            if let Some(step) = step.logits {
                self.last_logits = Some(step);
            }
        }
        Ok(json!({ "tokens": input.len(), "position": runtime.position() }))
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        let step = runtime.forward_token(token_id, true)?;
        self.last_logits = step.logits;
        Ok(step.argmax_token)
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.last_logits.clone())
    }

    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        let runtime = self
            .runtime
            .as_ref()
            .context("OpenCL adapter is not loaded")?;
        serde_json::to_vec(&json!({
            "schema": "kvslim.runtime.replay-snapshot/v1",
            "model_dir": self.model_dir,
            "tokens": runtime.session_token_ids(),
        }))
        .context("serialize OpenCL K/V snapshot")
    }

    fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        let envelope: Value =
            serde_json::from_slice(snapshot).context("parse OpenCL K/V snapshot")?;
        ensure!(
            envelope.get("schema").and_then(Value::as_str)
                == Some("kvslim.runtime.replay-snapshot/v1"),
            "unsupported runtime snapshot schema"
        );
        let expected_dir = envelope
            .get("model_dir")
            .and_then(Value::as_str)
            .context("OpenCL snapshot missing model identity")?;
        let current_dir = self
            .model_dir
            .as_deref()
            .context("OpenCL adapter is not loaded")?;
        ensure!(
            Path::new(expected_dir) == current_dir,
            "snapshot model identity mismatch"
        );
        let token_values = envelope
            .get("tokens")
            .and_then(Value::as_array)
            .context("OpenCL snapshot missing token history")?;
        let tokens: Vec<u32> = token_values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .context("invalid token in OpenCL snapshot")
            })
            .collect::<Result<_>>()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("OpenCL adapter is not loaded")?;
        runtime.reset_session()?;
        for token in tokens {
            runtime.forward_token(token, false)?;
        }
        Ok(())
    }
}
