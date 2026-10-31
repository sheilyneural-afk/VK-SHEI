//! LibTorch implementation of the universal adapter boundary.
//!
//! The current dense runtime exposes stateless generation. This adapter keeps
//! the session token history and reconstructs its K/V state by deterministic
//! replay, so restore is correct even before the runtime exposes a mutable
//! incremental cache API.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use mslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use serde_json::{Value, json};

use crate::model_bundle::ModelBundle;
use crate::torch_runtime::TorchModelRuntime;

pub struct TorchModelAdapter {
    capabilities: AdapterCapabilities,
    model_dir: Option<PathBuf>,
    runtime: Option<TorchModelRuntime>,
    session_tokens: Vec<i64>,
    last_logits: Option<Vec<f32>>,
}

impl Default for TorchModelAdapter {
    fn default() -> Self {
        Self {
            capabilities: AdapterCapabilities {
                adapter_id: "cpu-runtime".into(),
                backend: "cpu".into(),
                supports_prefill: true,
                supports_decode: true,
                supports_kv_restore: true,
                supports_streaming: false,
            },
            model_dir: None,
            runtime: None,
            session_tokens: Vec::new(),
            last_logits: None,
        }
    }
}

impl ModelAdapter for TorchModelAdapter {
    fn capabilities(&self) -> &AdapterCapabilities {
        &self.capabilities
    }

    fn inspect(&self, model_dir: &Path, _config: &Value) -> Result<()> {
        let bundle = ModelBundle::load(model_dir)?;
        bundle.spec.ensure_cpu_runtime_compatible()?;
        bundle.validate_decoder_layout()
    }

    fn load(&mut self, model_dir: &Path, config: &Value) -> Result<()> {
        self.inspect(model_dir, config)?;
        self.runtime =
            Some(TorchModelRuntime::load(model_dir).context("load LibTorch model adapter")?);
        self.model_dir = Some(model_dir.to_owned());
        self.session_tokens.clear();
        self.last_logits = None;
        Ok(())
    }

    fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        ensure!(!input.is_empty(), "Torch adapter prompt is empty");
        let runtime = self
            .runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        self.session_tokens = input.iter().map(|&token| i64::from(token)).collect();
        let generation = runtime.generate_token_ids_dense(&self.session_tokens, 1, 512)?;
        self.last_logits = generation.step_logits.first().cloned();
        Ok(json!({ "tokens": input.len() }))
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        self.session_tokens.push(i64::from(token_id));
        let generation = runtime.generate_token_ids_dense(&self.session_tokens, 1, 512)?;
        self.last_logits = generation.step_logits.first().cloned();
        generation
            .token_ids
            .first()
            .copied()
            .context("Torch adapter produced no token")
    }

    fn reset_session(&mut self) -> Result<()> {
        self.session_tokens.clear();
        self.last_logits = None;
        Ok(())
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.last_logits.clone())
    }

    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        self.runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        serde_json::to_vec(&json!({
            "schema": "mslim.runtime.replay-snapshot/v1",
            "model_dir": self.model_dir,
            "tokens": self.session_tokens,
        }))
        .context("serialize Torch K/V snapshot")
    }

    fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        let envelope: Value = serde_json::from_slice(snapshot).context("parse Torch snapshot")?;
        ensure!(
            envelope.get("schema").and_then(Value::as_str)
                == Some("mslim.runtime.replay-snapshot/v1"),
            "unsupported runtime snapshot schema"
        );
        let expected_dir = envelope
            .get("model_dir")
            .and_then(Value::as_str)
            .context("Torch snapshot missing model identity")?;
        let current_dir = self
            .model_dir
            .as_deref()
            .context("Torch adapter is not loaded")?;
        ensure!(
            Path::new(expected_dir) == current_dir,
            "snapshot model identity mismatch"
        );
        let values = envelope
            .get("tokens")
            .and_then(Value::as_array)
            .context("Torch snapshot missing token history")?;
        self.session_tokens = values
            .iter()
            .map(|value| value.as_i64().context("invalid token in Torch snapshot"))
            .collect::<Result<_>>()?;
        Ok(())
    }
}
