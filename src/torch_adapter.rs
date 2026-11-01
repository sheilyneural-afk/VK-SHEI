//! LibTorch implementation of the backend-neutral adapter contract.
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
        let session_tokens = input
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let generation = runtime.generate_token_ids_dense(&session_tokens, 1, 512)?;
        let logits = generation
            .step_logits
            .into_iter()
            .next()
            .context("Torch adapter prefill produced no logits")?;
        self.session_tokens = session_tokens;
        self.last_logits = Some(logits);
        Ok(json!({ "tokens": input.len() }))
    }

    fn prefill_token_only(&mut self, input: &[u32]) -> Result<(Value, u32)> {
        ensure!(!input.is_empty(), "Torch adapter prompt is empty");
        let runtime = self
            .runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        let session_tokens = input
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let next = runtime.next_token_id_dense(&session_tokens, 512)?;
        self.session_tokens = session_tokens;
        self.last_logits = None;
        Ok((
            json!({
                "tokens": input.len(),
                "full_vocabulary_logits_captured": false,
            }),
            next,
        ))
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        self.session_tokens.push(i64::from(token_id));
        let result = runtime
            .generate_token_ids_dense(&self.session_tokens, 1, 512)
            .and_then(|generation| {
                let logits = generation
                    .step_logits
                    .into_iter()
                    .next()
                    .context("Torch adapter decode produced no logits")?;
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .and_then(|(index, _)| u32::try_from(index).ok())
                    .context("Torch adapter decode logits are empty")?;
                Ok((logits, next))
            });
        match result {
            Ok((logits, next)) => {
                self.last_logits = Some(logits);
                Ok(next)
            }
            Err(error) => {
                self.session_tokens.pop();
                Err(error)
            }
        }
    }

    fn decode_token_only(&mut self, token_id: u32) -> Result<u32> {
        let runtime = self
            .runtime
            .as_ref()
            .context("Torch adapter is not loaded")?;
        self.session_tokens.push(i64::from(token_id));
        match runtime.next_token_id_dense(&self.session_tokens, 512) {
            Ok(next) => {
                self.last_logits = None;
                Ok(next)
            }
            Err(error) => {
                self.session_tokens.pop();
                Err(error)
            }
        }
    }

    fn reset_session(&mut self) -> Result<()> {
        self.session_tokens.clear();
        self.last_logits = None;
        Ok(())
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.last_logits.clone())
    }

    fn snapshot_kv(&mut self) -> Result<Vec<u8>> {
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
