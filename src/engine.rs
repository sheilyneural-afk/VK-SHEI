//! Public M-SLIM engine boundary.
//!
//! This module owns the model-adapter lifecycle used by the local engine. It
//! deliberately has no HTTP, provider, model-family, or accelerator-specific
//! policy. Concrete runtimes are selected by `adapter_catalog` and are used
//! only through the backend-neutral `ModelAdapter` contract.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use mslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    Created,
    Inspected,
    Loaded,
}

/// The M-SLIM execution engine for one loaded model session.
pub struct MslimEngine {
    adapter: Box<dyn ModelAdapter>,
    state: EngineState,
}

impl MslimEngine {
    pub fn new(adapter: Box<dyn ModelAdapter>) -> Self {
        Self {
            adapter,
            state: EngineState::Created,
        }
    }

    pub fn state(&self) -> EngineState {
        self.state
    }

    pub fn capabilities(&self) -> &AdapterCapabilities {
        self.adapter.capabilities()
    }

    pub fn inspect(&mut self, model_dir: &Path, config: &Value) -> Result<()> {
        self.adapter
            .inspect(model_dir, config)
            .context("M-SLIM engine model inspection failed")?;
        self.state = EngineState::Inspected;
        Ok(())
    }

    pub fn load(&mut self, model_dir: &Path, config: &Value) -> Result<()> {
        ensure!(
            self.state == EngineState::Inspected,
            "M-SLIM engine must inspect the model before loading it"
        );
        self.adapter
            .load(model_dir, config)
            .context("M-SLIM engine model load failed")?;
        self.state = EngineState::Loaded;
        Ok(())
    }

    pub fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter.prefill(input).context("M-SLIM prefill failed")
    }

    pub fn decode(&mut self, token_id: u32) -> Result<u32> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .decode(token_id)
            .context("M-SLIM decode failed")
    }

    pub fn snapshot_kv(&mut self) -> Result<Vec<u8>> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .snapshot_kv()
            .context("M-SLIM KV snapshot failed")
    }

    pub fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .last_logits()
            .context("M-SLIM logits capture failed")
    }

    pub fn kv_integrity_report(&mut self) -> Result<Option<Value>> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .kv_integrity_report()
            .context("M-SLIM KV integrity report failed")
    }

    pub fn reset_session(&mut self) -> Result<()> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .reset_session()
            .context("M-SLIM session reset failed")
    }

    pub fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        ensure!(
            self.state == EngineState::Loaded,
            "M-SLIM engine is not loaded"
        );
        self.adapter
            .restore_kv(snapshot)
            .context("M-SLIM KV restore failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::ensure;
    use serde_json::json;

    struct ContractAdapter {
        capabilities: AdapterCapabilities,
        loaded: bool,
        tokens: Vec<u32>,
    }

    impl ContractAdapter {
        fn new() -> Self {
            Self {
                capabilities: AdapterCapabilities {
                    adapter_id: "contract-test".into(),
                    backend: "test-runtime".into(),
                    supports_prefill: true,
                    supports_decode: true,
                    supports_kv_restore: true,
                    supports_streaming: false,
                },
                loaded: false,
                tokens: Vec::new(),
            }
        }
    }

    impl ModelAdapter for ContractAdapter {
        fn capabilities(&self) -> &AdapterCapabilities {
            &self.capabilities
        }

        fn inspect(&self, _model_dir: &Path, config: &Value) -> Result<()> {
            ensure!(config.get("model").is_some(), "model metadata is required");
            Ok(())
        }

        fn load(&mut self, _model_dir: &Path, _config: &Value) -> Result<()> {
            self.loaded = true;
            Ok(())
        }

        fn prefill(&mut self, input: &[u32]) -> Result<Value> {
            ensure!(self.loaded, "adapter is not loaded");
            self.tokens.extend_from_slice(input);
            Ok(json!({"tokens": input.len()}))
        }

        fn decode(&mut self, token_id: u32) -> Result<u32> {
            ensure!(self.loaded, "adapter is not loaded");
            self.tokens.push(token_id);
            Ok(token_id)
        }

        fn last_logits(&self) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0]))
        }

        fn reset_session(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }

        fn snapshot_kv(&mut self) -> Result<Vec<u8>> {
            Ok(serde_json::to_vec(&self.tokens)?)
        }

        fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
            self.tokens = serde_json::from_slice(snapshot)?;
            Ok(())
        }
    }

    #[test]
    fn backend_neutral_engine_enforces_lifecycle_and_roundtrips_kv() {
        let mut engine = MslimEngine::new(Box::new(ContractAdapter::new()));
        assert_eq!(engine.state(), EngineState::Created);
        assert!(engine.prefill(&[1]).is_err());

        engine
            .inspect(Path::new("model"), &json!({"model": "runtime-neutral"}))
            .unwrap();
        assert_eq!(engine.state(), EngineState::Inspected);
        engine
            .load(Path::new("model"), &json!({"model": "runtime-neutral"}))
            .unwrap();
        assert_eq!(engine.state(), EngineState::Loaded);
        engine.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(engine.decode(4).unwrap(), 4);

        let snapshot = engine.snapshot_kv().unwrap();
        engine.decode(5).unwrap();
        engine.restore_kv(&snapshot).unwrap();
        let restored = engine.snapshot_kv().unwrap();
        assert_eq!(snapshot, restored);
    }

    #[test]
    fn backend_neutral_engine_exposes_adapter_capabilities_without_backend_policy() {
        let engine = MslimEngine::new(Box::new(ContractAdapter::new()));
        assert_eq!(engine.capabilities().adapter_id, "contract-test");
        assert_eq!(engine.capabilities().backend, "test-runtime");
    }
}
