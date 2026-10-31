//! Session coordinator joining semantic generations with the backend-neutral engine.
//!
//! SessionStore owns semantic durability; MslimEngine owns adapter-specific KV.
use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::{
    engine::{EngineState, MslimEngine},
    session_store::{SessionGeneration, SessionStore},
};

pub struct SessionCoordinator {
    engine: MslimEngine,
    store: SessionStore,
}

impl SessionCoordinator {
    pub fn new(engine: MslimEngine, store: SessionStore) -> Self {
        Self { engine, store }
    }

    pub fn engine(&self) -> &MslimEngine {
        &self.engine
    }
    pub fn engine_mut(&mut self) -> &mut MslimEngine {
        &mut self.engine
    }

    pub fn snapshot_semantic(
        &self,
        handle: &str,
        generation: u64,
        state: Value,
    ) -> Result<SessionGeneration> {
        ensure!(
            self.engine.state() == EngineState::Loaded,
            "engine is not loaded"
        );
        self.store
            .prepare_and_commit(handle, generation, state)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn resume_semantic(&self, handle: &str) -> Result<Option<SessionGeneration>> {
        self.store
            .latest(handle)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn snapshot_physical_kv(&self) -> Result<Vec<u8>> {
        ensure!(
            self.engine.state() == EngineState::Loaded,
            "engine is not loaded"
        );
        self.engine.snapshot_kv().context("snapshot physical KV")
    }

    pub fn restore_physical_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        ensure!(
            self.engine.state() == EngineState::Loaded,
            "engine is not loaded"
        );
        self.engine
            .restore_kv(snapshot)
            .context("restore physical KV")
    }

    pub fn commit_successor_then_release(
        &self,
        handle: &str,
        generation: u64,
        state: Value,
    ) -> Result<SessionGeneration> {
        // SessionStore publishes CURRENT only after the complete successor is durable.
        self.snapshot_semantic(handle, generation, state)
    }
}
