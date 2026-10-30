//! Backend-neutral adapter conformance.
//!
//! This is the product contract. Concrete adapters only provide the runtime;
//! this module owns the checks that every runtime must satisfy.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use kvslim_adapter_api::ModelAdapter;
use serde::Serialize;
use serde_json::Value;

use crate::engine::{EngineState, KvslimEngine};

#[derive(Debug, Clone, Serialize)]
pub struct AdapterConformanceReport {
    pub adapter_id: String,
    pub backend: String,
    pub inspect: bool,
    pub load: bool,
    pub prefill: bool,
    pub decode: bool,
    pub kv_snapshot: bool,
    pub kv_restore: bool,
    pub status: &'static str,
}

/// Runs the same lifecycle and KV contract for any concrete adapter.
///
/// No model family, vendor, accelerator, precision, tokenizer, or backend
/// name is selected here. Applicability belongs to adapter inspection and
/// capability probing; correctness belongs to this contract.
pub fn run_adapter_conformance(
    adapter: Box<dyn ModelAdapter>,
    model_dir: &Path,
    config: &Value,
    probe_tokens: &[u32],
) -> Result<AdapterConformanceReport> {
    ensure!(
        !probe_tokens.is_empty(),
        "conformance probe cannot be empty"
    );
    let mut engine = KvslimEngine::new(adapter);
    let capabilities = engine.capabilities().clone();
    ensure!(
        capabilities.supports_prefill,
        "adapter does not support prefill"
    );
    ensure!(
        capabilities.supports_decode,
        "adapter does not support decode"
    );
    ensure!(
        capabilities.supports_kv_restore,
        "adapter does not support KV restore"
    );
    ensure!(
        engine.state() == EngineState::Created,
        "invalid initial engine state"
    );

    engine
        .inspect(model_dir, config)
        .context("conformance inspect")?;
    ensure!(
        engine.state() == EngineState::Inspected,
        "inspect did not advance state"
    );
    engine.load(model_dir, config).context("conformance load")?;
    ensure!(
        engine.state() == EngineState::Loaded,
        "load did not advance state"
    );
    engine
        .prefill(probe_tokens)
        .context("conformance prefill")?;
    engine
        .decode(probe_tokens[0])
        .context("conformance decode")?;
    let snapshot = engine.snapshot_kv().context("conformance snapshot")?;
    ensure!(
        !snapshot.is_empty(),
        "adapter returned an empty KV snapshot"
    );
    engine
        .restore_kv(&snapshot)
        .context("conformance restore")?;
    ensure!(
        engine
            .snapshot_kv()
            .context("conformance verify snapshot")?
            == snapshot,
        "KV snapshot changed after restore"
    );

    Ok(AdapterConformanceReport {
        adapter_id: capabilities.adapter_id,
        backend: capabilities.backend,
        inspect: true,
        load: true,
        prefill: true,
        decode: true,
        kv_snapshot: true,
        kv_restore: true,
        status: "passed",
    })
}
