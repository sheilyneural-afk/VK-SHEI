//! Backend-neutral adapter conformance.
//!
//! This is the product contract. Concrete adapters only provide the runtime;
//! this module owns the checks that every runtime must satisfy.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use mslim_adapter_api::ModelAdapter;
use serde::Serialize;
use serde_json::Value;

use crate::engine::{EngineState, MSlimEngine};

#[derive(Debug, Clone, Serialize)]
pub struct AdapterConformanceReport {
    pub adapter_id: String,
    pub backend: String,
    pub inspect: bool,
    pub load: bool,
    pub prefill: bool,
    pub decode: bool,
    /// Adapter lifecycle returned a non-empty opaque replay snapshot.
    /// This does not assert that physical K/V tensor bytes were persisted.
    pub kv_snapshot: bool,
    pub kv_restore: bool,
    pub snapshot_contract: &'static str,
    pub physical_kv_persistence_verified: bool,
    pub logits_captured: bool,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LogitParityReport {
    pub compared: usize,
    pub exact: bool,
    pub within_tolerance: bool,
    pub max_abs_difference: f32,
}

/// Compares logits produced by two real adapter executions. Missing vectors
/// are rejected by the caller; an absent capture must never become a passing
/// parity result.
pub fn compare_logits(
    reference: &[f32],
    candidate: &[f32],
    tolerance: f32,
) -> Result<LogitParityReport> {
    ensure!(!reference.is_empty(), "reference logits are empty");
    ensure!(!candidate.is_empty(), "candidate logits are empty");
    ensure!(
        reference.len() == candidate.len(),
        "logit vocabularies differ"
    );
    ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "invalid logit tolerance"
    );
    let max_abs_difference = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    Ok(LogitParityReport {
        compared: reference.len(),
        exact: reference == candidate,
        within_tolerance: max_abs_difference <= tolerance,
        max_abs_difference,
    })
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
    let mut engine = MSlimEngine::new(adapter);
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
    // ModelAdapter::snapshot_kv is an opaque logical replay contract. The
    // current API exposes no physical K/V tensor payload or hydration receipt,
    // so conformance must not turn round-trip equality into that stronger claim.
    let snapshot = engine
        .snapshot_kv()
        .context("conformance logical replay snapshot")?;
    let logits_captured = engine
        .last_logits()
        .context("conformance logits capture")?
        .is_some_and(|logits| !logits.is_empty());
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
        snapshot_contract: "opaque-logical-replay-not-physical-kv-proof",
        physical_kv_persistence_verified: false,
        logits_captured,
        status: "passed",
    })
}

#[cfg(test)]
mod tests {
    use super::compare_logits;

    #[test]
    fn logit_parity_reports_exact_and_tolerance_separately() {
        let report = compare_logits(&[1.0, 2.0], &[1.0, 2.0001], 0.001).unwrap();
        assert!(!report.exact);
        assert!(report.within_tolerance);
        assert!(report.max_abs_difference > 0.0);
    }
}
