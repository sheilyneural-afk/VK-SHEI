//! Backend-neutral adapter conformance.
//!
//! This is the product contract. Concrete adapters only provide the runtime;
//! this module owns the checks that every runtime must satisfy.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use mslim_adapter_api::ModelAdapter;
use serde::Serialize;
use serde_json::Value;

use crate::engine::{EngineState, MslimEngine};

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
    pub snapshot_contract: String,
    pub physical_kv_persistence_verified: bool,
    pub logits_captured: bool,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LogitParityReport {
    pub compared: usize,
    /// Equality of every finite FP32 bit pattern, including the sign of zero.
    pub exact: bool,
    pub within_tolerance: bool,
    pub max_abs_difference: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LogitDistributionReport {
    pub compared: usize,
    /// Equality of every finite FP32 bit pattern, including the sign of zero.
    pub exact: bool,
    pub max_abs_difference: f32,
    pub mean_abs_difference: f64,
    pub p95_abs_difference: f32,
    pub kl_reference_to_candidate: f64,
    pub reference_top1: usize,
    pub candidate_top1: usize,
    pub top1_equal: bool,
    pub reference_top1_top2_margin: f32,
    pub candidate_top1_top2_margin: f32,
}

fn top_two(values: &[f32]) -> (usize, f32, f32) {
    let mut first = (usize::MAX, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > first.1 {
            second = first.1;
            first = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (first.0, first.1, second)
}

/// Distribution-aware diagnostic for backend and same-backend comparisons.
/// KL is computed with a stable softmax in f64 and is not used as an implicit
/// activation threshold; policy must compare it with a predeclared limit.
pub fn compare_logit_distributions(
    reference: &[f32],
    candidate: &[f32],
) -> Result<LogitDistributionReport> {
    ensure!(!reference.is_empty(), "reference logits are empty");
    ensure!(
        reference.len() == candidate.len(),
        "logit vocabularies differ"
    );
    ensure!(
        reference
            .iter()
            .chain(candidate)
            .all(|value| value.is_finite()),
        "logit comparison requires finite values"
    );
    let mut differences = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (left - right).abs())
        .collect::<Vec<_>>();
    let max_abs_difference = differences.iter().copied().fold(0.0_f32, f32::max);
    let mean_abs_difference = differences
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>()
        / differences.len() as f64;
    differences.sort_by(f32::total_cmp);
    let p95_index = ((differences.len() - 1) * 95) / 100;
    let p95_abs_difference = differences[p95_index];

    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reference_exp = reference
        .iter()
        .map(|&value| f64::from(value - reference_max).exp())
        .collect::<Vec<_>>();
    let candidate_exp = candidate
        .iter()
        .map(|&value| f64::from(value - candidate_max).exp())
        .collect::<Vec<_>>();
    let reference_sum = reference_exp.iter().sum::<f64>();
    let candidate_sum = candidate_exp.iter().sum::<f64>();
    ensure!(
        reference_sum.is_finite() && candidate_sum.is_finite(),
        "logit softmax normalization is not finite"
    );
    let reference_log_sum = reference_sum.ln() + f64::from(reference_max);
    let candidate_log_sum = candidate_sum.ln() + f64::from(candidate_max);
    let kl_reference_to_candidate = reference_exp
        .iter()
        .zip(reference.iter().zip(candidate))
        .map(|(&left, (&reference_logit, &candidate_logit))| {
            let p = left / reference_sum;
            p * ((f64::from(reference_logit) - reference_log_sum)
                - (f64::from(candidate_logit) - candidate_log_sum))
        })
        .sum::<f64>()
        .max(0.0);
    let (reference_top1, reference_first, reference_second) = top_two(reference);
    let (candidate_top1, candidate_first, candidate_second) = top_two(candidate);
    Ok(LogitDistributionReport {
        compared: reference.len(),
        exact: reference
            .iter()
            .zip(candidate)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        max_abs_difference,
        mean_abs_difference,
        p95_abs_difference,
        kl_reference_to_candidate,
        reference_top1,
        candidate_top1,
        top1_equal: reference_top1 == candidate_top1,
        reference_top1_top2_margin: reference_first - reference_second,
        candidate_top1_top2_margin: candidate_first - candidate_second,
    })
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
    ensure!(
        reference
            .iter()
            .chain(candidate)
            .all(|value| value.is_finite()),
        "logit comparison requires finite values"
    );
    let max_abs_difference = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    Ok(LogitParityReport {
        compared: reference.len(),
        exact: reference
            .iter()
            .zip(candidate)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
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
    let mut engine = MslimEngine::new(adapter);
    let declared_capabilities = engine.capabilities().clone();
    ensure!(
        declared_capabilities.supports_prefill,
        "adapter does not support prefill"
    );
    ensure!(
        declared_capabilities.supports_decode,
        "adapter does not support decode"
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
    let capabilities = engine.capabilities().clone();
    ensure!(
        capabilities.supports_kv_restore,
        "loaded adapter configuration does not support KV restore"
    );
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
    let snapshot_schema = serde_json::from_slice::<Value>(&snapshot)
        .ok()
        .and_then(|value| {
            value
                .get("schema")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let physical_kv_persistence_verified =
        snapshot_schema.as_deref() == Some("mslim.runtime.physical-coldstore-snapshot/v1");
    let snapshot_contract = snapshot_schema.unwrap_or_else(|| "opaque-non-json-snapshot".into());
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
        snapshot_contract,
        physical_kv_persistence_verified,
        logits_captured,
        status: "passed",
    })
}

#[cfg(test)]
mod tests {
    use super::{compare_logit_distributions, compare_logits};

    #[test]
    fn logit_parity_reports_exact_and_tolerance_separately() {
        let report = compare_logits(&[1.0, 2.0], &[1.0, 2.0001], 0.001).unwrap();
        assert!(!report.exact);
        assert!(report.within_tolerance);
        assert!(report.max_abs_difference > 0.0);
    }

    #[test]
    fn distribution_report_captures_tail_and_top_token() {
        let report = compare_logit_distributions(&[0.0, 3.0, 1.0], &[0.1, 2.9, 1.0]).unwrap();
        assert_eq!(report.reference_top1, 1);
        assert!(report.top1_equal);
        assert!(report.max_abs_difference > 0.0);
        assert!(report.p95_abs_difference > 0.0);
        assert!(report.kl_reference_to_candidate >= 0.0);
    }

    #[test]
    fn signed_zero_is_numeric_but_not_bitwise_parity() {
        let report = compare_logits(&[0.0, 1.0], &[-0.0, 1.0], 0.0).unwrap();
        assert!(!report.exact);
        assert!(report.within_tolerance);
        assert_eq!(report.max_abs_difference, 0.0);
        assert!(
            !compare_logit_distributions(&[0.0, 1.0], &[-0.0, 1.0])
                .unwrap()
                .exact
        );
    }

    #[test]
    fn equal_finite_bit_patterns_pass_both_contracts() {
        let values = [-0.0, f32::from_bits(1), -1.25, 2.0];
        assert!(compare_logits(&values, &values, 0.0).unwrap().exact);
        assert!(compare_logit_distributions(&values, &values).unwrap().exact);
    }

    #[test]
    fn nonfinite_or_missing_logits_never_pass() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(compare_logits(&[invalid], &[invalid], 0.0).is_err());
            assert!(compare_logits(&[0.0], &[invalid], 1.0).is_err());
            assert!(compare_logit_distributions(&[invalid], &[invalid]).is_err());
        }
        assert!(compare_logits(&[], &[], 0.0).is_err());
        assert!(compare_logits(&[1.0], &[1.0, 2.0], 0.0).is_err());
    }
}
