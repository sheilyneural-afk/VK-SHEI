//! Bounded, advisory-only prefill cost learning for the existing calibrator.
//!
//! Three-parameter recursive least squares (RLS), with a fixed ridge prior.
//! Learns wall time, not correctness, memory capacity or permission to activate.
//! State is reconstructed from measured calibration reports; this module owns
//! no files, processes, sessions, tensors or profile activation authority.

use anyhow::{Result, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const DIM: usize = 3;
const MAX_PROFILES: usize = 32;
const MAX_OBSERVATIONS: usize = 4096;
const MIN_OBSERVATIONS: usize = 6;

#[derive(Debug, Clone, Serialize)]
pub struct PrefillPrediction {
    pub estimated_micros: f64,
    pub observations: usize,
    pub minimum_observed_tokens: usize,
    pub maximum_observed_tokens: usize,
    /// Empirical sequential error, NOT a confidence interval or a safety bound.
    pub mean_relative_prediction_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CostModel {
    theta: [f64; DIM],
    covariance: [[f64; DIM]; DIM],
    contexts: BTreeSet<usize>,
    observations: usize,
    relative_error_sum: f64,
    error_observations: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            theta: [0.0; DIM],
            covariance: [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
            contexts: BTreeSet::new(),
            observations: 0,
            relative_error_sum: 0.0,
            error_observations: 0,
        }
    }
}

fn features(tokens: usize) -> Result<[f64; DIM]> {
    ensure!(
        (1..=32768).contains(&tokens),
        "prefill length outside supported range"
    );
    let t = tokens as f64 / 1024.0;
    Ok([1.0, t, t * t])
}

fn positive_definite(matrix: &[[f64; DIM]; DIM]) -> bool {
    let mut l = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..=i {
            let v = matrix[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !v.is_finite() || v <= 0.0 {
                    return false;
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    true
}

impl CostModel {
    fn predict_seconds(&self, x: &[f64; DIM]) -> f64 {
        self.theta.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    fn observe(&mut self, tokens: usize, micros: f64) -> Result<()> {
        let x = features(tokens)?;
        ensure!(
            micros.is_finite() && micros > 0.0 && micros <= 86_400_000_000.0,
            "prefill time must be finite, positive and no longer than one day"
        );
        let seconds = micros / 1_000_000.0;
        let prediction = self.predict_seconds(&x);
        let residual = seconds - prediction;
        let mut px = [0.0; DIM];
        for (i, value) in px.iter_mut().enumerate() {
            *value = self.covariance[i].iter().zip(x).map(|(p, x)| p * x).sum();
        }
        // lambda=1: reproducible ridge least squares without hidden forgetting.
        let denominator = 1.0 + x.iter().zip(px).map(|(x, p)| x * p).sum::<f64>();
        ensure!(
            denominator.is_finite() && denominator > 0.0,
            "invalid RLS denominator"
        );
        let mut theta = self.theta;
        let mut covariance = self.covariance;
        for i in 0..DIM {
            theta[i] += px[i] / denominator * residual;
            for j in 0..DIM {
                covariance[i][j] -= px[i] * px[j] / denominator;
            }
        }
        for i in 1..DIM {
            let (previous, remaining) = covariance.split_at_mut(i);
            let row = &mut remaining[0];
            for (j, other) in previous.iter_mut().enumerate() {
                let average = (row[j] + other[i]) * 0.5;
                row[j] = average;
                other[i] = average;
            }
        }
        ensure!(
            theta.iter().all(|v| v.is_finite())
                && covariance.iter().flatten().all(|v| v.is_finite())
                && positive_definite(&covariance),
            "RLS numerical invariant failed"
        );
        // Commit only after every numerical check. A rejected update changes nothing.
        if self.observations >= MIN_OBSERVATIONS && prediction > 0.0 {
            self.relative_error_sum += residual.abs() / seconds;
            self.error_observations += 1;
        }
        self.theta = theta;
        self.covariance = covariance;
        self.contexts.insert(tokens);
        self.observations += 1;
        Ok(())
    }

    fn prediction(&self, tokens: usize) -> Option<PrefillPrediction> {
        let x = features(tokens).ok()?;
        if self.observations < MIN_OBSERVATIONS || self.contexts.len() < 3 {
            return None;
        }
        let minimum = *self.contexts.first()?;
        let maximum = *self.contexts.last()?;
        if !(minimum..=maximum).contains(&tokens) {
            return None;
        }
        let error = (self.error_observations > 0)
            .then(|| self.relative_error_sum / self.error_observations as f64);
        // Empirical abstention only; this does not certify future performance.
        if error.is_some_and(|v| !v.is_finite() || v > 0.5) {
            return None;
        }
        let micros = self.predict_seconds(&x) * 1_000_000.0;
        if !micros.is_finite() || micros <= 0.0 {
            return None;
        }
        Some(PrefillPrediction {
            estimated_micros: micros,
            observations: self.observations,
            minimum_observed_tokens: minimum,
            maximum_observed_tokens: maximum,
            mean_relative_prediction_error: error,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PrefillCostPredictor {
    environment_digest: String,
    models: BTreeMap<String, CostModel>,
    evidence_ids: BTreeSet<String>,
}

impl PrefillCostPredictor {
    pub fn new(environment_digest: String) -> Result<Self> {
        ensure!(
            environment_digest.len() == 64
                && environment_digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid environment digest"
        );
        Ok(Self {
            environment_digest,
            models: BTreeMap::new(),
            evidence_ids: BTreeSet::new(),
        })
    }

    /// Caller must supply actual, parity-checked timings, not another prediction.
    /// Evidence IDs prevent duplicate replay from increasing apparent support.
    pub fn observe(
        &mut self,
        environment: &str,
        profile: &str,
        evidence_id: &str,
        tokens: usize,
        micros: f64,
    ) -> Result<()> {
        ensure!(
            environment == self.environment_digest,
            "foreign calibration environment"
        );
        ensure!(
            !profile.trim().is_empty() && profile.len() <= 128,
            "invalid profile ID"
        );
        ensure!(
            !evidence_id.is_empty() && evidence_id.len() <= 192,
            "invalid evidence ID"
        );
        ensure!(
            !self.evidence_ids.contains(evidence_id),
            "duplicate calibration evidence"
        );
        ensure!(
            self.evidence_ids.len() < MAX_OBSERVATIONS,
            "predictor observation budget exhausted"
        );
        ensure!(
            self.models.contains_key(profile) || self.models.len() < MAX_PROFILES,
            "predictor profile budget exhausted"
        );
        let mut model = self.models.get(profile).cloned().unwrap_or_default();
        model.observe(tokens, micros)?;
        self.models.insert(profile.to_owned(), model);
        self.evidence_ids.insert(evidence_id.to_owned());
        Ok(())
    }

    pub fn predict(
        &self,
        environment: &str,
        profile: &str,
        tokens: usize,
    ) -> Option<PrefillPrediction> {
        if environment != self.environment_digest {
            return None;
        }
        self.models.get(profile)?.prediction(tokens)
    }

    /// Reorders only. Unknown profiles remain eligible and are NEVER skipped.
    pub fn order_candidates(
        &self,
        environment: &str,
        profiles: &[String],
        tokens: usize,
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..profiles.len()).collect();
        indices.sort_by(|&a, &b| {
            let pa = self.predict(environment, &profiles[a], tokens);
            let pb = self.predict(environment, &profiles[b], tokens);
            match (pa, pb) {
                (Some(a), Some(b)) => a.estimated_micros.total_cmp(&b.estimated_micros),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
            .then_with(|| a.cmp(&b))
        });
        indices
    }

    pub fn observation_count(&self) -> usize {
        self.evidence_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fitted() -> PrefillCostPredictor {
        let env = "a".repeat(64);
        let mut predictor = PrefillCostPredictor::new(env.clone()).unwrap();
        for repeat in 0..4 {
            for tokens in [256, 512, 1024, 2048] {
                predictor
                    .observe(
                        &env,
                        "p",
                        &format!("{repeat}-{tokens}"),
                        tokens,
                        1000.0 + 200.0 * tokens as f64,
                    )
                    .unwrap();
            }
        }
        predictor
    }
    #[test]
    fn rls_learns_cost_inside_observed_domain() {
        let p = fitted();
        let prediction = p.predict(&"a".repeat(64), "p", 768).unwrap();
        let expected = 1000.0 + 200.0 * 768.0;
        assert!((prediction.estimated_micros / expected - 1.0).abs() < 0.03);
    }
    #[test]
    fn refuses_extrapolation_and_foreign_environments() {
        let p = fitted();
        assert!(p.predict(&"a".repeat(64), "p", 4096).is_none());
        assert!(p.predict(&"b".repeat(64), "p", 512).is_none());
        assert!(p.predict(&"a".repeat(64), "missing", 512).is_none());
    }
    #[test]
    fn invalid_updates_and_duplicates_are_atomic() {
        let mut p = fitted();
        let before = serde_json::to_vec(&p).unwrap();
        for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(p.observe(&"a".repeat(64), "p", "bad", 512, value).is_err());
        }
        assert!(p.observe(&"a".repeat(64), "p", "0-256", 256, 1.0).is_err());
        assert!(p.observe(&"b".repeat(64), "p", "other", 512, 1.0).is_err());
        assert_eq!(serde_json::to_vec(&p).unwrap(), before);
    }
    #[test]
    fn cold_start_preserves_every_candidate_in_original_order() {
        let p = PrefillCostPredictor::new("a".repeat(64)).unwrap();
        let profiles = vec!["z".into(), "a".into(), "b".into()];
        assert_eq!(
            p.order_candidates(&"a".repeat(64), &profiles, 512),
            vec![0, 1, 2]
        );
    }
    #[test]
    fn predictions_do_not_train_or_authorize() {
        let p = fitted();
        let before = serde_json::to_vec(&p).unwrap();
        let profiles = vec!["unknown".into(), "p".into()];
        assert_eq!(
            p.order_candidates(&"a".repeat(64), &profiles, 512),
            vec![1, 0]
        );
        assert_eq!(serde_json::to_vec(&p).unwrap(), before);
    }
    #[test]
    fn repeated_single_length_is_not_sufficient_support() {
        let env = "a".repeat(64);
        let mut p = PrefillCostPredictor::new(env.clone()).unwrap();
        for i in 0..12 {
            p.observe(&env, "p", &i.to_string(), 512, 1000.0).unwrap();
        }
        assert!(p.predict(&env, "p", 512).is_none());
    }
    #[test]
    fn profile_budget_is_bounded() {
        let env = "a".repeat(64);
        let mut p = PrefillCostPredictor::new(env.clone()).unwrap();
        for i in 0..MAX_PROFILES {
            p.observe(&env, &format!("p{i}"), &i.to_string(), 512, 1000.0)
                .unwrap();
        }
        assert!(
            p.observe(&env, "one-too-many", "overflow", 512, 1000.0)
                .is_err()
        );
    }
}
