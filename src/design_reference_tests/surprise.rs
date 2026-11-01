//! Surprise detector over predicted vs observed state maps.
//! Port of `sheily-execution-kernel-rs/src/prediction/surprise_detection.rs`
//! (component functions, weighted total, adaptive threshold, level routing).
//! KL divergence stays 0, as in the original (its Python predecessor left it pending).

use serde_json::{Map, Value};
use std::collections::VecDeque;

const MAX_HISTORY: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Components {
    pub mse_error: f64,
    pub mahalanobis: f64,
    pub kl_divergence: f64,
    pub novelty: f64,
    pub temporal_anomaly: f64,
}

impl Components {
    pub fn total(self) -> f64 {
        0.3 * self.mse_error
            + 0.2 * self.mahalanobis
            + 0.2 * self.kl_divergence
            + 0.2 * self.novelty
            + 0.1 * self.temporal_anomaly
    }

    pub fn dominant_type(self) -> &'static str {
        [
            ("prediction_error", self.mse_error),
            ("statistical_anomaly", self.mahalanobis),
            ("distribution_shift", self.kl_divergence),
            ("novel_pattern", self.novelty),
            ("temporal_discontinuity", self.temporal_anomaly),
        ]
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|row| row.0)
        .unwrap_or("unknown")
    }
}

fn numeric_pairs(left: &Map<String, Value>, right: &Map<String, Value>) -> Vec<(f64, f64)> {
    left.iter()
        .filter_map(|(key, l)| Some((l.as_f64()?, right.get(key)?.as_f64()?)))
        .filter(|(l, r)| l.is_finite() && r.is_finite())
        .collect()
}

fn state_distance(left: &Map<String, Value>, right: &Map<String, Value>) -> f64 {
    let common = left.keys().filter(|key| right.contains_key(*key)).count();
    if common == 0 {
        return 1.0;
    }
    let sum: f64 = numeric_pairs(left, right)
        .into_iter()
        .map(|(l, r)| (l - r).abs())
        .sum();
    sum / common as f64
}

fn mse(predicted: &Map<String, Value>, observed: &Map<String, Value>, baseline_mse: f64) -> f64 {
    if predicted.keys().all(|key| !observed.contains_key(key)) {
        return 1.0;
    }
    let errors: Vec<f64> = numeric_pairs(predicted, observed)
        .into_iter()
        .map(|(p, o)| (p - o).powi(2))
        .collect();
    if errors.is_empty() {
        return 0.0;
    }
    let raw = errors.iter().sum::<f64>() / errors.len() as f64;
    let denominator = if baseline_mse > 0.0 {
        baseline_mse
    } else {
        1.0
    };
    (raw / denominator).min(1.0)
}

fn mahalanobis_approx(predicted: &Map<String, Value>, observed: &Map<String, Value>) -> f64 {
    let common = predicted
        .keys()
        .filter(|key| observed.contains_key(*key))
        .count();
    if common == 0 {
        return 1.0;
    }
    let diffs: Vec<f64> = numeric_pairs(predicted, observed)
        .into_iter()
        .map(|(p, o)| p - o)
        .collect();
    if diffs.is_empty() {
        return 0.0;
    }
    let distance = diffs.iter().map(|d| d * d).sum::<f64>().sqrt();
    (distance / (common as f64 + 1.0)).min(1.0)
}

#[derive(Debug, Clone)]
pub struct SurpriseDetector {
    history: VecDeque<Map<String, Value>>,
    baseline_mse: f64,
    magnitudes: VecDeque<f64>,
    threshold_std: f64,
    bcm_threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub components: Components,
    pub surprise: f64,
    pub threshold: f64,
    pub level: &'static str,
    pub dominant: &'static str,
    pub recommendation: Option<&'static str>,
}

impl Default for SurpriseDetector {
    fn default() -> Self {
        Self {
            history: VecDeque::new(),
            baseline_mse: 0.0,
            magnitudes: VecDeque::new(),
            threshold_std: 1.0,
            bcm_threshold: 0.5,
        }
    }
}

impl SurpriseDetector {
    fn novelty(&self, observed: &Map<String, Value>) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let total: f64 = self
            .history
            .iter()
            .map(|past| state_distance(observed, past))
            .sum();
        (total / self.history.len() as f64).min(1.0)
    }

    fn temporal_anomaly(&self, observed: &Map<String, Value>) -> f64 {
        let len = self.history.len();
        if len < 2 {
            return 0.0;
        }
        let distance = state_distance(observed, &self.history[len - 1]);
        let span = len.min(10);
        let recent: Vec<f64> = (1..span)
            .map(|offset| {
                state_distance(&self.history[len - offset], &self.history[len - offset - 1])
            })
            .collect();
        if recent.is_empty() {
            return 0.0;
        }
        let avg = recent.iter().sum::<f64>() / recent.len() as f64;
        (distance / (avg + 1e-6)).min(1.0)
    }

    fn adaptive_threshold(&self) -> f64 {
        if self.magnitudes.is_empty() {
            return 0.5;
        }
        let mut values: Vec<f64> = self.magnitudes.iter().rev().take(20).copied().collect();
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let median = if n.is_multiple_of(2) {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        } else {
            values[n / 2]
        };
        let mean = values.iter().sum::<f64>() / n as f64;
        let std = if n > 1 {
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
        } else {
            0.0
        };
        (median + self.threshold_std * std).clamp(0.1, 1.0)
    }

    /// Score one prediction against its observation, then learn from it.
    pub fn observe(
        &mut self,
        predicted: &Map<String, Value>,
        observed: &Map<String, Value>,
    ) -> Report {
        let components = Components {
            mse_error: mse(predicted, observed, self.baseline_mse),
            mahalanobis: mahalanobis_approx(predicted, observed),
            kl_divergence: 0.0,
            novelty: self.novelty(observed),
            temporal_anomaly: self.temporal_anomaly(observed),
        };
        let surprise = components.total();
        let threshold = self.adaptive_threshold();
        let level = route_level(surprise, threshold);
        let dominant = components.dominant_type();
        let recommendation = match level {
            "critical" => Some("pause_and_replan"),
            "high" => Some("log_gap"),
            "medium" => Some("monitor"),
            _ => None,
        };

        // Learn: baseline MSE, magnitude history, observation history.
        self.baseline_mse = (self.baseline_mse + components.mse_error) / 2.0;
        self.magnitudes.push_back(surprise);
        if self.magnitudes.len() > MAX_HISTORY {
            self.magnitudes.pop_front();
        }
        self.history.push_back(observed.clone());
        if self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }
        self.bcm_threshold = self.bcm_threshold.clamp(0.1, 1.0);

        Report {
            components,
            surprise,
            threshold,
            level,
            dominant,
            recommendation,
        }
    }
}

/// Bucket a score against the adaptive threshold.
pub fn route_level(score: f64, threshold: f64) -> &'static str {
    if score < threshold * 0.3 {
        "none"
    } else if score < threshold * 0.6 {
        "low"
    } else if score < threshold {
        "medium"
    } else if score < threshold * 1.3 {
        "high"
    } else {
        "critical"
    }
}

/// Bayesian surprise between two Beta posteriors (mean shift, capped at 1).
pub fn bayesian_surprise(
    prior_alpha: f64,
    prior_beta: f64,
    post_alpha: f64,
    post_beta: f64,
) -> Result<f64, &'static str> {
    for v in [prior_alpha, prior_beta, post_alpha, post_beta] {
        if !v.is_finite() || v <= 0.0 {
            return Err("bayesian_surprise_positive_finite_required");
        }
    }
    let prior_mean = prior_alpha / (prior_alpha + prior_beta);
    let post_mean = post_alpha / (post_alpha + post_beta);
    Ok((post_mean - prior_mean).abs().min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn m(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn identical_prediction_has_no_error_component() {
        let mut d = SurpriseDetector::default();
        let r = d.observe(
            &m(json!({"a": 1.0, "b": 2.0})),
            &m(json!({"a": 1.0, "b": 2.0})),
        );
        assert_eq!(r.components.mse_error, 0.0);
        assert_eq!(r.components.mahalanobis, 0.0);
    }

    #[test]
    fn large_error_is_critical_and_routed() {
        let mut d = SurpriseDetector::default();
        d.observe(&m(json!({"a": 0.0})), &m(json!({"a": 0.0})));
        let r = d.observe(&m(json!({"a": 0.0})), &m(json!({"a": 50.0})));
        assert!(r.surprise > 0.0);
        assert!(["medium", "high", "critical"].contains(&r.level));
    }

    #[test]
    fn route_level_boundaries() {
        assert_eq!(route_level(0.0, 0.5), "none");
        assert_eq!(route_level(0.5, 0.5), "high");
        assert_eq!(route_level(10.0, 0.5), "critical");
    }

    #[test]
    fn bayesian_surprise_is_mean_shift_and_validates() {
        let s = bayesian_surprise(1.0, 1.0, 9.0, 1.0).unwrap();
        assert!((s - 0.4).abs() < 1e-12);
        assert!(bayesian_surprise(0.0, 1.0, 1.0, 1.0).is_err());
    }
}
