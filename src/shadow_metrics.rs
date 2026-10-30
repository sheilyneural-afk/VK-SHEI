//! Pure, diagnostic-only aggregation for long-context certificate analysis.
//!
//! This module owns no tensors, storage or selection policy. Runtime code may
//! feed it scalar observations only after the canonical path has already
//! produced them. It must never authorize an omission.

use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ShadowEventId {
    pub sequence_position: usize,
    pub layer: usize,
    pub query_head: Option<usize>,
    pub block_id: Option<usize>,
    pub token_position: Option<usize>,
}

#[derive(Clone, Debug)]
struct ScalarSample {
    id: ShadowEventId,
    value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DistributionSummary {
    pub count: usize,
    pub minimum: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub maximum: f64,
    pub minimum_event: ShadowEventId,
    pub maximum_event: ShadowEventId,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShadowViolation {
    pub kind: &'static str,
    pub event: ShadowEventId,
    pub expected_upper: f64,
    pub observed_exact: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShadowDiagnosticReport {
    pub mode: &'static str,
    pub authority: &'static str,
    pub additional_cold_reads: u64,
    pub individual_omission_threshold: Option<DistributionSummary>,
    pub aabb_log_mass_slack: Option<DistributionSummary>,
    pub aabb_bound_violations: Vec<ShadowViolation>,
    pub attention_error_by_tolerance: Vec<AttentionErrorSummary>,
    pub limitations: Vec<&'static str>,
}

#[derive(Default)]
pub struct ShadowMetrics {
    individual_thresholds: Vec<ScalarSample>,
    region_slacks: Vec<ScalarSample>,
    violations: Vec<ShadowViolation>,
    attention_errors: Vec<AttentionErrorSample>,
}

#[derive(Clone, Debug)]
struct AttentionErrorSample {
    id: ShadowEventId,
    tolerance: f64,
    omitted_tokens: usize,
    certified_bound: f64,
    actual_linf_error: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttentionErrorSummary {
    pub tolerance: f64,
    pub head_events_total: usize,
    pub head_events_with_omission: usize,
    pub omitted_tokens_total: usize,
    pub zero_error_events_with_omission: usize,
    pub actual_linf_error: Option<DistributionSummary>,
    pub certified_bound: Option<DistributionSummary>,
    pub finite_bound_to_error_ratio: Option<DistributionSummary>,
    pub bound_violations: Vec<ShadowViolation>,
}

impl ShadowMetrics {
    pub fn record_individual_threshold(&mut self, id: ShadowEventId, threshold: f64) {
        if threshold.is_finite() && threshold >= 0.0 {
            self.individual_thresholds.push(ScalarSample {
                id,
                value: threshold,
            });
        }
    }

    pub fn record_region_bound(
        &mut self,
        id: ShadowEventId,
        aabb_log_mass_upper: f64,
        exact_log_mass_reference: f64,
    ) {
        if !aabb_log_mass_upper.is_finite() || !exact_log_mass_reference.is_finite() {
            self.violations.push(ShadowViolation {
                kind: "non_finite_region_bound",
                event: id,
                expected_upper: aabb_log_mass_upper,
                observed_exact: exact_log_mass_reference,
            });
            return;
        }
        let slack = aabb_log_mass_upper - exact_log_mass_reference;
        if slack < -1e-10 {
            self.violations.push(ShadowViolation {
                kind: "aabb_below_exact_region_score",
                event: id.clone(),
                expected_upper: aabb_log_mass_upper,
                observed_exact: exact_log_mass_reference,
            });
        }
        self.region_slacks.push(ScalarSample { id, value: slack });
    }

    pub fn record_attention_error(
        &mut self,
        id: ShadowEventId,
        tolerance: f64,
        omitted_tokens: usize,
        certified_bound: f64,
        actual_linf_error: f64,
    ) {
        if tolerance.is_finite()
            && tolerance >= 0.0
            && certified_bound.is_finite()
            && certified_bound >= 0.0
            && actual_linf_error.is_finite()
            && actual_linf_error >= 0.0
        {
            self.attention_errors.push(AttentionErrorSample {
                id,
                tolerance,
                omitted_tokens,
                certified_bound,
                actual_linf_error,
            });
        }
    }

    pub fn report(self) -> ShadowDiagnosticReport {
        ShadowDiagnosticReport {
            mode: "read_only_post_materialization_scalars",
            authority: "diagnostic_only_never_selects_or_omits",
            additional_cold_reads: 0,
            individual_omission_threshold: summarize(self.individual_thresholds),
            aabb_log_mass_slack: summarize(self.region_slacks),
            aabb_bound_violations: self.violations,
            attention_error_by_tolerance: summarize_attention_errors(self.attention_errors),
            limitations: vec![
                "samples_only_events_where_canonical_path_loaded_every_region",
                "omissions_are_simulated_only_in_local_attention_output",
                "no_shadow_hidden_state_or_logit_propagation_is_performed",
                "aabb_slack_uses_exact_scores_already_materialized_by_canonical_path",
            ],
        }
    }
}

fn summarize_attention_errors(
    mut samples: Vec<AttentionErrorSample>,
) -> Vec<AttentionErrorSummary> {
    samples.sort_by(|left, right| left.tolerance.total_cmp(&right.tolerance));
    let mut groups = Vec::<Vec<AttentionErrorSample>>::new();
    for sample in samples {
        if groups
            .last()
            .is_none_or(|group| group[0].tolerance.to_bits() != sample.tolerance.to_bits())
        {
            groups.push(Vec::new());
        }
        groups.last_mut().expect("grupo recién creado").push(sample);
    }
    groups
        .into_iter()
        .map(|group| {
            let tolerance = group[0].tolerance;
            let mut errors = Vec::with_capacity(group.len());
            let mut bounds = Vec::with_capacity(group.len());
            let mut ratios = Vec::with_capacity(group.len());
            let mut violations = Vec::new();
            let mut omitted_by_event = BTreeMap::<(usize, usize), usize>::new();
            let head_events_total = group.len();
            let mut head_events_with_omission = 0_usize;
            let mut zero_error_events_with_omission = 0_usize;
            for sample in group {
                omitted_by_event
                    .entry((sample.id.sequence_position, sample.id.layer))
                    .and_modify(|count| *count = (*count).max(sample.omitted_tokens))
                    .or_insert(sample.omitted_tokens);
                if sample.omitted_tokens == 0 {
                    continue;
                }
                head_events_with_omission += 1;
                errors.push(ScalarSample {
                    id: sample.id.clone(),
                    value: sample.actual_linf_error,
                });
                bounds.push(ScalarSample {
                    id: sample.id.clone(),
                    value: sample.certified_bound,
                });
                if sample.actual_linf_error == 0.0 {
                    zero_error_events_with_omission += 1;
                } else {
                    ratios.push(ScalarSample {
                        id: sample.id.clone(),
                        value: sample.certified_bound / sample.actual_linf_error,
                    });
                }
                if sample.actual_linf_error > sample.certified_bound + 1e-10 {
                    violations.push(ShadowViolation {
                        kind: "actual_attention_error_exceeds_certificate",
                        event: sample.id,
                        expected_upper: sample.certified_bound,
                        observed_exact: sample.actual_linf_error,
                    });
                }
            }
            AttentionErrorSummary {
                tolerance,
                head_events_total,
                head_events_with_omission,
                omitted_tokens_total: omitted_by_event.values().sum(),
                zero_error_events_with_omission,
                actual_linf_error: summarize(errors),
                certified_bound: summarize(bounds),
                finite_bound_to_error_ratio: summarize(ratios),
                bound_violations: violations,
            }
        })
        .collect()
}

fn summarize(mut samples: Vec<ScalarSample>) -> Option<DistributionSummary> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.value.total_cmp(&right.value));
    let minimum = samples.first().expect("muestras no vacías");
    let maximum = samples.last().expect("muestras no vacías");
    Some(DistributionSummary {
        count: samples.len(),
        minimum: minimum.value,
        p50: nearest_rank(&samples, 50),
        p90: nearest_rank(&samples, 90),
        p99: nearest_rank(&samples, 99),
        maximum: maximum.value,
        minimum_event: minimum.id.clone(),
        maximum_event: maximum.id.clone(),
    })
}

fn nearest_rank(samples: &[ScalarSample], percentile: usize) -> f64 {
    debug_assert!(!samples.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    let rank = (percentile * samples.len()).div_ceil(100);
    samples[rank.saturating_sub(1).min(samples.len() - 1)].value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(index: usize) -> ShadowEventId {
        ShadowEventId {
            sequence_position: index,
            layer: index % 2,
            query_head: Some(index % 4),
            block_id: Some(index),
            token_position: None,
        }
    }

    #[test]
    fn nearest_rank_summary_is_deterministic_and_keeps_event_identity() {
        let mut metrics = ShadowMetrics::default();
        for value in (1..=10).rev() {
            metrics.record_individual_threshold(event(value), value as f64);
        }
        let report = metrics.report();
        let summary = report.individual_omission_threshold.unwrap();
        assert_eq!(summary.count, 10);
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.p50, 5.0);
        assert_eq!(summary.p90, 9.0);
        assert_eq!(summary.p99, 10.0);
        assert_eq!(summary.maximum, 10.0);
        assert_eq!(summary.minimum_event.sequence_position, 1);
        assert_eq!(summary.maximum_event.sequence_position, 10);
    }

    #[test]
    fn aabb_slack_records_valid_bounds_and_flags_underestimates() {
        let mut metrics = ShadowMetrics::default();
        metrics.record_region_bound(event(1), 4.0, 3.5);
        metrics.record_region_bound(event(2), 2.0, 2.25);
        let report = metrics.report();
        let summary = report.aabb_log_mass_slack.unwrap();
        assert_eq!(summary.count, 2);
        assert_eq!(summary.minimum, -0.25);
        assert_eq!(summary.maximum, 0.5);
        assert_eq!(report.aabb_bound_violations.len(), 1);
        assert_eq!(
            report.aabb_bound_violations[0].kind,
            "aabb_below_exact_region_score"
        );
    }

    #[test]
    fn omitted_mass_bound_is_monotonic_for_non_negative_weights() {
        let value_max = 3.0;
        let weights = [0.001, 0.002, 0.004, 0.008];
        let mut omitted_mass = 0.0;
        let mut previous_bound = 0.0;
        for weight in weights {
            omitted_mass += weight;
            let bound = 2.0 * omitted_mass * value_max;
            assert!(bound >= previous_bound);
            previous_bound = bound;
        }
    }

    #[test]
    fn attention_error_is_grouped_by_tolerance_and_checks_the_bound() {
        let mut metrics = ShadowMetrics::default();
        metrics.record_attention_error(event(1), 1e-4, 2, 8e-5, 2e-5);
        metrics.record_attention_error(event(2), 1e-4, 3, 9e-5, 1e-4);
        metrics.record_attention_error(event(3), 1e-3, 5, 7e-4, 0.0);
        let report = metrics.report();
        assert_eq!(report.attention_error_by_tolerance.len(), 2);
        let first = &report.attention_error_by_tolerance[0];
        assert_eq!(first.head_events_total, 2);
        assert_eq!(first.head_events_with_omission, 2);
        assert_eq!(first.omitted_tokens_total, 5);
        assert_eq!(first.bound_violations.len(), 1);
        assert_eq!(
            first.finite_bound_to_error_ratio.as_ref().unwrap().maximum,
            4.0
        );
        let second = &report.attention_error_by_tolerance[1];
        assert_eq!(second.zero_error_events_with_omission, 1);
        assert!(second.finite_bound_to_error_ratio.is_none());
    }
}
