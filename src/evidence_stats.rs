//! Statistical evidence for measured claims.
//!
//! Pure arithmetic with no I/O and no dependency beyond `serde`. The results
//! are advisory: they qualify a measurement with its uncertainty and never
//! authorize an activation, a promotion or an omission. Each report states the
//! assumptions it does not verify.
//!
//! Two pieces live here:
//!
//! * [`PairedTimingEvidence`] compares paired timings of a baseline and a
//!   candidate and says whether the difference is distinguishable from noise.
//!   It is attached to calibration reports next to the existing gates.
//! * [`SequentialDegradationWatch`] watches a stream of paired losses for
//!   deterioration after a change. It is not connected to a runtime path yet:
//!   there is no post-promotion telemetry stream in this crate to feed it.

use serde::Serialize;
use std::fmt;

/// Invalid input to an evidence computation. The reason is a stable token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceError(&'static str);

impl EvidenceError {
    pub const fn reason(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid_evidence:{}", self.0)
    }
}

impl std::error::Error for EvidenceError {}

pub type EvidenceResult<T> = Result<T, EvidenceError>;

/// A point estimate with a closed interval. An unknown estimate is represented
/// by `None` at the call site, never by NaN.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub n: u64,
}

impl Estimate {
    pub fn new(value: f64, lo: f64, hi: f64, n: u64) -> EvidenceResult<Self> {
        if !(value.is_finite() && lo.is_finite() && hi.is_finite()) {
            return Err(EvidenceError("estimate_not_finite"));
        }
        if !(lo <= value && value <= hi) {
            return Err(EvidenceError("estimate_interval_order"));
        }
        Ok(Self { value, lo, hi, n })
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

/// Wilson score interval for an integer binomial count. The interval always
/// contains the observed proportion.
pub fn wilson(successes: u64, trials: u64, z: f64) -> EvidenceResult<(f64, f64)> {
    if !z.is_finite() || z <= 0.0 {
        return Err(EvidenceError("z_must_be_positive_and_finite"));
    }
    if successes > trials {
        return Err(EvidenceError("successes_exceed_trials"));
    }
    if trials == 0 {
        return Ok((0.0, 1.0));
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Ok(((center - half).clamp(0.0, p), (center + half).clamp(p, 1.0)))
}

/// Two-sided 95% Student-t critical value. Values between table entries are
/// rounded up to the lower-degree bin, so intervals are never narrower than
/// the exact ones.
fn student_t_975(degrees_of_freedom: u64) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match degrees_of_freedom {
        0 => f64::INFINITY,
        1..=30 => TABLE[(degrees_of_freedom - 1) as usize],
        31..=40 => 2.040,
        41..=60 => 2.020,
        61..=120 => 2.000,
        _ => 1.980,
    }
}

pub const PAIRED_TIMING_SCHEMA: &str = "mslim.paired_timing_evidence/v1";
/// Fewer pairs than this never support a claim of difference.
pub const MINIMUM_TIMING_PAIRS: u64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingClaim {
    InsufficientPairs,
    CandidateFasterWithConfidence,
    CandidateSlowerWithConfidence,
    NotDistinguishableFromNoise,
}

/// Paired comparison of baseline and candidate timings.
///
/// For each pair the log time ratio `ln(candidate / baseline)` is formed;
/// negative means the candidate was faster. The mean log ratio gets a 95%
/// Student-t interval and the share of pairs where the candidate was strictly
/// faster gets a 95% Wilson interval. The claim follows only the first one.
#[derive(Debug, Clone, Serialize)]
pub struct PairedTimingEvidence {
    pub schema: &'static str,
    pub pairs: u64,
    pub candidate_faster: u64,
    pub candidate_faster_rate: Option<Estimate>,
    pub mean_log_time_ratio: Option<Estimate>,
    pub claim: TimingClaim,
    pub authority: &'static str,
    pub limitations: &'static [&'static str],
}

impl PairedTimingEvidence {
    /// `pairs` holds `(baseline_time, candidate_time)` in one consistent unit.
    pub fn from_pairs(pairs: &[(f64, f64)]) -> EvidenceResult<Self> {
        let mut logs = Vec::with_capacity(pairs.len());
        let mut faster = 0_u64;
        for &(baseline, candidate) in pairs {
            if !(baseline.is_finite() && candidate.is_finite() && baseline > 0.0 && candidate > 0.0)
            {
                return Err(EvidenceError("timings_must_be_finite_and_positive"));
            }
            if candidate < baseline {
                faster += 1;
            }
            logs.push((candidate / baseline).ln());
        }
        let n = logs.len() as u64;

        let candidate_faster_rate = if n > 0 {
            let (lo, hi) = wilson(faster, n, 1.96)?;
            Some(Estimate::new(faster as f64 / n as f64, lo, hi, n)?)
        } else {
            None
        };

        let mean_log_time_ratio = if n >= 2 {
            let count = n as f64;
            let mean = logs.iter().sum::<f64>() / count;
            let variance = logs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0);
            let margin = student_t_975(n - 1) * (variance / count).sqrt();
            Some(Estimate::new(mean, mean - margin, mean + margin, n)?)
        } else {
            None
        };

        let claim = match mean_log_time_ratio {
            _ if n < MINIMUM_TIMING_PAIRS => TimingClaim::InsufficientPairs,
            Some(estimate) if estimate.hi < 0.0 => TimingClaim::CandidateFasterWithConfidence,
            Some(estimate) if estimate.lo > 0.0 => TimingClaim::CandidateSlowerWithConfidence,
            _ => TimingClaim::NotDistinguishableFromNoise,
        };

        Ok(Self {
            schema: PAIRED_TIMING_SCHEMA,
            pairs: n,
            candidate_faster: faster,
            candidate_faster_rate,
            mean_log_time_ratio,
            claim,
            authority: "advisory_never_changes_a_gate_decision",
            limitations: &[
                "pairs_are_repeats_over_few_contexts_not_independent_workloads",
                "baseline_and_candidate_run_in_separate_blocks_so_drift_is_not_cancelled",
                "log_ratio_interval_assumes_approximately_normal_differences",
            ],
        })
    }
}

/// Smallest loss scale used when none is known: 1024 machine epsilons.
const WATCH_EPSILON: f64 = f64::EPSILON * 1024.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DegradationObservation {
    pub samples: u64,
    pub wealth: f64,
    pub cusum: f64,
    pub relative_excess: f64,
    pub delta: f64,
    pub wealth_alarm: bool,
    pub cusum_alarm: bool,
    pub detected: bool,
}

/// Anytime-valid advisory detector for loss deterioration after a change.
///
/// It combines a capital process on a predictable bounded transform of the
/// paired loss difference with a relative CUSUM. The capital process is a
/// valid e-process only while no difference exceeds twice the predictable
/// bound (no clipping); the CUSUM keeps power over long streams. A false alarm
/// costs a review and a miss stops nothing, so the watch carries no authority.
///
/// Pass a realistic loss scale as `bound`. With `bound == 0` the first
/// observation is evaluated against a tiny scale and any first worsening
/// places the maximum bet.
#[derive(Debug, Clone, Serialize)]
pub struct SequentialDegradationWatch {
    wealth: f64,
    cusum: f64,
    bound: f64,
    samples: u64,
    lambda_bet: f64,
    cusum_allowance: f64,
    cusum_threshold: f64,
    wealth_alarm: f64,
}

impl SequentialDegradationWatch {
    /// Default betting fraction 0.25, CUSUM allowance 2%, CUSUM threshold 1.0
    /// and e-value alarm 20.
    pub fn new(bound: f64) -> EvidenceResult<Self> {
        Self::with_parameters(bound, 0.25, 0.02, 1.0, 20.0)
    }

    pub fn with_parameters(
        bound: f64,
        lambda_bet: f64,
        cusum_allowance: f64,
        cusum_threshold: f64,
        wealth_alarm: f64,
    ) -> EvidenceResult<Self> {
        if !bound.is_finite() || bound < 0.0 {
            return Err(EvidenceError("bound_must_be_finite_and_non_negative"));
        }
        if !lambda_bet.is_finite() || lambda_bet <= 0.0 || lambda_bet > 1.0 {
            return Err(EvidenceError("lambda_bet_must_be_in_0_1"));
        }
        if !cusum_allowance.is_finite() || cusum_allowance < 0.0 {
            return Err(EvidenceError("cusum_allowance_must_be_non_negative"));
        }
        if !cusum_threshold.is_finite() || cusum_threshold <= 0.0 {
            return Err(EvidenceError("cusum_threshold_must_be_positive"));
        }
        if !wealth_alarm.is_finite() || wealth_alarm <= 1.0 {
            return Err(EvidenceError("wealth_alarm_must_exceed_one"));
        }
        Ok(Self {
            wealth: 1.0,
            cusum: 0.0,
            bound: if bound == 0.0 { WATCH_EPSILON } else { bound },
            samples: 0,
            lambda_bet,
            cusum_allowance,
            cusum_threshold,
            wealth_alarm,
        })
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Feeds one paired observation. Invalid input leaves the state untouched.
    pub fn observe(
        &mut self,
        parent_loss: f64,
        live_loss: f64,
    ) -> EvidenceResult<DegradationObservation> {
        if !parent_loss.is_finite() || !live_loss.is_finite() {
            return Err(EvidenceError("losses_must_be_finite"));
        }
        if parent_loss < 0.0 || live_loss < 0.0 {
            return Err(EvidenceError("losses_must_be_non_negative"));
        }
        let delta = live_loss - parent_loss;
        let predictable_bound = self.bound.max(WATCH_EPSILON);
        let amplitude = (delta / (2.0 * predictable_bound)).clamp(-0.999, 0.999);
        self.wealth =
            (self.wealth * (1.0 + self.lambda_bet * amplitude)).clamp(f64::MIN_POSITIVE, 1e12);
        let relative = delta / parent_loss.abs().max(WATCH_EPSILON);
        self.cusum = (self.cusum + relative - self.cusum_allowance).max(0.0);
        self.bound = predictable_bound.max(delta.abs()).max(WATCH_EPSILON);
        self.samples += 1;
        let wealth_alarm = self.wealth >= self.wealth_alarm;
        let cusum_alarm = self.cusum >= self.cusum_threshold;
        Ok(DegradationObservation {
            samples: self.samples,
            wealth: self.wealth,
            cusum: self.cusum,
            relative_excess: relative,
            delta,
            wealth_alarm,
            cusum_alarm,
            detected: wealth_alarm || cusum_alarm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn wilson_matches_published_intervals_and_contains_the_proportion() {
        let (lo, hi) = wilson(5, 10, 1.96).unwrap();
        assert!(close(lo, 0.2366, 1e-3) && close(hi, 0.7634, 1e-3));
        let (lo, hi) = wilson(10, 10, 1.96).unwrap();
        assert!(close(lo, 0.7224, 1e-3));
        assert_eq!(hi, 1.0);
        let (lo, hi) = wilson(0, 10, 1.96).unwrap();
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0 && hi < 0.4);
        assert_eq!(wilson(0, 0, 1.96).unwrap(), (0.0, 1.0));
    }

    #[test]
    fn wilson_rejects_impossible_counts_and_bad_z() {
        assert!(wilson(3, 2, 1.96).is_err());
        assert!(wilson(1, 2, 0.0).is_err());
        assert!(wilson(1, 2, f64::NAN).is_err());
    }

    #[test]
    fn estimate_rejects_unordered_or_non_finite_intervals() {
        assert!(Estimate::new(1.0, 2.0, 3.0, 1).is_err());
        assert!(Estimate::new(f64::NAN, 0.0, 1.0, 1).is_err());
        assert!(close(
            Estimate::new(1.0, 0.5, 2.0, 1).unwrap().width(),
            1.5,
            1e-12
        ));
    }

    #[test]
    fn consistent_speedup_is_reported_with_confidence() {
        let evidence = PairedTimingEvidence::from_pairs(&[
            (1000.0, 800.0),
            (1010.0, 805.0),
            (990.0, 795.0),
            (1005.0, 790.0),
            (995.0, 802.0),
        ])
        .unwrap();
        assert_eq!(evidence.claim, TimingClaim::CandidateFasterWithConfidence);
        assert_eq!(evidence.candidate_faster, 5);
        let rate = evidence.candidate_faster_rate.unwrap();
        assert!(rate.lo > 0.5 && rate.hi == 1.0);
        let mean = evidence.mean_log_time_ratio.unwrap();
        assert!(mean.hi < 0.0 && close(mean.value, 0.8_f64.ln(), 0.01));
    }

    #[test]
    fn consistent_slowdown_is_reported_with_confidence() {
        let evidence = PairedTimingEvidence::from_pairs(&[
            (100.0, 130.0),
            (101.0, 131.0),
            (99.0, 129.0),
            (100.0, 128.0),
            (102.0, 133.0),
        ])
        .unwrap();
        assert_eq!(evidence.claim, TimingClaim::CandidateSlowerWithConfidence);
        assert_eq!(evidence.candidate_faster, 0);
    }

    #[test]
    fn jitter_around_equal_times_is_not_a_difference() {
        let evidence = PairedTimingEvidence::from_pairs(&[
            (100.0, 101.0),
            (100.0, 99.0),
            (100.0, 100.5),
            (100.0, 99.5),
            (100.0, 100.0),
        ])
        .unwrap();
        assert_eq!(evidence.claim, TimingClaim::NotDistinguishableFromNoise);
    }

    #[test]
    fn identical_timings_are_ties_not_wins() {
        let evidence =
            PairedTimingEvidence::from_pairs(&[(50.0, 50.0), (60.0, 60.0), (70.0, 70.0)]).unwrap();
        assert_eq!(evidence.candidate_faster, 0);
        assert_eq!(evidence.claim, TimingClaim::NotDistinguishableFromNoise);
    }

    #[test]
    fn too_few_pairs_never_support_a_claim() {
        let two = PairedTimingEvidence::from_pairs(&[(100.0, 50.0), (100.0, 50.0)]).unwrap();
        assert_eq!(two.claim, TimingClaim::InsufficientPairs);
        assert!(two.mean_log_time_ratio.is_some());
        let none = PairedTimingEvidence::from_pairs(&[]).unwrap();
        assert_eq!(none.claim, TimingClaim::InsufficientPairs);
        assert!(none.candidate_faster_rate.is_none() && none.mean_log_time_ratio.is_none());
    }

    #[test]
    fn invalid_timings_are_rejected() {
        for bad in [
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(PairedTimingEvidence::from_pairs(&[bad]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn evidence_serializes_with_its_authority_and_limits() {
        let value = serde_json::to_value(
            PairedTimingEvidence::from_pairs(&[(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]).unwrap(),
        )
        .unwrap();
        assert_eq!(value["schema"], PAIRED_TIMING_SCHEMA);
        assert_eq!(value["authority"], "advisory_never_changes_a_gate_decision");
        assert!(value["limitations"].as_array().unwrap().len() >= 3);
    }

    #[test]
    fn equal_losses_never_alarm_or_move_the_state() {
        let mut watch = SequentialDegradationWatch::new(0.1).unwrap();
        for _ in 0..1000 {
            let observation = watch.observe(1.0, 1.0).unwrap();
            assert!(!observation.detected);
            assert_eq!(observation.wealth, 1.0);
            assert_eq!(observation.cusum, 0.0);
        }
    }

    #[test]
    fn improvement_shrinks_wealth_and_never_alarms() {
        let mut watch = SequentialDegradationWatch::new(0.1).unwrap();
        let mut last = 1.0;
        for _ in 0..200 {
            let observation = watch.observe(1.0, 0.9).unwrap();
            assert!(!observation.detected);
            last = observation.wealth;
        }
        assert!(last < 1.0);
    }

    #[test]
    fn steady_ten_percent_degradation_trips_the_cusum_at_the_thirteenth_sample() {
        let mut watch = SequentialDegradationWatch::new(0.1).unwrap();
        let mut first = None;
        for i in 1..=30_u64 {
            let observation = watch.observe(1.0, 1.1).unwrap();
            if observation.detected {
                assert!(observation.cusum_alarm && !observation.wealth_alarm);
                first = Some(i);
                break;
            }
        }
        assert_eq!(first, Some(13));
    }

    #[test]
    fn capital_process_alone_trips_at_the_twenty_sixth_sample() {
        let mut watch =
            SequentialDegradationWatch::with_parameters(0.1, 0.25, 0.02, 1.0e9, 20.0).unwrap();
        let mut first = None;
        for i in 1..=40_u64 {
            let observation = watch.observe(1.0, 1.1).unwrap();
            if observation.wealth_alarm {
                assert!(!observation.cusum_alarm && observation.wealth >= 20.0);
                first = Some(i);
                break;
            }
        }
        assert_eq!(first, Some(26));
    }

    #[test]
    fn invalid_observations_leave_the_watch_untouched() {
        let mut watch = SequentialDegradationWatch::new(0.1).unwrap();
        watch.observe(1.0, 1.05).unwrap();
        let before = watch.clone();
        assert!(watch.observe(f64::NAN, 1.0).is_err());
        assert!(watch.observe(1.0, f64::INFINITY).is_err());
        assert!(watch.observe(-1.0, 1.0).is_err());
        assert!(watch.observe(1.0, -0.5).is_err());
        assert_eq!(watch.samples(), before.samples());
        assert_eq!(
            serde_json::to_value(&watch).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(SequentialDegradationWatch::new(-1.0).is_err());
        assert!(SequentialDegradationWatch::new(f64::NAN).is_err());
        assert!(SequentialDegradationWatch::with_parameters(0.1, 0.0, 0.02, 1.0, 20.0).is_err());
        assert!(SequentialDegradationWatch::with_parameters(0.1, 1.5, 0.02, 1.0, 20.0).is_err());
        assert!(SequentialDegradationWatch::with_parameters(0.1, 0.25, -0.1, 1.0, 20.0).is_err());
        assert!(SequentialDegradationWatch::with_parameters(0.1, 0.25, 0.02, 0.0, 20.0).is_err());
        assert!(SequentialDegradationWatch::with_parameters(0.1, 0.25, 0.02, 1.0, 1.0).is_err());
        assert!(SequentialDegradationWatch::new(0.0).is_ok());
    }
}
