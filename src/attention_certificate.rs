//! Exact omitted-mass certificate for query-adaptive attention views.

use anyhow::{Result, ensure};
use half::bf16;
use std::collections::BTreeSet;

pub fn certified(bounds: &[f64], tolerance: f64) -> bool {
    tolerance.is_finite()
        && tolerance >= 0.0
        && !bounds.is_empty()
        && bounds
            .iter()
            .all(|bound| bound.is_finite() && *bound <= tolerance)
}

/// Certified output-error bound for one attention head. `kept_scores` are
/// exact scores for materialized tokens; `omitted_score_uppers` are rigorous
/// upper bounds for every omitted region's log-mass (already including its
/// token count); value magnitudes are kept separate from softmax mass.
pub fn omitted_error_bound(
    kept_scores: &[f64],
    omitted_score_uppers: &[f64],
    omitted_value_abs_max: f64,
    kept_value_abs_max: f64,
) -> Result<f64> {
    ensure!(!kept_scores.is_empty(), "atención conservada vacía");
    ensure!(
        kept_scores.iter().all(|score| score.is_finite()),
        "score conservado no finito"
    );
    ensure!(
        omitted_score_uppers.iter().all(|score| score.is_finite()),
        "cota omitida no finita"
    );
    ensure!(
        omitted_value_abs_max.is_finite() && omitted_value_abs_max >= 0.0,
        "V omitido inválido"
    );
    ensure!(
        kept_value_abs_max.is_finite() && kept_value_abs_max >= 0.0,
        "V conservado inválido"
    );
    if omitted_score_uppers.is_empty() {
        return Ok(0.0);
    }
    let reference = kept_scores
        .iter()
        .copied()
        .chain(omitted_score_uppers.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let kept_mass = kept_scores
        .iter()
        .map(|score| (*score - reference).exp())
        .sum::<f64>();
    let omitted_mass = omitted_score_uppers
        .iter()
        .map(|score| (*score - reference).exp())
        .sum::<f64>();
    ensure!(
        kept_mass.is_finite() && omitted_mass.is_finite(),
        "masa no finita"
    );
    let omitted_fraction = omitted_mass / (kept_mass + omitted_mass);
    Ok(omitted_fraction * (omitted_value_abs_max + kept_value_abs_max))
}

/// Select the smallest per-head high-score prefixes whose exact omitted
/// softmax mass proves an output error no greater than `tolerance`, then take
/// their union.  `2 * alpha * max|V|` bounds the difference between selected
/// and full normalized attention.  This is the terminal refinement when
/// region boxes are too loose: it needs every K score, but not every V.
pub fn select_by_exact_mass(
    scores_by_head: &[Vec<f64>],
    value_abs_max: &[f64],
    tolerance: f64,
) -> Result<(Vec<usize>, Vec<f64>)> {
    ensure!(
        !scores_by_head.is_empty() && scores_by_head.len() == value_abs_max.len(),
        "geometría de masa exacta incompatible"
    );
    ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "tolerancia inválida"
    );
    let tokens = scores_by_head[0].len();
    ensure!(tokens > 0, "masa exacta sin tokens");
    ensure!(
        scores_by_head
            .iter()
            .all(|scores| scores.len() == tokens && scores.iter().all(|score| score.is_finite()))
            && value_abs_max
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        "masa exacta no finita"
    );
    let mut selected = BTreeSet::new();
    for (scores, value_max) in scores_by_head.iter().zip(value_abs_max) {
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let allowed_fraction = if *value_max == 0.0 {
            1.0
        } else {
            (tolerance / (2.0 * *value_max)).clamp(0.0, 1.0)
        };
        let mut ranked = weights.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let mut retained = 0.0_f64;
        for (token, weight) in ranked {
            selected.insert(token);
            retained += weight;
            if (total - retained).max(0.0) / total <= allowed_fraction {
                break;
            }
        }
    }
    let positions = selected.into_iter().collect::<Vec<_>>();
    let selected_set = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut bounds = Vec::with_capacity(scores_by_head.len());
    for (scores, value_max) in scores_by_head.iter().zip(value_abs_max) {
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0_f64;
        let mut omitted = 0.0_f64;
        for (token, score) in scores.iter().enumerate() {
            let weight = (*score - maximum).exp();
            total += weight;
            if !selected_set.contains(&token) {
                omitted += weight;
            }
        }
        bounds.push(2.0 * omitted / total * *value_max);
    }
    Ok((positions, bounds))
}

/// Select V tokens while preserving the complete softmax denominator. Unlike
/// [`select_by_exact_mass`], the retained probabilities are not renormalized:
/// only omitted `p_i * V_i` numerator terms disappear. The component-wise
/// output error is therefore bounded by `omitted_mass * max|V|`, without the
/// factor two required by renormalized sparse attention.
pub fn select_v_by_exact_contribution(
    scores_by_head: &[Vec<f64>],
    value_abs_max: &[f64],
    tolerance: f64,
) -> Result<(Vec<usize>, Vec<f64>)> {
    ensure!(
        !scores_by_head.is_empty() && scores_by_head.len() == value_abs_max.len(),
        "geometría de contribución V incompatible"
    );
    ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "tolerancia V inválida"
    );
    let tokens = scores_by_head[0].len();
    ensure!(tokens > 0, "contribución V sin tokens");
    ensure!(
        scores_by_head
            .iter()
            .all(|scores| scores.len() == tokens && scores.iter().all(|score| score.is_finite()))
            && value_abs_max
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        "contribución V no finita"
    );
    let mut selected = BTreeSet::new();
    for (scores, value_max) in scores_by_head.iter().zip(value_abs_max) {
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let allowed_fraction = if *value_max == 0.0 {
            1.0
        } else {
            (tolerance / *value_max).clamp(0.0, 1.0)
        };
        let mut ranked = weights.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let mut retained = 0.0_f64;
        for (token, weight) in ranked {
            selected.insert(token);
            retained += weight;
            if (total - retained).max(0.0) / total <= allowed_fraction {
                break;
            }
        }
    }
    let positions = selected.into_iter().collect::<Vec<_>>();
    let selected_set = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut bounds = Vec::with_capacity(scores_by_head.len());
    for (scores, value_max) in scores_by_head.iter().zip(value_abs_max) {
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0_f64;
        let mut omitted = 0.0_f64;
        for (token, score) in scores.iter().enumerate() {
            let weight = (*score - maximum).exp();
            total += weight;
            if !selected_set.contains(&token) {
                omitted += weight;
            }
        }
        bounds.push(omitted / total * *value_max);
    }
    Ok((positions, bounds))
}

/// Proves that every real value in `center ± per_head_error` rounds to the
/// same finite BF16 value as `center`. The interval must stay strictly inside
/// the two round-to-nearest midpoints; strictness deliberately rejects exact
/// tie cases instead of depending on the retained BF16 mantissa parity.
///
/// This is a sufficient output-quantization certificate, not by itself a
/// proof that a fused attention kernel computed `center`: callers must also
/// include conservative score/softmax/PV arithmetic error in each bound or
/// retain a canonical-kernel oracle.
pub fn bf16_rounding_cells_contain(
    center: &[f32],
    head_dim: usize,
    per_head_error: &[f64],
) -> bool {
    if head_dim == 0
        || per_head_error.is_empty()
        || center.len() != head_dim.saturating_mul(per_head_error.len())
        || per_head_error
            .iter()
            .any(|bound| !bound.is_finite() || *bound < 0.0)
    {
        return false;
    }
    let component_error = per_head_error
        .iter()
        .flat_map(|bound| std::iter::repeat_n(*bound, head_dim))
        .collect::<Vec<_>>();
    bf16_rounding_cells_contain_components(center, &component_error)
}

/// Component-wise form of [`bf16_rounding_cells_contain`].
///
/// `center` must be the *unrounded* value (for example an FP32 accumulation
/// of the candidate), never an output that was already rounded to BF16: a
/// BF16 value sits in the middle of its own cell, so centring the interval
/// on it hides on which side of a midpoint the real value lies.
pub fn bf16_rounding_cells_contain_components(center: &[f32], error: &[f64]) -> bool {
    !center.is_empty()
        && center.len() == error.len()
        && bf16_rounding_cell_count(center, error) == center.len()
}

/// Number of components whose interval `center ± error` lies strictly inside
/// a single BF16 round-to-nearest cell. Invalid geometry or a negative or
/// non-finite bound counts as zero certified components. This is the
/// per-component diagnostic behind
/// [`bf16_rounding_cells_contain_components`].
pub fn bf16_rounding_cell_count(center: &[f32], error: &[f64]) -> usize {
    if center.len() != error.len() || error.iter().any(|bound| !bound.is_finite() || *bound < 0.0) {
        return 0;
    }
    center
        .iter()
        .zip(error)
        .filter(|&(&value, &error)| bf16_rounding_cell_contains(value, error))
        .count()
}

/// Proves that every real value in `value ± error` rounds to the same finite
/// BF16 value as `value` (strictly inside both midpoints).
pub fn bf16_rounding_cell_contains(value: f32, error: f64) -> bool {
    if !value.is_finite() || !error.is_finite() || error < 0.0 {
        return false;
    }
    let rounded = bf16::from_f32(value);
    let bits = rounded.to_bits();
    let rounded_value = f64::from(rounded.to_f32());
    let (previous, next) = if bits == 0 {
        (
            -f64::from(bf16::from_bits(1).to_f32()),
            f64::from(bf16::from_bits(1).to_f32()),
        )
    } else if bits == 0x8000 {
        (
            f64::from(bf16::from_bits(0x8001).to_f32()),
            f64::from(bf16::from_bits(1).to_f32()),
        )
    } else if bits & 0x8000 == 0 {
        if bits >= 0x7f7f {
            return false;
        }
        (
            f64::from(bf16::from_bits(bits - 1).to_f32()),
            f64::from(bf16::from_bits(bits + 1).to_f32()),
        )
    } else {
        if bits >= 0xff7f {
            return false;
        }
        (
            f64::from(bf16::from_bits(bits + 1).to_f32()),
            f64::from(bf16::from_bits(bits - 1).to_f32()),
        )
    };
    let lower_midpoint = (previous + rounded_value) * 0.5;
    let upper_midpoint = (rounded_value + next) * 0.5;
    let value = f64::from(value);
    value - error > lower_midpoint && value + error < upper_midpoint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_mass_selection_is_variable_and_respects_every_head_bound() {
        let (positions, bounds) = select_by_exact_mass(
            &[
                vec![12.0, 1.0, 0.0, -1.0, -2.0],
                vec![-2.0, -1.0, 0.0, 1.0, 12.0],
            ],
            &[3.0, 4.0],
            1e-3,
        )
        .unwrap();
        assert_eq!(positions, vec![0, 4]);
        assert!(certified(&bounds, 1e-3));

        let (diffuse, diffuse_bounds) = select_by_exact_mass(&[vec![0.0; 7]], &[2.0], 0.6).unwrap();
        assert!(diffuse.len() > positions.len());
        assert!(certified(&diffuse_bounds, 0.6));
    }

    #[test]
    fn omitted_error_keeps_softmax_mass_and_value_magnitude_separate() {
        let error = omitted_error_bound(&[4.0], &[0.0], 3.0, 2.0).unwrap();
        let expected = (1.0 / (1.0 + 4.0_f64.exp())) * 5.0;
        assert!((error - expected).abs() < 1e-12);
        assert_eq!(omitted_error_bound(&[4.0], &[], 3.0, 2.0).unwrap(), 0.0);
    }

    #[test]
    fn full_denominator_v_selection_uses_single_omitted_mass_bound() {
        let scores = [vec![12.0, 1.0, 0.0, -1.0, -2.0]];
        let values = [3.0];
        let (renormalized, renormalized_bounds) =
            select_by_exact_mass(&scores, &values, 1e-3).unwrap();
        let (full_denominator, full_denominator_bounds) =
            select_v_by_exact_contribution(&scores, &values, 1e-3).unwrap();
        assert!(full_denominator.len() <= renormalized.len());
        assert!(certified(&full_denominator_bounds, 1e-3));
        assert!(certified(&renormalized_bounds, 1e-3));
        assert!(full_denominator_bounds[0] <= renormalized_bounds[0]);
    }

    #[test]
    fn bf16_rounding_cell_certificate_accepts_interior_and_rejects_boundary() {
        let one = bf16::from_f32(1.0).to_f32();
        let bits = bf16::from_f32(1.0).to_bits();
        let previous = bf16::from_bits(bits - 1).to_f32();
        let next = bf16::from_bits(bits + 1).to_f32();
        let radius = f64::from((one - previous).min(next - one)) * 0.5;
        assert!(bf16_rounding_cells_contain(&[1.0], 1, &[radius * 0.99]));
        assert!(bf16_rounding_cells_contain(&[-1.0], 1, &[0.001]));
        assert!(!bf16_rounding_cells_contain(&[1.0], 1, &[radius]));
        assert!(!bf16_rounding_cells_contain(&[f32::NAN], 1, &[0.0]));
    }

    #[test]
    fn bf16_rounding_cell_rejects_unrounded_center_next_to_midpoint() {
        // Midpoint between BF16 1.0 and its successor 1.0078125.
        let midpoint = 1.0 + 2_f64.powi(-8);
        // The true (FP32) value lies just above the midpoint, so dense BF16
        // rounds it up to 1.0078125. Centring on the already rounded value
        // (1.0 or 1.0078125) with a 3e-4 bound wrongly "certifies".
        let unrounded = (midpoint + 1e-4) as f32;
        assert_eq!(bf16::from_f32(unrounded).to_f32(), 1.0078125);
        assert!(bf16_rounding_cells_contain_components(&[1.0], &[3e-4]));
        assert!(!bf16_rounding_cells_contain_components(
            &[unrounded],
            &[3e-4]
        ));
        // The same bound certifies when the unrounded value is far from any
        // midpoint.
        assert!(bf16_rounding_cells_contain_components(&[1.002], &[3e-4]));
    }

    #[test]
    fn bf16_rounding_cell_count_reports_partial_components() {
        let center = [1.002_f32, 0.0, 0.25];
        let error = [3e-4, 1e-12, 0.0];
        // Exact zero with a non-subnormal bound cannot stay in its cell.
        assert_eq!(bf16_rounding_cell_count(&center, &error), 2);
        assert!(!bf16_rounding_cells_contain_components(&center, &error));
        assert!(bf16_rounding_cell_contains(0.0, 0.0));
        assert_eq!(bf16_rounding_cell_count(&center, &error[..2]), 0);
        assert!(!bf16_rounding_cells_contain_components(&[], &[]));
    }
}
