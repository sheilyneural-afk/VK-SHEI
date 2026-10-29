//! Exact omitted-mass certificate for query-adaptive attention views.

use anyhow::{Result, ensure};
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
}
