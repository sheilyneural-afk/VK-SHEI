//! Certified MICRO metadata / region pruning for the existing GPU LOD.
//!
//! The ORIGINAL CPU ParallelContextAxis supplies content-defined regions,
//! 4-bit LSH seeds and *communicating_dimensions* micro metadata. This
//! adjunct indexes exact BF16 K minima/maxima by those ORIGINAL region IDs.
//! It never uses a fingerprint, relevance or a 4-bit seal as the authority
//! for omission: only a conservative omitted-softmax-mass upper bound
//! authorizes avoiding real GPU Q·K and verified BF16 V retrieval.
//!
//! IMPORTANT: epsilon=0 ALWAYS uses the original full exact GPU route;
//! sparse selection is possible only at explicitly positive tolerances.
//! If region boxes are loose, the planner widens or returns None and the
//! original full-QK/ColdStore path remains authoritative.

use crate::{adaptive_extent::ParallelContextAxis, portable::AttentionGeometry};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::collections::{BTreeMap, BTreeSet};

// Memory guard. If the ORIGINAL CPU axis becomes excessively fragmented,
// release all auxiliary GPU micro boxes and use original full QK/ColdStore.
const MAX_MICRO_REGIONS_PER_LAYER: usize = 512;
const MICRO_TILE_TOKENS: usize = 8;

// (Original state ID, exact BF16 key score [lower,upper] per Q head,
// number of tokens in the original CPU state).
type RegionScoreBoxes = Vec<(usize, Vec<(f64, f64)>, usize)>;

#[derive(Debug, Clone)]
struct KeyRegion {
    state_id: usize,
    first: usize,
    end: usize,
    minimum: Vec<f32>, // original BF16 K converted exactly to f32
    maximum: Vec<f32>,
    sum: Vec<f64>,
}

impl KeyRegion {
    fn add(&mut self, position: usize, key: &[f32]) -> Result<()> {
        ensure!(
            self.end == position,
            "micro K region token ranges not contiguous"
        );
        ensure!(
            self.minimum.len() == key.len(),
            "micro K region shape mismatch"
        );
        for (i, &value) in key.iter().enumerate() {
            self.minimum[i] = self.minimum[i].min(value);
            self.maximum[i] = self.maximum[i].max(value);
            self.sum[i] += f64::from(value);
        }
        self.end += 1;
        Ok(())
    }

    fn can_extend(&self, position: usize, state_id: usize) -> bool {
        self.state_id == state_id && self.end == position && self.tokens() < MICRO_TILE_TOKENS
    }

    fn tokens(&self) -> usize {
        self.end - self.first
    }
}

#[derive(Debug, Clone)]
pub struct MicroRegionPlan {
    pub positions: Vec<usize>,
    pub error_bounds: Vec<f64>,
    pub selected_state_count: usize,
    pub omitted_state_count: usize,
    pub lsh_seed_regions: usize,
    pub linked_seed_regions: usize,
    pub avoided_k_dot_products: usize,
    pub score_upper_margin_max: f64,
}

/// Compact exact-key bounding boxes for the ORIGINAL CPU region IDs.
/// No new token segmentation or model-independent LSH is introduced.
#[derive(Debug)]
pub struct MicroKeyRegions {
    geometry: AttentionGeometry,
    layers: Vec<Vec<KeyRegion>>,
    total_bf16_values_indexed: usize,
    disabled_due_to_fragmentation: bool,
}

impl MicroKeyRegions {
    pub fn new(geometry: AttentionGeometry, num_layers: usize) -> Result<Self> {
        ensure!(num_layers > 0, "micro-key region layers cannot be empty");
        ensure!(
            geometry.query_heads > 0
                && geometry.kv_heads > 0
                && geometry.head_dim > 0
                && geometry.query_heads.is_multiple_of(geometry.kv_heads),
            "micro-key region attention geometry invalid"
        );
        Ok(Self {
            geometry,
            layers: (0..num_layers).map(|_| Vec::new()).collect(),
            total_bf16_values_indexed: 0,
            disabled_due_to_fragmentation: false,
        })
    }

    pub fn total_values_indexed(&self) -> usize {
        self.total_bf16_values_indexed
    }
    pub fn disabled_due_to_fragmentation(&self) -> bool {
        self.disabled_due_to_fragmentation
    }
    pub fn region_bounds_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|r| {
                (r.minimum.len() + r.maximum.len()) * std::mem::size_of::<f32>()
                    + r.sum.len() * std::mem::size_of::<f64>()
                    + 3 * std::mem::size_of::<usize>()
            })
            .sum()
    }

    pub fn layer_regions(&self, layer: usize) -> usize {
        self.layers.get(layer).map_or(0, Vec::len)
    }

    pub fn observe(
        &mut self,
        layer: usize,
        start: usize,
        original_k_bf16: &[u8],
        canonical_axis: &ParallelContextAxis,
    ) -> Result<()> {
        let token_bytes = self
            .geometry
            .kv_elements_per_token()
            .checked_mul(2)
            .context("micro K token-byte shape overflow")?;
        ensure!(
            !original_k_bf16.is_empty() && original_k_bf16.len().is_multiple_of(token_bytes),
            "micro K BF16 input not aligned to whole KV tokens"
        );
        let count = original_k_bf16.len() / token_bytes;
        if self.disabled_due_to_fragmentation {
            return Ok(());
        }
        ensure!(
            start
                .checked_add(count)
                .is_some_and(|end| end <= canonical_axis.token_len()),
            "micro K region would exceed canonical ParallelContextAxis"
        );
        let layers = self
            .layers
            .get_mut(layer)
            .context("micro region layer missing")?;
        // Validate complete input before updating region metadata.
        let mut parsed = Vec::with_capacity(count);
        for token_bytes in original_k_bf16.chunks_exact(token_bytes) {
            let values = token_bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect::<Vec<_>>();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "micro-key bound rejects NaN/Inf"
            );
            parsed.push(values);
        }
        // The original CPU region ranges are sorted. Walk once rather
        // than linearly searching every extent for each BF16 token.
        let regions = canonical_axis.regions();
        let mut range_idx = regions.partition_point(|r| r.token_end <= start);
        for (index, values) in parsed.iter().enumerate() {
            let position = start + index;
            while range_idx < regions.len() && position >= regions[range_idx].token_end {
                range_idx += 1;
            }
            let region = regions
                .get(range_idx)
                .context("canonical CPU region missing for exact micro K")?;
            ensure!(
                region.token_start <= position && position < region.token_end,
                "canonical CPU region/token mismatch"
            );
            let state = region.state_id;
            let extend = layers
                .last()
                .is_some_and(|current| current.can_extend(position, state));
            if extend {
                layers
                    .last_mut()
                    .expect("tile present")
                    .add(position, values)?;
            } else {
                if layers.len() >= MAX_MICRO_REGIONS_PER_LAYER {
                    self.disabled_due_to_fragmentation = true;
                    self.layers.iter_mut().for_each(Vec::clear);
                    self.total_bf16_values_indexed = 0;
                    return Ok(());
                }
                layers.push(KeyRegion {
                    state_id: state,
                    first: position,
                    end: position + 1,
                    minimum: values.clone(),
                    maximum: values.clone(),
                    sum: values.iter().map(|value| f64::from(*value)).collect(),
                });
            }
        }
        self.total_bf16_values_indexed += count * self.geometry.kv_elements_per_token();
        Ok(())
    }

    /// Compact n-dimensional boxes yield upper/lower QK score bounds per
    /// token and per query head. We add a generous error margin for
    /// GPU FP32 FMA and rsqrt so that scalar f64 bounds do not understate
    /// scores produced by the original GPU kernel. This is an
    /// engineering-conservative bound for finite BF16 Q/K, not a claim
    /// about arbitrary GPU drivers or transcendental implementations.
    fn score_box(&self, query: &[f32], region: &KeyRegion, qh: usize) -> Result<(f64, f64, f64)> {
        let dim = self.geometry.head_dim;
        let group = self.geometry.query_heads / self.geometry.kv_heads;
        let kvhead = qh / group;
        let qstart = qh * dim;
        let kstart = kvhead * dim;
        let mut lower = 0.0f64;
        let mut upper = 0.0f64;
        let mut magnitude = 0.0f64;
        for d in 0..dim {
            let q = f64::from(query[qstart + d]);
            let min = f64::from(region.minimum[kstart + d]);
            let max = f64::from(region.maximum[kstart + d]);
            let (lo, hi) = if q >= 0.0 {
                (q * min, q * max)
            } else {
                (q * max, q * min)
            };
            lower += lo;
            upper += hi;
            magnitude += (q * min).abs().max((q * max).abs());
        }
        let scale = (dim as f64).sqrt().recip();
        // BF16 multiplication operands are exact f32; OpenCL computes a
        // sequence of f32 FMA, then FP32 rsqrt. 0.001 + 0.1% of the
        // total absolute dot magnitude is conservative for <=8192 dims.
        let margin = 1e-3 + magnitude * scale * 1e-3;
        let aabb_low = lower * scale - margin;
        let aabb_high = upper * scale + margin;
        let count = region.tokens() as f64;
        let mut center_dot = 0.0f64;
        let mut query_norm = 0.0f64;
        let mut radius_sq = 0.0f64;
        for d in 0..dim {
            let q = f64::from(query[qstart + d]);
            let mean = region.sum[kstart + d] / count;
            center_dot += q * mean;
            query_norm += q * q;
            let left = mean - f64::from(region.minimum[kstart + d]);
            let right = f64::from(region.maximum[kstart + d]) - mean;
            radius_sq += left.max(right).max(0.0).powi(2);
        }
        let radius = radius_sq.sqrt();
        let query_norm = query_norm.sqrt();
        let sphere_low = (center_dot - query_norm * radius) * scale - margin;
        let sphere_high = (center_dot + query_norm * radius) * scale + margin;
        let low = aabb_low.max(sphere_low);
        let high = aabb_high.min(sphere_high).max(low);
        ensure!(
            low.is_finite() && high.is_finite() && margin.is_finite() && low <= high,
            "micro K regional score bounds nonfinite"
        );
        Ok((low, high, margin))
    }

    /// Bound the whole-head softmax mass potentially omitted by a
    /// region subset. This never consults 4-bit approximations for a
    /// mathematical exclusion decision.
    fn error_bounds(
        &self,
        boxes: &RegionScoreBoxes,
        selected: &BTreeSet<usize>,
        value_abs_max: &[f64],
    ) -> Result<Vec<f64>> {
        let nheads = self.geometry.query_heads;
        let group = nheads / self.geometry.kv_heads;
        ensure!(
            value_abs_max.len() == self.geometry.kv_heads,
            "micro V bound shape invalid"
        );
        let mut bounds = Vec::with_capacity(nheads);
        for h in 0..nheads {
            let reference = boxes
                .iter()
                .map(|(_, values, _)| values[h].1)
                .fold(f64::NEG_INFINITY, f64::max);
            ensure!(reference.is_finite(), "micro softmax reference invalid");
            let mut kept_lower = 0.0f64;
            let mut omitted_upper = 0.0f64;
            for (id, scores, count) in boxes {
                let term = if selected.contains(id) {
                    (scores[h].0 - reference).exp()
                } else {
                    (scores[h].1 - reference).exp()
                } * (*count as f64);
                if selected.contains(id) {
                    kept_lower += term;
                } else {
                    omitted_upper += term;
                }
            }
            let v = value_abs_max[h / group];
            ensure!(
                v.is_finite() && v >= 0.0 && kept_lower.is_finite() && omitted_upper.is_finite(),
                "micro score/value mass nonfinite"
            );
            let fraction = if omitted_upper == 0.0 {
                0.0
            } else if kept_lower == 0.0 {
                1.0
            } else {
                omitted_upper / (omitted_upper + kept_lower)
            };
            bounds.push(2.0 * v * fraction);
        }
        Ok(bounds)
    }

    /// The LSH seeds and the original micro-metadata graph are used
    /// ONLY TO PRIORITIZE which original regions are retained first.
    /// All pruning is gated by an exact BF16-key region-box mass bound.
    pub fn plan(
        &self,
        layer: usize,
        query_bf16: &[u8],
        axis: &ParallelContextAxis,
        seed_tokens: &BTreeSet<usize>,
        value_abs_max: &[f64],
        epsilon: f64,
    ) -> Result<Option<MicroRegionPlan>> {
        ensure!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon <= 0.1,
            "micro region sparse path requires positive epsilon <=0.1"
        );
        let nheads = self.geometry.query_heads;
        let elements = self.geometry.query_elements();
        ensure!(
            query_bf16.len() == elements * 2,
            "micro region query BF16 shape mismatch"
        );
        let query = query_bf16
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
            .collect::<Vec<_>>();
        ensure!(
            query.iter().all(|v| v.is_finite()),
            "nonfinite micro region query"
        );
        if self.disabled_due_to_fragmentation {
            return Ok(None);
        }
        let tiles = self
            .layers
            .get(layer)
            .context("micro region layer missing")?;
        if tiles.len() < 2 {
            return Ok(None);
        }
        let covered = tiles.iter().map(KeyRegion::tokens).sum::<usize>();
        if covered != axis.token_len() {
            // E.g. a new decode K token is still pending ColdStore append.
            return Ok(None);
        }
        let mut upper_lower: RegionScoreBoxes = Vec::with_capacity(tiles.len());
        let mut max_margin = 0.0f64;
        for (tile_id, reg) in tiles.iter().enumerate() {
            let mut values = Vec::with_capacity(nheads);
            for qh in 0..nheads {
                let (low, high, margin) = self.score_box(&query, reg, qh)?;
                values.push((low, high));
                max_margin = max_margin.max(margin);
            }
            upper_lower.push((tile_id, values, reg.tokens()));
        }

        // Exactly the original CPU semantic dimension ids and links:
        // no alternative index or segmentation is invented here.
        let seed_states = seed_tokens
            .iter()
            .filter_map(|&t| axis.state_id_for_token(t))
            .collect::<BTreeSet<_>>();
        let mut linked = BTreeSet::new();
        let mut linked_quality = BTreeMap::<usize, f32>::new();
        for &state in &seed_states {
            if let Some(extent) = axis.extents().get(state) {
                for (&next, &quality) in extent
                    .micro_metadata
                    .communicating_dimensions
                    .iter()
                    .zip(extent.micro_metadata.link_quality.iter())
                {
                    if quality >= 0.80
                        && axis
                            .extents()
                            .get(next)
                            .is_some_and(|peer| peer.micro_metadata.coherence >= 0.12)
                    {
                        linked.insert(next);
                        let entry = linked_quality.entry(next).or_insert(0.0);
                        *entry = (*entry).max(quality * extent.micro_metadata.coherence);
                    }
                }
            }
        }
        // Always include the latest original state (recency is a mandatory
        // recall safeguard). Additional tiles are chosen by the upper mass,
        // with measured micro-metadata only as an ORDERING bias.
        let latest_state = tiles.last().context("micro index has no regions")?.state_id;
        let mut selected = tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| tile.state_id == latest_state)
            .map(|(id, _)| id)
            .collect::<BTreeSet<_>>();
        let mut error = self.error_bounds(&upper_lower, &selected, value_abs_max)?;
        let initially_linked = linked.len();
        if !error.iter().all(|&v| v <= epsilon) {
            let mut candidates = upper_lower
                .iter()
                .filter(|(id, _, _)| !selected.contains(id))
                .map(|(id, scores, count)| {
                    let state = tiles[*id].state_id;
                    let upper = scores
                        .iter()
                        .map(|score| score.1)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let coherence = axis
                        .extents()
                        .get(state)
                        .map_or(0.0, |extent| extent.micro_metadata.coherence);
                    let boost = if seed_states.contains(&state) {
                        0.25 * f64::from(coherence.max(0.0))
                    } else if let Some(quality) = linked_quality.get(&state) {
                        0.15 * f64::from((*quality).max(0.0))
                    } else {
                        0.0
                    };
                    (*id, upper + (*count as f64).ln() + boost)
                })
                .collect::<Vec<_>>();
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (id, _) in candidates {
                selected.insert(id);
                error = self.error_bounds(&upper_lower, &selected, value_abs_max)?;
                if error.iter().all(|&v| v <= epsilon) {
                    break;
                }
            }
        }
        if !error.iter().all(|&v| v <= epsilon) {
            // Invalid/conservative box result cannot authorize pruning.
            return Ok(None);
        }
        let mut positions = Vec::new();
        for id in &selected {
            let region = &tiles[*id];
            positions.extend(region.first..region.end);
        }
        positions.sort_unstable();
        positions.dedup();
        ensure!(
            positions.len() <= covered,
            "micro region selection has invalid tokens"
        );
        let omitted = covered - positions.len();
        // Don't run multiple sparse GPU kernels to skip only a handful
        // of tokens; use original full-QK if benefit is insignificant.
        if omitted < 8 || omitted * 4 < covered {
            return Ok(None);
        }
        let selected_states = selected
            .iter()
            .map(|id| tiles[*id].state_id)
            .collect::<BTreeSet<_>>();
        let all_states = tiles
            .iter()
            .map(|tile| tile.state_id)
            .collect::<BTreeSet<_>>();
        Ok(Some(MicroRegionPlan {
            positions,
            error_bounds: error,
            selected_state_count: selected_states.len(),
            omitted_state_count: all_states.len() - selected_states.len(),
            lsh_seed_regions: seed_states.len(),
            linked_seed_regions: initially_linked,
            avoided_k_dot_products: omitted * nheads,
            score_upper_margin_max: max_margin,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_extent::AxisPolicy;
    fn bits(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
            .collect()
    }
    #[test]
    fn regional_micro_certificate_prunes_only_reliably_separated_original_states() {
        let geom = AttentionGeometry::new(2, 1, 4).unwrap();
        let mut index = MicroKeyRegions::new(geom, 1).unwrap();
        let mut axis = ParallelContextAxis::new(4, AxisPolicy::default()).unwrap();
        let keys = (0..32)
            .flat_map(|i| {
                if i < 16 {
                    [-20.0, 0.0, 0.0, 0.0]
                } else {
                    [20.0, 0.0, 0.0, 0.0]
                }
            })
            .collect::<Vec<_>>();
        for token in 0..32 {
            axis.observe(&keys[token * 4..token * 4 + 4]).unwrap();
        }
        assert_eq!(
            axis.regions().len(),
            2,
            "original CPU axis must detect semantic boundary"
        );
        index.observe(0, 0, &bits(&keys), &axis).unwrap();
        let q = bits(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let selected = index
            .plan(0, &q, &axis, &BTreeSet::new(), &[1.0], 0.001)
            .unwrap()
            .expect("separated BF16 states must be certifiable");
        assert_eq!(selected.positions, (16..32).collect::<Vec<_>>());
        assert_eq!(selected.avoided_k_dot_products, 16 * 2);
        assert!(selected.error_bounds.iter().all(|&b| b <= 0.001));
    }

    #[test]
    fn ambiguous_regions_fail_closed_to_full_scoring() {
        let geom = AttentionGeometry::new(2, 1, 4).unwrap();
        let mut axis = ParallelContextAxis::new(4, AxisPolicy::default()).unwrap();
        let mut idx = MicroKeyRegions::new(geom, 1).unwrap();
        let k = (0..32)
            .flat_map(|i| {
                if i < 16 {
                    [1.0, 0., 0., 0.]
                } else {
                    [-1.0, 0., 0., 0.]
                }
            })
            .collect::<Vec<_>>();
        for token in 0..32 {
            axis.observe(&k[token * 4..token * 4 + 4]).unwrap();
        }
        idx.observe(0, 0, &bits(&k), &axis).unwrap();
        let q = bits(&[0., 0., 0., 0., 0., 0., 0., 0.]);
        assert!(
            idx.plan(0, &q, &axis, &BTreeSet::new(), &[1.0], 0.001)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn metadata_never_authorizes_epsilon_zero_or_nan() {
        let geom = AttentionGeometry::new(2, 1, 4).unwrap();
        let mut idx = MicroKeyRegions::new(geom, 1).unwrap();
        let mut axis = ParallelContextAxis::new(4, AxisPolicy::default()).unwrap();
        axis.observe(&[0., 0., 0., 0.]).unwrap();
        assert!(
            idx.observe(0, 0, &bits(&[f32::NAN, 0., 0., 0.]), &axis)
                .is_err()
        );
        assert_eq!(idx.total_values_indexed(), 0);
        idx.observe(0, 0, &bits(&[0., 0., 0., 0.]), &axis).unwrap();
        assert!(
            idx.plan(0, &bits(&[0.; 8]), &axis, &BTreeSet::new(), &[1.0], 0.0)
                .is_err()
        );
    }
}
