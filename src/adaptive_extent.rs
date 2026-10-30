//! Reversible, content-defined coordinate axis over the token stream.
//!
//! This is deliberately not a second tokenizer. Tokens remain configured-model's
//! compute coordinates. The axis observes a compact signal already produced
//! during inference and creates semantic extents when that signal changes.
//! Query-time dimensional views cover extents; they are not physical KV
//! allocation units. Every view retains exact half-open token ranges.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AxisPolicy {
    /// Number of novelty observations required before an adaptive boundary.
    pub warmup_observations: usize,
    /// Boundary threshold: running novelty mean plus this many deviations.
    pub novelty_sigma: f32,
    /// Optional operational fuse. It is not a catalogue/page size: it merely
    /// prevents an unbounded open extent when the signal is constant forever.
    pub open_extent_safety_limit: Option<usize>,
}

impl Default for AxisPolicy {
    fn default() -> Self {
        Self {
            warmup_observations: 8,
            novelty_sigma: 3.0,
            open_extent_safety_limit: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SemanticExtent {
    pub id: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub centroid: Vec<f32>,
    pub minimum: Vec<f32>,
    pub maximum: Vec<f32>,
    pub boundary_novelty: f32,
    /// LSH routing signature over the extent centroid. This is metadata only;
    /// exact token coordinates remain the authoritative recovery address.
    pub lsh_signatures: [u64; 4],
    /// Dynamic dimensional micro-metadata. This is a routing/relationship
    /// layer; it never replaces the exact BF16 token coordinates.
    pub micro_metadata: DimensionalMicroMetadata,
}

/// Dynamic metadata carried by one content-defined dimension. Unlike static
/// document metadata, these fields evolve as neighbouring dimensions are
/// closed and communicate only compact deltas through the axis.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DimensionalMicroMetadata {
    /// Four-bit fast validation/routing seal. It is deliberately not the
    /// integrity authority; the persisted checksum/hash remains authoritative.
    pub validation_4bit: u8,
    /// Monotonic version of the last relationship update.
    pub relation_version: u64,
    /// Logical neighbours connected by a measured semantic graph. Adjacent
    /// extents are not linked unless their centroids actually agree.
    #[serde(default)]
    pub communicating_dimensions: Vec<usize>,
    /// Cosine similarity in `[0, 1]` aligned with `communicating_dimensions`.
    #[serde(default)]
    pub link_quality: Vec<f32>,
    /// Previous extent in token order. This is structure, not semantics.
    #[serde(default)]
    pub temporal_predecessor: Option<usize>,
    /// Highest measured link quality for this dimension, bounded to [0, 1].
    #[serde(default)]
    pub propagated_relevance: f32,
    /// Tightness of the extent around its centroid, bounded to (0, 1].
    #[serde(default)]
    pub coherence: f32,
    /// Number of compact deltas received by this dimension.
    pub received_deltas: u32,
}

impl DimensionalMicroMetadata {
    fn new(validation_4bit: u8, coherence: f32) -> Self {
        Self {
            validation_4bit: validation_4bit & 0x0f,
            relation_version: 0,
            communicating_dimensions: Vec::new(),
            link_quality: Vec::new(),
            temporal_predecessor: None,
            propagated_relevance: 0.0,
            coherence: coherence.clamp(0.0, 1.0),
            received_deltas: 0,
        }
    }

    fn payload_bytes(&self) -> usize {
        1 + std::mem::size_of::<u64>()
            + self.communicating_dimensions.len() * std::mem::size_of::<usize>()
            + self.link_quality.len() * std::mem::size_of::<f32>()
            + std::mem::size_of::<Option<usize>>()
            + std::mem::size_of::<f32>() * 2
            + std::mem::size_of::<u32>()
    }
}

/// A logical state on the parallel axis. Its token interval is only the exact
/// compatibility address needed to recover model KV; it does not define the
/// state and has no fixed width. The final state may still be open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisRegion {
    pub state_id: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub open: bool,
}

impl SemanticExtent {
    pub fn tokens(&self) -> usize {
        self.token_end - self.token_start
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionalView {
    /// Exact token intervals needed to reconstruct this view.
    pub token_ranges: Vec<(usize, usize)>,
    pub token_count: usize,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct RunningNovelty {
    count: usize,
    mean: f64,
    m2: f64,
}

impl RunningNovelty {
    fn observe(&mut self, value: f32) {
        self.count += 1;
        let delta = f64::from(value) - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = f64::from(value) - self.mean;
        self.m2 += delta * delta2;
    }

    fn threshold(&self, sigma: f32) -> Option<f32> {
        if self.count < 2 {
            return None;
        }
        let variance = self.m2 / (self.count - 1) as f64;
        Some((self.mean + f64::from(sigma) * variance.sqrt()) as f32)
    }

    fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OpenExtent {
    token_start: usize,
    count: usize,
    sum: Vec<f64>,
    minimum: Vec<f32>,
    maximum: Vec<f32>,
}

impl OpenExtent {
    fn new(token_start: usize, signal: &[f32]) -> Self {
        Self {
            token_start,
            count: 1,
            sum: signal.iter().map(|value| f64::from(*value)).collect(),
            minimum: signal.to_vec(),
            maximum: signal.to_vec(),
        }
    }

    fn centroid(&self) -> Vec<f32> {
        self.sum
            .iter()
            .map(|sum| (*sum / self.count as f64) as f32)
            .collect()
    }

    fn push(&mut self, signal: &[f32]) {
        self.count += 1;
        for (index, value) in signal.iter().copied().enumerate() {
            self.sum[index] += f64::from(value);
            self.minimum[index] = self.minimum[index].min(value);
            self.maximum[index] = self.maximum[index].max(value);
        }
    }
}

/// Four-bit search coordinate. This is catalogue metadata only: it never
/// replaces the model's BF16 K/V values. Signed digits use the symmetric
/// range `[-7, 7]`; two coordinates share one byte and one scale restores the
/// local magnitude when scoring a query.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct FourBitFingerprint {
    packed: Vec<u8>,
    scale: f32,
    len: usize,
}

impl FourBitFingerprint {
    fn encode(values: &[f32]) -> Self {
        let maximum = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
        let mut packed = vec![0_u8; values.len().div_ceil(2)];
        for (index, value) in values.iter().copied().enumerate() {
            let digit = (value / scale).round_ties_even().clamp(-7.0, 7.0) as i8;
            let nibble = (digit + 7) as u8;
            packed[index / 2] |= nibble << ((index & 1) * 4);
        }
        Self {
            packed,
            scale,
            len: values.len(),
        }
    }

    fn dot(&self, query: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), self.len);
        query
            .iter()
            .copied()
            .enumerate()
            .map(|(index, query)| {
                let nibble = (self.packed[index / 2] >> ((index & 1) * 4)) & 0x0f;
                query * (nibble as i8 - 7) as f32 * self.scale
            })
            .sum()
    }

    fn values(&self) -> Vec<f32> {
        (0..self.len)
            .map(|index| {
                let nibble = (self.packed[index / 2] >> ((index & 1) * 4)) & 0x0f;
                (nibble as i8 - 7) as f32 * self.scale
            })
            .collect()
    }

    fn payload_bytes(&self) -> usize {
        self.packed.len() + std::mem::size_of::<f32>()
    }
}

/// Locality-sensitive candidate index over compact fingerprints. It is only a
/// routing structure: exact BF16 K/V and the attention certificate remain the
/// authorities after candidate expansion.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct LshIndex {
    projections: Vec<Vec<Vec<f32>>>,
    buckets: Vec<HashMap<u64, Vec<usize>>>,
}

impl LshIndex {
    fn new(dimension: usize, bits: usize) -> Self {
        let mut seed = 0x9e37_79b9_u64;
        let projections = (0..4)
            .map(|_| {
                (0..bits)
                    .map(|_| {
                        (0..dimension)
                            .map(|_| {
                                seed ^= seed << 7;
                                seed ^= seed >> 9;
                                if seed & 1 == 0 { -1.0 } else { 1.0 }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            projections,
            buckets: (0..4).map(|_| HashMap::new()).collect(),
        }
    }

    fn signatures(&self, values: &[f32]) -> [u64; 4] {
        std::array::from_fn(|table| {
            self.projections[table].iter().enumerate().fold(
                0_u64,
                |signature, (bit, projection)| {
                    let score = projection
                        .iter()
                        .zip(values)
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>();
                    signature | u64::from(score >= 0.0) << bit
                },
            )
        })
    }

    fn insert(&mut self, token: usize, fingerprint: &FourBitFingerprint) {
        for (table, signature) in self
            .signatures(&fingerprint.values())
            .into_iter()
            .enumerate()
        {
            self.buckets[table]
                .entry(signature)
                .or_default()
                .push(token);
        }
    }

    fn candidates(&self, query: &[f32], max_hamming: usize) -> Vec<usize> {
        let mut votes = HashMap::<usize, usize>::new();
        for (table, signature) in self.signatures(query).into_iter().enumerate() {
            let limit = max_hamming.min(self.projections[table].len());
            visit_hamming_masks(0, limit, 0, self.projections[table].len(), &mut |mask| {
                if let Some(tokens) = self.buckets[table].get(&(signature ^ mask)) {
                    for token in tokens {
                        *votes.entry(*token).or_default() += 1;
                    }
                }
            });
        }
        let mut candidates = votes
            .into_iter()
            .filter(|(_, votes)| *votes >= 2)
            .map(|(token, _)| token)
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }
}

/// Enumerate every distinct 24-bit mask with Hamming weight at most `limit`
/// without allocating a per-query mask vector. The visit order and candidate
/// votes are identical to the former materialized combination list.
fn visit_hamming_masks(
    bit: usize,
    remaining: usize,
    current: u64,
    width: usize,
    visit: &mut impl FnMut(u64),
) {
    visit(current);
    if remaining == 0 {
        return;
    }
    for next in bit..width {
        visit_hamming_masks(
            next + 1,
            remaining - 1,
            current | (1_u64 << next),
            width,
            visit,
        );
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParallelContextAxis {
    dimension: usize,
    kv_heads: usize,
    head_dim: usize,
    fingerprint_dimension: usize,
    policy: AxisPolicy,
    next_token: usize,
    centroid_surprise: RunningNovelty,
    temporal_surprise: RunningNovelty,
    previous_signal: Option<Vec<f32>>,
    open: Option<OpenExtent>,
    extents: Vec<SemanticExtent>,
    token_fingerprints: Vec<Vec<FourBitFingerprint>>,
    lsh: Vec<LshIndex>,
    lsh_queries: usize,
    lsh_candidates: usize,
    lsh_misses: usize,
    lsh_probe_two: usize,
    lsh_probe_four: usize,
    relation_version: u64,
}

impl ParallelContextAxis {
    pub fn new(dimension: usize, policy: AxisPolicy) -> Result<Self> {
        Self::new_partitioned(dimension, infer_kv_heads(dimension), policy)
    }

    pub fn new_partitioned(dimension: usize, kv_heads: usize, policy: AxisPolicy) -> Result<Self> {
        ensure!(dimension > 0, "dimensión paralela vacía");
        ensure!(
            kv_heads > 0 && dimension.is_multiple_of(kv_heads),
            "partición KV inválida"
        );
        ensure!(policy.warmup_observations > 0, "warmup vacío");
        ensure!(
            policy.novelty_sigma.is_finite() && policy.novelty_sigma >= 0.0,
            "novelty_sigma inválido"
        );
        ensure!(
            policy
                .open_extent_safety_limit
                .is_none_or(|limit| limit > 0),
            "límite de seguridad vacío"
        );
        let head_dim = dimension / kv_heads;
        let fingerprint_dimension = head_dim.min(32);
        Ok(Self {
            dimension,
            kv_heads,
            head_dim,
            fingerprint_dimension,
            policy,
            next_token: 0,
            centroid_surprise: RunningNovelty::default(),
            temporal_surprise: RunningNovelty::default(),
            previous_signal: None,
            open: None,
            extents: Vec::new(),
            token_fingerprints: Vec::new(),
            // One LSH table family per KV head. Mixing heads into a single
            // 16-d average made query routing miss because protected Q only
            // occupies the serving head. 16 directional bits stay dense
            // enough for Hamming-2/4 probes.
            lsh: (0..kv_heads)
                .map(|_| LshIndex::new(fingerprint_dimension, 16))
                .collect(),
            lsh_queries: 0,
            lsh_candidates: 0,
            lsh_misses: 0,
            lsh_probe_two: 0,
            lsh_probe_four: 0,
            relation_version: 0,
        })
    }

    /// Observe one token-aligned signal already available in the model.
    /// Returns the extent closed immediately before this token, if any.
    pub fn observe(&mut self, signal: &[f32]) -> Result<Option<SemanticExtent>> {
        self.validate_signal(signal)?;
        let fingerprints = self.encode_heads(signal);
        let token = self.token_fingerprints.len();
        for (head, fingerprint) in fingerprints.iter().enumerate() {
            self.lsh[head].insert(token, fingerprint);
        }
        self.token_fingerprints.push(fingerprints);
        let Some(open) = self.open.as_ref() else {
            self.open = Some(OpenExtent::new(self.next_token, signal));
            self.previous_signal = Some(signal.to_vec());
            self.next_token += 1;
            return Ok(None);
        };
        let centroid_surprise = signal_distance(signal, &open.centroid());
        let temporal_surprise = self
            .previous_signal
            .as_deref()
            .map_or(0.0, |previous| signal_distance(signal, previous));
        let centroid_boundary = self.centroid_surprise.count >= self.policy.warmup_observations
            && self
                .centroid_surprise
                .threshold(self.policy.novelty_sigma)
                .is_some_and(|threshold| centroid_surprise > threshold);
        let temporal_boundary = self.temporal_surprise.count >= self.policy.warmup_observations
            && self
                .temporal_surprise
                .threshold(self.policy.novelty_sigma)
                .is_some_and(|threshold| temporal_surprise > threshold);
        let adaptive_boundary = centroid_boundary || temporal_boundary;
        let safety_boundary = self
            .policy
            .open_extent_safety_limit
            .is_some_and(|limit| open.count >= limit);
        let closed = if adaptive_boundary || safety_boundary {
            let boundary_surprise = centroid_surprise.max(temporal_surprise);
            let closed = self.close_open(boundary_surprise).expect("open comprobado");
            self.open = Some(OpenExtent::new(self.next_token, signal));
            self.centroid_surprise.clear();
            self.temporal_surprise.clear();
            Some(closed)
        } else {
            self.open.as_mut().expect("open comprobado").push(signal);
            self.centroid_surprise.observe(centroid_surprise);
            self.temporal_surprise.observe(temporal_surprise);
            None
        };
        self.previous_signal = Some(signal.to_vec());
        self.next_token += 1;
        Ok(closed)
    }

    /// Configure the operational open-extent fuse before the first token is
    /// observed. Explicit diagnostics can request tighter certificates
    /// without changing the canonical semantic axis.
    pub fn set_open_extent_safety_limit_before_observation(
        &mut self,
        limit: Option<usize>,
    ) -> Result<()> {
        ensure!(
            self.next_token == 0 && self.open.is_none() && self.extents.is_empty(),
            "no se puede cambiar el fusible tras observar tokens"
        );
        ensure!(limit.is_none_or(|value| value > 0), "fusible vacío");
        self.policy.open_extent_safety_limit = limit;
        Ok(())
    }

    /// Close the current semantic extent at a lifecycle boundary such as the
    /// end of prefill. This does not pad or retokenize it.
    pub fn flush(&mut self) -> Option<SemanticExtent> {
        self.close_open(0.0)
    }

    pub fn extents(&self) -> &[SemanticExtent] {
        &self.extents
    }

    /// Return the communicating dimensions for routing a query seed. The
    /// returned slice is metadata only; callers must still certify any BF16
    /// region they decide to materialize.
    pub fn communicating_dimensions(&self, extent_id: usize) -> &[usize] {
        self.extents.get(extent_id).map_or(&[], |extent| {
            &extent.micro_metadata.communicating_dimensions
        })
    }

    /// Bytes used by the communicating-dimension metadata. This is separate
    /// from BF16 residency and makes the metadata cost auditable.
    pub fn micro_metadata_bytes(&self) -> usize {
        self.extents
            .iter()
            .map(|extent| extent.micro_metadata.payload_bytes())
            .sum()
    }

    pub fn token_len(&self) -> usize {
        self.next_token
    }

    pub fn lsh_metrics(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.lsh_queries,
            self.lsh_candidates,
            self.lsh_misses,
            self.lsh_probe_two,
            self.lsh_probe_four,
        )
    }

    /// Intersections between a compatibility-token interval and logical
    /// dimension states. State ids, not storage frame numbers, are the units
    /// used by retrieval and attention certification.
    pub fn partition_regions(&self, start: usize, end: usize) -> Result<Vec<AxisRegion>> {
        ensure!(
            start < end && end <= self.next_token,
            "intervalo de eje inválido"
        );
        Ok(self
            .regions()
            .into_iter()
            .filter_map(|region| {
                let token_start = start.max(region.token_start);
                let token_end = end.min(region.token_end);
                (token_start < token_end).then_some(AxisRegion {
                    token_start,
                    token_end,
                    ..region
                })
            })
            .collect())
    }

    pub fn regions(&self) -> Vec<AxisRegion> {
        let mut regions = self
            .extents
            .iter()
            .map(|extent| AxisRegion {
                state_id: extent.id,
                token_start: extent.token_start,
                token_end: extent.token_end,
                open: false,
            })
            .collect::<Vec<_>>();
        if let Some(open) = &self.open {
            regions.push(AxisRegion {
                state_id: self.extents.len(),
                token_start: open.token_start,
                token_end: self.next_token,
                open: true,
            });
        }
        regions
    }

    pub fn state_id_for_token(&self, token: usize) -> Option<usize> {
        if token >= self.next_token {
            return None;
        }
        self.extents
            .iter()
            .find(|extent| extent.token_start <= token && token < extent.token_end)
            .map(|extent| extent.id)
            .or_else(|| {
                self.open
                    .as_ref()
                    .filter(|open| open.token_start <= token)
                    .map(|_| self.extents.len())
            })
    }

    /// Stable payload size of the four-bit token catalogue. Allocator and
    /// `Vec` bookkeeping are excluded so the value can be compared directly
    /// with the former f32 payload on every platform.
    pub fn fingerprint_payload_bytes(&self) -> usize {
        self.token_fingerprints
            .iter()
            .flatten()
            .map(FourBitFingerprint::payload_bytes)
            .sum()
    }

    pub fn f32_fingerprint_reference_bytes(&self) -> usize {
        self.token_fingerprints.len()
            * self.kv_heads
            * self.fingerprint_dimension
            * std::mem::size_of::<f32>()
    }

    /// Select a query-dependent number of reversible coordinates. The count is
    /// derived from the entropy of the compact 4-bit score distribution rather
    /// than from a fixed token/page budget. A concentrated query returns a
    /// small candidate set; an ambiguous query deliberately expands toward the
    /// complete catalogue before exact BF16 certification.
    pub fn materialize_query_tokens_adaptive(
        &mut self,
        query: &[f32],
    ) -> Result<Vec<DimensionalView>> {
        self.validate_signal(query)?;
        ensure!(
            !self.token_fingerprints.is_empty(),
            "catálogo de tokens vacío"
        );
        let query_heads = self.encode_heads(query);
        self.lsh_queries += 1;
        let mut candidate_tokens = self.candidates_from_heads(&query_heads, 2);
        if !candidate_tokens.is_empty() {
            self.lsh_probe_two += 1;
        }
        if candidate_tokens.is_empty() {
            // A wider probe remains advisory LSH routing. If it also misses,
            // the canonical certificate continues without heuristic seeds.
            candidate_tokens = self.candidates_from_heads(&query_heads, 4);
            if !candidate_tokens.is_empty() {
                self.lsh_probe_four += 1;
            }
        }
        if candidate_tokens.is_empty() {
            self.lsh_misses += 1;
            // LSH is a prioritizer, not the recall authority. Turning a miss
            // into the complete catalogue used to seed every logical state
            // before branch-and-bound, making certified omission impossible.
            // An empty routing result is safe: the attention certificate will
            // widen from the mandatory recent state until its error bound
            // closes.
            return Ok(Vec::new());
        }
        self.lsh_candidates += candidate_tokens.len();
        let query_signatures = self.signatures_for_heads(&query_heads);
        let extent_ids = self
            .extents
            .iter()
            .filter(|extent| {
                query_signatures
                    .into_iter()
                    .zip(extent.lsh_signatures)
                    .filter(|(query, extent)| (query ^ extent).count_ones() <= 2)
                    .count()
                    >= 2
            })
            .map(|extent| extent.id)
            .collect::<BTreeSet<_>>();
        let candidate_tokens_before_extent_filter = candidate_tokens.clone();
        let mut scored = candidate_tokens
            .into_iter()
            .filter(|token| {
                extent_ids.is_empty()
                    || self.extents.iter().any(|extent| {
                        extent_ids.contains(&extent.id)
                            && extent.token_start <= *token
                            && *token < extent.token_end
                    })
            })
            .map(|token| (token, self.token_dot(token, &query_heads)))
            .collect::<Vec<_>>();
        if scored.is_empty() {
            scored = candidate_tokens_before_extent_filter
                .into_iter()
                .map(|token| (token, self.token_dot(token, &query_heads)))
                .collect();
        }
        let maximum = scored
            .iter()
            .map(|(_, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let mean = scored.iter().map(|(_, score)| *score).sum::<f32>() / scored.len() as f32;
        let variance = scored
            .iter()
            .map(|(_, score)| (*score - mean).powi(2))
            .sum::<f32>()
            / scored.len() as f32;
        let spread = variance.sqrt();
        let spread = if spread.is_finite() && spread > 0.0 {
            spread
        } else {
            1.0
        };
        let scale = scored
            .iter()
            .map(|(_, score)| ((*score - maximum) / spread).exp())
            .sum::<f32>();
        let entropy = if scale.is_finite() && scale > 0.0 {
            scored
                .iter()
                .map(|(_, score)| {
                    let probability = ((*score - maximum) / spread).exp() / scale;
                    if probability > 0.0 {
                        -f64::from(probability) * f64::from(probability.ln())
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        } else {
            (scored.len() as f64).ln()
        };
        let candidate_count = entropy.exp().ceil().clamp(1.0, scored.len() as f64) as usize;
        scored.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        let tokens = scored
            .into_iter()
            .take(candidate_count)
            .map(|(token, _)| token)
            .collect::<Vec<_>>();
        self.materialize_token_positions(&tokens)
    }

    pub fn materialize_token_positions(&self, positions: &[usize]) -> Result<Vec<DimensionalView>> {
        ensure!(
            positions
                .iter()
                .all(|token| *token < self.token_fingerprints.len()),
            "posición token inexistente"
        );
        let mut tokens = positions.to_vec();
        tokens.sort_unstable();
        tokens.dedup();
        let mut views: BTreeMap<Option<usize>, DimensionalView> = BTreeMap::new();
        for token in tokens {
            let extent_id = self
                .extents
                .iter()
                .find(|extent| extent.token_start <= token && token < extent.token_end)
                .map(|extent| extent.id);
            let view = views.entry(extent_id).or_insert_with(|| DimensionalView {
                token_ranges: Vec::new(),
                token_count: 0,
            });
            if view.token_ranges.last().map(|range| range.1) == Some(token) {
                view.token_ranges.last_mut().expect("range presente").1 += 1;
            } else {
                view.token_ranges.push((token, token + 1));
            }
            view.token_count += 1;
        }
        let mut views = views.into_values().collect::<Vec<_>>();
        views.sort_unstable_by_key(|view| view.token_ranges[0].0);
        Ok(views)
    }

    /// Exact compatibility bridge back to the model's token coordinates.
    pub fn expand_token_positions(view: &DimensionalView) -> Vec<usize> {
        view.token_ranges
            .iter()
            .flat_map(|(start, end)| *start..*end)
            .collect()
    }

    fn validate_signal(&self, signal: &[f32]) -> Result<()> {
        ensure!(
            signal.len() == self.dimension,
            "dimensión de señal incompatible"
        );
        ensure!(
            signal.iter().all(|value| value.is_finite()),
            "señal no finita"
        );
        Ok(())
    }

    fn encode_heads(&self, signal: &[f32]) -> Vec<FourBitFingerprint> {
        (0..self.kv_heads)
            .map(|head| {
                let start = head * self.head_dim;
                FourBitFingerprint::encode(&self.pool_head(&signal[start..start + self.head_dim]))
            })
            .collect()
    }

    fn pool_head(&self, head: &[f32]) -> Vec<f32> {
        let mut fingerprint = vec![0.0_f32; self.fingerprint_dimension];
        let mut counts = vec![0_usize; self.fingerprint_dimension];
        for (index, value) in head.iter().copied().enumerate() {
            let bucket = index * self.fingerprint_dimension / self.head_dim;
            fingerprint[bucket] += value;
            counts[bucket] += 1;
        }
        for (value, count) in fingerprint.iter_mut().zip(counts) {
            *value /= count as f32;
        }
        fingerprint
    }

    fn candidates_from_heads(
        &self,
        query_heads: &[FourBitFingerprint],
        max_hamming: usize,
    ) -> Vec<usize> {
        let mut tokens = BTreeSet::new();
        for (head, fingerprint) in query_heads.iter().enumerate() {
            if head_energy(fingerprint) <= 1e-8 {
                continue;
            }
            tokens.extend(self.lsh[head].candidates(&fingerprint.values(), max_hamming));
        }
        tokens.into_iter().collect()
    }

    fn signatures_for_heads(&self, heads: &[FourBitFingerprint]) -> [u64; 4] {
        let mut combined = [0_u64; 4];
        for (head, fingerprint) in heads.iter().enumerate() {
            if head_energy(fingerprint) <= 1e-8 {
                continue;
            }
            let signature = self.lsh[head].signatures(&fingerprint.values());
            for (index, value) in signature.into_iter().enumerate() {
                combined[index] ^= value.rotate_left((head as u32 * 5) % 16);
            }
        }
        combined
    }

    fn token_dot(&self, token: usize, query_heads: &[FourBitFingerprint]) -> f32 {
        self.token_fingerprints[token]
            .iter()
            .zip(query_heads)
            .map(|(token_head, query_head)| token_head.dot(&query_head.values()))
            .sum()
    }

    fn close_open(&mut self, boundary_novelty: f32) -> Option<SemanticExtent> {
        let open = self.open.take()?;
        let centroid = open.centroid();
        let head_fingerprints = self.encode_heads(&centroid);
        let lsh_signatures = self.signatures_for_heads(&head_fingerprints);
        let validation_4bit = head_fingerprints
            .iter()
            .flat_map(|fingerprint| fingerprint.packed.iter().copied())
            .fold(0_u8, |seal, byte| seal ^ byte)
            & 0x0f;
        let coherence = extent_coherence(&centroid, &open.minimum, &open.maximum);
        let extent = SemanticExtent {
            id: self.extents.len(),
            token_start: open.token_start,
            token_end: open.token_start + open.count,
            centroid,
            minimum: open.minimum,
            maximum: open.maximum,
            boundary_novelty,
            lsh_signatures,
            micro_metadata: DimensionalMicroMetadata::new(validation_4bit, coherence),
        };
        self.extents.push(extent.clone());
        self.propagate_dimensional_delta(extent.id);
        Some(extent)
    }

    /// Connect a newly closed dimension to related dimensions using only
    /// compact metadata. Links require a high cosine agreement of centroids
    /// and non-trivial extent coherence. Adjacent-but-orthogonal extents are
    /// recorded as temporal structure, not as semantic neighbours. The BF16
    /// extents are not read or copied here.
    fn propagate_dimensional_delta(&mut self, new_id: usize) {
        let new_centroid = self.extents[new_id].centroid.clone();
        let new_coherence = self.extents[new_id].micro_metadata.coherence;
        if new_id > 0 {
            self.extents[new_id].micro_metadata.temporal_predecessor = Some(new_id - 1);
        }
        let mut neighbours = Vec::new();
        for candidate in 0..new_id {
            let quality = cosine_similarity(&new_centroid, &self.extents[candidate].centroid);
            let other_coherence = self.extents[candidate].micro_metadata.coherence;
            if quality >= SEMANTIC_LINK_QUALITY
                && new_coherence >= MIN_EXTENT_COHERENCE
                && other_coherence >= MIN_EXTENT_COHERENCE
            {
                neighbours.push((candidate, quality));
            }
        }
        self.relation_version = self.relation_version.saturating_add(1);
        let version = self.relation_version;
        for &(neighbour, quality) in &neighbours {
            insert_quality_link(&mut self.extents[neighbour].micro_metadata, new_id, quality);
            self.extents[neighbour].micro_metadata.relation_version = version;
            self.extents[neighbour].micro_metadata.received_deltas = self.extents[neighbour]
                .micro_metadata
                .received_deltas
                .saturating_add(1);
        }
        let metadata = &mut self.extents[new_id].micro_metadata;
        metadata.communicating_dimensions = neighbours.iter().map(|(id, _)| *id).collect();
        metadata.link_quality = neighbours.iter().map(|(_, quality)| *quality).collect();
        metadata.relation_version = version;
        metadata.propagated_relevance = neighbours
            .iter()
            .map(|(_, quality)| *quality)
            .fold(0.0_f32, f32::max);
    }
}

const SEMANTIC_LINK_QUALITY: f32 = 0.80;
const MIN_EXTENT_COHERENCE: f32 = 0.12;

fn infer_kv_heads(dimension: usize) -> usize {
    for head_dim in [128, 64, 32] {
        if dimension.is_multiple_of(head_dim) {
            let heads = dimension / head_dim;
            if (1..=16).contains(&heads) {
                return heads;
            }
        }
    }
    1
}

fn head_energy(fingerprint: &FourBitFingerprint) -> f32 {
    fingerprint
        .values()
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn extent_coherence(centroid: &[f32], minimum: &[f32], maximum: &[f32]) -> f32 {
    let mut centroid_norm = 0.0_f64;
    let mut radius = 0.0_f64;
    for ((centroid, minimum), maximum) in centroid.iter().zip(minimum).zip(maximum) {
        let value = f64::from(*centroid);
        centroid_norm += value * value;
        let left = value - f64::from(*minimum);
        let right = f64::from(*maximum) - value;
        radius += left.max(right).max(0.0).powi(2);
    }
    let centroid_norm = centroid_norm.sqrt();
    let radius = radius.sqrt();
    if centroid_norm + radius == 0.0 {
        1.0
    } else {
        (centroid_norm / (centroid_norm + radius)) as f32
    }
}

fn insert_quality_link(metadata: &mut DimensionalMicroMetadata, neighbour: usize, quality: f32) {
    if let Some(index) = metadata
        .communicating_dimensions
        .iter()
        .position(|id| *id == neighbour)
    {
        metadata.link_quality[index] = metadata.link_quality[index].max(quality);
    } else {
        metadata.communicating_dimensions.push(neighbour);
        metadata.link_quality.push(quality);
    }
    metadata.propagated_relevance = metadata.propagated_relevance.max(quality);
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    (1.0 - cosine_distance(left, right) / 2.0).clamp(0.0, 1.0)
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        dot += f64::from(*left) * f64::from(*right);
        left_norm += f64::from(*left) * f64::from(*left);
        right_norm += f64::from(*right) * f64::from(*right);
    }
    if left_norm == 0.0 && right_norm == 0.0 {
        return 0.0;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 1.0;
    }
    (1.0 - dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 2.0) as f32
}

fn normalized_magnitude_distance(left: &[f32], right: &[f32]) -> f32 {
    let mut difference = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let delta = f64::from(*left) - f64::from(*right);
        difference += delta * delta;
        left_norm += f64::from(*left) * f64::from(*left);
        right_norm += f64::from(*right) * f64::from(*right);
    }
    let denominator = left_norm.sqrt() + right_norm.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        (difference.sqrt() / denominator).clamp(0.0, 1.0) as f32
    }
}

fn signal_distance(left: &[f32], right: &[f32]) -> f32 {
    cosine_distance(left, right).max(normalized_magnitude_distance(left, right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_fingerprint_packs_two_coordinates_per_byte() {
        let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
        let fingerprint = FourBitFingerprint::encode(&values);
        assert_eq!(fingerprint.packed.len(), 8);
        assert_eq!(fingerprint.payload_bytes(), 12);
        assert_eq!(values.len() * std::mem::size_of::<f32>(), 64);
    }

    #[test]
    fn lsh_routes_identical_fingerprints_without_global_scan() {
        let mut index = LshIndex::new(4, 64);
        let first = FourBitFingerprint::encode(&[1.0, 0.0, 0.0, 0.0]);
        let second = FourBitFingerprint::encode(&[-1.0, 0.0, 0.0, 0.0]);
        index.insert(7, &first);
        index.insert(11, &second);
        let candidates = index.candidates(&first.values(), 0);
        assert!(candidates.contains(&7));
        assert!(!candidates.contains(&11));
    }

    #[test]
    fn logical_states_are_independent_of_storage_frame_boundaries() {
        let mut axis = ParallelContextAxis::new(
            2,
            AxisPolicy {
                warmup_observations: 8,
                novelty_sigma: 3.0,
                open_extent_safety_limit: Some(3),
            },
        )
        .unwrap();
        for token in 0..8 {
            axis.observe(&[token as f32, 1.0]).unwrap();
        }

        // Imaginary storage frames [0,2) and [2,4) both contain state 0;
        // frame [2,4) also contains state 1. Neither frame defines the state.
        assert_eq!(
            axis.partition_regions(0, 2).unwrap(),
            vec![AxisRegion {
                state_id: 0,
                token_start: 0,
                token_end: 2,
                open: false,
            }]
        );
        assert_eq!(
            axis.partition_regions(2, 4).unwrap(),
            vec![
                AxisRegion {
                    state_id: 0,
                    token_start: 2,
                    token_end: 3,
                    open: false,
                },
                AxisRegion {
                    state_id: 1,
                    token_start: 3,
                    token_end: 4,
                    open: false,
                },
            ]
        );
        assert_eq!(axis.state_id_for_token(2), Some(0));
        assert_eq!(axis.state_id_for_token(3), Some(1));
    }

    #[test]
    fn four_bit_fingerprint_preserves_clear_search_ranking() {
        let query = [1.0, 0.0, -1.0, 0.5, 0.0];
        let relevant = FourBitFingerprint::encode(&[5.0, 0.0, -5.0, 2.5, 0.0]);
        let irrelevant = FourBitFingerprint::encode(&[-5.0, 0.0, 5.0, -2.5, 0.0]);
        assert!(relevant.dot(&query) > irrelevant.dot(&query));
        assert_eq!(relevant.packed.len(), 3);
    }

    #[test]
    fn hamming_probe_streams_every_mask_once_without_materializing_a_vector() {
        let mut masks = BTreeSet::new();
        visit_hamming_masks(0, 4, 0, 24, &mut |mask| {
            assert!(masks.insert(mask), "máscara Hamming duplicada");
            assert!(mask.count_ones() <= 4);
        });
        assert_eq!(masks.len(), 1 + 24 + 276 + 2_024 + 10_626);
    }

    fn axis() -> ParallelContextAxis {
        ParallelContextAxis::new(
            2,
            AxisPolicy {
                warmup_observations: 2,
                novelty_sigma: 1.0,
                open_extent_safety_limit: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn content_changes_create_variable_extents_not_fixed_pages() {
        let mut axis = axis();
        for _ in 0..11 {
            axis.observe(&[1.0, 0.0]).unwrap();
        }
        for _ in 0..3 {
            axis.observe(&[0.0, 1.0]).unwrap();
        }
        axis.flush();
        let lengths = axis
            .extents()
            .iter()
            .map(SemanticExtent::tokens)
            .collect::<Vec<_>>();
        assert_eq!(lengths.iter().sum::<usize>(), 14);
        assert!(lengths.len() >= 2);
        assert!(lengths.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn magnitude_only_change_creates_a_dimensional_boundary() {
        let mut axis = axis();
        for _ in 0..10 {
            axis.observe(&[1.0, 0.0]).unwrap();
        }
        for _ in 0..5 {
            axis.observe(&[8.0, 0.0]).unwrap();
        }
        axis.flush();
        assert_eq!(axis.extents().len(), 2);
        assert_eq!(axis.extents()[0].tokens(), 10);
        assert_eq!(axis.extents()[1].tokens(), 5);
        assert!(axis.extents()[0].boundary_novelty > 0.0);
    }

    #[test]
    fn state_local_surprise_detects_a_recurring_regime() {
        let mut axis = axis();
        for signal in [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]] {
            for _ in 0..10 {
                axis.observe(&signal).unwrap();
            }
        }
        axis.flush();
        assert_eq!(
            axis.extents()
                .iter()
                .map(SemanticExtent::tokens)
                .collect::<Vec<_>>(),
            vec![10, 10, 10]
        );
    }

    #[test]
    fn token_coordinate_roundtrip_is_exact_and_gapless() {
        let mut axis = axis();
        for token in 0..37 {
            let signal = if !(9..28).contains(&token) {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            };
            axis.observe(&signal).unwrap();
        }
        axis.flush();
        let partitions = axis.partition_regions(5, 33).unwrap();
        assert_eq!(partitions.first().unwrap().token_start, 5);
        assert_eq!(partitions.last().unwrap().token_end, 33);
        assert_eq!(
            partitions
                .iter()
                .map(|region| region.token_end - region.token_start)
                .sum::<usize>(),
            28
        );
        assert!(
            partitions
                .windows(2)
                .all(|pair| pair[0].token_end == pair[1].token_start)
        );
    }

    #[test]
    fn dimensional_micro_metadata_rejects_orthogonal_adjacent_garbage() {
        let mut axis = axis();
        for _ in 0..10 {
            axis.observe(&[1.0, 0.0]).unwrap();
        }
        axis.flush();
        for _ in 0..10 {
            axis.observe(&[0.0, 8.0]).unwrap();
        }
        axis.flush();
        assert_eq!(axis.extents().len(), 2);
        let first = &axis.extents()[0].micro_metadata;
        let second = &axis.extents()[1].micro_metadata;
        assert!(first.validation_4bit < 16);
        assert!(second.validation_4bit < 16);
        assert!(
            first.coherence > 0.9,
            "a constant regime must produce a tight extent"
        );
        assert!(second.coherence > 0.9);
        assert!(
            first.communicating_dimensions.is_empty(),
            "orthogonal extents must not emit a semantic link"
        );
        assert!(second.communicating_dimensions.is_empty());
        assert_eq!(second.temporal_predecessor, Some(0));
        assert_eq!(second.link_quality.len(), 0);
        assert_eq!(second.propagated_relevance, 0.0);
        assert!(axis.micro_metadata_bytes() > 0);
    }

    #[test]
    fn dimensional_micro_metadata_links_only_high_quality_semantic_neighbours() {
        let mut axis = axis();
        for _ in 0..10 {
            axis.observe(&[1.0, 0.0]).unwrap();
        }
        axis.flush();
        for _ in 0..10 {
            axis.observe(&[0.0, 8.0]).unwrap();
        }
        axis.flush();
        for _ in 0..10 {
            axis.observe(&[1.0, 0.0]).unwrap();
        }
        axis.flush();
        assert_eq!(axis.extents().len(), 3);
        let first = &axis.extents()[0].micro_metadata;
        let middle = &axis.extents()[1].micro_metadata;
        let last = &axis.extents()[2].micro_metadata;
        assert_eq!(first.communicating_dimensions, vec![2]);
        assert_eq!(last.communicating_dimensions, vec![0]);
        assert!(first.link_quality[0] >= 0.99);
        assert!(last.link_quality[0] >= 0.99);
        assert!(middle.communicating_dimensions.is_empty());
        assert_eq!(last.temporal_predecessor, Some(1));
        assert!(last.propagated_relevance >= 0.99);
    }

    #[test]
    fn per_head_fourbit_index_routes_a_protected_query_without_scanning() {
        let mut axis = ParallelContextAxis::new(256, AxisPolicy::default()).unwrap();
        let mut serving = vec![0.0_f32; 256];
        let mut other = vec![0.0_f32; 256];
        serving[0] = 4.0;
        other[128] = 4.0;
        for _ in 0..8 {
            axis.observe(&serving).unwrap();
        }
        for _ in 0..8 {
            axis.observe(&other).unwrap();
        }
        let mut query = vec![0.0_f32; 256];
        query[0] = 4.0;
        let views = axis.materialize_query_tokens_adaptive(&query).unwrap();
        let tokens = views
            .iter()
            .flat_map(ParallelContextAxis::expand_token_positions)
            .collect::<Vec<_>>();
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|token| *token < 8));
        assert_eq!(axis.lsh_metrics().2, 0);
    }
}
