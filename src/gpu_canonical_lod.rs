//! GPU adapter for the EXISTING CPU KVSLIM LOD components.
//!
//! REUSES (no parallel 4-bit/index or cold-store implementation):
//!  - adaptive_extent::ParallelContextAxis: original 4-bit fingerprints,
//!    original LSH routing and content-defined token extents;
//!  - cold_store::ColdStore: same BF16 exact K/V page format and verified
//!    partial value reads, immutable commit footer and token checksums;
//!  - attention_certificate::select_by_exact_mass: original rigorous
//!    mass-bound fallback for all query heads.
//!
//! GPU supplies the REAL Q and full BF16 K score matrix. The CPU still
//! decides selection via ORIGINAL shared algorithms, GPU computes attention.
//! The original state-box branch-and-bound is not yet shared: this variant
//! directly uses the original exact-score final certificate every step.
//! K is kept BF16 on GPU for QK, V is stored BF16 in canonical CPU ColdStore
//! and streamed in bounded chunks for PV. Lossless epsilon=0 always reads all.
//! For epsilon>0 the bound is relative to the computed finite GPU FP32 scores,
//! not a promise of identical downstream BF16 logits vs LibTorch CPU.

use crate::{
    adaptive_extent::{AxisPolicy, ParallelContextAxis},
    attention_certificate::{certified, select_by_exact_mass},
    cold_store::{ColdStore, ColdStoreStats, PageGeometry},
    gpu_micro_regions::MicroKeyRegions,
    portable::AttentionGeometry,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Serialize;
use std::{collections::BTreeSet, path::PathBuf};
use tempfile::{Builder, TempDir};

#[derive(Clone, Debug)]
pub struct CanonicalGpuLodConfig {
    pub epsilon: f64,
    pub stage_tokens: usize,
    pub cold_directory: Option<PathBuf>,
}

impl Default for CanonicalGpuLodConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.0,
            stage_tokens: 64,
            cold_directory: None,
        }
    }
}

impl CanonicalGpuLodConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.epsilon.is_finite() && (0.0..=0.1).contains(&self.epsilon),
            "LOD epsilon must be finite within 0..=0.1"
        );
        ensure!(
            (1..=1024).contains(&self.stage_tokens),
            "LOD stage_tokens must be in 1..=1024"
        );
        if let Some(ref path) = self.cold_directory {
            ensure!(
                path.is_dir(),
                "LOD cold directory does not exist: {}",
                path.display()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CanonicalGpuLodStats {
    pub queries: usize,
    pub exact_k_scores_computed_on_gpu: usize,
    pub exact_v_tokens_read_from_cold: usize,
    pub exact_v_tokens_skipped: usize,
    pub exact_fallbacks: usize,
    pub certificates_met: usize,
    pub maximum_certified_error_bound: f64,
    pub four_bit_index_payload_bytes: usize,
    pub physical_cold_store_bytes: u64,
    pub physical_cold_store_pages: usize,
    pub cold_value_read_calls: u64,
    pub prefill_batched_page_tokens: usize,
    pub gpu_stage_tokens: usize,
    /// Micro-metadata region selection is an optional positive-epsilon
    /// optimization, never active for exact epsilon=0 inference.
    pub micro_sparse_queries: usize,
    pub micro_dense_fallback_queries: usize,
    pub micro_omitted_k_dot_products: usize,
    pub micro_omitted_regions: usize,
    pub micro_linked_regions_seen: usize,
    pub micro_fourbit_seeds_not_materialized: usize,
    /// CPU-only auxiliary bounding boxes; zero allocation in epsilon=0.
    pub micro_region_box_payload_bytes: usize,
    pub micro_region_states: usize,
    pub micro_fragmentation_fallback: bool,
    pub shared_original_cpu_algorithms: bool,
}

#[derive(Clone, Debug)]
pub struct CanonicalSelection {
    pub positions: Vec<usize>,
    pub selected_mass_per_head: Vec<f32>,
    pub bounds: Vec<f64>,
    pub candidates_from_original_4bit_lsh: usize,
    pub omitted: usize,
    pub tolerance: f64,
}

#[derive(Clone, Copy, Debug)]
struct PageAddress {
    page_id: u64,
    offset: usize,
}

pub struct CanonicalGpuLod {
    config: CanonicalGpuLodConfig,
    geometry: AttentionGeometry,
    axis: ParallelContextAxis,
    axis_policy: AxisPolicy,
    _directory: Option<TempDir>,
    cold: ColdStore,
    pages: Vec<Vec<PageAddress>>,
    value_abs_max: Vec<Vec<f64>>,
    // Same original CPU strategy: the protected LSH catalogue is built only
    // for layer zero and reused across the measured layer geometry.
    // Exact mass certification still evaluates each layer's real QK.
    last_route_token: Option<usize>,
    last_seed_positions: BTreeSet<usize>,
    micro_regions: Option<MicroKeyRegions>,
    pub stats: CanonicalGpuLodStats,
}

impl CanonicalGpuLod {
    pub fn new(
        geometry: AttentionGeometry,
        layer_count: usize,
        config: CanonicalGpuLodConfig,
    ) -> Result<Self> {
        Self::new_with_axis_policy(geometry, layer_count, config, AxisPolicy::default())
    }

    /// GPU-only policy tuning of the ORIGINAL CPU ParallelContextAxis,
    /// without modifying its implementation or touching the CPU service.
    /// Content-defined boundaries still use the original novelty test.
    pub fn new_with_axis_policy(
        geometry: AttentionGeometry,
        layer_count: usize,
        config: CanonicalGpuLodConfig,
        axis_policy: AxisPolicy,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(layer_count > 0, "LOD model must contain at least one layer");
        let axis = ParallelContextAxis::new_partitioned(
            geometry.kv_elements_per_token(),
            geometry.kv_heads,
            axis_policy.clone(),
        )?;
        let (directory, cold) = match config.cold_directory.as_deref() {
            Some(parent) => {
                let path = parent.join("pages-exact.kv");
                let store = if path.exists() {
                    ColdStore::open(&path, 0)?
                } else {
                    ColdStore::create(&path, 0)?
                };
                (None, store)
            }
            None => {
                let directory = Builder::new().prefix("kvslim-gpu-canonical-").tempdir()?;
                let store = ColdStore::create(directory.path().join("pages-exact.kv"), 0)?;
                (Some(directory), store)
            }
        };
        let stage_tokens = config.stage_tokens;
        // Do NOT allocate regional K bounding boxes in strict epsilon-zero
        // mode: it can never omit K/V, so such metadata would waste memory.
        let micro_regions = if config.epsilon > 0.0 {
            Some(MicroKeyRegions::new(geometry, layer_count)?)
        } else {
            None
        };
        let mut runtime = Self {
            config,
            geometry,
            axis,
            axis_policy,
            _directory: directory,
            cold,
            pages: (0..layer_count).map(|_| Vec::new()).collect(),
            value_abs_max: (0..layer_count)
                .map(|_| vec![0.0; geometry.kv_heads])
                .collect(),
            last_route_token: None,
            last_seed_positions: BTreeSet::new(),
            micro_regions,
            stats: CanonicalGpuLodStats {
                gpu_stage_tokens: stage_tokens,
                shared_original_cpu_algorithms: true,
                ..CanonicalGpuLodStats::default()
            },
        };
        runtime.restore_catalogue()?;
        Ok(runtime)
    }

    fn restore_catalogue(&mut self) -> Result<()> {
        let metadata = self.cold.metadata().to_vec();
        if metadata.is_empty() {
            return Ok(());
        }
        let row_bytes = self.geometry.head_dim * 2;
        let token_bytes = self.geometry.kv_elements_per_token() * 2;
        for meta in metadata {
            let layer = usize::try_from(meta.geometry.layer)?;
            ensure!(layer < self.pages.len(), "persisted KV layer out of range");
            ensure!(
                meta.geometry.heads as usize == self.geometry.kv_heads
                    && meta.geometry.head_dim as usize == self.geometry.head_dim,
                "persisted KV geometry does not match model"
            );
            let start = usize::try_from(meta.geometry.token_start)?;
            let count = usize::try_from(meta.geometry.tokens)?;
            ensure!(
                self.pages[layer].len() == start,
                "persisted KV pages are not a contiguous causal prefix"
            );
            let payload = self.cold.read_page(meta.page_id)?;
            ensure!(
                payload.key.len() == count * token_bytes
                    && payload.value.len() == count * token_bytes,
                "persisted KV payload shape mismatch"
            );
            let mut keys_token_major = vec![0_u8; payload.key.len()];
            let mut values_token_major = vec![0_u8; payload.value.len()];
            for head in 0..self.geometry.kv_heads {
                for token in 0..count {
                    let source = (head * count + token) * row_bytes;
                    let destination = (token * self.geometry.kv_heads + head) * row_bytes;
                    keys_token_major[destination..destination + row_bytes]
                        .copy_from_slice(&payload.key[source..source + row_bytes]);
                    values_token_major[destination..destination + row_bytes]
                        .copy_from_slice(&payload.value[source..source + row_bytes]);
                }
            }
            for token in 0..count {
                let value = &values_token_major[token * token_bytes..(token + 1) * token_bytes];
                for head in 0..self.geometry.kv_heads {
                    for dim in 0..self.geometry.head_dim {
                        let offset = (head * self.geometry.head_dim + dim) * 2;
                        let scalar =
                            bf16::from_bits(u16::from_le_bytes([value[offset], value[offset + 1]]))
                                .to_f32();
                        ensure!(scalar.is_finite(), "persisted KV contains nonfinite V");
                        self.value_abs_max[layer][head] =
                            self.value_abs_max[layer][head].max(f64::from(scalar.abs()));
                    }
                }
            }
            if layer == 0 {
                for token in 0..count {
                    let key = &keys_token_major[token * token_bytes..(token + 1) * token_bytes];
                    let signal = key
                        .chunks_exact(2)
                        .map(|bits| {
                            bf16::from_bits(u16::from_le_bytes([bits[0], bits[1]])).to_f32()
                        })
                        .collect::<Vec<_>>();
                    self.axis.observe(&signal)?;
                }
            }
            if let Some(index) = self.micro_regions.as_mut() {
                index.observe(layer, start, &keys_token_major, &self.axis)?;
            }
            self.pages[layer].extend((0..count).map(|offset| PageAddress {
                page_id: meta.page_id,
                offset,
            }));
        }
        self.axis.flush();
        self.stats.four_bit_index_payload_bytes = self.axis.fingerprint_payload_bytes();
        self.stats.physical_cold_store_pages = self.cold.metadata().len();
        self.stats.physical_cold_store_bytes = self.cold.stats().committed_bytes;
        if let Some(index) = self.micro_regions.as_ref() {
            self.stats.micro_region_box_payload_bytes = index.region_bounds_payload_bytes();
            self.stats.micro_region_states = index.layer_regions(0);
            self.stats.micro_fragmentation_fallback = index.disabled_due_to_fragmentation();
        }
        Ok(())
    }

    pub fn config(&self) -> &CanonicalGpuLodConfig {
        &self.config
    }
    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }
    pub fn stage_tokens(&self) -> usize {
        self.config.stage_tokens
    }
    pub fn storage_metrics(&self) -> ColdStoreStats {
        self.cold.stats()
    }
    pub fn axis(&self) -> &ParallelContextAxis {
        &self.axis
    }
    pub fn axis_policy(&self) -> &AxisPolicy {
        &self.axis_policy
    }

    /// This is allowed ONLY before the first inference token.
    /// Reuses original CPU novelty detection but allows the GPU to
    /// form finer logical regions from different application contexts.
    /// It cannot re-segment already committed BF16 K/V.
    pub fn set_gpu_axis_policy(&mut self, policy: AxisPolicy) -> Result<()> {
        ensure!(
            self.axis.token_len() == 0 && self.pages.iter().all(Vec::is_empty),
            "micro axis policy can only change before first token"
        );
        self.axis = ParallelContextAxis::new_partitioned(
            self.geometry.kv_elements_per_token(),
            self.geometry.kv_heads,
            policy.clone(),
        )?;
        self.axis_policy = policy;
        Ok(())
    }

    pub fn layer_tokens(&self, layer: usize) -> usize {
        self.pages.get(layer).map_or(0, Vec::len)
    }

    /// Single-token decode uses the exact same canonical ColdStore page
    /// writer as CPU. Prefill calls append_batch with 16-128 tokens/page,
    /// without changing the CPU ColdStore implementation or file format.
    pub fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &[u8],
        value: &[u8],
    ) -> Result<()> {
        self.append_batch(layer, position, key, value, 1)
    }

    /// Append exact BF16 K/V provided in GPU [token][kvhead][head_dim]
    /// layout. The canonical CPU ColdStore stores [head][token][dim].
    /// No BF16 value or weight is quantized. Every token is covered by
    /// the SAME per-token FNV hash table and committed page digest as CPU.
    /// Pages are grouped in the GPU adapter only; canonical CPU is untouched.
    pub fn append_batch(
        &mut self,
        layer: usize,
        position: usize,
        keys_token_major: &[u8],
        values_token_major: &[u8],
        page_tokens: usize,
    ) -> Result<()> {
        ensure!(
            (1..=512).contains(&page_tokens),
            "GPU ColdStore page size must be 1..512 tokens"
        );
        let token_bytes = self.geometry.kv_elements_per_token() * 2;
        ensure!(
            !keys_token_major.is_empty()
                && keys_token_major.len() == values_token_major.len()
                && keys_token_major.len().is_multiple_of(token_bytes),
            "GPU LOD batched BF16 K/V shape mismatch"
        );
        let count = keys_token_major.len() / token_bytes;
        let items = self
            .pages
            .get(layer)
            .context("GPU LOD batch layer out of range")?;
        ensure!(
            items.len() == position,
            "GPU LOD append needs sequential tokens"
        );
        let kvheads = self.geometry.kv_heads;
        let row_bytes = self.geometry.head_dim * 2;
        let mut maxima = self.value_abs_max[layer].clone();
        // Validate every value and each layer-zero key before committing
        // any ColdStore page, avoiding silently constructing a broken index.
        let mut axis_vectors = if layer == 0 {
            Vec::with_capacity(count)
        } else {
            Vec::new()
        };
        for i in 0..count {
            let k = &keys_token_major[i * token_bytes..(i + 1) * token_bytes];
            let v = &values_token_major[i * token_bytes..(i + 1) * token_bytes];
            if layer == 0 {
                let signal = k
                    .chunks_exact(2)
                    .map(|bits| bf16::from_bits(u16::from_le_bytes([bits[0], bits[1]])).to_f32())
                    .collect::<Vec<_>>();
                ensure!(
                    signal.iter().all(|x| x.is_finite()),
                    "GPU ColdStore K nonfinite"
                );
                axis_vectors.push(signal);
            }
            for (h, max) in maxima.iter_mut().enumerate() {
                for d in 0..self.geometry.head_dim {
                    let j = h * row_bytes + d * 2;
                    let value = bf16::from_bits(u16::from_le_bytes([v[j], v[j + 1]])).to_f32();
                    ensure!(value.is_finite(), "GPU ColdStore V nonfinite");
                    *max = max.max(value.abs() as f64);
                }
            }
        }
        let tile = if self.geometry.head_dim.is_multiple_of(16) {
            16
        } else {
            1
        };
        for start in (0..count).step_by(page_tokens) {
            let n = page_tokens.min(count - start);
            let mut keys = vec![0u8; n * token_bytes];
            let mut values = vec![0u8; n * token_bytes];
            for head in 0..kvheads {
                for token in 0..n {
                    let source = ((start + token) * kvheads + head) * row_bytes;
                    let dest = (head * n + token) * row_bytes;
                    keys[dest..dest + row_bytes]
                        .copy_from_slice(&keys_token_major[source..source + row_bytes]);
                    values[dest..dest + row_bytes]
                        .copy_from_slice(&values_token_major[source..source + row_bytes]);
                }
            }
            let meta = self.cold.append_page(
                PageGeometry {
                    layer: u32::try_from(layer)?,
                    token_start: (position + start) as u64,
                    tokens: u32::try_from(n)?,
                    heads: u32::try_from(kvheads)?,
                    head_dim: u32::try_from(self.geometry.head_dim)?,
                    tile,
                },
                &keys,
                &values,
                false,
            )?;
            self.pages[layer].extend((0..n).map(|offset| PageAddress {
                page_id: meta.page_id,
                offset,
            }));
            self.stats.prefill_batched_page_tokens = self.stats.prefill_batched_page_tokens.max(n);
        }
        self.value_abs_max[layer] = maxima;
        for vector in axis_vectors {
            self.axis.observe(&vector)?;
        }
        // A multi-token append is a prefill lifecycle boundary. Publish the
        // final open extent now so that the immediately following decode can
        // route through LSH and the hierarchy. Single-token decode appends stay
        // open until a real novelty boundary instead of creating one state per
        // token.
        if layer == 0 && count > 1 {
            self.axis.flush();
            self.last_route_token = None;
            self.last_seed_positions.clear();
        }
        // Indices reference the SAME original CPU semantic regions
        // and BF16 K that were just saved in the canonical ColdStore.
        // No secondary 4-bit index, segmentation or quantized K authority.
        if let Some(index) = self.micro_regions.as_mut() {
            index.observe(layer, position, keys_token_major, &self.axis)?;
            self.stats.micro_region_box_payload_bytes = index.region_bounds_payload_bytes();
            self.stats.micro_region_states = index.layer_regions(0);
            self.stats.micro_fragmentation_fallback = index.disabled_due_to_fragmentation();
        }
        self.stats.four_bit_index_payload_bytes = self.axis.fingerprint_payload_bytes();
        let info = self.cold.stats();
        self.stats.physical_cold_store_bytes = info.committed_bytes;
        self.stats.physical_cold_store_pages = info.pages;
        Ok(())
    }

    /// Same projection as original prefill_tch::protected_queries:
    /// each query head maps to a zero-filled KV-head planner vector.
    fn route_with_original_axis(&mut self, query: &[u8]) -> Result<BTreeSet<usize>> {
        let nheads = self.geometry.query_heads;
        let dim = self.geometry.head_dim;
        ensure!(
            query.len() == nheads * dim * 2,
            "GPU LOD protected Q shape mismatch"
        );
        let mut positions = BTreeSet::new();
        let group = nheads / self.geometry.kv_heads;
        for head in 0..nheads {
            let mut planner = vec![0.0_f32; self.geometry.kv_elements_per_token()];
            let kvhead = head / group;
            for d in 0..dim {
                let offset = (head * dim + d) * 2;
                planner[kvhead * dim + d] =
                    bf16::from_bits(u16::from_le_bytes([query[offset], query[offset + 1]]))
                        .to_f32();
            }
            if self.axis.token_len() > 0 {
                for view in self.axis.materialize_query_tokens_adaptive(&planner)? {
                    positions.extend(ParallelContextAxis::expand_token_positions(&view));
                }
            }
        }
        Ok(positions)
    }

    /// Reuse the original CPU 4-bit LSH query once per token across
    /// all layers, and keep its candidate graph only for priorities.
    fn seeds_for_query(
        &mut self,
        layer: usize,
        tokens: usize,
        query: &[u8],
    ) -> Result<BTreeSet<usize>> {
        if layer == 0 && self.last_route_token != Some(tokens) {
            let result = self.route_with_original_axis(query)?;
            self.last_route_token = Some(tokens);
            self.last_seed_positions = result.clone();
            Ok(result)
        } else if self.last_route_token == Some(tokens) {
            Ok(self.last_seed_positions.clone())
        } else {
            // If layer zero was skipped for any reason, never reuse
            // previous-token graph. The exact mass certificate is
            // authoritative and can run without LSH seeds.
            Ok(BTreeSet::new())
        }
    }

    /// OPTIONAL fast path: original dimensional micro links and 4-bit
    /// LSH seed IDs propose regions, then BF16 K min/max boxes and
    /// a conservative softmax-mass bound certify whether K AND V may
    /// safely be left unread. Returns None for any ambiguous case.
    ///
    /// The corresponding OpenCL sparse-QK kernels execute only on
    /// the selected token positions. epsilon=0 never calls this.
    pub fn plan_micro_regions(
        &mut self,
        layer: usize,
        query: &[u8],
    ) -> Result<Option<CanonicalSelection>> {
        if self.config.epsilon == 0.0 {
            return Ok(None);
        }
        let tokens = self.layer_tokens(layer);
        if tokens == 0 || self.micro_regions.is_none() {
            return Ok(None);
        }
        let seeds = self.seeds_for_query(layer, tokens, query)?;
        let plan = self.micro_regions.as_ref().unwrap().plan(
            layer,
            query,
            &self.axis,
            &seeds,
            &self.value_abs_max[layer],
            self.config.epsilon,
        )?;
        let Some(plan) = plan else {
            self.stats.micro_dense_fallback_queries += 1;
            return Ok(None);
        };
        ensure!(
            plan.positions.len() < tokens && !plan.positions.is_empty(),
            "micro region planner returned no useful certified skip"
        );
        ensure!(
            certified(&plan.error_bounds, self.config.epsilon),
            "uncertified micro region omission rejected"
        );
        let omitted = tokens - plan.positions.len();
        self.stats.queries += 1;
        self.stats.exact_k_scores_computed_on_gpu +=
            plan.positions.len() * self.geometry.query_heads;
        self.stats.exact_v_tokens_skipped += omitted;
        self.stats.certificates_met += 1;
        self.stats.micro_sparse_queries += 1;
        self.stats.micro_omitted_k_dot_products += plan.avoided_k_dot_products;
        self.stats.micro_omitted_regions += plan.omitted_state_count;
        self.stats.micro_linked_regions_seen += plan.linked_seed_regions;
        self.stats.maximum_certified_error_bound = self
            .stats
            .maximum_certified_error_bound
            .max(plan.error_bounds.iter().copied().fold(0.0, f64::max));
        Ok(Some(CanonicalSelection {
            positions: plan.positions,
            selected_mass_per_head: vec![1.0; self.geometry.query_heads],
            bounds: plan.error_bounds,
            candidates_from_original_4bit_lsh: plan.lsh_seed_regions,
            omitted,
            tolerance: self.config.epsilon,
        }))
    }

    /// Epsilon-zero has a mathematically predetermined exact result:
    /// every V token is required. The original 4-bit axis is still populated
    /// via append, but LSH searches and GPU QK readback cannot narrow the
    /// selection, so they are skipped to avoid pointless overhead.
    pub fn select_dense_exact(&mut self, layer: usize) -> Result<CanonicalSelection> {
        let tokens = self.layer_tokens(layer);
        ensure!(tokens > 0, "no BF16 K/V tokens for exact LOD");
        ensure!(
            self.axis.token_len() == tokens,
            "canonical axis not synchronized"
        );
        let heads = self.geometry.query_heads;
        self.stats.queries += 1;
        self.stats.exact_k_scores_computed_on_gpu += tokens * heads;
        self.stats.exact_fallbacks += 1;
        Ok(CanonicalSelection {
            positions: (0..tokens).collect(),
            selected_mass_per_head: vec![1.0; heads],
            bounds: vec![0.0; heads],
            candidates_from_original_4bit_lsh: 0,
            omitted: 0,
            tolerance: 0.0,
        })
    }

    /// Record a mathematically exact, full-history GPU attention while
    /// the BF16 K/V of an IN-FLIGHT request remain in device buffers.
    /// All V participate, so the omitted probability mass is exactly zero.
    /// The original canonical ColdStore is committed at the end of the
    /// turn BEFORE the response becomes visible to the caller.
    ///
    /// This never authorizes omission, approximate attention or disk I/O
    /// in a positive-epsilon LOD mode.
    pub fn observe_deferred_exact_attention(
        &mut self,
        layer: usize,
        query_tokens: usize,
    ) -> Result<()> {
        ensure!(
            self.config.epsilon == 0.0,
            "deferred exact attention cannot bypass positive-epsilon certification"
        );
        ensure!(
            query_tokens > 0 && layer < self.pages.len(),
            "deferred GPU attention geometry invalid"
        );
        let committed = self.layer_tokens(layer);
        ensure!(
            committed > 0 && committed <= query_tokens,
            "deferred GPU attention must have a committed BF16 prefill prefix"
        );
        ensure!(
            self.axis.token_len() == self.layer_tokens(0),
            "original four-bit axis is not aligned with committed prefill"
        );
        let count = query_tokens
            .checked_mul(self.geometry.query_heads)
            .context("exact GPU attention score accounting overflow")?;
        self.stats.queries += 1;
        self.stats.exact_k_scores_computed_on_gpu += count;
        self.stats.exact_fallbacks += 1;
        // Every token V was read by the full GPU attention kernel;
        // zero mass omitted, error bound=0. The counters above are
        // mathematically equivalent to select_dense_exact.
        Ok(())
    }

    /// A fully materialized GPU BF16 hot V cache is exact attention,
    /// so it is always a safe fallback even if epsilon was >0. Unlike
    /// micro selection, this path does not omit any K or V. It must
    /// still preserve the original exact ColdStore and four-bit index.
    pub fn observe_full_hot_attention(&mut self, layer: usize, tokens: usize) -> Result<()> {
        ensure!(
            layer < self.pages.len() && tokens > 0 && self.layer_tokens(layer) == tokens,
            "GPU hot full attention requires all BF16 K/V committed"
        );
        ensure!(
            self.axis.token_len() == tokens,
            "original CPU 4-bit index out of sync with BF16 hot attention"
        );
        self.stats.queries += 1;
        self.stats.exact_k_scores_computed_on_gpu += tokens * self.geometry.query_heads;
        self.stats.exact_fallbacks += 1;
        Ok(())
    }

    /// Scores must be read from the real GPU QK kernel BEFORE softmax.
    /// The selection and certification functions are the exact same
    /// functions used by the original KVSLIM CPU.
    pub fn select(
        &mut self,
        layer: usize,
        query: &[u8],
        scores: &[f32],
    ) -> Result<CanonicalSelection> {
        let tokens = self.layer_tokens(layer);
        ensure!(tokens > 0, "GPU LOD no BF16 KV recorded");
        ensure!(
            scores.len() == tokens * self.geometry.query_heads,
            "GPU LOD scores not [query_heads,tokens]"
        );
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "nonfinite real GPU QK scores"
        );
        ensure!(
            self.axis.token_len() == tokens,
            "GPU LOD original axis not updated for all tokens"
        );
        if self.config.epsilon == 0.0 {
            return self.select_dense_exact(layer);
        }
        let router_candidates = self.seeds_for_query(layer, tokens, query)?;
        self.stats.queries += 1;
        self.stats.exact_k_scores_computed_on_gpu += scores.len();
        let qheads = self.geometry.query_heads;
        let real_scores = scores
            .chunks_exact(tokens)
            .map(|head| head.iter().map(|&s| f64::from(s)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let value_max = (0..qheads)
            .map(|qh| {
                let kvhead = qh / (qheads / self.geometry.kv_heads);
                self.value_abs_max[layer][kvhead]
            })
            .collect::<Vec<_>>();
        // Original exact-score mass oracle and certification. Routing is
        // advisory; LSH misses never authorize skipping any BF16 V.
        let (exact_selection, _) =
            select_by_exact_mass(&real_scores, &value_max, self.config.epsilon)?;
        let candidate_count = router_candidates.len();
        // CRITICAL: previous GPU code forced every four-bit/LSH candidate
        // into the exact V read set, even when the CPU exact-mass
        // certificate had already selected all required BF16 tokens.
        // The original CPU micro/LSH graph provides PRIORITIES only:
        // it never authorizes extra cold reads or missed exact scores.
        // Exact mass selection is the final canonical recovery authority.
        let mut positions = exact_selection.into_iter().collect::<BTreeSet<_>>();
        self.stats.micro_fourbit_seeds_not_materialized +=
            router_candidates.difference(&positions).count();
        let mut bounds = vec![0.0; qheads];
        let mut normalizers = vec![1.0f32; qheads];
        for (head, scores) in real_scores.iter().enumerate() {
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0f64;
            let mut kept = 0.0f64;
            for (token, &score) in scores.iter().enumerate() {
                let weight = (score - maximum).exp();
                total += weight;
                if positions.contains(&token) {
                    kept += weight;
                }
            }
            ensure!(
                total > 0.0 && kept > 0.0 && total.is_finite() && kept.is_finite(),
                "GPU LOD softmax mass invalid"
            );
            bounds[head] = 2.0 * (total - kept).max(0.0) / total * value_max[head];
            normalizers[head] = (kept / total) as f32;
        }
        let certified_result = certified(&bounds, self.config.epsilon);
        if !certified_result {
            positions.extend(0..tokens);
            bounds.fill(0.0);
            normalizers.fill(1.0);
            self.stats.exact_fallbacks += 1;
        } else {
            self.stats.certificates_met += 1;
        }
        let positions = positions.into_iter().collect::<Vec<_>>();
        let omitted = tokens - positions.len();
        let bound = bounds.iter().copied().fold(0.0, f64::max);
        self.stats.exact_v_tokens_skipped += omitted;
        self.stats.maximum_certified_error_bound =
            self.stats.maximum_certified_error_bound.max(bound);
        Ok(CanonicalSelection {
            positions,
            selected_mass_per_head: normalizers,
            bounds,
            candidates_from_original_4bit_lsh: candidate_count,
            omitted,
            tolerance: self.config.epsilon,
        })
    }

    /// Canonical ColdStore partial V reads grouped by physical page.
    /// A single read_value_tokens page read validates its per-token hashes
    /// and returns head-major; convert to token-major for GPU PV staging.
    /// Sorted query positions ensure no duplicated reading of a page within
    /// one stage (unless the stage boundary splits a physical ColdStore page).
    pub fn read_values(&mut self, layer: usize, positions: &[usize]) -> Result<Vec<u8>> {
        let pages = self
            .pages
            .get(layer)
            .context("GPU LOD read layer missing")?;
        let bytes_per_token = self.geometry.kv_elements_per_token() * 2;
        let mut out = Vec::with_capacity(
            positions
                .len()
                .checked_mul(bytes_per_token)
                .context("GPU LOD V length overflow")?,
        );
        let mut cursor = 0usize;
        while cursor < positions.len() {
            let first = *pages
                .get(positions[cursor])
                .context("GPU LOD token not recorded")?;
            let mut offsets = vec![first.offset];
            let mut end = cursor + 1;
            while end < positions.len() {
                let next = *pages
                    .get(positions[end])
                    .context("GPU LOD token not recorded")?;
                if next.page_id != first.page_id {
                    break;
                }
                ensure!(
                    offsets.last().is_none_or(|&last| next.offset > last),
                    "GPU LOD selected positions must be sorted and unique"
                );
                offsets.push(next.offset);
                end += 1;
            }
            let data = self.cold.read_value_tokens(first.page_id, &offsets)?;
            ensure!(
                data.len() == offsets.len() * bytes_per_token,
                "canonical ColdStore partial page BF16 dimensions invalid"
            );
            let head_bytes = self.geometry.head_dim * 2;
            for local in 0..offsets.len() {
                for head in 0..self.geometry.kv_heads {
                    let address = (head * offsets.len() + local) * head_bytes;
                    out.extend_from_slice(&data[address..address + head_bytes]);
                }
            }
            cursor = end;
        }
        self.stats.exact_v_tokens_read_from_cold += positions.len();
        self.stats.cold_value_read_calls = self.cold.stats().reads;
        Ok(out)
    }

    /// Exact K counterpart of read_values, shared by every compute backend.
    /// Returned bytes are token-major BF16 and retain per-token verification.
    pub fn read_keys(&mut self, layer: usize, positions: &[usize]) -> Result<Vec<u8>> {
        let pages = self.pages.get(layer).context("LOD read layer missing")?;
        let bytes_per_token = self.geometry.kv_elements_per_token() * 2;
        let mut out = Vec::with_capacity(
            positions
                .len()
                .checked_mul(bytes_per_token)
                .context("LOD K length overflow")?,
        );
        let mut cursor = 0usize;
        while cursor < positions.len() {
            let first = *pages
                .get(positions[cursor])
                .context("LOD K token not recorded")?;
            let mut offsets = vec![first.offset];
            let mut end = cursor + 1;
            while end < positions.len() {
                let next = *pages
                    .get(positions[end])
                    .context("LOD K token not recorded")?;
                if next.page_id != first.page_id {
                    break;
                }
                ensure!(
                    offsets.last().is_none_or(|&last| next.offset > last),
                    "LOD selected K positions must be sorted and unique"
                );
                offsets.push(next.offset);
                end += 1;
            }
            let data = self.cold.read_key_tokens(first.page_id, &offsets)?;
            ensure!(
                data.len() == offsets.len() * bytes_per_token,
                "canonical ColdStore partial K BF16 dimensions invalid"
            );
            let head_bytes = self.geometry.head_dim * 2;
            for local in 0..offsets.len() {
                for head in 0..self.geometry.kv_heads {
                    let address = (head * offsets.len() + local) * head_bytes;
                    out.extend_from_slice(&data[address..address + head_bytes]);
                }
            }
            cursor = end;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(xs: &[f32]) -> Vec<u8> {
        xs.iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect()
    }
    fn geom() -> AttentionGeometry {
        AttentionGeometry::new(4, 2, 4).unwrap()
    }

    #[test]
    fn canonical_cpu_index_and_cold_store_reused_exactly() {
        let mut store = CanonicalGpuLod::new(geom(), 1, CanonicalGpuLodConfig::default()).unwrap();
        let k = bits(&[0., 1., 2., 3., 4., 5., 6., 7.]);
        let v = bits(&[1., 0., -2., 3., 4., -5., 6., 7.]);
        for t in 0..5 {
            store.append(0, t, &k, &v).unwrap();
        }
        assert_eq!(store.layer_tokens(0), 5);
        assert_eq!(store.axis().token_len(), 5);
        assert!(store.stats.four_bit_index_payload_bytes > 0);
        assert!(store.storage_metrics().committed_bytes > 0);
        let query = bits(&[0.0; 16]);
        let selection = store.select(0, &query, &[0.; 4 * 5]).unwrap();
        assert_eq!(selection.positions, vec![0, 1, 2, 3, 4]);
        assert_eq!(selection.omitted, 0);
        assert!(certified(&selection.bounds, 0.0));
        let exact = store.read_values(0, &selection.positions).unwrap();
        assert_eq!(
            exact,
            [v.clone(), v.clone(), v.clone(), v.clone(), v].concat()
        );
        assert_eq!(store.stats.exact_v_tokens_read_from_cold, 5);
    }

    #[test]
    fn approximate_mass_oracle_is_same_cpu_function() {
        let config = CanonicalGpuLodConfig {
            epsilon: 0.01,
            ..Default::default()
        };
        let mut cache = CanonicalGpuLod::new(geom(), 1, config).unwrap();
        let k = bits(&[1., 0., 0., 0., 1., 0., 0., 0.]);
        let v = bits(&[2.; 8]);
        for t in 0..24 {
            cache.append(0, t, &k, &v).unwrap();
        }
        let mut scores = vec![-30.0f32; 4 * 24];
        for qh in 0..4 {
            scores[qh * 24 + 2 + qh] = 30.0;
        }
        let query = bits(&[0.; 16]);
        let result = cache.select(0, &query, &scores).unwrap();
        assert!(certified(&result.bounds, 0.01));
        assert!(result.positions.len() < 24);
        assert!(result.positions.contains(&2));
        assert!(result.positions.contains(&5));
    }

    #[test]
    fn batched_pages_preserve_every_bf16_value_and_reduce_page_reads() {
        let count = 37usize;
        let key_items: Vec<f32> = (0..count * 8)
            .map(|i| (i as f32 % 59.0 - 29.0) * 0.03125)
            .collect();
        let value_items: Vec<f32> = (0..count * 8)
            .map(|i| (i as f32 % 61.0 - 30.0) * 0.0625)
            .collect();
        let keys = bits(&key_items);
        let values = bits(&value_items);
        let mut grouped =
            CanonicalGpuLod::new(geom(), 1, CanonicalGpuLodConfig::default()).unwrap();
        grouped.append_batch(0, 0, &keys, &values, 8).unwrap();
        assert_eq!(grouped.layer_tokens(0), count);
        assert_eq!(grouped.axis().token_len(), count);
        assert_eq!(grouped.storage_metrics().pages, 5);
        assert_eq!(grouped.stats.physical_cold_store_pages, 5);
        assert!(grouped.stats.four_bit_index_payload_bytes > 0);
        let original: Vec<usize> = (0..count).collect();
        assert_eq!(grouped.read_values(0, &original).unwrap(), values);
        assert_eq!(grouped.storage_metrics().reads, 5);
        // Sparse indices across several pages must retain order and be
        // bitwise identical to the original exact BF16 token payloads.
        let sparse = [0usize, 2, 4, 7, 8, 11, 23, 24, 36];
        let selected = grouped.read_values(0, &sparse).unwrap();
        let original_selected = sparse
            .iter()
            .flat_map(|&i| values[i * 16..(i + 1) * 16].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(selected, original_selected);
        let mut token_pages =
            CanonicalGpuLod::new(geom(), 1, CanonicalGpuLodConfig::default()).unwrap();
        for t in 0..count {
            token_pages
                .append(
                    0,
                    t,
                    &keys[t * 16..(t + 1) * 16],
                    &values[t * 16..(t + 1) * 16],
                )
                .unwrap();
        }
        assert_eq!(token_pages.storage_metrics().pages, count);
        assert_eq!(token_pages.read_values(0, &original).unwrap(), values);
        assert_eq!(token_pages.storage_metrics().reads, count as u64);
        assert_eq!(grouped.storage_metrics().reads, 10);
        assert!(
            grouped.storage_metrics().committed_bytes
                < token_pages.storage_metrics().committed_bytes
        );
    }

    #[test]
    fn configured_cold_directory_reopens_exact_prefix_and_axis() {
        let directory = tempfile::tempdir().unwrap();
        let config = CanonicalGpuLodConfig {
            cold_directory: Some(directory.path().to_path_buf()),
            ..Default::default()
        };
        let count = 9usize;
        let keys = bits(
            &(0..count * 8)
                .map(|index| index as f32 * 0.03125)
                .collect::<Vec<_>>(),
        );
        let values = bits(
            &(0..count * 8)
                .map(|index| 1.0 - index as f32 * 0.015625)
                .collect::<Vec<_>>(),
        );
        {
            let mut first = CanonicalGpuLod::new(geom(), 1, config.clone()).unwrap();
            first.append_batch(0, 0, &keys, &values, 4).unwrap();
            assert!(!first.axis().extents().is_empty());
        }
        let mut reopened = CanonicalGpuLod::new(geom(), 1, config).unwrap();
        assert_eq!(reopened.layer_tokens(0), count);
        assert_eq!(reopened.axis().token_len(), count);
        assert!(!reopened.axis().extents().is_empty());
        assert_eq!(
            reopened
                .read_values(0, &(0..count).collect::<Vec<_>>())
                .unwrap(),
            values
        );
    }

    #[test]
    fn canonical_exact_storage_rejects_nan() {
        let mut store = CanonicalGpuLod::new(geom(), 1, CanonicalGpuLodConfig::default()).unwrap();
        let key = bits(&[f32::NAN; 8]);
        let value = bits(&[1.0; 8]);
        assert!(store.append(0, 0, &key, &value).is_err());
        assert_eq!(store.layer_tokens(0), 0);
    }
}
