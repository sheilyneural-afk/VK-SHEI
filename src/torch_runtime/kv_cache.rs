//! KV cache, page management, and speculation for the LibTorch backend.
//!
//! Owns the KV memory model: dense pages, M-SLIM pages, attention certificates,
//! and the speculative drafting policies. Independent of hardware selection;
//! all tensor operations run through LibTorch on whatever device is active.

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::adaptive_extent::{AxisPolicy, ParallelContextAxis};
#[cfg(test)]
use crate::attention_certificate::omitted_error_bound;
use crate::attention_certificate::{
    bf16_rounding_cell_contains, certified, select_by_exact_mass, select_v_by_token_contribution,
};
use crate::cold_store::{ColdStore, ColdStoreStats, PageGeometry, PreparedPage};
use crate::conversion_worker::{Bf16PageJob, ConversionStats, ConversionWorker, PageShape};
use crate::model_config::ModelSpec;
use crate::prefill_tch::protected_queries;
use crate::shadow_metrics::{ShadowEventId, ShadowMetrics};

use super::weight_io::WeightArchive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChatTemplateFamily {
    TemplateDelimited,
}

pub(super) struct DecoderGeometry {
    pub(super) hidden: i64,
    pub(super) intermediate: i64,
    pub(super) vocab: i64,
    pub(super) n_layers: usize,
    pub(super) query_heads: i64,
    pub(super) kv_heads: i64,
    pub(super) head_dim: i64,
    pub(super) rope_theta: f64,
    pub(super) rms_eps: f64,
    pub(super) llama3_rope: bool,
    pub(super) eos: Vec<i64>,
    pub(super) chat: ChatTemplateFamily,
}

impl DecoderGeometry {
    pub(super) fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let intermediate = spec
            .intermediate_size
            .context("missing intermediate_size")?;
        let vocab = spec.vocab_size.context("missing vocab_size")?;
        let eos = spec.eos_token_ids.iter().map(|&id| i64::from(id)).collect();
        Ok(Self {
            hidden: spec.hidden_size as i64,
            intermediate: intermediate as i64,
            vocab: vocab as i64,
            n_layers: spec.num_hidden_layers,
            query_heads: spec.attention.query_heads as i64,
            kv_heads: spec.attention.kv_heads as i64,
            head_dim: spec.attention.head_dim as i64,
            rope_theta: spec.rope_theta.unwrap_or(10_000.0),
            rms_eps: spec.rms_norm_eps.unwrap_or(1e-5),
            llama3_rope: spec.rope_scaling.is_some(),
            eos,
            chat: ChatTemplateFamily::TemplateDelimited,
        })
    }

    pub(super) fn query_out(&self) -> i64 {
        self.query_heads * self.head_dim
    }

    pub(super) fn kv_out(&self) -> i64 {
        self.kv_heads * self.head_dim
    }
}

pub struct DecoderLayer {
    pub(super) input_norm: Tensor,
    pub(super) q: Tensor,
    pub(super) k: Tensor,
    pub(super) v: Tensor,
    pub(super) o: Tensor,
    pub(super) post_norm: Tensor,
    pub(super) gate: Tensor,
    pub(super) up: Tensor,
    pub(super) down: Tensor,
    pub(super) query_heads: i64,
    pub(super) kv_heads: i64,
    pub(super) head_dim: i64,
    pub(super) rope_theta: f64,
    pub(super) rms_eps: f64,
    pub(super) llama3_rope: bool,
}

pub struct DecoderModel {
    pub(super) embedding: Tensor,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: Tensor,
    pub(super) lm_head: Tensor,
    pub(super) rms_eps: f64,
    pub(super) eos: Vec<i64>,
    pub(super) chat: ChatTemplateFamily,
    pub(super) query_heads: i64,
    pub(super) kv_heads: i64,
    pub(super) head_dim: i64,
    pub(super) streaming_prefill: bool,
    pub(super) declared_context_limit: usize,
}

pub struct DenseKvCache {
    pub(super) layers: Vec<Option<(Tensor, Tensor)>>,
    pub(super) sequence_tokens: usize,
    pub(super) querywise_attention: bool,
}

pub trait AttentionCache {
    fn sequence_tokens(&self) -> usize;
    fn physical_bytes(&self) -> usize;
    fn attend(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<Tensor>;
    fn finish_step(&mut self, added_tokens: usize);
    fn record_layer_output(
        &mut self,
        _layer_index: usize,
        _sequence_start: usize,
        _output: &Tensor,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct LayerOutputTrace {
    pub(super) layer_index: usize,
    pub(super) sequence_end: usize,
    pub(super) last_token: Vec<f32>,
}

impl DenseKvCache {
    pub fn new() -> Self {
        Self::with_layers(24)
    }

    pub fn with_layers(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| None).collect(),
            sequence_tokens: 0,
            querywise_attention: false,
        }
    }

    pub(super) fn gqa_groups(query: &Tensor, key: &Tensor) -> Result<i64> {
        let query_heads = query.size()[1];
        let kv_heads = key.size()[1];
        ensure!(
            kv_heads > 0 && query_heads % kv_heads == 0,
            "GQA incompatible: {query_heads}Q / {kv_heads}KV"
        );
        Ok(query_heads / kv_heads)
    }

    pub fn sequence_tokens(&self) -> usize {
        self.sequence_tokens
    }

    pub fn physical_bytes(&self) -> usize {
        self.layers
            .iter()
            .filter_map(Option::as_ref)
            .map(|(key, value)| (key.numel() + value.numel()) * 2)
            .sum()
    }

    pub(super) fn state_digest(&self) -> Result<String> {
        let mut digest = Sha256::new();
        digest.update(b"mslim-dense-kv-state-v1");
        digest.update((self.sequence_tokens as u64).to_le_bytes());
        for (layer, state) in self.layers.iter().enumerate() {
            let (key, value) = state
                .as_ref()
                .with_context(|| format!("capa densa {layer} sin estado KV"))?;
            for tensor in [key, value] {
                ensure!(tensor.kind() == Kind::BFloat16, "estado KV no BF16");
                let contiguous = tensor.contiguous();
                // SAFETY: this CPU runtime's cache owns contiguous BF16 tensors.
                // The tensor remains alive and unmodified while its numel()*2
                // initialized bytes are borrowed for the digest; the slice does not escape.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        contiguous.data_ptr().cast::<u8>(),
                        contiguous.numel() * 2,
                    )
                };
                digest.update(bytes);
            }
        }
        Ok(digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }

    pub(super) fn compare_appended_state(
        &self,
        other: &Self,
        appended: usize,
    ) -> Result<KvStateDifference> {
        ensure!(
            self.sequence_tokens == other.sequence_tokens && appended <= self.sequence_tokens,
            "estados KV no comparables"
        );
        let start = self.sequence_tokens - appended;
        let mut first_divergent_layer = None;
        let mut divergent_layers = 0_usize;
        let mut key_mismatched_elements = 0_usize;
        let mut value_mismatched_elements = 0_usize;
        let mut elements_compared = 0_usize;
        let mut max_abs_key_difference = 0.0_f64;
        let mut max_abs_value_difference = 0.0_f64;
        for layer in 0..self.layers.len() {
            let (left_key, left_value) = self.layers[layer]
                .as_ref()
                .with_context(|| format!("capa secuencial {layer} sin KV"))?;
            let (right_key, right_value) = other.layers[layer]
                .as_ref()
                .with_context(|| format!("capa batch {layer} sin KV"))?;
            let mut layer_diverged = false;
            for (left, right, mismatch_total, max_difference) in [
                (
                    left_key,
                    right_key,
                    &mut key_mismatched_elements,
                    &mut max_abs_key_difference,
                ),
                (
                    left_value,
                    right_value,
                    &mut value_mismatched_elements,
                    &mut max_abs_value_difference,
                ),
            ] {
                let left = left.narrow(2, start as i64, appended as i64).contiguous();
                let right = right.narrow(2, start as i64, appended as i64).contiguous();
                ensure!(left.size() == right.size(), "colas KV incompatibles");
                // SAFETY: both tails are contiguous owned views of this CPU
                // runtime's BF16 cache, aligned for u16, with numel() initialized
                // elements. They remain alive and unmodified throughout comparison.
                let left_bits = unsafe {
                    std::slice::from_raw_parts(left.data_ptr().cast::<u16>(), left.numel())
                };
                // SAFETY: same CPU/BF16, alignment, extent and lifetime contract
                // as left_bits above; this slice is also used only within this scope.
                let right_bits = unsafe {
                    std::slice::from_raw_parts(right.data_ptr().cast::<u16>(), right.numel())
                };
                let mismatches = left_bits
                    .iter()
                    .zip(right_bits)
                    .filter(|(left, right)| left != right)
                    .count();
                *mismatch_total += mismatches;
                layer_diverged |= mismatches > 0;
                let difference = (left.to_kind(Kind::Float) - right.to_kind(Kind::Float))
                    .abs()
                    .amax([0_i64, 1_i64, 2_i64, 3_i64], false)
                    .double_value(&[]);
                *max_difference = (*max_difference).max(difference);
                elements_compared += left.numel();
            }
            if layer_diverged {
                first_divergent_layer.get_or_insert(layer);
                divergent_layers += 1;
            }
        }
        Ok(KvStateDifference {
            first_divergent_layer,
            divergent_layers,
            key_mismatched_elements,
            value_mismatched_elements,
            elements_compared,
            max_abs_key_difference,
            max_abs_value_difference,
        })
    }

    pub(super) fn truncate_to(&mut self, sequence_tokens: usize) -> Result<()> {
        ensure!(
            sequence_tokens <= self.sequence_tokens,
            "rollback KV denso hacia delante"
        );
        for (layer, state) in self.layers.iter_mut().enumerate() {
            let (key, value) = state
                .as_mut()
                .with_context(|| format!("capa densa {layer} sin estado para rollback"))?;
            ensure!(
                key.size()[2] as usize >= sequence_tokens
                    && value.size()[2] as usize >= sequence_tokens,
                "rollback KV fuera de la capa"
            );
            *key = key.narrow(2, 0, sequence_tokens as i64).contiguous();
            *value = value.narrow(2, 0, sequence_tokens as i64).contiguous();
        }
        self.sequence_tokens = sequence_tokens;
        Ok(())
    }
}

pub(super) struct KvStateDifference {
    pub(super) first_divergent_layer: Option<usize>,
    pub(super) divergent_layers: usize,
    pub(super) key_mismatched_elements: usize,
    pub(super) value_mismatched_elements: usize,
    pub(super) elements_compared: usize,
    pub(super) max_abs_key_difference: f64,
    pub(super) max_abs_value_difference: f64,
}

impl Default for DenseKvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionCache for DenseKvCache {
    fn sequence_tokens(&self) -> usize {
        self.sequence_tokens
    }

    fn physical_bytes(&self) -> usize {
        DenseKvCache::physical_bytes(self)
    }

    fn attend(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<Tensor> {
        ensure!(
            layer_index < self.layers.len(),
            "índice de capa KV inválido"
        );
        let (key, value) = match self.layers[layer_index].as_ref() {
            Some((old_key, old_value)) => (
                Tensor::cat(&[old_key, new_key], 2),
                Tensor::cat(&[old_value, new_value], 2),
            ),
            None => (new_key.shallow_clone(), new_value.shallow_clone()),
        };
        self.layers[layer_index] = Some((key.shallow_clone(), value.shallow_clone()));
        let groups = Self::gqa_groups(query, &key)?;
        let key = key.repeat_interleave_self_int(groups, 1, None);
        let value = value.repeat_interleave_self_int(groups, 1, None);
        let query_tokens = query.size()[2] as usize;
        let total_tokens = self.sequence_tokens + query_tokens;
        if self.querywise_attention && query_tokens > 1 {
            let mut outputs = Vec::with_capacity(query_tokens);
            for query_index in 0..query_tokens {
                let attended_tokens = self.sequence_tokens + query_index + 1;
                let query_token = query.narrow(2, query_index as i64, 1);
                let key_prefix = key.narrow(2, 0, attended_tokens as i64);
                let value_prefix = value.narrow(2, 0, attended_tokens as i64);
                let mask = Tensor::zeros(
                    [1, 1, 1, attended_tokens as i64],
                    (query.kind(), query.device()),
                );
                outputs.push(Tensor::scaled_dot_product_attention(
                    &query_token,
                    &key_prefix,
                    &value_prefix,
                    Some(&mask),
                    0.0,
                    false,
                    None,
                    false,
                ));
            }
            let output_refs = outputs.iter().collect::<Vec<_>>();
            return Ok(Tensor::cat(&output_refs, 2));
        }
        let mut mask_values = vec![0_f32; query_tokens * total_tokens];
        for query_index in 0..query_tokens {
            for key_index in self.sequence_tokens + query_index + 1..total_tokens {
                mask_values[query_index * total_tokens + key_index] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_slice(&mask_values)
            .view([1, 1, query_tokens as i64, total_tokens as i64])
            .to_kind(query.kind());
        Ok(Tensor::scaled_dot_product_attention(
            query,
            &key,
            &value,
            Some(&mask),
            0.0,
            false,
            None,
            false,
        ))
    }

    fn finish_step(&mut self, added_tokens: usize) {
        self.sequence_tokens += added_tokens;
    }
}

pub(super) struct MslimPage {
    pub(super) token_start: usize,
    pub(super) tokens: usize,
    pub(super) certificate: PageAttentionCertificate,
    pub(super) storage: PageStorage,
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct PageAttentionCertificate {
    pub(super) regions: Vec<KeyRegionCertificate>,
    /// Historical configured-model 2KV bound. Kept unchanged for v1 prefix
    /// compatibility and the original 16Q/2KV branch-and-bound path.
    pub(super) value_abs_max: [f32; 2],
    /// New, optional exact BF16 V bound for ALL KV heads (8 for Llama).
    /// A missing field in a legacy persisted prefix FAILS CLOSED
    /// to full dense BF16 V reads; it never authorizes an omission.
    #[serde(default)]
    pub(super) value_abs_max_by_kv_head: Vec<f32>,
    /// Per-KV-head, per-dimension maxima in head-major order. Legacy
    /// manifests lack this field and cannot certify component-wise omission.
    #[serde(default)]
    pub(super) value_abs_max_by_kv_head_dim: Vec<f32>,
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct KeyRegionCertificate {
    pub(super) block_id: usize,
    pub(super) state_id: usize,
    pub(super) local_start: usize,
    pub(super) local_end: usize,
    pub(super) key_minimum_bits: Vec<u16>,
    pub(super) key_maximum_bits: Vec<u16>,
    pub(super) value_abs_max: [f32; 2],
    /// Exact per-KV-head regional V maxima. Legacy prefixes omit this field
    /// and therefore cannot authorize a Llama 8-KV-head regional omission.
    #[serde(default)]
    pub(super) value_abs_max_by_kv_head: Vec<f32>,
}

pub(super) struct ParallelAxisMetrics {
    pub(super) extents: usize,
    pub(super) tokens: usize,
    pub(super) fingerprint_payload_bytes: usize,
    pub(super) f32_fingerprint_reference_bytes: usize,
    pub(super) micro_metadata_bytes: usize,
    pub(super) extent_min: usize,
    pub(super) extent_max: usize,
    pub(super) query_events: usize,
    pub(super) dimensional_views: usize,
    pub(super) view_tokens: usize,
    pub(super) lsh_queries: usize,
    pub(super) lsh_candidates: usize,
    pub(super) lsh_misses: usize,
    pub(super) lsh_probe_two: usize,
    pub(super) lsh_probe_four: usize,
}

pub(super) enum PageStorage {
    Cold {
        exact_id: u64,
        immutable: bool,
    },
    Pending {
        sequence: u64,
        key: Tensor,
        value: Tensor,
    },
}

#[derive(Deserialize, Serialize)]
pub(super) struct PersistentPrefixPage {
    pub(super) token_start: usize,
    pub(super) tokens: usize,
    pub(super) certificate: PageAttentionCertificate,
    pub(super) exact_id: u64,
}

#[derive(Deserialize, Serialize)]
pub(super) struct PersistentPrefixManifest {
    pub(super) schema: u32,
    pub(super) model_identity: String,
    pub(super) prefix_digest: String,
    pub(super) prefix_tokens: usize,
    pub(super) storage_frame_tokens: usize,
    pub(super) layers: Vec<Vec<PersistentPrefixPage>>,
    pub(super) parallel_axis: ParallelContextAxis,
}

const PREFIX_CACHE_SCHEMA: u32 = 1;

impl Clone for MslimPage {
    fn clone(&self) -> Self {
        Self {
            token_start: self.token_start,
            tokens: self.tokens,
            certificate: self.certificate.clone(),
            storage: match &self.storage {
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => PageStorage::Cold {
                    exact_id: *exact_id,
                    immutable: *immutable,
                },
                PageStorage::Pending {
                    sequence,
                    key,
                    value,
                } => PageStorage::Pending {
                    sequence: *sequence,
                    key: key.shallow_clone(),
                    value: value.shallow_clone(),
                },
            },
        }
    }
}

pub(super) struct MslimLayer {
    pub(super) pages: Vec<MslimPage>,
    pub(super) tail: Option<(Tensor, Tensor)>,
    pub(super) prefill_hot: Option<(Tensor, Tensor)>,
    /// Verified resident copy of `pages`, see [`ResidentCommittedKv`].
    pub(super) resident: Option<ResidentCommittedKv>,
}

/// Resident, already verified BF16 copy of a layer's committed pages.
///
/// Pages are append-only and immutable once sealed, so each cold page is
/// read and checksum-verified once, copied into a preallocated buffer that
/// grows geometrically, and every later block or decode step reuses it
/// instead of re-reading (and re-hashing) the whole history from disk. The
/// returned history is a view of the first `tokens` positions, which are
/// never rewritten, so its bytes are identical to the old `Tensor::cat`.
pub(super) struct ResidentCommittedKv {
    pub(super) key: Tensor,
    pub(super) value: Tensor,
    pub(super) tokens: usize,
    pub(super) capacity: usize,
    pub(super) spans: Vec<(usize, usize)>,
}

/// Default RAM budget for resident committed KV: `MSLIM_RESIDENT_KV_BYTES`
/// when set (0 disables it and restores the cold re-read path), otherwise a
/// third of physical memory capped at 24 GiB (Llama 3.1 8B at 111k tokens
/// needs ~13.6 GiB).
fn default_resident_kv_budget_bytes() -> usize {
    if let Ok(raw) = std::env::var("MSLIM_RESIDENT_KV_BYTES")
        && let Ok(bytes) = raw.trim().parse::<usize>()
    {
        return bytes;
    }
    let cap = 24_usize << 30;
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|meminfo| {
            meminfo
                .lines()
                .find_map(|line| line.strip_prefix("MemTotal:"))
                .and_then(|rest| {
                    rest.trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
        })
        .map_or(cap, |kib| (kib.saturating_mul(1024) / 3).min(cap))
}

pub struct MslimKvCache {
    pub(super) layers: Vec<MslimLayer>,
    pub(super) kv_heads: i64,
    pub(super) head_dim: i64,
    pub(super) query_heads: i64,
    pub(super) sequence_tokens: usize,
    pub(super) _cold_directory: tempfile::TempDir,
    pub(super) exact_cold: ColdStore,
    pub(super) conversion: ConversionWorker<PreparedPage>,
    pub(super) storage_frame_tokens: usize,
    pub(super) parallel_axis: ParallelContextAxis,
    pub(super) parallel_query_events: usize,
    pub(super) parallel_dimensional_views: usize,
    pub(super) parallel_view_tokens: usize,
    pub(super) parallel_selected_positions: Vec<usize>,
    pub(super) exact_k_tokens_scanned: usize,
    pub(super) exact_v_tokens_materialized: usize,
    pub(super) exact_mass_max_bound: f64,
    pub(super) decode_metadata_nanos: u64,
    pub(super) decode_key_materialization_nanos: u64,
    pub(super) decode_exact_scores_nanos: u64,
    pub(super) decode_region_bound_nanos: u64,
    pub(super) decode_key_assembly_nanos: u64,
    pub(super) decode_value_materialization_nanos: u64,
    pub(super) decode_sdpa_nanos: u64,
    /// Operation-count accounting for the attention authority. These are
    /// logical BF16 element counts, not claims about opaque LibTorch kernels.
    pub(super) attention_dense_qk_elements: u128,
    pub(super) attention_dense_pv_elements: u128,
    pub(super) attention_selected_qk_elements: u128,
    pub(super) attention_selected_pv_elements: u128,
    pub(super) attention_v_tokens_avoided: usize,
    /// Work submitted by the canonical dense path to LibTorch SDPA.  The
    /// scheduled counters describe the rectangular tensors handed to the
    /// opaque kernel; the causal counters describe the non-masked prefix
    /// pairs.  They are deliberately separate from the omission baseline
    /// above, which is also populated by diagnostic/selective paths.
    pub(super) dense_sdpa_calls: usize,
    pub(super) dense_sdpa_scheduled_qk_elements: u128,
    pub(super) dense_sdpa_scheduled_pv_elements: u128,
    pub(super) dense_sdpa_causal_qk_elements: u128,
    pub(super) dense_sdpa_causal_pv_elements: u128,
    pub(super) dense_sdpa_nanos: u64,
    /// Maximum known temporary allocation footprint of one autoregressive
    /// attention call. This is an accounting lower bound, not process RSS:
    /// opaque libtorch/SDPA workspace is reported separately as unknown.
    pub(super) known_attention_scratch_peak_bytes: usize,
    pub(super) prefill_working_peak_bytes: usize,
    pub(super) resident_kv_budget_bytes: usize,
    pub(super) resident_kv_bytes: usize,
    pub(super) resident_kv_peak_bytes: usize,
    pub(super) resident_kv_pages_reused: usize,
    pub(super) resident_kv_fallbacks: usize,
    pub(super) prefill_axis_nanos: u64,
    pub(super) prefill_hot_concat_nanos: u64,
    pub(super) prefill_certificate_nanos: u64,
    pub(super) prefill_handoff_nanos: u64,
    pub(super) prefill_commit_nanos: u64,
    pub(super) hierarchy_events: usize,
    pub(super) hierarchy_states_total: usize,
    pub(super) hierarchy_states_initial: usize,
    pub(super) hierarchy_states_loaded: usize,
    pub(super) hierarchy_widening_rounds: usize,
    pub(super) hierarchy_k_tokens_avoided: usize,
    pub(super) hierarchy_k_events: usize,
    pub(super) hierarchy_k_tokens_skippable: usize,
    pub(super) hierarchy_last_states_initial: usize,
    pub(super) hierarchy_last_states_loaded: usize,
    pub(super) hierarchy_last_widening_rounds: usize,
    pub(super) hierarchy_last_region_bound: f64,
    pub(super) hierarchy_last_token_bound: f64,
    pub(super) exact_oracle_events: usize,
    pub(super) exact_oracle_token_opportunities: usize,
    pub(super) exact_oracle_tokens_omissible_at_1e6: usize,
    pub(super) exact_oracle_tokens_omissible_at_1e5: usize,
    pub(super) exact_oracle_tokens_omissible_at_1e4: usize,
    pub(super) exact_oracle_tokens_omissible_at_1e3: usize,
    pub(super) exact_oracle_min_tolerance_for_one_omission: f64,
    pub(super) exact_oracle_enabled: bool,
    pub(super) shadow_metrics: Option<ShadowMetrics>,
    /// Explicit transaction mode for exact multi-token target verification.
    /// It is never inferred from tensor length because prefill is multi-token.
    pub(super) multitoken_verification: bool,
    pub(super) prefix_cold: Option<ColdStore>,
    pub(super) applied_mass_tolerance: Option<f64>,
    pub(super) prefix_copied_bytes: u64,
    pub(super) record_step_logits: bool,
    pub(super) step_logits: Vec<Vec<f32>>,
    pub(super) layer_output_traces: Vec<LayerOutputTrace>,
    pub(super) verify_exact_attention_outputs: bool,
    pub(super) exact_attention_verified_events: usize,
    pub(super) exact_attention_equal_events: usize,
    pub(super) exact_attention_candidate_v_tokens: usize,
    pub(super) exact_attention_rejected_v_tokens: usize,
    pub(super) rounding_cell_checked_events: usize,
    pub(super) rounding_cell_certified_events: usize,
    pub(super) rounding_cell_components_checked: usize,
    pub(super) rounding_cell_components_certified: usize,
    pub(super) suffix_against_committed: bool,
}

impl MslimKvCache {
    pub(super) fn conversion_worker(first_sequence: u64) -> Result<ConversionWorker<PreparedPage>> {
        ConversionWorker::spawn_from(16, 1, first_sequence, move |job| {
            let geometry = PageGeometry {
                layer: job.shape.layer as u32,
                token_start: job.shape.token_start as u64,
                tokens: job.shape.tokens as u32,
                heads: job.shape.heads as u32,
                head_dim: job.shape.dim as u32,
                tile: job.shape.tile as u32,
            };
            ColdStore::prepare_page(
                job.sequence,
                geometry,
                Self::u16_bytes(&job.key_bf16),
                Self::u16_bytes(&job.value_bf16),
            )
            .map_err(|error| error.to_string())
        })
        .map_err(anyhow::Error::msg)
        .context("workers de preparación KV")
    }

    pub fn try_new_with_storage_frame_tokens(storage_frame_tokens: usize) -> Result<Self> {
        Self::try_new_with_geometry(storage_frame_tokens, 24, 2, 16, 128)
    }

    pub(super) fn try_new_with_geometry(
        storage_frame_tokens: usize,
        n_layers: usize,
        kv_heads: i64,
        query_heads: i64,
        head_dim: i64,
    ) -> Result<Self> {
        ensure!(
            (16..=512).contains(&storage_frame_tokens) && storage_frame_tokens.is_power_of_two(),
            "storage_frame_tokens debe ser potencia de dos entre 16 y 512"
        );
        ensure!(
            n_layers > 0
                && kv_heads > 0
                && query_heads > 0
                && head_dim > 0
                && query_heads % kv_heads == 0,
            "geometría KV M-SLIM inválida"
        );
        let cold_directory = tempfile::tempdir().context("directorio KV frío")?;
        let exact_cold = ColdStore::create(cold_directory.path().join("pages-exact.kv"), 0)
            .context("almacén BF16 exacto frío")?;
        let conversion = Self::conversion_worker(0)?;
        let axis_dim = usize::try_from(kv_heads * head_dim).context("dimensión de eje")?;
        let axis_heads = usize::try_from(kv_heads).context("cabezas KV")?;
        Ok(Self {
            layers: (0..n_layers)
                .map(|_| MslimLayer {
                    pages: Vec::new(),
                    tail: None,
                    prefill_hot: None,
                    resident: None,
                })
                .collect(),
            kv_heads,
            head_dim,
            query_heads,
            sequence_tokens: 0,
            _cold_directory: cold_directory,
            exact_cold,
            conversion,
            storage_frame_tokens,
            parallel_axis: ParallelContextAxis::new_partitioned(
                axis_dim,
                axis_heads,
                AxisPolicy::default(),
            )?,
            parallel_query_events: 0,
            parallel_dimensional_views: 0,
            parallel_view_tokens: 0,
            parallel_selected_positions: Vec::new(),
            exact_k_tokens_scanned: 0,
            exact_v_tokens_materialized: 0,
            exact_mass_max_bound: 0.0,
            decode_metadata_nanos: 0,
            decode_key_materialization_nanos: 0,
            decode_exact_scores_nanos: 0,
            decode_region_bound_nanos: 0,
            decode_key_assembly_nanos: 0,
            decode_value_materialization_nanos: 0,
            decode_sdpa_nanos: 0,
            attention_dense_qk_elements: 0,
            attention_dense_pv_elements: 0,
            attention_selected_qk_elements: 0,
            attention_selected_pv_elements: 0,
            attention_v_tokens_avoided: 0,
            dense_sdpa_calls: 0,
            dense_sdpa_scheduled_qk_elements: 0,
            dense_sdpa_scheduled_pv_elements: 0,
            dense_sdpa_causal_qk_elements: 0,
            dense_sdpa_causal_pv_elements: 0,
            dense_sdpa_nanos: 0,
            known_attention_scratch_peak_bytes: 0,
            prefill_working_peak_bytes: 0,
            resident_kv_budget_bytes: default_resident_kv_budget_bytes(),
            resident_kv_bytes: 0,
            resident_kv_peak_bytes: 0,
            resident_kv_pages_reused: 0,
            resident_kv_fallbacks: 0,
            prefill_axis_nanos: 0,
            prefill_hot_concat_nanos: 0,
            prefill_certificate_nanos: 0,
            prefill_handoff_nanos: 0,
            prefill_commit_nanos: 0,
            hierarchy_events: 0,
            hierarchy_states_total: 0,
            hierarchy_states_initial: 0,
            hierarchy_states_loaded: 0,
            hierarchy_widening_rounds: 0,
            hierarchy_k_tokens_avoided: 0,
            hierarchy_k_events: 0,
            hierarchy_k_tokens_skippable: 0,
            hierarchy_last_states_initial: 0,
            hierarchy_last_states_loaded: 0,
            hierarchy_last_widening_rounds: 0,
            hierarchy_last_region_bound: 0.0,
            hierarchy_last_token_bound: 0.0,
            exact_oracle_events: 0,
            exact_oracle_token_opportunities: 0,
            exact_oracle_tokens_omissible_at_1e6: 0,
            exact_oracle_tokens_omissible_at_1e5: 0,
            exact_oracle_tokens_omissible_at_1e4: 0,
            exact_oracle_tokens_omissible_at_1e3: 0,
            exact_oracle_min_tolerance_for_one_omission: f64::INFINITY,
            exact_oracle_enabled: false,
            shadow_metrics: None,
            multitoken_verification: false,
            prefix_cold: None,
            applied_mass_tolerance: None,
            prefix_copied_bytes: 0,
            record_step_logits: false,
            step_logits: Vec::new(),
            layer_output_traces: Vec::new(),
            verify_exact_attention_outputs: false,
            exact_attention_verified_events: 0,
            exact_attention_equal_events: 0,
            exact_attention_candidate_v_tokens: 0,
            exact_attention_rejected_v_tokens: 0,
            rounding_cell_checked_events: 0,
            rounding_cell_certified_events: 0,
            rounding_cell_components_checked: 0,
            rounding_cell_components_certified: 0,
            suffix_against_committed: false,
        })
    }

    pub(super) fn cold_store(&mut self, immutable: bool) -> Result<&mut ColdStore> {
        if immutable {
            self.prefix_cold
                .as_mut()
                .context("página inmutable sin almacén de prefijo")
        } else {
            Ok(&mut self.exact_cold)
        }
    }

    pub(super) fn read_cold_key(&mut self, exact_id: u64, immutable: bool) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_key(exact_id)
            .map_err(anyhow::Error::from)
    }

    pub(super) fn read_cold_value(&mut self, exact_id: u64, immutable: bool) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_value(exact_id)
            .map_err(anyhow::Error::from)
    }

    pub(super) fn read_cold_key_tokens(
        &mut self,
        exact_id: u64,
        immutable: bool,
        local: &[usize],
    ) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_key_tokens(exact_id, local)
            .map_err(anyhow::Error::from)
    }

    pub(super) fn read_cold_value_tokens(
        &mut self,
        exact_id: u64,
        immutable: bool,
        local: &[usize],
    ) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_value_tokens(exact_id, local)
            .map_err(anyhow::Error::from)
    }

    /// Exact BF16 K/V tensors of `pages`, reading (and verifying) cold pages.
    pub(super) fn committed_page_tensors(
        &mut self,
        pages: &[MslimPage],
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut keys = Vec::with_capacity(pages.len());
        let mut values = Vec::with_capacity(pages.len());
        for page in pages {
            match &page.storage {
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => {
                    let key = self.read_cold_key(*exact_id, *immutable)?;
                    let value = self.read_cold_value(*exact_id, *immutable)?;
                    keys.push(self.exact_tensor_from_bytes(&key, page.tokens)?);
                    values.push(self.exact_tensor_from_bytes(&value, page.tokens)?);
                }
                PageStorage::Pending { key, value, .. } => {
                    keys.push(key.shallow_clone());
                    values.push(value.shallow_clone());
                }
            }
        }
        Ok((keys, values))
    }

    pub(super) fn resident_bytes_for(&self, capacity: usize) -> usize {
        2 * self.kv_heads as usize * capacity * self.head_dim as usize * 2
    }

    pub(super) fn release_resident(&mut self, layer_index: usize) {
        if let Some(resident) = self.layers[layer_index].resident.take() {
            self.resident_kv_bytes = self
                .resident_kv_bytes
                .saturating_sub(self.resident_bytes_for(resident.capacity));
        }
    }

    pub(super) fn assemble_committed_layer(
        &mut self,
        layer_index: usize,
    ) -> Result<(Tensor, Tensor)> {
        let pages = self.layers[layer_index].pages.clone();
        ensure!(!pages.is_empty(), "capa comprometida vacía");
        let spans = pages
            .iter()
            .map(|page| (page.token_start, page.tokens))
            .collect::<Vec<_>>();
        let total_tokens = spans.iter().map(|(_, tokens)| *tokens).sum::<usize>();
        // A resident copy is reused only while its pages are still an exact
        // prefix of the layer (pages are append-only; anything else drops it).
        if let Some(resident) = self.layers[layer_index].resident.as_ref()
            && (resident.spans.len() > spans.len()
                || spans[..resident.spans.len()] != resident.spans[..])
        {
            self.release_resident(layer_index);
        }
        let (old_tokens, old_capacity, first_new) = self.layers[layer_index]
            .resident
            .as_ref()
            .map_or((0, 0, 0), |resident| {
                (resident.tokens, resident.capacity, resident.spans.len())
            });
        // Grow geometrically (x1.5) when the budget allows, else exactly.
        let others = self.resident_kv_bytes - self.resident_bytes_for(old_capacity);
        let capacity = if total_tokens <= old_capacity {
            Some(old_capacity)
        } else {
            [
                total_tokens.max(old_capacity + old_capacity / 2),
                total_tokens,
            ]
            .into_iter()
            .find(|&capacity| {
                others + self.resident_bytes_for(capacity) <= self.resident_kv_budget_bytes
            })
        };
        let Some(capacity) = capacity else {
            // Over budget (or disabled with 0): original cold path, which
            // re-reads and re-verifies every page on each call.
            if first_new > 0 || self.resident_kv_budget_bytes > 0 {
                self.resident_kv_fallbacks += 1;
            }
            self.release_resident(layer_index);
            let (keys, values) = self.committed_page_tensors(&pages)?;
            let key_refs = keys.iter().collect::<Vec<_>>();
            let value_refs = values.iter().collect::<Vec<_>>();
            return Ok((Tensor::cat(&key_refs, 2), Tensor::cat(&value_refs, 2)));
        };
        let (new_keys, new_values) = self.committed_page_tensors(&pages[first_new..])?;
        self.resident_kv_pages_reused += first_new;
        let mut resident = match self.layers[layer_index].resident.take() {
            Some(resident) if resident.capacity == capacity => resident,
            previous => {
                let options = new_keys
                    .first()
                    .map(|tensor| (tensor.kind(), tensor.device()))
                    .unwrap_or((Kind::BFloat16, Device::Cpu));
                let shape = [1, self.kv_heads, capacity as i64, self.head_dim];
                let mut grown = ResidentCommittedKv {
                    key: Tensor::empty(shape, options),
                    value: Tensor::empty(shape, options),
                    tokens: 0,
                    capacity,
                    spans: Vec::new(),
                };
                self.resident_kv_peak_bytes = self
                    .resident_kv_peak_bytes
                    .max(self.resident_kv_bytes + self.resident_bytes_for(capacity));
                if let Some(previous) = previous {
                    if previous.tokens > 0 {
                        let n = previous.tokens as i64;
                        grown
                            .key
                            .narrow(2, 0, n)
                            .copy_(&previous.key.narrow(2, 0, n));
                        grown
                            .value
                            .narrow(2, 0, n)
                            .copy_(&previous.value.narrow(2, 0, n));
                    }
                    grown.tokens = previous.tokens;
                    grown.spans = previous.spans;
                    self.resident_kv_bytes -= self.resident_bytes_for(previous.capacity);
                }
                self.resident_kv_bytes += self.resident_bytes_for(capacity);
                grown
            }
        };
        debug_assert_eq!(resident.tokens, old_tokens);
        for ((key, value), span) in new_keys.iter().zip(&new_values).zip(&spans[first_new..]) {
            let offset = resident.tokens as i64;
            let tokens = span.1 as i64;
            ensure!(
                key.size()[2] == tokens && value.size()[2] == tokens,
                "página residente con geometría inconsistente"
            );
            resident.key.narrow(2, offset, tokens).copy_(key);
            resident.value.narrow(2, offset, tokens).copy_(value);
            resident.tokens += span.1;
            resident.spans.push(*span);
        }
        self.resident_kv_peak_bytes = self.resident_kv_peak_bytes.max(self.resident_kv_bytes);
        let history = (
            resident.key.narrow(2, 0, total_tokens as i64),
            resident.value.narrow(2, 0, total_tokens as i64),
        );
        self.layers[layer_index].resident = Some(resident);
        Ok(history)
    }

    pub(super) fn begin_multitoken_verification(&mut self) -> Result<()> {
        ensure!(
            !self.multitoken_verification,
            "verificación multí-token ya activa"
        );
        ensure!(
            self.layers.iter().all(|layer| layer.prefill_hot.is_none()),
            "verificación multí-token antes de sellar el prefill"
        );
        self.multitoken_verification = true;
        Ok(())
    }

    pub(super) fn end_multitoken_verification(&mut self) {
        self.multitoken_verification = false;
    }

    /// Roll back only provisional, unsealed tokens. Multi-token verification
    /// deliberately never submits them to conversion workers, so this cannot
    /// orphan a background job or mutate cold storage.
    pub(super) fn rollback_unsealed_to(&mut self, sequence_tokens: usize) -> Result<()> {
        ensure!(
            sequence_tokens <= self.sequence_tokens,
            "rollback KV frío hacia delante"
        );
        ensure!(
            !self.multitoken_verification,
            "rollback durante una verificación activa"
        );
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let sealed_tokens = layer.pages.iter().map(|page| page.tokens).sum::<usize>();
            ensure!(
                sequence_tokens >= sealed_tokens,
                "rollback de capa {layer_index} atravesaría almacenamiento sellado"
            );
            let desired_tail = sequence_tokens - sealed_tokens;
            let current_tail = layer
                .tail
                .as_ref()
                .map_or(0, |(key, _)| key.size()[2] as usize);
            ensure!(
                desired_tail <= current_tail,
                "rollback de capa {layer_index} fuera de la cola provisional"
            );
            layer.tail = if desired_tail == 0 {
                None
            } else {
                let (key, value) = layer.tail.as_ref().context("cola provisional ausente")?;
                Some((
                    key.narrow(2, 0, desired_tail as i64).contiguous(),
                    value.narrow(2, 0, desired_tail as i64).contiguous(),
                ))
            };
        }
        self.sequence_tokens = sequence_tokens;
        Ok(())
    }

    /// Hash the logical BF16 K/V state in the same layer-major order as the
    /// dense cache. Cold pages are verified by `ColdStore` while being read;
    /// the digest is diagnostic and does not become serving authority.
    pub(super) fn logical_state_digest(&mut self) -> Result<String> {
        self.collect_conversions()?;
        let mut digest = Sha256::new();
        digest.update(b"mslim-dense-kv-state-v1");
        digest.update((self.sequence_tokens as u64).to_le_bytes());
        let heads = usize::try_from(self.kv_heads).context("invalid KV heads for digest")?;
        let dimension = usize::try_from(self.head_dim).context("invalid KV head dimension")?;
        ensure!(
            heads > 0 && dimension > 0,
            "digest with invalid KV geometry"
        );
        for layer_index in 0..self.layers.len() {
            let pages = self.layers[layer_index].pages.clone();
            let tail = self.layers[layer_index]
                .tail
                .as_ref()
                .map(|(key, value)| (key.shallow_clone(), value.shallow_clone()));
            // The logical [1, kv_heads, tokens, head_dim] cache stores
            // head-major tokens. Cover every KV head and dimension, including
            // Llama 3.1 8B's eight heads, across physical page boundaries.
            for values in [false, true] {
                for head in 0..heads {
                    for page in &pages {
                        match &page.storage {
                            PageStorage::Cold {
                                exact_id,
                                immutable,
                            } => {
                                let bytes = if values {
                                    self.read_cold_value(*exact_id, *immutable)?
                                } else {
                                    self.read_cold_key(*exact_id, *immutable)?
                                };
                                let head_bytes = page.tokens * dimension * 2;
                                let expected = heads * head_bytes;
                                ensure!(
                                    bytes.len() == expected,
                                    "cold KV digest geometry mismatch in layer {layer_index}"
                                );
                                let start = head * head_bytes;
                                digest.update(&bytes[start..start + head_bytes]);
                            }
                            PageStorage::Pending { key, value, .. } => {
                                let tensor = if values { value } else { key }
                                    .narrow(1, head as i64, 1)
                                    .contiguous();
                                ensure!(tensor.kind() == Kind::BFloat16, "estado KV no BF16");
                                // SAFETY: pending pages are CPU BF16 tensors; narrow
                                // and contiguous retain owned storage through digest.update.
                                // Read exactly numel()*2 initialized bytes, with no mutation.
                                let bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        tensor.data_ptr().cast::<u8>(),
                                        tensor.numel() * 2,
                                    )
                                };
                                digest.update(bytes);
                            }
                        }
                    }
                    if let Some((key, value)) = &tail {
                        let tensor = if values { value } else { key }
                            .narrow(1, head as i64, 1)
                            .contiguous();
                        ensure!(tensor.kind() == Kind::BFloat16, "cola KV no BF16");
                        // SAFETY: the owned contiguous CPU BF16 tail outlives
                        // this read-only byte view and digest.update; the view spans
                        // exactly numel()*2 initialized bytes and never escapes.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                tensor.data_ptr().cast::<u8>(),
                                tensor.numel() * 2,
                            )
                        };
                        digest.update(bytes);
                    }
                }
            }
        }
        Ok(digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }

    pub(super) fn persistent_manifest(
        &self,
        model_identity: &str,
        prefix_digest: &str,
    ) -> Result<PersistentPrefixManifest> {
        ensure!(
            self.tail_tokens() == 0,
            "prefijo persistente conserva una cola"
        );
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                layer
                    .pages
                    .iter()
                    .map(|page| {
                        let exact_id = match page.storage {
                            PageStorage::Cold { exact_id, .. } => exact_id,
                            PageStorage::Pending { .. } => {
                                bail!("prefijo persistente contiene una página pendiente")
                            }
                        };
                        Ok(PersistentPrefixPage {
                            token_start: page.token_start,
                            tokens: page.tokens,
                            certificate: page.certificate.clone(),
                            exact_id,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PersistentPrefixManifest {
            schema: PREFIX_CACHE_SCHEMA,
            model_identity: model_identity.to_owned(),
            prefix_digest: prefix_digest.to_owned(),
            prefix_tokens: self.sequence_tokens,
            storage_frame_tokens: self.storage_frame_tokens,
            layers,
            parallel_axis: self.parallel_axis.clone(),
        })
    }

    pub(super) fn persist_prefix(
        &mut self,
        cache_root: &Path,
        cache_key: &str,
        model_identity: &str,
    ) -> Result<()> {
        self.drain_conversions()?;
        self.exact_cold
            .audit_mapped()
            .context("auditoría antes de persistir prefijo")?;
        fs::create_dir_all(cache_root)
            .with_context(|| format!("crear caché de prefijos {}", cache_root.display()))?;
        let destination = cache_root.join(cache_key);
        let manifest = self.persistent_manifest(model_identity, cache_key)?;
        let manifest_bytes = serde_json::to_vec(&manifest).context("serializar prefijo")?;
        let mut copied_bytes = 0_u64;
        if !destination.is_dir() {
            let staging = tempfile::Builder::new()
                .prefix(".mslim-prefix-")
                .tempdir_in(cache_root)
                .context("staging de prefijo persistente")?;
            let published_pages = staging.path().join("pages-exact.kv");
            // The inference worker owns this mutable store exclusively and no
            // suffix is processed until publication returns.  A hard link is
            // therefore an O(1) snapshot on the same filesystem.  Immediately
            // after atomic publication `adopt_published_prefix` unlinks the
            // mutable name, reopens the published inode read-only and creates
            // a fresh suffix store, so later appends cannot mutate the prefix.
            if fs::hard_link(self.exact_cold.path(), &published_pages).is_err() {
                copied_bytes = fs::copy(self.exact_cold.path(), &published_pages)
                    .context("copiar KV de prefijo entre sistemas de archivos")?;
            }
            fs::write(staging.path().join("manifest.json"), &manifest_bytes)
                .context("escribir manifiesto de prefijo")?;
            let staging_path = staging.keep();
            match fs::rename(&staging_path, &destination) {
                Ok(()) => {}
                Err(_error) if destination.is_dir() => {
                    fs::remove_dir_all(&staging_path).ok();
                }
                Err(error) => {
                    fs::remove_dir_all(&staging_path).ok();
                    return Err(error).context("publicar prefijo persistente");
                }
            }
        }
        self.adopt_published_prefix(&destination, &manifest_bytes, copied_bytes)
    }

    pub(super) fn adopt_published_prefix(
        &mut self,
        destination: &Path,
        expected_manifest: &[u8],
        copied_bytes: u64,
    ) -> Result<()> {
        let published_manifest = fs::read(destination.join("manifest.json"))
            .context("leer manifiesto recién publicado")?;
        ensure!(
            published_manifest == expected_manifest,
            "colisión de publicación con manifiesto de prefijo distinto"
        );
        let mut prefix_cold = ColdStore::open_read_only(destination.join("pages-exact.kv"), 0)
            .context("abrir prefijo recién publicado")?;
        prefix_cold
            .audit_mapped()
            .context("auditar prefijo recién publicado")?;
        let expected_pages = self
            .layers
            .iter()
            .map(|layer| layer.pages.len())
            .sum::<usize>();
        ensure!(
            prefix_cold.metadata().len() == expected_pages,
            "prefijo publicado no contiene todas las páginas"
        );
        ensure!(
            self.layers.iter().all(|layer| layer.prefill_hot.is_none()),
            "publicación antes de liberar el working set de prefill"
        );
        let mutable_path = self.exact_cold.path().to_path_buf();
        fs::remove_file(&mutable_path).with_context(|| {
            format!(
                "desvincular backing mutable promovido {}",
                mutable_path.display()
            )
        })?;
        let new_suffix = ColdStore::create(&mutable_path, 0)
            .context("crear ColdStore independiente para el sufijo")?;
        self.exact_cold = new_suffix;
        self.prefix_cold = Some(prefix_cold);
        for layer in &mut self.layers {
            for page in &mut layer.pages {
                match &mut page.storage {
                    PageStorage::Cold { immutable, .. } => *immutable = true,
                    PageStorage::Pending { .. } => {
                        bail!("página pendiente después de publicar el prefijo")
                    }
                }
            }
        }
        self.prefix_copied_bytes = copied_bytes;
        self.conversion = Self::conversion_worker(0)?;
        self.suffix_against_committed = true;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn restore_prefix(
        cache_root: &Path,
        cache_key: &str,
        model_identity: &str,
        storage_frame_tokens: usize,
    ) -> Result<Option<Self>> {
        Self::restore_prefix_geometry(
            cache_root,
            cache_key,
            model_identity,
            storage_frame_tokens,
            24,
            2,
            16,
            128,
        )
    }

    // Keeping the cache identity and complete tensor geometry explicit makes
    // a mismatched persistent prefix fail closed at this authority boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_prefix_geometry(
        cache_root: &Path,
        cache_key: &str,
        model_identity: &str,
        storage_frame_tokens: usize,
        n_layers: usize,
        kv_heads: i64,
        query_heads: i64,
        head_dim: i64,
    ) -> Result<Option<Self>> {
        let source = cache_root.join(cache_key);
        if !source.is_dir() {
            return Ok(None);
        }
        let manifest: PersistentPrefixManifest = serde_json::from_slice(
            &fs::read(source.join("manifest.json")).context("leer manifiesto de prefijo")?,
        )
        .context("validar manifiesto de prefijo")?;
        ensure!(
            manifest.schema == PREFIX_CACHE_SCHEMA,
            "schema de prefijo incompatible"
        );
        ensure!(
            manifest.model_identity == model_identity,
            "identidad de modelo distinta"
        );
        ensure!(
            manifest.prefix_digest == cache_key,
            "digest de prefijo inconsistente"
        );
        ensure!(
            manifest.storage_frame_tokens == storage_frame_tokens,
            "geometría de páginas distinta"
        );
        ensure!(
            manifest.layers.len() == n_layers,
            "número de capas persistidas inválido"
        );
        Self::restore_prefix_from_manifest(
            cache_root,
            cache_key,
            storage_frame_tokens,
            manifest,
            n_layers,
            kv_heads,
            query_heads,
            head_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_prefix_from_manifest(
        cache_root: &Path,
        cache_key: &str,
        storage_frame_tokens: usize,
        manifest: PersistentPrefixManifest,
        n_layers: usize,
        kv_heads: i64,
        query_heads: i64,
        head_dim: i64,
    ) -> Result<Option<Self>> {
        let source = cache_root.join(cache_key);
        let mut prefix_cold = ColdStore::open_read_only(source.join("pages-exact.kv"), 0)
            .context("abrir KV persistente inmutable")?;
        prefix_cold
            .audit_mapped()
            .context("auditar KV persistente inmutable")?;
        let layers = manifest
            .layers
            .into_iter()
            .map(|pages| MslimLayer {
                pages: pages
                    .into_iter()
                    .map(|page| MslimPage {
                        token_start: page.token_start,
                        tokens: page.tokens,
                        certificate: page.certificate,
                        storage: PageStorage::Cold {
                            exact_id: page.exact_id,
                            immutable: true,
                        },
                    })
                    .collect(),
                tail: None,
                prefill_hot: None,
                resident: None,
            })
            .collect();
        let mut cache = Self::try_new_with_geometry(
            storage_frame_tokens,
            n_layers,
            kv_heads,
            query_heads,
            head_dim,
        )?;
        cache.layers = layers;
        cache.sequence_tokens = manifest.prefix_tokens;
        cache.prefix_cold = Some(prefix_cold);
        cache.prefix_copied_bytes = 0;
        cache.suffix_against_committed = true;
        cache.conversion = Self::conversion_worker(0)?;
        cache.parallel_axis = manifest.parallel_axis;
        Ok(Some(cache))
    }

    #[cfg(test)]
    pub(super) fn hydrate_prefill_hot(&mut self) -> Result<()> {
        for layer_index in 0..self.layers.len() {
            self.layers[layer_index].prefill_hot =
                Some(self.assemble_committed_layer(layer_index)?);
        }
        Ok(())
    }

    pub(super) fn observe_parallel_axis(&mut self, key: &Tensor) -> Result<()> {
        let shape = key.size();
        let kv_heads = self.kv_heads as usize;
        let head_dim = self.head_dim as usize;
        ensure!(
            shape.len() == 4
                && shape[0] == 1
                && shape[1] == self.kv_heads
                && shape[3] == self.head_dim,
            "K incompatible con el eje paralelo"
        );
        let tokens = shape[2] as usize;
        let float = key.to_kind(Kind::Float).contiguous();
        let count = kv_heads * tokens * head_dim;
        let mut values = vec![0_f32; count];
        float.copy_data(&mut values, count);
        let axis_dim = kv_heads * head_dim;
        for token in 0..tokens {
            let mut signal = vec![0_f32; axis_dim];
            for head in 0..kv_heads {
                let source = (head * tokens + token) * head_dim;
                let target = head * head_dim;
                signal[target..target + head_dim]
                    .copy_from_slice(&values[source..source + head_dim]);
            }
            self.parallel_axis.observe(&signal)?;
        }
        Ok(())
    }

    pub(super) fn autocatalogue_parallel_axis(&mut self, query: &Tensor) -> Result<()> {
        if query.size()[2] != 1 || self.parallel_axis.extents().is_empty() {
            return Ok(());
        }
        let mut token_votes = BTreeMap::<usize, usize>::new();
        for protected in protected_queries(query)? {
            for page in self
                .parallel_axis
                .materialize_query_tokens_adaptive(&protected.planner_vector)?
            {
                for token in ParallelContextAxis::expand_token_positions(&page) {
                    *token_votes.entry(token).or_default() += 1;
                }
            }
        }
        let mut region_votes = BTreeMap::<usize, usize>::new();
        for (token, votes) in &token_votes {
            let state_id = self
                .parallel_axis
                .state_id_for_token(*token)
                .context("token de catálogo sin estado dimensional")?;
            *region_votes.entry(state_id).or_default() += votes;
        }
        let logical_regions = self.parallel_axis.regions().len();
        let vote_total = region_votes.values().sum::<usize>() as f64;
        let region_entropy = if vote_total > 0.0 {
            region_votes
                .values()
                .map(|votes| {
                    let probability = *votes as f64 / vote_total;
                    -probability * probability.ln()
                })
                .sum::<f64>()
        } else {
            0.0
        };
        let region_budget = region_entropy
            .exp()
            .ceil()
            .clamp(1.0, logical_regions as f64) as usize;
        let selected_regions = {
            let mut ranked = region_votes.into_iter().collect::<Vec<_>>();
            ranked.sort_unstable_by(|left, right| {
                right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0))
            });
            ranked
                .into_iter()
                .take(region_budget.max(1))
                .map(|(region, _)| region)
                .collect::<BTreeSet<_>>()
        };
        let mut ranked_tokens = token_votes
            .into_iter()
            .filter(|(token, _)| {
                self.parallel_axis
                    .state_id_for_token(*token)
                    .is_some_and(|state| selected_regions.contains(&state))
            })
            .collect::<Vec<_>>();
        ranked_tokens.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0))
        });
        let mut positions = ranked_tokens
            .into_iter()
            .map(|(token, _)| token)
            .collect::<Vec<_>>();
        positions.sort_unstable();
        let views = self.parallel_axis.materialize_token_positions(&positions)?;
        self.parallel_query_events += 1;
        self.parallel_dimensional_views += views.len();
        self.parallel_view_tokens += views.iter().map(|view| view.token_count).sum::<usize>();
        self.parallel_selected_positions = positions;
        Ok(())
    }

    pub(super) fn bf16_bits(tensor: &Tensor) -> Result<Vec<u16>> {
        ensure!(tensor.kind() == Kind::BFloat16, "página no BF16");
        let contiguous = tensor.contiguous();
        // SAFETY: callers supply BF16 tensors from the CPU-only runtime.
        // contiguous owns aligned storage for numel() initialized u16 bit patterns;
        // to_vec copies them before that tensor can be dropped or mutated.
        let source = unsafe {
            std::slice::from_raw_parts(contiguous.data_ptr().cast::<u16>(), contiguous.numel())
        };
        Ok(source.to_vec())
    }

    pub(super) fn u16_bytes(values: &[u16]) -> &[u8] {
        // SAFETY: u16 has two initialized bytes and no padding. u8 has alignment
        // one; the shared byte view has exactly the input extent and borrowed lifetime.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) }
    }

    /// Encode interval endpoints conservatively in BF16. Nearest conversion
    /// is not sufficient for a certificate: minima must round downward and
    /// maxima upward in the numeric ordering of BF16 values.
    pub(super) fn outward_bf16_bounds(values: &[f32], lower: bool) -> Vec<u16> {
        values
            .iter()
            .copied()
            .map(|value| {
                let mut bits = half::bf16::from_f32(value).to_bits();
                let decoded = half::bf16::from_bits(bits).to_f32();
                if lower && decoded > value {
                    bits = if value.is_sign_negative() {
                        bits.saturating_add(1)
                    } else {
                        bits.saturating_sub(1)
                    };
                } else if !lower && decoded < value {
                    bits = if value.is_sign_negative() {
                        bits.saturating_sub(1)
                    } else {
                        bits.saturating_add(1)
                    };
                }
                bits
            })
            .collect()
    }

    pub(super) fn page_attention_certificate(
        &self,
        key: &Tensor,
        value: &Tensor,
        token_start: usize,
    ) -> Result<PageAttentionCertificate> {
        let key_shape = key.size();
        ensure!(
            key_shape.len() == 4
                && key_shape[0] == 1
                && key_shape[1] == self.kv_heads
                && key_shape[2] > 0
                && key_shape[3] == self.head_dim
                && value.size() == key_shape,
            "geometría de certificado KV inválida"
        );
        let value_maximum = value
            .abs()
            .amax([2_i64, 3_i64], false)
            .to_kind(Kind::Float)
            .contiguous();
        let tokens = key_shape[2] as usize;
        // Authoritative per-head V maxima for GQA Llama 32Q/8KV.
        // The original configured-model 16Q/2KV certificates below remain
        // byte-for-byte compatible with their existing prefix cache.
        let mut value_abs_max_by_kv_head = vec![0.0f32; self.kv_heads as usize];
        value_maximum.copy_data(&mut value_abs_max_by_kv_head, self.kv_heads as usize);
        ensure!(
            value_abs_max_by_kv_head
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "original BF16 V per-KV-head maxima nonfinite"
        );
        let value_component_maximum = value
            .abs()
            .amax([2_i64], false)
            .to_kind(Kind::Float)
            .contiguous();
        let component_count = self.kv_heads as usize * self.head_dim as usize;
        let mut value_abs_max_by_kv_head_dim = vec![0.0f32; component_count];
        value_component_maximum.copy_data(&mut value_abs_max_by_kv_head_dim, component_count);
        ensure!(
            value_abs_max_by_kv_head_dim
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "original BF16 V per-component maxima nonfinite"
        );
        let regions = self
            .parallel_axis
            .partition_regions(token_start, token_start + tokens)?
            .into_iter()
            .map(|region_axis| {
                let local_start = region_axis.token_start - token_start;
                let local_end = region_axis.token_end - token_start;
                let region = key.narrow(2, local_start as i64, (local_end - local_start) as i64);
                // Keep interval endpoints conservative when storing them in
                // BF16: nearest rounding can move a minimum upward or a
                // maximum downward and would invalidate the certificate.
                let minimum = region
                    .amin([2_i64], false)
                    .to_kind(Kind::Float)
                    .contiguous();
                let maximum = region
                    .amax([2_i64], false)
                    .to_kind(Kind::Float)
                    .contiguous();
                let key_elements = self.kv_heads as usize * self.head_dim as usize;
                let mut minimum_values = vec![0.0_f32; key_elements];
                let mut maximum_values = vec![0.0_f32; key_elements];
                minimum.copy_data(&mut minimum_values, key_elements);
                maximum.copy_data(&mut maximum_values, key_elements);
                let value_region =
                    value.narrow(2, local_start as i64, (local_end - local_start) as i64);
                let value_max = value_region
                    .abs()
                    .amax([2_i64, 3_i64], false)
                    .to_kind(Kind::Float)
                    .contiguous();
                let mut value_abs_max_by_kv_head = vec![0.0_f32; self.kv_heads as usize];
                value_max.copy_data(&mut value_abs_max_by_kv_head, self.kv_heads as usize);
                ensure!(
                    value_abs_max_by_kv_head
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0),
                    "certificado V regional no finito"
                );
                let mut value_abs_max = [0.0_f32; 2];
                value_abs_max[..2.min(value_abs_max_by_kv_head.len())].copy_from_slice(
                    &value_abs_max_by_kv_head[..2.min(value_abs_max_by_kv_head.len())],
                );
                Ok(KeyRegionCertificate {
                    block_id: token_start + local_start,
                    state_id: region_axis.state_id,
                    local_start,
                    local_end,
                    key_minimum_bits: Self::outward_bf16_bounds(&minimum_values, true),
                    key_maximum_bits: Self::outward_bf16_bounds(&maximum_values, false),
                    value_abs_max,
                    value_abs_max_by_kv_head,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut value_abs_max = [0.0_f32; 2];
        value_maximum.copy_data(&mut value_abs_max, 2);
        ensure!(
            value_abs_max.iter().all(|value| value.is_finite()),
            "certificado KV no finito"
        );
        Ok(PageAttentionCertificate {
            regions,
            value_abs_max,
            value_abs_max_by_kv_head,
            value_abs_max_by_kv_head_dim,
        })
    }

    pub(super) fn commit_conversion(
        &mut self,
        result: crate::conversion_worker::ConversionResult<PreparedPage>,
    ) -> Result<()> {
        let prepared = result.encoded.map_err(anyhow::Error::msg)?;
        let exact = self
            .exact_cold
            .append_prepared(prepared, false)
            .context("commit BF16 exacto frío")?;
        let page = self.layers[result.shape.layer]
            .pages
            .iter_mut()
            .find(|page| matches!(page.storage, PageStorage::Pending { sequence, .. } if sequence == result.sequence))
            .context("resultado de conversión sin página pendiente")?;
        page.storage = PageStorage::Cold {
            exact_id: exact.page_id,
            immutable: false,
        };
        Ok(())
    }

    pub(super) fn collect_conversions(&mut self) -> Result<()> {
        for result in self.conversion.try_collect_ordered() {
            self.commit_conversion(result)?;
        }
        Ok(())
    }

    pub(super) fn gqa_groups(query: &Tensor, key: &Tensor) -> i64 {
        let query_heads = query.size()[1];
        let kv_heads = key.size()[1].max(1);
        if query_heads % kv_heads == 0 {
            query_heads / kv_heads
        } else {
            1
        }
    }

    pub(super) fn record_dense_sdpa(
        &mut self,
        query_tokens: usize,
        attended_tokens: usize,
        sequence_tokens: usize,
        elapsed_nanos: u64,
    ) {
        let heads = self.query_heads as u128;
        let head_dim = self.head_dim as u128;
        let scheduled_pairs = (query_tokens as u128).saturating_mul(attended_tokens as u128);
        let causal_pairs = (0..query_tokens).fold(0_u128, |sum, query_index| {
            sum.saturating_add(
                sequence_tokens
                    .saturating_add(query_index)
                    .saturating_add(1) as u128,
            )
        });
        self.dense_sdpa_calls = self.dense_sdpa_calls.saturating_add(1);
        self.dense_sdpa_scheduled_qk_elements = self
            .dense_sdpa_scheduled_qk_elements
            .saturating_add(scheduled_pairs.saturating_mul(heads));
        self.dense_sdpa_scheduled_pv_elements =
            self.dense_sdpa_scheduled_pv_elements.saturating_add(
                scheduled_pairs
                    .saturating_mul(heads)
                    .saturating_mul(head_dim),
            );
        self.dense_sdpa_causal_qk_elements = self
            .dense_sdpa_causal_qk_elements
            .saturating_add(causal_pairs.saturating_mul(heads));
        self.dense_sdpa_causal_pv_elements = self
            .dense_sdpa_causal_pv_elements
            .saturating_add(causal_pairs.saturating_mul(heads).saturating_mul(head_dim));
        self.dense_sdpa_nanos = self.dense_sdpa_nanos.saturating_add(elapsed_nanos);
    }

    pub(super) fn seal_and_release_prefill(&mut self) -> Result<()> {
        self.seal_partial_storage_frames()?;
        for layer in &mut self.layers {
            layer.prefill_hot = None;
        }
        self.suffix_against_committed = true;
        Ok(())
    }

    pub fn drain_conversions(&mut self) -> Result<()> {
        let submitted = self.conversion.stats().submitted;
        while self.conversion.stats().delivered < submitted {
            let result = self.conversion.wait_next().map_err(anyhow::Error::msg)?;
            self.commit_conversion(result)?;
        }
        Ok(())
    }

    pub fn conversion_metrics(&self) -> ConversionStats {
        self.conversion.stats()
    }

    pub fn storage_frame_tokens(&self) -> usize {
        self.storage_frame_tokens
    }

    pub(super) fn parallel_axis_metrics(&self) -> ParallelAxisMetrics {
        let extents = self.parallel_axis.extents();
        let minimum = extents
            .iter()
            .map(|extent| extent.tokens())
            .min()
            .unwrap_or(0);
        let maximum = extents
            .iter()
            .map(|extent| extent.tokens())
            .max()
            .unwrap_or(0);
        ParallelAxisMetrics {
            extents: extents.len(),
            tokens: self.parallel_axis.token_len(),
            fingerprint_payload_bytes: self.parallel_axis.fingerprint_payload_bytes(),
            f32_fingerprint_reference_bytes: self.parallel_axis.f32_fingerprint_reference_bytes(),
            micro_metadata_bytes: self.parallel_axis.micro_metadata_bytes(),
            extent_min: minimum,
            extent_max: maximum,
            query_events: self.parallel_query_events,
            dimensional_views: self.parallel_dimensional_views,
            view_tokens: self.parallel_view_tokens,
            lsh_queries: self.parallel_axis.lsh_metrics().0,
            lsh_candidates: self.parallel_axis.lsh_metrics().1,
            lsh_misses: self.parallel_axis.lsh_metrics().2,
            lsh_probe_two: self.parallel_axis.lsh_metrics().3,
            lsh_probe_four: self.parallel_axis.lsh_metrics().4,
        }
    }

    /// Seal the final, possibly short, storage frame of every layer after prefill.
    /// Generation can then start with the complete prompt in cold storage
    /// instead of retaining up to `storage_frame_tokens - 1` BF16 tokens per layer.
    pub fn seal_partial_storage_frames(&mut self) -> Result<()> {
        self.collect_conversions()?;
        self.parallel_axis.flush();
        for layer_index in 0..self.layers.len() {
            let Some((key_page, value_page)) = self.layers[layer_index].tail.take() else {
                continue;
            };
            let tokens = key_page.size()[2] as usize;
            if tokens == 0 {
                continue;
            }
            let token_start = self.sequence_tokens.saturating_sub(tokens);
            let certificate =
                self.page_attention_certificate(&key_page, &value_page, token_start)?;
            let shape = PageShape {
                layer: layer_index,
                token_start,
                heads: self.kv_heads as usize,
                tokens,
                dim: self.head_dim as usize,
                tile: 16,
            };
            let job = Bf16PageJob::owned(
                shape,
                Self::bf16_bits(&key_page)?,
                Self::bf16_bits(&value_page)?,
            )
            .map_err(anyhow::Error::msg)?;
            let sequence = self.conversion.submit(job).map_err(anyhow::Error::msg)?;
            let page = MslimPage {
                token_start,
                tokens,
                certificate,
                storage: PageStorage::Pending {
                    sequence,
                    key: key_page,
                    value: value_page,
                },
            };
            self.layers[layer_index].pages.push(page);
        }
        // The full BF16 working state serves prefill while exact physical
        // frames are committed in parallel. One phase barrier makes cold
        // storage authoritative before autoregressive generation.
        self.drain_conversions()?;
        if !self.exact_cold.metadata().is_empty() {
            self.exact_cold
                .audit_mapped()
                .context("auditoría BF16 fría mapeada")?;
        }
        for layer in &mut self.layers {
            layer.prefill_hot = None;
        }
        Ok(())
    }

    pub fn storage_frame_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.pages.len()).sum()
    }

    pub fn tail_tokens(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|layer| layer.tail.as_ref())
            .map(|(key, _)| key.size()[2] as usize)
            .sum()
    }

    pub fn exact_cold_metrics(&self) -> ColdStoreStats {
        let mut stats = self.exact_cold.stats();
        if let Some(prefix) = &self.prefix_cold {
            let extra = prefix.stats();
            stats.pages += extra.pages;
            stats.committed_bytes += extra.committed_bytes;
            stats.reads += extra.reads;
            stats.cache_hits += extra.cache_hits;
            stats.cache_misses += extra.cache_misses;
            stats.evictions += extra.evictions;
            stats.hot_pages += extra.hot_pages;
            stats.hot_bytes += extra.hot_bytes;
            stats.bytes_read += extra.bytes_read;
            stats.partial_value_reads += extra.partial_value_reads;
            stats.partial_value_payload_bytes += extra.partial_value_payload_bytes;
            stats.partial_value_bytes_avoided += extra.partial_value_bytes_avoided;
            stats.partial_key_reads += extra.partial_key_reads;
            stats.partial_key_payload_bytes += extra.partial_key_payload_bytes;
            stats.partial_key_bytes_avoided += extra.partial_key_bytes_avoided;
            stats.cryptographic_digests_verified += extra.cryptographic_digests_verified;
        }
        stats
    }

    pub fn resident_bytes(&self) -> usize {
        let tails = self
            .layers
            .iter()
            .filter_map(|layer| layer.tail.as_ref())
            .map(|(key, value)| (key.numel() + value.numel()) * 2)
            .sum::<usize>();
        let pending = self
            .layers
            .iter()
            .flat_map(|layer| &layer.pages)
            .filter_map(|page| match &page.storage {
                PageStorage::Pending { key, value, .. } => Some((key.numel() + value.numel()) * 2),
                PageStorage::Cold { .. } => None,
            })
            .sum::<usize>();
        let certificates = self
            .layers
            .iter()
            .flat_map(|layer| &layer.pages)
            .map(|page| {
                page.certificate
                    .regions
                    .iter()
                    .map(|region| {
                        (region.key_minimum_bits.len() + region.key_maximum_bits.len())
                            * std::mem::size_of::<u16>()
                            + region.value_abs_max.len() * std::mem::size_of::<f32>()
                    })
                    .sum::<usize>()
                    + page.certificate.value_abs_max.len() * std::mem::size_of::<f32>()
            })
            .sum::<usize>();
        tails
            + pending
            + certificates
            + self.parallel_axis.fingerprint_payload_bytes()
            + self.exact_cold.stats().hot_bytes
    }

    /// Tokens committed to the immutable BF16 cold store and no longer held as
    /// active K/V tensors. They remain fully recoverable; this is residency
    /// state, not deletion.
    pub fn inactive_zero_consumption_tokens(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| &layer.pages)
            .filter_map(|page| match page.storage {
                PageStorage::Cold { .. } => Some(page.tokens),
                PageStorage::Pending { .. } => None,
            })
            .sum()
    }

    pub fn active_resident_kv_bytes(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|layer| layer.tail.as_ref())
            .map(|(key, value)| (key.numel() + value.numel()) * 2)
            .sum::<usize>()
            + self
                .layers
                .iter()
                .flat_map(|layer| &layer.pages)
                .filter_map(|page| match &page.storage {
                    PageStorage::Pending { key, value, .. } => {
                        Some((key.numel() + value.numel()) * 2)
                    }
                    PageStorage::Cold { .. } => None,
                })
                .sum::<usize>()
    }

    pub(super) fn exact_tensor_from_bytes(&self, bytes: &[u8], tokens: usize) -> Result<Tensor> {
        let expected = self.kv_heads as usize * tokens * self.head_dim as usize * 2;
        ensure!(bytes.len() == expected, "geometría BF16 fría inválida");
        Ok(Tensor::from_data_size(
            bytes,
            &[1, self.kv_heads, tokens as i64, self.head_dim],
            Kind::BFloat16,
        ))
    }

    /// Optional 32Q/8KV Llama exact-score mass filter for BF16 V.
    ///
    /// Previously Llama always assembled *every* V from ColdStore,
    /// including requests that opted into applied_mass_tolerance. This
    /// method reads full exact K (needed for score mass) but retrieves
    /// only the certified selected BF16 V token indices. The original
    /// ColdStore is never rewritten or quantized, and with no proof
    /// (missing old-prefix metadata, diffuse scores, NaN, zero tolerance)
    /// returns None so the caller executes the ORIGINAL full SDPA.
    ///
    /// The error bound is a per-head mathematical softmax-mass bound
    /// computed from actual BF16 Q/K operands in FP32 and conservative
    /// per-KV-head |V| maxima; it does NOT prove token/logit identity.
    // The optional preassembled score/probability tensors identify which
    // physical work already happened; bundling them would obscure ownership
    // and make accidental recomputation harder to audit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn llama_certified_value_attention(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        pages: &[MslimPage],
        tail: Option<&(Tensor, Tensor)>,
        preassembled_key: Option<&Tensor>,
        precomputed_scores: Option<&Tensor>,
        sequence_tokens: usize,
        tolerance: f64,
    ) -> Result<Option<Tensor>> {
        ensure!(
            layer_index < self.layers.len(),
            "Llama ColdStore layer out of range"
        );
        if !(tolerance.is_finite() && tolerance > 0.0 && tolerance <= 1e-2) {
            return Ok(None); // only explicit positive, bounded epsilon
        }
        if self.query_heads != 32
            || self.kv_heads != 8
            || self.head_dim != 128
            || query.size() != [1, 32, 1, 128]
        {
            return Ok(None); // never reinterpret unrelated geometry
        }
        // Logical dense baseline for this Llama attention event. The regional
        // planner may avoid K/V work, but these counters remain the honest
        // dense Transformer comparison and are never reported as savings.
        let dense_pairs = sequence_tokens.saturating_add(1);
        self.attention_dense_qk_elements = self
            .attention_dense_qk_elements
            .saturating_add(u128::from(dense_pairs as u64) * 32);
        self.attention_dense_pv_elements = self
            .attention_dense_pv_elements
            .saturating_add(u128::from(dense_pairs as u64) * 32 * 128);
        let mut max_per_kv = [0.0f64; 8];
        let mut max_per_kv_component = vec![0.0f64; 8 * 128];
        let mut tail_value_max = [0.0f64; 8];
        for page in pages {
            // A legacy v1 prefix lacked complete 8-KV-head bounds:
            // its first-two-head certificate is NEVER sufficient
            // authority for Llama omission. Restore old full attention.
            if page.certificate.value_abs_max_by_kv_head.len() != 8
                || page.certificate.value_abs_max_by_kv_head_dim.len() != 8 * 128
            {
                return Ok(None);
            }
            if page.certificate.regions.iter().any(|region| {
                region.key_minimum_bits.len() != 8 * 128
                    || region.key_maximum_bits.len() != 8 * 128
                    || region.value_abs_max_by_kv_head.len() != 8
            }) {
                // A legacy or partially written regional certificate cannot
                // authorize a Llama 32Q/8KV omission. Preserve dense BF16.
                return Ok(None);
            }
            for (dst, &maxval) in max_per_kv
                .iter_mut()
                .zip(&page.certificate.value_abs_max_by_kv_head)
            {
                if !(maxval.is_finite() && maxval >= 0.0) {
                    return Ok(None);
                }
                *dst = dst.max(f64::from(maxval));
            }
            for (dst, &maxval) in max_per_kv_component
                .iter_mut()
                .zip(&page.certificate.value_abs_max_by_kv_head_dim)
            {
                if !(maxval.is_finite() && maxval >= 0.0) {
                    return Ok(None);
                }
                *dst = dst.max(f64::from(maxval));
            }
        }
        if let Some((key, value)) = tail {
            if key.size() != value.size() || key.size()[1] != 8 {
                return Ok(None);
            }
            let maxima = value
                .abs()
                .amax([2_i64, 3_i64], false)
                .to_kind(Kind::Float)
                .contiguous();
            let mut bf16_max = vec![0.0f32; 8];
            maxima.copy_data(&mut bf16_max, 8);
            for (dst, &maxval) in max_per_kv.iter_mut().zip(&bf16_max) {
                if !(maxval.is_finite() && maxval >= 0.0) {
                    return Ok(None);
                }
                *dst = dst.max(f64::from(maxval));
            }
            for (dst, &maxval) in tail_value_max.iter_mut().zip(&bf16_max) {
                *dst = f64::from(maxval);
            }
            let component_maxima = value
                .abs()
                .amax([2_i64], false)
                .to_kind(Kind::Float)
                .contiguous();
            let mut bf16_component_max = vec![0.0f32; 8 * 128];
            component_maxima.copy_data(&mut bf16_component_max, 8 * 128);
            for (dst, &maxval) in max_per_kv_component.iter_mut().zip(&bf16_component_max) {
                if !(maxval.is_finite() && maxval >= 0.0) {
                    return Ok(None);
                }
                *dst = dst.max(f64::from(maxval));
            }
        }
        // Parity-first Llama path: score the complete K history so omitted V
        // terms can be removed without deleting their probability mass from
        // the softmax denominator. The regional QK-pruning experiment remains
        // available to focused tests, but cannot authorize this path because
        // sparse SDPA renormalizes retained tokens and perturbs every term.
        let cursor = sequence_tokens + 1;
        let full_key = if let Some(key) = preassembled_key {
            ensure!(
                key.size()[2] as usize >= cursor,
                "Llama preassembled K does not cover causal query"
            );
            key.narrow(2, 0, cursor as i64)
        } else {
            let mut keys = Vec::with_capacity(pages.len() + usize::from(tail.is_some()));
            let mut covered = 0usize;
            for page in pages {
                ensure!(
                    page.token_start == covered && page.tokens > 0,
                    "Llama canonical ColdStore K page ordering invalid"
                );
                let key = match &page.storage {
                    PageStorage::Pending { key, .. } => key.shallow_clone(),
                    PageStorage::Cold {
                        exact_id,
                        immutable,
                    } => {
                        let bf16 = self.read_cold_key(*exact_id, *immutable)?;
                        self.exact_tensor_from_bytes(&bf16, page.tokens)?
                    }
                };
                keys.push(key);
                covered += page.tokens;
            }
            if let Some((k, _)) = tail {
                keys.push(k.shallow_clone());
                covered += k.size()[2] as usize;
            }
            if keys.is_empty() {
                return Ok(None);
            }
            ensure!(covered == cursor, "Llama exact K history incomplete");
            let refs = keys.iter().collect::<Vec<_>>();
            Tensor::cat(&refs, 2)
        };
        ensure!(
            full_key.size()[2] as usize == cursor,
            "Llama exact-score audit does not cover complete BF16 K history"
        );
        let gqa = self.query_heads / self.kv_heads;
        let queries = query.to_kind(Kind::Float);
        let scores = if let Some(scores) = precomputed_scores {
            ensure!(
                scores.size() == [1, 32, 1, cursor as i64],
                "scores Llama precomputados incompatibles"
            );
            scores.shallow_clone()
        } else {
            let repeated = full_key
                .repeat_interleave_self_int(gqa, 1, None)
                .to_kind(Kind::Float);
            (queries.matmul(&repeated.transpose(-2, -1)) / (self.head_dim as f64).sqrt())
                .to_kind(Kind::Float)
                .contiguous()
        };
        let mut scores_flat = vec![0.0f32; 32 * cursor];
        scores.copy_data(&mut scores_flat, 32 * cursor);
        if scores_flat.iter().any(|value| !value.is_finite()) {
            return Ok(None); // nonfinite attention never certifies omission
        }
        let scores_by_head = scores_flat
            .chunks_exact(cursor)
            .map(|head| {
                head.iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Validate the newly dimensioned 32Q/8KV regional bounds against the
        // exact scores before they can ever authorize future QK pruning.
        // This is deliberately a verification-only step for now: a failed
        // regional proof falls back to dense attention.
        let mut query_values = vec![0.0_f32; 32 * 128];
        queries.copy_data(&mut query_values, 32 * 128);
        let regional_uppers = state_score_uppers_generic::<32, 8>(
            pages,
            &query_values,
            1.0 / (self.head_dim as f64).sqrt(),
        );
        for page in pages {
            for region in &page.certificate.regions {
                let start = page.token_start + region.local_start;
                let end = page.token_start + region.local_end;
                let upper = match regional_uppers.get(&region.block_id) {
                    Some(upper) => upper,
                    None => return Ok(None),
                };
                for head in 0..32 {
                    let exact_log_mass =
                        logsumexp_finite(scores_by_head[head][start..end].iter().copied());
                    if !exact_log_mass.is_finite() || exact_log_mass > upper[head] + 1e-8 {
                        return Ok(None);
                    }
                }
            }
        }
        // Measurement only: how many K tokens a best-first regional proof
        // (the configured-model branch-and-bound, renormalized) would let this event
        // skip. The parity-first Llama path still scores every K token, so
        // these are reported as `hierarchy_k_tokens_skippable`, never as
        // avoided work. A failed simulation counts nothing and never aborts
        // inference.
        if let Ok(skippable) = llama_regional_k_tokens_skippable(
            pages,
            &regional_uppers,
            &scores_by_head,
            tail_value_max,
            tolerance * 0.5,
        ) {
            self.hierarchy_k_events += 1;
            self.hierarchy_k_tokens_skippable =
                self.hierarchy_k_tokens_skippable.saturating_add(skippable);
        }
        // Per-token |V| upper bounds per KV head: the maximum of the token's
        // region when the page's regions cover it completely, otherwise the
        // page maximum, and the tail maximum for tail tokens. Each is a
        // bound for that token, so the omission bound stays sound while a
        // few large-|V| tokens no longer inflate the bound of every token.
        let mut token_max_by_kv = (0..8)
            .map(|_| Vec::with_capacity(cursor))
            .collect::<Vec<Vec<f64>>>();
        for page in pages {
            let covered = page
                .certificate
                .regions
                .iter()
                .map(|region| region.local_end.saturating_sub(region.local_start))
                .sum::<usize>()
                == page.tokens
                && page
                    .certificate
                    .regions
                    .iter()
                    .all(|region| region.value_abs_max_by_kv_head.len() == 8);
            if covered {
                let mut regions = page.certificate.regions.iter().collect::<Vec<_>>();
                regions.sort_by_key(|region| region.local_start);
                for region in regions {
                    for (kv_head, maxima) in token_max_by_kv.iter_mut().enumerate() {
                        let value = f64::from(region.value_abs_max_by_kv_head[kv_head]);
                        maxima.extend(std::iter::repeat_n(
                            value,
                            region.local_end - region.local_start,
                        ));
                    }
                }
            } else {
                for (kv_head, maxima) in token_max_by_kv.iter_mut().enumerate() {
                    let value = f64::from(page.certificate.value_abs_max_by_kv_head[kv_head]);
                    maxima.extend(std::iter::repeat_n(value, page.tokens));
                }
            }
        }
        for (kv_head, maxima) in token_max_by_kv.iter_mut().enumerate() {
            let remaining = cursor.saturating_sub(maxima.len());
            maxima.extend(std::iter::repeat_n(tail_value_max[kv_head], remaining));
        }
        if token_max_by_kv
            .iter()
            .zip(&max_per_kv)
            .any(|(maxima, global)| maxima.len() != cursor || maxima.iter().any(|v| v > global))
        {
            return Ok(None);
        }
        let token_max_per_query = (0..32)
            .map(|head| token_max_by_kv[head / (32 / 8)].clone())
            .collect::<Vec<_>>();
        // Reserve half the requested budget for differences between explicit
        // FP32 score/softmax planning and LibTorch BF16 SDPA.
        let allowed = tolerance * 0.5;
        let (chosen, bounds) =
            select_v_by_token_contribution(&scores_by_head, &token_max_per_query, allowed)?;
        ensure!(
            certified(&bounds, allowed),
            "Llama exact-mass certificate failed under conservative threshold"
        );
        if chosen.len() == cursor || chosen.is_empty() {
            // No BF16 V was physically avoided: record NO omission and
            // preserve the original dense SDPA result bit-for-bit.
            return Ok(None);
        }
        let selection = chosen.iter().copied().collect::<BTreeSet<_>>();
        let mut values = Vec::with_capacity(pages.len() + usize::from(tail.is_some()));
        let mut offset = 0usize;
        for page in pages {
            let positions = selection
                .range(offset..offset + page.tokens)
                .map(|position| position - offset)
                .collect::<Vec<_>>();
            if !positions.is_empty() {
                match &page.storage {
                    PageStorage::Pending { value, .. } => {
                        values.push(select_token_indices(value, &positions));
                    }
                    PageStorage::Cold {
                        exact_id,
                        immutable,
                    } => {
                        // Partial-I/O, token hashes and BF16 bytes are
                        // verified by the original ColdStore implementation.
                        let bf16 =
                            self.read_cold_value_tokens(*exact_id, *immutable, &positions)?;
                        values.push(self.exact_tensor_from_bytes(&bf16, positions.len())?);
                    }
                }
            }
            offset += page.tokens;
        }
        if let Some((_, tail_value)) = tail {
            let n = tail_value.size()[2] as usize;
            let positions = selection
                .range(offset..offset + n)
                .map(|position| position - offset)
                .collect::<Vec<_>>();
            if !positions.is_empty() {
                values.push(select_token_indices(tail_value, &positions));
            }
        }
        ensure!(!values.is_empty(), "Llama certified V selection empty");
        let selected_indices =
            Tensor::from_slice(&chosen.iter().map(|&x| x as i64).collect::<Vec<_>>());
        let value_refs = values.iter().collect::<Vec<_>>();
        let selected_values = Tensor::cat(&value_refs, 2);
        // Preserve the canonical LibTorch SDPA geometry and accumulation
        // order.  The omitted V positions are represented by exact BF16
        // zeros, so their payload is never read from ColdStore, while full K
        // still supplies the original softmax denominator.  Using a compact
        // FP32 softmax+matmul here would mix two effects in the parity oracle:
        // omitted V and a different attention kernel.
        let zero_values = Tensor::zeros(
            [1, self.kv_heads, cursor as i64, self.head_dim],
            (query.kind(), query.device()),
        );
        let sparse_values = zero_values.index_copy(2, &selected_indices, &selected_values);
        let output = Self::scaled_attention(
            query,
            &[full_key.shallow_clone()],
            &[sparse_values],
            sequence_tokens,
        );
        // BF16 rounding-cell diagnostic. Its centre must be the UNROUNDED
        // candidate: the BF16 SDPA output above is already rounded, so a cell
        // centred on it says nothing about which side of a midpoint the real
        // value lies (and practically never passes). Recompute the candidate
        // in FP32 from the same FP32 score plan that chose the omission:
        // full-denominator softmax probabilities times the selected BF16 V
        // (exact in FP32). This is diagnostic only: it never changes the
        // returned output or the omission decision, and it is relative to the
        // FP32 score plan, not a proof about the fused BF16 SDPA kernel.
        let unit_roundoff = 2_f64.powi(-24);
        let operations = (chosen.len().saturating_mul(2).saturating_add(64)) as f64;
        let gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff);
        let selected_set = chosen.iter().copied().collect::<BTreeSet<_>>();
        let mut omitted_fraction = Vec::with_capacity(32);
        let mut kept_probabilities = Vec::with_capacity(32 * chosen.len());
        for head_scores in &scores_by_head {
            let maximum = head_scores
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let weights = head_scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            let mut omitted = 0.0;
            for (token, weight) in weights.iter().enumerate() {
                if !selected_set.contains(&token) {
                    omitted += weight;
                }
            }
            omitted_fraction.push(omitted / total);
            kept_probabilities.extend(chosen.iter().map(|&token| (weights[token] / total) as f32));
        }
        let probabilities =
            Tensor::from_slice(&kept_probabilities).view([1, 32, 1, chosen.len() as i64]);
        let center = probabilities
            .matmul(
                &selected_values
                    .to_kind(Kind::Float)
                    .repeat_interleave_self_int(gqa, 1, None),
            )
            .contiguous();
        let mut center_components = vec![0.0f32; 32 * 128];
        center.copy_data(&mut center_components, 32 * 128);
        let output_float = output.to_kind(Kind::Float).contiguous();
        let mut output_components = vec![0.0f32; 32 * 128];
        output_float.copy_data(&mut output_components, 32 * 128);
        // Per component: omitted contribution plus conservative FP32
        // accumulation and probability-rounding error (gamma covers both).
        let rounding_bounds = (0..32)
            .flat_map(|head| {
                let kv_head = head / 4;
                let fraction = omitted_fraction[head];
                let maxima = &max_per_kv_component[kv_head * 128..(kv_head + 1) * 128];
                maxima
                    .iter()
                    .map(move |maximum| (fraction + gamma + unit_roundoff) * *maximum)
            })
            .collect::<Vec<_>>();
        // A component counts only when its whole interval stays inside one
        // BF16 cell AND the emitted BF16 candidate is that cell's value.
        let certified_components = center_components
            .iter()
            .zip(&rounding_bounds)
            .zip(&output_components)
            .filter(|&((&center, &bound), &emitted)| {
                bf16_rounding_cell_contains(center, bound)
                    && half::bf16::from_f32(center).to_bits()
                        == half::bf16::from_f32(emitted).to_bits()
            })
            .count();
        self.rounding_cell_checked_events += 1;
        self.rounding_cell_components_checked += center_components.len();
        self.rounding_cell_components_certified += certified_components;
        if certified_components == center_components.len() {
            self.rounding_cell_certified_events += 1;
        }
        let omitted = cursor.saturating_sub(chosen.len());
        self.exact_k_tokens_scanned += cursor;
        self.exact_v_tokens_materialized += chosen.len();
        self.attention_selected_qk_elements = self
            .attention_selected_qk_elements
            .saturating_add(u128::from(cursor as u64) * 32);
        // The sparse payload avoids V materialization, but the zero-filled
        // canonical tensor still submits a dense PV geometry to SDPA.  Do not
        // misreport I/O savings as Transformer arithmetic savings.
        self.attention_selected_pv_elements = self
            .attention_selected_pv_elements
            .saturating_add(u128::from(cursor as u64) * 32 * 128);
        self.attention_v_tokens_avoided = self.attention_v_tokens_avoided.saturating_add(omitted);
        self.exact_mass_max_bound = self
            .exact_mass_max_bound
            .max(bounds.into_iter().fold(0.0f64, f64::max));
        Ok(Some(output))
    }

    /// Layer-major causal selective prefill for Llama. Q/K/V projections and
    /// the MLP are produced for the complete caller tile by `forward_with_cache`;
    /// only attention is evaluated query-by-query so each token keeps its own
    /// causal prefix and certificate. If any query cannot certify, return None
    /// and let the caller execute the existing dense causal tile.
    pub(super) fn llama_querywise_certified_prefill(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        new_key: &Tensor,
        new_value: &Tensor,
        tolerance: f64,
    ) -> Result<Option<Tensor>> {
        let query_tokens = query.size()[2] as usize;
        if query_tokens < 2
            || query.size()[1] != 32
            || new_key.size()[1] != 8
            || new_key.size() != new_value.size()
            || new_key.size()[2] as usize != query_tokens
        {
            return Ok(None);
        }
        let pages = self.layers[layer_index].pages.clone();
        let existing_tail = self.layers[layer_index]
            .tail
            .as_ref()
            .map(|(key, value)| (key.shallow_clone(), value.shallow_clone()));
        let logical_snapshot = (
            self.exact_k_tokens_scanned,
            self.exact_v_tokens_materialized,
            self.attention_dense_qk_elements,
            self.attention_dense_pv_elements,
            self.attention_selected_qk_elements,
            self.attention_selected_pv_elements,
            self.attention_v_tokens_avoided,
            self.exact_mass_max_bound,
        );
        let preassembled_key =
            self.llama_dense_tile_key(&pages, existing_tail.as_ref(), new_key)?;
        let repeated_key = preassembled_key
            .repeat_interleave_self_int(4, 1, None)
            .to_kind(Kind::Float);
        let precomputed_scores = (query
            .to_kind(Kind::Float)
            .matmul(&repeated_key.transpose(-2, -1))
            / (self.head_dim as f64).sqrt())
        .to_kind(Kind::Float)
        .contiguous();
        // Exact diagnostic authority must have the same batched geometry and
        // causal mask as the canonical prefill. A sequence of one-query SDPA
        // calls is mathematically equivalent, but not necessarily bit-exact
        // because LibTorch may select a different kernel/accumulation order.
        let reference_tile = if self.verify_exact_attention_outputs {
            Some(self.llama_dense_tile_attention(
                query,
                &pages,
                existing_tail.as_ref(),
                &preassembled_key,
                new_value,
            )?)
        } else {
            None
        };
        let mut outputs = Vec::with_capacity(query_tokens);
        let mut selective_queries = 0usize;
        for query_index in 0..query_tokens {
            let prefix_tokens = query_index + 1;
            let prefix_key = new_key.narrow(2, 0, prefix_tokens as i64);
            let prefix_value = new_value.narrow(2, 0, prefix_tokens as i64);
            let tail = match &existing_tail {
                Some((key, value)) => (
                    Tensor::cat(&[key, &prefix_key], 2),
                    Tensor::cat(&[value, &prefix_value], 2),
                ),
                None => (prefix_key, prefix_value),
            };
            let query_token = query.narrow(2, query_index as i64, 1);
            let logical_history = self.sequence_tokens + query_index;
            let causal_tokens = logical_history + 1;
            let query_scores = precomputed_scores
                .narrow(2, query_index as i64, 1)
                .narrow(3, 0, causal_tokens as i64)
                .contiguous();
            let avoided_before = self.attention_v_tokens_avoided;
            let candidate = self.llama_certified_value_attention(
                layer_index,
                &query_token,
                &pages,
                Some(&tail),
                Some(&preassembled_key),
                Some(&query_scores),
                logical_history,
                tolerance,
            )?;
            let output = if let Some(candidate) = candidate {
                selective_queries += 1;
                if let Some(canonical) = reference_tile.as_ref() {
                    let omitted = self
                        .attention_v_tokens_avoided
                        .saturating_sub(avoided_before);
                    self.exact_attention_verified_events += 1;
                    self.exact_attention_candidate_v_tokens = self
                        .exact_attention_candidate_v_tokens
                        .saturating_add(omitted);
                    let reference_query = canonical.narrow(2, query_index as i64, 1);
                    if candidate.equal(&reference_query) {
                        self.exact_attention_equal_events += 1;
                    } else {
                        self.exact_attention_rejected_v_tokens = self
                            .exact_attention_rejected_v_tokens
                            .saturating_add(omitted);
                    }
                    reference_query
                } else {
                    candidate
                }
            } else if let Some(canonical) = reference_tile.as_ref() {
                canonical.narrow(2, query_index as i64, 1)
            } else {
                self.llama_dense_single_query_attention(
                    &query_token,
                    &pages,
                    Some(&tail),
                    logical_history,
                )?
            };
            outputs.push(output);
        }
        if let Some(canonical) = reference_tile {
            // The oracle deliberately read complete V and executed canonical
            // dense attention once. Candidate counters describe discovered
            // opportunities, but no physical/Transformer saving is claimed.
            self.exact_k_tokens_scanned = logical_snapshot.0;
            self.exact_v_tokens_materialized = logical_snapshot.1;
            self.attention_dense_qk_elements = logical_snapshot.2;
            self.attention_dense_pv_elements = logical_snapshot.3;
            self.attention_selected_qk_elements = logical_snapshot.4;
            self.attention_selected_pv_elements = logical_snapshot.5;
            self.attention_v_tokens_avoided = logical_snapshot.6;
            self.exact_mass_max_bound = logical_snapshot.7;
            return Ok(Some(canonical));
        }
        if selective_queries == 0 {
            // No omission survived this tile. Keep physical I/O metrics (the
            // attempted work happened), but let the caller use the faster
            // canonical batched SDPA and claim no logical savings.
            self.exact_k_tokens_scanned = logical_snapshot.0;
            self.exact_v_tokens_materialized = logical_snapshot.1;
            self.attention_dense_qk_elements = logical_snapshot.2;
            self.attention_dense_pv_elements = logical_snapshot.3;
            self.attention_selected_qk_elements = logical_snapshot.4;
            self.attention_selected_pv_elements = logical_snapshot.5;
            self.attention_v_tokens_avoided = logical_snapshot.6;
            self.exact_mass_max_bound = logical_snapshot.7;
            return Ok(None);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Ok(Some(Tensor::cat(&refs, 2)))
    }

    pub(super) fn llama_dense_tile_attention(
        &mut self,
        query: &Tensor,
        pages: &[MslimPage],
        existing_tail: Option<&(Tensor, Tensor)>,
        preassembled_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<Tensor> {
        let mut values = Vec::with_capacity(pages.len() + 2);
        let mut covered = 0usize;
        for page in pages {
            ensure!(
                page.token_start == covered && page.tokens > 0,
                "páginas Llama densas fuera de orden"
            );
            let value = match &page.storage {
                PageStorage::Pending { value, .. } => value.shallow_clone(),
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => {
                    let value = self.read_cold_value(*exact_id, *immutable)?;
                    self.exact_tensor_from_bytes(&value, page.tokens)?
                }
            };
            values.push(value);
            covered += page.tokens;
        }
        if let Some((key, value)) = existing_tail {
            ensure!(key.size() == value.size(), "cola Llama K/V incompatible");
            covered += key.size()[2] as usize;
            values.push(value.shallow_clone());
        }
        ensure!(
            covered == self.sequence_tokens,
            "histórico Llama denso incompleto antes del bloque"
        );
        values.push(new_value.shallow_clone());
        ensure!(
            preassembled_key.size()[2] as usize == self.sequence_tokens + query.size()[2] as usize,
            "bloque Llama K preensamblado incompatible"
        );
        let query_tokens = query.size()[2] as usize;
        let attended_tokens = preassembled_key.size()[2] as usize;
        let started = std::time::Instant::now();
        let output = Self::scaled_attention(
            query,
            &[preassembled_key.shallow_clone()],
            &values,
            self.sequence_tokens,
        );
        self.record_dense_sdpa(
            query_tokens,
            attended_tokens,
            self.sequence_tokens,
            elapsed_nanos(started),
        );
        Ok(output)
    }

    pub(super) fn llama_dense_tile_key(
        &mut self,
        pages: &[MslimPage],
        existing_tail: Option<&(Tensor, Tensor)>,
        new_key: &Tensor,
    ) -> Result<Tensor> {
        let mut keys = Vec::with_capacity(pages.len() + 2);
        let mut covered = 0usize;
        for page in pages {
            ensure!(
                page.token_start == covered && page.tokens > 0,
                "páginas Llama K fuera de orden"
            );
            let key = match &page.storage {
                PageStorage::Pending { key, .. } => key.shallow_clone(),
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => {
                    let bytes = self.read_cold_key(*exact_id, *immutable)?;
                    self.exact_tensor_from_bytes(&bytes, page.tokens)?
                }
            };
            keys.push(key);
            covered += page.tokens;
        }
        if let Some((key, value)) = existing_tail {
            ensure!(key.size() == value.size(), "cola Llama K/V incompatible");
            covered += key.size()[2] as usize;
            keys.push(key.shallow_clone());
        }
        ensure!(
            covered == self.sequence_tokens,
            "histórico Llama K incompleto antes del bloque"
        );
        keys.push(new_key.shallow_clone());
        let refs = keys.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&refs, 2))
    }

    pub(super) fn llama_dense_single_query_attention(
        &mut self,
        query: &Tensor,
        pages: &[MslimPage],
        tail: Option<&(Tensor, Tensor)>,
        sequence_tokens: usize,
    ) -> Result<Tensor> {
        let mut keys = Vec::with_capacity(pages.len() + usize::from(tail.is_some()));
        let mut values = Vec::with_capacity(keys.capacity());
        let mut covered = 0usize;
        for page in pages {
            ensure!(
                page.token_start == covered && page.tokens > 0,
                "páginas Llama densas fuera de orden"
            );
            let (key, value) = match &page.storage {
                PageStorage::Pending { key, value, .. } => {
                    (key.shallow_clone(), value.shallow_clone())
                }
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => {
                    let key = self.read_cold_key(*exact_id, *immutable)?;
                    let value = self.read_cold_value(*exact_id, *immutable)?;
                    (
                        self.exact_tensor_from_bytes(&key, page.tokens)?,
                        self.exact_tensor_from_bytes(&value, page.tokens)?,
                    )
                }
            };
            keys.push(key);
            values.push(value);
            covered += page.tokens;
        }
        if let Some((key, value)) = tail {
            ensure!(key.size() == value.size(), "cola Llama K/V incompatible");
            covered += key.size()[2] as usize;
            keys.push(key.shallow_clone());
            values.push(value.shallow_clone());
        }
        ensure!(
            covered == sequence_tokens + 1 && !keys.is_empty(),
            "histórico Llama denso incompleto"
        );
        let started = std::time::Instant::now();
        let output = Self::scaled_attention(query, &keys, &values, sequence_tokens);
        self.record_dense_sdpa(
            query.size()[2] as usize,
            covered,
            sequence_tokens,
            elapsed_nanos(started),
        );
        Ok(output)
    }

    pub(super) fn exact_mass_attention(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        pages: &[MslimPage],
        tail: Option<&(Tensor, Tensor)>,
        tolerance: f64,
    ) -> Result<Tensor> {
        let metadata_started = std::time::Instant::now();
        ensure!(query.size() == [1, 16, 1, 128], "Q no autoregresiva");
        let query_float = query.to_kind(Kind::Float).contiguous();
        let mut query_values = vec![0.0_f32; 16 * 128];
        query_float.copy_data(&mut query_values, 16 * 128);
        let scale = 1.0 / (128.0_f64).sqrt();
        let mut value_max = [0.0_f64; 2];
        for page in pages {
            for (head, maximum) in page.certificate.value_abs_max.iter().enumerate() {
                value_max[head] = value_max[head].max(f64::from(*maximum));
            }
        }
        let mut tail_start = None;
        let mut tail_value_max = [0.0_f64; 2];
        if let Some((tail_key, tail_value)) = tail {
            let tail_tokens = tail_key.size()[2] as usize;
            let total_tokens = self.sequence_tokens + query.size()[2] as usize;
            tail_start = Some(total_tokens - tail_tokens);
            let maximum = tail_value
                .abs()
                .amax([2_i64, 3_i64], false)
                .to_kind(Kind::Float)
                .contiguous();
            let mut tail_max = [0.0_f32; 2];
            maximum.copy_data(&mut tail_max, 2);
            for head in 0..2 {
                value_max[head] = value_max[head].max(f64::from(tail_max[head]));
                tail_value_max[head] = f64::from(tail_max[head]);
            }
        }
        let all_states = pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .map(|region| region.block_id)
            .collect::<BTreeSet<_>>();
        let region_count = pages
            .iter()
            .map(|page| page.certificate.regions.len())
            .sum::<usize>();
        ensure!(
            all_states.len() == region_count,
            "identificador jerárquico duplicado"
        );
        let state_uppers = state_score_uppers(pages, &query_values, scale);
        let state_value_max = pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .fold(BTreeMap::<usize, [f64; 2]>::new(), |mut maxima, region| {
                let entry = maxima.entry(region.block_id).or_insert([0.0; 2]);
                for (slot, value) in entry.iter_mut().enumerate() {
                    *value = value.max(f64::from(region.value_abs_max[slot]));
                }
                maxima
            });
        let mut loaded = BTreeSet::new();
        for position in &self.parallel_selected_positions {
            for page in pages {
                if *position < page.token_start || *position >= page.token_start + page.tokens {
                    continue;
                }
                let local = *position - page.token_start;
                for region in &page.certificate.regions {
                    if self.parallel_axis.state_id_for_token(*position) == Some(region.state_id)
                        && (region.local_start..region.local_end).contains(&local)
                    {
                        loaded.insert(region.block_id);
                    }
                }
            }
        }
        if let Some(latest_state) = all_states.last() {
            loaded.insert(*latest_state);
        }
        let loaded_state_ids = pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .filter(|region| loaded.contains(&region.block_id))
            .map(|region| region.state_id)
            .collect::<BTreeSet<_>>();
        let related_state_ids = loaded_state_ids
            .iter()
            .flat_map(|state_id| self.parallel_axis.communicating_dimensions(*state_id))
            .copied()
            .collect::<BTreeSet<_>>();
        let metadata_priority = pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .filter(|region| related_state_ids.contains(&region.state_id))
            .map(|region| region.block_id)
            .collect::<BTreeSet<_>>();
        let initial_loaded_states = loaded.len();
        let region_tolerance = tolerance * 0.5;
        let mut widening_rounds = 0_usize;
        // Materialization is incremental for the lifetime of this attention
        // call. Branch-and-bound may add one region per round, but an already
        // opened region must never be read from cold storage again. Keeping
        // this cache local preserves the bounded steady-state KV design: all
        // tensors are released as soon as the current token finishes.
        let mut materialized_region_keys = BTreeMap::<usize, Tensor>::new();
        let mut materialized_region_scores = BTreeMap::<usize, Vec<Vec<f64>>>::new();
        let tail_scores = tail.map(|(tail_key, _)| {
            let tail_tokens = tail_key.size()[2] as usize;
            let tail_float = tail_key.to_kind(Kind::Float).contiguous();
            let mut tail_values = vec![0.0_f32; 2 * tail_tokens * 128];
            tail_float.copy_data(&mut tail_values, 2 * tail_tokens * 128);
            exact_scores(&query_values, &tail_values, tail_tokens, scale)
        });
        let mut regional_certificate = IncrementalRegionCertificate::<16, 2>::new(
            &state_uppers,
            &state_value_max,
            tail_scores.as_ref(),
            tail_value_max,
        )?;
        let mut candidate_order = all_states.iter().copied().collect::<Vec<_>>();
        candidate_order.sort_unstable_by(|left, right| {
            let left_upper = state_uppers[left]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let right_upper = state_uppers[right]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            right_upper.total_cmp(&left_upper).then_with(|| {
                metadata_priority
                    .contains(right)
                    .cmp(&metadata_priority.contains(left))
            })
        });
        let mut next_candidate = 0_usize;
        self.decode_metadata_nanos = self
            .decode_metadata_nanos
            .saturating_add(elapsed_nanos(metadata_started));
        let region_bounds = loop {
            for page in pages {
                for region in page
                    .certificate
                    .regions
                    .iter()
                    .filter(|region| loaded.contains(&region.block_id))
                {
                    if materialized_region_keys.contains_key(&region.block_id) {
                        continue;
                    }
                    let key_materialization_started = std::time::Instant::now();
                    let local = (region.local_start..region.local_end).collect::<Vec<_>>();
                    ensure!(!local.is_empty(), "región jerárquica vacía");
                    let key = match &page.storage {
                        PageStorage::Pending { key, .. } => select_token_indices(key, &local),
                        PageStorage::Cold {
                            exact_id,
                            immutable,
                        } => {
                            let bytes = self
                                .read_cold_key_tokens(*exact_id, *immutable, &local)
                                .with_context(|| {
                                    format!("lectura K BF16 parcial exacta {exact_id}")
                                })?;
                            self.exact_tensor_from_bytes(&bytes, local.len())?
                        }
                    };
                    let region_tokens = local.len();
                    let key_float = key.to_kind(Kind::Float).contiguous();
                    let mut key_values = vec![0.0_f32; 2 * region_tokens * 128];
                    key_float.copy_data(&mut key_values, 2 * region_tokens * 128);
                    self.decode_key_materialization_nanos = self
                        .decode_key_materialization_nanos
                        .saturating_add(elapsed_nanos(key_materialization_started));
                    let exact_scores_started = std::time::Instant::now();
                    let scores = exact_scores(&query_values, &key_values, region_tokens, scale);
                    self.decode_exact_scores_nanos = self
                        .decode_exact_scores_nanos
                        .saturating_add(elapsed_nanos(exact_scores_started));
                    regional_certificate.materialize(
                        region.block_id,
                        &scores,
                        &state_uppers,
                        &state_value_max,
                    )?;
                    materialized_region_scores.insert(region.block_id, scores);
                    materialized_region_keys.insert(region.block_id, key);
                }
            }
            let bound_started = std::time::Instant::now();
            let region_bounds = regional_certificate.bounds();
            self.decode_region_bound_nanos = self
                .decode_region_bound_nanos
                .saturating_add(elapsed_nanos(bound_started));
            if certified(&region_bounds, region_tolerance) || loaded.len() == all_states.len() {
                break region_bounds;
            }
            while next_candidate < candidate_order.len()
                && loaded.contains(&candidate_order[next_candidate])
            {
                next_candidate += 1;
            }
            ensure!(
                next_candidate < candidate_order.len(),
                "ensanchamiento sin región candidata"
            );
            // Best-first branch-and-bound: materialize exactly one next
            // certified state, then recompute the bound.  Growing by
            // `loaded.len()` used to overshoot the proof boundary and could
            // load the whole history even when the next single extent was
            // already sufficient.
            loaded.insert(candidate_order[next_candidate]);
            next_candidate += 1;
            widening_rounds += 1;
        };
        let final_assembly_started = std::time::Instant::now();
        let mut positions = Vec::new();
        let mut scores_by_head = vec![Vec::<f64>::new(); 16];
        for page in pages {
            for region in page
                .certificate
                .regions
                .iter()
                .filter(|region| loaded.contains(&region.block_id))
            {
                positions.extend(
                    (region.local_start..region.local_end).map(|token| page.token_start + token),
                );
                let region_scores = materialized_region_scores
                    .get(&region.block_id)
                    .context("scores K jerárquicos ausentes")?;
                for head in 0..16 {
                    scores_by_head[head].extend_from_slice(&region_scores[head]);
                }
            }
        }
        if let Some((tail_key, _)) = tail {
            let start = tail_start.expect("tail_start presente");
            positions.extend(start..start + tail_key.size()[2] as usize);
            let tail_scores = tail_scores.as_ref().expect("scores tail presentes");
            for head in 0..16 {
                scores_by_head[head].extend_from_slice(&tail_scores[head]);
            }
        }
        ensure!(!positions.is_empty(), "selección jerárquica sin K");
        let mut key_parts = Vec::new();
        for page in pages {
            for region in page
                .certificate
                .regions
                .iter()
                .filter(|region| loaded.contains(&region.block_id))
            {
                key_parts.push(
                    materialized_region_keys
                        .get(&region.block_id)
                        .context("K jerárquica materializada ausente")?
                        .shallow_clone(),
                );
            }
        }
        if let Some((tail_key, _)) = tail {
            key_parts.push(tail_key.shallow_clone());
        }
        ensure!(!key_parts.is_empty(), "selección jerárquica sin K final");
        let key_refs = key_parts.iter().collect::<Vec<_>>();
        let full_key = Tensor::cat(&key_refs, 2);
        self.decode_key_assembly_nanos = self
            .decode_key_assembly_nanos
            .saturating_add(elapsed_nanos(final_assembly_started));
        let region_max_bound = region_bounds.iter().copied().fold(0.0_f64, f64::max);
        self.hierarchy_events += 1;
        self.hierarchy_states_total += all_states.len();
        self.hierarchy_states_initial += initial_loaded_states;
        self.hierarchy_states_loaded += loaded.len();
        self.hierarchy_widening_rounds += widening_rounds;
        self.hierarchy_last_states_initial = initial_loaded_states;
        self.hierarchy_last_states_loaded = loaded.len();
        self.hierarchy_last_widening_rounds = widening_rounds;
        self.hierarchy_last_region_bound = region_max_bound;
        let value_max_by_head = (0..16).map(|head| value_max[head / 8]).collect::<Vec<_>>();
        // When every region is already materialized, the scores are a free
        // exact oracle: quantify what an ideal geometric certificate could
        // omit without performing an additional dense read. If future
        // regional pruning succeeds, this diagnostic deliberately stops
        // sampling rather than destroying the achieved speedup.
        if self.exact_oracle_enabled && loaded.len() == all_states.len() {
            self.exact_oracle_events += 1;
            self.exact_oracle_token_opportunities += positions.len();
            self.exact_oracle_tokens_omissible_at_1e6 +=
                exact_oracle_omissible_tokens(&scores_by_head, &value_max_by_head, 1e-6)?;
            self.exact_oracle_tokens_omissible_at_1e5 +=
                exact_oracle_omissible_tokens(&scores_by_head, &value_max_by_head, 1e-5)?;
            self.exact_oracle_tokens_omissible_at_1e4 +=
                exact_oracle_omissible_tokens(&scores_by_head, &value_max_by_head, 1e-4)?;
            self.exact_oracle_tokens_omissible_at_1e3 +=
                exact_oracle_omissible_tokens(&scores_by_head, &value_max_by_head, 1e-3)?;
            let (minimum_threshold, token_offset, limiting_head) =
                exact_oracle_min_tolerance_candidate(&scores_by_head, &value_max_by_head);
            self.exact_oracle_min_tolerance_for_one_omission = self
                .exact_oracle_min_tolerance_for_one_omission
                .min(minimum_threshold);
            if let Some(metrics) = self.shadow_metrics.as_mut() {
                metrics.record_individual_threshold(
                    ShadowEventId {
                        sequence_position: self.sequence_tokens,
                        layer: layer_index,
                        query_head: Some(limiting_head),
                        block_id: None,
                        token_position: positions.get(token_offset).copied(),
                    },
                    minimum_threshold,
                );
                let position_offsets = positions
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(offset, position)| (position, offset))
                    .collect::<BTreeMap<_, _>>();
                for page in pages {
                    for region in &page.certificate.regions {
                        let token_count = region.local_end - region.local_start;
                        let global_start = page.token_start + region.local_start;
                        let offsets = (global_start..global_start + token_count)
                            .filter_map(|position| position_offsets.get(&position).copied())
                            .collect::<Vec<_>>();
                        if offsets.len() != token_count {
                            continue;
                        }
                        for head in 0..16 {
                            let exact_max = offsets
                                .iter()
                                .map(|offset| scores_by_head[head][*offset])
                                .fold(f64::NEG_INFINITY, f64::max);
                            let exact_log_mass_reference = exact_max + (token_count as f64).ln();
                            metrics.record_region_bound(
                                ShadowEventId {
                                    sequence_position: self.sequence_tokens,
                                    layer: layer_index,
                                    query_head: Some(head),
                                    block_id: Some(region.block_id),
                                    token_position: Some(global_start),
                                },
                                state_uppers[&region.block_id][head],
                                exact_log_mass_reference,
                            );
                        }
                    }
                }
            }
        }
        let avoided_k_tokens = pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .filter(|region| !loaded.contains(&region.block_id))
            .map(|region| region.local_end - region.local_start)
            .sum::<usize>();
        self.hierarchy_k_tokens_avoided += avoided_k_tokens;
        // In the configured-model path the proof is applied, so every skippable K
        // token is physically avoided.
        self.hierarchy_k_events += 1;
        self.hierarchy_k_tokens_skippable += avoided_k_tokens;
        let token_tolerance = (tolerance - region_max_bound).max(0.0);
        let minimum_individual_tolerance =
            exact_oracle_min_tolerance_candidate(&scores_by_head, &value_max_by_head).0;
        let (selected_offsets, token_bounds) = if token_tolerance < minimum_individual_tolerance {
            // If not even the cheapest individual token can be omitted under
            // the monotone mass bound, no non-empty omitted set can qualify.
            // Avoid sorting every head only to rediscover the dense set.
            ((0..positions.len()).collect(), vec![0.0; 16])
        } else {
            select_by_exact_mass(&scores_by_head, &value_max_by_head, token_tolerance)?
        };
        let combined_bounds = region_bounds
            .iter()
            .zip(&token_bounds)
            .map(|(region, token)| region + token)
            .collect::<Vec<_>>();
        self.hierarchy_last_token_bound = token_bounds.iter().copied().fold(0.0_f64, f64::max);
        ensure!(
            certified(&combined_bounds, tolerance),
            "masa jerárquica exacta no certificada"
        );
        self.exact_mass_max_bound = self
            .exact_mass_max_bound
            .max(combined_bounds.iter().copied().fold(0.0_f64, f64::max));
        let selected_positions = selected_offsets
            .iter()
            .map(|offset| positions[*offset])
            .collect::<BTreeSet<_>>();
        // The dimensional catalog is a routing/prioritization signal only.
        // Do not force its voted positions into the historical V materialization:
        // the exact-mass certificate is the authority for the active attention
        // set, while every omitted token remains recoverable from cold BF16.
        let selected_offsets = positions
            .iter()
            .enumerate()
            .filter_map(|(offset, position)| {
                selected_positions.contains(position).then_some(offset)
            })
            .collect::<Vec<_>>();
        let selected_all_materialized = selected_offsets.len() == positions.len();
        let selected_key = if selected_all_materialized {
            full_key.shallow_clone()
        } else {
            let indices = Tensor::from_slice(
                &selected_offsets
                    .iter()
                    .map(|offset| *offset as i64)
                    .collect::<Vec<_>>(),
            );
            full_key.index_select(2, &indices)
        };
        let value_materialization_started = std::time::Instant::now();
        let mut value_parts = Vec::new();
        let mut value_source_tokens = 0_usize;
        for page in pages {
            let local = selected_positions
                .range(page.token_start..page.token_start + page.tokens)
                .map(|position| position - page.token_start)
                .collect::<Vec<_>>();
            if local.is_empty() {
                continue;
            }
            let value = match &page.storage {
                PageStorage::Pending { value, .. } => {
                    value_source_tokens += page.tokens;
                    value.shallow_clone()
                }
                PageStorage::Cold {
                    exact_id,
                    immutable,
                    ..
                } => {
                    let bytes = self
                        .read_cold_value_tokens(*exact_id, *immutable, &local)
                        .with_context(|| format!("lectura V BF16 parcial exacta {exact_id}"))?;
                    value_source_tokens += local.len();
                    self.exact_tensor_from_bytes(&bytes, local.len())?
                }
            };
            value_parts.push(match &page.storage {
                PageStorage::Pending { .. } => select_token_indices(&value, &local),
                PageStorage::Cold { .. } => value,
            });
        }
        if let Some((tail_key, tail_value)) = tail {
            let tail_tokens = tail_key.size()[2] as usize;
            let total_tokens = self.sequence_tokens + 1;
            let tail_start = total_tokens - tail_tokens;
            let local = selected_positions
                .range(tail_start..total_tokens)
                .map(|position| position - tail_start)
                .collect::<Vec<_>>();
            if !local.is_empty() {
                value_source_tokens += tail_tokens;
                value_parts.push(select_token_indices(tail_value, &local));
            }
        }
        ensure!(!value_parts.is_empty(), "selección exacta sin V");
        self.decode_value_materialization_nanos = self
            .decode_value_materialization_nanos
            .saturating_add(elapsed_nanos(value_materialization_started));
        let tokens = positions.len();
        if self.shadow_metrics.is_some() && selected_offsets.len() == tokens {
            let value_refs = value_parts.iter().collect::<Vec<_>>();
            let full_value = Tensor::cat(&value_refs, 2)
                .to_kind(Kind::Float)
                .contiguous();
            let mut value_values = vec![0.0_f32; 2 * tokens * 128];
            full_value.copy_data(&mut value_values, 2 * tokens * 128);
            if let Some(metrics) = self.shadow_metrics.as_mut() {
                record_shadow_attention_errors(
                    metrics,
                    self.sequence_tokens,
                    layer_index,
                    &scores_by_head,
                    &value_values,
                    tokens,
                    &value_max_by_head,
                )?;
            }
        }
        self.exact_k_tokens_scanned += tokens;
        self.exact_v_tokens_materialized += selected_offsets.len();
        // Known live allocations at the handoff to SDPA. The repeated K/V
        // tensors dominate this lower bound. Tensor allocator padding and
        // internal SDPA/libtorch workspaces cannot be observed here.
        let selected = selected_offsets.len();
        let known_scratch = tokens * (512 + 1024 + 1024 + 128 + 8)
            + selected * (512 + 512 + 8192 + 4)
            + value_source_tokens * 512
            + 16 * 128 * std::mem::size_of::<f32>();
        self.known_attention_scratch_peak_bytes =
            self.known_attention_scratch_peak_bytes.max(known_scratch);
        let sdpa_started = std::time::Instant::now();
        let output =
            Self::scaled_attention(query, &[selected_key], &value_parts, self.sequence_tokens);
        self.decode_sdpa_nanos = self
            .decode_sdpa_nanos
            .saturating_add(elapsed_nanos(sdpa_started));
        Ok(output)
    }

    /// Exact multi-token verification over the complete cold history. The
    /// historical BF16 pages are materialized once for the block; each query
    /// still uses a length-one SDPA call and its own causal prefix, preserving
    /// the numerical authority of sequential decode while avoiding repeated
    /// certification and cold reads for every candidate.
    pub(super) fn querywise_dense_cold_attention(
        &mut self,
        query: &Tensor,
        pages: &[MslimPage],
        existing_tail: Option<&(Tensor, Tensor)>,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<Tensor> {
        let query_tokens = query.size()[2] as usize;
        let causal_pairs = (0..query_tokens)
            .map(|index| self.sequence_tokens + index + 1)
            .sum::<usize>();
        self.attention_dense_qk_elements = self
            .attention_dense_qk_elements
            .saturating_add(u128::from(causal_pairs as u64) * u128::from(self.query_heads as u64));
        self.attention_dense_pv_elements = self.attention_dense_pv_elements.saturating_add(
            u128::from(causal_pairs as u64)
                * u128::from(self.query_heads as u64)
                * u128::from(self.head_dim as u64),
        );
        ensure!(query_tokens > 1, "verificación querywise sin bloque");
        let mut key_parts = Vec::new();
        let mut value_parts = Vec::new();
        for page in pages {
            let (key, value) = match &page.storage {
                PageStorage::Pending { key, value, .. } => {
                    (key.shallow_clone(), value.shallow_clone())
                }
                PageStorage::Cold {
                    exact_id,
                    immutable,
                } => {
                    let key = self
                        .read_cold_key(*exact_id, *immutable)
                        .with_context(|| format!("lectura K BF16 completa {exact_id}"))?;
                    let value = self
                        .read_cold_value(*exact_id, *immutable)
                        .with_context(|| format!("lectura V BF16 completa {exact_id}"))?;
                    (
                        self.exact_tensor_from_bytes(&key, page.tokens)?,
                        self.exact_tensor_from_bytes(&value, page.tokens)?,
                    )
                }
            };
            key_parts.push(key);
            value_parts.push(value);
        }
        if let Some((tail_key, tail_value)) = existing_tail {
            key_parts.push(tail_key.shallow_clone());
            value_parts.push(tail_value.shallow_clone());
        }
        key_parts.push(new_key.shallow_clone());
        value_parts.push(new_value.shallow_clone());
        let key_refs = key_parts.iter().collect::<Vec<_>>();
        let value_refs = value_parts.iter().collect::<Vec<_>>();
        let full_key = Tensor::cat(&key_refs, 2);
        let full_value = Tensor::cat(&value_refs, 2);
        ensure!(
            full_key.size()[2] as usize == self.sequence_tokens + query_tokens
                && full_value.size()[2] as usize == self.sequence_tokens + query_tokens,
            "histórico querywise incompatible"
        );
        let mut outputs = Vec::with_capacity(query_tokens);
        for query_index in 0..query_tokens {
            let attended_tokens = self.sequence_tokens + query_index + 1;
            let query_token = query.narrow(2, query_index as i64, 1);
            let groups = Self::gqa_groups(query, &full_key);
            let key = full_key
                .narrow(2, 0, attended_tokens as i64)
                .repeat_interleave_self_int(groups, 1, None);
            let value = full_value
                .narrow(2, 0, attended_tokens as i64)
                .repeat_interleave_self_int(groups, 1, None);
            let mask = Tensor::zeros(
                [1, 1, 1, attended_tokens as i64],
                (query.kind(), query.device()),
            );
            let started = std::time::Instant::now();
            let output = Tensor::scaled_dot_product_attention(
                &query_token,
                &key,
                &value,
                Some(&mask),
                0.0,
                false,
                None,
                false,
            );
            self.record_dense_sdpa(
                1,
                attended_tokens,
                attended_tokens.saturating_sub(1),
                elapsed_nanos(started),
            );
            outputs.push(output);
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&output_refs, 2))
    }

    pub(super) fn scaled_attention(
        query: &Tensor,
        key_pages: &[Tensor],
        value_pages: &[Tensor],
        sequence_tokens: usize,
    ) -> Tensor {
        let key_refs = key_pages.iter().collect::<Vec<_>>();
        let value_refs = value_pages.iter().collect::<Vec<_>>();
        let key = Tensor::cat(&key_refs, 2);
        let value = Tensor::cat(&value_refs, 2);
        let attended_tokens = key.size()[2] as usize;
        let groups = Self::gqa_groups(query, &key);
        let key = key.repeat_interleave_self_int(groups, 1, None);
        let value = value.repeat_interleave_self_int(groups, 1, None);
        let query_tokens = query.size()[2] as usize;
        let mut mask_values = vec![0_f32; query_tokens * attended_tokens];
        if query_tokens > 1 {
            for query_index in 0..query_tokens {
                for key_index in sequence_tokens + query_index + 1..attended_tokens {
                    mask_values[query_index * attended_tokens + key_index] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_slice(&mask_values)
            .view([1, 1, query_tokens as i64, attended_tokens as i64])
            .to_kind(query.kind());
        Tensor::scaled_dot_product_attention(
            query,
            &key,
            &value,
            Some(&mask),
            0.0,
            false,
            None,
            false,
        )
    }
}

fn state_score_uppers(
    pages: &[MslimPage],
    query_values: &[f32],
    scale: f64,
) -> BTreeMap<usize, [f64; 16]> {
    state_score_uppers_generic::<16, 2>(pages, query_values, scale)
}

/// Measurement for the parity-first Llama path: replays the best-first
/// regional K branch-and-bound of the configured-model path on exact scores that were
/// already computed, and returns how many K tokens it would have left
/// unread under `region_tolerance` (renormalized-omission bound). Nothing is
/// skipped; geometry it cannot account for completely yields `Ok(0)`.
fn llama_regional_k_tokens_skippable(
    pages: &[MslimPage],
    regional_uppers: &BTreeMap<usize, [f64; 32]>,
    scores_by_head: &[Vec<f64>],
    tail_value_max: [f64; 8],
    region_tolerance: f64,
) -> Result<usize> {
    ensure!(scores_by_head.len() == 32, "scores Llama sin 32 cabezas");
    let cursor = scores_by_head[0].len();
    let mut block_scores = BTreeMap::<usize, Vec<Vec<f64>>>::new();
    let mut block_tokens = BTreeMap::<usize, usize>::new();
    let mut block_value_max = BTreeMap::<usize, [f64; 8]>::new();
    let mut paged_tokens = 0usize;
    for page in pages {
        let covered = page
            .certificate
            .regions
            .iter()
            .map(|region| region.local_end.saturating_sub(region.local_start))
            .sum::<usize>();
        if page.token_start != paged_tokens || covered != page.tokens {
            return Ok(0); // tokens outside any region: no sound measurement
        }
        paged_tokens += page.tokens;
        for region in &page.certificate.regions {
            ensure!(
                region.value_abs_max_by_kv_head.len() == 8,
                "región Llama sin máximos V por cabeza"
            );
            let start = page.token_start + region.local_start;
            let end = page.token_start + region.local_end;
            ensure!(
                start < end && end <= cursor,
                "región Llama fuera del cursor"
            );
            let scores = block_scores
                .entry(region.block_id)
                .or_insert_with(|| vec![Vec::new(); 32]);
            for (head, head_scores) in scores.iter_mut().enumerate() {
                head_scores.extend_from_slice(&scores_by_head[head][start..end]);
            }
            *block_tokens.entry(region.block_id).or_insert(0) += end - start;
            let maxima = block_value_max.entry(region.block_id).or_insert([0.0; 8]);
            for (slot, value) in maxima.iter_mut().enumerate() {
                *value = value.max(f64::from(region.value_abs_max_by_kv_head[slot]));
            }
        }
    }
    if block_scores.is_empty() || !regional_uppers.keys().eq(block_scores.keys()) {
        return Ok(0);
    }
    let tail_scores = (paged_tokens < cursor).then(|| {
        scores_by_head
            .iter()
            .map(|head| head[paged_tokens..].to_vec())
            .collect::<Vec<_>>()
    });
    let mut certificate = IncrementalRegionCertificate::<32, 8>::new(
        regional_uppers,
        &block_value_max,
        tail_scores.as_ref(),
        tail_value_max,
    )?;
    let mut candidate_order = block_scores.keys().copied().collect::<Vec<_>>();
    candidate_order.sort_unstable_by(|left, right| {
        let upper = |state: &usize| {
            regional_uppers[state]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        upper(right).total_cmp(&upper(left))
    });
    // Like configured-model, the most recent region is always materialized first.
    let mut loaded = BTreeSet::new();
    let latest = *block_scores.keys().next_back().expect("bloques no vacíos");
    certificate.materialize(
        latest,
        &block_scores[&latest],
        regional_uppers,
        &block_value_max,
    )?;
    loaded.insert(latest);
    let mut next_candidate = 0usize;
    while !certified(&certificate.bounds(), region_tolerance) && loaded.len() < block_scores.len() {
        while loaded.contains(&candidate_order[next_candidate]) {
            next_candidate += 1;
        }
        let state = candidate_order[next_candidate];
        certificate.materialize(
            state,
            &block_scores[&state],
            regional_uppers,
            &block_value_max,
        )?;
        loaded.insert(state);
    }
    Ok(block_tokens
        .iter()
        .filter(|(state, _)| !loaded.contains(state))
        .map(|(_, tokens)| *tokens)
        .sum())
}

pub(super) fn state_score_uppers_generic<const Q: usize, const KV: usize>(
    pages: &[MslimPage],
    query_values: &[f32],
    scale: f64,
) -> BTreeMap<usize, [f64; Q]> {
    debug_assert_eq!(Q % KV, 0);
    debug_assert_eq!(query_values.len(), Q * 128);
    let mut fragments = BTreeMap::<usize, [Vec<f64>; Q]>::new();
    for page in pages {
        for region in &page.certificate.regions {
            let state = fragments
                .entry(region.block_id)
                .or_insert_with(|| std::array::from_fn(|_| Vec::new()));
            for head in 0..Q {
                let kv_head = head / (Q / KV);
                let query = &query_values[head * 128..(head + 1) * 128];
                let mut upper = 0.0_f64;
                let mut absolute_product_sum = 0.0_f64;
                for (dimension, query_value) in query.iter().copied().enumerate() {
                    let index = kv_head * 128 + dimension;
                    let minimum =
                        f64::from(half::bf16::from_bits(region.key_minimum_bits[index]).to_f32());
                    let maximum =
                        f64::from(half::bf16::from_bits(region.key_maximum_bits[index]).to_f32());
                    let bound = if query_value >= 0.0 { maximum } else { minimum };
                    upper += f64::from(query_value) * bound;
                    absolute_product_sum +=
                        f64::from(query_value).abs() * minimum.abs().max(maximum.abs());
                }
                // The interval dot product above is evaluated in f64, while
                // the authoritative regional score is a LibTorch FP32
                // matmul followed by FP32 scaling. Bound both rounding stages
                // using Higham's standard gamma_n model instead of an
                // empirical epsilon. This keeps the interval conservative
                // for any accumulation order of 128 finite FP32 products.
                let unit_roundoff = 2_f64.powi(-24);
                let gamma_128 = (128.0 * unit_roundoff) / (1.0 - 128.0 * unit_roundoff);
                let dot_upper = upper + gamma_128 * absolute_product_sum;
                let scaled = dot_upper * scale;
                let scale_rounding = unit_roundoff / (1.0 - unit_roundoff) * scaled.abs();
                state[head].push(
                    scaled + scale_rounding + ((region.local_end - region.local_start) as f64).ln(),
                );
            }
        }
    }
    fragments
        .into_iter()
        .map(|(state_id, heads)| {
            (
                state_id,
                std::array::from_fn(|head| logsumexp_finite(heads[head].iter().copied())),
            )
        })
        .collect()
}

fn exact_scores(
    query_values: &[f32],
    key_values: &[f32],
    tokens: usize,
    scale: f64,
) -> Vec<Vec<f64>> {
    exact_scores_generic::<16, 2>(query_values, key_values, tokens, scale)
}

fn exact_scores_generic<const Q: usize, const KV: usize>(
    query_values: &[f32],
    key_values: &[f32],
    tokens: usize,
    scale: f64,
) -> Vec<Vec<f64>> {
    debug_assert_eq!(Q % KV, 0);
    debug_assert_eq!(query_values.len(), Q * 128);
    debug_assert_eq!(key_values.len(), KV * tokens * 128);
    (0..Q)
        .into_par_iter()
        .map(|head| {
            let kv_head = head / (Q / KV);
            let query = &query_values[head * 128..(head + 1) * 128];
            (0..tokens)
                .map(|token| {
                    let start = (kv_head * tokens + token) * 128;
                    query
                        .iter()
                        .zip(&key_values[start..start + 128])
                        .map(|(left, right)| f64::from(*left) * f64::from(*right))
                        .sum::<f64>()
                        * scale
                })
                .collect()
        })
        .collect()
}

/// Select token coordinates without copying when they form one contiguous
/// interval. `index_select` remains the exact fallback for genuinely sparse
/// selections.
pub(super) fn select_token_indices(tensor: &Tensor, indices: &[usize]) -> Tensor {
    debug_assert!(!indices.is_empty());
    let contiguous = indices.windows(2).all(|pair| pair[1] == pair[0] + 1);
    if contiguous {
        let start = indices[0];
        if start == 0 && indices.len() == tensor.size()[2] as usize {
            tensor.shallow_clone()
        } else {
            tensor.narrow(2, start as i64, indices.len() as i64)
        }
    } else {
        let selection = Tensor::from_slice(
            &indices
                .iter()
                .map(|index| *index as i64)
                .collect::<Vec<_>>(),
        );
        tensor.index_select(2, &selection)
    }
}

pub(super) fn exact_oracle_omissible_tokens(
    scores_by_head: &[Vec<f64>],
    value_abs_max: &[f64],
    tolerance: f64,
) -> Result<usize> {
    let tokens = scores_by_head.first().map_or(0, Vec::len);
    let (selected, _) = select_by_exact_mass(scores_by_head, value_abs_max, tolerance)?;
    Ok(tokens.saturating_sub(selected.len()))
}

/// Smallest component-wise attention-output tolerance for which at least one
/// token can be omitted simultaneously for every query head. This uses exact
/// QK scores; no geometric region bound participates in the result.
pub(super) fn exact_oracle_min_tolerance_candidate(
    scores_by_head: &[Vec<f64>],
    value_abs_max: &[f64],
) -> (f64, usize, usize) {
    let tokens = scores_by_head.first().map_or(0, Vec::len);
    let per_head_thresholds = scores_by_head
        .iter()
        .zip(value_abs_max)
        .map(|(scores, value_max)| {
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f64>();
            scores
                .iter()
                .map(|score| 2.0 * (*score - maximum).exp() / total * *value_max)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (0..tokens)
        .map(|token| {
            let (limiting_head, threshold) = per_head_thresholds
                .iter()
                .enumerate()
                .map(|(head, thresholds)| (head, thresholds[token]))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap_or((0, f64::INFINITY));
            (threshold, token, limiting_head)
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .unwrap_or((f64::INFINITY, 0, 0))
}

fn record_shadow_attention_errors(
    metrics: &mut ShadowMetrics,
    sequence_position: usize,
    layer: usize,
    scores_by_head: &[Vec<f64>],
    value_values: &[f32],
    tokens: usize,
    value_abs_max: &[f64],
) -> Result<()> {
    ensure!(
        value_values.len() == 2 * tokens * 128,
        "V shadow incompatible"
    );
    for tolerance in [1e-4_f64, 1e-3_f64] {
        let (selected, bounds) = select_by_exact_mass(scores_by_head, value_abs_max, tolerance)?;
        let mut kept = vec![false; tokens];
        for token in &selected {
            kept[*token] = true;
        }
        let omitted_tokens = tokens - selected.len();
        for head in 0..16 {
            let scores = &scores_by_head[head];
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let dense_denominator = weights.iter().sum::<f64>();
            let kept_denominator = weights
                .iter()
                .enumerate()
                .filter(|(token, _)| kept[*token])
                .map(|(_, weight)| *weight)
                .sum::<f64>();
            ensure!(
                dense_denominator.is_finite()
                    && dense_denominator > 0.0
                    && kept_denominator.is_finite()
                    && kept_denominator > 0.0,
                "masa shadow inválida"
            );
            let kv_head = head / 8;
            let mut actual_linf_error = 0.0_f64;
            for dimension in 0..128 {
                let mut dense_numerator = 0.0_f64;
                let mut kept_numerator = 0.0_f64;
                for token in 0..tokens {
                    let value =
                        f64::from(value_values[(kv_head * tokens + token) * 128 + dimension]);
                    dense_numerator += weights[token] * value;
                    if kept[token] {
                        kept_numerator += weights[token] * value;
                    }
                }
                let dense = dense_numerator / dense_denominator;
                let partial = kept_numerator / kept_denominator;
                actual_linf_error = actual_linf_error.max((partial - dense).abs());
            }
            metrics.record_attention_error(
                ShadowEventId {
                    sequence_position,
                    layer,
                    query_head: Some(head),
                    block_id: None,
                    token_position: None,
                },
                tolerance,
                omitted_tokens,
                bounds[head],
                actual_linf_error,
            );
        }
    }
    Ok(())
}

fn logsumexp_finite(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum
        + values
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn elapsed_nanos(started: std::time::Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) fn reference_draft(prompt: &[i64], generated: &[u32], maximum: usize) -> Vec<i64> {
    if maximum == 0 || generated.is_empty() {
        return Vec::new();
    }
    let generated = generated
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let longest = generated.len().min(32).min(prompt.len());
    for suffix_len in (1..=longest).rev() {
        let suffix = &generated[generated.len() - suffix_len..];
        for start in (0..=prompt.len() - suffix_len).rev() {
            let continuation = start + suffix_len;
            if prompt[start..continuation] == *suffix && continuation < prompt.len() {
                return prompt[continuation..]
                    .iter()
                    .copied()
                    .take(maximum)
                    .collect();
            }
        }
    }
    Vec::new()
}

#[derive(Clone, Default)]
pub(super) struct SuffixAutomatonState {
    pub(super) maximum_length: usize,
    pub(super) suffix_link: Option<usize>,
    pub(super) transitions: HashMap<i64, usize>,
    pub(super) terminal: bool,
}

/// Linear-size index of every prompt substring. A state with exactly one
/// outgoing edge has a unique observed continuation. Terminal states are not
/// extended because end-of-document is an alternative continuation.
pub(super) struct DeterministicSuffixDrafter {
    pub(super) states: Vec<SuffixAutomatonState>,
}

impl DeterministicSuffixDrafter {
    pub(super) fn build(tokens: &[i64]) -> Self {
        let mut states = vec![SuffixAutomatonState::default()];
        let mut last = 0_usize;
        for &token in tokens {
            let current = states.len();
            states.push(SuffixAutomatonState {
                maximum_length: states[last].maximum_length + 1,
                ..SuffixAutomatonState::default()
            });
            let mut cursor = Some(last);
            while let Some(state) = cursor {
                if states[state].transitions.contains_key(&token) {
                    break;
                }
                states[state].transitions.insert(token, current);
                cursor = states[state].suffix_link;
            }
            if let Some(state) = cursor {
                let target = states[state].transitions[&token];
                if states[state].maximum_length + 1 == states[target].maximum_length {
                    states[current].suffix_link = Some(target);
                } else {
                    let clone = states.len();
                    let mut clone_state = states[target].clone();
                    clone_state.maximum_length = states[state].maximum_length + 1;
                    clone_state.terminal = false;
                    states.push(clone_state);
                    let mut redirect = Some(state);
                    while let Some(candidate) = redirect {
                        if states[candidate].transitions.get(&token) != Some(&target) {
                            break;
                        }
                        states[candidate].transitions.insert(token, clone);
                        redirect = states[candidate].suffix_link;
                    }
                    states[target].suffix_link = Some(clone);
                    states[current].suffix_link = Some(clone);
                }
            } else {
                states[current].suffix_link = Some(0);
            }
            last = current;
        }
        let mut terminal = Some(last);
        while let Some(state) = terminal {
            states[state].terminal = true;
            terminal = states[state].suffix_link;
        }
        Self { states }
    }

    pub(super) fn walk(&self, tokens: &[i64]) -> Option<usize> {
        let mut state = 0_usize;
        for token in tokens {
            state = *self.states[state].transitions.get(token)?;
        }
        Some(state)
    }

    pub(super) fn longest_suffix_state(&self, tokens: &[i64]) -> Option<usize> {
        let maximum_context = tokens.len().min(64);
        (1..=maximum_context)
            .rev()
            .find_map(|length| self.walk(&tokens[tokens.len() - length..]))
    }

    /// The first token is already the target model's exact argmax from the
    /// previous forward. Only the remaining suffix is drafted from unique
    /// prompt continuations.
    pub(super) fn draft(
        &self,
        generated: &[u32],
        exact_first_token: i64,
        maximum: usize,
    ) -> Vec<i64> {
        if maximum == 0 {
            return Vec::new();
        }
        let mut draft = vec![exact_first_token];
        if maximum == 1 {
            return draft;
        }
        let mut context = generated
            .iter()
            .rev()
            .take(63)
            .map(|token| i64::from(*token))
            .collect::<Vec<_>>();
        context.reverse();
        context.push(exact_first_token);
        let Some(mut state) = self.longest_suffix_state(&context) else {
            return draft;
        };
        while draft.len() < maximum {
            let node = &self.states[state];
            if node.terminal || node.transitions.len() != 1 {
                break;
            }
            let (&token, &next) = node.transitions.iter().next().expect("una transición");
            draft.push(token);
            state = next;
        }
        draft
    }
}

#[derive(Debug)]
pub(super) struct AdaptiveSpeculationPolicy {
    pub(super) k: usize,
    pub(super) cooldown_tokens: usize,
    pub(super) bypassed_tokens: usize,
}

impl AdaptiveSpeculationPolicy {
    pub(super) fn new() -> Self {
        Self {
            k: 2,
            cooldown_tokens: 0,
            bypassed_tokens: 0,
        }
    }

    pub(super) fn maximum_draft(&self) -> usize {
        self.k
    }

    pub(super) fn should_speculate(&self) -> bool {
        self.cooldown_tokens == 0
    }

    pub(super) fn record_sequential_bypass(&mut self) {
        self.cooldown_tokens = self.cooldown_tokens.saturating_sub(1);
        self.bypassed_tokens += 1;
    }

    pub(super) fn record_round(&mut self, proposed: usize, accepted: usize) {
        if accepted == proposed {
            self.k = (self.k * 2).min(8);
            self.cooldown_tokens = 0;
            return;
        }
        let rejected = proposed.saturating_sub(accepted);
        self.k = 2;
        // A rejected token already paid for a provisional target computation.
        // Require several exact sequential observations before speculating
        // again, proportional to the discarded work and bounded for recovery.
        self.cooldown_tokens = (rejected * 2).clamp(2, 8);
    }
}

#[cfg(test)]
pub(super) fn omitted_region_bounds(
    selected_scores: &[Vec<f64>],
    state_uppers: &BTreeMap<usize, [f64; 16]>,
    state_value_max: &BTreeMap<usize, [f64; 2]>,
    loaded: &BTreeSet<usize>,
) -> Vec<f64> {
    (0..16)
        .map(|head| {
            let omitted_value_max = state_uppers
                .keys()
                .filter(|state| !loaded.contains(state))
                .map(|state| state_value_max[state][head / 8])
                .fold(0.0_f64, f64::max);
            let kept_value_max = state_value_max
                .keys()
                .filter(|state| loaded.contains(state))
                .map(|state| state_value_max[state][head / 8])
                .fold(0.0_f64, f64::max);
            let omitted = state_uppers
                .iter()
                .filter(|(state, _)| !loaded.contains(state))
                .filter_map(|(state, upper)| state_value_max.get(state).map(|_| upper[head]))
                .collect::<Vec<_>>();
            if omitted.is_empty() {
                return 0.0;
            }
            // Keep the document's proof boundary in one canonical function:
            // softmax mass and V magnitude are independent factors.
            omitted_error_bound(
                &selected_scores[head],
                &omitted,
                omitted_value_max,
                kept_value_max,
            )
            .unwrap_or(f64::INFINITY)
        })
        .collect()
}

/// Incremental form of the regional omitted-mass certificate.
///
/// Region priorities and upper masses are immutable during one decode event,
/// so rebuilding the complete kept-score vectors and sorting every remaining
/// region after each widening round is unnecessary.  This accumulator keeps
/// the same proof quantities in a fixed exponential reference frame and
/// updates only the state that has just been materialized.
pub(super) struct IncrementalRegionCertificate<const Q: usize, const KV: usize> {
    pub(super) reference: [f64; Q],
    pub(super) kept_mass: [f64; Q],
    pub(super) omitted_upper_mass: [f64; Q],
    pub(super) kept_value_max: [f64; KV],
    pub(super) omitted_value_desc: [Vec<(usize, f64)>; KV],
    pub(super) omitted_value_cursor: [usize; KV],
    pub(super) remaining: BTreeSet<usize>,
}

impl<const Q: usize, const KV: usize> IncrementalRegionCertificate<Q, KV> {
    pub(super) fn new(
        state_uppers: &BTreeMap<usize, [f64; Q]>,
        state_value_max: &BTreeMap<usize, [f64; KV]>,
        always_kept_scores: Option<&Vec<Vec<f64>>>,
        always_kept_value_max: [f64; KV],
    ) -> Result<Self> {
        ensure!(
            !state_uppers.is_empty(),
            "certificado incremental sin regiones"
        );
        ensure!(
            state_uppers.keys().eq(state_value_max.keys()),
            "geometría incremental K/V incompatible"
        );
        let reference = std::array::from_fn(|head| {
            state_uppers
                .values()
                .map(|upper| upper[head])
                .chain(
                    always_kept_scores
                        .into_iter()
                        .flat_map(|scores| scores[head].iter().copied()),
                )
                .fold(f64::NEG_INFINITY, f64::max)
        });
        ensure!(
            reference.iter().all(|value| value.is_finite()),
            "referencia incremental no finita"
        );
        let kept_mass = std::array::from_fn(|head| {
            always_kept_scores.map_or(0.0, |scores| {
                scores[head]
                    .iter()
                    .map(|score| (*score - reference[head]).exp())
                    .sum()
            })
        });
        let omitted_upper_mass = std::array::from_fn(|head| {
            state_uppers
                .values()
                .map(|upper| (upper[head] - reference[head]).exp())
                .sum()
        });
        let omitted_value_desc = std::array::from_fn(|kv_head| {
            let mut values = state_value_max
                .iter()
                .map(|(state, maxima)| (*state, maxima[kv_head]))
                .collect::<Vec<_>>();
            values.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            values
        });
        Ok(Self {
            reference,
            kept_mass,
            omitted_upper_mass,
            kept_value_max: always_kept_value_max,
            omitted_value_desc,
            omitted_value_cursor: std::array::from_fn(|_| 0),
            remaining: state_uppers.keys().copied().collect(),
        })
    }

    pub(super) fn materialize(
        &mut self,
        state: usize,
        exact_scores: &[Vec<f64>],
        state_uppers: &BTreeMap<usize, [f64; Q]>,
        state_value_max: &BTreeMap<usize, [f64; KV]>,
    ) -> Result<()> {
        ensure!(
            self.remaining.remove(&state),
            "región materializada dos veces"
        );
        ensure!(
            exact_scores.len() == Q
                && exact_scores.iter().all(
                    |scores| !scores.is_empty() && scores.iter().all(|score| score.is_finite())
                ),
            "scores incrementales inválidos"
        );
        let upper = state_uppers
            .get(&state)
            .context("cota incremental ausente")?;
        for head in 0..Q {
            let exact_log_mass = logsumexp_finite(exact_scores[head].iter().copied());
            ensure!(
                exact_log_mass <= upper[head] + 1e-8,
                "cota regional inferior a la masa exacta"
            );
            self.kept_mass[head] += (exact_log_mass - self.reference[head]).exp();
            self.omitted_upper_mass[head] = (self.omitted_upper_mass[head]
                - (upper[head] - self.reference[head]).exp())
            .max(0.0);
        }
        let maxima = state_value_max
            .get(&state)
            .context("máximo V incremental ausente")?;
        for (kv_head, maximum) in maxima.iter().copied().enumerate() {
            self.kept_value_max[kv_head] = self.kept_value_max[kv_head].max(maximum);
            while self.omitted_value_cursor[kv_head] < self.omitted_value_desc[kv_head].len()
                && !self.remaining.contains(
                    &self.omitted_value_desc[kv_head][self.omitted_value_cursor[kv_head]].0,
                )
            {
                self.omitted_value_cursor[kv_head] += 1;
            }
        }
        Ok(())
    }

    pub(super) fn bounds(&self) -> Vec<f64> {
        (0..Q)
            .map(|head| {
                if self.remaining.is_empty() {
                    return 0.0;
                }
                // Query heads per KV head (8 for configured-model 16Q/2KV, 4 for
                // Llama 32Q/8KV); a fixed `/ 8` was only right for configured-model.
                let kv_head = head / (Q / KV);
                let omitted_value_max = self.omitted_value_desc[kv_head]
                    .get(self.omitted_value_cursor[kv_head])
                    .map_or(0.0, |(_, value)| *value);
                let omitted = self.omitted_upper_mass[head];
                let denominator = self.kept_mass[head] + omitted;
                if denominator == 0.0 {
                    f64::INFINITY
                } else {
                    omitted / denominator * (omitted_value_max + self.kept_value_max[kv_head])
                }
            })
            .collect()
    }
}

impl AttentionCache for MslimKvCache {
    fn sequence_tokens(&self) -> usize {
        self.sequence_tokens
    }

    fn physical_bytes(&self) -> usize {
        let pages: usize = self
            .exact_cold
            .metadata()
            .iter()
            .chain(
                self.prefix_cold
                    .as_ref()
                    .map(ColdStore::metadata)
                    .into_iter()
                    .flatten(),
            )
            .map(|page| (page.key_len + page.value_len) as usize)
            .sum();
        let tails: usize = self
            .layers
            .iter()
            .filter_map(|layer| layer.tail.as_ref())
            .map(|(key, value)| (key.numel() + value.numel()) * 2)
            .sum();
        let pending: usize = self
            .layers
            .iter()
            .flat_map(|layer| &layer.pages)
            .filter_map(|page| match &page.storage {
                PageStorage::Pending { key, value, .. } => Some((key.numel() + value.numel()) * 2),
                PageStorage::Cold { .. } => None,
            })
            .sum();
        let certificates: usize = self
            .layers
            .iter()
            .flat_map(|layer| &layer.pages)
            .map(|page| {
                page.certificate
                    .regions
                    .iter()
                    .map(|region| {
                        (region.key_minimum_bits.len() + region.key_maximum_bits.len())
                            * std::mem::size_of::<u16>()
                            + region.value_abs_max.len() * std::mem::size_of::<f32>()
                            + region.value_abs_max_by_kv_head.len() * std::mem::size_of::<f32>()
                    })
                    .sum::<usize>()
                    + page.certificate.value_abs_max.len() * std::mem::size_of::<f32>()
                    + page.certificate.value_abs_max_by_kv_head.len() * std::mem::size_of::<f32>()
                    + page.certificate.value_abs_max_by_kv_head_dim.len()
                        * std::mem::size_of::<f32>()
            })
            .sum();
        pages + tails + pending + certificates + self.parallel_axis.fingerprint_payload_bytes()
    }

    fn attend(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<Tensor> {
        ensure!(
            layer_index < self.layers.len(),
            "índice de capa KV inválido"
        );
        let query_tokens = query.size()[2] as usize;
        let is_multitoken_verification = self.multitoken_verification && query_tokens > 1;
        let is_prefill = query_tokens > 1 && !is_multitoken_verification;
        if layer_index == 0 {
            let started = std::time::Instant::now();
            self.observe_parallel_axis(new_key)?;
            if self.query_heads == 16 && self.kv_heads == 2 {
                self.autocatalogue_parallel_axis(query)?;
            }
            if is_prefill {
                self.prefill_axis_nanos = self
                    .prefill_axis_nanos
                    .saturating_add(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
            }
        }
        let commit_started = std::time::Instant::now();
        self.collect_conversions()?;
        if is_prefill {
            self.prefill_commit_nanos = self.prefill_commit_nanos.saturating_add(
                commit_started
                    .elapsed()
                    .as_nanos()
                    .min(u128::from(u64::MAX)) as u64,
            );
        }
        let selective_prefill_output = if is_prefill
            && self.query_heads == 32
            && self.kv_heads == 8
            && let Some(tolerance) = self.applied_mass_tolerance
        {
            self.llama_querywise_certified_prefill(
                layer_index,
                query,
                new_key,
                new_value,
                tolerance,
            )?
        } else {
            None
        };
        let prefill_hot = if is_prefill && selective_prefill_output.is_none() {
            let started = std::time::Instant::now();
            let previous = self.layers[layer_index].prefill_hot.take();
            if self.suffix_against_committed {
                let (history_key, history_value) = self.assemble_committed_layer(layer_index)?;
                let suffix = match previous {
                    Some((key, value)) => (
                        Tensor::cat(&[&key, new_key], 2),
                        Tensor::cat(&[&value, new_value], 2),
                    ),
                    None => (new_key.shallow_clone(), new_value.shallow_clone()),
                };
                let combined = (
                    Tensor::cat(&[&history_key, &suffix.0], 2),
                    Tensor::cat(&[&history_value, &suffix.1], 2),
                );
                self.layers[layer_index].prefill_hot =
                    Some((suffix.0.shallow_clone(), suffix.1.shallow_clone()));
                let working_bytes = (combined.0.numel() + combined.1.numel()) * 2
                    + self
                        .layers
                        .iter()
                        .filter_map(|layer| layer.prefill_hot.as_ref())
                        .map(|(key, value)| (key.numel() + value.numel()) * 2)
                        .sum::<usize>();
                self.prefill_working_peak_bytes =
                    self.prefill_working_peak_bytes.max(working_bytes);
                self.prefill_hot_concat_nanos = self
                    .prefill_hot_concat_nanos
                    .saturating_add(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
                Some(combined)
            } else {
                let combined = match previous {
                    Some((key, value)) => (
                        Tensor::cat(&[&key, new_key], 2),
                        Tensor::cat(&[&value, new_value], 2),
                    ),
                    None => (new_key.shallow_clone(), new_value.shallow_clone()),
                };
                self.layers[layer_index].prefill_hot =
                    Some((combined.0.shallow_clone(), combined.1.shallow_clone()));
                let working_bytes = self
                    .layers
                    .iter()
                    .filter_map(|layer| layer.prefill_hot.as_ref())
                    .map(|(key, value)| (key.numel() + value.numel()) * 2)
                    .sum::<usize>();
                self.prefill_working_peak_bytes =
                    self.prefill_working_peak_bytes.max(working_bytes);
                self.prefill_hot_concat_nanos = self
                    .prefill_hot_concat_nanos
                    .saturating_add(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
                Some(combined)
            }
        } else {
            None
        };
        let verification_output = if is_multitoken_verification {
            let pages = self.layers[layer_index].pages.clone();
            let existing_tail = self.layers[layer_index]
                .tail
                .as_ref()
                .map(|(key, value)| (key.shallow_clone(), value.shallow_clone()));
            Some(self.querywise_dense_cold_attention(
                query,
                &pages,
                existing_tail.as_ref(),
                new_key,
                new_value,
            )?)
        } else {
            None
        };
        let previous_tail = self.layers[layer_index].tail.take();
        let (combined_key, combined_value) = match previous_tail {
            Some((tail_key, tail_value)) => (
                Tensor::cat(&[&tail_key, new_key], 2),
                Tensor::cat(&[&tail_value, new_value], 2),
            ),
            None => (new_key.shallow_clone(), new_value.shallow_clone()),
        };
        let tokens = combined_key.size()[2] as usize;
        let combined_start = self
            .sequence_tokens
            .saturating_sub(tokens - new_key.size()[2] as usize);
        let mut offset = 0_usize;
        let mut sealed = Vec::new();
        while !is_multitoken_verification && tokens - offset >= self.storage_frame_tokens {
            let storage_frame_tokens = self.storage_frame_tokens;
            let key_page = combined_key
                .narrow(2, offset as i64, storage_frame_tokens as i64)
                .contiguous();
            let value_page = combined_value
                .narrow(2, offset as i64, storage_frame_tokens as i64)
                .contiguous();
            let token_start = combined_start + offset;
            let certificate_started = std::time::Instant::now();
            let certificate =
                self.page_attention_certificate(&key_page, &value_page, token_start)?;
            if is_prefill {
                self.prefill_certificate_nanos = self.prefill_certificate_nanos.saturating_add(
                    certificate_started
                        .elapsed()
                        .as_nanos()
                        .min(u128::from(u64::MAX)) as u64,
                );
            }
            let shape = PageShape {
                layer: layer_index,
                token_start,
                heads: self.kv_heads as usize,
                tokens: storage_frame_tokens,
                dim: self.head_dim as usize,
                tile: 16,
            };
            let handoff_started = std::time::Instant::now();
            let job = Bf16PageJob::owned(
                shape,
                Self::bf16_bits(&key_page)?,
                Self::bf16_bits(&value_page)?,
            )
            .map_err(anyhow::Error::msg)?;
            let sequence = self.conversion.submit(job).map_err(anyhow::Error::msg)?;
            if is_prefill {
                self.prefill_handoff_nanos = self.prefill_handoff_nanos.saturating_add(
                    handoff_started
                        .elapsed()
                        .as_nanos()
                        .min(u128::from(u64::MAX)) as u64,
                );
            }
            sealed.push(MslimPage {
                token_start,
                tokens: storage_frame_tokens,
                certificate,
                storage: PageStorage::Pending {
                    sequence,
                    key: key_page,
                    value: value_page,
                },
            });
            offset += storage_frame_tokens;
        }
        let tail = if offset < tokens {
            Some((
                combined_key
                    .narrow(2, offset as i64, (tokens - offset) as i64)
                    .contiguous(),
                combined_value
                    .narrow(2, offset as i64, (tokens - offset) as i64)
                    .contiguous(),
            ))
        } else {
            None
        };
        {
            let layer = &mut self.layers[layer_index];
            layer.pages.extend(sealed);
            layer.tail = tail;
        }
        if let Some(output) = verification_output {
            return Ok(output);
        }
        let pages = self.layers[layer_index].pages.clone();
        let tail = self.layers[layer_index]
            .tail
            .as_ref()
            .map(|(key, value)| (key.shallow_clone(), value.shallow_clone()));
        if query_tokens > 1 {
            if let Some(output) = selective_prefill_output {
                return Ok(output);
            }
            let (key, value) = prefill_hot.context("prefill BF16 sin estado de trabajo")?;
            let attended_tokens = key.size()[2] as usize;
            let started = std::time::Instant::now();
            let output = Self::scaled_attention(query, &[key], &[value], self.sequence_tokens);
            self.record_dense_sdpa(
                query_tokens,
                attended_tokens,
                self.sequence_tokens,
                elapsed_nanos(started),
            );
            return Ok(output);
        }
        if self.query_heads != 16 || self.kv_heads != 2 {
            if let Some(epsilon) = self.applied_mass_tolerance
                && let Some(selected) = self.llama_certified_value_attention(
                    layer_index,
                    query,
                    &pages,
                    tail.as_ref(),
                    None,
                    None,
                    self.sequence_tokens,
                    epsilon,
                )?
            {
                return Ok(selected);
            }
            let mut keys = Vec::new();
            let mut values = Vec::new();
            if !self.layers[layer_index].pages.is_empty() {
                let (key, value) = self.assemble_committed_layer(layer_index)?;
                keys.push(key);
                values.push(value);
            }
            if let Some((tail_key, tail_value)) = tail.as_ref() {
                keys.push(tail_key.shallow_clone());
                values.push(tail_value.shallow_clone());
            }
            ensure!(!keys.is_empty(), "KV Llama sin páginas ni cola");
            let attended_tokens = keys.iter().map(|key| key.size()[2] as usize).sum();
            let started = std::time::Instant::now();
            let output = Self::scaled_attention(query, &keys, &values, self.sequence_tokens);
            self.record_dense_sdpa(
                query_tokens,
                attended_tokens,
                self.sequence_tokens,
                elapsed_nanos(started),
            );
            return Ok(output);
        }
        self.exact_mass_attention(
            layer_index,
            query,
            &pages,
            tail.as_ref(),
            self.applied_mass_tolerance.unwrap_or(1e-6),
        )
    }

    fn finish_step(&mut self, added_tokens: usize) {
        self.sequence_tokens += added_tokens;
    }

    fn record_layer_output(
        &mut self,
        layer_index: usize,
        sequence_start: usize,
        output: &Tensor,
    ) -> Result<()> {
        if !self.record_step_logits {
            return Ok(());
        }
        ensure!(
            output.size().len() == 3 && output.size()[0] == 1 && output.size()[1] > 0,
            "salida de capa inválida para diagnóstico"
        );
        let tokens = output.size()[1] as usize;
        let last = output
            .narrow(1, (tokens - 1) as i64, 1)
            .to_kind(Kind::Float)
            .contiguous();
        let mut values = vec![0.0_f32; last.numel()];
        let count = values.len();
        last.copy_data(&mut values, count);
        self.layer_output_traces.push(LayerOutputTrace {
            layer_index,
            sequence_end: sequence_start + tokens,
            last_token: values,
        });
        Ok(())
    }
}

impl DecoderModel {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let spec = ModelSpec::inspect(&model_dir)?;
        spec.ensure_cpu_runtime_compatible()?;
        let geometry = DecoderGeometry::from_spec(&spec)?;
        let weights = WeightArchive::open_dir(&model_dir)?;
        let embedding = weights.require_bf16(
            "model.embed_tokens.weight",
            &[geometry.vocab, geometry.hidden],
        )?;
        let final_norm = weights.require_bf16("model.norm.weight", &[geometry.hidden])?;
        let lm_head = weights.require_bf16("lm_head.weight", &[geometry.vocab, geometry.hidden])?;
        let layers = (0..geometry.n_layers)
            .map(|layer| DecoderLayer::load(&weights, layer, &geometry))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            rms_eps: geometry.rms_eps,
            eos: geometry.eos,
            chat: geometry.chat,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            streaming_prefill: geometry.chat == ChatTemplateFamily::TemplateDelimited,
            declared_context_limit: spec
                .max_position_embeddings
                .context("config.json necesita max_position_embeddings")?,
        })
    }

    pub(super) fn is_eos(&self, token: i64) -> bool {
        self.eos.contains(&token)
    }

    pub(super) fn mslim_cache(&self, storage_frame_tokens: usize) -> Result<MslimKvCache> {
        MslimKvCache::try_new_with_geometry(
            storage_frame_tokens,
            self.layers.len(),
            self.kv_heads,
            self.query_heads,
            self.head_dim,
        )
    }

    pub fn forward_with_cache<C: AttentionCache>(
        &self,
        token_ids: &[i64],
        cache: &mut C,
    ) -> Result<Tensor> {
        ensure!(!token_ids.is_empty(), "secuencia de tokens vacía");
        let past_tokens = cache.sequence_tokens();
        let ids = Tensor::from_slice(token_ids);
        let mut hidden = self.embedding.index_select(0, &ids).unsqueeze(0);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_cached(&hidden, layer_index, past_tokens, cache)?;
        }
        cache.finish_step(token_ids.len());
        let hidden = DecoderLayer::rms_norm(&hidden, &self.final_norm, self.rms_eps);
        Ok(hidden.matmul(&self.lm_head.transpose(0, 1)))
    }

    /// Layer-major causal prefill used by the explicit positive-tolerance
    /// Llama experiment. The cache applies the one-query certificate inside
    /// each attention layer while projections and MLPs remain batched for the
    /// whole tile, matching the canonical/OpenCL prefill execution order.
    pub(super) fn forward_certified_prefill<C: AttentionCache>(
        &self,
        token_ids: &[i64],
        cache: &mut C,
    ) -> Result<Tensor> {
        ensure!(!token_ids.is_empty(), "prefill certificado sin tokens");
        self.forward_with_cache(token_ids, cache)
    }
}
