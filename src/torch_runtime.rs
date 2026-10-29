//! Native Rust ownership of the BF16 libtorch execution backend.
//!
//! This module never invokes Python. It maps the original Hugging Face
//! safetensors file and constructs libtorch tensors directly from its bytes.

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use tch::{Device, Kind, Tensor};
use tokenizers::Tokenizer;

use crate::adaptive_extent::{AxisPolicy, ParallelContextAxis};
#[cfg(test)]
use crate::attention_certificate::omitted_error_bound;
use crate::attention_certificate::{certified, select_by_exact_mass};
use crate::cold_store::{ColdStore, ColdStoreStats, PageGeometry, PreparedPage};
use crate::conversion_worker::{Bf16PageJob, ConversionStats, ConversionWorker, PageShape};
use crate::model_config::ModelSpec;
use crate::prefill_tch::protected_queries;
use crate::shadow_metrics::{ShadowDiagnosticReport, ShadowEventId, ShadowMetrics};

pub struct WeightFile {
    path: PathBuf,
    mapping: Mmap,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).with_context(|| format!("no se pudo abrir {}", path.display()))?;
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("no se pudo mapear {}", path.display()))?;
        SafeTensors::deserialize(&mapping)
            .with_context(|| format!("safetensors inválido: {}", path.display()))?;
        Ok(Self { path, mapping })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let archive = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("safetensors inválido: {}", self.path.display()))?;
        let view = archive
            .tensor(name)
            .with_context(|| format!("peso ausente: {name}"))?;
        let kind = match view.dtype() {
            Dtype::BF16 => Kind::BFloat16,
            Dtype::F16 => Kind::Half,
            Dtype::F32 => Kind::Float,
            other => bail!("dtype de peso no soportado para {name}: {other:?}"),
        };
        let shape = view.shape().iter().map(|&v| v as i64).collect::<Vec<_>>();
        Tensor::f_from_data_size(view.data(), &shape, kind)
            .with_context(|| format!("no se pudo materializar {name}"))
    }

    pub fn require_bf16(&self, name: &str, expected: &[i64]) -> Result<Tensor> {
        let tensor = self.tensor(name)?;
        ensure!(tensor.kind() == Kind::BFloat16, "{name} no es BF16");
        ensure!(
            tensor.size() == expected,
            "geometría incorrecta para {name}"
        );
        Ok(tensor.to_device(Device::Cpu))
    }

    fn tensor_names(&self) -> Result<Vec<String>> {
        let archive = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("safetensors inválido: {}", self.path.display()))?;
        Ok(archive.names().into_iter().cloned().collect())
    }
}

pub struct WeightArchive {
    shards: Vec<WeightFile>,
    owners: HashMap<String, usize>,
}

impl WeightArchive {
    pub fn open_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut paths = Vec::new();
        let direct = model_dir.join("model.safetensors");
        if direct.is_file() {
            paths.push(direct);
        } else {
            paths = std::fs::read_dir(model_dir)
                .with_context(|| format!("no se pudo listar {}", model_dir.display()))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("model-") && name.ends_with(".safetensors")
                        })
                })
                .collect();
            paths.sort();
        }
        ensure!(
            !paths.is_empty(),
            "no hay shards safetensors en {}",
            model_dir.display()
        );
        let shards = paths
            .into_iter()
            .map(WeightFile::open)
            .collect::<Result<Vec<_>>>()?;
        let mut owners = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            for name in shard.tensor_names()? {
                owners.entry(name).or_insert(index);
            }
        }
        Ok(Self { shards, owners })
    }

    pub fn require_bf16(&self, name: &str, expected: &[i64]) -> Result<Tensor> {
        let index = self
            .owners
            .get(name)
            .copied()
            .with_context(|| format!("peso ausente: {name}"))?;
        self.shards[index].require_bf16(name, expected)
    }
}

pub fn model_weight_file(model_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let model_dir = model_dir.as_ref();
    let direct = model_dir.join("model.safetensors");
    if direct.is_file() {
        return Ok(direct);
    }
    let mut shards = std::fs::read_dir(model_dir)
        .with_context(|| format!("no se pudo listar {}", model_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        })
        .collect::<Vec<_>>();
    shards.sort();
    ensure!(
        shards.len() == 1,
        "esta fase requiere exactamente un shard safetensors"
    );
    Ok(shards.remove(0))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChatTemplateFamily {
    MiniCpm5,
    Llama3,
}

struct DecoderGeometry {
    hidden: i64,
    intermediate: i64,
    vocab: i64,
    n_layers: usize,
    query_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    rope_theta: f64,
    rms_eps: f64,
    llama3_rope: bool,
    eos: Vec<i64>,
    chat: ChatTemplateFamily,
}

impl DecoderGeometry {
    fn from_spec(spec: &ModelSpec) -> Result<Self> {
        if spec.legacy_bf16_geometry_match {
            Ok(Self {
                hidden: 1_536,
                intermediate: 4_608,
                vocab: 130_560,
                n_layers: 24,
                query_heads: 16,
                kv_heads: 2,
                head_dim: 128,
                rope_theta: 5_000_000.0,
                rms_eps: 1e-6,
                llama3_rope: false,
                eos: vec![1, 130_073],
                chat: ChatTemplateFamily::MiniCpm5,
            })
        } else if spec.is_llama31_8b_instruct_bf16() {
            Ok(Self {
                hidden: 4_096,
                intermediate: 14_336,
                vocab: 128_256,
                n_layers: 32,
                query_heads: 32,
                kv_heads: 8,
                head_dim: 128,
                rope_theta: spec.rope_theta.unwrap_or(500_000.0),
                rms_eps: spec.rms_norm_eps.unwrap_or(1e-5),
                llama3_rope: true,
                eos: vec![128_001, 128_008, 128_009],
                chat: ChatTemplateFamily::Llama3,
            })
        } else {
            bail!(
                "geometría no soportada por el runtime CPU: {}",
                spec.model_type
            )
        }
    }

    fn query_out(&self) -> i64 {
        self.query_heads * self.head_dim
    }

    fn kv_out(&self) -> i64 {
        self.kv_heads * self.head_dim
    }
}

pub struct MiniCpmLayer {
    input_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    post_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    query_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    rope_theta: f64,
    rms_eps: f64,
    llama3_rope: bool,
}

pub struct MiniCpmModel {
    embedding: Tensor,
    layers: Vec<MiniCpmLayer>,
    final_norm: Tensor,
    lm_head: Tensor,
    rms_eps: f64,
    eos: Vec<i64>,
    chat: ChatTemplateFamily,
    query_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    streaming_prefill: bool,
    declared_context_limit: usize,
}

pub struct DenseKvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    sequence_tokens: usize,
    querywise_attention: bool,
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

    fn gqa_groups(query: &Tensor, key: &Tensor) -> Result<i64> {
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

    fn state_digest(&self) -> Result<String> {
        let mut digest = Sha256::new();
        digest.update(b"terrsvusion-dense-kv-state-v1");
        digest.update((self.sequence_tokens as u64).to_le_bytes());
        for (layer, state) in self.layers.iter().enumerate() {
            let (key, value) = state
                .as_ref()
                .with_context(|| format!("capa densa {layer} sin estado KV"))?;
            for tensor in [key, value] {
                ensure!(tensor.kind() == Kind::BFloat16, "estado KV no BF16");
                let contiguous = tensor.contiguous();
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

    fn compare_appended_state(&self, other: &Self, appended: usize) -> Result<KvStateDifference> {
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
                let left_bits = unsafe {
                    std::slice::from_raw_parts(left.data_ptr().cast::<u16>(), left.numel())
                };
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

    fn truncate_to(&mut self, sequence_tokens: usize) -> Result<()> {
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

struct KvStateDifference {
    first_divergent_layer: Option<usize>,
    divergent_layers: usize,
    key_mismatched_elements: usize,
    value_mismatched_elements: usize,
    elements_compared: usize,
    max_abs_key_difference: f64,
    max_abs_value_difference: f64,
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

struct TerrsvusionPage {
    token_start: usize,
    tokens: usize,
    certificate: PageAttentionCertificate,
    storage: PageStorage,
}

#[derive(Clone, Deserialize, Serialize)]
struct PageAttentionCertificate {
    regions: Vec<KeyRegionCertificate>,
    /// Historical MiniCPM5 2KV bound. Kept unchanged for v1 prefix
    /// compatibility and the original 16Q/2KV branch-and-bound path.
    value_abs_max: [f32; 2],
    /// New, optional exact BF16 V bound for ALL KV heads (8 for Llama).
    /// A missing field in a legacy persisted prefix FAILS CLOSED
    /// to full dense BF16 V reads; it never authorizes an omission.
    #[serde(default)]
    value_abs_max_by_kv_head: Vec<f32>,
}

#[derive(Clone, Deserialize, Serialize)]
struct KeyRegionCertificate {
    block_id: usize,
    state_id: usize,
    local_start: usize,
    local_end: usize,
    key_minimum_bits: Vec<u16>,
    key_maximum_bits: Vec<u16>,
    value_abs_max: [f32; 2],
}

struct ParallelAxisMetrics {
    extents: usize,
    tokens: usize,
    fingerprint_payload_bytes: usize,
    f32_fingerprint_reference_bytes: usize,
    micro_metadata_bytes: usize,
    extent_min: usize,
    extent_max: usize,
    query_events: usize,
    dimensional_views: usize,
    view_tokens: usize,
    lsh_queries: usize,
    lsh_candidates: usize,
    lsh_misses: usize,
    lsh_probe_two: usize,
    lsh_probe_four: usize,
}

enum PageStorage {
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
struct PersistentPrefixPage {
    token_start: usize,
    tokens: usize,
    certificate: PageAttentionCertificate,
    exact_id: u64,
}

#[derive(Deserialize, Serialize)]
struct PersistentPrefixManifest {
    schema: u32,
    model_identity: String,
    prefix_digest: String,
    prefix_tokens: usize,
    storage_frame_tokens: usize,
    layers: Vec<Vec<PersistentPrefixPage>>,
    parallel_axis: ParallelContextAxis,
}

const PREFIX_CACHE_SCHEMA: u32 = 1;

impl Clone for TerrsvusionPage {
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

struct TerrsvusionLayer {
    pages: Vec<TerrsvusionPage>,
    tail: Option<(Tensor, Tensor)>,
    prefill_hot: Option<(Tensor, Tensor)>,
}

pub struct TerrsvusionKvCache {
    layers: Vec<TerrsvusionLayer>,
    kv_heads: i64,
    head_dim: i64,
    query_heads: i64,
    sequence_tokens: usize,
    _cold_directory: tempfile::TempDir,
    exact_cold: ColdStore,
    conversion: ConversionWorker<PreparedPage>,
    storage_frame_tokens: usize,
    parallel_axis: ParallelContextAxis,
    parallel_query_events: usize,
    parallel_dimensional_views: usize,
    parallel_view_tokens: usize,
    parallel_selected_positions: Vec<usize>,
    exact_k_tokens_scanned: usize,
    exact_v_tokens_materialized: usize,
    exact_mass_max_bound: f64,
    decode_metadata_nanos: u64,
    decode_key_materialization_nanos: u64,
    decode_exact_scores_nanos: u64,
    decode_region_bound_nanos: u64,
    decode_key_assembly_nanos: u64,
    decode_value_materialization_nanos: u64,
    decode_sdpa_nanos: u64,
    /// Operation-count accounting for the attention authority. These are
    /// logical BF16 element counts, not claims about opaque LibTorch kernels.
    attention_dense_qk_elements: u128,
    attention_dense_pv_elements: u128,
    attention_v_tokens_avoided: usize,
    /// Maximum known temporary allocation footprint of one autoregressive
    /// attention call. This is an accounting lower bound, not process RSS:
    /// opaque libtorch/SDPA workspace is reported separately as unknown.
    known_attention_scratch_peak_bytes: usize,
    prefill_working_peak_bytes: usize,
    prefill_axis_nanos: u64,
    prefill_hot_concat_nanos: u64,
    prefill_certificate_nanos: u64,
    prefill_handoff_nanos: u64,
    prefill_commit_nanos: u64,
    hierarchy_events: usize,
    hierarchy_states_total: usize,
    hierarchy_states_initial: usize,
    hierarchy_states_loaded: usize,
    hierarchy_widening_rounds: usize,
    hierarchy_k_tokens_avoided: usize,
    hierarchy_last_states_initial: usize,
    hierarchy_last_states_loaded: usize,
    hierarchy_last_widening_rounds: usize,
    hierarchy_last_region_bound: f64,
    hierarchy_last_token_bound: f64,
    exact_oracle_events: usize,
    exact_oracle_token_opportunities: usize,
    exact_oracle_tokens_omissible_at_1e6: usize,
    exact_oracle_tokens_omissible_at_1e5: usize,
    exact_oracle_tokens_omissible_at_1e4: usize,
    exact_oracle_tokens_omissible_at_1e3: usize,
    exact_oracle_min_tolerance_for_one_omission: f64,
    exact_oracle_enabled: bool,
    shadow_metrics: Option<ShadowMetrics>,
    /// Explicit transaction mode for exact multi-token target verification.
    /// It is never inferred from tensor length because prefill is multi-token.
    multitoken_verification: bool,
    prefix_cold: Option<ColdStore>,
    applied_mass_tolerance: Option<f64>,
    prefix_copied_bytes: u64,
    record_step_logits: bool,
    step_logits: Vec<Vec<f32>>,
    suffix_against_committed: bool,
}

impl TerrsvusionKvCache {
    fn conversion_worker(first_sequence: u64) -> Result<ConversionWorker<PreparedPage>> {
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

    fn try_new_with_geometry(
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
            "geometría KV Terrsvusion inválida"
        );
        let cold_directory = tempfile::tempdir().context("directorio KV frío")?;
        let exact_cold = ColdStore::create(cold_directory.path().join("pages-exact.kv"), 0)
            .context("almacén BF16 exacto frío")?;
        let conversion = Self::conversion_worker(0)?;
        let axis_dim = usize::try_from(kv_heads * head_dim).context("dimensión de eje")?;
        let axis_heads = usize::try_from(kv_heads).context("cabezas KV")?;
        Ok(Self {
            layers: (0..n_layers)
                .map(|_| TerrsvusionLayer {
                    pages: Vec::new(),
                    tail: None,
                    prefill_hot: None,
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
            attention_v_tokens_avoided: 0,
            known_attention_scratch_peak_bytes: 0,
            prefill_working_peak_bytes: 0,
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
            suffix_against_committed: false,
        })
    }

    fn cold_store(&mut self, immutable: bool) -> Result<&mut ColdStore> {
        if immutable {
            self.prefix_cold
                .as_mut()
                .context("página inmutable sin almacén de prefijo")
        } else {
            Ok(&mut self.exact_cold)
        }
    }

    fn read_cold_key(&mut self, exact_id: u64, immutable: bool) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_key(exact_id)
            .map_err(anyhow::Error::from)
    }

    fn read_cold_value(&mut self, exact_id: u64, immutable: bool) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_value(exact_id)
            .map_err(anyhow::Error::from)
    }

    fn read_cold_key_tokens(
        &mut self,
        exact_id: u64,
        immutable: bool,
        local: &[usize],
    ) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_key_tokens(exact_id, local)
            .map_err(anyhow::Error::from)
    }

    fn read_cold_value_tokens(
        &mut self,
        exact_id: u64,
        immutable: bool,
        local: &[usize],
    ) -> Result<Vec<u8>> {
        self.cold_store(immutable)?
            .read_value_tokens(exact_id, local)
            .map_err(anyhow::Error::from)
    }

    fn assemble_committed_layer(&mut self, layer_index: usize) -> Result<(Tensor, Tensor)> {
        let pages = self.layers[layer_index].pages.clone();
        ensure!(!pages.is_empty(), "capa comprometida vacía");
        let mut keys = Vec::with_capacity(pages.len());
        let mut values = Vec::with_capacity(pages.len());
        for page in &pages {
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
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        Ok((Tensor::cat(&key_refs, 2), Tensor::cat(&value_refs, 2)))
    }

    fn begin_multitoken_verification(&mut self) -> Result<()> {
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

    fn end_multitoken_verification(&mut self) {
        self.multitoken_verification = false;
    }

    /// Roll back only provisional, unsealed tokens. Multi-token verification
    /// deliberately never submits them to conversion workers, so this cannot
    /// orphan a background job or mutate cold storage.
    fn rollback_unsealed_to(&mut self, sequence_tokens: usize) -> Result<()> {
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
    fn logical_state_digest(&mut self) -> Result<String> {
        self.collect_conversions()?;
        let mut digest = Sha256::new();
        digest.update(b"terrsvusion-dense-kv-state-v1");
        digest.update((self.sequence_tokens as u64).to_le_bytes());
        for layer_index in 0..self.layers.len() {
            let pages = self.layers[layer_index].pages.clone();
            let tail = self.layers[layer_index]
                .tail
                .as_ref()
                .map(|(key, value)| (key.shallow_clone(), value.shallow_clone()));
            // A contiguous [1, 2, tokens, 128] tensor stores every token of
            // head 0 before head 1. Reproduce that logical layout across page
            // boundaries instead of hashing page-major physical records.
            for values in [false, true] {
                for head in 0..2 {
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
                                let head_bytes = page.tokens * 128 * 2;
                                let start = head * head_bytes;
                                digest.update(&bytes[start..start + head_bytes]);
                            }
                            PageStorage::Pending { key, value, .. } => {
                                let tensor = if values { value } else { key }
                                    .narrow(1, head as i64, 1)
                                    .contiguous();
                                ensure!(tensor.kind() == Kind::BFloat16, "estado KV no BF16");
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

    fn persistent_manifest(
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

    fn persist_prefix(
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
        if destination.is_dir() {
            return Ok(());
        }
        let staging = tempfile::Builder::new()
            .prefix(".terrsvusion-prefix-")
            .tempdir_in(cache_root)
            .context("staging de prefijo persistente")?;
        fs::copy(
            self.exact_cold.path(),
            staging.path().join("pages-exact.kv"),
        )
        .context("copiar KV de prefijo persistente")?;
        let manifest = self.persistent_manifest(model_identity, cache_key)?;
        let manifest_bytes = serde_json::to_vec(&manifest).context("serializar prefijo")?;
        fs::write(staging.path().join("manifest.json"), manifest_bytes)
            .context("escribir manifiesto de prefijo")?;
        let staging_path = staging.keep();
        match fs::rename(&staging_path, &destination) {
            Ok(()) => Ok(()),
            Err(_error) if destination.is_dir() => {
                fs::remove_dir_all(&staging_path).ok();
                Ok(())
            }
            Err(error) => Err(error).context("publicar prefijo persistente"),
        }
    }

    #[cfg(test)]
    fn restore_prefix(
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

    fn restore_prefix_geometry(
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

    fn restore_prefix_from_manifest(
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
            .map(|pages| TerrsvusionLayer {
                pages: pages
                    .into_iter()
                    .map(|page| TerrsvusionPage {
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

    #[allow(dead_code)]
    fn hydrate_prefill_hot(&mut self) -> Result<()> {
        for layer_index in 0..self.layers.len() {
            self.layers[layer_index].prefill_hot =
                Some(self.assemble_committed_layer(layer_index)?);
        }
        Ok(())
    }

    fn observe_parallel_axis(&mut self, key: &Tensor) -> Result<()> {
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

    fn autocatalogue_parallel_axis(&mut self, query: &Tensor) -> Result<()> {
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

    fn bf16_bits(tensor: &Tensor) -> Result<Vec<u16>> {
        ensure!(tensor.kind() == Kind::BFloat16, "página no BF16");
        let contiguous = tensor.contiguous();
        let source = unsafe {
            std::slice::from_raw_parts(contiguous.data_ptr().cast::<u16>(), contiguous.numel())
        };
        Ok(source.to_vec())
    }

    fn u16_bytes(values: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) }
    }

    /// Encode interval endpoints conservatively in BF16. Nearest conversion
    /// is not sufficient for a certificate: minima must round downward and
    /// maxima upward in the numeric ordering of BF16 values.
    fn outward_bf16_bounds(values: &[f32], lower: bool) -> Vec<u16> {
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

    fn page_attention_certificate(
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
        // The original MiniCPM 16Q/2KV certificates below remain
        // byte-for-byte compatible with their existing prefix cache.
        let mut value_abs_max_by_kv_head = vec![0.0f32; self.kv_heads as usize];
        value_maximum.copy_data(&mut value_abs_max_by_kv_head, self.kv_heads as usize);
        ensure!(
            value_abs_max_by_kv_head
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "original BF16 V per-KV-head maxima nonfinite"
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
                let mut minimum_values = vec![0.0_f32; 2 * 128];
                let mut maximum_values = vec![0.0_f32; 2 * 128];
                minimum.copy_data(&mut minimum_values, 2 * 128);
                maximum.copy_data(&mut maximum_values, 2 * 128);
                let value_region =
                    value.narrow(2, local_start as i64, (local_end - local_start) as i64);
                let value_max = value_region
                    .abs()
                    .amax([2_i64, 3_i64], false)
                    .to_kind(Kind::Float)
                    .contiguous();
                let mut value_abs_max = [0.0_f32; 2];
                value_max.copy_data(&mut value_abs_max, 2);
                ensure!(
                    value_abs_max.iter().all(|value| value.is_finite()),
                    "certificado V regional no finito"
                );
                Ok(KeyRegionCertificate {
                    block_id: token_start + local_start,
                    state_id: region_axis.state_id,
                    local_start,
                    local_end,
                    key_minimum_bits: Self::outward_bf16_bounds(&minimum_values, true),
                    key_maximum_bits: Self::outward_bf16_bounds(&maximum_values, false),
                    value_abs_max,
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
        })
    }

    fn commit_conversion(
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

    fn collect_conversions(&mut self) -> Result<()> {
        for result in self.conversion.try_collect_ordered() {
            self.commit_conversion(result)?;
        }
        Ok(())
    }

    fn gqa_groups(query: &Tensor, key: &Tensor) -> i64 {
        let query_heads = query.size()[1];
        let kv_heads = key.size()[1].max(1);
        if query_heads % kv_heads == 0 {
            query_heads / kv_heads
        } else {
            1
        }
    }

    fn seal_and_release_prefill(&mut self) -> Result<()> {
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

    fn parallel_axis_metrics(&self) -> ParallelAxisMetrics {
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
            let page = TerrsvusionPage {
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

    fn exact_tensor_from_bytes(&self, bytes: &[u8], tokens: usize) -> Result<Tensor> {
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
    fn llama_certified_value_attention(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        pages: &[TerrsvusionPage],
        tail: Option<&(Tensor, Tensor)>,
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
        let mut max_per_kv = [0.0f64; 8];
        for page in pages {
            // A legacy v1 prefix lacked complete 8-KV-head bounds:
            // its first-two-head certificate is NEVER sufficient
            // authority for Llama omission. Restore old full attention.
            if page.certificate.value_abs_max_by_kv_head.len() != 8 {
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
        }
        let mut keys = Vec::with_capacity(pages.len() + usize::from(tail.is_some()));
        let mut cursor = 0usize;
        for page in pages {
            ensure!(
                page.token_start == cursor && page.tokens > 0,
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
            cursor += page.tokens;
        }
        if let Some((k, _)) = tail {
            keys.push(k.shallow_clone());
            cursor += k.size()[2] as usize;
        }
        if keys.is_empty() {
            return Ok(None);
        }
        ensure!(
            cursor == self.sequence_tokens + 1,
            "Llama exact-score audit does not cover complete BF16 K history"
        );
        let key_refs = keys.iter().collect::<Vec<_>>();
        let full_key = Tensor::cat(&key_refs, 2);
        let gqa = self.query_heads / self.kv_heads;
        let queries = query.to_kind(Kind::Float);
        let repeated = full_key
            .repeat_interleave_self_int(gqa, 1, None)
            .to_kind(Kind::Float);
        let scores = (queries.matmul(&repeated.transpose(-2, -1)) / (self.head_dim as f64).sqrt())
            .to_kind(Kind::Float)
            .contiguous();
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
        let max_per_query = (0..32)
            .map(|head| max_per_kv[head / (32 / 8)])
            .collect::<Vec<_>>();
        // Slack for numerical differences between FP32 score planning
        // and LibTorch BF16 SDPA. Not a guarantee of arbitrary GPU
        // backend parity; keep original path for epsilon=0.
        let allowed = tolerance * 0.5;
        let (chosen, bounds) = select_by_exact_mass(&scores_by_head, &max_per_query, allowed)?;
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
        let key_indices = Tensor::from_slice(&chosen.iter().map(|&x| x as i64).collect::<Vec<_>>());
        let selected_key = full_key.index_select(2, &key_indices);
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
        let output = Self::scaled_attention(query, &[selected_key], &values, self.sequence_tokens);
        self.exact_k_tokens_scanned += cursor;
        self.exact_v_tokens_materialized += chosen.len();
        self.attention_v_tokens_avoided = self
            .attention_v_tokens_avoided
            .saturating_add(cursor.saturating_sub(chosen.len()));
        self.exact_mass_max_bound = self
            .exact_mass_max_bound
            .max(bounds.into_iter().fold(0.0f64, f64::max));
        Ok(Some(output))
    }

    fn exact_mass_attention(
        &mut self,
        layer_index: usize,
        query: &Tensor,
        pages: &[TerrsvusionPage],
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
        self.hierarchy_k_tokens_avoided += pages
            .iter()
            .flat_map(|page| &page.certificate.regions)
            .filter(|region| !loaded.contains(&region.block_id))
            .map(|region| region.local_end - region.local_start)
            .sum::<usize>();
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
    fn querywise_dense_cold_attention(
        &mut self,
        query: &Tensor,
        pages: &[TerrsvusionPage],
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
            outputs.push(Tensor::scaled_dot_product_attention(
                &query_token,
                &key,
                &value,
                Some(&mask),
                0.0,
                false,
                None,
                false,
            ));
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&output_refs, 2))
    }

    fn scaled_attention(
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
    pages: &[TerrsvusionPage],
    query_values: &[f32],
    scale: f64,
) -> BTreeMap<usize, [f64; 16]> {
    state_score_uppers_generic::<16, 2>(pages, query_values, scale)
}

fn state_score_uppers_generic<const Q: usize, const KV: usize>(
    pages: &[TerrsvusionPage],
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
                for (dimension, query_value) in query.iter().copied().enumerate() {
                    let index = kv_head * 128 + dimension;
                    let bound = if query_value >= 0.0 {
                        region.key_maximum_bits[index]
                    } else {
                        region.key_minimum_bits[index]
                    };
                    upper +=
                        f64::from(query_value) * f64::from(half::bf16::from_bits(bound).to_f32());
                }
                state[head]
                    .push(upper * scale + ((region.local_end - region.local_start) as f64).ln());
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
    (0..16)
        .into_par_iter()
        .map(|head| {
            let kv_head = head / 8;
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
fn select_token_indices(tensor: &Tensor, indices: &[usize]) -> Tensor {
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

fn exact_oracle_omissible_tokens(
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
fn exact_oracle_min_tolerance_candidate(
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

fn reference_draft(prompt: &[i64], generated: &[u32], maximum: usize) -> Vec<i64> {
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
struct SuffixAutomatonState {
    maximum_length: usize,
    suffix_link: Option<usize>,
    transitions: HashMap<i64, usize>,
    terminal: bool,
}

/// Linear-size index of every prompt substring. A state with exactly one
/// outgoing edge has a unique observed continuation. Terminal states are not
/// extended because end-of-document is an alternative continuation.
struct DeterministicSuffixDrafter {
    states: Vec<SuffixAutomatonState>,
}

impl DeterministicSuffixDrafter {
    fn build(tokens: &[i64]) -> Self {
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

    fn walk(&self, tokens: &[i64]) -> Option<usize> {
        let mut state = 0_usize;
        for token in tokens {
            state = *self.states[state].transitions.get(token)?;
        }
        Some(state)
    }

    fn longest_suffix_state(&self, tokens: &[i64]) -> Option<usize> {
        let maximum_context = tokens.len().min(64);
        (1..=maximum_context)
            .rev()
            .find_map(|length| self.walk(&tokens[tokens.len() - length..]))
    }

    /// The first token is already the target model's exact argmax from the
    /// previous forward. Only the remaining suffix is drafted from unique
    /// prompt continuations.
    fn draft(&self, generated: &[u32], exact_first_token: i64, maximum: usize) -> Vec<i64> {
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
struct AdaptiveSpeculationPolicy {
    k: usize,
    cooldown_tokens: usize,
    bypassed_tokens: usize,
}

impl AdaptiveSpeculationPolicy {
    fn new() -> Self {
        Self {
            k: 2,
            cooldown_tokens: 0,
            bypassed_tokens: 0,
        }
    }

    fn maximum_draft(&self) -> usize {
        self.k
    }

    fn should_speculate(&self) -> bool {
        self.cooldown_tokens == 0
    }

    fn record_sequential_bypass(&mut self) {
        self.cooldown_tokens = self.cooldown_tokens.saturating_sub(1);
        self.bypassed_tokens += 1;
    }

    fn record_round(&mut self, proposed: usize, accepted: usize) {
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
fn omitted_region_bounds(
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
struct IncrementalRegionCertificate<const Q: usize, const KV: usize> {
    reference: [f64; Q],
    kept_mass: [f64; Q],
    omitted_upper_mass: [f64; Q],
    kept_value_max: [f64; KV],
    omitted_value_desc: [Vec<(usize, f64)>; KV],
    omitted_value_cursor: [usize; KV],
    remaining: BTreeSet<usize>,
}

impl<const Q: usize, const KV: usize> IncrementalRegionCertificate<Q, KV> {
    fn new(
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

    fn materialize(
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

    fn bounds(&self) -> Vec<f64> {
        (0..Q)
            .map(|head| {
                if self.remaining.is_empty() {
                    return 0.0;
                }
                let omitted_value_max = self.omitted_value_desc[head / 8]
                    .get(self.omitted_value_cursor[head / 8])
                    .map_or(0.0, |(_, value)| *value);
                let omitted = self.omitted_upper_mass[head];
                let denominator = self.kept_mass[head] + omitted;
                if denominator == 0.0 {
                    f64::INFINITY
                } else {
                    omitted / denominator * (omitted_value_max + self.kept_value_max[head / 8])
                }
            })
            .collect()
    }
}

impl AttentionCache for TerrsvusionKvCache {
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
                    })
                    .sum::<usize>()
                    + page.certificate.value_abs_max.len() * std::mem::size_of::<f32>()
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
        let prefill_hot = if is_prefill {
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
            sealed.push(TerrsvusionPage {
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
            let (key, value) = prefill_hot.context("prefill BF16 sin estado de trabajo")?;
            return Ok(Self::scaled_attention(
                query,
                &[key],
                &[value],
                self.sequence_tokens,
            ));
        }
        if self.query_heads != 16 || self.kv_heads != 2 {
            if let Some(epsilon) = self.applied_mass_tolerance
                && let Some(selected) = self.llama_certified_value_attention(
                    layer_index,
                    query,
                    &pages,
                    tail.as_ref(),
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
            return Ok(Self::scaled_attention(
                query,
                &keys,
                &values,
                self.sequence_tokens,
            ));
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
}

impl MiniCpmModel {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let spec = ModelSpec::inspect(&model_dir)?;
        spec.ensure_torch_runtime_compatible()?;
        let geometry = DecoderGeometry::from_spec(&spec)?;
        let weights = WeightArchive::open_dir(&model_dir)?;
        let embedding = weights.require_bf16(
            "model.embed_tokens.weight",
            &[geometry.vocab, geometry.hidden],
        )?;
        let final_norm = weights.require_bf16("model.norm.weight", &[geometry.hidden])?;
        let lm_head = weights.require_bf16("lm_head.weight", &[geometry.vocab, geometry.hidden])?;
        let layers = (0..geometry.n_layers)
            .map(|layer| MiniCpmLayer::load(&weights, layer, &geometry))
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
            streaming_prefill: geometry.chat == ChatTemplateFamily::Llama3,
            declared_context_limit: spec
                .max_position_embeddings
                .context("config.json necesita max_position_embeddings")?,
        })
    }

    fn is_eos(&self, token: i64) -> bool {
        self.eos.iter().any(|&id| id == token)
    }

    fn terrsvusion_cache(&self, storage_frame_tokens: usize) -> Result<TerrsvusionKvCache> {
        TerrsvusionKvCache::try_new_with_geometry(
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
        let hidden = MiniCpmLayer::rms_norm(&hidden, &self.final_norm, self.rms_eps);
        Ok(hidden.matmul(&self.lm_head.transpose(0, 1)))
    }

    /// Causal prefill path used only by the explicit positive-tolerance Llama
    /// experiment. A multi-token SDPA call cannot apply the one-query
    /// certificate safely, so this path advances one token at a time and
    /// lets the cache decide whether that query can omit V. The exact default
    /// continues to use batched prefill.
    fn forward_certified_prefill<C: AttentionCache>(
        &self,
        token_ids: &[i64],
        cache: &mut C,
    ) -> Result<Tensor> {
        ensure!(!token_ids.is_empty(), "prefill certificado sin tokens");
        let mut logits = None;
        for &token in token_ids {
            logits = Some(self.forward_with_cache(&[token], cache)?);
        }
        logits.context("prefill certificado sin salida")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TorchChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Default)]
pub struct TorchGeneration {
    pub content: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub kv_tokens: usize,
    pub latency_seconds: f64,
    pub prefill_seconds: f64,
    pub seal_seconds: f64,
    pub decode_seconds: f64,
    pub drain_seconds: f64,
    pub kv_physical_bytes: usize,
    pub kv_resident_bytes: usize,
    pub kv_known_attention_scratch_peak_bytes: usize,
    pub kv_prefill_working_peak_bytes: usize,
    pub prefill_axis_nanos: u64,
    pub prefill_hot_concat_nanos: u64,
    pub prefill_certificate_nanos: u64,
    pub prefill_handoff_nanos: u64,
    pub prefill_commit_nanos: u64,
    pub kv_bf16_reference_bytes: usize,
    pub kv_steady_resident_savings_percent: f64,
    pub kv_active_resident_bytes: usize,
    pub kv_inactive_zero_consumption_tokens: usize,
    pub kv_storage_frames: usize,
    pub kv_storage_frame_tokens: usize,
    pub kv_tail_tokens_across_layers: usize,
    pub kv_exact_cold_file_bytes: u64,
    pub kv_exact_cold_reads: u64,
    pub kv_exact_cold_cache_hits: u64,
    pub kv_exact_cold_cache_misses: u64,
    pub kv_exact_cold_evictions: u64,
    pub kv_exact_hot_pages: usize,
    pub kv_exact_hot_bytes: usize,
    pub kv_exact_bytes_read: u64,
    pub kv_exact_cryptographic_digests_verified: u64,
    pub kv_exact_partial_value_reads: u64,
    pub kv_exact_partial_value_payload_bytes: u64,
    pub kv_exact_partial_value_bytes_avoided: u64,
    pub kv_exact_partial_key_reads: u64,
    pub kv_exact_partial_key_payload_bytes: u64,
    pub kv_exact_partial_key_bytes_avoided: u64,
    pub kv_conversion_submitted: u64,
    pub kv_conversion_delivered: u64,
    pub kv_conversion_buffered_out_of_order: usize,
    pub kv_conversion_worker_nanos: u64,
    pub parallel_extents: usize,
    pub parallel_axis_tokens: usize,
    pub parallel_fingerprint_payload_bytes: usize,
    pub parallel_f32_fingerprint_reference_bytes: usize,
    pub parallel_micro_metadata_bytes: usize,
    pub parallel_extent_min_tokens: usize,
    pub parallel_extent_max_tokens: usize,
    pub parallel_query_events: usize,
    pub parallel_dimensional_views: usize,
    pub parallel_view_tokens: usize,
    pub parallel_lsh_queries: usize,
    pub parallel_lsh_candidates: usize,
    pub parallel_lsh_misses: usize,
    pub parallel_lsh_probe_two: usize,
    pub parallel_lsh_probe_four: usize,
    pub exact_k_tokens_scanned: usize,
    pub exact_v_tokens_materialized: usize,
    pub exact_mass_max_bound: f64,
    pub decode_metadata_nanos: u64,
    pub decode_key_materialization_nanos: u64,
    pub decode_exact_scores_nanos: u64,
    pub decode_region_bound_nanos: u64,
    pub decode_key_assembly_nanos: u64,
    pub decode_value_materialization_nanos: u64,
    pub decode_sdpa_nanos: u64,
    pub attention_dense_qk_elements: u128,
    pub attention_dense_pv_elements: u128,
    pub attention_v_tokens_avoided: usize,
    pub hierarchy_events: usize,
    pub hierarchy_states_total: usize,
    pub hierarchy_states_initial: usize,
    pub hierarchy_states_loaded: usize,
    pub hierarchy_widening_rounds: usize,
    pub hierarchy_k_tokens_avoided: usize,
    pub hierarchy_last_states_initial: usize,
    pub hierarchy_last_states_loaded: usize,
    pub hierarchy_last_widening_rounds: usize,
    pub hierarchy_last_region_bound: f64,
    pub hierarchy_last_token_bound: f64,
    pub exact_oracle_events: usize,
    pub exact_oracle_token_opportunities: usize,
    pub exact_oracle_tokens_omissible_at_1e6: usize,
    pub exact_oracle_tokens_omissible_at_1e5: usize,
    pub exact_oracle_tokens_omissible_at_1e4: usize,
    pub exact_oracle_tokens_omissible_at_1e3: usize,
    pub exact_oracle_min_tolerance_for_one_omission: Option<f64>,
    pub exact_oracle_min_logit_margin: Option<f64>,
    pub prefix_cache_hit: bool,
    pub prefix_cache_tokens: usize,
    pub prefix_cache_lookup_seconds: f64,
    pub prefix_cache_hydrate_seconds: f64,
    pub prefix_cache_copied_bytes: u64,
    pub prefix_immutable: bool,
    pub certified_v_tokens_omitted: usize,
    pub applied_mass_tolerance: Option<f64>,
    #[serde(skip_serializing)]
    pub step_logits: Vec<Vec<f32>>,
    pub kv_sha256: String,
    pub speculative_enabled: bool,
    pub speculative_rounds: usize,
    pub speculative_rejected_rounds: usize,
    pub speculative_proposed_tokens: usize,
    pub speculative_accepted_tokens: usize,
    pub speculative_bypassed_tokens: usize,
    pub speculative_target_forward_calls: usize,
    pub speculative_maximum_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_parity: Option<NativeInferenceParity>,
}

#[derive(Debug, Serialize)]
pub struct AppliedOmissionReport {
    pub tolerance: f64,
    pub exact_token_parity: bool,
    pub max_abs_logit_difference: f64,
    pub logit_steps_compared: usize,
    pub certified_v_tokens_omitted: usize,
    pub hierarchy_k_tokens_avoided: usize,
    pub canonical_token_ids: Vec<u32>,
    pub applied_token_ids: Vec<u32>,
    pub canonical: TorchGeneration,
    pub applied: TorchGeneration,
}

#[derive(Debug, Serialize)]
pub struct DenseGeneration {
    pub content: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub kv_tokens: usize,
    pub latency_seconds: f64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub kv_bf16_bytes: usize,
    pub kv_sha256: String,
    #[serde(skip_serializing)]
    pub step_logits: Vec<Vec<f32>>,
}

impl From<DenseGeneration> for TorchGeneration {
    fn from(dense: DenseGeneration) -> Self {
        Self {
            content: dense.content,
            token_ids: dense.token_ids,
            prompt_tokens: dense.prompt_tokens,
            completion_tokens: dense.completion_tokens,
            kv_tokens: dense.kv_tokens,
            latency_seconds: dense.latency_seconds,
            prefill_seconds: dense.prefill_seconds,
            decode_seconds: dense.decode_seconds,
            kv_physical_bytes: dense.kv_bf16_bytes,
            kv_bf16_reference_bytes: dense.kv_bf16_bytes,
            kv_sha256: dense.kv_sha256,
            step_logits: dense.step_logits,
            ..Self::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct NativeStepDivergence {
    pub step: usize,
    pub dense_argmax: u32,
    pub terrsvusion_argmax: u32,
    pub dense_logit_at_dense_argmax: f32,
    pub terrsvusion_logit_at_dense_argmax: f32,
    pub dense_logit_at_terrsvusion_argmax: f32,
    pub terrsvusion_logit_at_terrsvusion_argmax: f32,
    pub linf: f64,
}

#[derive(Debug, Serialize)]
pub struct NativeInferenceParity {
    pub prompt_tokens: usize,
    pub exact_token_parity: bool,
    pub first_token_mismatch: Option<usize>,
    pub max_abs_logit_difference: f64,
    pub logit_steps_compared: usize,
    pub first_logit_step_above_1e3: Option<usize>,
    pub kv_sha256_equal: bool,
    pub dense_kv_sha256: String,
    pub terrsvusion_kv_sha256: String,
    pub dense_tokens: Vec<u32>,
    pub terrsvusion_tokens: Vec<u32>,
    pub dense_text: String,
    pub terrsvusion_text: String,
    pub dense_prefill_seconds: f64,
    pub terrsvusion_prefill_seconds: f64,
    pub certified_v_tokens_omitted: usize,
    pub terrsvusion_kv_resident_bytes: usize,
    pub dense_kv_bf16_bytes: usize,
    pub terrsvusion_kv_bf16_reference_bytes: usize,
    pub prompt_authority: &'static str,
    pub divergence: Option<NativeStepDivergence>,
}

fn capture_last_logits(last: &Tensor) -> Vec<f32> {
    let flat = last.to_kind(Kind::Float).contiguous();
    let mut values = vec![0.0_f32; flat.numel()];
    let count = values.len();
    flat.copy_data(&mut values, count);
    values
}

fn argmax_logit(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn first_step_divergence(step: usize, dense: &[f32], terrsvusion: &[f32]) -> NativeStepDivergence {
    let dense_argmax = argmax_logit(dense);
    let terrsvusion_argmax = argmax_logit(terrsvusion);
    let mut linf = 0.0_f64;
    for (left, right) in dense.iter().zip(terrsvusion) {
        linf = linf.max((*left as f64 - *right as f64).abs());
    }
    NativeStepDivergence {
        step,
        dense_argmax,
        terrsvusion_argmax,
        dense_logit_at_dense_argmax: dense[dense_argmax as usize],
        terrsvusion_logit_at_dense_argmax: terrsvusion[dense_argmax as usize],
        dense_logit_at_terrsvusion_argmax: dense[terrsvusion_argmax as usize],
        terrsvusion_logit_at_terrsvusion_argmax: terrsvusion[terrsvusion_argmax as usize],
        linf,
    }
}

#[derive(Debug, Serialize)]
pub struct MultiTokenVerificationOracle {
    pub candidate_tokens: Vec<u32>,
    pub sequential_next_tokens: Vec<u32>,
    pub batched_next_tokens: Vec<u32>,
    pub sequential_seconds: f64,
    pub batched_seconds: f64,
    pub verification_speedup: f64,
    pub argmax_parity: bool,
    pub max_abs_logit_difference: f64,
    pub sequence_tokens_equal: bool,
    pub physical_bytes_equal: bool,
    pub sequential_kv_sha256: String,
    pub batched_kv_sha256: String,
    pub kv_sha256_equal: bool,
    pub canonical_commit_safe: bool,
    pub first_divergent_layer: Option<usize>,
    pub divergent_layers: usize,
    pub key_bf16_mismatched_elements: usize,
    pub value_bf16_mismatched_elements: usize,
    pub candidate_kv_elements_compared: usize,
    pub max_abs_key_difference: f64,
    pub max_abs_value_difference: f64,
    pub authority: &'static str,
    pub scope: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ReferenceSpeculativeGeneration {
    pub content: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub target_forward_calls: usize,
    pub speculative_rounds: usize,
    pub rejected_rounds: usize,
    pub proposed_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub acceptance_percent: f64,
    pub maximum_k_used: usize,
    pub final_k: usize,
    pub adaptive_bypass_tokens: usize,
    pub final_cooldown_tokens: usize,
    pub final_kv_sha256: String,
    pub kv_tokens: usize,
    pub hierarchy_events: Option<usize>,
    pub hierarchy_widening_rounds: Option<usize>,
    pub exact_k_tokens_scanned: Option<usize>,
    pub exact_v_tokens_materialized: Option<usize>,
    pub exact_cold_reads: Option<u64>,
    pub authority: &'static str,
    pub scope: &'static str,
}

pub struct TorchModelRuntime {
    model: MiniCpmModel,
    tokenizer: Tokenizer,
    prefix_cache_dir: Option<PathBuf>,
    model_identity: String,
}

/// Canonical prefix cache stores immutable original BF16 K/V keyed
/// only by model/tokens/frame geometry, not by attention epsilon.
/// An explicitly approximating decoder MUST NOT reuse or publish a
/// prefix under the same key as an exact BF16 turn.
fn safe_prefix_tokens_for_omission(requested: usize, applied_mass_tolerance: Option<f64>) -> usize {
    if applied_mass_tolerance.is_some() {
        0
    } else {
        requested
    }
}

#[derive(Default)]
struct GenerationInstrumentation {
    exact_oracle_enabled: bool,
    shadow_metrics_enabled: bool,
    prefix_tokens: usize,
    applied_mass_tolerance: Option<f64>,
    record_step_logits: bool,
}

impl TorchModelRuntime {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_internal(model_dir.as_ref(), None)
    }

    pub fn load_with_prefix_cache(
        model_dir: impl AsRef<Path>,
        prefix_cache_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::load_internal(
            model_dir.as_ref(),
            Some(prefix_cache_dir.as_ref().to_path_buf()),
        )
    }

    fn load_internal(model_dir: &Path, prefix_cache_dir: Option<PathBuf>) -> Result<Self> {
        let spec = ModelSpec::inspect(model_dir)?;
        spec.ensure_torch_runtime_compatible()?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_identity = if spec.legacy_bf16_geometry_match {
            let weight_path = model_weight_file(model_dir)?;
            Self::runtime_identity(&weight_path, &tokenizer_path)?
        } else {
            Self::runtime_identity_from_shards(model_dir, &spec, &tokenizer_path)?
        };
        let model = MiniCpmModel::load(model_dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow::anyhow!("tokenizer {}: {error}", tokenizer_path.display()))?;
        Ok(Self {
            model,
            tokenizer,
            prefix_cache_dir,
            model_identity,
        })
    }

    fn runtime_identity(weight_path: &Path, tokenizer_path: &Path) -> Result<String> {
        let mut digest = Sha256::new();
        for path in [weight_path, tokenizer_path] {
            let mut file = File::open(path)
                .with_context(|| format!("abrir identidad runtime {}", path.display()))?;
            let mut buffer = vec![0_u8; 1024 * 1024];
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
            }
        }
        Ok(Self::hex_digest(digest.finalize().as_slice()))
    }

    fn runtime_identity_from_shards(
        model_dir: &Path,
        spec: &ModelSpec,
        tokenizer_path: &Path,
    ) -> Result<String> {
        let mut digest = Sha256::new();
        digest.update(b"terrsvusion-runtime-id-shards-v1");
        for relative in ["config.json", "tokenizer.json"] {
            let path = if relative == "tokenizer.json" {
                tokenizer_path.to_path_buf()
            } else {
                model_dir.join(relative)
            };
            let bytes = std::fs::read(&path)
                .with_context(|| format!("abrir identidad runtime {}", path.display()))?;
            digest.update(&(bytes.len() as u64).to_le_bytes());
            digest.update(&bytes);
        }
        for shard in &spec.weight_shards {
            let path = model_dir.join(shard);
            digest.update(shard.as_bytes());
            digest.update(
                path.metadata()
                    .with_context(|| format!("metadatos {}", path.display()))?
                    .len()
                    .to_le_bytes(),
            );
        }
        Ok(Self::hex_digest(digest.finalize().as_slice()))
    }

    fn hex_digest(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn prefix_digest(&self, prefix_ids: &[i64], storage_frame_tokens: usize) -> String {
        let mut digest = Sha256::new();
        digest.update(b"terrsvusion-prefix-v1");
        digest.update(self.model_identity.as_bytes());
        digest.update((storage_frame_tokens as u64).to_le_bytes());
        for token in prefix_ids {
            digest.update(token.to_le_bytes());
        }
        Self::hex_digest(digest.finalize().as_slice())
    }

    pub fn render_messages(messages: &[TorchChatMessage]) -> Result<String> {
        Self::render_messages_with_thinking(messages, false)
    }

    pub fn render_messages_with_thinking(
        messages: &[TorchChatMessage],
        enable_thinking: bool,
    ) -> Result<String> {
        ensure!(!messages.is_empty(), "messages no puede estar vacío");
        let mut prompt = String::from("<s>");
        for message in messages {
            ensure!(
                matches!(message.role.as_str(), "system" | "user" | "assistant"),
                "rol no soportado: {}",
                message.role
            );
            prompt.push_str("<|im_start|>");
            prompt.push_str(&message.role);
            prompt.push('\n');
            prompt.push_str(&message.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        // MiniCPM5's chat template: enable_thinking=false seeds an empty
        // think block so decode starts at the visible answer, not a CoT.
        if enable_thinking {
            prompt.push_str("<think>\n");
        } else {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        Ok(prompt)
    }

    fn render_llama3_messages(
        messages: &[TorchChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        ensure!(!messages.is_empty(), "messages no puede estar vacío");
        let mut prompt = String::from("<|begin_of_text|>");
        for message in messages {
            ensure!(
                matches!(message.role.as_str(), "system" | "user" | "assistant"),
                "rol no soportado: {}",
                message.role
            );
            prompt.push_str("<|start_header_id|>");
            prompt.push_str(&message.role);
            prompt.push_str("<|end_header_id|>\n\n");
            prompt.push_str(&message.content);
            prompt.push_str("<|eot_id|>");
        }
        if add_generation_prompt {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        Ok(prompt)
    }

    fn chat_prompt(&self, messages: &[TorchChatMessage], enable_thinking: bool) -> Result<String> {
        match self.model.chat {
            ChatTemplateFamily::MiniCpm5 => {
                Self::render_messages_with_thinking(messages, enable_thinking)
            }
            ChatTemplateFamily::Llama3 => Self::render_llama3_messages(messages, true),
        }
    }

    fn chat_prefix(&self, messages: &[TorchChatMessage]) -> Result<String> {
        match self.model.chat {
            ChatTemplateFamily::MiniCpm5 => Self::render_message_prefix(messages),
            ChatTemplateFamily::Llama3 => Self::render_llama3_messages(messages, false),
        }
    }

    fn render_message_prefix(messages: &[TorchChatMessage]) -> Result<String> {
        ensure!(!messages.is_empty(), "prefijo messages vacío");
        let mut prompt = String::from("<s>");
        for message in messages {
            ensure!(
                matches!(message.role.as_str(), "system" | "user" | "assistant"),
                "rol no soportado: {}",
                message.role
            );
            prompt.push_str("<|im_start|>");
            prompt.push_str(&message.role);
            prompt.push('\n');
            prompt.push_str(&message.content);
            prompt.push_str("<|im_end|>\n");
        }
        Ok(prompt)
    }

    pub fn generate_chat(&self, user: &str, max_tokens: usize) -> Result<TorchGeneration> {
        self.generate_messages(
            &[TorchChatMessage {
                role: "user".to_owned(),
                content: user.to_owned(),
            }],
            max_tokens,
        )
    }

    pub fn generate_messages(
        &self,
        messages: &[TorchChatMessage],
        max_tokens: usize,
    ) -> Result<TorchGeneration> {
        self.generate_messages_cancellable(messages, max_tokens, false, &AtomicBool::new(false))
    }

    const DEFAULT_CHAT_SYSTEM: &str = "Te llamas MiniCPM. Eres el asistente de Terrsvusion. Responde de forma directa y breve, en el idioma del usuario. No repitas la pregunta. No pidas aclaraciones si ya se entiende.";
    const DEFAULT_LLAMA_CHAT_SYSTEM: &str = "Eres Llama 3.1 8B Instruct, el asistente de Terrsvusion. Responde de forma directa y breve, en el idioma del usuario. No repitas la pregunta. No pidas aclaraciones si ya se entiende.";

    fn default_system_prompt(&self) -> &'static str {
        match self.model.chat {
            ChatTemplateFamily::MiniCpm5 => Self::DEFAULT_CHAT_SYSTEM,
            ChatTemplateFamily::Llama3 => Self::DEFAULT_LLAMA_CHAT_SYSTEM,
        }
    }

    fn sentence_case_if_shouting(text: &str) -> String {
        let letters = text
            .chars()
            .filter(|character| character.is_alphabetic())
            .collect::<Vec<_>>();
        let words = text
            .split_whitespace()
            .filter(|word| word.chars().any(|character| character.is_alphabetic()))
            .count();
        if words < 2
            || letters.len() < 8
            || letters.iter().any(|character| character.is_lowercase())
        {
            return text.to_owned();
        }
        let mut out = String::with_capacity(text.len());
        let mut capitalize = true;
        for character in text.chars() {
            if character.is_alphabetic() {
                if capitalize {
                    out.extend(character.to_uppercase());
                    capitalize = false;
                } else {
                    out.extend(character.to_lowercase());
                }
            } else {
                out.push(character);
            }
        }
        out
    }

    fn spanish_question_word(word: &str) -> Option<&'static str> {
        match word.to_lowercase().as_str() {
            "que" | "qué" => Some("Qué"),
            "cual" | "cuál" => Some("Cuál"),
            "cuales" | "cuáles" => Some("Cuáles"),
            "quien" | "quién" => Some("Quién"),
            "quienes" | "quiénes" => Some("Quiénes"),
            "como" | "cómo" => Some("Cómo"),
            "donde" | "dónde" => Some("Dónde"),
            "cuando" | "cuándo" => Some("Cuándo"),
            "cuanto" | "cuánto" | "cuanta" | "cuánta" => Some("Cuánto"),
            "cuantos" | "cuántos" | "cuantas" | "cuántas" => Some("Cuántos"),
            _ => None,
        }
    }

    fn spanish_question_if_informal(text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.starts_with('¿') {
            return text.to_owned();
        }
        let without_mark = trimmed.trim_end_matches('?').trim_end();
        let mut parts = without_mark.split_whitespace();
        let Some(first) = parts.next() else {
            return text.to_owned();
        };
        let second = parts.next();
        let (replacement, skip_second) = if first.eq_ignore_ascii_case("por")
            && second.is_some_and(|word| matches!(word.to_lowercase().as_str(), "que" | "qué"))
        {
            ("Por qué", true)
        } else if first.eq_ignore_ascii_case("a")
            && second.is_some_and(|word| matches!(word.to_lowercase().as_str(), "que" | "qué"))
        {
            ("A qué", true)
        } else if let Some(word) = Self::spanish_question_word(first) {
            (word, false)
        } else {
            return text.to_owned();
        };
        let rest = if skip_second {
            without_mark
                .split_whitespace()
                .skip(2)
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            without_mark
                .split_whitespace()
                .skip(1)
                .collect::<Vec<_>>()
                .join(" ")
        };
        if rest.is_empty() {
            format!("¿{replacement}?")
        } else {
            format!("¿{replacement} {rest}?")
        }
    }

    fn expand_definition_question(text: &str) -> String {
        let trimmed = text.trim();
        let topic = trimmed
            .strip_prefix("¿Qué es ")
            .or_else(|| trimmed.strip_prefix("¿Qué son "));
        let Some(topic) = topic else {
            return text.to_owned();
        };
        let topic = topic.trim_end_matches('?').trim();
        if topic.is_empty() {
            return text.to_owned();
        }
        format!(
            "Explica de forma concreta qué es {topic}. Da una definición; no repitas la pregunta."
        )
    }

    fn expand_short_greeting(text: &str) -> String {
        let compact = text
            .trim()
            .trim_matches(|character: char| matches!(character, '!' | '.' | '¡' | '?' | '¿'))
            .trim()
            .to_lowercase();
        if matches!(
            compact.as_str(),
            "hola"
                | "hello"
                | "hi"
                | "hey"
                | "buenas"
                | "buenos dias"
                | "buenas tardes"
                | "buenas noches"
        ) {
            "Saluda en una frase corta y espera la siguiente pregunta.".to_owned()
        } else {
            text.to_owned()
        }
    }

    fn normalize_user_chat_text(text: &str) -> String {
        let shouting = Self::sentence_case_if_shouting(text);
        let greeted = Self::expand_short_greeting(&shouting);
        if greeted != shouting {
            return greeted;
        }
        Self::expand_definition_question(&Self::spanish_question_if_informal(&shouting))
    }

    fn prepare_chat_messages_with_system(
        messages: &[TorchChatMessage],
        system: &str,
    ) -> Vec<TorchChatMessage> {
        let mut prepared = Vec::with_capacity(messages.len() + 1);
        if !messages.iter().any(|message| message.role == "system") {
            prepared.push(TorchChatMessage {
                role: "system".to_owned(),
                content: system.to_owned(),
            });
        }
        prepared.extend(messages.iter().map(|message| {
            if message.role == "user" {
                TorchChatMessage {
                    role: message.role.clone(),
                    content: Self::normalize_user_chat_text(&message.content),
                }
            } else {
                message.clone()
            }
        }));
        prepared
    }

    #[cfg(test)]
    fn prepare_chat_messages(messages: &[TorchChatMessage]) -> Vec<TorchChatMessage> {
        Self::prepare_chat_messages_with_system(messages, Self::DEFAULT_CHAT_SYSTEM)
    }

    fn prepare_runtime_chat_messages(
        &self,
        messages: &[TorchChatMessage],
    ) -> Vec<TorchChatMessage> {
        Self::prepare_chat_messages_with_system(messages, self.default_system_prompt())
    }

    fn wrap_question_if_reference_text(messages: &mut [TorchChatMessage]) {
        let has_reference = messages.iter().any(|message| {
            message.role == "user"
                && message
                    .content
                    .starts_with("Texto de referencia (única fuente")
        });
        if !has_reference {
            return;
        }
        let Some(last) = messages.last_mut() else {
            return;
        };
        if last.role != "user"
            || last
                .content
                .starts_with("Texto de referencia (única fuente")
            || last.content.starts_with("Responde con un dato concreto")
        {
            return;
        }
        last.content = format!(
            "Responde con un dato concreto del texto de referencia. No repitas la pregunta. Pregunta: {}",
            last.content
        );
    }

    fn visible_completion(content: &str, enable_thinking: bool) -> String {
        if enable_thinking {
            return content.to_owned();
        }
        if let Some(index) = content.rfind("</think>") {
            content[index + "</think>".len()..].trim().to_owned()
        } else {
            content.to_owned()
        }
    }

    fn encode_chat_prompt_ids(
        &self,
        messages: &[TorchChatMessage],
        enable_thinking: bool,
    ) -> Result<(Vec<TorchChatMessage>, Vec<i64>)> {
        let mut messages = self.prepare_runtime_chat_messages(messages);
        Self::wrap_question_if_reference_text(&mut messages);
        let prompt = self.chat_prompt(&messages, enable_thinking)?;
        // The chat template already includes bos `<s>`. Encoding with
        // add_special_tokens would prepend a second BOS and derail chat.
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow::anyhow!("tokenización falló: {error}"))?;
        let prompt_ids = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i64)
            .collect::<Vec<_>>();
        Ok((messages, prompt_ids))
    }

    pub fn encode_chat_prompt_token_ids(
        &self,
        messages: &[TorchChatMessage],
        enable_thinking: bool,
    ) -> Result<Vec<i64>> {
        Ok(self.encode_chat_prompt_ids(messages, enable_thinking)?.1)
    }

    pub fn encode_raw_chat_prompt_token_ids(
        &self,
        messages: &[TorchChatMessage],
        enable_thinking: bool,
    ) -> Result<Vec<i64>> {
        let prompt = self.chat_prompt(messages, enable_thinking)?;
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow::anyhow!("tokenización falló: {error}"))?;
        Ok(encoded
            .get_ids()
            .iter()
            .map(|&token| token as i64)
            .collect())
    }

    /// Official MiniCPM chat template + dense BF16 KV. No Terrsvusion cache,
    /// no default system, no Spanish/document rewrites.
    pub fn generate_messages_virgin_dense(
        &self,
        messages: &[TorchChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
    ) -> Result<TorchGeneration> {
        ensure!(!messages.is_empty(), "messages no puede estar vacío");
        let prompt = self.chat_prompt(messages, enable_thinking)?;
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow::anyhow!("tokenización falló: {error}"))?;
        let prompt_ids = encoded
            .get_ids()
            .iter()
            .map(|&token| token as i64)
            .collect::<Vec<_>>();
        let mut dense = self.generate_token_ids_dense(&prompt_ids, max_tokens, 512)?;
        dense.content = Self::visible_completion(&dense.content, enable_thinking);
        Ok(TorchGeneration::from(dense))
    }

    /// Dense BF16 attention on the same chat template, tokenizer and greedy
    /// decoding as `generate_messages`. Isolates model errors from Terrsvusion.
    pub fn generate_messages_dense(
        &self,
        messages: &[TorchChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
    ) -> Result<DenseGeneration> {
        let (_messages, prompt_ids) = self.encode_chat_prompt_ids(messages, enable_thinking)?;
        let mut generation = self.generate_token_ids_dense(&prompt_ids, max_tokens, 512)?;
        generation.content = Self::visible_completion(&generation.content, enable_thinking);
        Ok(generation)
    }

    pub fn generate_messages_cancellable(
        &self,
        messages: &[TorchChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
        cancel: &AtomicBool,
    ) -> Result<TorchGeneration> {
        self.generate_messages_cancellable_with_mass_tolerance(
            messages,
            max_tokens,
            enable_thinking,
            None,
            cancel,
        )
    }

    /// Opt-in only: explicit positive omitted-V softmax-mass tolerance.
    /// The historical chat normalization, system prompt, document
    /// wrapper, Llama tokenizer, greedy and original ColdStore remain
    /// unchanged. No tolerance means the untouched original route.
    pub fn generate_messages_cancellable_with_mass_tolerance(
        &self,
        messages: &[TorchChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
        applied_mass_tolerance: Option<f64>,
        cancel: &AtomicBool,
    ) -> Result<TorchGeneration> {
        if let Some(epsilon) = applied_mass_tolerance {
            ensure!(
                epsilon.is_finite() && epsilon > 0.0 && epsilon <= 1e-2,
                "explicit certified V omission tolerance must be in (0,0.01]"
            );
        }
        let (messages, prompt_ids) = self.encode_chat_prompt_ids(messages, enable_thinking)?;
        let prefix_tokens = if self.prefix_cache_dir.is_some() && messages.len() > 1 {
            let prefix = self.chat_prefix(&messages[..messages.len() - 1])?;
            let encoded_prefix = self
                .tokenizer
                .encode(prefix, false)
                .map_err(|error| anyhow::anyhow!("tokenización de prefijo falló: {error}"))?;
            let prefix_ids = encoded_prefix
                .get_ids()
                .iter()
                .map(|&token| token as i64)
                .collect::<Vec<_>>();
            ensure!(
                prompt_ids.starts_with(&prefix_ids),
                "el tokenizer no conserva la frontera exacta del prefijo"
            );
            prefix_ids.len()
        } else {
            0
        };
        let mut generation = self
            .generate_token_ids_instrumented(
                &prompt_ids,
                max_tokens,
                512,
                512,
                GenerationInstrumentation {
                    prefix_tokens,
                    applied_mass_tolerance,
                    ..GenerationInstrumentation::default()
                },
                Some(cancel),
            )?
            .0;
        generation.content = Self::visible_completion(&generation.content, enable_thinking);
        Ok(generation)
    }

    pub fn context_limit_tokens(&self) -> usize {
        self.model.declared_context_limit
    }

    /// Encode an externally supplied document/query for a real long-context
    /// evaluation. The model and tokenizer remain the single authority.
    pub fn encode_context_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        self.encode_text(text, add_special_tokens)
    }

    pub fn build_book_tokens(&self, book_tokens: usize) -> Result<Vec<i64>> {
        let prefix = self.encode_text(
            "Lee este libro completo y conserva sus datos para responder la pregunta final. ",
            true,
        )?;
        let needle = self.encode_text(
            " Capítulo central. Dato único: la clave secreta verificable del libro es AZULEJO-7319. ",
            false,
        )?;
        let question = self.encode_text(
            "\nFin del libro. Pregunta: ¿Cuál es la clave secreta verificable? Responde solamente con la clave:",
            false,
        )?;
        let filler = self.encode_text(
            " Capítulo de historia. La biblioteca conservaba mapas, cartas, inventarios y recuerdos de muchas generaciones.",
            false,
        )?;
        let fixed = prefix.len() + needle.len() + question.len();
        ensure!(
            book_tokens > fixed && !filler.is_empty(),
            "libro demasiado pequeño"
        );
        let filler_tokens = book_tokens - fixed;
        let mut repeated = Vec::with_capacity(filler_tokens);
        while repeated.len() < filler_tokens {
            let remaining = filler_tokens - repeated.len();
            repeated.extend_from_slice(&filler[..remaining.min(filler.len())]);
        }
        let split = filler_tokens / 2;
        let mut ids = Vec::with_capacity(book_tokens);
        ids.extend(prefix);
        ids.extend_from_slice(&repeated[..split]);
        ids.extend(needle);
        ids.extend_from_slice(&repeated[split..]);
        ids.extend(question);
        ensure!(ids.len() == book_tokens, "longitud de libro incorrecta");
        Ok(ids)
    }

    pub fn build_adversarial_book_tokens(&self, book_tokens: usize) -> Result<Vec<i64>> {
        let prefix = self.encode_text(
            "Auditoría de un archivo de expedientes. Para responder, aplica estas reglas: un registro solo tiene autoridad si ESTADO=VIGENTE; los registros REVOCADO y BORRADOR son distractores aunque tengan una revisión mayor o aparezcan después. Entre registros VIGENTE del mismo expediente manda la revisión mayor. Conserva los identificadores exactamente.\n",
            true,
        )?;
        let authority = self.encode_text(
            "\nEXPEDIENTE K-0713 | REVISION 42 | ESTADO=VIGENTE | CODIGO=QUARTZ-5842 | Este es el valor autorizado.\n",
            false,
        )?;
        let late_conflict = self.encode_text(
            "\nEXPEDIENTE K-0713 | REVISION 99 | ESTADO=REVOCADO | CODIGO=QUARTZ-9999 | Distractor posterior sin autoridad.\nEXPEDIENTE K-0713 | REVISION 100 | ESTADO=BORRADOR | CODIGO=QUARTZ-0000 | Distractor final sin autoridad.\n",
            false,
        )?;
        let question = self.encode_text(
            "\nFIN DEL ARCHIVO. Según las reglas, ¿cuál es el CODIGO autorizado del EXPEDIENTE K-0713? Responde solamente con el código, sin explicación:",
            false,
        )?;
        let fixed = prefix.len() + authority.len() + late_conflict.len() + question.len();
        ensure!(
            book_tokens > fixed + 64,
            "contexto adversarial demasiado pequeño"
        );
        let filler_tokens = book_tokens - fixed;
        let mut filler = Vec::with_capacity(filler_tokens);
        let mut record = 0_usize;
        while filler.len() < filler_tokens {
            let state = match record % 3 {
                0 => "VIGENTE",
                1 => "REVOCADO",
                _ => "BORRADOR",
            };
            let text = format!(
                "EXPEDIENTE K-{id:04} | REVISION {revision} | ESTADO={state} | CODIGO=QUARTZ-{code:04} | Registro independiente.\n",
                id = (record * 37 + 19) % 10_000,
                revision = (record * 11 + 3) % 101,
                code = (record * 97 + 31) % 10_000,
            );
            let encoded = self.encode_text(&text, false)?;
            let remaining = filler_tokens - filler.len();
            filler.extend_from_slice(&encoded[..remaining.min(encoded.len())]);
            record += 1;
        }
        let first_split = filler_tokens / 3;
        let second_split = filler_tokens * 9 / 10;
        let mut ids = Vec::with_capacity(book_tokens);
        ids.extend(prefix);
        ids.extend_from_slice(&filler[..first_split]);
        ids.extend(authority);
        ids.extend_from_slice(&filler[first_split..second_split]);
        ids.extend(late_conflict);
        ids.extend_from_slice(&filler[second_split..]);
        ids.extend(question);
        ensure!(ids.len() == book_tokens, "longitud adversarial incorrecta");
        Ok(ids)
    }

    pub fn generate_token_ids(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
    ) -> Result<TorchGeneration> {
        self.generate_token_ids_with_storage_frame_tokens(prompt_ids, max_tokens, chunk_size, 512)
    }

    pub fn generate_token_ids_with_storage_frame_tokens(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
    ) -> Result<TorchGeneration> {
        self.generate_token_ids_with_oracle(
            prompt_ids,
            max_tokens,
            chunk_size,
            storage_frame_tokens,
            false,
        )
    }

    pub fn generate_token_ids_with_oracle(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
        exact_oracle_enabled: bool,
    ) -> Result<TorchGeneration> {
        Ok(self
            .generate_token_ids_instrumented(
                prompt_ids,
                max_tokens,
                chunk_size,
                storage_frame_tokens,
                GenerationInstrumentation {
                    exact_oracle_enabled,
                    ..GenerationInstrumentation::default()
                },
                None,
            )?
            .0)
    }

    /// Explicit benchmark-only diagnostics. The returned report is separate
    /// from `TorchGeneration`; normal chat and server paths never enable it.
    pub fn generate_token_ids_with_shadow_diagnostics(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
    ) -> Result<(TorchGeneration, ShadowDiagnosticReport)> {
        let (generation, report) = self.generate_token_ids_instrumented(
            prompt_ids,
            max_tokens,
            chunk_size,
            storage_frame_tokens,
            GenerationInstrumentation {
                exact_oracle_enabled: true,
                shadow_metrics_enabled: true,
                prefix_tokens: 0,
                ..GenerationInstrumentation::default()
            },
            None,
        )?;
        Ok((generation, report.context("informe shadow ausente")?))
    }

    pub fn generate_token_ids_with_applied_omission(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
        tolerance: f64,
    ) -> Result<AppliedOmissionReport> {
        ensure!(
            tolerance.is_finite() && tolerance > 0.0,
            "tolerancia de omisión aplicada inválida"
        );
        let canonical = self
            .generate_token_ids_instrumented(
                prompt_ids,
                max_tokens,
                chunk_size,
                storage_frame_tokens,
                GenerationInstrumentation {
                    record_step_logits: true,
                    ..GenerationInstrumentation::default()
                },
                None,
            )?
            .0;
        let applied = self
            .generate_token_ids_instrumented(
                prompt_ids,
                max_tokens,
                chunk_size,
                storage_frame_tokens,
                GenerationInstrumentation {
                    applied_mass_tolerance: Some(tolerance),
                    record_step_logits: true,
                    ..GenerationInstrumentation::default()
                },
                None,
            )?
            .0;
        let steps = canonical.step_logits.len().min(applied.step_logits.len());
        let mut max_abs_logit_difference = 0.0_f64;
        for index in 0..steps {
            ensure!(
                canonical.step_logits[index].len() == applied.step_logits[index].len(),
                "logits de omisión aplicada con geometría distinta"
            );
            for (left, right) in canonical.step_logits[index]
                .iter()
                .zip(&applied.step_logits[index])
            {
                max_abs_logit_difference =
                    max_abs_logit_difference.max((*left as f64 - *right as f64).abs());
            }
        }
        ensure!(
            max_abs_logit_difference <= tolerance,
            "omisión aplicada rechazada: L∞ de logits {:.9} supera la tolerancia {:.9}",
            max_abs_logit_difference,
            tolerance
        );
        Ok(AppliedOmissionReport {
            tolerance,
            exact_token_parity: canonical.token_ids == applied.token_ids,
            max_abs_logit_difference,
            logit_steps_compared: steps,
            certified_v_tokens_omitted: applied.certified_v_tokens_omitted,
            hierarchy_k_tokens_avoided: applied.hierarchy_k_tokens_avoided,
            canonical_token_ids: canonical.token_ids.clone(),
            applied_token_ids: applied.token_ids.clone(),
            canonical,
            applied,
        })
    }

    fn generate_token_ids_instrumented(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
        instrumentation: GenerationInstrumentation,
        cancel: Option<&AtomicBool>,
    ) -> Result<(TorchGeneration, Option<ShadowDiagnosticReport>)> {
        let GenerationInstrumentation {
            exact_oracle_enabled,
            shadow_metrics_enabled,
            prefix_tokens,
            applied_mass_tolerance,
            record_step_logits,
        } = instrumentation;
        // SAFETY: a positive tolerance can alter BF16 attention and thus
        // *future* K/V. Never load/publish the same persistent prefix key
        // as a native/exact turn: an approximate KV snapshot must not be
        // reused by a later exact request (or vice versa).
        // Recompute the whole prompt for explicit experimental omission.
        let prefix_tokens = safe_prefix_tokens_for_omission(prefix_tokens, applied_mass_tolerance);
        let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::SeqCst));
        if cancelled() {
            bail!("inferencia cancelada");
        }
        ensure!((1..=512).contains(&max_tokens), "max_tokens fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        ensure!(
            prefix_tokens <= prompt_ids.len(),
            "prefijo fuera del prompt"
        );
        let started = std::time::Instant::now();
        ensure!(
            prompt_ids.len() + max_tokens <= self.model.declared_context_limit,
            "prompt y generación exceden el límite declarado de {} tokens",
            self.model.declared_context_limit
        );
        let prefix_key = (prefix_tokens > 0)
            .then(|| self.prefix_digest(&prompt_ids[..prefix_tokens], storage_frame_tokens));
        let lookup_started = std::time::Instant::now();
        let restored = match (&self.prefix_cache_dir, &prefix_key) {
            (Some(root), Some(key)) => TerrsvusionKvCache::restore_prefix_geometry(
                root,
                key,
                &self.model_identity,
                storage_frame_tokens,
                self.model.layers.len(),
                self.model.kv_heads,
                self.model.query_heads,
                self.model.head_dim,
            )?,
            _ => None,
        };
        let prefix_cache_lookup_seconds = lookup_started.elapsed().as_secs_f64();
        let prefix_cache_hit = restored.is_some();
        let mut cache = restored.unwrap_or(self.model.terrsvusion_cache(storage_frame_tokens)?);
        cache.exact_oracle_enabled = exact_oracle_enabled || shadow_metrics_enabled;
        cache.shadow_metrics = shadow_metrics_enabled.then(ShadowMetrics::default);
        cache.applied_mass_tolerance = applied_mass_tolerance;
        cache.record_step_logits = record_step_logits;
        let mut generated = Vec::with_capacity(max_tokens);
        let mut prefill_seconds = 0.0_f64;
        let mut seal_seconds = 0.0_f64;
        let mut decode_seconds = 0.0_f64;
        let mut min_logit_margin = f64::INFINITY;
        let mut prefix_cache_hydrate_seconds = 0.0_f64;
        let speculative_enabled = !exact_oracle_enabled
            && !shadow_metrics_enabled
            && applied_mass_tolerance.is_none()
            && !record_step_logits;
        let certified_prefill = self.model.streaming_prefill && applied_mass_tolerance.is_some();
        let drafter = DeterministicSuffixDrafter::build(prompt_ids);
        let mut speculation_policy = AdaptiveSpeculationPolicy::new();
        let mut speculative_rounds = 0_usize;
        let mut speculative_rejected_rounds = 0_usize;
        let mut speculative_proposed_tokens = 0_usize;
        let mut speculative_accepted_tokens = 0_usize;
        let mut speculative_target_forward_calls = 0_usize;
        let mut speculative_maximum_k = 0_usize;
        tch::no_grad(|| -> Result<()> {
            let prefill_started = std::time::Instant::now();
            let mut logits;
            if prefix_cache_hit {
                let suffix = &prompt_ids[prefix_tokens..];
                ensure!(
                    !suffix.is_empty(),
                    "prefijo cacheado consume todo el prompt"
                );
                let hydrate_started = std::time::Instant::now();
                let mut chunks = suffix.chunks(chunk_size).peekable();
                logits = loop {
                    if cancelled() {
                        bail!("inferencia cancelada");
                    }
                    let chunk = chunks.next().context("sufijo sin tokens")?;
                    let output = if certified_prefill {
                        self.model.forward_certified_prefill(chunk, &mut cache)?
                    } else {
                        self.model.forward_with_cache(chunk, &mut cache)?
                    };
                    if chunks.peek().is_none() {
                        break output;
                    }
                    if self.model.streaming_prefill {
                        cache.seal_and_release_prefill()?;
                    }
                };
                prefix_cache_hydrate_seconds = hydrate_started.elapsed().as_secs_f64();
            } else if let (Some(root), Some(key)) = (&self.prefix_cache_dir, &prefix_key) {
                let mut chunks = prompt_ids[..prefix_tokens].chunks(chunk_size).peekable();
                logits = loop {
                    if cancelled() {
                        bail!("inferencia cancelada");
                    }
                    let chunk = chunks.next().context("prefijo sin tokens")?;
                    let output = if certified_prefill {
                        self.model.forward_certified_prefill(chunk, &mut cache)?
                    } else {
                        self.model.forward_with_cache(chunk, &mut cache)?
                    };
                    if chunks.peek().is_none() {
                        break output;
                    }
                    if self.model.streaming_prefill {
                        cache.seal_and_release_prefill()?;
                    }
                };
                cache.seal_partial_storage_frames()?;
                cache.persist_prefix(root, key, &self.model_identity)?;
                cache.suffix_against_committed = true;
                let hydrate_started = std::time::Instant::now();
                for chunk in prompt_ids[prefix_tokens..].chunks(chunk_size) {
                    if cancelled() {
                        bail!("inferencia cancelada");
                    }
                    logits = if certified_prefill {
                        self.model.forward_certified_prefill(chunk, &mut cache)?
                    } else {
                        self.model.forward_with_cache(chunk, &mut cache)?
                    };
                }
                prefix_cache_hydrate_seconds = hydrate_started.elapsed().as_secs_f64();
            } else {
                let mut chunks = prompt_ids.chunks(chunk_size).peekable();
                logits = loop {
                    if cancelled() {
                        bail!("inferencia cancelada");
                    }
                    let chunk = chunks.next().context("prompt sin tokens")?;
                    let output = if certified_prefill {
                        self.model.forward_certified_prefill(chunk, &mut cache)?
                    } else {
                        self.model.forward_with_cache(chunk, &mut cache)?
                    };
                    if chunks.peek().is_none() {
                        break output;
                    }
                    if self.model.streaming_prefill {
                        cache.seal_and_release_prefill()?;
                    }
                }
            }
            prefill_seconds = prefill_started.elapsed().as_secs_f64();
            let seal_started = std::time::Instant::now();
            cache.seal_partial_storage_frames()?;
            seal_seconds = seal_started.elapsed().as_secs_f64();
            let decode_started = std::time::Instant::now();
            while generated.len() < max_tokens {
                if cancelled() {
                    bail!("inferencia cancelada");
                }
                let last = logits.select(1, logits.size()[1] - 1);
                if record_step_logits {
                    let flat = last.to_kind(Kind::Float).contiguous();
                    let mut values = vec![0.0_f32; flat.numel()];
                    let count = values.len();
                    flat.copy_data(&mut values, count);
                    cache.step_logits.push(values);
                }
                if exact_oracle_enabled || shadow_metrics_enabled {
                    let top_two = last.topk(2, -1, true, true).0;
                    let margin = top_two.double_value(&[0, 0]) - top_two.double_value(&[0, 1]);
                    min_logit_margin = min_logit_margin.min(margin);
                }
                let token = last.argmax(-1, false).int64_value(&[0]);
                if self.model.is_eos(token) {
                    break;
                }
                if !speculative_enabled || !speculation_policy.should_speculate() {
                    if speculative_enabled {
                        speculation_policy.record_sequential_bypass();
                    }
                    generated.push(token as u32);
                    if generated.len() < max_tokens {
                        logits = self.model.forward_with_cache(&[token], &mut cache)?;
                        speculative_target_forward_calls += usize::from(speculative_enabled);
                    }
                    continue;
                }
                let remaining = max_tokens - generated.len();
                let draft = drafter.draft(
                    &generated,
                    token,
                    speculation_policy.maximum_draft().min(remaining),
                );
                if draft.len() < 2 {
                    generated.push(token as u32);
                    if generated.len() < max_tokens {
                        logits = self.model.forward_with_cache(&[token], &mut cache)?;
                        speculative_target_forward_calls += 1;
                    }
                    continue;
                }
                speculative_rounds += 1;
                speculative_proposed_tokens += draft.len();
                speculative_maximum_k = speculative_maximum_k.max(draft.len());
                let checkpoint = cache.sequence_tokens();
                cache.begin_multitoken_verification()?;
                let verified_result = self.model.forward_with_cache(&draft, &mut cache);
                cache.end_multitoken_verification();
                let verified = verified_result?;
                speculative_target_forward_calls += 1;
                let mut accepted = 0_usize;
                let mut rejection_target = None;
                let mut stop_reached = false;
                for (index, candidate) in draft.iter().copied().enumerate() {
                    let expected = if index == 0 {
                        token
                    } else {
                        verified
                            .select(1, index as i64 - 1)
                            .argmax(-1, false)
                            .int64_value(&[0])
                    };
                    if self.model.is_eos(expected) {
                        stop_reached = true;
                        break;
                    }
                    if candidate != expected {
                        rejection_target = Some(expected);
                        break;
                    }
                    accepted += 1;
                }
                speculative_accepted_tokens += accepted;
                generated.extend(draft[..accepted].iter().map(|candidate| *candidate as u32));
                speculation_policy.record_round(draft.len(), accepted);
                if stop_reached {
                    cache.rollback_unsealed_to(checkpoint + accepted)?;
                    break;
                }
                if accepted == draft.len() {
                    logits = verified;
                    continue;
                }
                speculative_rejected_rounds += 1;
                cache.rollback_unsealed_to(checkpoint + accepted)?;
                let expected = rejection_target.context("rechazo sin token objetivo")?;
                if generated.len() >= max_tokens {
                    break;
                }
                generated.push(expected as u32);
                if generated.len() < max_tokens {
                    logits = self.model.forward_with_cache(&[expected], &mut cache)?;
                    speculative_target_forward_calls += 1;
                }
            }
            decode_seconds = decode_started.elapsed().as_secs_f64();
            Ok(())
        })?;
        let drain_started = std::time::Instant::now();
        cache.drain_conversions()?;
        let drain_seconds = drain_started.elapsed().as_secs_f64();
        let content = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("decodificación falló: {error}"))?;
        let completion_tokens = generated.len();
        let physical = cache.physical_bytes();
        let resident = cache.resident_bytes();
        let active_resident_kv_bytes = cache.active_resident_kv_bytes();
        let inactive_zero_consumption_tokens = cache.inactive_zero_consumption_tokens();
        let reference = cache.sequence_tokens()
            * cache.layers.len()
            * cache.kv_heads as usize
            * 2
            * cache.head_dim as usize
            * 2;
        let exact_cold = cache.exact_cold_metrics();
        let conversion = cache.conversion_metrics();
        let storage_frame_tokens = cache.storage_frame_tokens();
        let parallel = cache.parallel_axis_metrics();
        let shadow_report = cache.shadow_metrics.take().map(ShadowMetrics::report);
        // Compute after taking operational counters: diagnostic digest reads
        // must not masquerade as decode I/O.
        let kv_sha256 = cache.logical_state_digest()?;
        Ok((
            TorchGeneration {
                content,
                token_ids: generated,
                prompt_tokens: prompt_ids.len(),
                completion_tokens,
                kv_tokens: cache.sequence_tokens(),
                latency_seconds: started.elapsed().as_secs_f64(),
                prefill_seconds,
                seal_seconds,
                decode_seconds,
                drain_seconds,
                kv_physical_bytes: physical,
                kv_resident_bytes: resident,
                kv_known_attention_scratch_peak_bytes: cache.known_attention_scratch_peak_bytes,
                kv_prefill_working_peak_bytes: cache.prefill_working_peak_bytes,
                prefill_axis_nanos: cache.prefill_axis_nanos,
                prefill_hot_concat_nanos: cache.prefill_hot_concat_nanos,
                prefill_certificate_nanos: cache.prefill_certificate_nanos,
                prefill_handoff_nanos: cache.prefill_handoff_nanos,
                prefill_commit_nanos: cache.prefill_commit_nanos,
                kv_bf16_reference_bytes: reference,
                kv_steady_resident_savings_percent: if reference == 0 {
                    0.0
                } else {
                    100.0 * (1.0 - resident as f64 / reference as f64)
                },
                kv_active_resident_bytes: active_resident_kv_bytes,
                kv_inactive_zero_consumption_tokens: inactive_zero_consumption_tokens,
                kv_storage_frames: cache.storage_frame_count(),
                kv_storage_frame_tokens: storage_frame_tokens,
                kv_tail_tokens_across_layers: cache.tail_tokens(),
                kv_exact_cold_file_bytes: exact_cold.committed_bytes,
                kv_exact_cold_reads: exact_cold.reads,
                kv_exact_cold_cache_hits: exact_cold.cache_hits,
                kv_exact_cold_cache_misses: exact_cold.cache_misses,
                kv_exact_cold_evictions: exact_cold.evictions,
                kv_exact_hot_pages: exact_cold.hot_pages,
                kv_exact_hot_bytes: exact_cold.hot_bytes,
                kv_exact_bytes_read: exact_cold.bytes_read,
                kv_exact_cryptographic_digests_verified: exact_cold.cryptographic_digests_verified,
                kv_exact_partial_value_reads: exact_cold.partial_value_reads,
                kv_exact_partial_value_payload_bytes: exact_cold.partial_value_payload_bytes,
                kv_exact_partial_value_bytes_avoided: exact_cold.partial_value_bytes_avoided,
                kv_exact_partial_key_reads: exact_cold.partial_key_reads,
                kv_exact_partial_key_payload_bytes: exact_cold.partial_key_payload_bytes,
                kv_exact_partial_key_bytes_avoided: exact_cold.partial_key_bytes_avoided,
                kv_conversion_submitted: conversion.submitted,
                kv_conversion_delivered: conversion.delivered,
                kv_conversion_buffered_out_of_order: conversion.buffered_out_of_order,
                kv_conversion_worker_nanos: conversion.worker_nanos,
                parallel_extents: parallel.extents,
                parallel_axis_tokens: parallel.tokens,
                parallel_fingerprint_payload_bytes: parallel.fingerprint_payload_bytes,
                parallel_f32_fingerprint_reference_bytes: parallel.f32_fingerprint_reference_bytes,
                parallel_micro_metadata_bytes: parallel.micro_metadata_bytes,
                parallel_extent_min_tokens: parallel.extent_min,
                parallel_extent_max_tokens: parallel.extent_max,
                parallel_query_events: parallel.query_events,
                parallel_dimensional_views: parallel.dimensional_views,
                parallel_view_tokens: parallel.view_tokens,
                parallel_lsh_queries: parallel.lsh_queries,
                parallel_lsh_candidates: parallel.lsh_candidates,
                parallel_lsh_misses: parallel.lsh_misses,
                parallel_lsh_probe_two: parallel.lsh_probe_two,
                parallel_lsh_probe_four: parallel.lsh_probe_four,
                exact_k_tokens_scanned: cache.exact_k_tokens_scanned,
                exact_v_tokens_materialized: cache.exact_v_tokens_materialized,
                exact_mass_max_bound: cache.exact_mass_max_bound,
                decode_metadata_nanos: cache.decode_metadata_nanos,
                decode_key_materialization_nanos: cache.decode_key_materialization_nanos,
                decode_exact_scores_nanos: cache.decode_exact_scores_nanos,
                decode_region_bound_nanos: cache.decode_region_bound_nanos,
                decode_key_assembly_nanos: cache.decode_key_assembly_nanos,
                decode_value_materialization_nanos: cache.decode_value_materialization_nanos,
                decode_sdpa_nanos: cache.decode_sdpa_nanos,
                attention_dense_qk_elements: cache.attention_dense_qk_elements,
                attention_dense_pv_elements: cache.attention_dense_pv_elements,
                attention_v_tokens_avoided: cache.attention_v_tokens_avoided,
                hierarchy_events: cache.hierarchy_events,
                hierarchy_states_total: cache.hierarchy_states_total,
                hierarchy_states_initial: cache.hierarchy_states_initial,
                hierarchy_states_loaded: cache.hierarchy_states_loaded,
                hierarchy_widening_rounds: cache.hierarchy_widening_rounds,
                hierarchy_k_tokens_avoided: cache.hierarchy_k_tokens_avoided,
                hierarchy_last_states_initial: cache.hierarchy_last_states_initial,
                hierarchy_last_states_loaded: cache.hierarchy_last_states_loaded,
                hierarchy_last_widening_rounds: cache.hierarchy_last_widening_rounds,
                hierarchy_last_region_bound: cache.hierarchy_last_region_bound,
                hierarchy_last_token_bound: cache.hierarchy_last_token_bound,
                exact_oracle_events: cache.exact_oracle_events,
                exact_oracle_token_opportunities: cache.exact_oracle_token_opportunities,
                exact_oracle_tokens_omissible_at_1e6: cache.exact_oracle_tokens_omissible_at_1e6,
                exact_oracle_tokens_omissible_at_1e5: cache.exact_oracle_tokens_omissible_at_1e5,
                exact_oracle_tokens_omissible_at_1e4: cache.exact_oracle_tokens_omissible_at_1e4,
                exact_oracle_tokens_omissible_at_1e3: cache.exact_oracle_tokens_omissible_at_1e3,
                exact_oracle_min_tolerance_for_one_omission: cache
                    .exact_oracle_min_tolerance_for_one_omission
                    .is_finite()
                    .then_some(cache.exact_oracle_min_tolerance_for_one_omission),
                exact_oracle_min_logit_margin: min_logit_margin
                    .is_finite()
                    .then_some(min_logit_margin),
                prefix_cache_hit,
                prefix_cache_tokens: prefix_tokens,
                prefix_cache_lookup_seconds,
                prefix_cache_hydrate_seconds,
                prefix_cache_copied_bytes: cache.prefix_copied_bytes,
                prefix_immutable: cache.prefix_cold.is_some(),
                certified_v_tokens_omitted: cache
                    .exact_k_tokens_scanned
                    .saturating_sub(cache.exact_v_tokens_materialized),
                applied_mass_tolerance: cache.applied_mass_tolerance,
                step_logits: std::mem::take(&mut cache.step_logits),
                kv_sha256,
                speculative_enabled,
                speculative_rounds,
                speculative_rejected_rounds,
                speculative_proposed_tokens,
                speculative_accepted_tokens,
                speculative_bypassed_tokens: speculation_policy.bypassed_tokens,
                speculative_target_forward_calls,
                speculative_maximum_k,
                native_parity: None,
            },
            shadow_report,
        ))
    }

    /// Benchmark-only oracle for the hardware/backend precondition behind
    /// lossless speculative decoding. Two independent dense BF16 caches ingest
    /// the same prompt. One verifies `k` teacher-forced candidates through `k`
    /// one-token forwards; the other verifies the same candidates in one
    /// causal multi-token forward. No result is committed to the server path.
    pub fn benchmark_multitoken_verification(
        &self,
        prompt_ids: &[i64],
        candidates: usize,
        chunk_size: usize,
        querywise_attention: bool,
    ) -> Result<MultiTokenVerificationOracle> {
        ensure!((2..=16).contains(&candidates), "candidatos fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        ensure!(
            prompt_ids.len() + candidates <= self.model.declared_context_limit,
            "prompt y candidatos exceden el límite declarado de {} tokens",
            self.model.declared_context_limit
        );
        tch::no_grad(|| -> Result<MultiTokenVerificationOracle> {
            let mut sequential_cache = DenseKvCache::with_layers(self.model.layers.len());
            let mut batched_cache = DenseKvCache::with_layers(self.model.layers.len());
            let mut sequential_logits = None;
            let mut batched_prompt_logits = None;
            for chunk in prompt_ids.chunks(chunk_size) {
                sequential_logits = Some(
                    self.model
                        .forward_with_cache(chunk, &mut sequential_cache)?,
                );
                batched_prompt_logits =
                    Some(self.model.forward_with_cache(chunk, &mut batched_cache)?);
            }
            let mut logits = sequential_logits.context("prefill secuencial sin logits")?;
            let _ = batched_prompt_logits.context("prefill batch sin logits")?;
            batched_cache.querywise_attention = querywise_attention;
            let mut candidate_tokens = Vec::with_capacity(candidates);
            let mut sequential_outputs = Vec::with_capacity(candidates);
            let sequential_started = std::time::Instant::now();
            for _ in 0..candidates {
                let last = logits.select(1, logits.size()[1] - 1);
                let token = last.argmax(-1, false).int64_value(&[0]);
                candidate_tokens.push(token);
                logits = self
                    .model
                    .forward_with_cache(&[token], &mut sequential_cache)?;
                sequential_outputs.push(logits.shallow_clone());
            }
            let sequential_seconds = sequential_started.elapsed().as_secs_f64();
            let batched_started = std::time::Instant::now();
            let batched_logits = self
                .model
                .forward_with_cache(&candidate_tokens, &mut batched_cache)?;
            let batched_seconds = batched_started.elapsed().as_secs_f64();
            let sequential_refs = sequential_outputs.iter().collect::<Vec<_>>();
            let sequential_logits = Tensor::cat(&sequential_refs, 1);
            ensure!(
                sequential_logits.size() == batched_logits.size(),
                "logits secuenciales y batch incompatibles"
            );
            let sequential_argmax = sequential_logits.argmax(-1, false);
            let batched_argmax = batched_logits.argmax(-1, false);
            let mut sequential_next_tokens = vec![0_i64; candidates];
            let mut batched_next_tokens = vec![0_i64; candidates];
            sequential_argmax.copy_data(&mut sequential_next_tokens, candidates);
            batched_argmax.copy_data(&mut batched_next_tokens, candidates);
            let max_abs_logit_difference = (&sequential_logits - &batched_logits)
                .abs()
                .amax([0_i64, 1_i64, 2_i64], false)
                .double_value(&[]);
            let sequence_tokens_equal =
                sequential_cache.sequence_tokens() == batched_cache.sequence_tokens();
            let physical_bytes_equal =
                sequential_cache.physical_bytes() == batched_cache.physical_bytes();
            let sequential_kv_sha256 = sequential_cache.state_digest()?;
            let batched_kv_sha256 = batched_cache.state_digest()?;
            let kv_sha256_equal = sequential_kv_sha256 == batched_kv_sha256;
            let kv_difference =
                sequential_cache.compare_appended_state(&batched_cache, candidates)?;
            let argmax_parity = sequential_argmax.equal(&batched_argmax);
            Ok(MultiTokenVerificationOracle {
                candidate_tokens: candidate_tokens
                    .into_iter()
                    .map(|token| token as u32)
                    .collect(),
                sequential_next_tokens: sequential_next_tokens
                    .into_iter()
                    .map(|token| token as u32)
                    .collect(),
                batched_next_tokens: batched_next_tokens
                    .into_iter()
                    .map(|token| token as u32)
                    .collect(),
                sequential_seconds,
                batched_seconds,
                verification_speedup: sequential_seconds / batched_seconds,
                argmax_parity,
                max_abs_logit_difference,
                sequence_tokens_equal,
                physical_bytes_equal,
                sequential_kv_sha256,
                batched_kv_sha256,
                kv_sha256_equal,
                canonical_commit_safe: argmax_parity
                    && sequence_tokens_equal
                    && physical_bytes_equal
                    && kv_sha256_equal,
                first_divergent_layer: kv_difference.first_divergent_layer,
                divergent_layers: kv_difference.divergent_layers,
                key_bf16_mismatched_elements: kv_difference.key_mismatched_elements,
                value_bf16_mismatched_elements: kv_difference.value_mismatched_elements,
                candidate_kv_elements_compared: kv_difference.elements_compared,
                max_abs_key_difference: kv_difference.max_abs_key_difference,
                max_abs_value_difference: kv_difference.max_abs_value_difference,
                authority: "dense_bf16_target_full_kv",
                scope: if querywise_attention {
                    "benchmark_only_querywise_attention_no_server_commit"
                } else {
                    "benchmark_only_batched_attention_no_server_commit"
                },
            })
        })
    }

    /// Benchmark-only greedy speculative generation with a zero-weight
    /// reference drafter. Draft tokens come exclusively from continuations in
    /// the prompt; the dense BF16 target remains the sole acceptance authority.
    /// Querywise attention preserves the byte-exact sequential KV trajectory.
    pub fn generate_reference_speculative(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
    ) -> Result<ReferenceSpeculativeGeneration> {
        ensure!((1..=512).contains(&max_tokens), "max_tokens fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        let mut cache = DenseKvCache::with_layers(self.model.layers.len());
        let mut generated = Vec::<u32>::with_capacity(max_tokens);
        let mut prefill_seconds = 0.0_f64;
        let mut decode_seconds = 0.0_f64;
        let mut target_forward_calls = 0_usize;
        let mut speculative_rounds = 0_usize;
        let mut rejected_rounds = 0_usize;
        let mut proposed_tokens = 0_usize;
        let mut accepted_draft_tokens = 0_usize;
        let mut maximum_k_used = 0_usize;
        let mut current_k = 2_usize;
        tch::no_grad(|| -> Result<()> {
            let prefill_started = std::time::Instant::now();
            let mut logits = None;
            for chunk in prompt_ids.chunks(chunk_size) {
                logits = Some(self.model.forward_with_cache(chunk, &mut cache)?);
            }
            let mut logits = logits.context("prefill especulativo sin logits")?;
            prefill_seconds = prefill_started.elapsed().as_secs_f64();
            cache.querywise_attention = true;
            let decode_started = std::time::Instant::now();
            while generated.len() < max_tokens {
                let last = logits.select(1, logits.size()[1] - 1);
                let target_token = last.argmax(-1, false).int64_value(&[0]);
                if self.model.is_eos(target_token) {
                    break;
                }
                let remaining = max_tokens - generated.len();
                let draft = reference_draft(prompt_ids, &generated, current_k.min(remaining));
                if draft.len() < 2 {
                    generated.push(target_token as u32);
                    if generated.len() < max_tokens {
                        logits = self.model.forward_with_cache(&[target_token], &mut cache)?;
                        target_forward_calls += 1;
                    }
                    continue;
                }
                speculative_rounds += 1;
                proposed_tokens += draft.len();
                maximum_k_used = maximum_k_used.max(draft.len());
                let checkpoint = cache.sequence_tokens();
                let verified = self.model.forward_with_cache(&draft, &mut cache)?;
                target_forward_calls += 1;
                let mut accepted = 0_usize;
                let mut rejection_target = None;
                for (index, candidate) in draft.iter().copied().enumerate() {
                    let expected = if index == 0 {
                        target_token
                    } else {
                        verified
                            .select(1, index as i64 - 1)
                            .argmax(-1, false)
                            .int64_value(&[0])
                    };
                    if candidate != expected {
                        rejection_target = Some(expected);
                        break;
                    }
                    accepted += 1;
                }
                accepted_draft_tokens += accepted;
                generated.extend(draft[..accepted].iter().map(|token| *token as u32));
                if accepted == draft.len() {
                    logits = verified;
                    current_k = (current_k * 2).min(8);
                    continue;
                }
                rejected_rounds += 1;
                cache.truncate_to(checkpoint + accepted)?;
                current_k = (current_k / 2).max(2);
                let expected = rejection_target.context("rechazo sin token objetivo")?;
                if self.model.is_eos(expected) || generated.len() >= max_tokens {
                    break;
                }
                generated.push(expected as u32);
                if generated.len() < max_tokens {
                    logits = self.model.forward_with_cache(&[expected], &mut cache)?;
                    target_forward_calls += 1;
                }
            }
            decode_seconds = decode_started.elapsed().as_secs_f64();
            Ok(())
        })?;
        let content = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("decodificación falló: {error}"))?;
        let final_kv_sha256 = cache.state_digest()?;
        let kv_tokens = cache.sequence_tokens();
        Ok(ReferenceSpeculativeGeneration {
            content,
            token_ids: generated.clone(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated.len(),
            prefill_seconds,
            decode_seconds,
            target_forward_calls,
            speculative_rounds,
            rejected_rounds,
            proposed_tokens,
            accepted_draft_tokens,
            acceptance_percent: if proposed_tokens == 0 {
                0.0
            } else {
                accepted_draft_tokens as f64 * 100.0 / proposed_tokens as f64
            },
            maximum_k_used,
            final_k: current_k,
            adaptive_bypass_tokens: 0,
            final_cooldown_tokens: 0,
            final_kv_sha256,
            kv_tokens,
            hierarchy_events: None,
            hierarchy_widening_rounds: None,
            exact_k_tokens_scanned: None,
            exact_v_tokens_materialized: None,
            exact_cold_reads: None,
            authority: "dense_bf16_target_querywise_exact",
            scope: "benchmark_only_reference_drafter_no_server_commit",
        })
    }

    /// The same reference-drafter experiment on Terrsvusion's real cold KV
    /// path. Candidate blocks use exact length-one querywise SDPA while each
    /// cold page is materialized once per layer and block. Provisional K/V is
    /// kept outside cold storage until acceptance, enabling exact rollback.
    pub fn generate_reference_speculative_terrsvusion(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
    ) -> Result<ReferenceSpeculativeGeneration> {
        ensure!((1..=512).contains(&max_tokens), "max_tokens fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        let mut cache = self.model.terrsvusion_cache(storage_frame_tokens)?;
        let mut generated = Vec::<u32>::with_capacity(max_tokens);
        let mut prefill_seconds = 0.0_f64;
        let mut decode_seconds = 0.0_f64;
        let mut target_forward_calls = 0_usize;
        let mut speculative_rounds = 0_usize;
        let mut rejected_rounds = 0_usize;
        let mut proposed_tokens = 0_usize;
        let mut accepted_draft_tokens = 0_usize;
        let mut maximum_k_used = 0_usize;
        let mut policy = AdaptiveSpeculationPolicy::new();
        let drafter = DeterministicSuffixDrafter::build(prompt_ids);
        tch::no_grad(|| -> Result<()> {
            let prefill_started = std::time::Instant::now();
            let mut logits = None;
            for chunk in prompt_ids.chunks(chunk_size) {
                logits = Some(self.model.forward_with_cache(chunk, &mut cache)?);
            }
            let mut logits = logits.context("prefill especulativo frío sin logits")?;
            cache.seal_partial_storage_frames()?;
            prefill_seconds = prefill_started.elapsed().as_secs_f64();
            let decode_started = std::time::Instant::now();
            while generated.len() < max_tokens {
                let last = logits.select(1, logits.size()[1] - 1);
                let target_token = last.argmax(-1, false).int64_value(&[0]);
                if self.model.is_eos(target_token) {
                    break;
                }
                if !policy.should_speculate() {
                    policy.record_sequential_bypass();
                    generated.push(target_token as u32);
                    if generated.len() < max_tokens {
                        logits = self.model.forward_with_cache(&[target_token], &mut cache)?;
                        target_forward_calls += 1;
                    }
                    continue;
                }
                let remaining = max_tokens - generated.len();
                let draft = drafter.draft(
                    &generated,
                    target_token,
                    policy.maximum_draft().min(remaining),
                );
                if draft.len() < 2 {
                    generated.push(target_token as u32);
                    if generated.len() < max_tokens {
                        logits = self.model.forward_with_cache(&[target_token], &mut cache)?;
                        target_forward_calls += 1;
                    }
                    continue;
                }
                speculative_rounds += 1;
                proposed_tokens += draft.len();
                maximum_k_used = maximum_k_used.max(draft.len());
                let checkpoint = cache.sequence_tokens();
                cache.begin_multitoken_verification()?;
                let verified_result = self.model.forward_with_cache(&draft, &mut cache);
                cache.end_multitoken_verification();
                let verified = verified_result?;
                target_forward_calls += 1;
                let mut accepted = 0_usize;
                let mut rejection_target = None;
                for (index, candidate) in draft.iter().copied().enumerate() {
                    let expected = if index == 0 {
                        target_token
                    } else {
                        verified
                            .select(1, index as i64 - 1)
                            .argmax(-1, false)
                            .int64_value(&[0])
                    };
                    if candidate != expected {
                        rejection_target = Some(expected);
                        break;
                    }
                    accepted += 1;
                }
                accepted_draft_tokens += accepted;
                generated.extend(draft[..accepted].iter().map(|token| *token as u32));
                policy.record_round(draft.len(), accepted);
                if accepted == draft.len() {
                    logits = verified;
                    continue;
                }
                rejected_rounds += 1;
                cache.rollback_unsealed_to(checkpoint + accepted)?;
                let expected = rejection_target.context("rechazo frío sin token objetivo")?;
                if self.model.is_eos(expected) || generated.len() >= max_tokens {
                    break;
                }
                generated.push(expected as u32);
                if generated.len() < max_tokens {
                    logits = self.model.forward_with_cache(&[expected], &mut cache)?;
                    target_forward_calls += 1;
                }
            }
            decode_seconds = decode_started.elapsed().as_secs_f64();
            Ok(())
        })?;
        let content = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("decodificación falló: {error}"))?;
        let kv_tokens = cache.sequence_tokens();
        let exact_cold_reads = cache.exact_cold_metrics().reads;
        let final_kv_sha256 = cache.logical_state_digest()?;
        Ok(ReferenceSpeculativeGeneration {
            content,
            token_ids: generated.clone(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated.len(),
            prefill_seconds,
            decode_seconds,
            target_forward_calls,
            speculative_rounds,
            rejected_rounds,
            proposed_tokens,
            accepted_draft_tokens,
            acceptance_percent: if proposed_tokens == 0 {
                0.0
            } else {
                accepted_draft_tokens as f64 * 100.0 / proposed_tokens as f64
            },
            maximum_k_used,
            final_k: policy.maximum_draft(),
            adaptive_bypass_tokens: policy.bypassed_tokens,
            final_cooldown_tokens: policy.cooldown_tokens,
            final_kv_sha256,
            kv_tokens,
            hierarchy_events: Some(cache.hierarchy_events),
            hierarchy_widening_rounds: Some(cache.hierarchy_widening_rounds),
            exact_k_tokens_scanned: Some(cache.exact_k_tokens_scanned),
            exact_v_tokens_materialized: Some(cache.exact_v_tokens_materialized),
            exact_cold_reads: Some(exact_cold_reads),
            authority: "terrsvusion_cold_bf16_target_querywise_exact",
            scope: "benchmark_only_reference_drafter_no_server_commit",
        })
    }

    /// Dense BF16 control using the same model, tokenizer, prompt chunks and
    /// greedy decoding as Terrsvusion. This is a benchmark oracle, not a
    /// second serving route.
    /// Optional GPU validation helper. Runs the original dense BF16 model
    /// with the exact same prompt tokens and already-generated autoregressive
    /// continuation, and returns logical BF16 KV bytes, layer by layer.
    /// The reference path is unchanged when opencl-backend is disabled.
    #[cfg(feature = "opencl-backend")]
    pub fn debug_dense_kv_bytes(
        &self,
        prompt_ids: &[i64],
        generated: &[u32],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        ensure!(!prompt_ids.is_empty(), "empty parity prompt");
        ensure!(
            prompt_ids.len() + generated.len() <= 4096,
            "GPU diagnostic readback limited to 4096 tokens"
        );
        let mut cache = DenseKvCache::with_layers(self.model.layers.len());
        tch::no_grad(|| -> Result<()> {
            let _ = self.model.forward_with_cache(prompt_ids, &mut cache)?;
            for &id in generated.iter().take(generated.len().saturating_sub(1)) {
                let _ = self
                    .model
                    .forward_with_cache(&[i64::from(id)], &mut cache)?;
            }
            Ok(())
        })?;
        let mut result = Vec::with_capacity(cache.layers.len());
        for (index, state) in cache.layers.iter().enumerate() {
            let (key, value) = state
                .as_ref()
                .with_context(|| format!("dense KV layer {index} missing"))?;
            let read = |tensor: &Tensor| -> Vec<u8> {
                let contiguous = tensor.contiguous();
                // SAFETY: Exactly numel()*2 bytes of contiguous BF16,
                // copied to owned storage before local Tensor drops.
                unsafe {
                    std::slice::from_raw_parts(
                        contiguous.data_ptr().cast::<u8>(),
                        contiguous.numel() * 2,
                    )
                    .to_vec()
                }
            };
            result.push((read(key), read(value)));
        }
        Ok(result)
    }

    pub fn generate_token_ids_dense(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
    ) -> Result<DenseGeneration> {
        ensure!((1..=512).contains(&max_tokens), "max_tokens fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        ensure!(
            prompt_ids.len() + max_tokens <= self.model.declared_context_limit,
            "prompt y generación exceden el límite declarado de {} tokens",
            self.model.declared_context_limit
        );
        let started = std::time::Instant::now();
        let mut cache = DenseKvCache::with_layers(self.model.layers.len());
        let mut generated = Vec::with_capacity(max_tokens);
        let mut step_logits = Vec::with_capacity(max_tokens);
        let mut prefill_seconds = 0.0_f64;
        let mut decode_seconds = 0.0_f64;
        tch::no_grad(|| -> Result<()> {
            let prefill_started = std::time::Instant::now();
            let mut chunks = prompt_ids.chunks(chunk_size).peekable();
            let mut logits = loop {
                let chunk = chunks.next().context("prompt sin tokens")?;
                let output = self.model.forward_with_cache(chunk, &mut cache)?;
                if chunks.peek().is_none() {
                    break output;
                }
            };
            prefill_seconds = prefill_started.elapsed().as_secs_f64();
            let decode_started = std::time::Instant::now();
            for step in 0..max_tokens {
                let last = logits.select(1, logits.size()[1] - 1);
                step_logits.push(capture_last_logits(&last));
                let token = last.argmax(-1, false).int64_value(&[0]);
                if self.model.is_eos(token) {
                    break;
                }
                generated.push(token as u32);
                if step + 1 < max_tokens {
                    logits = self.model.forward_with_cache(&[token], &mut cache)?;
                }
            }
            decode_seconds = decode_started.elapsed().as_secs_f64();
            Ok(())
        })?;
        let content = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("decodificación falló: {error}"))?;
        let kv_sha256 = cache.state_digest()?;
        Ok(DenseGeneration {
            content,
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated.len(),
            token_ids: generated,
            kv_tokens: cache.sequence_tokens(),
            latency_seconds: started.elapsed().as_secs_f64(),
            prefill_seconds,
            decode_seconds,
            kv_bf16_bytes: cache.physical_bytes(),
            kv_sha256,
            step_logits,
        })
    }

    /// Same prompt token ids, greedy argmax, Terrsvusion KV vs dense BF16 KV.
    /// If greedy tokens diverge, inspect logits and KV hashes at the first step.
    pub fn compare_native_inference(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
    ) -> Result<NativeInferenceParity> {
        let dense = self.generate_token_ids_dense(prompt_ids, max_tokens, 512)?;
        let terrsvusion = self
            .generate_token_ids_instrumented(
                prompt_ids,
                max_tokens,
                512,
                512,
                GenerationInstrumentation {
                    record_step_logits: true,
                    ..GenerationInstrumentation::default()
                },
                None,
            )?
            .0;
        let steps = dense.step_logits.len().min(terrsvusion.step_logits.len());
        let mut max_abs_logit_difference = 0.0_f64;
        let mut first_logit_step_above_1e3 = None;
        for index in 0..steps {
            ensure!(
                dense.step_logits[index].len() == terrsvusion.step_logits[index].len(),
                "logits nativos y Terrsvusion con vocabulario distinto"
            );
            let mut step_linf = 0.0_f64;
            for (left, right) in dense.step_logits[index]
                .iter()
                .zip(&terrsvusion.step_logits[index])
            {
                step_linf = step_linf.max((*left as f64 - *right as f64).abs());
            }
            max_abs_logit_difference = max_abs_logit_difference.max(step_linf);
            if first_logit_step_above_1e3.is_none() && step_linf > 1e-3 {
                first_logit_step_above_1e3 = Some(index);
            }
        }
        let first_token_mismatch = dense
            .token_ids
            .iter()
            .zip(&terrsvusion.token_ids)
            .position(|(left, right)| left != right)
            .or_else(|| {
                (dense.token_ids.len() != terrsvusion.token_ids.len())
                    .then_some(dense.token_ids.len().min(terrsvusion.token_ids.len()))
            });
        let divergence = first_token_mismatch.and_then(|index| {
            let dense_logits = dense.step_logits.get(index)?;
            let terrsvusion_logits = terrsvusion.step_logits.get(index)?;
            Some(first_step_divergence(
                index,
                dense_logits,
                terrsvusion_logits,
            ))
        });
        Ok(NativeInferenceParity {
            prompt_tokens: prompt_ids.len(),
            exact_token_parity: first_token_mismatch.is_none(),
            first_token_mismatch,
            max_abs_logit_difference,
            logit_steps_compared: steps,
            first_logit_step_above_1e3,
            kv_sha256_equal: dense.kv_sha256 == terrsvusion.kv_sha256,
            dense_kv_sha256: dense.kv_sha256,
            terrsvusion_kv_sha256: terrsvusion.kv_sha256,
            dense_tokens: dense.token_ids,
            terrsvusion_tokens: terrsvusion.token_ids,
            dense_text: dense.content,
            terrsvusion_text: terrsvusion.content,
            dense_prefill_seconds: dense.prefill_seconds,
            terrsvusion_prefill_seconds: terrsvusion.prefill_seconds,
            certified_v_tokens_omitted: terrsvusion.certified_v_tokens_omitted,
            terrsvusion_kv_resident_bytes: terrsvusion.kv_resident_bytes,
            dense_kv_bf16_bytes: dense.kv_bf16_bytes,
            terrsvusion_kv_bf16_reference_bytes: terrsvusion.kv_bf16_reference_bytes,
            prompt_authority: "identical_prompt_ids_raw_chat_template_no_default_system",
            divergence,
        })
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| {
                encoding
                    .get_ids()
                    .iter()
                    .map(|&token| token as i64)
                    .collect()
            })
            .map_err(|error| anyhow::anyhow!("tokenización falló: {error}"))
    }
}

impl MiniCpmLayer {
    fn load(weights: &WeightArchive, layer: usize, geometry: &DecoderGeometry) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        let hidden = geometry.hidden;
        let query_out = geometry.query_out();
        let kv_out = geometry.kv_out();
        let intermediate = geometry.intermediate;
        Ok(Self {
            input_norm: weights.require_bf16(&format!("{p}.input_layernorm.weight"), &[hidden])?,
            q: weights.require_bf16(
                &format!("{p}.self_attn.q_proj.weight"),
                &[query_out, hidden],
            )?,
            k: weights.require_bf16(&format!("{p}.self_attn.k_proj.weight"), &[kv_out, hidden])?,
            v: weights.require_bf16(&format!("{p}.self_attn.v_proj.weight"), &[kv_out, hidden])?,
            o: weights.require_bf16(
                &format!("{p}.self_attn.o_proj.weight"),
                &[hidden, query_out],
            )?,
            post_norm: weights
                .require_bf16(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?,
            gate: weights.require_bf16(
                &format!("{p}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?,
            up: weights
                .require_bf16(&format!("{p}.mlp.up_proj.weight"), &[intermediate, hidden])?,
            down: weights.require_bf16(
                &format!("{p}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            rope_theta: geometry.rope_theta,
            rms_eps: geometry.rms_eps,
            llama3_rope: geometry.llama3_rope,
        })
    }

    fn linear(input: &Tensor, weight: &Tensor) -> Tensor {
        input.matmul(&weight.transpose(0, 1))
    }

    fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        let dtype = input.kind();
        let input_f32 = input.to_kind(Kind::Float);
        let variance =
            input_f32
                .pow_tensor_scalar(2)
                .mean_dim(Some([-1_i64].as_slice()), true, Kind::Float);
        (input_f32 * (variance + eps).rsqrt()).to_kind(dtype) * weight
    }

    fn rotate_half(input: &Tensor) -> Tensor {
        let half = input.size()[3] / 2;
        Tensor::cat(
            &[
                &input.narrow(3, half, half).neg(),
                &input.narrow(3, 0, half),
            ],
            3,
        )
    }

    fn llama3_inv_freq(dim: i64, theta: f64) -> Tensor {
        let mut values = Vec::with_capacity((dim / 2) as usize);
        for index in (0..dim).step_by(2) {
            let mut inv_freq = (-(index as f64) * (theta.ln() / dim as f64)).exp();
            let wavelength = 2.0 * std::f64::consts::PI / inv_freq;
            let original_max = 8192.0;
            let low_freq_factor = 1.0;
            let high_freq_factor = 4.0;
            let factor = 8.0;
            let low_freq_wavelen = original_max / low_freq_factor;
            let high_freq_wavelen = original_max / high_freq_factor;
            if wavelength > low_freq_wavelen {
                inv_freq /= factor;
            } else if wavelength < high_freq_wavelen {
            } else {
                let smooth = (original_max / wavelength - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                inv_freq = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq;
            }
            values.push(inv_freq as f32);
        }
        Tensor::from_slice(&values)
    }

    fn rope(&self, input: &Tensor, position_start: usize) -> Tensor {
        let seq = input.size()[2];
        let dim = input.size()[3];
        let positions = Tensor::arange_start(
            position_start as i64,
            position_start as i64 + seq,
            (Kind::Float, Device::Cpu),
        )
        .unsqueeze(1);
        let inv_freq = if self.llama3_rope {
            Self::llama3_inv_freq(dim, self.rope_theta)
        } else {
            let dimensions = Tensor::arange_start_step(0, dim, 2, (Kind::Float, Device::Cpu));
            (-dimensions * (self.rope_theta.ln() / dim as f64)).exp()
        };
        let frequencies = positions.matmul(&inv_freq.unsqueeze(0));
        let embedding = Tensor::cat(&[&frequencies, &frequencies], 1);
        let cos = embedding
            .cos()
            .to_kind(input.kind())
            .unsqueeze(0)
            .unsqueeze(0);
        let sin = embedding
            .sin()
            .to_kind(input.kind())
            .unsqueeze(0)
            .unsqueeze(0);
        input * &cos + Self::rotate_half(input) * &sin
    }

    fn forward_cached<C: AttentionCache>(
        &self,
        input: &Tensor,
        layer_index: usize,
        past_tokens: usize,
        cache: &mut C,
    ) -> Result<Tensor> {
        let residual = input;
        let normalized = Self::rms_norm(input, &self.input_norm, self.rms_eps);
        let batch = normalized.size()[0];
        let seq = normalized.size()[1];
        let q = self.rope(
            &Self::linear(&normalized, &self.q)
                .view([batch, seq, self.query_heads, self.head_dim])
                .transpose(1, 2),
            past_tokens,
        );
        let new_key = self.rope(
            &Self::linear(&normalized, &self.k)
                .view([batch, seq, self.kv_heads, self.head_dim])
                .transpose(1, 2),
            past_tokens,
        );
        let new_value = Self::linear(&normalized, &self.v)
            .view([batch, seq, self.kv_heads, self.head_dim])
            .transpose(1, 2);
        let attention = cache
            .attend(layer_index, &q, &new_key, &new_value)?
            .transpose(1, 2)
            .contiguous()
            .view([batch, seq, self.query_heads * self.head_dim]);
        let hidden = Self::linear(&attention, &self.o) + residual;
        let normalized = Self::rms_norm(&hidden, &self.post_norm, self.rms_eps);
        let mlp =
            Self::linear(&normalized, &self.gate).silu() * Self::linear(&normalized, &self.up);
        Ok(hidden + Self::linear(&mlp, &self.down))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_working_set_is_released_after_exact_cold_barrier() {
        let mut cache = TerrsvusionKvCache::try_new_with_storage_frame_tokens(16).unwrap();
        let query = Tensor::zeros([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let key = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let value = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));

        let _ = cache.attend(0, &query, &key, &value).unwrap();
        cache.finish_step(2);
        assert!(cache.layers[0].prefill_hot.is_some());
        assert_eq!(cache.prefill_working_peak_bytes, 2 * 2 * 2 * 128 * 2);

        cache.seal_partial_storage_frames().unwrap();
        assert!(cache.layers.iter().all(|layer| layer.prefill_hot.is_none()));
        assert_eq!(cache.conversion_metrics().submitted, 1);
        assert_eq!(cache.conversion_metrics().delivered, 1);
        assert_eq!(cache.exact_cold.metadata().len(), 1);
        assert!(matches!(
            cache.layers[0].pages[0].storage,
            PageStorage::Cold { .. }
        ));
    }

    #[test]
    fn persistent_prefix_roundtrip_restores_verified_kv_and_axis() {
        let root = tempfile::tempdir().unwrap();
        let mut cache = TerrsvusionKvCache::try_new_with_storage_frame_tokens(16).unwrap();
        let query = Tensor::zeros([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let key = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let value = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        for layer in 0..24 {
            let _ = cache.attend(layer, &query, &key, &value).unwrap();
        }
        cache.finish_step(2);
        cache.seal_partial_storage_frames().unwrap();
        cache
            .persist_prefix(root.path(), "prefix-digest", "model-digest")
            .unwrap();

        let mut restored =
            TerrsvusionKvCache::restore_prefix(root.path(), "prefix-digest", "model-digest", 16)
                .unwrap()
                .expect("prefijo persistido");
        assert_eq!(restored.sequence_tokens(), 2);
        assert_eq!(restored.parallel_axis.token_len(), 2);
        assert_eq!(restored.prefix_copied_bytes, 0);
        assert!(
            restored
                .prefix_cold
                .as_ref()
                .is_some_and(|store| { store.is_read_only() && store.metadata().len() == 24 })
        );
        assert!(restored.exact_cold.metadata().is_empty());
        assert!(restored.layers.iter().all(|layer| layer.pages.len() == 1));
        restored.hydrate_prefill_hot().unwrap();
        assert!(restored.layers.iter().all(|layer| {
            layer
                .prefill_hot
                .as_ref()
                .is_some_and(|(key, value)| key.size()[2] == 2 && value.size()[2] == 2)
        }));
    }

    #[test]
    fn restored_prefix_attends_suffix_without_copying_or_keeping_history_hot() {
        let root = tempfile::tempdir().unwrap();
        let mut cache = TerrsvusionKvCache::try_new_with_storage_frame_tokens(16).unwrap();
        let query = Tensor::zeros([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let key = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let value = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        for layer in 0..24 {
            let _ = cache.attend(layer, &query, &key, &value).unwrap();
        }
        cache.finish_step(2);
        cache.seal_partial_storage_frames().unwrap();
        cache
            .persist_prefix(root.path(), "prefix-digest", "model-digest")
            .unwrap();
        let mut restored =
            TerrsvusionKvCache::restore_prefix(root.path(), "prefix-digest", "model-digest", 16)
                .unwrap()
                .expect("prefijo persistido");
        assert_eq!(restored.prefix_copied_bytes, 0);
        let suffix_query = Tensor::ones([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let suffix_key = Tensor::ones([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let suffix_value = Tensor::ones([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        for layer in 0..24 {
            let _ = restored
                .attend(layer, &suffix_query, &suffix_key, &suffix_value)
                .unwrap();
        }
        assert!(restored.layers.iter().all(|layer| {
            layer
                .prefill_hot
                .as_ref()
                .is_some_and(|(key, value)| key.size()[2] == 2 && value.size()[2] == 2)
        }));
        assert!(restored.prefill_working_peak_bytes > 0);
        assert!(restored.prefill_working_peak_bytes < 24 * 2 * (2 + 2) * 128 * 2 * 2);
    }

    #[test]
    fn querywise_multitoken_attention_is_byte_exact_with_sequential_decode() {
        let prefix_query = Tensor::arange(3 * 16 * 128, (Kind::Float, Device::Cpu))
            .view([1, 16, 3, 128])
            .to_kind(Kind::BFloat16);
        let prefix_key = Tensor::arange(3 * 2 * 128, (Kind::Float, Device::Cpu))
            .view([1, 2, 3, 128])
            .to_kind(Kind::BFloat16);
        let prefix_value = (&prefix_key * 0.5).to_kind(Kind::BFloat16);
        let candidate_query = Tensor::arange(2 * 16 * 128, (Kind::Float, Device::Cpu))
            .view([1, 16, 2, 128])
            .to_kind(Kind::BFloat16);
        let candidate_key = Tensor::arange(2 * 2 * 128, (Kind::Float, Device::Cpu))
            .view([1, 2, 2, 128])
            .to_kind(Kind::BFloat16);
        let candidate_value = (&candidate_key * 0.25).to_kind(Kind::BFloat16);
        let mut sequential = DenseKvCache::new();
        let mut querywise = DenseKvCache::new();
        let _ = sequential
            .attend(0, &prefix_query, &prefix_key, &prefix_value)
            .unwrap();
        sequential.finish_step(3);
        let _ = querywise
            .attend(0, &prefix_query, &prefix_key, &prefix_value)
            .unwrap();
        querywise.finish_step(3);
        let mut sequential_outputs = Vec::new();
        for token in 0..2 {
            sequential_outputs.push(
                sequential
                    .attend(
                        0,
                        &candidate_query.narrow(2, token, 1),
                        &candidate_key.narrow(2, token, 1),
                        &candidate_value.narrow(2, token, 1),
                    )
                    .unwrap(),
            );
            sequential.finish_step(1);
        }
        querywise.querywise_attention = true;
        let querywise_output = querywise
            .attend(0, &candidate_query, &candidate_key, &candidate_value)
            .unwrap();
        querywise.finish_step(2);
        let sequential_refs = sequential_outputs.iter().collect::<Vec<_>>();
        let sequential_output = Tensor::cat(&sequential_refs, 2);
        assert!(sequential_output.equal(&querywise_output));
        let (sequential_key, sequential_value) = sequential.layers[0].as_ref().unwrap();
        let (querywise_key, querywise_value) = querywise.layers[0].as_ref().unwrap();
        assert!(sequential_key.equal(querywise_key));
        assert!(sequential_value.equal(querywise_value));
    }

    #[test]
    fn reference_drafter_uses_longest_generated_suffix_and_bounds_the_proposal() {
        let prompt = [9_i64, 20, 30, 40, 50, 20, 30, 60, 70];
        assert_eq!(reference_draft(&prompt, &[20, 30], 2), vec![60, 70]);
        assert_eq!(
            reference_draft(&prompt, &[9, 20, 30], 8),
            vec![40, 50, 20, 30, 60, 70]
        );
        assert!(reference_draft(&prompt, &[], 4).is_empty());
        assert!(reference_draft(&prompt, &[999], 4).is_empty());
    }

    #[test]
    fn adaptive_speculation_policy_expands_on_success_and_cools_down_on_rejection() {
        let mut policy = AdaptiveSpeculationPolicy::new();
        assert!(policy.should_speculate());
        assert_eq!(policy.maximum_draft(), 2);
        policy.record_round(2, 2);
        assert_eq!(policy.maximum_draft(), 4);
        policy.record_round(4, 1);
        assert_eq!(policy.maximum_draft(), 2);
        assert_eq!(policy.cooldown_tokens, 6);
        for _ in 0..6 {
            assert!(!policy.should_speculate());
            policy.record_sequential_bypass();
        }
        assert!(policy.should_speculate());
        assert_eq!(policy.bypassed_tokens, 6);
    }

    #[test]
    fn suffix_automaton_drafts_only_unique_non_terminal_continuations() {
        let unique_then_branch = DeterministicSuffixDrafter::build(&[1, 2, 3, 1, 2, 3, 4]);
        assert_eq!(unique_then_branch.draft(&[1], 2, 8), vec![2, 3]);

        let immediate_branch = DeterministicSuffixDrafter::build(&[9, 2, 3, 8, 2, 4]);
        assert_eq!(immediate_branch.draft(&[], 2, 8), vec![2]);

        let terminal = DeterministicSuffixDrafter::build(&[1, 2, 3]);
        assert_eq!(terminal.draft(&[1], 2, 8), vec![2, 3]);
        assert_eq!(terminal.draft(&[99], 100, 8), vec![100]);
    }

    #[test]
    fn cold_multitoken_rollback_truncates_only_unsealed_tails() {
        let mut cache = TerrsvusionKvCache::try_new_with_storage_frame_tokens(16).unwrap();
        for layer in &mut cache.layers {
            layer.tail = Some((
                Tensor::zeros([1, 2, 5, 128], (Kind::BFloat16, Device::Cpu)),
                Tensor::ones([1, 2, 5, 128], (Kind::BFloat16, Device::Cpu)),
            ));
        }
        cache.sequence_tokens = 5;
        cache.begin_multitoken_verification().unwrap();
        cache.end_multitoken_verification();
        cache.rollback_unsealed_to(3).unwrap();
        assert_eq!(cache.sequence_tokens(), 3);
        assert!(cache.layers.iter().all(|layer| {
            layer
                .tail
                .as_ref()
                .is_some_and(|(key, value)| key.size()[2] == 3 && value.size()[2] == 3)
        }));
    }

    #[test]
    fn regional_mass_bound_certifies_a_far_below_region() {
        let selected_scores = vec![vec![20.0]; 16];
        let low = BTreeMap::from([(0, [(-20.0 + 100.0_f64.ln()); 16])]);
        let value_max = BTreeMap::from([(0, [1.0, 1.0])]);
        let bounds = omitted_region_bounds(&selected_scores, &low, &value_max, &BTreeSet::new());
        assert!(certified(&bounds, 1e-12));

        let high = BTreeMap::from([(0, [(20.0 + 100.0_f64.ln()); 16])]);
        let ambiguous =
            omitted_region_bounds(&selected_scores, &high, &value_max, &BTreeSet::new());
        assert!(!certified(&ambiguous, 1e-3));
    }

    #[test]
    fn incremental_region_certificate_matches_canonical_bound() {
        let state_uppers = BTreeMap::from([(10, [-3.0; 16]), (20, [-7.0; 16]), (30, [-11.0; 16])]);
        let state_value_max =
            BTreeMap::from([(10, [1.0, 1.5]), (20, [2.0, 2.5]), (30, [3.0, 3.5])]);
        let mut incremental = IncrementalRegionCertificate::<16, 2>::new(
            &state_uppers,
            &state_value_max,
            None,
            [0.0; 2],
        )
        .unwrap();
        let exact = vec![vec![-8.0, -9.0]; 16];
        incremental
            .materialize(20, &exact, &state_uppers, &state_value_max)
            .unwrap();
        let canonical = omitted_region_bounds(
            &exact,
            &state_uppers,
            &state_value_max,
            &BTreeSet::from([20]),
        );
        for (left, right) in incremental.bounds().iter().zip(canonical) {
            assert!((left - right).abs() < 1e-12, "{left} != {right}");
        }
    }

    #[test]
    fn incremental_region_certificate_accounts_for_hot_tail_values() {
        let state_uppers = BTreeMap::from([(10, [-10.0; 16])]);
        let state_value_max = BTreeMap::from([(10, [1.0, 1.0])]);
        let tail_scores = vec![vec![0.0]; 16];
        let without_tail_value = IncrementalRegionCertificate::<16, 2>::new(
            &state_uppers,
            &state_value_max,
            Some(&tail_scores),
            [0.0; 2],
        )
        .unwrap();
        let with_tail_value = IncrementalRegionCertificate::<16, 2>::new(
            &state_uppers,
            &state_value_max,
            Some(&tail_scores),
            [100.0; 2],
        )
        .unwrap();
        assert!(with_tail_value.bounds()[0] > without_tail_value.bounds()[0] * 50.0);
    }

    #[test]
    fn incremental_region_certificate_supports_llama_32q_8kv_geometry() {
        let state_uppers = BTreeMap::from([(10, [(-3.0_f64); 32]), (20, [(-7.0_f64); 32])]);
        let state_value_max = BTreeMap::from([(10, [1.0_f64; 8]), (20, [2.0_f64; 8])]);
        let mut certificate = IncrementalRegionCertificate::<32, 8>::new(
            &state_uppers,
            &state_value_max,
            None,
            [0.0; 8],
        )
        .unwrap();
        let exact = vec![vec![-8.0, -9.0]; 32];
        certificate
            .materialize(20, &exact, &state_uppers, &state_value_max)
            .unwrap();
        assert_eq!(certificate.bounds().len(), 32);
        assert!(certificate.bounds().iter().all(|bound| bound.is_finite()));
    }

    #[test]
    fn exact_oracle_reports_the_true_first_omission_threshold() {
        let scores = vec![vec![0.0, -10.0]; 16];
        let value_max = vec![2.0; 16];
        let (threshold, token, limiting_head) =
            exact_oracle_min_tolerance_candidate(&scores, &value_max);
        let expected = 4.0 * (-10.0_f64).exp() / (1.0 + (-10.0_f64).exp());
        assert!((threshold - expected).abs() < 1e-12);
        assert_eq!(token, 1);
        assert!(limiting_head < 16);
        assert_eq!(
            exact_oracle_omissible_tokens(&scores, &value_max, threshold * 0.5).unwrap(),
            0
        );
        assert_eq!(
            exact_oracle_omissible_tokens(&scores, &value_max, threshold * 1.01).unwrap(),
            1
        );
    }

    #[test]
    fn bf16_interval_endpoints_round_outward() {
        let values = [0.12345_f32, -0.12345_f32];
        let lower = TerrsvusionKvCache::outward_bf16_bounds(&values, true);
        let upper = TerrsvusionKvCache::outward_bf16_bounds(&values, false);
        for index in 0..values.len() {
            assert!(half::bf16::from_bits(lower[index]).to_f32() <= values[index]);
            assert!(half::bf16::from_bits(upper[index]).to_f32() >= values[index]);
        }
    }

    #[test]
    fn chat_prompt_can_seed_an_empty_think_block() {
        let messages = [TorchChatMessage {
            role: "user".to_owned(),
            content: "Hola".to_owned(),
        }];
        let silent = TorchModelRuntime::render_messages_with_thinking(&messages, false).unwrap();
        assert!(silent.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let thinking = TorchModelRuntime::render_messages_with_thinking(&messages, true).unwrap();
        assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!thinking.contains("</think>"));
    }

    #[test]
    fn llama_32q8kv_certified_v_read_from_physical_coldstore_is_partial_and_bounded() {
        // Test the actual CPU libtorch SDPA and original physical ColdStore;
        // no Llama checkpoint/GPU is needed. The 32Q/8KV head geometry,
        // BF16 precision, and 128-dimensional RoPE Q/K match Llama 3.1 8B.
        let mut cache = TerrsvusionKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        let tokens = 16usize;
        let heads = 8usize;
        let dim = 128usize;
        let mut key_data = Vec::with_capacity(heads * tokens * dim);
        let mut value_data = Vec::with_capacity(heads * tokens * dim);
        for _head in 0..heads {
            for token in 0..tokens {
                for d in 0..dim {
                    let dominant = token == 15;
                    key_data.push(if d == 0 {
                        if dominant { 20.0f32 } else { -20.0 }
                    } else {
                        0.0
                    });
                    value_data.push(if d == 0 {
                        if dominant { 1.0f32 } else { -1.0 }
                    } else {
                        0.0
                    });
                }
            }
        }
        let k = Tensor::from_slice(&key_data)
            .view([1, 8, 16, 128])
            .to_kind(Kind::BFloat16)
            .contiguous();
        let v = Tensor::from_slice(&value_data)
            .view([1, 8, 16, 128])
            .to_kind(Kind::BFloat16)
            .contiguous();
        for token in 0..16 {
            let mut signal = vec![0.0f32; 8 * 128];
            for head in 0..8 {
                signal[head * 128] = if token == 15 { 20.0 } else { -20.0 };
            }
            cache.parallel_axis.observe(&signal).unwrap();
        }
        let certificate = cache.page_attention_certificate(&k, &v, 0).unwrap();
        assert_eq!(certificate.value_abs_max_by_kv_head.len(), 8);
        assert_eq!(certificate.value_abs_max_by_kv_head, [1.0f32; 8]);
        let key_bits = TerrsvusionKvCache::bf16_bits(&k).unwrap();
        let value_bits = TerrsvusionKvCache::bf16_bits(&v).unwrap();
        let bf16_k = TerrsvusionKvCache::u16_bytes(&key_bits);
        let bf16_v = TerrsvusionKvCache::u16_bytes(&value_bits);
        let committed = cache
            .exact_cold
            .append_page(
                PageGeometry {
                    layer: 0,
                    token_start: 0,
                    tokens: 16,
                    heads: 8,
                    head_dim: 128,
                    tile: 16,
                },
                &bf16_k,
                &bf16_v,
                false,
            )
            .unwrap();
        assert_eq!(committed.page_id, 0);
        cache.sequence_tokens = 16;
        let page = TerrsvusionPage {
            token_start: 0,
            tokens: 16,
            certificate,
            storage: PageStorage::Cold {
                exact_id: 0,
                immutable: false,
            },
        };
        let qdata = (0..32 * 128)
            .map(|i| if i % 128 == 0 { 20.0f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let q = Tensor::from_slice(&qdata)
            .view([1, 32, 1, 128])
            .to_kind(Kind::BFloat16);
        let tail_data = (0..8 * 128)
            .map(|i| if i % 128 == 0 { 20.0f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let vtdata = (0..8 * 128)
            .map(|i| if i % 128 == 0 { 1.0f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let tailk = Tensor::from_slice(&tail_data)
            .view([1, 8, 1, 128])
            .to_kind(Kind::BFloat16);
        let tailv = Tensor::from_slice(&vtdata)
            .view([1, 8, 1, 128])
            .to_kind(Kind::BFloat16);
        let full = TerrsvusionKvCache::scaled_attention(
            &q,
            &[k.shallow_clone(), tailk.shallow_clone()],
            &[v.shallow_clone(), tailv.shallow_clone()],
            16,
        );
        let output = cache
            .llama_certified_value_attention(
                0,
                &q,
                &[page.clone()],
                Some(&(tailk.shallow_clone(), tailv.shallow_clone())),
                1e-3,
            )
            .unwrap()
            .expect("32Q8KV exact QK bound must certify omission");
        let count = cache.exact_k_tokens_scanned;
        let selected = cache.exact_v_tokens_materialized;
        assert_eq!(count, 17);
        assert_eq!(selected, 2);
        assert_eq!(count - selected, 15);
        let result = output.to_kind(Kind::Float).contiguous();
        let reference = full.to_kind(Kind::Float).contiguous();
        let mut actual = vec![0.0f32; 32 * 128];
        let mut expected = vec![0.0f32; 32 * 128];
        result.copy_data(&mut actual, 32 * 128);
        reference.copy_data(&mut expected, 32 * 128);
        let linf = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from((*a - *b).abs()))
            .fold(0.0f64, f64::max);
        assert!(linf <= 1e-3, "actual BF16 SDPA error exceeded local bound");
        assert!(cache.exact_mass_max_bound <= 5e-4);
        let stats = cache.exact_cold.stats();
        assert_eq!(stats.partial_value_reads, 1);
        assert!(
            stats.partial_value_bytes_avoided > 0,
            "ColdStore skipped no physical BF16 V bytes"
        );
        cache.exact_cold.audit_mapped().unwrap();

        // Legacy persisted prefix has only MiniCPM-style two-head
        // metadata. It MUST fail closed rather than fabricate 8KV bounds.
        let mut legacy = page.clone();
        legacy.certificate.value_abs_max_by_kv_head.clear();
        assert!(
            cache
                .llama_certified_value_attention(
                    0,
                    &q,
                    &[legacy],
                    Some(&(tailk.shallow_clone(), tailv.shallow_clone())),
                    1e-3
                )
                .unwrap()
                .is_none()
        );
        assert!(
            cache
                .llama_certified_value_attention(
                    0,
                    &q,
                    &[page.clone()],
                    Some(&(tailk.shallow_clone(), tailv.shallow_clone())),
                    0.0
                )
                .unwrap()
                .is_none()
        );
        let zero = Tensor::zeros([1, 32, 1, 128], (Kind::BFloat16, Device::Cpu));
        assert!(
            cache
                .llama_certified_value_attention(0, &zero, &[page], Some(&(tailk, tailv)), 1e-3)
                .unwrap()
                .is_none(),
            "diffuse equal scores must not certify Llama KV omission"
        );
        eprintln!(
            "LLAMA_32Q8KV_CERTIFIED_PHYSICAL_V_PASS K_scored={} V_materialized={} V_omitted={} cold_partial_value_reads={} cold_partial_bytes_avoided={} BF16_SDPA_Linf={} bound={} epsilon=0.001 old_prefix_fail_closed=true exact_mode_fail_closed=true",
            count,
            selected,
            count - selected,
            stats.partial_value_reads,
            stats.partial_value_bytes_avoided,
            linf,
            cache.exact_mass_max_bound
        );
    }

    #[test]
    fn omission_policy_never_reuses_exact_bf16_prefix_cache() {
        assert_eq!(safe_prefix_tokens_for_omission(7143, None), 7143);
        assert_eq!(safe_prefix_tokens_for_omission(7143, Some(0.001)), 0);
        assert_eq!(safe_prefix_tokens_for_omission(111632, Some(0.01)), 0);
        assert_eq!(safe_prefix_tokens_for_omission(0, None), 0);
    }

    #[test]
    fn llama_missing_eight_kv_bounds_deserializes_as_fail_closed() {
        let old = serde_json::json!({
            "regions":[],
            "value_abs_max":[1.0,2.0]
        });
        let parsed: PageAttentionCertificate = serde_json::from_value(old).unwrap();
        assert!(parsed.value_abs_max_by_kv_head.is_empty());
        let fresh = serde_json::json!({
            "regions":[],
            "value_abs_max":[1.0,2.0],
            "value_abs_max_by_kv_head":[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
        });
        let parsed: PageAttentionCertificate = serde_json::from_value(fresh).unwrap();
        assert_eq!(parsed.value_abs_max_by_kv_head.len(), 8);
    }

    #[test]
    fn llama3_chat_template_uses_begin_of_text_headers() {
        let prompt = TorchModelRuntime::render_llama3_messages(
            &[
                TorchChatMessage {
                    role: "system".to_owned(),
                    content: "Sé breve.".to_owned(),
                },
                TorchChatMessage {
                    role: "user".to_owned(),
                    content: "Qué es un coche?".to_owned(),
                },
            ],
            true,
        )
        .unwrap();
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
        assert!(!prompt.contains("<|im_start|>"));
    }

    #[test]
    fn chat_template_string_already_contains_bos_so_encode_must_not_add_another() {
        let prompt = TorchModelRuntime::render_messages(&[TorchChatMessage {
            role: "user".to_owned(),
            content: "Hola".to_owned(),
        }])
        .unwrap();
        assert!(prompt.starts_with("<s>"));
    }

    #[test]
    fn shouting_user_text_becomes_sentence_case() {
        assert_eq!(
            TorchModelRuntime::sentence_case_if_shouting("QUE ES UN COCHE"),
            "Que es un coche"
        );
        assert_eq!(
            TorchModelRuntime::sentence_case_if_shouting("Qué es un coche"),
            "Qué es un coche"
        );
        assert_eq!(TorchModelRuntime::sentence_case_if_shouting("NASA"), "NASA");
        let prepared = TorchModelRuntime::prepare_chat_messages(&[TorchChatMessage {
            role: "user".to_owned(),
            content: "QUE ES UN COCHE".to_owned(),
        }]);
        assert_eq!(prepared[0].role, "system");
        assert_eq!(
            prepared[1].content,
            "Explica de forma concreta qué es un coche. Da una definición; no repitas la pregunta."
        );
        assert_eq!(
            TorchModelRuntime::normalize_user_chat_text("como te llamas?"),
            "¿Cómo te llamas?"
        );
        assert_eq!(
            TorchModelRuntime::normalize_user_chat_text("que es el amor?"),
            "Explica de forma concreta qué es el amor. Da una definición; no repitas la pregunta."
        );
        assert_eq!(
            TorchModelRuntime::normalize_user_chat_text("What is a car?"),
            "What is a car?"
        );
        assert_eq!(
            TorchModelRuntime::normalize_user_chat_text("hola"),
            "Saluda en una frase corta y espera la siguiente pregunta."
        );
        assert_eq!(
            TorchModelRuntime::normalize_user_chat_text("a que hora llego el tren?"),
            "¿A qué hora llego el tren?"
        );
        assert_eq!(
            TorchModelRuntime::visible_completion(
                "razonamiento interno</think>\n\nEl tren llegó a las 6:40.",
                false
            ),
            "El tren llegó a las 6:40."
        );
    }

    #[test]
    fn contiguous_token_selection_matches_sparse_fallback() {
        let tensor = Tensor::arange_start(0, 6, (Kind::Int64, Device::Cpu)).view([1, 1, 6, 1]);
        let full = select_token_indices(&tensor, &[0, 1, 2, 3, 4, 5]);
        let contiguous = select_token_indices(&tensor, &[1, 2, 3]);
        let sparse = select_token_indices(&tensor, &[0, 2, 5]);
        let mut contiguous_values = vec![0_i64; 3];
        let mut sparse_values = vec![0_i64; 3];
        contiguous.copy_data(&mut contiguous_values, 3);
        sparse.copy_data(&mut sparse_values, 3);
        assert_eq!(contiguous_values, vec![1, 2, 3]);
        assert_eq!(sparse_values, vec![0, 2, 5]);
        assert_eq!(full.data_ptr(), tensor.data_ptr());
        assert_eq!(
            contiguous.data_ptr() as usize,
            tensor.data_ptr() as usize + std::mem::size_of::<i64>()
        );
        assert_ne!(sparse.data_ptr(), tensor.data_ptr());
    }
}
