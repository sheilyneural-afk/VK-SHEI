//! LibTorch model runtime and generation output types.
//!
//! TorchModelRuntime owns the loaded model, tokenizer, and prefix cache.
//! DecoderLayer performs the forward pass. Generation structs carry
//! per-token and aggregate statistics for every inference run.
//!
//! This is where LibTorch (tch) calls concentrate; it is the FFI boundary
//! for the torch-backend feature.

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};
use tokenizers::Tokenizer;

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
use crate::shadow_metrics::{ShadowDiagnosticReport, ShadowEventId, ShadowMetrics};

use super::kv_cache::{
    AttentionCache, DecoderGeometry, DecoderLayer, DecoderModel, DenseKvCache, LayerOutputTrace,
    MslimKvCache,
};
use super::weight_io::{WeightArchive, model_weight_file};

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
    pub prefill_forward_calls: usize,
    pub seal_seconds: f64,
    pub decode_seconds: f64,
    pub drain_seconds: f64,
    pub kv_physical_bytes: usize,
    pub kv_resident_bytes: usize,
    pub kv_known_attention_scratch_peak_bytes: usize,
    pub kv_prefill_working_peak_bytes: usize,
    /// Resident verified committed KV (bytes now / peak), pages reused from
    /// it instead of being re-read from the ColdStore, and calls that fell
    /// back to the cold re-read path because of the RAM budget.
    #[serde(default)]
    pub kv_committed_resident_bytes: usize,
    #[serde(default)]
    pub kv_committed_resident_peak_bytes: usize,
    #[serde(default)]
    pub kv_committed_resident_pages_reused: usize,
    #[serde(default)]
    pub kv_committed_resident_fallbacks: usize,
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
    pub attention_selected_qk_elements: u128,
    pub attention_selected_pv_elements: u128,
    pub attention_v_tokens_avoided: usize,
    pub dense_sdpa_calls: usize,
    pub dense_sdpa_scheduled_qk_elements: u128,
    pub dense_sdpa_scheduled_pv_elements: u128,
    pub dense_sdpa_causal_qk_elements: u128,
    pub dense_sdpa_causal_pv_elements: u128,
    pub dense_sdpa_nanos: u64,
    pub hierarchy_events: usize,
    pub hierarchy_states_total: usize,
    pub hierarchy_states_initial: usize,
    pub hierarchy_states_loaded: usize,
    pub hierarchy_widening_rounds: usize,
    /// K tokens physically left unread by the regional proof. Only the
    /// configured-model path applies the proof; the parity-first Llama path scores the
    /// complete K history, so this stays 0 there by design.
    pub hierarchy_k_tokens_avoided: usize,
    /// Attention events where the regional K proof was evaluated (configured-model
    /// applied, Llama measured).
    #[serde(default)]
    pub hierarchy_k_events: usize,
    /// K tokens the regional proof would let skip at the event's tolerance.
    /// configured-model: equal to `hierarchy_k_tokens_avoided`. Llama: measured on the
    /// exact scores without skipping anything (renormalized-omission bound).
    #[serde(default)]
    pub hierarchy_k_tokens_skippable: usize,
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
    pub exact_attention_verified_events: usize,
    pub exact_attention_equal_events: usize,
    pub exact_attention_candidate_v_tokens: usize,
    pub exact_attention_rejected_v_tokens: usize,
    pub rounding_cell_checked_events: usize,
    pub rounding_cell_certified_events: usize,
    /// Output components checked by the BF16 rounding-cell diagnostic.
    #[serde(default)]
    pub rounding_cell_components_checked: usize,
    /// Components whose unrounded FP32 interval stays in one BF16 cell.
    #[serde(default)]
    pub rounding_cell_components_certified: usize,
    pub applied_mass_tolerance: Option<f64>,
    #[serde(skip_serializing)]
    pub step_logits: Vec<Vec<f32>>,
    #[serde(skip_serializing)]
    #[doc(hidden)]
    pub layer_output_traces: Vec<LayerOutputTrace>,
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
    pub omission_accepted: bool,
    pub exact_logit_parity: bool,
    pub within_requested_logit_tolerance: bool,
    pub exact_token_parity: bool,
    pub max_abs_logit_difference: f64,
    pub logit_steps_compared: usize,
    pub layer_output_events_compared: usize,
    pub first_divergent_layer: Option<usize>,
    pub first_divergent_sequence_end: Option<usize>,
    pub first_divergent_layer_linf: f64,
    pub max_layer_output_linf: f64,
    pub max_divergent_layer: Option<usize>,
    pub max_divergent_sequence_end: Option<usize>,
    pub certified_v_tokens_omitted: usize,
    pub hierarchy_k_tokens_avoided: usize,
    #[serde(default)]
    pub hierarchy_k_tokens_skippable: usize,
    pub reference_token_ids: Vec<u32>,
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
    pub mslim_argmax: u32,
    pub dense_logit_at_dense_argmax: f32,
    pub mslim_logit_at_dense_argmax: f32,
    pub dense_logit_at_mslim_argmax: f32,
    pub mslim_logit_at_mslim_argmax: f32,
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
    pub mslim_kv_sha256: String,
    pub dense_tokens: Vec<u32>,
    pub mslim_tokens: Vec<u32>,
    pub dense_text: String,
    pub mslim_text: String,
    pub dense_prefill_seconds: f64,
    pub mslim_prefill_seconds: f64,
    pub certified_v_tokens_omitted: usize,
    pub mslim_kv_resident_bytes: usize,
    pub dense_kv_bf16_bytes: usize,
    pub mslim_kv_bf16_reference_bytes: usize,
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

fn first_step_divergence(step: usize, dense: &[f32], mslim: &[f32]) -> NativeStepDivergence {
    let dense_argmax = argmax_logit(dense);
    let mslim_argmax = argmax_logit(mslim);
    let mut linf = 0.0_f64;
    for (left, right) in dense.iter().zip(mslim) {
        linf = linf.max((*left as f64 - *right as f64).abs());
    }
    NativeStepDivergence {
        step,
        dense_argmax,
        mslim_argmax,
        dense_logit_at_dense_argmax: dense[dense_argmax as usize],
        mslim_logit_at_dense_argmax: mslim[dense_argmax as usize],
        dense_logit_at_mslim_argmax: dense[mslim_argmax as usize],
        mslim_logit_at_mslim_argmax: mslim[mslim_argmax as usize],
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
    pub reference_commit_safe: bool,
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
    model: DecoderModel,
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
    verify_exact_attention_outputs: bool,
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
        spec.ensure_cpu_runtime_compatible()?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_identity = Self::runtime_identity_from_shards(model_dir, &spec, &tokenizer_path)?;
        let model = DecoderModel::load(model_dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow::anyhow!("tokenizer {}: {error}", tokenizer_path.display()))?;
        Ok(Self {
            model,
            tokenizer,
            prefix_cache_dir,
            model_identity,
        })
    }

    /// Build another movable executor whose LibTorch tensor handles reference
    /// the exact same immutable checkpoint storage. KV caches remain local to
    /// each generation call and are never shared by this operation.
    pub fn shallow_clone_shared_weights(&self) -> Result<Self> {
        let clone = Self {
            model: DecoderModel {
                embedding: self.model.embedding.shallow_clone(),
                layers: self
                    .model
                    .layers
                    .iter()
                    .map(DecoderLayer::shallow_clone_shared_weights)
                    .collect(),
                final_norm: self.model.final_norm.shallow_clone(),
                lm_head: self.model.lm_head.shallow_clone(),
                rms_eps: self.model.rms_eps,
                eos: self.model.eos.clone(),
                chat: self.model.chat,
                query_heads: self.model.query_heads,
                kv_heads: self.model.kv_heads,
                head_dim: self.model.head_dim,
                streaming_prefill: self.model.streaming_prefill,
                declared_context_limit: self.model.declared_context_limit,
            },
            tokenizer: self.tokenizer.clone(),
            prefix_cache_dir: self.prefix_cache_dir.clone(),
            model_identity: self.model_identity.clone(),
        };
        ensure!(
            self.shares_all_weight_storage_with(&clone),
            "el clon LibTorch no comparte todos los pesos del checkpoint"
        );
        Ok(clone)
    }

    pub fn weight_storage_count(&self) -> usize {
        self.weight_storage_ids().len()
    }

    pub fn shares_all_weight_storage_with(&self, other: &Self) -> bool {
        self.weight_storage_ids() == other.weight_storage_ids()
    }

    fn weight_storage_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(3 + self.model.layers.len() * 9);
        ids.push(self.model.embedding.data_ptr() as usize);
        for layer in &self.model.layers {
            layer.weight_storage_ids(&mut ids);
        }
        ids.push(self.model.final_norm.data_ptr() as usize);
        ids.push(self.model.lm_head.data_ptr() as usize);
        ids
    }

    fn runtime_identity_from_shards(
        model_dir: &Path,
        spec: &ModelSpec,
        tokenizer_path: &Path,
    ) -> Result<String> {
        let mut digest = Sha256::new();
        digest.update(b"mslim-runtime-id-shards-v1");
        for relative in ["config.json", "tokenizer.json"] {
            let path = if relative == "tokenizer.json" {
                tokenizer_path.to_path_buf()
            } else {
                model_dir.join(relative)
            };
            let bytes = std::fs::read(&path)
                .with_context(|| format!("abrir identidad runtime {}", path.display()))?;
            digest.update((bytes.len() as u64).to_le_bytes());
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
        digest.update(b"mslim-prefix-v1");
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
        // configured-model's chat template: enable_thinking=false seeds an empty
        // think block so decode starts at the visible answer, not a CoT.
        if enable_thinking {
            prompt.push_str("<think>\n");
        } else {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        Ok(prompt)
    }

    fn render_template_messages(
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

    fn chat_prompt(&self, messages: &[TorchChatMessage], _enable_thinking: bool) -> Result<String> {
        match self.model.chat {
            ChatTemplateFamily::TemplateDelimited => Self::render_template_messages(messages, true),
        }
    }

    fn chat_prefix(&self, messages: &[TorchChatMessage]) -> Result<String> {
        match self.model.chat {
            ChatTemplateFamily::TemplateDelimited => {
                Self::render_template_messages(messages, false)
            }
        }
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

    const DEFAULT_TEMPLATE_CHAT_SYSTEM: &str =
        "Responde de forma directa y breve, en el idioma del usuario.";

    fn default_system_prompt(&self) -> &'static str {
        match self.model.chat {
            ChatTemplateFamily::TemplateDelimited => Self::DEFAULT_TEMPLATE_CHAT_SYSTEM,
        }
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
        // Input text is authoritative: never rewrite messages before tokenization.
        prepared.extend(messages.iter().cloned());
        prepared
    }

    #[cfg(test)]
    fn prepare_chat_messages(messages: &[TorchChatMessage]) -> Vec<TorchChatMessage> {
        Self::prepare_chat_messages_with_system(messages, Self::DEFAULT_TEMPLATE_CHAT_SYSTEM)
    }

    fn prepare_runtime_chat_messages(
        &self,
        messages: &[TorchChatMessage],
    ) -> Vec<TorchChatMessage> {
        Self::prepare_chat_messages_with_system(messages, self.default_system_prompt())
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
        let messages = self.prepare_runtime_chat_messages(messages);
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

    /// Official configured-model chat template + dense BF16 KV. No M-SLIM cache,
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
    /// decoding as `generate_messages`. Isolates model errors from M-SLIM.
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
                    verify_exact_attention_outputs: true,
                    ..GenerationInstrumentation::default()
                },
                None,
            )?
            .0;
        ensure!(
            !canonical.step_logits.is_empty()
                && canonical.step_logits.len() == applied.step_logits.len(),
            "comparación de omisión necesita el mismo número positivo de pasos de logits"
        );
        let steps = canonical.step_logits.len();
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
                ensure!(
                    left.is_finite() && right.is_finite(),
                    "logit no finito en comparación de omisión aplicada"
                );
                max_abs_logit_difference =
                    max_abs_logit_difference.max((*left as f64 - *right as f64).abs());
            }
        }
        ensure!(
            canonical.layer_output_traces.len() == applied.layer_output_traces.len(),
            "trazas de capa canónica y aplicada con longitudes distintas"
        );
        let mut first_divergent_layer = None;
        let mut first_divergent_sequence_end = None;
        let mut first_divergent_layer_linf = 0.0_f64;
        let mut max_layer_output_linf = 0.0_f64;
        let mut max_divergent_layer = None;
        let mut max_divergent_sequence_end = None;
        for (reference_trace, applied_trace) in canonical
            .layer_output_traces
            .iter()
            .zip(&applied.layer_output_traces)
        {
            ensure!(
                reference_trace.layer_index == applied_trace.layer_index
                    && reference_trace.sequence_end == applied_trace.sequence_end
                    && reference_trace.last_token.len() == applied_trace.last_token.len(),
                "eventos de capa canónico y aplicado incompatibles"
            );
            let linf = reference_trace
                .last_token
                .iter()
                .zip(&applied_trace.last_token)
                .map(|(left, right)| (*left as f64 - *right as f64).abs())
                .fold(0.0_f64, f64::max);
            if linf > max_layer_output_linf {
                max_layer_output_linf = linf;
                max_divergent_layer = Some(reference_trace.layer_index);
                max_divergent_sequence_end = Some(reference_trace.sequence_end);
            }
            if linf > 0.0 && first_divergent_layer.is_none() {
                first_divergent_layer = Some(reference_trace.layer_index);
                first_divergent_sequence_end = Some(reference_trace.sequence_end);
                first_divergent_layer_linf = linf;
            }
        }
        let exact_logit_parity = max_abs_logit_difference == 0.0;
        let within_requested_logit_tolerance = max_abs_logit_difference <= tolerance;
        Ok(AppliedOmissionReport {
            tolerance,
            omission_accepted: exact_logit_parity && applied.certified_v_tokens_omitted > 0,
            exact_logit_parity,
            within_requested_logit_tolerance,
            exact_token_parity: canonical.token_ids == applied.token_ids,
            max_abs_logit_difference,
            logit_steps_compared: steps,
            layer_output_events_compared: canonical.layer_output_traces.len(),
            first_divergent_layer,
            first_divergent_sequence_end,
            first_divergent_layer_linf,
            max_layer_output_linf,
            max_divergent_layer,
            max_divergent_sequence_end,
            certified_v_tokens_omitted: applied.certified_v_tokens_omitted,
            hierarchy_k_tokens_avoided: applied.hierarchy_k_tokens_avoided,
            hierarchy_k_tokens_skippable: applied.hierarchy_k_tokens_skippable,
            reference_token_ids: canonical.token_ids.clone(),
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
            verify_exact_attention_outputs,
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
            (Some(root), Some(key)) => MslimKvCache::restore_prefix_geometry(
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
        let mut cache = restored.unwrap_or(self.model.mslim_cache(storage_frame_tokens)?);
        cache.exact_oracle_enabled = exact_oracle_enabled || shadow_metrics_enabled;
        cache.shadow_metrics = shadow_metrics_enabled.then(ShadowMetrics::default);
        cache.applied_mass_tolerance = applied_mass_tolerance;
        if applied_mass_tolerance.is_some()
            && cache.query_heads == 32
            && cache.kv_heads == 8
            && cache.head_dim == 128
        {
            // Per-token regions are intentionally diagnostic-only: they make
            // the regional Llama proof tight enough to measure real QK/PV
            // pruning, but their metadata and launch cost are not acceptable
            // as the canonical 131K representation.
            cache
                .parallel_axis
                .set_open_extent_safety_limit_before_observation(Some(1))?;
        }
        cache.record_step_logits = record_step_logits;
        cache.verify_exact_attention_outputs = verify_exact_attention_outputs;
        let mut generated = Vec::with_capacity(max_tokens);
        let mut prefill_seconds = 0.0_f64;
        let mut prefill_forward_calls = 0usize;
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
                    prefill_forward_calls += 1;
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
                    prefill_forward_calls += 1;
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
                    prefill_forward_calls += 1;
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
                    prefill_forward_calls += 1;
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
                prefill_forward_calls,
                seal_seconds,
                decode_seconds,
                drain_seconds,
                kv_physical_bytes: physical,
                kv_resident_bytes: resident,
                kv_known_attention_scratch_peak_bytes: cache.known_attention_scratch_peak_bytes,
                kv_prefill_working_peak_bytes: cache.prefill_working_peak_bytes,
                kv_committed_resident_bytes: cache.resident_kv_bytes,
                kv_committed_resident_peak_bytes: cache.resident_kv_peak_bytes,
                kv_committed_resident_pages_reused: cache.resident_kv_pages_reused,
                kv_committed_resident_fallbacks: cache.resident_kv_fallbacks,
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
                attention_selected_qk_elements: cache.attention_selected_qk_elements,
                attention_selected_pv_elements: cache.attention_selected_pv_elements,
                attention_v_tokens_avoided: cache.attention_v_tokens_avoided,
                dense_sdpa_calls: cache.dense_sdpa_calls,
                dense_sdpa_scheduled_qk_elements: cache.dense_sdpa_scheduled_qk_elements,
                dense_sdpa_scheduled_pv_elements: cache.dense_sdpa_scheduled_pv_elements,
                dense_sdpa_causal_qk_elements: cache.dense_sdpa_causal_qk_elements,
                dense_sdpa_causal_pv_elements: cache.dense_sdpa_causal_pv_elements,
                dense_sdpa_nanos: cache.dense_sdpa_nanos,
                hierarchy_events: cache.hierarchy_events,
                hierarchy_states_total: cache.hierarchy_states_total,
                hierarchy_states_initial: cache.hierarchy_states_initial,
                hierarchy_states_loaded: cache.hierarchy_states_loaded,
                hierarchy_widening_rounds: cache.hierarchy_widening_rounds,
                hierarchy_k_tokens_avoided: cache.hierarchy_k_tokens_avoided,
                hierarchy_k_events: cache.hierarchy_k_events,
                hierarchy_k_tokens_skippable: cache.hierarchy_k_tokens_skippable,
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
                certified_v_tokens_omitted: cache.attention_v_tokens_avoided,
                exact_attention_verified_events: cache.exact_attention_verified_events,
                exact_attention_equal_events: cache.exact_attention_equal_events,
                exact_attention_candidate_v_tokens: cache.exact_attention_candidate_v_tokens,
                exact_attention_rejected_v_tokens: cache.exact_attention_rejected_v_tokens,
                rounding_cell_checked_events: cache.rounding_cell_checked_events,
                rounding_cell_certified_events: cache.rounding_cell_certified_events,
                rounding_cell_components_checked: cache.rounding_cell_components_checked,
                rounding_cell_components_certified: cache.rounding_cell_components_certified,
                applied_mass_tolerance: cache.applied_mass_tolerance,
                step_logits: std::mem::take(&mut cache.step_logits),
                layer_output_traces: std::mem::take(&mut cache.layer_output_traces),
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
                reference_commit_safe: argmax_parity
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

    /// The same reference-drafter experiment on M-SLIM's real cold KV
    /// path. Candidate blocks use exact length-one querywise SDPA while each
    /// cold page is materialized once per layer and block. Provisional K/V is
    /// kept outside cold storage until acceptance, enabling exact rollback.
    pub fn generate_reference_speculative_mslim(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
        chunk_size: usize,
        storage_frame_tokens: usize,
    ) -> Result<ReferenceSpeculativeGeneration> {
        ensure!((1..=512).contains(&max_tokens), "max_tokens fuera de rango");
        ensure!(!prompt_ids.is_empty(), "prompt sin tokens");
        ensure!(chunk_size > 0, "chunk_size debe ser positivo");
        let mut cache = self.model.mslim_cache(storage_frame_tokens)?;
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
            authority: "mslim_cold_bf16_target_querywise_exact",
            scope: "benchmark_only_reference_drafter_no_server_commit",
        })
    }

    /// Dense BF16 control using the same model, tokenizer, prompt chunks and
    /// greedy decoding as M-SLIM. This is a benchmark oracle, not a
    /// second serving route.
    /// Optional GPU validation helper. Runs the original dense BF16 model
    /// with the exact same prompt tokens and already-generated autoregressive
    /// continuation, and returns logical BF16 KV bytes, layer by layer.
    /// The reference path is unchanged when accelerator-backend is disabled.
    #[cfg(feature = "accelerator-backend")]
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

    /// Same prompt token ids, greedy argmax, M-SLIM KV vs dense BF16 KV.
    /// If greedy tokens diverge, inspect logits and KV hashes at the first step.
    pub fn compare_native_inference(
        &self,
        prompt_ids: &[i64],
        max_tokens: usize,
    ) -> Result<NativeInferenceParity> {
        let dense = self.generate_token_ids_dense(prompt_ids, max_tokens, 512)?;
        let mslim = self
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
        let steps = dense.step_logits.len().min(mslim.step_logits.len());
        let mut max_abs_logit_difference = 0.0_f64;
        let mut first_logit_step_above_1e3 = None;
        for index in 0..steps {
            ensure!(
                dense.step_logits[index].len() == mslim.step_logits[index].len(),
                "logits nativos y M-SLIM con vocabulario distinto"
            );
            let mut step_linf = 0.0_f64;
            for (left, right) in dense.step_logits[index]
                .iter()
                .zip(&mslim.step_logits[index])
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
            .zip(&mslim.token_ids)
            .position(|(left, right)| left != right)
            .or_else(|| {
                (dense.token_ids.len() != mslim.token_ids.len())
                    .then_some(dense.token_ids.len().min(mslim.token_ids.len()))
            });
        let divergence = first_token_mismatch.and_then(|index| {
            let dense_logits = dense.step_logits.get(index)?;
            let mslim_logits = mslim.step_logits.get(index)?;
            Some(first_step_divergence(index, dense_logits, mslim_logits))
        });
        Ok(NativeInferenceParity {
            prompt_tokens: prompt_ids.len(),
            exact_token_parity: first_token_mismatch.is_none(),
            first_token_mismatch,
            max_abs_logit_difference,
            logit_steps_compared: steps,
            first_logit_step_above_1e3,
            kv_sha256_equal: dense.kv_sha256 == mslim.kv_sha256,
            dense_kv_sha256: dense.kv_sha256,
            mslim_kv_sha256: mslim.kv_sha256,
            dense_tokens: dense.token_ids,
            mslim_tokens: mslim.token_ids,
            dense_text: dense.content,
            mslim_text: mslim.content,
            dense_prefill_seconds: dense.prefill_seconds,
            mslim_prefill_seconds: mslim.prefill_seconds,
            certified_v_tokens_omitted: mslim.certified_v_tokens_omitted,
            mslim_kv_resident_bytes: mslim.kv_resident_bytes,
            dense_kv_bf16_bytes: dense.kv_bf16_bytes,
            mslim_kv_bf16_reference_bytes: mslim.kv_bf16_reference_bytes,
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

impl DecoderLayer {
    fn shallow_clone_shared_weights(&self) -> Self {
        Self {
            input_norm: self.input_norm.shallow_clone(),
            q: self.q.shallow_clone(),
            k: self.k.shallow_clone(),
            v: self.v.shallow_clone(),
            o: self.o.shallow_clone(),
            post_norm: self.post_norm.shallow_clone(),
            gate: self.gate.shallow_clone(),
            up: self.up.shallow_clone(),
            down: self.down.shallow_clone(),
            query_heads: self.query_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            rms_eps: self.rms_eps,
            llama3_rope: self.llama3_rope,
        }
    }

    fn weight_storage_ids(&self, ids: &mut Vec<usize>) {
        ids.extend([
            self.input_norm.data_ptr() as usize,
            self.q.data_ptr() as usize,
            self.k.data_ptr() as usize,
            self.v.data_ptr() as usize,
            self.o.data_ptr() as usize,
            self.post_norm.data_ptr() as usize,
            self.gate.data_ptr() as usize,
            self.up.data_ptr() as usize,
            self.down.data_ptr() as usize,
        ]);
    }

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
        let output = hidden + Self::linear(&mlp, &self.down);
        cache.record_layer_output(layer_index, past_tokens, &output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resident bytes of the VMA that contains `address`, from smaps.
    #[cfg(target_os = "linux")]
    fn mapping_rss_bytes(address: usize) -> u64 {
        let smaps = fs::read_to_string("/proc/self/smaps").unwrap();
        let mut inside = false;
        for line in smaps.lines() {
            let first = line.split_whitespace().next().unwrap_or("");
            if let Some((low, high)) = first.split_once('-')
                && let (Ok(low), Ok(high)) = (
                    usize::from_str_radix(low, 16),
                    usize::from_str_radix(high, 16),
                )
            {
                inside = (low..high).contains(&address);
                continue;
            }
            if inside && let Some(rest) = line.strip_prefix("Rss:") {
                let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap();
                return kib * 1024;
            }
        }
        panic!("no se encontró el mapeo en smaps");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn weight_load_releases_mapped_pages_after_copy() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let rows = 4096usize;
        let cols = 2048usize; // 16 MiB of BF16
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|index| half::bf16::from_f32((index % 251) as f32).to_le_bytes())
            .collect();
        let header = format!(
            "{{\"w\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{cols}],\"data_offsets\":[0,{}]}}}}",
            data.len()
        );
        let mut header = header.into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        fs::write(&path, &bytes).unwrap();

        let weights = WeightFile::open(&path).unwrap();
        let tensor = weights
            .require_bf16("w", &[rows as i64, cols as i64])
            .unwrap();
        // The copy is intact even though the source pages were dropped.
        let mut copied = vec![0_u8; data.len()];
        tensor
            .flatten(0, -1)
            .view_dtype(Kind::Int16)
            .copy_data_u8(&mut copied, data.len() / 2);
        assert!(copied == data);
        // Only the header page (and at most one boundary page) may remain.
        let resident = mapping_rss_bytes(weights.mapping.as_ptr() as usize + header.len() + 8);
        assert!(
            resident <= 64 * 1024,
            "el mmap sigue residente tras copiar: {resident} bytes"
        );
        // The mapping stays valid for later reads of the same tensor.
        let again = weights.tensor("w").unwrap();
        assert!(again.equal(&tensor));
    }

    fn resident_test_page(
        cache: &mut MslimKvCache,
        token_start: usize,
        tokens: usize,
        seed: f32,
    ) -> (Tensor, Tensor) {
        let elements = 8 * tokens * 128;
        let key = Tensor::from_slice(
            &(0..elements)
                .map(|index| seed + (index % 97) as f32 * 0.125)
                .collect::<Vec<_>>(),
        )
        .view([1, 8, tokens as i64, 128])
        .to_kind(Kind::BFloat16);
        let value = (&key * -0.5).to_kind(Kind::BFloat16);
        let key_bits = MslimKvCache::bf16_bits(&key).unwrap();
        let value_bits = MslimKvCache::bf16_bits(&value).unwrap();
        let key_bytes = MslimKvCache::u16_bytes(&key_bits);
        let value_bytes = MslimKvCache::u16_bytes(&value_bits);
        let committed = cache
            .exact_cold
            .append_page(
                PageGeometry {
                    layer: 0,
                    token_start: token_start as u64,
                    tokens: tokens as u32,
                    heads: 8,
                    head_dim: 128,
                    tile: 16,
                },
                key_bytes,
                value_bytes,
                false,
            )
            .unwrap();
        cache.layers[0].pages.push(MslimPage {
            token_start,
            tokens,
            certificate: PageAttentionCertificate {
                regions: Vec::new(),
                value_abs_max: [0.0; 2],
                value_abs_max_by_kv_head: Vec::new(),
                value_abs_max_by_kv_head_dim: Vec::new(),
            },
            storage: PageStorage::Cold {
                exact_id: committed.page_id,
                immutable: false,
            },
        });
        (key, value)
    }

    #[test]
    fn logical_kv_digest_covers_all_llama_heads_and_matches_dense_across_cold_and_tail() {
        let mut cache = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        let mut dense = DenseKvCache::with_layers(1);
        let (cold_key, cold_value) = resident_test_page(&mut cache, 0, 16, 1.0);
        cache.sequence_tokens = 16;
        dense.sequence_tokens = 16;
        dense.layers[0] = Some((cold_key.shallow_clone(), cold_value.shallow_clone()));
        assert_eq!(
            cache.logical_state_digest().unwrap(),
            dense.state_digest().unwrap()
        );

        let tail_key = Tensor::arange(8 * 128, (Kind::Float, Device::Cpu))
            .view([1, 8, 1, 128])
            .to_kind(Kind::BFloat16);
        let tail_value = (&tail_key + 5.0).to_kind(Kind::BFloat16);
        cache.layers[0].tail = Some((tail_key.shallow_clone(), tail_value.shallow_clone()));
        cache.sequence_tokens = 17;
        dense.layers[0] = Some((
            Tensor::cat(&[&cold_key, &tail_key], 2),
            Tensor::cat(&[&cold_value, &tail_value], 2),
        ));
        dense.sequence_tokens = 17;
        let baseline = cache.logical_state_digest().unwrap();
        assert_eq!(baseline, dense.state_digest().unwrap());

        // A changed BF16 value in the eighth KV head MUST affect the digest.
        let modified_key = tail_key.copy();
        let _ = modified_key.narrow(1, 7, 1).fill_(20.0);
        cache.layers[0].tail = Some((modified_key, tail_value.shallow_clone()));
        assert_ne!(baseline, cache.logical_state_digest().unwrap());
        let modified_value = tail_value.copy();
        let _ = modified_value.narrow(1, 7, 1).fill_(30.0);
        cache.layers[0].tail = Some((tail_key, modified_value));
        assert_ne!(baseline, cache.logical_state_digest().unwrap());
    }

    #[test]
    fn resident_committed_kv_reads_each_cold_page_once_and_matches_cat() {
        let mut cache = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        cache.resident_kv_budget_bytes = 64 << 20;
        let first = resident_test_page(&mut cache, 0, 16, 1.0);
        let second = resident_test_page(&mut cache, 16, 16, -3.0);
        let (key, value) = cache.assemble_committed_layer(0).unwrap();
        let reads_after_first = cache.exact_cold.stats().reads;
        assert_eq!(reads_after_first, 4, "K y V de dos páginas, una vez");
        assert!(key.equal(&Tensor::cat(&[&first.0, &second.0], 2)));
        assert!(value.equal(&Tensor::cat(&[&first.1, &second.1], 2)));
        // Same history again: served from RAM, no cold read, identical bytes.
        let (again_key, again_value) = cache.assemble_committed_layer(0).unwrap();
        assert_eq!(cache.exact_cold.stats().reads, reads_after_first);
        assert!(again_key.equal(&key) && again_value.equal(&value));
        // A new page: only that page is read; earlier views stay intact.
        let third = resident_test_page(&mut cache, 32, 16, 7.0);
        let (grown_key, grown_value) = cache.assemble_committed_layer(0).unwrap();
        assert_eq!(cache.exact_cold.stats().reads, reads_after_first + 2);
        assert!(grown_key.equal(&Tensor::cat(&[&first.0, &second.0, &third.0], 2)));
        assert!(grown_value.equal(&Tensor::cat(&[&first.1, &second.1, &third.1], 2)));
        assert!(key.equal(&Tensor::cat(&[&first.0, &second.0], 2)));
        assert_eq!(cache.resident_kv_pages_reused, 2 + 2);
        assert!(cache.resident_kv_bytes > 0);
        assert!(cache.resident_kv_bytes <= cache.resident_kv_budget_bytes);
        assert_eq!(cache.resident_kv_fallbacks, 0);
    }

    #[test]
    fn resident_committed_kv_respects_budget_and_falls_back_to_cold_reads() {
        let mut cache = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        // One 16-token page fits (64 KiB), two do not.
        cache.resident_kv_budget_bytes = 2 * 8 * 16 * 128 * 2;
        let first = resident_test_page(&mut cache, 0, 16, 2.0);
        let _ = cache.assemble_committed_layer(0).unwrap();
        assert_eq!(cache.resident_kv_bytes, cache.resident_kv_budget_bytes);
        let second = resident_test_page(&mut cache, 16, 16, 5.0);
        let (key, value) = cache.assemble_committed_layer(0).unwrap();
        assert!(key.equal(&Tensor::cat(&[&first.0, &second.0], 2)));
        assert!(value.equal(&Tensor::cat(&[&first.1, &second.1], 2)));
        assert_eq!(cache.resident_kv_fallbacks, 1);
        assert_eq!(
            cache.resident_kv_bytes, 0,
            "el residente se libera al pasarse"
        );
        let reads = cache.exact_cold.stats().reads;
        let _ = cache.assemble_committed_layer(0).unwrap();
        assert_eq!(
            cache.exact_cold.stats().reads,
            reads + 4,
            "ruta fría original"
        );
        // Budget 0 disables residency entirely (original behaviour).
        let mut disabled = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        disabled.resident_kv_budget_bytes = 0;
        let _ = resident_test_page(&mut disabled, 0, 16, 1.0);
        let _ = disabled.assemble_committed_layer(0).unwrap();
        let _ = disabled.assemble_committed_layer(0).unwrap();
        assert_eq!(disabled.exact_cold.stats().reads, 4);
        assert_eq!(disabled.resident_kv_fallbacks, 0);
    }

    #[test]
    fn dense_sdpa_accounting_covers_remainder_chunks_and_all_llama_layers() {
        let mut cache = MslimKvCache::try_new_with_geometry(512, 32, 8, 32, 128).unwrap();
        let mut sequence_tokens = 0usize;
        for query_tokens in [512usize, 512, 512, 512, 512, 512, 512, 416] {
            let attended_tokens = sequence_tokens + query_tokens;
            for _ in 0..32 {
                cache.record_dense_sdpa(query_tokens, attended_tokens, sequence_tokens, 1);
            }
            sequence_tokens = attended_tokens;
        }
        assert_eq!(sequence_tokens, 4000);
        assert_eq!(cache.dense_sdpa_calls, 8 * 32);
        assert_eq!(cache.dense_sdpa_scheduled_qk_elements, 9_220_128_768);
        assert_eq!(cache.dense_sdpa_scheduled_pv_elements, 1_180_176_482_304);
        assert_eq!(cache.dense_sdpa_causal_qk_elements, 8_194_048_000);
        assert_eq!(cache.dense_sdpa_causal_pv_elements, 1_048_838_144_000);
        assert_eq!(cache.dense_sdpa_nanos, 8 * 32);
    }

    #[test]
    fn prefill_working_set_is_released_after_exact_cold_barrier() {
        let mut cache = MslimKvCache::try_new_with_storage_frame_tokens(16).unwrap();
        let query = Tensor::zeros([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let key = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let value = Tensor::zeros([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));

        let _ = cache.attend(0, &query, &key, &value).unwrap();
        cache.finish_step(2);
        assert_eq!(cache.dense_sdpa_calls, 1);
        assert_eq!(cache.dense_sdpa_scheduled_qk_elements, 2 * 2 * 16);
        assert_eq!(cache.dense_sdpa_scheduled_pv_elements, 2 * 2 * 16 * 128);
        assert_eq!(cache.dense_sdpa_causal_qk_elements, (1 + 2) * 16);
        assert_eq!(cache.dense_sdpa_causal_pv_elements, (1 + 2) * 16 * 128);
        assert!(cache.dense_sdpa_nanos > 0);
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
        let mut cache = MslimKvCache::try_new_with_storage_frame_tokens(16).unwrap();
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
        assert_eq!(cache.prefix_copied_bytes, 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let suffix_inode = fs::metadata(cache.exact_cold.path()).unwrap().ino();
            let prefix_inode =
                fs::metadata(root.path().join("prefix-digest").join("pages-exact.kv"))
                    .unwrap()
                    .ino();
            assert_ne!(suffix_inode, prefix_inode);
        }
        assert!(
            cache
                .prefix_cold
                .as_ref()
                .is_some_and(|store| store.is_read_only() && store.metadata().len() == 24)
        );
        assert!(cache.exact_cold.metadata().is_empty());
        assert!(cache.layers.iter().all(|layer| {
            layer.pages.iter().all(|page| {
                matches!(
                    page.storage,
                    PageStorage::Cold {
                        immutable: true,
                        ..
                    }
                )
            })
        }));
        let suffix_query = Tensor::ones([1, 16, 2, 128], (Kind::BFloat16, Device::Cpu));
        let suffix_key = Tensor::ones([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        let suffix_value = Tensor::ones([1, 2, 2, 128], (Kind::BFloat16, Device::Cpu));
        for layer in 0..24 {
            let _ = cache
                .attend(layer, &suffix_query, &suffix_key, &suffix_value)
                .unwrap();
        }
        cache.finish_step(2);
        cache.seal_partial_storage_frames().unwrap();
        assert_eq!(cache.exact_cold.metadata().len(), 24);
        assert_eq!(
            cache
                .prefix_cold
                .as_ref()
                .map(|store| store.metadata().len()),
            Some(24)
        );

        let mut restored =
            MslimKvCache::restore_prefix(root.path(), "prefix-digest", "model-digest", 16)
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
        let mut cache = MslimKvCache::try_new_with_storage_frame_tokens(16).unwrap();
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
            MslimKvCache::restore_prefix(root.path(), "prefix-digest", "model-digest", 16)
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
    fn llama_batched_causal_attention_matches_querywise_full_attention() {
        let prefix_tokens = 5_i64;
        let query_tokens = 4_i64;
        let prefix_query = (Tensor::arange(32 * prefix_tokens * 128, (Kind::Float, Device::Cpu))
            * 0.0017)
            .sin()
            .view([1, 32, prefix_tokens, 128])
            .to_kind(Kind::BFloat16);
        let prefix_key = (Tensor::arange(8 * prefix_tokens * 128, (Kind::Float, Device::Cpu))
            * 0.0023)
            .cos()
            .view([1, 8, prefix_tokens, 128])
            .to_kind(Kind::BFloat16);
        let prefix_value = (Tensor::arange(8 * prefix_tokens * 128, (Kind::Float, Device::Cpu))
            * 0.0031)
            .sin()
            .view([1, 8, prefix_tokens, 128])
            .to_kind(Kind::BFloat16);
        let query = (Tensor::arange(32 * query_tokens * 128, (Kind::Float, Device::Cpu)) * 0.0041)
            .cos()
            .view([1, 32, query_tokens, 128])
            .to_kind(Kind::BFloat16);
        let key = (Tensor::arange(8 * query_tokens * 128, (Kind::Float, Device::Cpu)) * 0.0053)
            .sin()
            .view([1, 8, query_tokens, 128])
            .to_kind(Kind::BFloat16);
        let value = (Tensor::arange(8 * query_tokens * 128, (Kind::Float, Device::Cpu)) * 0.0067)
            .cos()
            .view([1, 8, query_tokens, 128])
            .to_kind(Kind::BFloat16);

        let mut batched = DenseKvCache::with_layers(1);
        let mut querywise = DenseKvCache::with_layers(1);
        let _ = batched
            .attend(0, &prefix_query, &prefix_key, &prefix_value)
            .unwrap();
        batched.finish_step(prefix_tokens as usize);
        let _ = querywise
            .attend(0, &prefix_query, &prefix_key, &prefix_value)
            .unwrap();
        querywise.finish_step(prefix_tokens as usize);
        querywise.querywise_attention = true;

        let batched_output = batched.attend(0, &query, &key, &value).unwrap();
        let querywise_output = querywise.attend(0, &query, &key, &value).unwrap();
        let linf = (&batched_output - &querywise_output)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            batched_output.equal(&querywise_output),
            "SDPA causal batida y querywise completa divergen: L∞={linf}"
        );
    }

    #[test]
    fn shallow_runtime_clone_shares_every_weight_storage() {
        let weight = || Tensor::zeros([1], (Kind::BFloat16, Device::Cpu));
        let authority = TorchModelRuntime {
            model: DecoderModel {
                embedding: weight(),
                layers: vec![DecoderLayer {
                    input_norm: weight(),
                    q: weight(),
                    k: weight(),
                    v: weight(),
                    o: weight(),
                    post_norm: weight(),
                    gate: weight(),
                    up: weight(),
                    down: weight(),
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: 1,
                    rope_theta: 1.0,
                    rms_eps: 1e-5,
                    llama3_rope: false,
                }],
                final_norm: weight(),
                lm_head: weight(),
                rms_eps: 1e-5,
                eos: vec![1],
                chat: ChatTemplateFamily::TemplateDelimited,
                query_heads: 1,
                kv_heads: 1,
                head_dim: 1,
                streaming_prefill: false,
                declared_context_limit: 16,
            },
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            prefix_cache_dir: None,
            model_identity: "test-authority".to_owned(),
        };
        let clone = authority.shallow_clone_shared_weights().unwrap();
        assert_eq!(authority.weight_storage_count(), 12);
        assert!(authority.shares_all_weight_storage_with(&clone));
        assert_eq!(authority.model_identity, clone.model_identity);
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
        let mut cache = MslimKvCache::try_new_with_storage_frame_tokens(16).unwrap();
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
    fn llama_regional_upper_bounds_cover_libtorch_fp32_matmul() {
        let mut cache = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
        let key_values = (0..8 * 128)
            .map(|index| {
                let signed = (index % 29) as f32 - 14.0;
                signed * 0.03125
            })
            .collect::<Vec<_>>();
        let query_values = (0..32 * 128)
            .map(|index| {
                let signed = (index % 31) as f32 - 15.0;
                signed * 0.046875
            })
            .collect::<Vec<_>>();
        let key = Tensor::from_slice(&key_values)
            .view([1, 8, 1, 128])
            .to_kind(Kind::BFloat16);
        let value = Tensor::ones([1, 8, 1, 128], (Kind::BFloat16, Device::Cpu));
        cache.observe_parallel_axis(&key).unwrap();
        let page = MslimPage {
            token_start: 0,
            tokens: 1,
            certificate: cache.page_attention_certificate(&key, &value, 0).unwrap(),
            storage: PageStorage::Pending {
                sequence: 0,
                key: key.shallow_clone(),
                value,
            },
        };
        let query = Tensor::from_slice(&query_values)
            .view([1, 32, 1, 128])
            .to_kind(Kind::BFloat16)
            .to_kind(Kind::Float);
        let mut query_float = vec![0.0_f32; 32 * 128];
        query.copy_data(&mut query_float, 32 * 128);
        let scale = 1.0 / 128.0_f64.sqrt();
        let uppers = state_score_uppers_generic::<32, 8>(&[page], &query_float, scale);
        let repeated = key
            .repeat_interleave_self_int(4, 1, None)
            .to_kind(Kind::Float);
        let scores = (query.matmul(&repeated.transpose(-2, -1)) / 128.0_f64.sqrt()).contiguous();
        let mut exact = vec![0.0_f32; 32];
        scores.copy_data(&mut exact, 32);
        let upper = uppers.get(&0).unwrap();
        for head in 0..32 {
            assert!(
                f64::from(exact[head]) <= upper[head],
                "score LibTorch fuera de cota en cabeza {head}: {} > {}",
                exact[head],
                upper[head]
            );
        }
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
        let lower = MslimKvCache::outward_bf16_bounds(&values, true);
        let upper = MslimKvCache::outward_bf16_bounds(&values, false);
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
        let mut cache = MslimKvCache::try_new_with_geometry(16, 1, 8, 32, 128).unwrap();
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
        let key_bits = MslimKvCache::bf16_bits(&k).unwrap();
        let value_bits = MslimKvCache::bf16_bits(&v).unwrap();
        let bf16_k = MslimKvCache::u16_bytes(&key_bits);
        let bf16_v = MslimKvCache::u16_bytes(&value_bits);
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
                bf16_k,
                bf16_v,
                false,
            )
            .unwrap();
        assert_eq!(committed.page_id, 0);
        cache.sequence_tokens = 16;
        let page = MslimPage {
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
        let full = MslimKvCache::scaled_attention(
            &q,
            &[k.shallow_clone(), tailk.shallow_clone()],
            &[v.shallow_clone(), tailv.shallow_clone()],
            16,
        );
        let output = cache
            .llama_certified_value_attention(
                0,
                &q,
                std::slice::from_ref(&page),
                Some(&(tailk.shallow_clone(), tailv.shallow_clone())),
                None,
                None,
                16,
                1e-3,
            )
            .unwrap()
            .expect("32Q8KV exact QK bound must certify omission");
        let count = cache.exact_k_tokens_scanned;
        let selected = cache.exact_v_tokens_materialized;
        assert_eq!(
            count, 17,
            "la ruta de paridad debe conservar el denominador QK completo"
        );
        assert!(
            selected <= 2,
            "la selección de contribución materializó demasiada V: {selected}"
        );
        assert!(count >= selected);
        assert!(
            output.equal(&full),
            "la V omitida y rellenada con cero debe conservar la salida BF16 exacta"
        );
        assert_eq!(cache.rounding_cell_checked_events, 1);
        assert_eq!(cache.rounding_cell_certified_events, 1);
        assert_eq!(cache.rounding_cell_components_checked, 32 * 128);
        assert_eq!(cache.rounding_cell_components_certified, 32 * 128);
        // The regional proof is measured on the Llama path: the 15 tokens of
        // the low-score region could be skipped, but the parity-first path
        // still scored all 17 K tokens, so nothing is reported as avoided.
        assert_eq!(page.certificate.regions.len(), 2);
        assert_eq!(cache.hierarchy_k_events, 1);
        assert_eq!(cache.hierarchy_k_tokens_skippable, 15);
        assert_eq!(cache.hierarchy_k_tokens_avoided, 0);
        assert_eq!(cache.attention_v_tokens_avoided, 17 - selected);
        assert_eq!(
            cache.attention_selected_pv_elements,
            17 * 32 * 128,
            "omitir lecturas V no debe declararse como PV disperso: SDPA recibió la geometría completa"
        );
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

        // Legacy persisted prefix has only configured-model-style two-head
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
                    None,
                    None,
                    16,
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
                    std::slice::from_ref(&page),
                    Some(&(tailk.shallow_clone(), tailv.shallow_clone())),
                    None,
                    None,
                    16,
                    0.0
                )
                .unwrap()
                .is_none()
        );
        let zero = Tensor::zeros([1, 32, 1, 128], (Kind::BFloat16, Device::Cpu));
        assert!(
            cache
                .llama_certified_value_attention(
                    0,
                    &zero,
                    &[page],
                    Some(&(tailk, tailv)),
                    None,
                    None,
                    16,
                    1e-3,
                )
                .unwrap()
                .is_none(),
            "diffuse equal scores must not certify Llama KV omission"
        );
        eprintln!(
            "LLAMA_32Q8KV_CERTIFIED_PHYSICAL_V_PASS K_scored={} V_materialized={} V_omitted={} cold_partial_value_reads={} cold_partial_bytes_avoided={} BF16_SDPA_Linf={} bound={} epsilon=0.001 old_prefix_fail_closed=true exact_mode_fail_closed=true",
            count,
            selected,
            cache.attention_v_tokens_avoided,
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
        let prompt = TorchModelRuntime::render_template_messages(
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
    fn chat_preparation_preserves_user_and_assistant_messages_exactly() {
        let exact_cases = [
            "Como dije antes, el código es TEST-CODE.",
            "que es lo que pasó ayer?",
            "QUE ES UN COCHE",
            "a que hora llego el tren?",
            "hola",
            "  Código: a = {x: 3}  ",
            "Texto de referencia (única fuente): TEST-DOCUMENT.",
        ];
        for content in exact_cases {
            let originals = vec![
                TorchChatMessage {
                    role: "assistant".to_owned(),
                    content: "He leído el texto anterior".to_owned(),
                },
                TorchChatMessage {
                    role: "user".to_owned(),
                    content: content.to_owned(),
                },
            ];
            let prepared = TorchModelRuntime::prepare_chat_messages(&originals);
            assert_eq!(prepared.len(), 3);
            assert_eq!(prepared[0].role, "system");
            for (before, after) in originals.iter().zip(prepared.iter().skip(1)) {
                assert_eq!(before.role, after.role);
                assert_eq!(before.content, after.content);
            }
        }
        let explicit = vec![
            TorchChatMessage {
                role: "system".to_owned(),
                content: "No cambies nada".to_owned(),
            },
            TorchChatMessage {
                role: "user".to_owned(),
                content: "¿Qué es TEST-CODE?".to_owned(),
            },
        ];
        let prepared = TorchModelRuntime::prepare_chat_messages(&explicit);
        assert_eq!(prepared.len(), explicit.len());
        assert!(
            prepared
                .iter()
                .zip(&explicit)
                .all(|(a, b)| a.role == b.role && a.content == b.content)
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
