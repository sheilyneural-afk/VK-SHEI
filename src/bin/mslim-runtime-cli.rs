#![recursion_limit = "256"]
//! Experimental full-model BF16 inference with Rust/OpenCL on a physical GPU.
//! Local runtime command line entry point for a verified model bundle.

use anyhow::{Context, Result, bail, ensure};
use mslim::gpu_region_store::GpuRegionConfig;
use mslim::opencl_llm::{OpenClLlm, WeightMode};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

fn raw_user_chat(user: &str, thinking: bool) -> String {
    let mut text = format!("<s><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    if thinking {
        text.push_str("<think>\n");
    } else {
        text.push_str("<think>\n\n</think>\n\n");
    }
    text
}

fn parse_ids(text: &str) -> Result<Vec<u32>> {
    text.split(',')
        .map(|t| {
            t.trim()
                .parse::<u32>()
                .with_context(|| format!("token id inválido: {t:?}"))
        })
        .collect()
}

fn main() -> Result<()> {
    let mut model = PathBuf::from("deploy/local-production/runtime/models/active");
    let mut user = String::from("Hola");
    let mut ids = None::<Vec<u32>>;
    let mut max_new_tokens = 1usize;
    let mut max_context_tokens = 128usize;
    let mut mode = WeightMode::Stream;
    let mut load_only = false;
    let mut verify_weights = false;
    let mut capture_logits = false;
    let mut thinking = false;
    let mut lod = None::<GpuRegionConfig>;
    let mut gpu_micro_sigma: Option<f32> = None;
    let mut gpu_micro_warmup: Option<usize> = None;
    let mut gpu_micro_open_fuse: Option<usize> = None;
    let mut prefill_batch: Option<usize> = None;
    let mut weight_reuse = 1usize;
    let mut fused_prefill_mask = 0u8;
    let mut hot_v_tokens = 0usize;
    let mut verified_prompt_replay_tokens = 0usize;
    let mut verified_shared_prefix = false;
    let mut micro_hot_v_tokens = 0usize;
    let mut deferred_cold = false;
    let mut fused_decode = false;
    let mut exact_hot_fused = false;
    let mut decode_unroll4 = false;
    let mut native_hip = false;
    let mut prefill_profile: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().context("--model value missing")?.into(),
            "--prompt" => user = args.next().context("--prompt value missing")?,
            "--ids" => ids = Some(parse_ids(&args.next().context("--ids value missing")?)?),
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .context("--max-new-tokens value missing")?
                    .parse()?
            }
            "--max-context-tokens" => {
                max_context_tokens = args
                    .next()
                    .context("--max-context-tokens value missing")?
                    .parse()?
            }
            "--resident" => mode = WeightMode::Resident,
            "--stream-weights" => mode = WeightMode::Stream,
            "--load-only" => load_only = true,
            "--verify-weights" => verify_weights = true,
            "--logits" => capture_logits = true,
            "--thinking" => thinking = true,
            "--prefill-profile" => {
                prefill_profile = Some(
                    args.next()
                        .context("missing --prefill-profile path")?
                        .into(),
                );
            }
            "--gpu-prefill-batch" => {
                prefill_batch = Some(
                    args.next()
                        .context("missing --gpu-prefill-batch token count")?
                        .parse()?,
                );
            }
            "--gpu-prefill-fallback" => {
                prefill_batch = None;
            }
            "--gpu-prefill-weight-reuse" => {
                weight_reuse = args
                    .next()
                    .context("missing --gpu-prefill-weight-reuse")?
                    .parse()?;
            }
            "--gpu-prefill-fused-projections" => {
                fused_prefill_mask = 3;
            }
            "--gpu-prefill-fused-qkv" => {
                fused_prefill_mask = 1;
            }
            "--gpu-prefill-fused-swiglu" => {
                fused_prefill_mask = 2;
            }
            "--gpu-prefill-separate-projections" => {
                fused_prefill_mask = 0;
            }
            "--gpu-native-hipblaslt" => {
                native_hip = true;
            }
            "--gpu-decode-unroll4" => {
                decode_unroll4 = true;
            }
            "--gpu-fused-decode" => {
                fused_decode = true;
            }
            "--gpu-exact-fused-attention" => {
                exact_hot_fused = true;
            }
            "--gpu-exact-original-attention" => {
                exact_hot_fused = false;
            }
            "--gpu-unfused-decode" => {
                fused_decode = false;
            }
            "--gpu-deferred-cold" => {
                deferred_cold = true;
            }
            "--gpu-immediate-cold" => {
                deferred_cold = false;
            }
            "--gpu-hot-v-tokens" => {
                hot_v_tokens = args.next().context("missing --gpu-hot-v-tokens")?.parse()?;
            }
            "--gpu-verified-prompt-replay" => {
                verified_prompt_replay_tokens = args
                    .next()
                    .context("missing --gpu-verified-prompt-replay max tokens")?
                    .parse()?;
            }
            "--gpu-no-prompt-replay" => {
                verified_prompt_replay_tokens = 0;
            }
            "--gpu-verified-shared-prefix" => {
                verified_shared_prefix = true;
            }
            "--gpu-no-shared-prefix" => {
                verified_shared_prefix = false;
            }
            "--gpu-micro-hot-v-tokens" => {
                micro_hot_v_tokens = args
                    .next()
                    .context("missing --gpu-micro-hot-v-tokens")?
                    .parse()?;
            }
            "--gpu-lod" => {
                lod.get_or_insert_with(GpuRegionConfig::default);
            }
            "--gpu-dense-kv" => {
                lod = None;
            }
            "--gpu-micro-novelty-sigma" => {
                gpu_micro_sigma = Some(
                    args.next()
                        .context("missing --gpu-micro-novelty-sigma")?
                        .parse()?,
                );
            }
            "--gpu-micro-warmup" => {
                gpu_micro_warmup =
                    Some(args.next().context("missing --gpu-micro-warmup")?.parse()?);
            }
            "--gpu-micro-open-fuse" => {
                gpu_micro_open_fuse = Some(
                    args.next()
                        .context("missing --gpu-micro-open-fuse")?
                        .parse()?,
                );
            }
            "--lod-epsilon" => {
                let value = args
                    .next()
                    .context("missing --lod-epsilon value")?
                    .parse()?;
                lod.get_or_insert_with(GpuRegionConfig::default).epsilon = value;
            }
            "--lod-stage" => {
                let value = args.next().context("missing --lod-stage value")?.parse()?;
                lod.get_or_insert_with(GpuRegionConfig::default)
                    .stage_tokens = value;
            }
            "--lod-cold-dir" => {
                let value = args.next().context("missing --lod-cold-dir path")?.into();
                lod.get_or_insert_with(GpuRegionConfig::default)
                    .cold_directory = Some(value);
            }
            "--help" | "-h" => {
                println!("M-SLIM BF16 Rust/OpenCL full LLM inference, experimental");
                println!(
                    "Usage: mslim-runtime-cli [--model dir] [--prompt Hola | --ids 1,2,3] [--max-new-tokens 1] [--max-context-tokens 128] [--resident|--stream-weights] [--load-only] [--verify-weights] [--logits] [--thinking] [--gpu-lod] [--lod-epsilon 0] [--lod-stage 64] [--gpu-dense-kv] [--gpu-prefill-batch 128|--gpu-prefill-fallback] [--prefill-profile PATH]"
                );
                println!(
                    "GPU computes all transformer layers. --stream-weights minimizes GPU allocations at the expense of repeated weight transfers."
                );
                return Ok(());
            }
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(
        (1..=512).contains(&max_new_tokens),
        "max_new_tokens must be 1..=512"
    );
    let prompt_ids = if let Some(ids) = ids {
        ids
    } else {
        let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let chat = raw_user_chat(&user, thinking);
        let enc = tokenizer
            .encode(chat.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
        enc.get_ids().to_vec()
    };
    ensure!(!prompt_ids.is_empty(), "no input tokens");
    ensure!(
        prompt_ids.len() + max_new_tokens <= max_context_tokens,
        "max_context_tokens smaller than prompt + generated tokens"
    );
    let startup = Instant::now();
    eprintln!("Starting BF16 OpenCL model upload/validation (mode={mode:?})...");
    let mut runtime = if native_hip {
        #[cfg(feature = "hip-backend")]
        {
            if let Some(config) = lod {
                OpenClLlm::load_with_gpu_region_native_hip(
                    &model,
                    max_context_tokens,
                    mode,
                    config,
                )?
            } else {
                OpenClLlm::load_dense_native_hip(&model, max_context_tokens, mode)?
            }
        }
        #[cfg(not(feature = "hip-backend"))]
        anyhow::bail!("Native gfx1103 hipBLASLt was not compiled; enable --features hip-backend")
    } else if let Some(config) = lod {
        OpenClLlm::load_with_gpu_region(&model, max_context_tokens, mode, config)?
    } else {
        OpenClLlm::load(&model, max_context_tokens, mode)?
    };
    if gpu_micro_sigma.is_some() || gpu_micro_warmup.is_some() || gpu_micro_open_fuse.is_some() {
        let mut policy = mslim::adaptive_extent::AxisPolicy::default();
        if let Some(sigma) = gpu_micro_sigma {
            policy.novelty_sigma = sigma;
        }
        if let Some(warmup) = gpu_micro_warmup {
            policy.warmup_observations = warmup;
        }
        if let Some(limit) = gpu_micro_open_fuse {
            policy.open_extent_safety_limit = Some(limit);
        }
        runtime.set_gpu_micro_axis_policy(policy)?;
    }
    runtime.set_batch_prefill_tokens(prefill_batch)?;
    runtime.set_prefill_weight_reuse_tokens(weight_reuse)?;
    runtime.set_prefill_projection_fusion_mode(fused_prefill_mask)?;
    runtime.set_exact_hot_v_tokens(hot_v_tokens)?;
    runtime.set_verified_gpu_prompt_replay(verified_prompt_replay_tokens)?;
    runtime.set_verified_gpu_prefix_replay(verified_shared_prefix)?;
    runtime.set_micro_hot_v_tokens(micro_hot_v_tokens)?;
    runtime.set_deferred_cold_commit(deferred_cold)?;
    runtime.set_decode_unroll4(decode_unroll4)?;
    runtime.set_decode_fused_kernels(fused_decode)?;
    runtime.set_exact_hot_fused_attention(exact_hot_fused)?;
    if let Some(path) = &prefill_profile {
        runtime.apply_promoted_prefill_profile(path)?;
        eprintln!("Applied measured prefill profile: {}", path.display());
    }

    eprintln!("Physical GPU: {}", runtime.device_name());
    eprintln!(
        "GPU checkpoint upload/init seconds: {:.3}",
        runtime.model_load_seconds
    );
    let weight_verification = if verify_weights {
        Some(runtime.verify_resident_weights()?)
    } else {
        None
    };
    if load_only {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "loaded": true, "full_gpu_model_weights_uploaded": mode == WeightMode::Resident,
                "mode": format!("{mode:?}"), "gpu": runtime.device_name(),
                "weight_bytes_from_checkpoint": runtime.gpu_weight_payload_bytes,
            "weight_verification":weight_verification,
            "gpu_reported_global_memory_bytes": runtime.device_global_memory_bytes(),
                "startup_seconds": startup.elapsed().as_secs_f64(),
                "note": "No token generated in --load-only mode",
            }))?
        );
        return Ok(());
    }
    eprintln!(
        "Model/GPU initialized; {} prompt IDs, {} max output",
        prompt_ids.len(),
        max_new_tokens
    );
    let generation = runtime.generate(&prompt_ids, max_new_tokens, capture_logits)?;
    let content = Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer readback: {e}"))?
        .decode(&generation.tokens, true)
        .map_err(|e| anyhow::anyhow!("decode generated token ids: {e}"))?;
    let without_logits: Vec<_> = generation
        .steps
        .iter()
        .map(|s| {
            json!({"position":s.position,
            "argmax_token":s.argmax_token,"forward_seconds":s.forward_seconds})
        })
        .collect();
    let logits_summary = generation
        .steps
        .iter()
        .map(|s| {
            s.logits.as_ref().map(|logits| {
                json!({
                    "max":logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max),
                    "min":logits.iter().cloned().fold(f32::INFINITY,f32::min),
                    "count":logits.len(),
                })
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "engine": generation.runtime, "device": generation.device_name,
            "full_llm_layers_computed_on_gpu": true,
            "cpu_for_device_matrix_arithmetic": false,
            "torch_backend_used": false,
            "mslim_lod_active": false,
            "input_prompt": user,
            "prompt_ids": prompt_ids,
            "completion_ids": generation.tokens,
            "completion_text": content,
            "weight_mode":format!("{mode:?}"),
            "gpu_region_components_active":generation.gpu_region_components_active,
            "gpu_region_stats":generation.gpu_region_stats,
            "gpu_kv_capacity_bytes":generation.gpu_kv_capacity_bytes,
            "gpu_micro_axis_novelty_sigma":runtime.gpu_micro_axis_policy().map(|p|p.novelty_sigma),
            "gpu_micro_axis_warmup":runtime.gpu_micro_axis_policy().map(|p|p.warmup_observations),
            "gpu_micro_axis_open_safety_fuse":runtime.gpu_micro_axis_policy().and_then(|p|p.open_extent_safety_limit),
            "calibrated_prefill_profile": runtime.calibrated_prefill_profile(),
            "gpu_prefill_batch_tokens":generation.gpu_prefill_batch_tokens,
            "gpu_prefill_weight_reuse_tokens":generation.gpu_prefill_weight_reuse_tokens,
            "gpu_prefill_fused_projections":generation.gpu_prefill_fused_projections,
            "gpu_prefill_fusion_mask":generation.gpu_prefill_fusion_mask,
            "gpu_verified_prompt_replay_max_tokens":generation.gpu_verified_prompt_replay_max_tokens,
            "gpu_verified_prefix_replay_enabled":generation.gpu_verified_prefix_replay_enabled,
            "gpu_verified_prefix_reused_tokens":generation.gpu_verified_prefix_reused_tokens,
            "gpu_verified_prefix_restored_bytes":generation.gpu_verified_prefix_restored_bytes,
            "gpu_verified_prompt_replay_hit":generation.gpu_verified_prompt_replay_hit,
            "gpu_verified_prompt_replay_cached_tokens":generation.gpu_verified_prompt_replay_cached_tokens,
            "gpu_verified_prompt_replay_cached_bytes":generation.gpu_verified_prompt_replay_cached_bytes,
            "gpu_verified_prompt_replay_restored_bytes":generation.gpu_verified_prompt_replay_restored_bytes,
            "gpu_verified_prompt_replay_sha256_verified":generation.gpu_verified_prompt_replay_sha256_verified,
            "gpu_decode_unroll4":generation.gpu_decode_unroll4,
            "gpu_decode_fused_kernels":generation.gpu_decode_fused_kernels,
            "gpu_exact_hot_fused_attention":generation.gpu_exact_hot_fused_attention,
            "gpu_deferred_cold_enabled":generation.gpu_deferred_cold_enabled,
            "gpu_deferred_cold_activated":generation.gpu_deferred_cold_activated,
            "gpu_deferred_cold_commit_seconds":generation.gpu_deferred_cold_commit_seconds,
            "gpu_deferred_cold_committed_token_layers":generation.gpu_deferred_cold_committed_token_layers,
            "gpu_hot_v_capacity_tokens":generation.gpu_hot_v_capacity_tokens,
            "gpu_micro_hot_v_capacity_tokens":generation.gpu_micro_hot_v_capacity_tokens,
            "gpu_micro_hot_sparse_calls":generation.gpu_micro_hot_sparse_calls,
            "gpu_micro_hot_exact_fallback_calls":generation.gpu_micro_hot_exact_fallback_calls,
            "gpu_micro_selection_upload_calls":generation.gpu_micro_selection_upload_calls,
            "gpu_micro_selection_cache_hits":generation.gpu_micro_selection_cache_hits,
            "gpu_hot_v_attention_calls":generation.gpu_hot_v_attention_calls,
            "gpu_hot_v_cold_fallback_calls":generation.gpu_hot_v_cold_fallback_calls,
            "gpu_hot_v_prefetch_bytes":generation.gpu_hot_v_prefetch_bytes,
            "gpu_hot_v_valid_prefix_tokens":generation.gpu_hot_v_valid_prefix_tokens,
            "gpu_prefill_batches":generation.gpu_prefill_batches,
            "gpu_prefill_cold_seal_seconds":generation.gpu_prefill_cold_seal_seconds,
            "gpu_prefill_temporary_v_capacity_bytes":generation.gpu_prefill_temporary_v_capacity_bytes,
            "model_weights_dtype":generation.model_weights_dtype,
            "gpu_compute":generation.gpu_compute,
            "native_hip_bf16_gemm_active":generation.hip_bf16_gemm_active,
            "steps":without_logits,
            "logits_summary":logits_summary,
            "prefill_seconds":generation.prefill_seconds,
            "total_generation_seconds":generation.total_seconds,
            "model_load_seconds":generation.model_load_seconds,
            "model_weight_bytes":generation.gpu_weight_payload_bytes,
            "model_config":runtime.spec().support_summary(),
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_raw_chat_template_format() {
        assert_eq!(
            raw_user_chat("Hola", false),
            "<s><|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
