//! Strict, opt-in comparison of native BF16 Rust/OpenCL GPU inference versus
//! the already verified original MiniCPM5-1B dense BF16 LibTorch CPU model.
//! Same original safetensors, tokenizer, greedy input IDs and generation steps.
//! Neither the model checkpoint nor the Terrsvusion historical path is changed.

use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::path::PathBuf;
use terrsvusion::opencl_llm::{OpenClLlm, WeightMode};
use terrsvusion::torch_runtime::{TorchChatMessage, TorchModelRuntime};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let mut model = PathBuf::from("/home/yo/models/MiniCPM5-1B");
    let mut prompt = String::from("Hola");
    let mut max_tokens = 2usize;
    let mut max_context = 128usize;
    let mut mode = WeightMode::Resident;
    let mut token_ids = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().context("missing --model")?.into(),
            "--prompt" => prompt = it.next().context("missing --prompt")?,
            "--max-tokens" => max_tokens = it.next().context("missing --max-tokens")?.parse()?,
            "--max-context" => max_context = it.next().context("missing --max-context")?.parse()?,
            "--stream-weights" => mode = WeightMode::Stream,
            "--ids" => {
                token_ids = Some(
                    it.next()
                        .context("missing --ids")?
                        .split(',')
                        .map(|id| id.trim().parse::<u32>().map_err(anyhow::Error::from))
                        .collect::<Result<Vec<u32>>>()?,
                )
            }
            "--help" | "-h" => {
                println!(
                    "terrsvusion-compare-opencl [--model path] [--prompt Hola] [--ids 1,2,3] [--max-tokens 2] [--max-context 128] [--stream-weights]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
    }
    ensure!((1..=128).contains(&max_tokens), "max_tokens must be 1..128");
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let ids = if let Some(ids) = token_ids {
        ids
    } else {
        let messages = vec![TorchChatMessage {
            role: "user".to_owned(),
            content: prompt.clone(),
        }];
        let chat = TorchModelRuntime::render_messages_with_thinking(&messages, false)?;
        tokenizer
            .encode(chat.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec()
    };
    ensure!(
        !ids.is_empty() && ids.len() + max_tokens <= max_context,
        "insufficient GPU context capacity"
    );

    eprintln!("GPU model and BF16 KV reference comparison: {:?}", ids);
    let mut gpu = OpenClLlm::load(&model, max_context, mode)?;
    eprintln!(
        "GPU={} weights_loaded={}",
        gpu.device_name(),
        gpu.gpu_weight_payload_bytes
    );
    let gpu_generation = gpu.generate(&ids, max_tokens, true)?;
    let gpu_kv_sha = gpu.logical_kv_sha256()?;
    let gpu_kv_bytes = gpu.debug_kv_layer_bytes()?;
    let gpu_device = gpu.device_name().to_owned();
    eprintln!(
        "GPU inference completed: tokens={:?}",
        gpu_generation.tokens
    );
    drop(gpu); // release GPU weights before loading dense BF16 CPU oracle
    let original = TorchModelRuntime::load(&model)?;
    let dense_ids = ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>();
    let cpu_generation = original.generate_token_ids_dense(&dense_ids, max_tokens, 512)?;
    let mut step_rows = Vec::new();
    let mut max_abs_diff = 0.0f64;
    let mut first_token_mismatch = None;
    let mut all_logit_exact = true;
    for (step, (gpu_step, cpu_step)) in gpu_generation
        .steps
        .iter()
        .zip(cpu_generation.step_logits.iter())
        .enumerate()
    {
        let logits = gpu_step.logits.as_ref().context("GPU logits missing")?;
        ensure!(
            logits.len() == cpu_step.len(),
            "GPU/CPU logit vocabulary sizes differ"
        );
        let mut row_max = 0.0f64;
        let mut mean_abs = 0.0f64;
        let mut exact_count = 0usize;
        let mut first_logit_mismatch = None;
        for (idx, (a, b)) in logits.iter().zip(cpu_step).enumerate() {
            let diff = (*a as f64 - *b as f64).abs();
            if diff > 0.0 {
                first_logit_mismatch.get_or_insert(idx);
            } else {
                exact_count += 1;
            }
            row_max = row_max.max(diff);
            mean_abs += diff;
        }
        mean_abs /= logits.len() as f64;
        max_abs_diff = max_abs_diff.max(row_max);
        all_logit_exact &= row_max == 0.0;
        let cpu_token = cpu_generation.token_ids.get(step).copied();
        let gpu_token = gpu_generation.tokens.get(step).copied();
        if cpu_token != gpu_token {
            first_token_mismatch.get_or_insert(step);
        }
        step_rows.push(json!({
            "step":step,
            "cpu_token":cpu_token,"gpu_token":gpu_token,
            "max_abs_logit_difference":row_max,
            "mean_abs_logit_difference":mean_abs,
            "first_logit_mismatch_index":first_logit_mismatch,
            "exactly_equal_logits_count":exact_count,
            "vocab_count":logits.len(),
        }));
    }
    let dense_kv_bytes = original.debug_dense_kv_bytes(&dense_ids, &gpu_generation.tokens)?;
    ensure!(
        gpu_kv_bytes.len() == dense_kv_bytes.len(),
        "GPU and dense KV have different layer counts"
    );
    let mut kv_mismatch_layers = Vec::new();
    let mut total_mismatches = 0usize;
    let mut max_abs_kv_difference = 0.0f64;
    let mut first_kv_divergent_layer = None;
    for (layer, ((gk, gv), (dk, dv))) in gpu_kv_bytes.iter().zip(&dense_kv_bytes).enumerate() {
        let mut key_mismatches = 0usize;
        let mut value_mismatches = 0usize;
        let mut layer_max = 0.0f64;
        for (ga, da, counter) in [
            (gk, dk, &mut key_mismatches),
            (gv, dv, &mut value_mismatches),
        ] {
            ensure!(
                ga.len() == da.len() && ga.len() % 2 == 0,
                "GPU/dense KV payload size mismatch in layer {}",
                layer
            );
            for (a, b) in ga.chunks_exact(2).zip(da.chunks_exact(2)) {
                let abits = u16::from_le_bytes([a[0], a[1]]);
                let bbits = u16::from_le_bytes([b[0], b[1]]);
                if abits != bbits {
                    *counter += 1;
                    let x = half::bf16::from_bits(abits).to_f32();
                    let y = half::bf16::from_bits(bbits).to_f32();
                    layer_max = layer_max.max((x as f64 - y as f64).abs());
                }
            }
        }
        total_mismatches += key_mismatches + value_mismatches;
        max_abs_kv_difference = max_abs_kv_difference.max(layer_max);
        if key_mismatches + value_mismatches > 0 {
            first_kv_divergent_layer.get_or_insert(layer);
        }
        kv_mismatch_layers.push(json!({
            "layer":layer,
            "key_bf16_mismatched_elements":key_mismatches,
            "value_bf16_mismatched_elements":value_mismatches,
            "max_abs_kv_difference":layer_max,
        }));
    }
    let token_parity = gpu_generation.tokens == cpu_generation.token_ids;
    let kv_sha_equal = gpu_kv_sha == cpu_generation.kv_sha256;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "comparison":"MiniCPM5-1B_dense_BF16_CPU_vs_native_Rust_OpenCL_iGPU",
            "same_weights":true,
            "same_tokenizer":true,
            "same_input_token_ids":ids,
            "same_greedy_decoding":true,
            "gpu_device":gpu_device,
            "gpu_outputs":gpu_generation.tokens,
            "dense_cpu_outputs":cpu_generation.token_ids,
            "token_parity":token_parity,
            "first_token_mismatch":first_token_mismatch,
            "logits_bit_exact":all_logit_exact,
            "max_abs_logit_difference":max_abs_diff,
            "logit_comparisons":step_rows,
            "gpu_kv_sha256":gpu_kv_sha,
            "dense_kv_sha256":cpu_generation.kv_sha256,
            "kv_sha256_equal":kv_sha_equal,
            "first_kv_divergent_layer":first_kv_divergent_layer,
            "kv_total_bf16_element_mismatches":total_mismatches,
            "max_abs_kv_difference":max_abs_kv_difference,
            "kv_layer_differences":kv_mismatch_layers,
            "gpu_total_seconds":gpu_generation.total_seconds,
            "gpu_load_seconds":gpu_generation.model_load_seconds,
            "dense_cpu_seconds":cpu_generation.latency_seconds,
            "gpu_historical_terravision_lod_active":false,
        }))?
    );
    Ok(())
}
