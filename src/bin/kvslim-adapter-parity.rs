//! Real cross-adapter parity harness.
//!
//! This command intentionally requires both concrete runtimes. It never
//! substitutes synthetic data or a missing accelerator with a passing result.

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use kvslim::{
    adapter_catalog::create_adapter, conformance::compare_logits, engine::KvslimEngine,
    model_bundle::ModelBundle,
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn ordered_bf16(bits: u16) -> i32 {
    if bits & 0x8000 == 0 {
        0x8000 + i32::from(bits)
    } else {
        0x8000 - i32::from(bits & 0x7fff)
    }
}

fn within_one_bf16_ulp(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(&a, &b)| {
            if !a.is_finite() || !b.is_finite() {
                return a.to_bits() == b.to_bits();
            }
            let a = ordered_bf16(bf16::from_f32(a).to_bits());
            let b = ordered_bf16(bf16::from_f32(b).to_bits());
            (a - b).abs() <= 1
        })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("uso: kvslim-adapter-parity <bundle>")?);
    ensure!(args.next().is_none(), "uso: kvslim-adapter-parity <bundle>");
    ensure!(
        model.is_dir(),
        "el bundle no es un directorio: {}",
        model.display()
    );

    let bundle = ModelBundle::load(&model)?;
    let tokens = bundle.probe_token_ids()?;
    ensure!(tokens.len() >= 2, "la prueba necesita al menos dos tokens");
    let config: serde_json::Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;

    let mut cpu = KvslimEngine::new(create_adapter("native-cpu-runtime")?);
    let mut accelerator = KvslimEngine::new(create_adapter("accelerator-runtime")?);
    cpu.inspect(&model, &config).context("CPU inspect")?;
    accelerator
        .inspect(&model, &config)
        .context("accelerator inspect")?;
    cpu.load(&model, &config).context("CPU load")?;
    accelerator
        .load(&model, &config)
        .context("accelerator load")?;

    cpu.prefill(&tokens).context("CPU prefill")?;
    accelerator
        .prefill(&tokens)
        .context("accelerator prefill")?;
    let cpu_prefill_logits = cpu
        .last_logits()?
        .context("CPU did not capture prefill logits")?;
    let accelerator_prefill_logits = accelerator
        .last_logits()?
        .context("accelerator did not capture prefill logits")?;
    let prefill_parity = compare_logits(&cpu_prefill_logits, &accelerator_prefill_logits, 1e-3)?;

    let cpu_token = cpu.decode(tokens[0]).context("CPU decode")?;
    let accelerator_token = accelerator
        .decode(tokens[0])
        .context("accelerator decode")?;
    let cpu_decode_logits = cpu
        .last_logits()?
        .context("CPU did not capture decode logits")?;
    let accelerator_decode_logits = accelerator
        .last_logits()?
        .context("accelerator did not capture decode logits")?;
    let decode_parity = compare_logits(&cpu_decode_logits, &accelerator_decode_logits, 1e-3)?;
    let cpu_snapshot = cpu.snapshot_kv()?;
    let accelerator_snapshot = accelerator.snapshot_kv()?;
    let kv_equal = cpu_snapshot == accelerator_snapshot;
    let prefill_bf16_ulp_equivalent =
        within_one_bf16_ulp(&cpu_prefill_logits, &accelerator_prefill_logits);
    let decode_bf16_ulp_equivalent =
        within_one_bf16_ulp(&cpu_decode_logits, &accelerator_decode_logits);
    let passed = prefill_bf16_ulp_equivalent
        && decode_bf16_ulp_equivalent
        && cpu_token == accelerator_token
        && kv_equal;

    let receipt = json!({
        "schema": "kvslim.adapter-parity/v1",
        "status": if passed { "passed" } else { "failed" },
        "model": model,
        "probe_tokens": tokens.len(),
        "cpu_token": cpu_token,
        "accelerator_token": accelerator_token,
        "prefill_logits": prefill_parity,
        "decode_logits": decode_parity,
        "prefill_within_one_bf16_ulp": prefill_bf16_ulp_equivalent,
        "decode_within_one_bf16_ulp": decode_bf16_ulp_equivalent,
        "bit_exact": prefill_parity.exact && decode_parity.exact,
        "kv_snapshot_equal": kv_equal,
        "measured_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "activation": if passed { "eligible-for-reviewed-activation" } else { "fallback-required" }
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !passed {
        bail!("adapter parity failed; exact fallback remains required");
    }
    Ok(())
}
