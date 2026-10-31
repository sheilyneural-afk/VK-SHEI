//! Real cross-adapter parity gate.
//!
//! The logical replay snapshot exposed by the adapter API is checked as a
//! lifecycle contract. It is not, and must never be reported as, a comparison
//! of physical K/V tensor bytes.

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use mslim::{
    adapter_catalog::create_adapter, conformance::compare_logits, engine::MSlimEngine,
    model_bundle::ModelBundle,
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_DECODE_STEPS: usize = 8;

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
    let model = PathBuf::from(
        args.next()
            .context("uso: mslim-adapter-parity <bundle> [--decode-steps N]")?,
    );
    let mut decode_steps = DEFAULT_DECODE_STEPS;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--decode-steps" => {
                decode_steps = args.next().context("falta --decode-steps")?.parse()?
            }
            other => bail!("argumento desconocido: {other}"),
        }
    }
    ensure!(
        (1..=128).contains(&decode_steps),
        "--decode-steps debe ser 1..=128"
    );
    ensure!(
        model.is_dir(),
        "el bundle no es un directorio: {}",
        model.display()
    );
    let bundle_id = model
        .file_name()
        .and_then(|value| value.to_str())
        .context("bundle sin identidad válida")?;
    ensure!(
        bundle_id.len() == 64 && bundle_id.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "el bundle debe estar identificado por su SHA-256"
    );

    let bundle = ModelBundle::load(&model)?;
    let tokens = bundle.probe_token_ids()?;
    ensure!(tokens.len() >= 2, "la prueba necesita al menos dos tokens");
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    let parity_cold = tempfile::tempdir()?;
    config
        .as_object_mut()
        .context("model config must be a JSON object")?
        .insert("mslim_cold_directory".into(), json!(parity_cold.path()));

    let mut cpu = MSlimEngine::new(create_adapter("native-cpu-runtime")?);
    let mut accelerator = MSlimEngine::new(create_adapter("accelerator-runtime")?);
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
    let prefill_bf16_ulp_equivalent =
        within_one_bf16_ulp(&cpu_prefill_logits, &accelerator_prefill_logits);

    let mut input_token = tokens[0];
    let mut decode_rows = Vec::with_capacity(decode_steps);
    let mut all_decode_ulp_equivalent = true;
    let mut all_decode_tokens_equal = true;
    let mut all_decode_logits_exact = true;
    for step in 0..decode_steps {
        let cpu_token = cpu.decode(input_token).context("CPU decode")?;
        let accelerator_token = accelerator
            .decode(input_token)
            .context("accelerator decode")?;
        let cpu_logits = cpu
            .last_logits()?
            .context("CPU did not capture decode logits")?;
        let accelerator_logits = accelerator
            .last_logits()?
            .context("accelerator did not capture decode logits")?;
        let parity = compare_logits(&cpu_logits, &accelerator_logits, 1e-3)?;
        let ulp_equivalent = within_one_bf16_ulp(&cpu_logits, &accelerator_logits);
        all_decode_ulp_equivalent &= ulp_equivalent;
        all_decode_tokens_equal &= cpu_token == accelerator_token;
        all_decode_logits_exact &= parity.exact;
        decode_rows.push(json!({
            "step": step,
            "input_token": input_token,
            "cpu_token": cpu_token,
            "accelerator_token": accelerator_token,
            "token_equal": cpu_token == accelerator_token,
            "logits": parity,
            "within_one_bf16_ulp": ulp_equivalent,
        }));
        // Both adapters receive the same reference continuation. A divergence
        // cannot fork the two histories and hide the first causal mismatch.
        input_token = cpu_token;
    }

    let cpu_snapshot: serde_json::Value = serde_json::from_slice(&cpu.snapshot_kv()?)?;
    let accelerator_snapshot: serde_json::Value =
        serde_json::from_slice(&accelerator.snapshot_kv()?)?;
    let cpu_snapshot_tokens = cpu_snapshot
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .context("CPU snapshot missing token history")?;
    let accelerator_snapshot_tokens = accelerator_snapshot
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .context("accelerator snapshot missing token history")?;
    // Backends deliberately use different snapshot authorities: CPU exposes a
    // logical replay recipe while OpenCL references physical canonical
    // ColdStore. Compare their shared token history, never raw envelopes.
    let logical_token_history_equal = cpu_snapshot_tokens == accelerator_snapshot_tokens;
    let accelerator_physical_snapshot = accelerator_snapshot
        .get("schema")
        .and_then(serde_json::Value::as_str)
        == Some("mslim.runtime.physical-coldstore-snapshot/v1")
        && accelerator_snapshot
            .get("authority")
            .and_then(serde_json::Value::as_str)
            == Some("canonical-bf16-coldstore");
    let passed = prefill_bf16_ulp_equivalent
        && all_decode_ulp_equivalent
        && all_decode_tokens_equal
        && logical_token_history_equal
        && accelerator_physical_snapshot;

    let receipt = json!({
        "schema": "mslim.adapter-parity/v2",
        "status": if passed { "passed" } else { "failed" },
        "bundle_sha256": bundle_id,
        "model": model,
        "probe_tokens": tokens.len(),
        "decode_steps": decode_steps,
        "prefill_logits": prefill_parity,
        "prefill_within_one_bf16_ulp": prefill_bf16_ulp_equivalent,
        "decode": decode_rows,
        "all_decode_tokens_equal": all_decode_tokens_equal,
        "all_decode_logits_exact": all_decode_logits_exact,
        "all_decode_within_one_bf16_ulp": all_decode_ulp_equivalent,
        "bit_exact": prefill_parity.exact && all_decode_logits_exact,
        "snapshot_contract": {
            "cpu_kind": "logical-replay-token-history",
            "accelerator_kind": "physical-coldstore-handle",
            "logical_token_history_equal": logical_token_history_equal,
            "accelerator_physical_snapshot": accelerator_physical_snapshot,
            "physical_kv_tensor_bytes_compared": false,
            "physical_kv_persistence_verified_by_parity_gate": false,
            "note": "La paridad compara el historial lógico compartido y exige que el acelerador use autoridad ColdStore; la prueba física de reapertura se ejecuta por separado."
        },
        "measured_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "activation": if passed {
            "eligible-for-reviewed-activation"
        } else {
            "blocked-fallback-required"
        }
    });
    let receipt_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("deploy/local-production/runtime/validation");
    fs::create_dir_all(&receipt_dir)?;
    let receipt_path = receipt_dir.join(format!("adapter-parity-{bundle_id}.json"));
    let temporary = tempfile::NamedTempFile::new_in(&receipt_dir)?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&receipt)?)?;
    temporary
        .persist(&receipt_path)
        .map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !passed {
        bail!("adapter parity failed; accelerator activation is blocked");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::within_one_bf16_ulp;

    #[test]
    fn bf16_ulp_comparison_rejects_large_drift() {
        assert!(within_one_bf16_ulp(&[1.0], &[1.0]));
        assert!(!within_one_bf16_ulp(&[1.0], &[1.5]));
    }
}
