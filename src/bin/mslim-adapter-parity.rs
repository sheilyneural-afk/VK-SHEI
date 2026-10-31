//! Four-route validation matrix for the native CPU and OpenCL runtimes.
//!
//! Activation is decided only by deterministic same-backend comparisons:
//! CPU hot-dense vs CPU exact ColdStore, and GPU dense vs GPU exact ColdStore.
//! CPU vs GPU remains a numerical diagnostic and never impersonates a
//! physical K/V integrity certificate.

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use mslim::{
    adapter_catalog::create_adapter, conformance::compare_logit_distributions, engine::MslimEngine,
    model_bundle::ModelBundle,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_DECODE_STEPS: usize = 8;

#[derive(Clone)]
struct StepTrace {
    input_token: u32,
    predicted_token: u32,
    logits: Vec<f32>,
}

#[derive(Clone)]
struct ExecutionTrace {
    prefill_logits: Vec<f32>,
    decode: Vec<StepTrace>,
    kv_integrity: Value,
    snapshot: Value,
}

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

fn object(config: &mut Value) -> Result<&mut serde_json::Map<String, Value>> {
    config
        .as_object_mut()
        .context("model config must be a JSON object")
}

fn frozen_common_config(base: &Value) -> Result<Value> {
    let mut config = base.clone();
    let values = object(&mut config)?;
    values.insert("mslim_lod_epsilon".into(), json!(0.0));
    values.insert("mslim_epsilon".into(), json!(0.0));
    values.insert("mslim_stage_tokens".into(), json!(64));
    values.insert("mslim_prefill_batch_tokens".into(), json!(128));
    values.insert("mslim_prompt_replay_tokens".into(), json!(0));
    values.insert("mslim_shared_prefix".into(), json!(false));
    values.insert("mslim_deferred_cold_commit".into(), json!(false));
    values.insert("mslim_exact_hot_fused_attention".into(), json!(true));
    values.insert("mslim_decode_unroll4".into(), json!(false));
    values.insert("mslim_decode_fused_kernels".into(), json!(true));
    values.insert("mslim_prefill_fuse_projections".into(), json!(true));
    Ok(config)
}

fn cpu_config(base: &Value, dense_hot: bool) -> Result<Value> {
    let mut config = frozen_common_config(base)?;
    object(&mut config)?.insert(
        "native_hot_kv_bytes".into(),
        if dense_hot { json!(u64::MAX) } else { json!(0) },
    );
    Ok(config)
}

fn gpu_config(base: &Value, exact: bool, cold_root: &Path) -> Result<Value> {
    let mut config = frozen_common_config(base)?;
    let values = object(&mut config)?;
    values.insert(
        "mslim_kv_mode".into(),
        json!(if exact { "exact-coldstore" } else { "dense" }),
    );
    if exact {
        values.insert("mslim_cold_directory".into(), json!(cold_root));
    } else {
        values.remove("mslim_cold_directory");
        values.remove("mslim_cold_dir");
    }
    Ok(config)
}

fn run_once(
    engine: &mut MslimEngine,
    prompt: &[u32],
    decode_inputs: &[u32],
) -> Result<ExecutionTrace> {
    engine.prefill(prompt)?;
    let prefill_logits = engine
        .last_logits()?
        .context("runtime did not capture prefill logits")?;
    let mut decode = Vec::with_capacity(decode_inputs.len());
    for &input_token in decode_inputs {
        let predicted_token = engine.decode(input_token)?;
        let logits = engine
            .last_logits()?
            .context("runtime did not capture decode logits")?;
        decode.push(StepTrace {
            input_token,
            predicted_token,
            logits,
        });
    }
    let kv_integrity = engine
        .kv_integrity_report()?
        .context("runtime does not expose a physical/logical KV integrity proof")?;
    let snapshot = serde_json::from_slice(&engine.snapshot_kv()?)?;
    Ok(ExecutionTrace {
        prefill_logits,
        decode,
        kv_integrity,
        snapshot,
    })
}

fn run_mode(
    adapter: &str,
    model: &Path,
    config: &Value,
    prompt: &[u32],
    decode_inputs: &[u32],
) -> Result<(ExecutionTrace, Value)> {
    let mut engine = MslimEngine::new(create_adapter(adapter)?);
    engine.inspect(model, config)?;
    engine.load(model, config)?;
    let first = run_once(&mut engine, prompt, decode_inputs)?;
    engine.reset_session()?;
    let second = run_once(&mut engine, prompt, decode_inputs)?;
    let determinism = compare_traces("run-1", &first, "run-2", &second)?;
    Ok((first, determinism))
}

fn kv_digest(trace: &ExecutionTrace) -> Result<&str> {
    trace
        .kv_integrity
        .get("logical_kv_sha256")
        .and_then(Value::as_str)
        .context("KV integrity report missing logical_kv_sha256")
}

fn compare_traces(
    reference_name: &str,
    reference: &ExecutionTrace,
    candidate_name: &str,
    candidate: &ExecutionTrace,
) -> Result<Value> {
    ensure!(
        reference.decode.len() == candidate.decode.len(),
        "decode trace lengths differ"
    );
    let prefill =
        compare_logit_distributions(&reference.prefill_logits, &candidate.prefill_logits)?;
    let prefill_ulp = within_one_bf16_ulp(&reference.prefill_logits, &candidate.prefill_logits);
    let mut rows = Vec::with_capacity(reference.decode.len());
    let mut all_logits_exact = prefill.exact;
    let mut all_tokens_equal = true;
    let mut all_top1_equal = prefill.top1_equal;
    let mut all_within_one_bf16_ulp = prefill_ulp;
    for (step, (left, right)) in reference.decode.iter().zip(&candidate.decode).enumerate() {
        ensure!(
            left.input_token == right.input_token,
            "comparison used different teacher-forced input at step {step}"
        );
        let distribution = compare_logit_distributions(&left.logits, &right.logits)?;
        let ulp = within_one_bf16_ulp(&left.logits, &right.logits);
        all_logits_exact &= distribution.exact;
        all_tokens_equal &= left.predicted_token == right.predicted_token;
        all_top1_equal &= distribution.top1_equal;
        all_within_one_bf16_ulp &= ulp;
        rows.push(json!({
            "step": step,
            "input_token": left.input_token,
            "reference_token": left.predicted_token,
            "candidate_token": right.predicted_token,
            "token_equal": left.predicted_token == right.predicted_token,
            "distribution": distribution,
            "within_one_bf16_ulp": ulp,
        }));
    }
    let reference_digest = kv_digest(reference)?;
    let candidate_digest = kv_digest(candidate)?;
    Ok(json!({
        "reference": reference_name,
        "candidate": candidate_name,
        "prefill": {
            "distribution": prefill,
            "within_one_bf16_ulp": prefill_ulp,
        },
        "decode": rows,
        "all_logits_exact": all_logits_exact,
        "all_tokens_equal": all_tokens_equal,
        "all_top1_equal": all_top1_equal,
        "all_within_one_bf16_ulp": all_within_one_bf16_ulp,
        "kv": {
            "reference": reference.kv_integrity,
            "candidate": candidate.kv_integrity,
            "logical_bytes_equal": reference_digest == candidate_digest,
        },
        "exact_same_backend_passed": all_logits_exact
            && all_tokens_equal
            && reference_digest == candidate_digest,
    }))
}

fn comparison_passed(value: &Value) -> bool {
    value
        .get("exact_same_backend_passed")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn physical_snapshot(trace: &ExecutionTrace) -> bool {
    trace.snapshot.get("schema").and_then(Value::as_str)
        == Some("mslim.runtime.physical-coldstore-snapshot/v1")
        && trace.snapshot.get("authority").and_then(Value::as_str)
            == Some("canonical-bf16-coldstore")
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
    let prompt = bundle.probe_token_ids()?;
    ensure!(prompt.len() >= 2, "la prueba necesita al menos dos tokens");
    let decode_inputs = prompt
        .iter()
        .copied()
        .cycle()
        .take(decode_steps)
        .collect::<Vec<_>>();
    let base_config: Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    let gpu_cold = tempfile::tempdir()?;

    let (cpu_dense, cpu_dense_determinism) = run_mode(
        "native-cpu-runtime",
        &model,
        &cpu_config(&base_config, true)?,
        &prompt,
        &decode_inputs,
    )?;
    let (cpu_exact, cpu_exact_determinism) = run_mode(
        "native-cpu-runtime",
        &model,
        &cpu_config(&base_config, false)?,
        &prompt,
        &decode_inputs,
    )?;
    let cpu_same_backend = compare_traces(
        "cpu-hot-dense",
        &cpu_dense,
        "cpu-exact-coldstore",
        &cpu_exact,
    )?;

    let (gpu_dense, gpu_dense_determinism) = run_mode(
        "accelerator-runtime",
        &model,
        &gpu_config(&base_config, false, gpu_cold.path())?,
        &prompt,
        &decode_inputs,
    )?;
    let (gpu_exact, gpu_exact_determinism) = run_mode(
        "accelerator-runtime",
        &model,
        &gpu_config(&base_config, true, gpu_cold.path())?,
        &prompt,
        &decode_inputs,
    )?;
    let gpu_same_backend =
        compare_traces("gpu-dense", &gpu_dense, "gpu-exact-coldstore", &gpu_exact)?;
    let cross_backend = compare_traces("cpu-hot-dense", &cpu_dense, "gpu-dense", &gpu_dense)?;

    let deterministic = [
        &cpu_dense_determinism,
        &cpu_exact_determinism,
        &gpu_dense_determinism,
        &gpu_exact_determinism,
    ]
    .into_iter()
    .all(comparison_passed);
    let same_backend_exact =
        comparison_passed(&cpu_same_backend) && comparison_passed(&gpu_same_backend);
    let accelerator_physical_snapshot = physical_snapshot(&gpu_exact);
    let passed = deterministic && same_backend_exact && accelerator_physical_snapshot;

    let receipt = json!({
        "schema": "mslim.validation-matrix/v1",
        "status": if passed { "passed" } else { "failed" },
        "source_revision": mslim::build_info::source_revision(),
        "bundle_sha256": bundle_id,
        "model": model,
        "probe_tokens": prompt.len(),
        "decode_steps": decode_steps,
        "teacher_forced_decode_inputs": decode_inputs,
        "policy": {
            "activation_depends_on_cross_backend_bit_parity": false,
            "same_backend_requires_exact_logits": true,
            "same_backend_requires_equal_logical_kv_bytes": true,
            "same_backend_requires_equal_tokens": true,
            "approximation_epsilon": 0.0,
            "cross_backend_role": "diagnostic-only",
        },
        "frozen_runtime_configuration": {
            "stage_tokens": 64,
            "prefill_batch_tokens": 128,
            "prompt_replay_tokens": 0,
            "shared_prefix": false,
            "deferred_cold_commit": false,
            "exact_hot_fused_attention": true,
            "decode_unroll4": false,
            "decode_fused_kernels": true,
            "prefill_fuse_projections": true,
        },
        "determinism": {
            "cpu_dense": cpu_dense_determinism,
            "cpu_exact": cpu_exact_determinism,
            "gpu_dense": gpu_dense_determinism,
            "gpu_exact": gpu_exact_determinism,
            "all_passed": deterministic,
        },
        "same_backend": {
            "cpu_dense_vs_exact": cpu_same_backend,
            "gpu_dense_vs_exact": gpu_same_backend,
            "all_passed": same_backend_exact,
        },
        "cross_backend_diagnostic": cross_backend,
        "snapshot_contract": {
            "accelerator_exact_physical_snapshot": accelerator_physical_snapshot,
            "physical_kv_tensor_bytes_compared_via_logical_sha256": true,
            "logical_serialization": "k-then-v-layer-major-head-token-dim",
        },
        "measured_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "activation": if passed {
            "eligible-for-reviewed-activation"
        } else {
            "blocked-by-same-backend-or-integrity-gate"
        },
    });
    let receipt_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("deploy/local-production/runtime/validation");
    fs::create_dir_all(&receipt_dir)?;
    let receipt_path = receipt_dir.join(format!("validation-matrix-{bundle_id}.json"));
    let temporary = tempfile::NamedTempFile::new_in(&receipt_dir)?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&receipt)?)?;
    temporary
        .persist(&receipt_path)
        .map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !passed {
        bail!("same-backend validation failed; inspect validation matrix before activation");
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
