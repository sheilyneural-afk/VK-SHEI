//! Real AMD/Intel/NVIDIA OpenCL GPU attention benchmark (not full LLM inference).
//! Uses the same GQA geometry as configured-model by default but synthetic KV/Q.

use anyhow::{Context, Result, bail, ensure};
use kvslim::opencl_backend::OpenClBackend;
use kvslim::portable::{
    AttentionBackend, AttentionGeometry, CpuReferenceBackend, ExactKvFrames, FloatFormat, RawTensor,
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn parse_dtype(text: &str) -> Result<FloatFormat> {
    match text.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(FloatFormat::Bf16),
        "f16" | "fp16" => Ok(FloatFormat::F16),
        "f32" | "fp32" => Ok(FloatFormat::F32),
        other => bail!("unknown dtype {other}; expected bf16, f16, or f32"),
    }
}

fn next_f32(seed: &mut u64) -> f32 {
    // Deterministic synthetic data; no model weight claims.
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (*seed >> 40) as u32;
    (u as f32 / 16_777_216.0f32 - 0.5) * 0.75
}

fn raw_values(format: FloatFormat, values: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(values.len() * format.bytes_per_element());
    for value in values {
        match format {
            FloatFormat::Bf16 => {
                output.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes())
            }
            FloatFormat::F16 => {
                output.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes())
            }
            FloatFormat::F32 => output.extend_from_slice(&value.to_le_bytes()),
        }
    }
    output
}

fn quantized_count(dtype: FloatFormat, data: &[f32]) -> usize {
    match dtype {
        FloatFormat::F32 => data.len(),
        FloatFormat::F16 | FloatFormat::Bf16 => data.len(),
    }
}

fn make_cache(
    geo: AttentionGeometry,
    dtype: FloatFormat,
    tokens: usize,
    frame_tokens: usize,
    seed: u64,
) -> Result<ExactKvFrames> {
    let mut rng = seed;
    let mut cache = ExactKvFrames::new(geo);
    let mut remaining = tokens;
    while remaining > 0 {
        let frame = frame_tokens.min(remaining);
        let n = frame * geo.kv_elements_per_token();
        let mut keys = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            keys.push(next_f32(&mut rng));
            values.push(next_f32(&mut rng));
        }
        let shape = vec![frame, geo.kv_heads, geo.head_dim];
        cache.append(
            RawTensor::from_le_bytes(dtype, shape.clone(), raw_values(dtype, &keys))?,
            RawTensor::from_le_bytes(dtype, shape, raw_values(dtype, &values))?,
        )?;
        remaining -= frame;
    }
    ensure!(cache.tokens() == tokens, "incorrect cache tokens");
    Ok(cache)
}

fn median_ms(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let mut tokens = 1024usize;
    let mut qheads = 16usize;
    let mut kvheads = 2usize;
    let mut dim = 128usize;
    let mut framesize = 256usize;
    let mut rounds = 3usize;
    let mut dtype = FloatFormat::Bf16;
    let mut model = None::<PathBuf>;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => tokens = args.next().context("missing --tokens")?.parse()?,
            "--qheads" => qheads = args.next().context("missing --qheads")?.parse()?,
            "--kvheads" => kvheads = args.next().context("missing --kvheads")?.parse()?,
            "--dim" => dim = args.next().context("missing --dim")?.parse()?,
            "--frame-tokens" => {
                framesize = args.next().context("missing --frame-tokens")?.parse()?
            }
            "--rounds" => rounds = args.next().context("missing --rounds")?.parse()?,
            "--dtype" => dtype = parse_dtype(&args.next().context("missing --dtype")?)?,
            "--model" => model = Some(PathBuf::from(args.next().context("missing --model")?)),
            "--help" | "-h" => {
                println!(
                    "Usage: kvslim-igpu-bench [--tokens 1024] [--dtype bf16|f16|f32] [--qheads 16] [--kvheads 2] [--dim 128] [--rounds 3] [--model /model]"
                );
                println!(
                    "Test actual OpenCL GPU attention against Rust CPU reference. Model --model reads config.json only; Q/K/V benchmark inputs are synthetic."
                );
                return Ok(());
            }
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(
        (1..=131_072).contains(&tokens),
        "tokens must be between 1 and 131072"
    );
    ensure!(
        (1..=4096).contains(&framesize),
        "frame size must be 1..=4096"
    );
    ensure!((1..=20).contains(&rounds), "rounds must be 1..=20");
    let geo = AttentionGeometry::new(qheads, kvheads, dim)?;
    ensure!(geo.query_elements() <= 65536, "too many Q elements");
    if let Some(path) = model.as_ref() {
        let spec = kvslim::model_config::ModelSpec::inspect(path)?;
        ensure!(
            spec.attention == geo,
            "supplied model requires Q={} KV={} dim={} (benchmark requested Q={} KV={} dim={})",
            spec.attention.query_heads,
            spec.attention.kv_heads,
            spec.attention.head_dim,
            geo.query_heads,
            geo.kv_heads,
            geo.head_dim
        );
    }
    let backend_started = Instant::now();
    let gpu = OpenClBackend::new()?;
    let gpu_compile_ms = backend_started.elapsed().as_secs_f64() * 1000.0;
    let cache = make_cache(geo, dtype, tokens, framesize, 0x7A1C_2026_1029)?;
    let mut random = 0x2026_1029_EB13_u64;
    let q: Vec<f32> = (0..geo.query_elements())
        .map(|_| next_f32(&mut random))
        .collect();
    ensure!(quantized_count(dtype, &q) > 0, "empty Q");

    let cpu_start = Instant::now();
    let cpu_out = cache.decode(&CpuReferenceBackend, &q)?;
    let cpu_wall_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let mut gpu_times = Vec::with_capacity(rounds);
    let mut gpu_out = Vec::new();
    let mut gpu_kernel_ms = Vec::new();
    for _ in 0..rounds {
        let started = Instant::now();
        gpu_out = cache.decode(&gpu, &q)?;
        gpu_times.push(started.elapsed().as_secs_f64() * 1000.0);
        gpu_kernel_ms.push(gpu.last_timings().context("missing GPU profiling events")?);
    }
    let max_error = cpu_out
        .iter()
        .zip(&gpu_out)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let rms_error = (cpu_out
        .iter()
        .zip(&gpu_out)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / cpu_out.len() as f64)
        .sqrt();
    let max_cpu_value = cpu_out
        .iter()
        .map(|v| v.abs() as f64)
        .fold(0.0f64, f64::max);
    let tolerance = 0.0005f64;
    let passed = max_error <= tolerance;
    let gpu_median = median_ms(&gpu_times);
    let metrics = gpu
        .last_timings()
        .context("missing OpenCL kernel metrics")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "engine": "kvslim-opencl-rust-native",
            "backend": gpu.name(),
            "hardware": gpu.device_info(),
            "physical_gpu_discovered_via": "OpenCL CL_DEVICE_TYPE_GPU",
            "model_config_used_only_for_geometry": model.map(|p|p.display().to_string()),
            "full_llm_inference_on_gpu": false,
            "test_data": "deterministic_synthetic_qkv",
            "format": dtype.label(),
            "geometry": geo,
            "tokens": tokens,
            "frames": cache.frames().len(),
            "kv_payload_bytes": cache.payload_bytes(),
            "q_elements": q.len(),
            "gpu_backend_compile_ms": gpu_compile_ms,
            "cpu_reference_ms": cpu_wall_ms,
            "gpu_wall_ms_by_round": gpu_times,
            "gpu_wall_median_ms": gpu_median,
            "gpu_kernel_qk_ms_last": metrics.qk_kernel_ms,
            "gpu_kernel_softmax_ms_last": metrics.softmax_kernel_ms,
            "gpu_kernel_pv_ms_last": metrics.pv_kernel_ms,
            "gpu_score_scratch_bytes": metrics.scratch_score_bytes,
            "gpu_input_kv_bytes": metrics.input_kv_bytes,
            "gpu_output_bytes": metrics.output_bytes,
            "gpu_vs_cpu_reference_ratio": cpu_wall_ms/gpu_median,
            "max_abs_attention_difference": max_error,
            "rms_attention_difference": rms_error,
            "max_abs_cpu_attention_value": max_cpu_value,
            "acceptance_abs_tolerance": tolerance,
            "numerical_check_passed": passed,
            "gpu_kernel_samples": gpu_kernel_ms.iter().map(|m|json!({
                "qk_ms":m.qk_kernel_ms,
                "softmax_ms":m.softmax_kernel_ms,
                "pv_ms":m.pv_kernel_ms,
                "dispatch_host_ms":m.dispatch_ms
            })).collect::<Vec<_>>(),
        }))?
    );
    if !passed {
        bail!("GPU output differs from CPU reference by {max_error} > tolerance {tolerance}");
    }
    Ok(())
}
