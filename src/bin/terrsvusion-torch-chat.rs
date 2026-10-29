use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use terrsvusion::torch_runtime::TorchModelRuntime;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/yo/models/MiniCPM5-1B"));
    let prompt = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "Responde exactamente: HOLA".to_owned());
    let load_started = Instant::now();
    let runtime = TorchModelRuntime::load(&model_dir)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let generation_started = Instant::now();
    let generation = runtime.generate_chat(&prompt, 16)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "runtime": "rust-tch-libtorch",
            "model": model_dir,
            "weights_dtype": "bf16",
            "weights_transformed": false,
            "load_seconds": load_seconds,
            "generation_seconds": generation_started.elapsed().as_secs_f64(),
            "generation": generation
        }))?
    );
    Ok(())
}
