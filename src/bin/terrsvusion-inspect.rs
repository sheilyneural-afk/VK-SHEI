//! Hardware-neutral model metadata and native CPU reference probe.
//! This CLI does NOT load weights or claim arbitrary LLM inference.
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::path::PathBuf;
use terrsvusion::model_config::ModelSpec;
use terrsvusion::portable::{
    AttentionGeometry, CpuReferenceBackend, ExactKvFrames, RawTensor, require_backend,
};

fn reference_smoke() -> Result<serde_json::Value> {
    use terrsvusion::portable::AttentionBackend;
    let geometry = AttentionGeometry::new(4, 2, 2)?;
    let mut kv = ExactKvFrames::new(geometry);
    let keys = RawTensor::from_f32(vec![1, 2, 2], &[0.0; 4])?;
    let values = RawTensor::from_f32(vec![1, 2, 2], &[1.0, -1.0, 2.0, -2.0])?;
    kv.append(keys, values)?;
    let cpu = CpuReferenceBackend;
    let output = kv.decode(&cpu, &[0.0; 8])?;
    anyhow::ensure!(
        output == [1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 2.0, -2.0],
        "la referencia CPU no coincide con el vector conocido"
    );
    Ok(json!({"backend": cpu.name(), "dtype": "FP32",
        "geometry": geometry, "output": output, "passed": true}))
}

fn main() -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut device = String::from("cpu");
    let mut self_test = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().context("falta ruta --model")?)),
            "--device" => device = args.next().context("falta valor --device")?,
            "--self-test" => self_test = true,
            "--help" | "-h" => {
                println!(
                    "Uso: terrsvusion-inspect [--model /ruta/modelo] [--device cpu|auto] [--self-test]"
                );
                println!(
                    "Inspección de metadatos BF16/FP16/FP32 sin PyTorch. No ejecuta el LLM completo."
                );
                return Ok(());
            }
            other => bail!("argumento desconocido: {other}"),
        }
    }
    let backend = require_backend(&device)?;
    if !self_test && model.is_none() {
        bail!("indica --model o --self-test");
    }
    let inspected = model.map(ModelSpec::inspect).transpose()?;
    let result = json!({
        "tool": "terrsvusion-inspect",
        "implementation_language": "rust",
        "compute_backend": format!("rust-cpu-reference ({})", device),
        "kernel_gpu_supported": false,
        "weights_modified": false,
        "model": inspected,
        "support": inspected.as_ref().map(ModelSpec::support_summary),
        "self_test": if self_test { Some(reference_smoke()?) } else { None },
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    let _ = backend; // Explicit fail-closed device validation.
    Ok(())
}
