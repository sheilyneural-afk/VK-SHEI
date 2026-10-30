//! Measure the real local execution path and persist only observations.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Serialize)]
struct CalibrationReceipt {
    schema: &'static str,
    bundle: String,
    adapter: String,
    identity_digest: String,
    started_unix_seconds: u64,
    prefill_micros: u128,
    decode_micros: u128,
    kv_snapshot_bytes: usize,
    kv_report: Value,
    peak_rss_bytes: Option<u64>,
    context_samples: Vec<ContextSample>,
    checks: Vec<&'static str>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct ContextSample {
    tokens_added: usize,
    tokens_total: usize,
    prefill_micros: u128,
    snapshot_bytes: usize,
    peak_rss_bytes: Option<u64>,
    status: &'static str,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bundle = PathBuf::from(args.first().context(
        "uso: kvslim-calibrate <deploy/local-production/runtime/models/<digest>> [--force]",
    )?);
    let force = args.iter().any(|arg| arg == "--force");
    let skip_context_sweep = args.iter().any(|arg| arg == "--skip-context-sweep");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundle = if bundle.is_absolute() {
        bundle
    } else {
        root.join(bundle)
    };
    let digest = bundle
        .file_name()
        .and_then(|name| name.to_str())
        .context("digest inválido")?;
    let destination = root.join("deploy/local-production/runtime/calibration");
    let path = destination.join(format!("{digest}.json"));
    let config: Value = serde_json::from_slice(&fs::read(bundle.join("config.json"))?)?;
    let selection = kvslim::adapter_catalog::select_available_for_model(&bundle, None, None)?;
    let host = kvslim::adapter_catalog::host_capabilities();
    let identity_input = serde_json::json!({
        "model_digest": digest,
        "config": config,
        "adapter": selection,
        "host": host,
        "engine_schema": "kvslim.engine/v1",
    });
    let identity_digest = hex::encode(Sha256::digest(serde_json::to_vec(&identity_input)?));
    if !force && let Ok(bytes) = fs::read(&path) {
        let existing: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("certificación existente inválida: {}", path.display()))?;
        if existing.get("schema").and_then(Value::as_str) == Some("kvslim.calibration_receipt/v1")
            && existing.get("status").and_then(Value::as_str) == Some("measured-basic-execution")
            && existing.get("identity_digest").and_then(Value::as_str) == Some(&identity_digest)
        {
            println!("{}", serde_json::to_string_pretty(&existing)?);
            return Ok(());
        }
    }
    let adapter = kvslim::adapter_catalog::create_adapter(&selection.adapter_id)?;
    let mut engine = kvslim::engine::KvslimEngine::new(adapter);
    engine.inspect(&bundle, &config)?;
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let load_start = Instant::now();
    engine.load(&bundle, &config)?;
    let _load_elapsed = load_start.elapsed();
    let prefill_start = Instant::now();
    let result = engine.prefill(&[0])?;
    let prefill_micros = prefill_start.elapsed().as_micros();
    ensure!(result.is_object(), "prefill no estructurado");
    let kv_report = result
        .get("kv")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"authority": "adapter-managed"}));
    let decode_start = Instant::now();
    let _ = engine.decode(0)?;
    let decode_micros = decode_start.elapsed().as_micros();
    let snapshot = engine.snapshot_kv()?;
    ensure!(!snapshot.is_empty(), "snapshot KV vacío");
    engine.restore_kv(&snapshot)?;
    let mut context_samples = vec![ContextSample {
        tokens_added: 1,
        tokens_total: 1,
        prefill_micros,
        snapshot_bytes: snapshot.len(),
        peak_rss_bytes: peak_rss_bytes(),
        status: "stable",
    }];
    let max_context = config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let mut tokens_total = 1usize;
    if !skip_context_sweep {
        while let Some(limit) = max_context {
            if tokens_total >= limit {
                break;
            }
            let target = tokens_total.saturating_mul(2).min(limit);
            let tokens_added = target - tokens_total;
            let input = vec![0u32; tokens_added];
            let start = Instant::now();
            let result = engine.prefill(&input);
            match result {
                Ok(value) if value.is_object() => {
                    tokens_total = target;
                    let snapshot = engine.snapshot_kv()?;
                    context_samples.push(ContextSample {
                        tokens_added,
                        tokens_total,
                        prefill_micros: start.elapsed().as_micros(),
                        snapshot_bytes: snapshot.len(),
                        peak_rss_bytes: peak_rss_bytes(),
                        status: "stable",
                    });
                }
                _ => break,
            }
        }
    }
    let receipt = CalibrationReceipt {
        schema: "kvslim.calibration_receipt/v1",
        bundle: digest.to_owned(),
        adapter: selection.adapter_id,
        identity_digest,
        started_unix_seconds: started,
        prefill_micros,
        decode_micros,
        kv_snapshot_bytes: snapshot.len(),
        kv_report,
        peak_rss_bytes: peak_rss_bytes(),
        context_samples,
        checks: vec![
            "inspect",
            "load",
            "prefill",
            "decode",
            "kv_snapshot",
            "kv_restore",
        ],
        status: "measured-basic-execution",
    };
    fs::create_dir_all(&destination)?;
    let temporary = tempfile::NamedTempFile::new_in(&destination)?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&receipt)?)?;
    temporary.persist(&path).map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
