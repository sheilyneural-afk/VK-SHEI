//! Optional external model runtime process.
//! Concrete device/model code lives here, outside the universal host.
use anyhow::{Context, Result};
use kvslim::{
    ipc_transport::{read_request, write_response},
    opencl_adapter::OpenClModelAdapter,
    runtime_host::RuntimeHost,
};
use kvslim_adapter_api::{ModelAdapter, ModelAdapterSession, ModelIdentity, RuntimeSession};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
};

fn digest(path: &std::path::Path) -> Result<String> {
    Ok(hex::encode(Sha256::digest(fs::read(path)?)))
}

fn available_memory_bytes() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| text.lines().find(|line| line.starts_with("MemAvailable:")).and_then(|line| line.split_whitespace().nth(1)).and_then(|value| value.parse::<u64>().ok()))
        .unwrap_or(0)
        .saturating_mul(1024)
}

fn load(
    model: &std::path::Path,
    weight_mode: &str,
    cold_dir: &std::path::Path,
) -> Result<Box<dyn RuntimeSession>> {
    let selected_mode = if weight_mode == "auto" {
        let model_bytes = fs::read_dir(model)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.metadata().ok())
            .map(|meta| meta.len())
            .sum::<u64>();
        // Resident avoids rereading the full weight bundle for every decode.
        // Keep a dynamic safety margin for KV, allocator fragmentation and the OS.
        // The old fixed 3x rule forced streaming on 64 GB hosts even when the
        // model plus a bounded 8 GiB working set fit comfortably.
        let safety_margin = 8_u64 * 1024 * 1024 * 1024;
        if available_memory_bytes() >= model_bytes.saturating_add(safety_margin) {
            "resident"
        } else {
            "stream"
        }
    } else {
        weight_mode
    };
    eprintln!("runtime weight mode={selected_mode} requested={weight_mode}");
    let mut config: Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    if config.get("max_context_tokens").is_none() {
        let max = config
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .context("model context missing")?;
        config["max_context_tokens"] = max.into();
    }
    // Stream is the safe default; resident can be selected for high-memory hosts.
    config["kvslim_weight_mode"] = Value::String(selected_mode.into());
    config["kvslim_decode_fused_kernels"] = Value::Bool(true);
    config["kvslim_prefill_fuse_projections"] = Value::Bool(true);
    fs::create_dir_all(cold_dir)?;
    config["kvslim_cold_directory"] = Value::String(cold_dir.display().to_string());
    config["kvslim_hot_v_tokens"] = 512u64.into();
    config["kvslim_decode_unroll4"] = Value::Bool(true);
    let mut adapter = OpenClModelAdapter::default();
    adapter.load(model, &config)?;
    let identity = ModelIdentity {
        model_digest: digest(&model.join("config.json"))?,
        tokenizer_digest: digest(&model.join("tokenizer.json"))?,
        architecture_digest: digest(&model.join("config.json"))?,
    };
    Ok(Box::new(ModelAdapterSession::new(adapter, identity)))
}
fn serve(mut stream: UnixStream, session: Box<dyn RuntimeSession>, id: &str) -> Result<()> {
    let mut host = RuntimeHost::new(session, id)?;
    loop {
        let request = match read_request(&mut stream) {
            Ok(r) => r,
            Err(e) if e.to_string().contains("failed to fill whole buffer") => break,
            Err(e) => return Err(e),
        };
        let response = host.dispatch(request);
        let released =
            response.ok && response.payload.get("released").and_then(Value::as_bool) == Some(true);
        write_response(&mut stream, &response)?;
        if released {
            break;
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| PathBuf::from(&w[1]))
        .context("--model DIR is required")?;
    let socket = args
        .windows(2)
        .find(|w| w[0] == "--socket")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("/tmp/kvslim-provider.sock"));
    let weight_mode = args
        .windows(2)
        .find(|w| w[0] == "--weight-mode")
        .map(|w| w[1].as_str())
        .unwrap_or("stream");
    anyhow::ensure!(
        matches!(weight_mode, "auto" | "stream" | "resident"),
        "--weight-mode must be auto, stream or resident"
    );
    let cold_dir = args
        .windows(2)
        .find(|w| w[0] == "--cold-dir")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| {
            PathBuf::from("deploy/local-production/runtime/coldstore/external-provider")
        });
    let session_id = args
        .windows(2)
        .find(|w| w[0] == "--session-id")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "external-model-session".into());
    if socket.exists() {
        fs::remove_file(&socket)?;
    }
    let session = load(&model, weight_mode, &cold_dir)?;
    let listener = UnixListener::bind(&socket)?;
    println!(
        "external runtime provider listening on {}",
        socket.display()
    );
    if let Some(stream) = listener.incoming().next() {
        serve(stream?, session, &session_id)?;
    }
    Ok(())
}
