//! Generic runtime host process for the provider-neutral IPC protocol.
//!
//! This binary is a protocol harness: it owns a RuntimeSession supplied by the
//! host process and never selects a device or provider. The built-in session is
//! deterministic and exists only for smoke testing the transport; production
//! launches replace it with an externally supplied runtime implementation.
use anyhow::{Context, Result, bail};
use kvslim::{
    ipc_transport::{read_request, write_response},
    opencl_adapter::OpenClModelAdapter,
    runtime_host::RuntimeHost,
};
use kvslim_adapter_api::{
    AdapterCapabilities, CacheMode, CacheReceipt, GenerationRequest, GenerationResult,
    GenerationStats, ModelAdapter, ModelAdapterSession, ModelIdentity, RuntimeSession,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
};

struct DeterministicSession {
    caps: AdapterCapabilities,
    identity: ModelIdentity,
    snapshot: Vec<u8>,
}
impl DeterministicSession {
    fn new() -> Self {
        Self {
            caps: AdapterCapabilities {
                adapter_id: "runtime-session".into(),
                backend: "external".into(),
                supports_prefill: true,
                supports_decode: true,
                supports_kv_restore: true,
                supports_streaming: false,
            },
            identity: ModelIdentity {
                model_digest: "0".repeat(64),
                tokenizer_digest: "0".repeat(64),
                architecture_digest: "0".repeat(64),
            },
            snapshot: vec![0x4b, 0x56],
        }
    }
}
impl RuntimeSession for DeterministicSession {
    fn capabilities(&self) -> &AdapterCapabilities {
        &self.caps
    }
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    fn generate(&mut self, request: GenerationRequest) -> Result<GenerationResult> {
        if request.input_tokens.is_empty() {
            bail!("input_tokens must not be empty");
        }
        Ok(GenerationResult {
            output_tokens: vec![request.input_tokens.last().copied().unwrap_or(0)],
            prompt_tokens: request.input_tokens.len(),
            finish_reason: "stop".into(),
            stats: GenerationStats {
                backend: Some("external".into()),
                ..GenerationStats::default()
            },
        })
    }
    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        Ok(self.snapshot.clone())
    }
    fn restore_kv(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes != self.snapshot {
            bail!("snapshot digest/content mismatch");
        }
        Ok(())
    }
    fn snapshot_receipt(&self) -> Result<CacheReceipt> {
        Ok(CacheReceipt {
            schema: "v1".into(),
            operation: "snapshot".into(),
            model_digest: self.identity.model_digest.clone(),
            tokenizer_digest: self.identity.tokenizer_digest.clone(),
            architecture_digest: self.identity.architecture_digest.clone(),
            prefix_digest: "0".repeat(64),
            generation: 0,
            reused_tokens: 0,
            restored_bytes: self.snapshot.len() as u64,
            mode: CacheMode::Exact,
        })
    }
    fn restore_receipt(&mut self, receipt: &CacheReceipt, bytes: &[u8]) -> Result<()> {
        if receipt.model_digest != self.identity.model_digest {
            bail!("model digest mismatch");
        }
        self.restore_kv(bytes)
    }
}

fn digest_file(path: &std::path::Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn load_session(model: Option<&std::path::Path>) -> Result<Box<dyn RuntimeSession>> {
    let Some(model) = model else {
        return Ok(Box::new(DeterministicSession::new()));
    };
    let mut config: Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    if config.get("max_context_tokens").is_none() {
        let max = config
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .context("model context missing")?;
        config["max_context_tokens"] = json!(max);
    }
    config["kvslim_cold_directory"] = Value::Null;
    let mut adapter = OpenClModelAdapter::default();
    adapter.load(model, &config)?;
    let identity = ModelIdentity {
        model_digest: digest_file(&model.join("config.json"))?,
        tokenizer_digest: digest_file(&model.join("tokenizer.json"))?,
        architecture_digest: digest_file(&model.join("config.json"))?,
    };
    Ok(Box::new(ModelAdapterSession::new(adapter, identity)))
}

fn serve_connection(
    mut stream: UnixStream,
    session: Box<dyn RuntimeSession>,
    session_id: &str,
) -> Result<()> {
    let mut host = RuntimeHost::new(session, session_id)?;
    loop {
        let request = match read_request(&mut stream) {
            Ok(request) => request,
            Err(error) if error.to_string().contains("failed to fill whole buffer") => break,
            Err(error) => return Err(error),
        };
        let response = host.dispatch(request);
        write_response(&mut stream, &response)?;
        if response.ok && matches!(response.payload.get("released"), Some(Value::Bool(true))) {
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut socket = PathBuf::from("/tmp/kvslim-runtime-host.sock");
    let mut session_id = String::from("runtime-session");
    let mut self_test = false;
    let mut model: Option<PathBuf> = None;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" => {
                i += 1;
                socket = PathBuf::from(args.get(i).context("--socket needs a path")?);
            }
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(
                    args.get(i).context("--model needs a directory")?,
                ));
            }
            "--session-id" => {
                i += 1;
                session_id = args.get(i).context("--session-id needs a value")?.clone();
            }
            "--self-test" => self_test = true,
            "--help" => {
                println!(
                    "kvslim-runtime-host --socket PATH [--model DIR] [--session-id ID] [--self-test]"
                );
                return Ok(());
            }
            other => bail!("unrecognized option {other}"),
        }
        i += 1;
    }
    if self_test {
        println!(
            "generic runtime host self-test requires a client; socket={}",
            socket.display()
        );
    }
    if socket.exists() {
        fs::remove_file(&socket).context("remove stale runtime socket")?;
    }
    let session = load_session(model.as_deref())?;
    let listener = UnixListener::bind(&socket).context("bind runtime IPC socket")?;
    println!("generic runtime host listening on {}", socket.display());
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(error) = serve_connection(stream, session, &session_id) {
                    eprintln!("connection failed: {error:#}");
                }
                break;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
