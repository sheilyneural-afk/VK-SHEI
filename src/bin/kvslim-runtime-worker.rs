#![recursion_limit = "256"]
//! Local model runtime worker behind the backend-neutral KVSLIM gateway.
//! This worker is an independent local runtime behind the KVSLIM gateway.
//! No PyTorch or LibTorch needed: Rust + local runtime executes all transformer
//! neural arithmetic, retaining original BF16 weights and BF16 KV in device RAM.

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use kvslim::{
    continuity_contract::ContinuityRequest,
    gpu_canonical_lod::CanonicalGpuLodConfig,
    inference_worker::{InferenceError, InferenceExecutor, InferenceWorker, SubmitError},
    model_config::ModelSpec,
    opencl_llm::{OpenClGeneration, OpenClLlm, WeightMode},
};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::AtomicBool,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::Tokenizer;

const MAX_BODY_BYTES: usize = 1_048_576;
const CHAT_HTML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/chat-runtime.html"
));

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default)]
    native: bool,
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    user: Option<String>,
    /// Optional provider-neutral continuity capability. The worker validates it
    /// fail-closed; physical resume/snapshot is only enabled once a backend
    /// implementation is attached.
    #[serde(default)]
    continuity: Option<ContinuityRequest>,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": {"message": self.1}}))).into_response()
    }
}

fn invalid(code: StatusCode, message: impl Into<String>) -> ApiError {
    ApiError(code, message.into())
}

fn validate(request: &ChatRequest, served_model: &str) -> Result<usize, ApiError> {
    if request.messages.is_empty() {
        return Err(invalid(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
        ));
    }
    let model = request.model.as_deref().unwrap_or(served_model);
    if model != served_model {
        return Err(invalid(
            StatusCode::NOT_FOUND,
            format!("model {model} is not served by this local runtime; use {served_model}"),
        ));
    }
    let supported_roles = ["system", "user", "assistant"];
    if request
        .messages
        .iter()
        .any(|msg| !supported_roles.contains(&msg.role.as_str()))
    {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "runtime chat supports only system/user/assistant text messages",
        ));
    }
    if request.native {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "native=true is not supported by this runtime contract",
        ));
    }
    if request.document.as_ref().is_some_and(|d| !d.is_empty()) {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "document prefix caching is not implemented for the OpenCL engine",
        ));
    }
    if let (Some(a), Some(b)) = (request.max_tokens, request.max_completion_tokens)
        && a != b
    {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "max_tokens conflicts with max_completion_tokens",
        ));
    }
    let max_tokens = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(64);
    if !(1..=512).contains(&max_tokens) {
        return Err(invalid(
            StatusCode::BAD_REQUEST,
            "max_tokens must be in 1..=512",
        ));
    }
    let mut unsupported = Vec::new();
    if request.temperature.is_some_and(|t| t != 0.0) {
        unsupported.push("temperature sampling");
    }
    if request.top_p.is_some_and(|p| p != 1.0) {
        unsupported.push("top_p sampling");
    }
    if request.stream {
        unsupported.push("stream=true (SSE)");
    }
    if request.n.is_some_and(|n| n != 1) {
        unsupported.push("n != 1");
    }
    if request.stop.as_ref().is_some_and(|x| !x.is_null()) {
        unsupported.push("stop");
    }
    if request.tools.as_ref().is_some_and(|x| !x.is_empty()) {
        unsupported.push("tools");
    }
    if request
        .tool_choice
        .as_ref()
        .is_some_and(|x| !x.is_null() && x != "none")
    {
        unsupported.push("tool_choice");
    }
    if request
        .response_format
        .as_ref()
        .is_some_and(|x| !x.is_null() && x.get("type").and_then(Value::as_str) != Some("text"))
    {
        unsupported.push("response_format");
    }
    if request.logprobs == Some(true) {
        unsupported.push("logprobs");
    }
    if request.frequency_penalty.is_some_and(|x| x != 0.0) {
        unsupported.push("frequency_penalty");
    }
    if request.presence_penalty.is_some_and(|x| x != 0.0) {
        unsupported.push("presence_penalty");
    }
    if request.seed.is_some() {
        unsupported.push("seed");
    }
    if !request.other.is_empty() {
        unsupported.push("unknown extra parameters");
    }
    if !unsupported.is_empty() {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("GPU engine cannot provide: {}", unsupported.join(", ")),
        ));
    }
    let _ = &request.user;
    Ok(max_tokens)
}

fn render_prompt(
    template_source: &str,
    messages: &[ChatMessage],
    thinking: bool,
) -> Result<String> {
    ensure!(!messages.is_empty(), "no messages");
    let mut env = Environment::new();
    env.add_template("chat", template_source)
        .context("invalid model chat template")?;
    let rendered = env.get_template("chat")?.render(context! {
        messages => messages.iter().map(|message| json!({
            "role": message.role,
            "content": message.content,
        })).collect::<Vec<_>>(),
        add_generation_prompt => true,
        enable_thinking => thinking,
        bos_token => "",
        eos_token => "",
    })?;
    ensure!(
        !rendered.is_empty(),
        "model chat template rendered an empty prompt"
    );
    Ok(rendered)
}

#[derive(Clone)]
struct GpuJob {
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    enable_thinking: bool,
}

struct GpuAnswer {
    text: String,
    generation: OpenClGeneration,
}

struct GpuExecutor {
    model: OpenClLlm,
    tokenizer: Tokenizer,
    chat_template: String,
}

impl InferenceExecutor for GpuExecutor {
    type Request = GpuJob;
    type Response = GpuAnswer;
    type Error = anyhow::Error;

    fn execute(&mut self, request: GpuJob, cancel: &AtomicBool) -> Result<GpuAnswer> {
        // Keep the verified BF16 prompt snapshot across turns. Reset only the
        // active execution state after a completed request; reset_session()
        // deliberately preserves the bounded SHA-256 verified replay cache.
        // On the first request the runtime is already fresh, so no reset or
        // device barrier is necessary.
        if self.model.position() != 0 {
            self.model.reset_session()?;
        }
        let text = render_prompt(
            &self.chat_template,
            &request.messages,
            request.enable_thinking,
        )?;
        let input = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let ids = input.get_ids();
        ensure!(!ids.is_empty(), "empty tokenized prompt");
        let generated = self
            .model
            .generate_cancellable(ids, request.max_tokens, false, cancel)?;
        let output = self
            .tokenizer
            .decode(&generated.tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        Ok(GpuAnswer {
            text: output,
            generation: generated,
        })
    }
}

#[derive(Clone)]
struct AppState {
    worker: InferenceWorker<GpuJob, GpuAnswer, anyhow::Error>,
    model_id: String,
    model_dir: String,
    gpu_device: String,
    load_seconds: f64,
    max_context_tokens: usize,
    weight_mode: &'static str,
    lod_epsilon: Option<f64>,
    lod_stage: Option<usize>,
    gpu_micro_axis_sigma: Option<f32>,
    gpu_micro_axis_warmup: Option<usize>,
    gpu_micro_axis_safety_fuse: Option<usize>,
    gpu_kv_capacity_bytes: usize,
    gpu_prefill_batch_tokens: Option<usize>,
    gpu_prefill_weight_reuse_tokens: usize,
    gpu_hot_v_capacity_tokens: usize,
    gpu_verified_prompt_replay_max_tokens: usize,
    gpu_verified_shared_prefix_enabled: bool,
    gpu_micro_hot_v_capacity_tokens: usize,
    gpu_decode_unroll4: bool,
    gpu_deferred_cold_enabled: bool,
    gpu_decode_fused_kernels: bool,
    gpu_exact_hot_fused_attention: bool,
    timeout: Duration,
}

async fn index() -> Html<&'static str> {
    Html(CHAT_HTML)
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status":"ready",
        "engine":"kvslim-engine",
        "model":state.model_id,
        "model_directory": state.model_dir,
        "running_backend": "rust-opencl-runtime",
        "device":state.gpu_device,
        "weight_mode":state.weight_mode,
        "load_seconds":state.load_seconds,
        "model_weights":"original_BF16_unchanged",
        "neural_arithmetic":"measured-device-kernels-with-declared-precision",
        "full_llm_layers_computed_on_gpu":true,
        "kvslim_lod_active":state.lod_epsilon.is_some(),
        "canonical_cpu_kvslim_lod_components_active":state.lod_epsilon.is_some(),
        "canonical_cpu_lod_modules":["adaptive_extent::ParallelContextAxis",
            "cold_store::ColdStore","attention_certificate::select_by_exact_mass"],
        "gpu_lod_epsilon":state.lod_epsilon,
        "gpu_micro_axis_novelty_sigma":state.gpu_micro_axis_sigma,
        "gpu_micro_axis_warmup":state.gpu_micro_axis_warmup,
        "gpu_micro_axis_open_safety_fuse":state.gpu_micro_axis_safety_fuse,
        "gpu_micro_metadata_region_pruning_active":state.lod_epsilon.is_some_and(|e|e>0.0),
        "gpu_micro_metadata_regions":"original CPU ParallelContextAxis communicating_dimensions and 4-bit LSH; original BF16 K bounding boxes and conservative omitted-mass certificate",
        "gpu_micro_epsilon_zero":"disabled: all BF16 K/V retained with exact scores",
        "gpu_lod_stage_tokens":state.lod_stage,
        "gpu_kv_capacity_bytes":state.gpu_kv_capacity_bytes,
        "gpu_prefill_batch_tokens":state.gpu_prefill_batch_tokens,
        "gpu_prefill_weight_reuse_tokens":state.gpu_prefill_weight_reuse_tokens,
        "gpu_hot_v_capacity_tokens":state.gpu_hot_v_capacity_tokens,
        "gpu_verified_prompt_replay_max_tokens":state.gpu_verified_prompt_replay_max_tokens,
        "gpu_verified_shared_prefix_enabled":state.gpu_verified_shared_prefix_enabled,
        "gpu_verified_shared_prefix_policy":"enabled when exact canonical KV is configured: identical tokenizer-ID prefix, SHA256 original BF16 K/V, CPU ColdStore/4-bit index restored; only the causal suffix is physically executed",
        "gpu_verified_prompt_replay_policy":"bounded RAM-only exact replay: identical tokenizer IDs and SHA256 original BF16 K/V required before ColdStore/four-bit rehydration; mismatches recompute full prefill",
        "gpu_micro_hot_v_capacity_tokens":state.gpu_micro_hot_v_capacity_tokens,
        "gpu_micro_hot_policy":"Optional positive epsilon only: GPU resident BF16 V and cached certified token IDs, original CPU ColdStore BF16 commits preserved; full BF16 GPU fallback when no K-region certificate",
        "gpu_decode_unroll4":state.gpu_decode_unroll4,
        "gpu_deferred_cold_enabled":state.gpu_deferred_cold_enabled,
        "gpu_decode_fused_kernels":state.gpu_decode_fused_kernels,
        "gpu_exact_hot_fused_attention":state.gpu_exact_hot_fused_attention,
        "gpu_prefill_backend":if state.gpu_prefill_batch_tokens.is_some(){
            "batched-device-prefill"
        }else{"sequential-device-prefill"},
        "gpu_kv_cache":if state.lod_epsilon.is_some() {
            "exact-device-kv-with-cold-store-authority"
        } else {"dense-device-kv"},
        "max_context_tokens":state.max_context_tokens,
        "no_libtorch":true,
        "exact_dense_logits_parity_proven":false
    }))
}

async fn ready(State(_state): State<AppState>) -> Json<Value> {
    Json(json!({"status":"ready","workers":1,"model_loaded":true}))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"object":"list","data":[
        {"id":state.model_id,"object":"model","created":0,"owned_by":"kvslim-local-runtime"}
    ]}))
}

async fn capabilities(State(state): State<AppState>) -> Json<Value> {
    let model_path = std::path::Path::new(&state.model_dir);
    let model_digest = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    let calibration_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("deploy/local-production/runtime/calibration")
        .join(format!("{model_digest}.json"));
    let calibration = std::fs::read(&calibration_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|value| {
            value.get("schema").and_then(Value::as_str) == Some("kvslim.calibration_receipt/v2")
                && value.get("status").and_then(Value::as_str)
                    == Some("measured-sustained-execution")
        });
    Json(json!({
        "engine":"kvslim-engine",
        "model_checkpoint":state.model_dir,
        "model_digest":model_digest,
        "calibration":calibration,
        "model":state.model_id,"gpu_device":state.gpu_device,
        "gpu_model_loaded":true,"gpu_full_llm_inference":true,
        "compute_precision":"FP32 accumulations, BF16 model weights and KV",
        "models_verified":[],
        "kvslim_lod_active":state.lod_epsilon.is_some(),
        "canonical_cpu_lod_components_active":state.lod_epsilon.is_some(),
        "canonical_lod_epsilon":state.lod_epsilon,
        "gpu_micro_axis_novelty_sigma":state.gpu_micro_axis_sigma,
        "gpu_micro_axis_warmup":state.gpu_micro_axis_warmup,
        "gpu_micro_axis_open_safety_fuse":state.gpu_micro_axis_safety_fuse,
        "micro_metadata_sparse_qk_opt_in":state.lod_epsilon.is_some_and(|e|e>0.0),
        "micro_metadata_scope":"original CPU 4bit/LSH and communicating_dimensions prioritize CPU semantic regions. BF16 K min/max and omitted softmax mass bound certify possible QK/V avoidance. Ambiguous boxes fall back to original full GPU QK and ColdStore. Exact epsilon=0 never prunes",
        "micro_metadata_memory":"Bounded 512 regions per layer, disables sparse metadata on fragmentation; exact K/V always in original BF16 storage",
        "canonical_lod_stage_tokens":state.lod_stage,
        "gpu_kv_capacity_bytes":state.gpu_kv_capacity_bytes,
        "gpu_prefill_batch_tokens":state.gpu_prefill_batch_tokens,
        "gpu_prefill_weight_reuse_tokens":state.gpu_prefill_weight_reuse_tokens,
        "gpu_hot_v_capacity_tokens":state.gpu_hot_v_capacity_tokens,
        "gpu_verified_prompt_replay_max_tokens":state.gpu_verified_prompt_replay_max_tokens,
        "gpu_verified_shared_prefix_enabled":state.gpu_verified_shared_prefix_enabled,
        "gpu_verified_shared_prefix_policy":"enabled when exact canonical KV is configured: identical tokenizer-ID prefix, SHA256 original BF16 K/V, CPU ColdStore/4-bit index restored; only the causal suffix is physically executed",
        "gpu_verified_prompt_replay_policy":"bounded RAM-only exact replay: identical tokenizer IDs and SHA256 original BF16 K/V required before ColdStore/four-bit rehydration; mismatches recompute full prefill",
        "gpu_micro_hot_v_capacity_tokens":state.gpu_micro_hot_v_capacity_tokens,
        "gpu_micro_hot_policy":"Optional positive epsilon only: GPU resident BF16 V and cached certified token IDs, original CPU ColdStore BF16 commits preserved; full BF16 GPU fallback when no K-region certificate",
        "gpu_decode_unroll4":state.gpu_decode_unroll4,
        "gpu_deferred_cold_enabled":state.gpu_deferred_cold_enabled,
        "gpu_decode_fused_kernels":state.gpu_decode_fused_kernels,
        "gpu_exact_hot_fused_attention":state.gpu_exact_hot_fused_attention,
        "gpu_exact_hot_fused_scope":"Single exact QK-softmax-PV BF16 GPU kernel over ALL K/V, cold store and CPU unchanged, opt-in with hot V <=2048",
        "gpu_deferred_cold_policy":"exact BF16 K/V from GPU committed to original CPU ColdStore BEFORE response, only for epsilon zero and whole turn within bounded hot V",
        "gpu_hot_v_policy":"Canonical ColdStore V exact always; bounded BF16 hot tier only at epsilon=0",
        "gpu_prefill_cold_pages":"batched up to 64 BF16 tokens/page using original ColdStore",
        "gpu_prefill_temporary_v_residency":"released after cold seal; no permanent extra GPU V cache",
         "canonical_lod_scope":"hierarchical best-first Grid/Chunk/Page selection over original CPU index/LSH and micro-links; ColdStore and conservative BF16-box omitted-mass certificates authorize positive-epsilon pruning; exact epsilon=0 remains dense",
        "parity_with_cpu":null,
        "server_backend":"local accelerator runtime",
        "max_context_tokens":state.max_context_tokens,
        "max_output_tokens":512,
        "API":{"OpenAI_compatible_subset":true,"stream":false,
               "tools":false,"sampling":false,"temperature":0}
    }))
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<Value>, ApiError> {
    let max_tokens = validate(&request, &state.model_id)?;
    if let Some(continuity) = request.continuity.as_ref() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| invalid(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .as_secs();
        continuity
            .validate_at(now)
            .map_err(|e| invalid(StatusCode::BAD_REQUEST, e.to_string()))?;
        return Err(invalid(
            StatusCode::NOT_IMPLEMENTED,
            "continuity contract is validated, but this worker build has no physical session store attached; refusing to pretend resume/snapshot/release succeeded",
        ));
    }
    let response = state
        .worker
        .try_submit(GpuJob {
            messages: request.messages,
            max_tokens,
            enable_thinking: request.enable_thinking,
        })
        .map_err(|err| match err {
            SubmitError::QueueFull => invalid(
                StatusCode::TOO_MANY_REQUESTS,
                "local runtime worker queue is full",
            ),
            SubmitError::WorkerStopped => invalid(
                StatusCode::SERVICE_UNAVAILABLE,
                "local runtime worker is unavailable",
            ),
        })?;
    let result = tokio::time::timeout(state.timeout, response)
        .await
        .map_err(|_| {
            invalid(
                StatusCode::GATEWAY_TIMEOUT,
                "local runtime inference request deadline expired",
            )
        })?
        .map_err(|err| match err {
            InferenceError::Executor(error) => {
                let msg = error.to_string();
                if msg.contains("cancelled") {
                    invalid(StatusCode::REQUEST_TIMEOUT, msg)
                } else if msg.contains("capacity") || msg.contains("context") {
                    invalid(StatusCode::BAD_REQUEST, msg)
                } else {
                    invalid(StatusCode::INTERNAL_SERVER_ERROR, msg)
                }
            }
            InferenceError::Cancelled => invalid(StatusCode::REQUEST_TIMEOUT, "cancelled"),
            InferenceError::ExecutorPanicked | InferenceError::WorkerStopped => invalid(
                StatusCode::SERVICE_UNAVAILABLE,
                "local runtime worker stopped",
            ),
        })?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| invalid(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .as_secs();
    let generation = result.generation;
    let finish_reason = if generation.tokens.len() >= max_tokens {
        "length"
    } else {
        "stop"
    };
    Ok(Json(json!({
        "id":format!("chatcmpl-kvslim-igpu-{created}"),
        "object":"chat.completion",
        "created":created,
        "model":state.model_id,
        "choices":[{"index":0,"message":{"role":"assistant","content":result.text},
            "finish_reason":finish_reason}],
        "usage":{
            "prompt_tokens":generation.prompt_tokens,
            "completion_tokens":generation.tokens.len(),
            "total_tokens":generation.prompt_tokens+generation.tokens.len(),
            "prefill_seconds":generation.prefill_seconds,
            "latency_seconds":generation.total_seconds,
            "gpu_model_load_seconds":generation.model_load_seconds,
            "engine":"local-accelerator-runtime",
            "gpu_weight_bytes":generation.gpu_weight_payload_bytes,
            "gpu_device":generation.device_name,
            "kvslim_lod_active":generation.canonical_cpu_lod_components_active,
            "canonical_cpu_lod_components_active":generation.canonical_cpu_lod_components_active,
            "canonical_cpu_lod_stats":generation.canonical_cpu_lod_stats,
            "gpu_kv_capacity_bytes":generation.gpu_kv_capacity_bytes,
            "gpu_prefill_batch_tokens":generation.gpu_prefill_batch_tokens,
            "gpu_prefill_weight_reuse_tokens":generation.gpu_prefill_weight_reuse_tokens,
            "gpu_hot_v_capacity_tokens":generation.gpu_hot_v_capacity_tokens,
            "gpu_verified_prompt_replay_max_tokens":generation.gpu_verified_prompt_replay_max_tokens,
            "gpu_verified_prefix_replay_enabled":generation.gpu_verified_prefix_replay_enabled,
            "gpu_verified_prefix_reused_tokens":generation.gpu_verified_prefix_reused_tokens,
            "gpu_verified_prefix_restored_bytes":generation.gpu_verified_prefix_restored_bytes,
            "gpu_verified_prompt_replay_hit":generation.gpu_verified_prompt_replay_hit,
            "gpu_verified_prompt_replay_cached_tokens":generation.gpu_verified_prompt_replay_cached_tokens,
            "gpu_verified_prompt_replay_cached_bytes":generation.gpu_verified_prompt_replay_cached_bytes,
            "gpu_verified_prompt_replay_restored_bytes":generation.gpu_verified_prompt_replay_restored_bytes,
            "gpu_verified_prompt_replay_sha256_verified":generation.gpu_verified_prompt_replay_sha256_verified,
            "gpu_micro_hot_v_capacity_tokens":generation.gpu_micro_hot_v_capacity_tokens,
            "gpu_micro_hot_sparse_calls":generation.gpu_micro_hot_sparse_calls,
            "gpu_micro_hot_exact_fallback_calls":generation.gpu_micro_hot_exact_fallback_calls,
            "gpu_micro_selection_upload_calls":generation.gpu_micro_selection_upload_calls,
            "gpu_micro_selection_cache_hits":generation.gpu_micro_selection_cache_hits,
            "gpu_hot_v_attention_calls":generation.gpu_hot_v_attention_calls,
            "gpu_hot_v_cold_fallback_calls":generation.gpu_hot_v_cold_fallback_calls,
            "gpu_hot_v_prefetch_bytes":generation.gpu_hot_v_prefetch_bytes,
            "gpu_decode_unroll4":generation.gpu_decode_unroll4,
            "gpu_decode_fused_kernels":generation.gpu_decode_fused_kernels,
            "gpu_exact_hot_fused_attention":generation.gpu_exact_hot_fused_attention,
            "gpu_deferred_cold_enabled":generation.gpu_deferred_cold_enabled,
            "gpu_deferred_cold_activated":generation.gpu_deferred_cold_activated,
            "gpu_deferred_cold_commit_seconds":generation.gpu_deferred_cold_commit_seconds,
            "gpu_deferred_cold_committed_token_layers":generation.gpu_deferred_cold_committed_token_layers,
            "gpu_prefill_batches":generation.gpu_prefill_batches,
            "gpu_prefill_cold_seal_seconds":generation.gpu_prefill_cold_seal_seconds,
            "gpu_prefill_temporary_v_capacity_bytes":generation.gpu_prefill_temporary_v_capacity_bytes,
            "torch_used":false
        }
    })))
}

fn validate_model_bundle(model: &std::path::Path) -> Result<String> {
    let _spec = ModelSpec::inspect(model)?;
    let config: Value = serde_json::from_slice(
        &std::fs::read(model.join("tokenizer_config.json"))
            .context("tokenizer_config.json is required for message rendering")?,
    )?;
    let template = config
        .get("chat_template")
        .and_then(Value::as_str)
        .context("tokenizer_config.json has no chat_template")?;
    ensure!(!template.trim().is_empty(), "model chat_template is empty");
    Ok(template.to_owned())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut model = PathBuf::from("deploy/local-production/runtime/models/active");
    let mut host = String::from("127.0.0.1");
    let mut port = 8188u16;
    let mut max_context = 2048usize;
    let mut timeout_secs = 300u64;
    let mut mode = WeightMode::Resident;
    // Exact canonical KV is the production default: BF16 pages remain the
    // authority and enable verified prefix replay between independent turns.
    // Positive-epsilon pruning is still explicit; --gpu-dense-kv remains the
    // low-overhead escape hatch for short, stateless workloads.
    let mut lod: Option<CanonicalGpuLodConfig> = Some(CanonicalGpuLodConfig {
        epsilon: 0.0,
        ..CanonicalGpuLodConfig::default()
    });
    let mut micro_sigma: Option<f32> = None;
    let mut micro_warmup: Option<usize> = None;
    let mut micro_safety_fuse: Option<usize> = None;
    // AMD Phoenix: 128-token layer-major causal BF16 prefill benchmarked
    // and verified bit-exact against the previous OpenCL dense/LOD engine.
    // Explicit fallback keeps the conservative execution path available.
    let mut prefill_batch_tokens = Some(128usize);
    // Safe exact optimizations are enabled coherently. They preserve original
    // BF16 K/V, retain a bounded verified replay snapshot in host RAM, and
    // fail closed to physical prefill on a token/digest mismatch. Hardware-
    // specific GEMV unrolling stays opt-in until calibration proves a win.
    let mut prefill_weight_reuse = 1usize;
    let mut hot_v_tokens = 512usize;
    let mut verified_prompt_replay_tokens = 512usize;
    let mut verified_shared_prefix = true;
    let mut micro_hot_v_tokens = 0usize;
    let mut decode_unroll4 = false;
    let mut deferred_cold = true;
    let mut fused_decode = false;
    let mut exact_hot_fused = true;
    let mut allow_network = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().context("missing --model")?.into(),
            "--host" => host = args.next().context("missing --host")?,
            "--port" => port = args.next().context("missing --port")?.parse()?,
            "--max-context-tokens" => {
                max_context = args
                    .next()
                    .context("missing --max-context-tokens")?
                    .parse()?
            }
            "--timeout-secs" => {
                timeout_secs = args.next().context("missing --timeout-secs")?.parse()?
            }
            "--stream-weights" => mode = WeightMode::Stream,
            "--resident" => mode = WeightMode::Resident,
            "--gpu-prefill-batch" => {
                prefill_batch_tokens = Some(
                    args.next()
                        .context("missing --gpu-prefill-batch")?
                        .parse()?,
                );
            }
            "--gpu-prefill-fallback" => {
                prefill_batch_tokens = None;
            }
            "--gpu-prefill-weight-reuse" => {
                prefill_weight_reuse = args
                    .next()
                    .context("missing --gpu-prefill-weight-reuse")?
                    .parse()?;
            }
            "--gpu-hot-v-tokens" => {
                hot_v_tokens = args.next().context("missing --gpu-hot-v-tokens")?.parse()?;
            }
            "--gpu-verified-prompt-replay" => {
                verified_prompt_replay_tokens = args
                    .next()
                    .context("missing --gpu-verified-prompt-replay max tokens")?
                    .parse()?;
            }
            "--gpu-no-prompt-replay" => {
                verified_prompt_replay_tokens = 0;
            }
            "--gpu-verified-shared-prefix" => {
                verified_shared_prefix = true;
            }
            "--gpu-no-shared-prefix" => {
                verified_shared_prefix = false;
            }
            "--gpu-micro-hot-v-tokens" => {
                micro_hot_v_tokens = args
                    .next()
                    .context("missing --gpu-micro-hot-v-tokens")?
                    .parse()?;
            }
            "--gpu-decode-unroll4" => {
                decode_unroll4 = true;
            }
            "--gpu-decode-original" => {
                decode_unroll4 = false;
            }
            "--gpu-deferred-cold" => {
                deferred_cold = true;
            }
            "--gpu-immediate-cold" => {
                deferred_cold = false;
            }
            "--gpu-fused-decode" => {
                fused_decode = true;
            }
            "--gpu-exact-fused-attention" => {
                exact_hot_fused = true;
            }
            "--gpu-exact-original-attention" => {
                exact_hot_fused = false;
            }
            "--gpu-unfused-decode" => {
                fused_decode = false;
            }
            "--gpu-lod" => {
                lod.get_or_insert_with(CanonicalGpuLodConfig::default);
            }
            "--gpu-dense-kv" => {
                lod = None;
            }
            "--gpu-micro-novelty-sigma" => {
                micro_sigma = Some(
                    args.next()
                        .context("missing --gpu-micro-novelty-sigma")?
                        .parse()?,
                );
            }
            "--gpu-micro-warmup" => {
                micro_warmup = Some(args.next().context("missing --gpu-micro-warmup")?.parse()?);
            }
            "--gpu-micro-open-fuse" => {
                micro_safety_fuse = Some(
                    args.next()
                        .context("missing --gpu-micro-open-fuse")?
                        .parse()?,
                );
            }
            "--lod-epsilon" => {
                let e = args.next().context("missing --lod-epsilon")?.parse()?;
                lod.get_or_insert_with(CanonicalGpuLodConfig::default)
                    .epsilon = e;
            }
            "--lod-stage" => {
                let n = args.next().context("missing --lod-stage")?.parse()?;
                lod.get_or_insert_with(CanonicalGpuLodConfig::default)
                    .stage_tokens = n;
            }
            "--lod-cold-dir" => {
                let p = args.next().context("missing --lod-cold-dir")?.into();
                lod.get_or_insert_with(CanonicalGpuLodConfig::default)
                    .cold_directory = Some(p);
            }
            "--allow-network" => allow_network = true,
            "--help" | "-h" => {
                println!("KVSLIM local runtime worker (local model runtime)");
                println!(
                    "--model DIR --host 127.0.0.1 --port 8188 --max-context-tokens 2048 --timeout-secs 300 --resident|--stream-weights --gpu-lod|--gpu-dense-kv --lod-epsilon 0 --lod-stage 64 --gpu-prefill-batch 128|--gpu-prefill-fallback --gpu-prefill-weight-reuse 8 --gpu-hot-v-tokens 512 --gpu-decode-unroll4|--gpu-decode-original --gpu-deferred-cold|--gpu-immediate-cold --gpu-fused-decode|--gpu-unfused-decode"
                );
                println!(
                    "No direct authentication: bind loopback only; use the authenticated gateway for clients."
                );
                return Ok(());
            }
            other => bail!("unrecognized option {other}"),
        }
    }
    if host != "127.0.0.1" && host != "localhost" && host != "::1" && !allow_network {
        bail!("GPU server has NO authentication: remote bind requires --allow-network");
    }
    ensure!(
        (1..=32768).contains(&max_context),
        "invalid GPU max context"
    );
    hot_v_tokens = hot_v_tokens.min(max_context);
    verified_prompt_replay_tokens = verified_prompt_replay_tokens.min(hot_v_tokens);
    if lod.is_none() {
        // Replay and deferred commit require the exact canonical ColdStore.
        // An explicit dense route must never advertise or attempt them.
        verified_prompt_replay_tokens = 0;
        verified_shared_prefix = false;
        deferred_cold = false;
    }
    if verified_prompt_replay_tokens < 32 {
        verified_shared_prefix = false;
    }
    let chat_template = validate_model_bundle(&model)?;
    let model_id = model
        .file_name()
        .and_then(|value| value.to_str())
        .context("model path must have a stable directory name")?
        .to_owned();
    let started = Instant::now();
    let mut engine = if let Some(config) = lod {
        OpenClLlm::load_with_canonical_lod(&model, max_context, mode, config)?
    } else {
        OpenClLlm::load(&model, max_context, mode)?
    };
    if micro_sigma.is_some() || micro_warmup.is_some() || micro_safety_fuse.is_some() {
        let mut policy = kvslim::adaptive_extent::AxisPolicy::default();
        if let Some(value) = micro_sigma {
            policy.novelty_sigma = value;
        }
        if let Some(value) = micro_warmup {
            policy.warmup_observations = value;
        }
        if let Some(value) = micro_safety_fuse {
            policy.open_extent_safety_limit = Some(value);
        }
        engine.set_gpu_micro_axis_policy(policy)?;
    }
    let gpu_micro_axis_sigma = engine.gpu_micro_axis_policy().map(|p| p.novelty_sigma);
    let gpu_micro_axis_warmup = engine
        .gpu_micro_axis_policy()
        .map(|p| p.warmup_observations);
    let gpu_micro_axis_safety_fuse = engine
        .gpu_micro_axis_policy()
        .and_then(|p| p.open_extent_safety_limit);
    engine.set_batch_prefill_tokens(prefill_batch_tokens)?;
    engine.set_prefill_weight_reuse_tokens(prefill_weight_reuse)?;
    engine.set_exact_hot_v_tokens(hot_v_tokens)?;
    engine.set_verified_gpu_prompt_replay(verified_prompt_replay_tokens)?;
    engine.set_verified_gpu_prefix_replay(verified_shared_prefix)?;
    engine.set_micro_hot_v_tokens(micro_hot_v_tokens)?;
    engine.set_decode_unroll4(decode_unroll4)?;
    engine.set_deferred_cold_commit(deferred_cold)?;
    engine.set_decode_fused_kernels(fused_decode)?;
    engine.set_exact_hot_fused_attention(exact_hot_fused)?;
    let gpu_exact_hot_fused_attention = engine.exact_hot_fused_attention_enabled();
    let gpu_deferred_cold_enabled = engine.deferred_cold_enabled();
    let gpu_decode_fused_kernels = engine.decode_fused_kernels();
    let gpu_prefill_batch_tokens = engine.batch_prefill_tokens();
    let gpu_prefill_weight_reuse_tokens = engine.prefill_weight_reuse_tokens();
    let gpu_hot_v_capacity_tokens = engine.exact_hot_v_capacity_tokens();
    let gpu_verified_prompt_replay_max_tokens = engine.verified_gpu_prompt_replay_max_tokens();
    let gpu_verified_shared_prefix_enabled = engine.verified_gpu_prefix_replay_enabled();
    let gpu_micro_hot_v_capacity_tokens = engine.micro_hot_v_capacity_tokens();
    let gpu_decode_unroll4 = engine.decode_unroll4();
    let lod_epsilon = engine.canonical_lod_config().map(|x| x.epsilon);
    let lod_stage = engine.canonical_lod_config().map(|x| x.stage_tokens);
    let gpu_kv_capacity_bytes = engine.canonical_lod_gpu_kv_capacity_bytes();
    let gpu_device = engine.device_name().to_owned();
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let worker = InferenceWorker::spawn(
        GpuExecutor {
            model: engine,
            tokenizer,
            chat_template,
        },
        4,
    )?;
    let state = AppState {
        worker,
        model_id,
        model_dir: model.display().to_string(),
        gpu_device,
        load_seconds: started.elapsed().as_secs_f64(),
        max_context_tokens: max_context,
        weight_mode: if mode == WeightMode::Resident {
            "Resident"
        } else {
            "Stream"
        },
        lod_epsilon,
        lod_stage,
        gpu_micro_axis_sigma,
        gpu_micro_axis_warmup,
        gpu_micro_axis_safety_fuse,
        gpu_kv_capacity_bytes,
        gpu_prefill_batch_tokens,
        gpu_prefill_weight_reuse_tokens,
        gpu_hot_v_capacity_tokens,
        gpu_verified_prompt_replay_max_tokens,
        gpu_verified_shared_prefix_enabled,
        gpu_micro_hot_v_capacity_tokens,
        gpu_decode_unroll4,
        gpu_deferred_cold_enabled,
        gpu_decode_fused_kernels,
        gpu_exact_hot_fused_attention,
        timeout: Duration::from_secs(timeout_secs.max(1)),
    };
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);
    println!("KVSLIM local runtime listo en http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(value: Value) -> ChatRequest {
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn runtime_accepts_supported_architectures_when_template_is_declared() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 1536,
            "intermediate_size": 4608,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 130560,
            "rope_theta": 5000000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072
        });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages[0]['content'] }}"}"#,
        )
        .unwrap();
        assert!(validate_model_bundle(tmp.path()).is_ok());
        let mut llama = config;
        llama["hidden_size"] = json!(4096);
        llama["num_attention_heads"] = json!(32);
        llama["num_key_value_heads"] = json!(8);
        llama["num_hidden_layers"] = json!(32);
        llama["head_dim"] = json!(128);
        llama["vocab_size"] = json!(128256);
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_vec(&llama).unwrap(),
        )
        .unwrap();
        assert!(validate_model_bundle(tmp.path()).is_ok());
    }

    #[test]
    fn gpu_browser_request_is_valid() {
        let req = parse(json!({"model":"model-a",
            "messages":[{"role":"user","content":"Hola"}],
            "temperature":0,"stream":false,"max_tokens":2}));
        assert_eq!(validate(&req, "model-a").unwrap(), 2);
    }
    #[test]
    fn gpu_api_does_not_silently_accept_unimplemented_modes() {
        for param in [
            json!({"stream":true}),
            json!({"temperature":0.8}),
            json!({"tools":[{"type":"function"}]}),
            json!({"native":true}),
            json!({"foo":"bar"}),
            json!({"response_format":{"type":"json_object"}}),
        ] {
            let mut request =
                json!({"model":"model-a","messages":[{"role":"user","content":"Hola"}]});
            request
                .as_object_mut()
                .unwrap()
                .extend(param.as_object().unwrap().clone());
            assert!(validate(&parse(request), "model-a").is_err());
        }
    }
    #[test]
    fn exact_raw_template_matches_reference() {
        let messages = [ChatMessage {
            role: "user".into(),
            content: "Hola".into(),
        }];
        assert_eq!(
            render_prompt(
                "{{ messages[0]['role'] }}: {{ messages[0]['content'] }}",
                &messages,
                false
            )
            .unwrap(),
            "user: Hola"
        );
    }
}
