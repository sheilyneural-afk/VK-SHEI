//! Canonical KVSLIM local gateway.
//!
//! This binary is the public entry point. Runtime workers are selected from
//! measured capabilities and readiness; the gateway itself does not own model
//! weights or KV state.

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant},
};

const INDEX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/chat-unified.html"
));
const CPU_URL: &str = "http://127.0.0.1:8088";
const GPU_URL: &str = "http://127.0.0.1:8188";
const CPU_MODEL: &str = "kvslim-rust";
const DENSE_MODEL: &str = "minicpm-dense-bf16";
const GPU_MODEL: &str = "kvslim-opencl-bf16";
const AUTO_MODEL: &str = "kvslim-auto";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hardware {
    Cpu,
    Gpu,
    Auto,
}
impl Hardware {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Auto => "auto",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "gpu" | "igpu" => Some(Self::Gpu),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}
#[derive(Debug, Deserialize)]
struct Incoming {
    messages: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    hardware: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    native: bool,
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    user: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":{"message":self.1}}))).into_response()
    }
}
fn invalid(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, msg.into())
}

#[derive(Debug, Clone, Copy)]
struct Route {
    selected: Hardware,
    requested: Hardware,
    fallback: bool,
    dense: bool,
}

fn parse_input(req: &Incoming) -> Result<(Hardware, bool, usize), ApiError> {
    if req.messages.is_empty() || req.messages.len() > 256 {
        return Err(invalid(
            StatusCode::BAD_REQUEST,
            "messages debe contener 1..=256 elementos",
        ));
    }
    for (i, msg) in req.messages.iter().enumerate() {
        let role = msg.get("role").and_then(Value::as_str);
        let content = msg.get("content").and_then(Value::as_str);
        if !matches!(role, Some("system" | "assistant" | "user")) || content.is_none() {
            return Err(invalid(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("messages[{i}] requiere role válido y content de texto"),
            ));
        }
    }
    let model = req.model.as_deref();
    let requested_by_model = match model {
        None | Some(AUTO_MODEL) => None,
        Some(CPU_MODEL) | Some(DENSE_MODEL) => Some(Hardware::Cpu),
        Some(GPU_MODEL) => Some(Hardware::Gpu),
        Some(other) => {
            return Err(invalid(
                StatusCode::NOT_FOUND,
                format!("Modelo desconocido: {other}"),
            ));
        }
    };
    let requested_by_hardware = match req.hardware.as_deref() {
        None => None,
        Some(s) => Some(Hardware::parse(s).ok_or_else(|| {
            invalid(
                StatusCode::UNPROCESSABLE_ENTITY,
                "hardware: cpu, gpu o auto",
            )
        })?),
    };
    if let (Some(a), Some(b)) = (requested_by_model, requested_by_hardware)
        && a != b
    {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model y hardware solicitan dispositivos distintos",
        ));
    }
    let requested = requested_by_model
        .or(requested_by_hardware)
        .unwrap_or(Hardware::Auto);
    let dense = req.native || model == Some(DENSE_MODEL);
    if dense && requested == Hardware::Gpu {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "La ruta densa BF16 solo existe en CPU",
        ));
    }
    if dense && model == Some(CPU_MODEL) {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model=kvslim-rust incompatible con native=true",
        ));
    }
    let document = req
        .document
        .as_deref()
        .is_some_and(|d| !d.trim().is_empty());
    if document && (dense || requested == Hardware::Gpu) {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Los documentos con prefijo KV solo se pueden usar con CPU KVSLIM",
        ));
    }
    if let (Some(a), Some(b)) = (req.max_tokens, req.max_completion_tokens)
        && a != b
    {
        return Err(invalid(
            StatusCode::BAD_REQUEST,
            "Límites de tokens contradictorios",
        ));
    }
    let max = req.max_tokens.or(req.max_completion_tokens).unwrap_or(64);
    if !(1..=512).contains(&max) {
        return Err(invalid(
            StatusCode::BAD_REQUEST,
            "max_tokens debe ser 1..=512",
        ));
    }
    if req.stream
        || req.temperature.is_some_and(|t| t != 0.0)
        || req.top_p.is_some_and(|p| p != 1.0)
        || req.n.is_some_and(|n| n != 1)
        || req.tools.as_ref().is_some_and(|t| !t.is_empty())
        || req.stop.as_ref().is_some_and(|s| !s.is_null())
        || req
            .response_format
            .as_ref()
            .is_some_and(|f| !f.is_null() && f.get("type").and_then(Value::as_str) != Some("text"))
        || !req.extra.is_empty()
    {
        return Err(invalid(
            StatusCode::UNPROCESSABLE_ENTITY,
            "La API unificada implementa greedy, una respuesta, sin SSE/tools/sampling; parámetros incompatibles",
        ));
    }
    let _ = &req.user;
    Ok((requested, dense, max))
}

fn choose_mode(
    requested: Hardware,
    prefer: Hardware,
    cpu_ok: bool,
    gpu_ok: bool,
    requires_cpu: bool,
    dense: bool,
) -> Result<Route, ApiError> {
    let selected = match requested {
        Hardware::Cpu if cpu_ok => Hardware::Cpu,
        Hardware::Cpu => {
            return Err(invalid(
                StatusCode::SERVICE_UNAVAILABLE,
                "CPU solicitada, pero el backend CPU no responde; sin fallback",
            ));
        }
        Hardware::Gpu if gpu_ok => Hardware::Gpu,
        Hardware::Gpu => {
            return Err(invalid(
                StatusCode::SERVICE_UNAVAILABLE,
                "GPU solicitada, pero OpenCL no responde; sin fallback",
            ));
        }
        Hardware::Auto if requires_cpu || dense => {
            if cpu_ok {
                Hardware::Cpu
            } else {
                return Err(invalid(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Esta solicitud necesita CPU LOD/prefijo/densa y CPU no responde",
                ));
            }
        }
        Hardware::Auto => {
            let prefer_gpu = prefer == Hardware::Gpu;
            if prefer_gpu && gpu_ok {
                Hardware::Gpu
            } else if !prefer_gpu && cpu_ok {
                Hardware::Cpu
            } else if gpu_ok {
                Hardware::Gpu
            } else if cpu_ok {
                Hardware::Cpu
            } else {
                return Err(invalid(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Ninguno de los backends CPU/GPU está disponible",
                ));
            }
        }
    };
    Ok(Route {
        selected,
        requested,
        fallback: requested == Hardware::Auto && selected != prefer && !requires_cpu && !dense,
        dense,
    })
}

#[derive(Clone)]
struct AppState {
    client: Client,
    cpu_url: String,
    gpu_url: String,
    prefer: Hardware,
    timeout: Duration,
    started: Instant,
}
impl AppState {
    fn endpoint(&self, hardware: Hardware, path: &str) -> String {
        let base = match hardware {
            Hardware::Cpu => &self.cpu_url,
            Hardware::Gpu => &self.gpu_url,
            Hardware::Auto => unreachable!(),
        };
        format!("{}{}", base.trim_end_matches('/'), path)
    }
    async fn available(&self, hardware: Hardware) -> bool {
        let response = self
            .client
            .get(self.endpoint(hardware, "/ready"))
            .timeout(Duration::from_secs(2))
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => {
                let body: Value = response.json().await.unwrap_or(Value::Null);
                body.get("status").and_then(Value::as_str) == Some("ready")
                    || body.get("model_loaded").and_then(Value::as_bool) == Some(true)
            }
            _ => false,
        }
    }
    async fn healths(&self) -> (bool, bool) {
        tokio::join!(self.available(Hardware::Cpu), self.available(Hardware::Gpu))
    }
}

async fn index() -> Html<&'static str> {
    Html(INDEX)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (cpu, gpu) = state.healths().await;
    Json(json!({
        "status":if cpu||gpu {"ready"} else {"unavailable"},
        "engine":"kvslim-hardware-router-rust",
        "cpu":{"available":cpu,"url":state.cpu_url,"lod":"historical_cpu_branch_bound_and_ColdStore"},
        "gpu":{"available":gpu,"url":state.gpu_url,
            "lod":"shared_canonical_cpu_4bit_LSH_ColdStore_exact_mass"},
        "auto_prefer":state.prefer.label(),
        "up_seconds":state.started.elapsed().as_secs_f64(),
        "model_weights_modified":false,
        "backends_reconfigured":false
    }))
}
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let (cpu, gpu) = state.healths().await;
    let code = if cpu || gpu {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(json!({"status":if cpu||gpu {"ready"} else {"unavailable"},
        "cpu":cpu,"gpu":gpu})),
    )
        .into_response()
}
async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (cpu, gpu) = state.healths().await;
    let mut items = vec![];
    if cpu {
        items.push(json!({"id":CPU_MODEL,"object":"model","created":0,"owned_by":"cpu"}));
        items.push(json!({"id":DENSE_MODEL,"object":"model","created":0,"owned_by":"cpu"}));
    }
    if gpu {
        items.push(json!({"id":GPU_MODEL,"object":"model","created":0,"owned_by":"gpu"}));
    }
    if cpu || gpu {
        items.push(json!({"id":AUTO_MODEL,"object":"model","created":0,"owned_by":"router"}));
    }
    Json(json!({"object":"list","data":items}))
}
async fn capabilities(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (cpu, gpu) = state.healths().await;
    Json(json!({
        "router":"Rust_only_no_LibTorch_required","select_modes":["cpu","gpu","auto"],
        "default_preference":state.prefer.label(),
        "cpu":{"available":cpu,"lod":"exact_BF16_cold_4bit_LSH_branch_bound"},
        "gpu":{"available":gpu,"lod":"same_CPU_axis_index_cold_store_exact_mass_fallback",
            "full_CPU_regional_branch_bound_ported":false,
            "logits_bit_exact_with_BF16_CPU":false},
        "auto_fallback":"when preferred unavailable, only for compatible requests",
        "explicit_selection_never_falls_back":true,
        "weights_and_cpu_configuration_unchanged":true,
        "api":{"stream":false,"tools":false,"temperature":0,"n":1}
    }))
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(request): Json<Incoming>,
) -> Result<Response, ApiError> {
    let (requested, dense, max) = parse_input(&request)?;
    let (cpu, gpu) = state.healths().await;
    let requires_cpu = dense
        || request
            .document
            .as_deref()
            .is_some_and(|d| !d.trim().is_empty());
    let route = choose_mode(requested, state.prefer, cpu, gpu, requires_cpu, dense)?;
    let mut payload = json!({
        "messages":request.messages,
        "max_tokens":max,
        "temperature":0,
        "stream":false,
        "enable_thinking":request.enable_thinking
    });
    if route.selected == Hardware::Cpu {
        payload["native"] = json!(route.dense);
        if let Some(doc) = request.document.as_deref()
            && !doc.trim().is_empty()
        {
            payload["document"] = json!(doc);
        }
    } else {
        payload["model"] = json!(GPU_MODEL);
    }
    let started = Instant::now();
    let response = state
        .client
        .post(state.endpoint(route.selected, "/v1/chat/completions"))
        .timeout(state.timeout)
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            invalid(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                },
                format!(
                    "{} backend failed (no retry to avoid duplicate generation): {}",
                    route.selected.label(),
                    e
                ),
            )
        })?;
    let status = response.status();
    let mut result: Value = response.json().await.map_err(|e| {
        invalid(
            StatusCode::BAD_GATEWAY,
            format!("Selected backend returned invalid JSON: {e}"),
        )
    })?;
    if !result.is_object() {
        return Err(invalid(
            StatusCode::BAD_GATEWAY,
            "Backend did not return JSON object",
        ));
    }
    result["kvslim_route"] = json!({
        "requested":route.requested.label(),
        "selected":route.selected.label(),
        "fallback":route.fallback,
        "cpu_original_unchanged":true,
        "gpu_original_unchanged":true,
        "precision":"CPU/GPU bitwise equivalence is not required; cross-backend comparisons are diagnostic and GPU LOD is checked against dense GPU",
        "total_router_wall_seconds":started.elapsed().as_secs_f64(),
    });
    Ok((status, Json(result)).into_response())
}

fn parse_upstream(raw: &str) -> Result<String> {
    let url = reqwest::Url::parse(raw)?;
    ensure!(
        url.scheme() == "http",
        "Only loopback HTTP upstreams permitted"
    );
    ensure!(
        url.path() == "/" && url.query().is_none() && url.fragment().is_none(),
        "Backend URL must point to root without query/fragment"
    );
    let host = url.host_str().context("no upstream hostname")?;
    let ip = host.parse::<IpAddr>();
    ensure!(
        ip.is_ok_and(|addr| addr.is_loopback()) || host.eq_ignore_ascii_case("localhost"),
        "Backend upstream not loopback (router must not become open proxy)"
    );
    Ok(raw.trim_end_matches('/').to_owned())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut host = String::from("127.0.0.1");
    let mut port = 8288u16;
    let mut cpu_url = CPU_URL.to_owned();
    let mut gpu_url = GPU_URL.to_owned();
    let mut prefer = Hardware::Cpu;
    let mut timeout_seconds = 300u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--host" => host = args.next().context("missing --host")?,
            "--port" => port = args.next().context("missing --port")?.parse()?,
            "--cpu-url" => cpu_url = args.next().context("missing --cpu-url")?,
            "--gpu-url" => gpu_url = args.next().context("missing --gpu-url")?,
            "--auto-prefer" => {
                let arg = args.next().context("missing --auto-prefer")?;
                prefer = Hardware::parse(&arg).context("expected cpu|gpu")?;
                ensure!(
                    prefer != Hardware::Auto,
                    "auto cannot be its own preference"
                );
            }
            "--timeout-secs" => {
                timeout_seconds = args.next().context("missing --timeout-secs")?.parse()?
            }
            "--help" | "-h" => {
                println!("KVSLIM canonical local gateway");
                println!(
                    "--host 127.0.0.1 --port 8288 --cpu-url http://127.0.0.1:8088 --gpu-url http://127.0.0.1:8188 --auto-prefer cpu|gpu --timeout-secs 300"
                );
                return Ok(());
            }
            other => bail!("Unrecognized gateway argument: {other}"),
        }
    }
    let ip = host
        .parse::<IpAddr>()
        .context("--host must be a loopback IP address")?;
    ensure!(
        ip.is_loopback(),
        "No authentication: router must bind loopback only"
    );
    ensure!(
        (1..=3600).contains(&timeout_seconds),
        "timeout must be 1..=3600"
    );
    cpu_url = parse_upstream(&cpu_url)?;
    gpu_url = parse_upstream(&gpu_url)?;
    ensure!(cpu_url != gpu_url, "CPU/GPU upstreams must remain separate");
    let state = Arc::new(AppState {
        client: Client::builder().pool_max_idle_per_host(4).build()?,
        cpu_url,
        gpu_url,
        prefer,
        timeout: Duration::from_secs(timeout_seconds),
        started: Instant::now(),
    });
    let address = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    let router = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(1_048_576))
        .with_state(state);
    println!("KVSLIM canonical local gateway at http://{address}/");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(other: Value) -> Incoming {
        let mut base = json!({"messages":[{"role":"user","content":"Hola"}]});
        base.as_object_mut()
            .unwrap()
            .extend(other.as_object().unwrap().clone());
        serde_json::from_value(base).unwrap()
    }
    #[test]
    fn cpu_explicit_has_no_unexpected_gpu_fallback() {
        assert!(choose_mode(Hardware::Cpu, Hardware::Gpu, false, true, false, false).is_err());
        assert_eq!(
            choose_mode(Hardware::Cpu, Hardware::Gpu, true, true, false, false)
                .unwrap()
                .selected,
            Hardware::Cpu
        );
    }
    #[test]
    fn gpu_explicit_has_no_unexpected_cpu_fallback() {
        assert!(choose_mode(Hardware::Gpu, Hardware::Cpu, true, false, false, false).is_err());
        assert_eq!(
            choose_mode(Hardware::Gpu, Hardware::Cpu, true, true, false, false)
                .unwrap()
                .selected,
            Hardware::Gpu
        );
    }
    #[test]
    fn auto_prefer_cpu_and_fallback_gpu_when_compatible() {
        assert_eq!(
            choose_mode(Hardware::Auto, Hardware::Cpu, true, true, false, false)
                .unwrap()
                .selected,
            Hardware::Cpu
        );
        let r = choose_mode(Hardware::Auto, Hardware::Cpu, false, true, false, false).unwrap();
        assert_eq!(r.selected, Hardware::Gpu);
        assert!(r.fallback);
        assert!(choose_mode(Hardware::Auto, Hardware::Cpu, false, true, true, false).is_err());
    }
    #[test]
    fn cpu_lod_documents_cannot_be_sent_to_gpu() {
        let req = request(json!({"hardware":"auto","document":"Libro de referencia"}));
        let (_, dense, _) = parse_input(&req).unwrap();
        assert!(!dense);
        assert!(choose_mode(Hardware::Auto, Hardware::Cpu, false, true, true, false).is_err());
        assert!(parse_input(&request(json!({"hardware":"gpu","document":"Libro"}))).is_err());
    }
    #[test]
    fn rejects_unknown_models_sampling_and_streaming() {
        for other in [
            json!({"model":"does-not-exist"}),
            json!({"stream":true}),
            json!({"temperature":0.5}),
            json!({"top_p":0.75}),
            json!({"tools":[{"type":"function"}]}),
            json!({"unhandled":true}),
            json!({"model":GPU_MODEL,"hardware":"cpu"}),
        ] {
            assert!(parse_input(&request(other)).is_err());
        }
    }
    #[test]
    fn parses_model_aliases_and_bf16_dense() {
        assert_eq!(
            parse_input(&request(json!({"model":GPU_MODEL}))).unwrap().0,
            Hardware::Gpu
        );
        let (_, dense, max) =
            parse_input(&request(json!({"model":DENSE_MODEL,"max_tokens":3}))).unwrap();
        assert!(dense);
        assert_eq!(max, 3);
        let (hardware, _, _) = parse_input(&request(json!({"model":AUTO_MODEL}))).unwrap();
        assert_eq!(hardware, Hardware::Auto);
    }
    #[test]
    fn forbids_remote_upstreams() {
        assert!(parse_upstream(CPU_URL).is_ok());
        assert!(parse_upstream("http://8.8.8.8:8099").is_err());
        assert!(parse_upstream("http://localhost:8188/x").is_err());
    }
}
