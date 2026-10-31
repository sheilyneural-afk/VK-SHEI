//! Canonical backend-neutral M-SLIM local gateway.

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
};
use reqwest::Client;
use serde_json::{Value, json};
use std::{net::IpAddr, sync::Arc, time::Duration};

#[derive(Clone)]
struct AppState {
    client: Client,
    workers: Arc<Vec<(String, String)>>,
    auth_token: Arc<[u8]>,
}

#[derive(Debug)]
struct GatewayError(StatusCode, String);

impl IntoResponse for GatewayError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(json!({"error": {"message": self.1}}))).into_response()
    }
}

fn constant_time_token_eq(expected: &[u8], supplied: &[u8]) -> bool {
    let max = expected.len().max(supplied.len());
    let mut difference = expected.len() ^ supplied.len();
    for index in 0..max {
        difference |= expected.get(index).copied().unwrap_or(0) as usize
            ^ supplied.get(index).copied().unwrap_or(0) as usize;
    }
    difference == 0
}

async fn authenticate(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    let supplied = bearer_token(request.headers());
    if supplied.is_some_and(|token| constant_time_token_eq(&state.auth_token, token.as_bytes())) {
        next.run(request).await
    } else {
        GatewayError(StatusCode::UNAUTHORIZED, "authentication required".into()).into_response()
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    value
        .strip_prefix("Bearer ")
        .filter(|token| !token.is_empty())
}

fn loopback_url(raw: &str) -> Result<String> {
    let url = reqwest::Url::parse(raw).context("worker endpoint is not a URL")?;
    ensure!(url.scheme() == "http", "worker endpoint must use http");
    let host = url.host_str().context("worker endpoint has no host")?;
    let ip = host.parse::<IpAddr>();
    ensure!(
        ip.is_ok_and(|address| address.is_loopback()) || host.eq_ignore_ascii_case("localhost"),
        "workers must be local loopback endpoints"
    );
    ensure!(
        url.path() == "/" && url.query().is_none() && url.fragment().is_none(),
        "worker endpoint must point to its root"
    );
    Ok(raw.trim_end_matches('/').to_owned())
}

/// WorkerRegistry rejects empty and duplicate ids; the gateway adds the
/// loopback-only restriction on top.
fn parse_workers(assignments: &[String]) -> Result<Vec<(String, String)>> {
    let registry = mslim::worker_registry::WorkerRegistry::from_assignments(assignments)?;
    registry
        .workers()
        .iter()
        .map(|worker| Ok((worker.id.clone(), loopback_url(&worker.endpoint)?)))
        .collect()
}

async fn ready(state: State<Arc<AppState>>) -> impl IntoResponse {
    let mut available = Vec::new();
    for (id, endpoint) in state.workers.iter() {
        let result = state
            .client
            .get(format!("{endpoint}/ready"))
            .timeout(Duration::from_secs(2))
            .send()
            .await;
        available.push(json!({
            "id": id,
            "endpoint": endpoint,
            "ready": result.is_ok_and(|response| response.status().is_success())
        }));
    }
    let any = available
        .iter()
        .any(|item| item.get("ready") == Some(&Value::Bool(true)));
    let status = if any {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "status": if any {"ready"} else {"unavailable"},
            "workers": available
        })),
    )
}

async fn select_worker(state: &AppState) -> Result<String, GatewayError> {
    for (_, endpoint) in state.workers.iter() {
        let response = state
            .client
            .get(format!("{endpoint}/ready"))
            .timeout(Duration::from_secs(2))
            .send()
            .await;
        if !response.is_ok_and(|value| value.status().is_success()) {
            continue;
        }
        let capabilities = match state
            .client
            .get(format!("{endpoint}/v1/capabilities"))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => response.json::<Value>().await.ok(),
            _ => None,
        };
        if worker_capabilities_are_valid(capabilities.as_ref()) {
            return Ok(endpoint.clone());
        }
    }
    Err(GatewayError(
        StatusCode::SERVICE_UNAVAILABLE,
        "no registered worker is ready".into(),
    ))
}

fn worker_capabilities_are_valid(capabilities: Option<&Value>) -> bool {
    let Some(capabilities) = capabilities.and_then(Value::as_object) else {
        return false;
    };
    let has_engine = capabilities
        .get("engine")
        .or_else(|| capabilities.get("server"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let has_model = capabilities
        .get("model_checkpoint")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
        || capabilities
            .get("worker_capabilities")
            .and_then(Value::as_object)
            .is_some();
    let has_runtime_description = capabilities
        .get("model_metadata")
        .or_else(|| capabilities.get("worker_capabilities"))
        .is_some_and(Value::is_object);
    let model_digest = capabilities
        .get("model_digest")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let checkpoint_digest = capabilities
        .get("model_checkpoint")
        .and_then(Value::as_str)
        .and_then(|value| value.trim_end_matches('/').rsplit('/').next())
        .filter(|value| !value.trim().is_empty());
    let calibration = capabilities.get("calibration");
    let calibration_digest = calibration
        .and_then(|calibration| calibration.get("bundle"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let has_calibration = model_digest.is_some_and(|digest| {
        calibration.is_some_and(|receipt| {
            mslim::calibration_contract::validate_calibration_receipt(receipt, digest).is_ok()
        })
    });
    let identity_matches = model_digest.is_some()
        && model_digest == checkpoint_digest
        && model_digest == calibration_digest;
    has_engine && has_model && has_runtime_description && has_calibration && identity_matches
}

async fn models(state: State<Arc<AppState>>) -> Result<Json<Value>, GatewayError> {
    let worker = select_worker(&state).await?;
    let response = state
        .client
        .get(format!("{worker}/v1/models"))
        .send()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?;
    let status = response.status();
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?;
    if !status.is_success() {
        return Err(GatewayError(status, body.to_string()));
    }
    Ok(Json(body))
}

async fn capabilities(state: State<Arc<AppState>>) -> Result<Json<Value>, GatewayError> {
    let worker = select_worker(&state).await?;
    let body = state
        .client
        .get(format!("{worker}/v1/capabilities"))
        .send()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?
        .json::<Value>()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?;
    Ok(Json(json!({
        "server": "mslim-server",
        "worker_capabilities": body,
        "routing": "readiness-and-worker-registry"
    })))
}

async fn chat(
    state: State<Arc<AppState>>,
    Json(request): Json<Value>,
) -> Result<Json<Value>, GatewayError> {
    let worker = select_worker(&state).await?;
    let response = state
        .client
        .post(format!("{worker}/v1/chat/completions"))
        .timeout(Duration::from_secs(3600))
        .json(&request)
        .send()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?;
    let status = response.status();
    let mut body = response
        .json::<Value>()
        .await
        .map_err(|error| GatewayError(StatusCode::BAD_GATEWAY, error.to_string()))?;
    if !status.is_success() {
        return Err(GatewayError(status, body.to_string()));
    }
    if let Some(object) = body.as_object_mut() {
        object.insert("mslim_server".into(), json!({"routing": "dynamic-worker"}));
    }
    Ok(Json(body))
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8288u16;
    let mut assignments = Vec::new();
    let auth_token = std::env::var("MSLIM_GATEWAY_TOKEN")
        .context("MSLIM_GATEWAY_TOKEN must be set; refusing unauthenticated gateway")?;
    ensure!(
        !auth_token.is_empty(),
        "MSLIM_GATEWAY_TOKEN cannot be empty"
    );
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().context("missing --host")?,
            "--port" => port = args.next().context("missing --port")?.parse()?,
            "--worker" => assignments.push(args.next().context("missing --worker id=endpoint")?),
            "--help" | "-h" => {
                println!("mslim-server --worker <id=http://127.0.0.1:port> [--worker ...]");
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    ensure!(
        !assignments.is_empty(),
        "at least one --worker id=endpoint is required"
    );
    let host_ip: IpAddr = host.parse().context("--host must be an IP address")?;
    ensure!(
        host_ip.is_loopback(),
        "the local gateway must bind loopback"
    );
    let workers = parse_workers(&assignments)?;
    let state = Arc::new(AppState {
        client: Client::builder().pool_max_idle_per_host(16).build()?,
        workers: Arc::new(workers),
        auth_token: Arc::<[u8]>::from(auth_token.into_bytes()),
    });
    let address = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    let protected = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat))
        .layer(middleware::from_fn_with_state(state.clone(), authenticate));
    let app = Router::new()
        .route("/health", get(ready))
        .route("/ready", get(ready))
        .merge(protected)
        .with_state(state);
    println!("M-SLIM server listening at http://{address}");
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

    fn assignments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parse_workers_accepts_loopback_and_trims() {
        let workers = parse_workers(&assignments(&[
            " a = http://127.0.0.1:9000/ ",
            "b=http://localhost:9001",
        ]))
        .unwrap();
        assert_eq!(
            workers,
            vec![
                ("a".to_owned(), "http://127.0.0.1:9000".to_owned()),
                ("b".to_owned(), "http://localhost:9001".to_owned()),
            ]
        );
    }

    #[test]
    fn parse_workers_rejects_duplicate_and_empty_ids() {
        assert!(
            parse_workers(&assignments(&[
                "a=http://127.0.0.1:9000",
                "a=http://127.0.0.1:9001",
            ]))
            .is_err()
        );
        assert!(parse_workers(&assignments(&["=http://127.0.0.1:9000"])).is_err());
        assert!(parse_workers(&assignments(&["missing-separator"])).is_err());
    }

    #[test]
    fn parse_workers_rejects_non_loopback_or_non_root() {
        assert!(parse_workers(&assignments(&["a=http://10.0.0.5:9000"])).is_err());
        assert!(parse_workers(&assignments(&["a=https://127.0.0.1:9000"])).is_err());
        assert!(parse_workers(&assignments(&["a=http://127.0.0.1:9000/v1"])).is_err());
    }

    #[test]
    fn worker_certificate_requires_matching_model_identity() {
        let bundle = "a".repeat(64);
        let valid = json!({
            "engine": "mslim-engine",
            "model_checkpoint": format!("runtime/models/{bundle}"),
            "model_digest": bundle,
            "model_metadata": {},
            "calibration": {
                "schema": "mslim.calibration_receipt/v3",
                "status": "measured-sustained-execution",
                "bundle": "a".repeat(64),
                "adapter": "accelerator-runtime",
                "identity_digest": "b".repeat(64),
                "started_unix_seconds": 1,
                "warmup_runs": 1,
                "measured_runs": 3,
                "decode_tokens_per_run": 8,
                "context_samples": [{
                    "prompt_tokens": 1,
                    "prefill_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                    "decode_tokens": 8,
                    "decode_micros": {"samples": [1, 1, 1], "minimum": 1, "median": 1, "p95": 1, "maximum": 1},
                    "status": "measured"
                }],
                "snapshot_contract": {
                    "kind": "physical-coldstore-handle",
                    "physical_kv_persisted": true,
                    "restore_latency_measured_as_physical_kv_hydration": false
                },
                "validation_matrix_gate": "same-backend-and-kv-integrity-passed"
            }
        });
        assert!(worker_capabilities_are_valid(Some(&valid)));
        let mut invalid = valid.clone();
        invalid["calibration"]["bundle"] = json!("c".repeat(64));
        assert!(!worker_capabilities_are_valid(Some(&invalid)));
        let mut legacy = valid;
        legacy["calibration"]["schema"] = json!("kvslim.calibration_receipt/v2");
        assert!(!worker_capabilities_are_valid(Some(&legacy)));
    }

    #[test]
    fn bearer_auth_rejects_missing_or_wrong_tokens_without_plaintext_fallback() {
        let mut headers = HeaderMap::new();
        assert!(bearer_token(&headers).is_none());
        headers.insert("authorization", "Basic secret".parse().unwrap());
        assert!(bearer_token(&headers).is_none());
        headers.insert("authorization", "Bearer right".parse().unwrap());
        let supplied = bearer_token(&headers).unwrap();
        assert!(constant_time_token_eq(b"right", supplied.as_bytes()));
        assert!(!constant_time_token_eq(b"wrong", supplied.as_bytes()));
    }
}
