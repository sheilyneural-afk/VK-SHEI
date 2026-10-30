//! Canonical backend-neutral KVSLIM local gateway.

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
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
}

#[derive(Debug)]
struct GatewayError(StatusCode, String);

impl IntoResponse for GatewayError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(json!({"error": {"message": self.1}}))).into_response()
    }
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
    let has_calibration = capabilities
        .get("calibration")
        .and_then(Value::as_object)
        .is_some_and(|calibration| {
            calibration.get("schema").and_then(Value::as_str)
                == Some("kvslim.calibration_receipt/v1")
                && calibration.get("status").and_then(Value::as_str)
                    == Some("measured-basic-execution")
        });
    let model_digest = capabilities
        .get("model_digest")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let checkpoint_digest = capabilities
        .get("model_checkpoint")
        .and_then(Value::as_str)
        .and_then(|value| value.trim_end_matches('/').rsplit('/').next())
        .filter(|value| !value.trim().is_empty());
    let calibration_digest = capabilities
        .get("calibration")
        .and_then(Value::as_object)
        .and_then(|calibration| calibration.get("bundle"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let identity_matches = model_digest.is_some()
        && model_digest == checkpoint_digest
        && model_digest == calibration_digest;
    let has_identity = capabilities
        .get("calibration")
        .and_then(Value::as_object)
        .and_then(|calibration| calibration.get("identity_digest"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    has_engine
        && has_model
        && has_runtime_description
        && has_calibration
        && identity_matches
        && has_identity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_certificate_requires_matching_model_identity() {
        let valid = json!({
            "engine": "kvslim-engine",
            "model_checkpoint": "runtime/models/model-a",
            "model_digest": "model-a",
            "model_metadata": {},
            "calibration": {
                "schema": "kvslim.calibration_receipt/v1",
                "status": "measured-basic-execution",
                "bundle": "model-a",
                "identity_digest": "environment-a"
            }
        });
        assert!(worker_capabilities_are_valid(Some(&valid)));
        let mut invalid = valid.clone();
        invalid["calibration"]["bundle"] = json!("model-b");
        assert!(!worker_capabilities_are_valid(Some(&invalid)));
    }
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
        "server": "kvslim-server",
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
        object.insert("kvslim_server".into(), json!({"routing": "dynamic-worker"}));
    }
    Ok(Json(body))
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8288u16;
    let mut assignments = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().context("missing --host")?,
            "--port" => port = args.next().context("missing --port")?.parse()?,
            "--worker" => assignments.push(args.next().context("missing --worker id=endpoint")?),
            "--help" | "-h" => {
                println!("kvslim-server --worker <id=http://127.0.0.1:port> [--worker ...]");
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
    let mut workers = Vec::new();
    for assignment in assignments {
        let (id, endpoint) = assignment
            .split_once('=')
            .context("worker must be id=endpoint")?;
        workers.push((id.trim().to_owned(), loopback_url(endpoint.trim())?));
    }
    let state = Arc::new(AppState {
        client: Client::builder().pool_max_idle_per_host(16).build()?,
        workers: Arc::new(workers),
    });
    let address = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    let app = Router::new()
        .route("/health", get(ready.clone()))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat))
        .with_state(state);
    println!("KVSLIM server listening at http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
