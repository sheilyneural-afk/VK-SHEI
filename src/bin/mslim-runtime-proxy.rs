//! Provider-neutral HTTP proxy over the generic runtime IPC session.
use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use mslim::{
    generic_runtime::{IPC_SCHEMA, IpcOperation, IpcRequest},
    ipc_transport::IpcClient,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    env,
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone)]
struct AppState {
    client: Arc<Mutex<IpcClient>>,
    session_id: String,
}

#[derive(Deserialize)]
struct GenerateRequest {
    input_tokens: Vec<u32>,
    #[serde(default = "default_tokens")]
    max_new_tokens: usize,
}
fn default_tokens() -> usize {
    1
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    runtime_session_id: String,
    external_runtime: bool,
}

async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ready",
        runtime_session_id: state.session_id,
        external_runtime: true,
    })
}

async fn generate(
    State(state): State<AppState>,
    Json(input): Json<GenerateRequest>,
) -> Result<Json<Value>, (axum::http::StatusCode, String)> {
    if input.input_tokens.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "input_tokens must not be empty".into(),
        ));
    }
    let req = IpcRequest {
        schema: IPC_SCHEMA.into(),
        operation: IpcOperation::Prefill,
        request_id: format!(
            "generate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ),
        session_id: Some(state.session_id.clone()),
        tensors: vec![],
        payload: json!({"input_tokens":input.input_tokens,"max_new_tokens":input.max_new_tokens,"temperature":null,"seed":null}),
    };
    let mut client = state.client.lock().map_err(|_| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "IPC mutex poisoned".into(),
        )
    })?;
    let response = client
        .request(&req)
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !response.ok {
        return Err((
            axum::http::StatusCode::BAD_GATEWAY,
            response
                .error
                .unwrap_or_else(|| "runtime rejected request".into()),
        ));
    }
    Ok(Json(response.payload))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let socket = args
        .windows(2)
        .find(|w| w[0] == "--socket")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "/tmp/mslim-runtime-host.sock".into());
    let port: u16 = args
        .windows(2)
        .find(|w| w[0] == "--port")
        .map(|w| w[1].parse())
        .transpose()?
        .unwrap_or(8190);
    let mut client =
        IpcClient::connect(&socket, Duration::from_secs(10)).context("connect runtime host")?;
    let probe = IpcRequest {
        schema: IPC_SCHEMA.into(),
        operation: IpcOperation::Probe,
        request_id: "proxy-probe".into(),
        session_id: None,
        tensors: vec![],
        payload: Value::Null,
    };
    let response = client.request(&probe)?;
    if !response.ok {
        bail!("runtime probe rejected: {:?}", response.error);
    }
    let open = IpcRequest {
        schema: IPC_SCHEMA.into(),
        operation: IpcOperation::OpenSession,
        request_id: "proxy-open".into(),
        session_id: None,
        tensors: vec![],
        payload: Value::Null,
    };
    let response = client.request(&open)?;
    if !response.ok {
        bail!("runtime open rejected: {:?}", response.error);
    }
    let session_id = response
        .session_id
        .context("runtime returned no session id")?;
    let state = AppState {
        client: Arc::new(Mutex::new(client)),
        session_id,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/generate", post(generate))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("runtime proxy listening on http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
