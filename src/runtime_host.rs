//! Generic IPC request dispatcher for one executable runtime session.
use anyhow::{Result, bail};
use kvslim_adapter_api::{GenerationRequest, RuntimeSession};
use serde_json::{Value, json};

use crate::generic_runtime::{IPC_SCHEMA, IpcOperation, IpcRequest, IpcResponse};

pub struct RuntimeHost {
    session: Box<dyn RuntimeSession>,
    session_id: String,
}

impl RuntimeHost {
    pub fn new(session: Box<dyn RuntimeSession>, session_id: impl Into<String>) -> Result<Self> {
        let session_id = session_id.into();
        if session_id.is_empty() {
            bail!("session id is required");
        }
        Ok(Self {
            session,
            session_id,
        })
    }

    pub fn dispatch(&mut self, request: IpcRequest) -> IpcResponse {
        let id = request.request_id.clone();
        let result = self.dispatch_inner(&request);
        match result {
            Ok((session_id, payload, receipts)) => IpcResponse {
                schema: IPC_SCHEMA.into(),
                request_id: id,
                ok: true,
                session_id,
                receipts,
                tensors: Vec::new(),
                payload,
                error: None,
            },
            Err(error) => IpcResponse {
                schema: IPC_SCHEMA.into(),
                request_id: id,
                ok: false,
                session_id: Some(self.session_id.clone()),
                receipts: Vec::new(),
                tensors: Vec::new(),
                payload: Value::Null,
                error: Some(error.to_string()),
            },
        }
    }

    fn dispatch_inner(
        &mut self,
        request: &IpcRequest,
    ) -> Result<(Option<String>, Value, Vec<Value>)> {
        if request.schema != IPC_SCHEMA {
            bail!("unsupported IPC schema");
        }
        if request.request_id.is_empty() {
            bail!("request_id is required");
        }
        match request.operation {
            IpcOperation::Probe => Ok((
                None,
                json!({"capabilities": self.session.capabilities()}),
                Vec::new(),
            )),
            IpcOperation::OpenSession => Ok((
                Some(self.session_id.clone()),
                json!({"identity": self.session.identity()}),
                Vec::new(),
            )),
            IpcOperation::Prefill | IpcOperation::Decode => {
                let generation: GenerationRequest = serde_json::from_value(request.payload.clone())
                    .map_err(|e| anyhow::anyhow!("invalid generation payload: {e}"))?;
                let result = self.session.generate(generation)?;
                Ok((
                    Some(self.session_id.clone()),
                    serde_json::to_value(result)?,
                    Vec::new(),
                ))
            }
            IpcOperation::SnapshotKv => {
                let bytes = self.session.snapshot_kv()?;
                let receipt = self.session.snapshot_receipt()?;
                Ok((
                    Some(self.session_id.clone()),
                    json!({"snapshot_bytes": bytes, "receipt": receipt}),
                    vec![serde_json::to_value(receipt)?],
                ))
            }
            IpcOperation::RestoreKv => {
                let receipt: kvslim_adapter_api::CacheReceipt = serde_json::from_value(
                    request
                        .payload
                        .get("receipt")
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("receipt is required"))?,
                )?;
                let bytes: Vec<u8> = serde_json::from_value(
                    request
                        .payload
                        .get("snapshot_bytes")
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("snapshot_bytes is required"))?,
                )?;
                self.session.restore_receipt(&receipt, &bytes)?;
                Ok((
                    Some(self.session_id.clone()),
                    json!({"restored": true}),
                    vec![serde_json::to_value(receipt)?],
                ))
            }
            IpcOperation::Release => Ok((
                Some(self.session_id.clone()),
                json!({"released": true}),
                Vec::new(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use kvslim_adapter_api::{
        AdapterCapabilities, CacheMode, GenerationResult, GenerationStats, ModelIdentity,
    };
    struct Fake {
        caps: AdapterCapabilities,
        id: ModelIdentity,
        snap: Vec<u8>,
    }
    impl RuntimeSession for Fake {
        fn capabilities(&self) -> &AdapterCapabilities {
            &self.caps
        }
        fn identity(&self) -> &ModelIdentity {
            &self.id
        }
        fn generate(&mut self, r: GenerationRequest) -> Result<GenerationResult> {
            Ok(GenerationResult {
                output_tokens: vec![7],
                prompt_tokens: r.input_tokens.len(),
                finish_reason: "stop".into(),
                stats: GenerationStats::default(),
            })
        }
        fn snapshot_kv(&self) -> Result<Vec<u8>> {
            Ok(self.snap.clone())
        }
        fn restore_kv(&mut self, _: &[u8]) -> Result<()> {
            Ok(())
        }
        fn snapshot_receipt(&self) -> Result<kvslim_adapter_api::CacheReceipt> {
            Ok(kvslim_adapter_api::CacheReceipt {
                schema: "v1".into(),
                operation: "snapshot".into(),
                model_digest: self.id.model_digest.clone(),
                tokenizer_digest: self.id.tokenizer_digest.clone(),
                architecture_digest: self.id.architecture_digest.clone(),
                prefix_digest: "0".repeat(64),
                generation: 0,
                reused_tokens: 0,
                restored_bytes: 0,
                mode: CacheMode::Exact,
            })
        }
        fn restore_receipt(
            &mut self,
            r: &kvslim_adapter_api::CacheReceipt,
            b: &[u8],
        ) -> Result<()> {
            if r.model_digest != self.id.model_digest || b.is_empty() {
                bail!("invalid receipt or snapshot")
            }
            Ok(())
        }
    }
    fn host() -> RuntimeHost {
        RuntimeHost::new(
            Box::new(Fake {
                caps: AdapterCapabilities {
                    adapter_id: "generic".into(),
                    backend: "external".into(),
                    supports_prefill: true,
                    supports_decode: true,
                    supports_kv_restore: true,
                    supports_streaming: false,
                },
                id: ModelIdentity {
                    model_digest: "a".repeat(64),
                    tokenizer_digest: "b".repeat(64),
                    architecture_digest: "c".repeat(64),
                },
                snap: vec![1],
            }),
            "s",
        )
        .unwrap()
    }
    #[test]
    fn rejects_missing_request_id_and_schema() {
        let mut h = host();
        let r = h.dispatch(IpcRequest {
            schema: IPC_SCHEMA.into(),
            operation: IpcOperation::Probe,
            request_id: String::new(),
            session_id: None,
            tensors: vec![],
            payload: Value::Null,
        });
        assert!(!r.ok);
        assert_eq!(r.request_id, "");
        assert!(r.error.unwrap().contains("request_id"));
        let r = h.dispatch(IpcRequest {
            schema: "wrong".into(),
            operation: IpcOperation::Probe,
            request_id: "bad-schema".into(),
            session_id: None,
            tensors: vec![],
            payload: Value::Null,
        });
        assert!(!r.ok);
        assert!(r.error.unwrap().contains("schema"));
    }

    #[test]
    fn dispatches_probe_and_generation() {
        let mut h = host();
        let r = h.dispatch(IpcRequest {
            schema: IPC_SCHEMA.into(),
            operation: IpcOperation::Probe,
            request_id: "1".into(),
            session_id: None,
            tensors: vec![],
            payload: Value::Null,
        });
        assert!(r.ok);
        let r = h.dispatch(IpcRequest {
            schema: IPC_SCHEMA.into(),
            operation: IpcOperation::Prefill,
            request_id: "2".into(),
            session_id: None,
            tensors: vec![],
            payload: serde_json::to_value(GenerationRequest {
                input_tokens: vec![1],
                max_new_tokens: 1,
                temperature: None,
                seed: None,
            })
            .unwrap(),
        });
        assert!(r.ok);
    }
}
