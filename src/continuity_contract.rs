//! Versioned, provider-neutral continuity contracts for physical KV sessions.
//!
//! The caller owns the mapping from governed identity to the opaque handle.
//! M-SLIM sees no tenant, principal, prompt text, model weights, or store path.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

pub use mslim_adapter_api::CacheMode;

pub const CONTINUITY_REQUEST_SCHEMA: &str = "mslim.continuity.request/v1";
pub const CACHE_RECEIPT_SCHEMA: &str = "mslim.continuity.cache-receipt/v2";
const REQUEST_HASH_DOMAIN: &[u8] = b"mslim-continuity-request-v1\0";
const RECEIPT_HASH_DOMAIN: &[u8] = b"mslim-continuity-cache-receipt-v2\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError {
    field: &'static str,
    reason: &'static str,
}

impl ContractError {
    const fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid {}: {}", self.field, self.reason)
    }
}

impl Error for ContractError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityOperation {
    Resume,
    Snapshot,
    Release,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContinuityBounds {
    pub max_reused_tokens: u64,
    pub max_restored_bytes: u64,
}

impl ContinuityBounds {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.max_reused_tokens == 0 {
            return Err(ContractError::new(
                "bounds.max_reused_tokens",
                "must be positive",
            ));
        }
        if self.max_restored_bytes == 0 {
            return Err(ContractError::new(
                "bounds.max_restored_bytes",
                "must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContinuityRequest {
    pub schema: String,
    pub operation: ContinuityOperation,
    pub opaque_session_handle: String,
    pub expected_generation: u64,
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub prefix_digest: String,
    pub policy_digest: String,
    pub expires_at_unix_s: u64,
    pub bounds: ContinuityBounds,
}

impl ContinuityRequest {
    pub fn validate_at(&self, now_unix_s: u64) -> Result<(), ContractError> {
        validate_schema(&self.schema, CONTINUITY_REQUEST_SCHEMA, "request.schema")?;
        validate_handle(&self.opaque_session_handle)?;
        validate_digest(&self.model_digest, "request.model_digest")?;
        validate_digest(&self.tokenizer_digest, "request.tokenizer_digest")?;
        validate_digest(&self.prefix_digest, "request.prefix_digest")?;
        validate_digest(&self.policy_digest, "request.policy_digest")?;
        if self.expires_at_unix_s <= now_unix_s {
            return Err(ContractError::new(
                "request.expires_at_unix_s",
                "capability is expired",
            ));
        }
        self.bounds.validate()
    }

    pub fn canonical_hash_at(&self, now_unix_s: u64) -> Result<String, ContractError> {
        self.validate_at(now_unix_s)?;
        canonical_hash(REQUEST_HASH_DOMAIN, self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContinuityCacheReceipt {
    pub project: String,
    pub product: String,
    pub technical_namespace: String,
    pub schema: String,
    pub operation: ContinuityOperation,
    pub opaque_session_handle: String,
    pub previous_generation: u64,
    pub resulting_generation: u64,
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub prefix_digest: String,
    pub policy_digest: String,
    pub reused_tokens: u64,
    pub restored_bytes: u64,
    pub mode: CacheMode,
}

impl ContinuityCacheReceipt {
    pub fn validate_for_at(
        &self,
        request: &ContinuityRequest,
        now_unix_s: u64,
    ) -> Result<(), ContractError> {
        request.validate_at(now_unix_s)?;
        if self.project != crate::project_identity::PROJECT
            || self.product != crate::project_identity::PRODUCT
            || self.technical_namespace != crate::project_identity::TECHNICAL_NAMESPACE
        {
            return Err(ContractError::new(
                "receipt.identity",
                "unsupported identity",
            ));
        }
        validate_schema(&self.schema, CACHE_RECEIPT_SCHEMA, "receipt.schema")?;
        validate_handle(&self.opaque_session_handle)?;
        validate_digest(&self.model_digest, "receipt.model_digest")?;
        validate_digest(&self.tokenizer_digest, "receipt.tokenizer_digest")?;
        validate_digest(&self.prefix_digest, "receipt.prefix_digest")?;
        validate_digest(&self.policy_digest, "receipt.policy_digest")?;
        if self.operation != request.operation
            || self.opaque_session_handle != request.opaque_session_handle
            || self.previous_generation != request.expected_generation
            || self.model_digest != request.model_digest
            || self.tokenizer_digest != request.tokenizer_digest
            || self.prefix_digest != request.prefix_digest
            || self.policy_digest != request.policy_digest
        {
            return Err(ContractError::new("receipt", "does not bind to request"));
        }
        if self.resulting_generation < self.previous_generation {
            return Err(ContractError::new(
                "receipt.resulting_generation",
                "generation regressed",
            ));
        }
        if self.reused_tokens > request.bounds.max_reused_tokens {
            return Err(ContractError::new(
                "receipt.reused_tokens",
                "exceeds authorized bound",
            ));
        }
        if self.restored_bytes > request.bounds.max_restored_bytes {
            return Err(ContractError::new(
                "receipt.restored_bytes",
                "exceeds authorized bound",
            ));
        }
        if !matches!(self.operation, ContinuityOperation::Resume)
            && (self.reused_tokens != 0 || self.restored_bytes != 0)
        {
            return Err(ContractError::new(
                "receipt",
                "non-resume operation reports restored cache",
            ));
        }
        Ok(())
    }

    pub fn canonical_hash_for_at(
        &self,
        request: &ContinuityRequest,
        now_unix_s: u64,
    ) -> Result<String, ContractError> {
        self.validate_for_at(request, now_unix_s)?;
        canonical_hash(RECEIPT_HASH_DOMAIN, self)
    }
}

fn validate_schema(
    actual: &str,
    expected: &'static str,
    field: &'static str,
) -> Result<(), ContractError> {
    if actual != expected {
        return Err(ContractError::new(field, "unsupported schema"));
    }
    Ok(())
}

fn validate_handle(handle: &str) -> Result<(), ContractError> {
    if handle.is_empty() || handle.len() > 256 {
        return Err(ContractError::new(
            "opaque_session_handle",
            "invalid length",
        ));
    }
    if !handle
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~'))
    {
        return Err(ContractError::new(
            "opaque_session_handle",
            "forbidden characters",
        ));
    }
    Ok(())
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), ContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(ContractError::new(field, "must be lowercase SHA-256 hex"));
    }
    Ok(())
}

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, ContractError> {
    let encoded = serde_json::to_vec(value)
        .map_err(|_| ContractError::new("contract", "canonical serialization failed"))?;
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((encoded.len() as u64).to_le_bytes());
    hash.update(encoded);
    Ok(hex::encode(hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    fn request(operation: ContinuityOperation) -> ContinuityRequest {
        ContinuityRequest {
            schema: CONTINUITY_REQUEST_SCHEMA.into(),
            operation,
            opaque_session_handle: "session_7fP3-opaque".into(),
            expected_generation: 7,
            model_digest: digest('a'),
            tokenizer_digest: digest('b'),
            prefix_digest: digest('c'),
            policy_digest: digest('d'),
            expires_at_unix_s: 2_000,
            bounds: ContinuityBounds {
                max_reused_tokens: 4_096,
                max_restored_bytes: 64 * 1024 * 1024,
            },
        }
    }

    fn receipt(request: &ContinuityRequest) -> ContinuityCacheReceipt {
        ContinuityCacheReceipt {
            project: crate::project_identity::PROJECT.into(),
            product: crate::project_identity::PRODUCT.into(),
            technical_namespace: crate::project_identity::TECHNICAL_NAMESPACE.into(),
            schema: CACHE_RECEIPT_SCHEMA.into(),
            operation: request.operation,
            opaque_session_handle: request.opaque_session_handle.clone(),
            previous_generation: request.expected_generation,
            resulting_generation: request.expected_generation + 1,
            model_digest: request.model_digest.clone(),
            tokenizer_digest: request.tokenizer_digest.clone(),
            prefix_digest: request.prefix_digest.clone(),
            policy_digest: request.policy_digest.clone(),
            reused_tokens: 512,
            restored_bytes: 1_048_576,
            mode: CacheMode::Exact,
        }
    }

    #[test]
    fn valid_resume_roundtrips_and_hashes_deterministically() {
        let request = request(ContinuityOperation::Resume);
        let decoded: ContinuityRequest =
            serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(
            request.canonical_hash_at(1_000).unwrap(),
            request.canonical_hash_at(1_000).unwrap()
        );
        let receipt = receipt(&request);
        receipt.validate_for_at(&request, 1_000).unwrap();
        assert_eq!(
            receipt
                .canonical_hash_for_at(&request, 1_000)
                .unwrap()
                .len(),
            64
        );
    }

    #[test]
    fn serde_rejects_unknown_fields_and_enum_variants() {
        let mut value = serde_json::to_value(request(ContinuityOperation::Resume)).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("tenant_id".into(), serde_json::json!("forbidden"));
        assert!(serde_json::from_value::<ContinuityRequest>(value).is_err());

        let mut value = serde_json::to_value(request(ContinuityOperation::Resume)).unwrap();
        value["operation"] = serde_json::json!("delete_everything");
        assert!(serde_json::from_value::<ContinuityRequest>(value).is_err());
    }

    #[test]
    fn request_validation_is_fail_closed() {
        let mut value = request(ContinuityOperation::Resume);
        value.schema = "mslim.continuity.request/v2".into();
        assert!(value.validate_at(1_000).is_err());

        let mut value = request(ContinuityOperation::Resume);
        value.expires_at_unix_s = 1_000;
        assert!(value.validate_at(1_000).is_err());

        let mut value = request(ContinuityOperation::Resume);
        value.model_digest = digest('A');
        assert!(value.validate_at(1_000).is_err());

        let mut value = request(ContinuityOperation::Resume);
        value.opaque_session_handle = "../session".into();
        assert!(value.validate_at(1_000).is_err());
    }

    #[test]
    fn receipt_must_bind_to_request_and_respect_bounds() {
        let resume_request = request(ContinuityOperation::Resume);
        let mut value = receipt(&resume_request);
        value.prefix_digest = digest('e');
        assert!(value.validate_for_at(&resume_request, 1_000).is_err());

        let mut value = receipt(&resume_request);
        value.reused_tokens = resume_request.bounds.max_reused_tokens + 1;
        assert!(value.validate_for_at(&resume_request, 1_000).is_err());

        let snapshot = request(ContinuityOperation::Snapshot);
        let mut value = receipt(&snapshot);
        value.reused_tokens = 1;
        value.restored_bytes = 0;
        assert!(value.validate_for_at(&snapshot, 1_000).is_err());
    }
}
