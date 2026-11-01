//! Canonical JSON hashing and HMAC-SHA256 authorization.
//! Port of `sheily-turn-engine-rs/src/resilience/integrity.rs`.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Deterministic JSON encoding: object keys sorted recursively.
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&sort_value(value)).unwrap_or_else(|_| "{}".to_string())
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .iter()
                .map(|(key, value)| (key.clone(), sort_value(value)))
                .collect();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

/// RFC 2104 HMAC-SHA256, hex encoded.
pub fn hmac_sha256_hex(secret: &[u8], payload: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut key = [0u8; BLOCK];
    if secret.len() > BLOCK {
        let digest = Sha256::digest(secret);
        key[..digest.len()].copy_from_slice(&digest);
    } else {
        key[..secret.len()].copy_from_slice(secret);
    }
    let mut inner_pad = [0x36u8; BLOCK];
    let mut outer_pad = [0x5cu8; BLOCK];
    for index in 0..BLOCK {
        inner_pad[index] ^= key[index];
        outer_pad[index] ^= key[index];
    }
    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(payload);
    let inner_digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner_digest);
    hex::encode(outer.finalize())
}

/// Constant-time comparison for signatures.
pub fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

/// Verify an HMAC over the canonical body (every field except `signature`).
pub fn verify_signed_body(secret: &[u8], token: &Map<String, Value>) -> bool {
    let signature = token.get("signature").and_then(Value::as_str).unwrap_or("");
    if signature.is_empty() {
        return false;
    }
    let mut body = token.clone();
    body.remove("signature");
    let expected = hmac_sha256_hex(secret, canonical_json(&Value::Object(body)).as_bytes());
    constant_time_eq(&expected, signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        let a = json!({"b": 1, "a": {"y": 2, "x": [ {"d":1,"c":2} ]}});
        assert_eq!(
            canonical_json(&a),
            r#"{"a":{"x":[{"c":2,"d":1}],"y":2},"b":1}"#
        );
    }

    #[test]
    fn hmac_matches_rfc4231_case_2() {
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn signed_body_detects_tampering() {
        let secret = b"s3cret";
        let mut token = json!({"actor": "a", "expires_at": 9999999999u64})
            .as_object()
            .unwrap()
            .clone();
        let sig = hmac_sha256_hex(
            secret,
            canonical_json(&Value::Object(token.clone())).as_bytes(),
        );
        token.insert("signature".into(), json!(sig));
        assert!(verify_signed_body(secret, &token));
        token.insert("actor".into(), json!("evil"));
        assert!(!verify_signed_body(secret, &token));
    }
}
