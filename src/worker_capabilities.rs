//! Shared gateway/worker capability contract.
//!
//! The worker produces this shape and the gateway validates the same shape;
//! tests must not invent a second, more permissive representation.

use serde_json::{Value, json};

pub const WORKER_CAPABILITIES_SCHEMA: &str = "mslim.worker_capabilities/v1";

pub fn effective_configuration(model_digest: &str, backend: &str, configuration: Value) -> Value {
    json!({
        "schema": WORKER_CAPABILITIES_SCHEMA,
        "model_digest": model_digest,
        "backend": backend,
        "effective_configuration": configuration,
    })
}

pub fn validate(value: &Value, expected_model_digest: &str) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.get("schema").and_then(Value::as_str) == Some(WORKER_CAPABILITIES_SCHEMA)
        && object.get("model_digest").and_then(Value::as_str) == Some(expected_model_digest)
        && object
            .get("backend")
            .and_then(Value::as_str)
            .is_some_and(|backend| !backend.trim().is_empty())
        && object
            .get("effective_configuration")
            .and_then(Value::as_object)
            .is_some_and(|configuration| !configuration.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_real_configuration_bound_to_the_model() {
        let digest = "a".repeat(64);
        let value =
            effective_configuration(&digest, "accelerator-runtime", json!({"lod_epsilon": 0.0}));
        assert!(validate(&value, &digest));
        assert!(!validate(&value, &"b".repeat(64)));
        assert!(!validate(&json!({}), &digest));
    }
}
