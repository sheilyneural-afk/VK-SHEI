//! Canonical identity carried by human-facing headers and machine receipts.

use serde_json::{Value, json};

pub const PROJECT: &str = "KV-SHEI";
pub const PRODUCT: &str = "M-SLIM";
pub const TECHNICAL_NAMESPACE: &str = "mslim";

// Canonical identity-bearing contracts.  The previous versions predate the
// mandatory project/product/namespace identity and remain legacy input only;
// producers must never emit them again under their old version number.
pub const MODEL_BUNDLE_SCHEMA: &str = "mslim.model_bundle/v2";
pub const VALIDATION_RECEIPT_SCHEMA: &str = "mslim.validation_receipt/v2";
pub const VALIDATION_MATRIX_SCHEMA: &str = "mslim.validation-matrix/v2";
pub const CALIBRATION_RECEIPT_SCHEMA: &str = "mslim.calibration_receipt/v4";
pub const LOCAL_RELEASE_SCHEMA: &str = "mslim.local_release/v2";

pub const LEGACY_MODEL_BUNDLE_SCHEMA: &str = "mslim.model_bundle/v1";
pub const LEGACY_VALIDATION_RECEIPT_SCHEMA: &str = "mslim.validation_receipt/v1";
pub const LEGACY_VALIDATION_MATRIX_SCHEMA: &str = "mslim.validation-matrix/v1";
pub const LEGACY_CALIBRATION_RECEIPT_SCHEMA: &str = "mslim.calibration_receipt/v3";

pub fn receipt_fields() -> Value {
    json!({
        "project": PROJECT,
        "product": PRODUCT,
        "technical_namespace": TECHNICAL_NAMESPACE,
    })
}
