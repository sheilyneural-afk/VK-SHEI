//! Source revision captured when the binary was built (see `build.rs`).
//!
//! Validation, calibration and certification receipts embed this value so a
//! receipt can be traced to a commit. It identifies `git HEAD` and whether
//! tracked files were modified at build time; for a dirty tree the commit does
//! not by itself reproduce the exact sources.

use serde_json::{Value, json};

pub const SOURCE_COMMIT: &str = env!("MSLIM_SOURCE_COMMIT");
const SOURCE_DIRTY: &str = env!("MSLIM_SOURCE_DIRTY");

pub fn source_revision() -> Value {
    json!({
        "commit": SOURCE_COMMIT,
        "tracked_files_modified_at_build": match SOURCE_DIRTY {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::Null,
        },
        "meaning": "git HEAD when the binary was built; reproducible only when tracked_files_modified_at_build is false",
    })
}

/// Reusable evidence must come from this exact commit and from a clean build.
/// A dirty build may still be useful for development, but it cannot authorize
/// a cached production receipt.
pub fn is_current_clean_source_revision(value: Option<&Value>) -> bool {
    let Some(value) = value else {
        return false;
    };
    SOURCE_COMMIT.len() == 40
        && SOURCE_COMMIT.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.get("commit").and_then(Value::as_str) == Some(SOURCE_COMMIT)
        && value
            .get("tracked_files_modified_at_build")
            .and_then(Value::as_bool)
            == Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_is_a_full_hash_or_explicitly_unknown() {
        assert!(
            SOURCE_COMMIT == "unknown"
                || (SOURCE_COMMIT.len() == 40
                    && SOURCE_COMMIT.bytes().all(|byte| byte.is_ascii_hexdigit()))
        );
    }

    #[test]
    fn revision_value_has_the_documented_shape() {
        let value = source_revision();
        assert_eq!(value["commit"], SOURCE_COMMIT);
        assert!(
            value["tracked_files_modified_at_build"].is_boolean()
                || value["tracked_files_modified_at_build"].is_null()
        );
        assert!(value["meaning"].is_string());
    }

    #[test]
    fn reusable_revision_requires_the_current_commit_and_a_clean_build() {
        if SOURCE_COMMIT == "unknown" {
            assert!(!is_current_clean_source_revision(Some(&source_revision())));
            return;
        }
        let clean = json!({
            "commit": SOURCE_COMMIT,
            "tracked_files_modified_at_build": false,
        });
        assert!(is_current_clean_source_revision(Some(&clean)));
        let dirty = json!({
            "commit": SOURCE_COMMIT,
            "tracked_files_modified_at_build": true,
        });
        assert!(!is_current_clean_source_revision(Some(&dirty)));
        let foreign = json!({
            "commit": "0".repeat(40),
            "tracked_files_modified_at_build": false,
        });
        assert!(!is_current_clean_source_revision(Some(&foreign)));
    }
}
