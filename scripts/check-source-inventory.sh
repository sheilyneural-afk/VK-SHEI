#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT

cargo build --locked --manifest-path tools/inventory-syntax/Cargo.toml

python3 scripts/source_inventory.py \
  --json "$temporary/inventory.json" \
  --markdown "$temporary/inventory.md" \
  --call-graph

python3 - "$temporary/inventory.json" scripts/inventory-duplicate-allowlist.json scripts/inventory-entrypoint-allowlist.json <<'PY'
import json
import pathlib
import sys

inventory = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
allowlist = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
entrypoint_allowlist = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
assert allowlist.get("schema") == "mslim.inventory.duplicate-allowlist/v1"

allowed = {
    tuple(sorted(group["members"]))
    for group in allowlist.get("groups", [])
}
observed = {
    tuple(sorted(f'{member["file"]}::{member["name"]}' for member in group["members"]))
    for group in inventory["normalized_duplicate_groups"]
}
unexpected = observed - allowed
stale = allowed - observed
if unexpected:
    raise SystemExit(f"unapproved normalized function clones: {sorted(unexpected)}")
if stale:
    raise SystemExit(f"stale duplicate allowlist entries: {sorted(stale)}")

assert entrypoint_allowlist.get("schema") == "mslim.inventory.indirect-entrypoint-allowlist/v1"
allowed_entrypoints = {
    entry["function"] for entry in entrypoint_allowlist.get("entries", [])
    if entry.get("reason", "").strip()
}

functions = inventory["functions"]
if any(record["name"] == "__attribute__" for record in functions):
    raise SystemExit("OpenCL attributes were misclassified as declarations")
if not any(record["language"] == "opencl" and record["kind"] == "opencl_kernel" for record in functions):
    raise SystemExit("no OpenCL kernel declarations were recovered by Clang")
if not any(record["language"] == "cpp" for record in functions):
    raise SystemExit("no C++ declarations were recovered by Clang")

rust = [record for record in functions if record["language"] == "rust"]
resolved = [
    record for record in rust
    if record["caller_resolution"] == "rust_analyzer_lsif_no_default_and_all_features_union"
]
if not resolved:
    raise SystemExit("rust-analyzer produced no semantically resolved call references")

private_orphans = {
    f'{record["file"]}::{record["name"]}'
    for record in resolved
    if record["kind"] in {"rust_free_function", "rust_module_function"}
    and record["declared_visibility"] == "private"
    and not record["test"]
    and record["name"] != "main"
    and not record["verified_callers"]
}
unexpected_orphans = private_orphans - allowed_entrypoints
stale_entrypoints = allowed_entrypoints - private_orphans
if unexpected_orphans:
    raise SystemExit(f"private Rust functions without resolved callers: {sorted(unexpected_orphans)}")
if stale_entrypoints:
    raise SystemExit(f"stale indirect-entrypoint allowlist entries: {sorted(stale_entrypoints)}")

for language in ("cpp", "opencl", "python", "javascript", "shell"):
    rows = [record for record in functions if record["language"] == language]
    if rows and any(record["caller_resolution"].startswith("not_implemented") for record in rows):
        raise SystemExit(f"caller resolution is not implemented for {language}")

clone_languages = {
    record["language"]
    for record in functions
    if record.get("normalized_body_sha256")
}
missing_clone_languages = {
    record["language"] for record in functions
} - clone_languages
if missing_clone_languages:
    raise SystemExit(
        "normalized duplicate detection has no eligible body in languages: "
        + ", ".join(sorted(missing_clone_languages))
    )

print(
    "inventory gate OK: "
    f'{inventory["source_file_count"]} files, {inventory["function_count"]} declarations, '
    f'{len(resolved)} Rust declarations indexed in the feature-graph union'
)
print(f'build_input_digest={inventory["build_input_digest"]}')
PY
