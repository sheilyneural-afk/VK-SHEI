#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"
env_file="$root/deploy/local-production/local-production.env"
manifest="$root/deploy/local-production/manifest.json"

fail() { echo "VERIFY-FAIL: $*" >&2; exit 1; }

[ -f "$env_file" ] || fail "missing local-production.env"
[ -f "$manifest" ] || fail "missing manifest.json"

# shellcheck disable=SC1090
source "$env_file"

command -v python3 >/dev/null 2>&1 || fail "python3 is required to validate manifest.json"
build_input_digest=$(python3 "$root/scripts/source_inventory.py" --print-build-input-digest) ||
  fail "cannot calculate build_input_digest"
python3 - "$manifest" "${MSLIM_BINARY_DIGEST:-}" "${MSLIM_WORKER_BINARY_DIGEST:-}" \
  "${MSLIM_MODEL_DIGEST:-}" "${MSLIM_MODEL_DIR:-}" "$build_input_digest" <<'PY' || fail "manifest.json does not match the effective release configuration"
import json
import pathlib
import sys

path, binary_digest, worker_digest, model_digest, model_dir, build_input_digest = sys.argv[1:]
manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
root = pathlib.Path(path).resolve().parents[2]
configured_model = pathlib.Path(model_dir)
if not configured_model.is_absolute():
    configured_model = root / configured_model
configured_model = configured_model.resolve()
expected_model = (root / "deploy/local-production/runtime/models" / model_digest).resolve()
assert configured_model == expected_model
expected_model_path = f"runtime/models/{model_digest}"
assert manifest.get("project") == "KV-SHEI"
assert manifest.get("product") == "M-SLIM"
assert manifest.get("technical_namespace") == "mslim"
assert manifest.get("schema") == "mslim.local_release/v3"
assert manifest.get("build_input_digest") == build_input_digest
assert manifest.get("binary", {}).get("path") == "runtime/bin/mslim-gateway"
assert manifest.get("binary", {}).get("sha256") == binary_digest
assert manifest.get("worker_binary", {}).get("path") == "runtime/bin/mslim-runtime-worker"
assert manifest.get("worker_binary", {}).get("sha256") == worker_digest
assert manifest.get("model", {}).get("relative_path") == expected_model_path
assert manifest.get("model", {}).get("bundle_sha256") == model_digest
PY

[ "${MSLIM_HOST:-}" = "127.0.0.1" ] || fail "host must be 127.0.0.1"
case "${MSLIM_ENABLE_LOD:-}" in
  true|false) ;;
  *) fail "MSLIM_ENABLE_LOD must be true or false" ;;
esac
[ "${MSLIM_COLDSTORE_FORMAT:-}" = "TSKVPG04" ] || fail "unsupported ColdStore format"

is_inside_project() {
  case "$1" in
    "$root"/*) return 0 ;;
    *) return 1 ;;
  esac
}

for value in "${MSLIM_MODEL_DIR:-}" "${MSLIM_COLDSTORE_DIR:-}" "${MSLIM_RECEIPTS_DIR:-}" "${MSLIM_LOG_DIR:-}"; do
  [ -n "$value" ] || fail "empty runtime path"
  case "$value" in /*) path="$value" ;; *) path="$root/$value" ;; esac
  is_inside_project "$path" || fail "path escapes project: $path"
done

binary="$runtime/bin/mslim-gateway"
worker_binary="$runtime/bin/mslim-runtime-worker"
[ -f "$binary" ] || fail "missing binary: $binary"
[ -x "$binary" ] || fail "binary is not executable: $binary"
[ -f "$worker_binary" ] || fail "missing worker binary: $worker_binary"
[ -x "$worker_binary" ] || fail "worker binary is not executable: $worker_binary"

case "${MSLIM_MODEL_DIR:-}" in
  /*) model_path="$MSLIM_MODEL_DIR" ;;
  *) model_path="$root/$MSLIM_MODEL_DIR" ;;
esac
model_path=$(realpath -e -- "$model_path") || fail "cannot resolve model path"
[ "$model_path" = "$runtime/models/${MSLIM_MODEL_DIGEST:-}" ] ||
  fail "model path does not match runtime/models/MSLIM_MODEL_DIGEST"
[ -d "$model_path" ] || fail "missing model directory: $model_path"

actual_binary=$(sha256sum "$binary" | awk '{print $1}')
[ "$actual_binary" = "${MSLIM_BINARY_DIGEST:-}" ] || fail "binary digest mismatch"
actual_worker_binary=$(sha256sum "$worker_binary" | awk '{print $1}')
[ "$actual_worker_binary" = "${MSLIM_WORKER_BINARY_DIGEST:-}" ] ||
  fail "worker binary digest mismatch"

model_digest=$(python3 "$root/scripts/model-bundle-digest.py" "$model_path") ||
  fail "cannot calculate model content digest"
[ "$model_digest" = "${MSLIM_MODEL_DIGEST:-}" ] || fail "model bundle content digest mismatch"

if find "$model_path" -type f -perm /022 -print -quit | grep -q .; then
  fail "model files must not be group/world writable"
fi

echo "VERIFY-OK: local release is structurally valid"
echo "binary_sha256=$actual_binary"
echo "worker_binary_sha256=$actual_worker_binary"
echo "model_bundle_sha256=$model_digest"
echo "host=$MSLIM_HOST"
echo "lod=$MSLIM_ENABLE_LOD"
