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

[ "${KVSLIM_HOST:-}" = "127.0.0.1" ] || fail "host must be 127.0.0.1"
case "${KVSLIM_ENABLE_LOD:-}" in
  true|false) ;;
  *) fail "KVSLIM_ENABLE_LOD must be true or false" ;;
esac
[ "${KVSLIM_COLDSTORE_FORMAT:-}" = "TSKVPG04" ] || fail "unsupported ColdStore format"

is_inside_project() {
  case "$1" in
    "$root"/*) return 0 ;;
    *) return 1 ;;
  esac
}

for value in "${KVSLIM_MODEL_DIR:-}" "${KVSLIM_COLDSTORE_DIR:-}" "${KVSLIM_RECEIPTS_DIR:-}" "${KVSLIM_LOG_DIR:-}"; do
  [ -n "$value" ] || fail "empty runtime path"
  case "$value" in /*) path="$value" ;; *) path="$root/$value" ;; esac
  is_inside_project "$path" || fail "path escapes project: $path"
done

binary="$runtime/bin/kvslim-server"
worker_binary="$runtime/bin/kvslim-runtime-worker"
[ -f "$binary" ] || fail "missing binary: $binary"
[ -x "$binary" ] || fail "binary is not executable: $binary"
[ -f "$worker_binary" ] || fail "missing worker binary: $worker_binary"
[ -x "$worker_binary" ] || fail "worker binary is not executable: $worker_binary"

case "${KVSLIM_MODEL_DIR:-}" in
  /*) model_path="$KVSLIM_MODEL_DIR" ;;
  *) model_path="$root/$KVSLIM_MODEL_DIR" ;;
esac
case "$model_path" in "$runtime"/*) ;; *) fail "model path is outside runtime" ;; esac
[ -d "$model_path" ] || fail "missing model directory: $model_path"

actual_binary=$(sha256sum "$binary" | awk '{print $1}')
[ "$actual_binary" = "${KVSLIM_BINARY_DIGEST:-}" ] || fail "binary digest mismatch"
actual_worker_binary=$(sha256sum "$worker_binary" | awk '{print $1}')
[ "$actual_worker_binary" = "${KVSLIM_WORKER_BINARY_DIGEST:-}" ] ||
  fail "worker binary digest mismatch"

model_digest=$(find "$model_path" -type f -not -path '*/.git/*' -print0 | sort -z | sha256sum | awk '{print $1}')
[ "$model_digest" = "${KVSLIM_MODEL_DIGEST:-}" ] || fail "model tree digest mismatch"

if find "$model_path" -type f -perm /022 -print -quit | grep -q .; then
  fail "model files must not be group/world writable"
fi

echo "VERIFY-OK: local release is structurally valid"
echo "binary_sha256=$actual_binary"
echo "worker_binary_sha256=$actual_worker_binary"
echo "model_tree_sha256=$model_digest"
echo "host=$KVSLIM_HOST"
echo "lod=$KVSLIM_ENABLE_LOD"
