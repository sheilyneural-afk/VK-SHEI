#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"
env_file="$root/deploy/local-production/local-production.env"

[ -f "$env_file" ] || { echo "missing $env_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

case "${MSLIM_MODEL_DIR:?MSLIM_MODEL_DIR must be set}" in
  /*) model_dir="$MSLIM_MODEL_DIR" ;;
  *) model_dir="$root/$MSLIM_MODEL_DIR" ;;
esac
model_digest=$(basename -- "$model_dir")
actual_model_digest=$(python3 "$root/scripts/model-bundle-digest.py" "$model_dir")
[ "$actual_model_digest" = "$model_digest" ] || {
  echo "model content digest does not match its directory" >&2
  exit 1
}

# Bind the binaries to one stable source snapshot. A writer racing the build
# must invalidate the release instead of producing a manifest for different
# inputs than the compiler observed.
build_input_digest_before=$(python3 "$root/scripts/source_inventory.py" --print-build-input-digest)

"$root/deploy/local-production/scripts/cargo-local.sh" build --locked --release \
  --bin mslim-runtime-worker --features accelerator-backend
"$root/deploy/local-production/scripts/cargo-local.sh" build --locked --release \
  --bin mslim-gateway --features unified-router

build_input_digest=$(python3 "$root/scripts/source_inventory.py" --print-build-input-digest)
[ "$build_input_digest" = "$build_input_digest_before" ] || {
  echo "source or build input changed while compiling; release not installed" >&2
  exit 1
}

# Do not replace the resident release until both builds have completed from one
# stable input snapshot. The temporary names keep each individual replacement
# atomic after that gate has passed.
mkdir -p "$runtime/bin"
install -m 0755 "$runtime/build-target/release/mslim-runtime-worker" "$runtime/bin/.mslim-runtime-worker.new"
install -m 0755 "$runtime/build-target/release/mslim-gateway" "$runtime/bin/.mslim-gateway.new"
mv -f "$runtime/bin/.mslim-runtime-worker.new" "$runtime/bin/mslim-runtime-worker"
mv -f "$runtime/bin/.mslim-gateway.new" "$runtime/bin/mslim-gateway"

python3 "$root/scripts/write-local-release.py" \
  --root "$root" \
  --build-input-digest "$build_input_digest" \
  --model-digest "$model_digest"

"$script_dir/verify-release.sh"
