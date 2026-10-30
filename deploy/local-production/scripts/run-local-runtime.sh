#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
env_file="$root/deploy/local-production/local-production.env"

[ -f "$env_file" ] || { echo "missing $env_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

"$script_dir/verify-release.sh"

binary="$root/deploy/local-production/runtime/bin/kvslim-server"
case "${KVSLIM_MODEL_DIR:-}" in
  /*) model_dir="$KVSLIM_MODEL_DIR" ;;
  *) model_dir="$root/$KVSLIM_MODEL_DIR" ;;
esac
exec "$binary" \
  --model "$model_dir" \
  --host 127.0.0.1 \
  --port "${KVSLIM_PORT:-8088}" \
  --max-context-tokens "${KVSLIM_MAX_CONTEXT_TOKENS:-8192}" \
  --timeout-secs "${KVSLIM_REQUEST_TIMEOUT_SECS:-300}"
