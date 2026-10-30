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
: "${KVSLIM_GATEWAY_TOKEN:?KVSLIM_GATEWAY_TOKEN must be set}"
: "${KVSLIM_WORKER_ENDPOINT:?KVSLIM_WORKER_ENDPOINT must be set (for example http://127.0.0.1:8188)}"
exec "$binary" \
  --host 127.0.0.1 \
  --port "${KVSLIM_PORT:-8088}" \
  --worker "local=${KVSLIM_WORKER_ENDPOINT}"
