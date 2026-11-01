#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
env_file="$root/deploy/local-production/local-production.env"

[ -f "$env_file" ] || { echo "missing $env_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

"$script_dir/verify-release.sh"

if [ -n "${MSLIM_RUNTIME_MANIFEST:-}" ]; then
  [ -f "$MSLIM_RUNTIME_MANIFEST" ] || { echo "runtime manifest missing: $MSLIM_RUNTIME_MANIFEST" >&2; exit 1; }
  "$script_dir/verify-runtime-manifest.sh" "$MSLIM_RUNTIME_MANIFEST"
fi

binary="$root/deploy/local-production/runtime/bin/mslim-server"
: "${MSLIM_GATEWAY_TOKEN:?MSLIM_GATEWAY_TOKEN must be set}"
export MSLIM_GATEWAY_TOKEN

backend="${MSLIM_BACKEND:-auto}"
configured="${MSLIM_WORKER_ENDPOINT:-}"
declare -a candidates=()
if [ -n "$configured" ]; then
  candidates+=("$configured")
fi
if [ "$backend" = "auto" ] || [ "$backend" = "accelerator" ]; then
  candidates+=("http://127.0.0.1:8188")
fi
if [ "$backend" = "auto" ] || [ "$backend" = "cpu" ]; then
  candidates+=("http://127.0.0.1:8099")
fi

selected=""
for endpoint in "${candidates[@]}"; do
  if curl --silent --show-error --fail --max-time 2 "$endpoint/ready" >/dev/null 2>&1 &&
     curl --silent --show-error --fail --max-time 2 "$endpoint/v1/capabilities" >/dev/null 2>&1; then
    selected="$endpoint"
    break
  fi
done

if [ -z "$selected" ]; then
  echo "No hay ningún worker local certificado disponible." >&2
  echo "M-SLIM no exige un backend concreto: inicia un worker compatible y vuelve a ejecutar este script." >&2
  exit 1
fi

exec "$binary" \
  --host 127.0.0.1 \
  --port "${MSLIM_PORT:-8088}" \
  --worker "auto=$selected"
