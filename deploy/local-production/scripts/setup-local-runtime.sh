#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"
env_file="$root/deploy/local-production/local-production.env"
report="$runtime/state/setup-report.json"
"$script_dir/install-layout.sh" >/dev/null
if [ ! -f "$env_file" ]; then
  cp "$root/deploy/local-production/env.example" "$env_file"
  if command -v openssl >/dev/null 2>&1; then
    token=$(openssl rand -hex 32)
    sed -i "s/KVSLIM_GATEWAY_TOKEN=REPLACE_WITH_LOCAL_SECRET/KVSLIM_GATEWAY_TOKEN=$token/" "$env_file"
  fi
  chmod 0600 "$env_file"
fi
# El archivo local tiene prioridad sobre variables heredadas para evitar activar
# accidentalmente un manifest de otra instalación.
# shellcheck disable=SC1090
source "$env_file"
os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
mem_kb=$(awk '/MemTotal:/{print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
opencl="unavailable"
if command -v clinfo >/dev/null 2>&1 && clinfo >/dev/null 2>&1; then opencl="available"; fi
vulkan="unavailable"
if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then vulkan="available"; fi
cpu="available"
command -v cargo >/dev/null 2>&1 || cpu="unavailable"
manifest="${KVSLIM_RUNTIME_MANIFEST:-}"
if [ -n "$manifest" ]; then "$script_dir/verify-runtime-manifest.sh" "$manifest"; fi
if [ -n "$manifest" ]; then manifest_status="verified"; else manifest_status="not-selected"; fi
if command -v jq >/dev/null 2>&1; then
  jq -n --arg os "$os" --arg arch "$arch" --arg opencl "$opencl" --arg vulkan "$vulkan" --arg cpu "$cpu" --arg manifest "$manifest_status" --argjson mem_kb "$mem_kb" \
    '{schema:"kvslim.setup.report/v1",os:$os,architecture:$arch,memory_kb:$mem_kb,capabilities:{opencl:$opencl,vulkan:$vulkan,cpu:$cpu},runtime_manifest:$manifest,drivers_are_external:true}' > "$report"
else
  printf '{"schema":"kvslim.setup.report/v1","os":"%s","architecture":"%s","memory_kb":%s,"runtime_manifest":"%s","drivers_are_external":true}\n' "$os" "$arch" "$mem_kb" "$manifest_status" > "$report"
fi
chmod 0640 "$report"
echo "KV-SHEI preparado: $report"
echo "Sistema: $os/$arch; OpenCL: $opencl; Vulkan: $vulkan; CPU: $cpu"
if [ "$manifest_status" = "not-selected" ]; then echo "No se ha activado un runtime externo: falta seleccionar un manifest verificado."; fi
