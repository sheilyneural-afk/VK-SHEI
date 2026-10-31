#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# Real external-runtime certification:
#   provider (resident/stream) -> HTTP worker -> SessionStore -> worker restart -> restore
# The model is never hard-coded. Set KVSLIM_MODEL_DIR to a verified local bundle.

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model="${KVSLIM_MODEL_DIR:?Set KVSLIM_MODEL_DIR to a verified model bundle}"
port="${KVSLIM_EXTERNAL_E2E_PORT:-18790}"
socket="${KVSLIM_EXTERNAL_E2E_SOCKET:-/tmp/kvslim-real-provider-e2e.sock}"
weight_mode="${KVSLIM_WEIGHT_MODE:-stream}"
timeout_secs="${KVSLIM_EXTERNAL_E2E_TIMEOUT_SECS:-600}"
output_root="$root/deploy/local-production/runtime/validation/external-e2e"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
out_dir="${KVSLIM_EXTERNAL_E2E_OUTPUT:-$output_root/$run_id}"
session_store="${KVSLIM_EXTERNAL_E2E_SESSION_STORE:-$out_dir/sessions}"
cold_dir="${KVSLIM_EXTERNAL_E2E_COLD_DIR:-$out_dir/coldstore}"
provider_log="$out_dir/provider.log"
worker_log="$out_dir/worker.log"
manifest="$out_dir/runtime-manifest.json"
generation_json="$out_dir/generation.json"
snapshot_json="$out_dir/snapshot.json"
resume_json="$out_dir/resume.json"
worker_pid=""
provider_pid=""
session_handle="external-http-certification-$RANDOM-$$"

mkdir -p "$out_dir" "$session_store" "$cold_dir"
rm -f "$socket" "$manifest" "$generation_json" "$snapshot_json" "$resume_json"

cleanup() {
  kill "$worker_pid" "$provider_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
  wait "$provider_pid" 2>/dev/null || true
  rm -f "$socket"
}
trap cleanup EXIT

cargo build --offline --locked --features accelerator-backend \
  --bin kvslim-runtime-provider --bin kvslim-runtime-worker

provider="$root/deploy/local-production/runtime/build-target/debug/kvslim-runtime-provider"
worker="$root/deploy/local-production/runtime/build-target/debug/kvslim-runtime-worker"

python3 - "$manifest" "$socket" <<'PY'
import json, sys
path, endpoint = sys.argv[1:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({
        "schema": "kvslim.runtime.manifest/v1",
        "id": "real-runtime-provider",
        "capabilities": ["prefill", "decode", "snapshot_kv", "restore_kv"],
        "priority": 100,
        "protocol": "kvslim.runtime.ipc/v1",
        "endpoint": endpoint,
    }, f)
PY

nohup "$provider" --model "$model" --socket "$socket" \
  --weight-mode "$weight_mode" --cold-dir "$cold_dir" >"$provider_log" 2>&1 < /dev/null &
provider_pid=$!
for _ in $(seq 1 600); do
  [ -S "$socket" ] && break
  kill -0 "$provider_pid" 2>/dev/null || { cat "$provider_log" >&2; exit 1; }
  sleep 1
done
[ -S "$socket" ] || { cat "$provider_log" >&2; exit 1; }

start_worker() {
  nohup env KVSLIM_SESSION_STORE="$session_store" "$worker" \
    --model "$model" --runtime-manifest "$manifest" --port "$port" \
    --timeout-secs "$timeout_secs" >"$worker_log" 2>&1 < /dev/null &
  worker_pid=$!
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      return 0
    fi
    kill -0 "$worker_pid" 2>/dev/null || { cat "$worker_log" >&2; return 1; }
    sleep 1
  done
  cat "$worker_log" >&2
  return 1
}

start_worker

model_digest=$(sha256sum "$model/config.json" | awk '{print $1}')
tokenizer_digest=$(sha256sum "$model/tokenizer.json" | awk '{print $1}')
expires_at=$(($(date +%s) + 3600))
python3 - "$port" "$timeout_secs" >"$generation_json" <<'PY'
import json, sys, time, urllib.error, urllib.request
port, timeout = sys.argv[1], int(sys.argv[2])
body = {
    "messages": [{"role": "user", "content": "Responde solo con una palabra."}],
    "max_tokens": 1,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
started = time.monotonic()
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode(errors="replace"), file=sys.stderr)
    raise
result["measured_wall_seconds"] = time.monotonic() - started
json.dump(result, sys.stdout, sort_keys=True)
print()
assert result["object"] == "chat.completion"
assert result["usage"]["completion_tokens"] == 1
PY

python3 - "$port" "$expires_at" "$model_digest" "$tokenizer_digest" "$session_handle" >"$snapshot_json" <<'PY'
import json, sys, urllib.error, urllib.request
port, expires, model_digest, tokenizer_digest, session_handle = sys.argv[1:]
z = "0" * 64
body = {
    "messages": [],
    "continuity": {
        "schema": "kvslim.continuity.request/v1",
        "operation": "snapshot",
        "opaque_session_handle": session_handle,
        "expected_generation": 0,
        "model_digest": model_digest,
        "tokenizer_digest": tokenizer_digest,
        "prefix_digest": z,
        "policy_digest": z,
        "expires_at_unix_s": int(expires),
        "bounds": {"max_reused_tokens": 4096, "max_restored_bytes": 268435456},
    },
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode(errors="replace"), file=sys.stderr)
    raise
json.dump(result, sys.stdout, sort_keys=True)
print()
assert result["operation"] == "snapshot"
PY

kill "$worker_pid"
wait "$worker_pid" 2>/dev/null || true
worker_pid=""
start_worker

python3 - "$port" "$expires_at" "$model_digest" "$tokenizer_digest" "$session_handle" >"$resume_json" <<'PY'
import json, sys, urllib.error, urllib.request
port, expires, model_digest, tokenizer_digest, session_handle = sys.argv[1:]
z = "0" * 64
body = {
    "messages": [],
    "continuity": {
        "schema": "kvslim.continuity.request/v1",
        "operation": "resume",
        "opaque_session_handle": session_handle,
        "expected_generation": 0,
        "model_digest": model_digest,
        "tokenizer_digest": tokenizer_digest,
        "prefix_digest": z,
        "policy_digest": z,
        "expires_at_unix_s": int(expires),
        "bounds": {"max_reused_tokens": 4096, "max_restored_bytes": 268435456},
    },
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode(errors="replace"), file=sys.stderr)
    raise
json.dump(result, sys.stdout, sort_keys=True)
print()
assert result["operation"] == "resume"
assert result["receipt"]["restored_bytes"] > 0
PY

python3 - "$generation_json" "$snapshot_json" "$resume_json" <<'PY'
import json, sys
generation, snapshot, resume = (json.load(open(p)) for p in sys.argv[1:])
print("REAL_EXTERNAL_RESTART_E2E_OK")
print(json.dumps({
    "generation_wall_seconds": generation["measured_wall_seconds"],
    "prefill_seconds": generation["usage"]["prefill_seconds"],
    "decode_seconds": generation["usage"]["runtime_diagnostics"]["stats"]["decode_seconds"],
    "snapshot_generation": snapshot["receipt"]["resulting_generation"],
    "restored_bytes": resume["receipt"]["restored_bytes"],
}, sort_keys=True))
PY
