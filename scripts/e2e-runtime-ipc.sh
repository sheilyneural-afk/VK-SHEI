#!/usr/bin/env bash
set -euo pipefail

# Real Unix-socket E2E for the generic runtime host.
# This harness intentionally uses the deterministic host session and proves
# framing, request routing, session identity, generation and KV restore.
# It does not claim model inference; the external runtime must provide that.

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sock="${MSLIM_E2E_SOCKET:-/tmp/mslim-runtime-e2e.sock}"
log="${MSLIM_E2E_LOG:-${root}/deploy/local-production/runtime/logs/runtime-e2e.log}"
mkdir -p "$(dirname -- "$log")"
rm -f "$sock" "$log"

cargo run --offline --locked --features accelerator-backend \
  --bin mslim-runtime-host -- --socket "$sock" >"$log" 2>&1 &
host_pid=$!
cleanup() {
  kill "$host_pid" 2>/dev/null || true
  wait "$host_pid" 2>/dev/null || true
  rm -f "$sock"
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  [ -S "$sock" ] && break
  kill -0 "$host_pid" 2>/dev/null || { cat "$log" >&2; exit 1; }
  sleep 0.2
done
[ -S "$sock" ] || { cat "$log" >&2; echo "host socket not created" >&2; exit 1; }

python3 - "$sock" <<'PY'
import json, socket, struct, sys

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])

def call(operation, request_id, payload=None, session_id=None):
    request = {
        "schema": "mslim.runtime.ipc/v1",
        "operation": operation,
        "request_id": request_id,
        "session_id": session_id,
        "tensors": [],
        "payload": payload,
    }
    body = json.dumps(request, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(body)) + body)
    header = sock.recv(4)
    if len(header) != 4:
        raise SystemExit(f"{operation}: missing response header")
    size = struct.unpack(">I", header)[0]
    response = b""
    while len(response) < size:
        chunk = sock.recv(size - len(response))
        if not chunk:
            raise SystemExit(f"{operation}: truncated response")
        response += chunk
    result = json.loads(response)
    if result.get("request_id") != request_id:
        raise SystemExit(f"{operation}: request id mismatch")
    if not result.get("ok"):
        raise SystemExit(f"{operation}: {result.get('error')}")
    return result

probe = call("probe", "e2e-probe")
caps = probe["payload"]["capabilities"]
assert caps["supports_prefill"] and caps["supports_decode"] and caps["supports_kv_restore"]

opened = call("open_session", "e2e-open")
session = opened["session_id"]
assert session

generated = call(
    "prefill",
    "e2e-prefill",
    {"input_tokens": [11, 22, 33], "max_new_tokens": 1,
     "temperature": None, "seed": None},
    session,
)
assert generated["payload"]["output_tokens"] == [33]

snapshot = call("snapshot_kv", "e2e-snapshot", None, session)
payload = snapshot["payload"]
call("restore_kv", "e2e-restore",
     {"snapshot_bytes": payload["snapshot_bytes"], "receipt": payload["receipt"]},
     session)
call("release", "e2e-release", session_id=session)
print("IPC_E2E_OK")
PY
