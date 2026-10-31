#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
env_file="$root/deploy/local-production/local-production.env"
[ -f "$env_file" ] || { echo "missing $env_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

binary="$root/deploy/local-production/runtime/bin/mslim-runtime-worker"
case "${MSLIM_MODEL_DIR:?MSLIM_MODEL_DIR must be set}" in
  /*) model_dir="$MSLIM_MODEL_DIR" ;;
  *) model_dir="$root/$MSLIM_MODEL_DIR" ;;
esac
case "${MSLIM_COLDSTORE_DIR:?MSLIM_COLDSTORE_DIR must be set}" in
  /*) cold_dir="$MSLIM_COLDSTORE_DIR" ;;
  *) cold_dir="$root/$MSLIM_COLDSTORE_DIR" ;;
esac
mkdir -p "$cold_dir"

declare -a profile_args=()
if [ "${MSLIM_ENABLE_LOD:-false}" = "true" ]; then
  profile_args+=(
    --gpu-lod
    --lod-cold-dir "$cold_dir"
    --lod-epsilon "${MSLIM_LOD_EPSILON:-0}"
    --lod-stage "${MSLIM_LOD_STAGE_TOKENS:-64}"
  )
else
  profile_args+=(--gpu-dense-kv)
fi

if [ "${MSLIM_PREFILL_BATCH_TOKENS:-0}" -gt 0 ]; then
  profile_args+=(--gpu-prefill-batch "$MSLIM_PREFILL_BATCH_TOKENS")
else
  profile_args+=(--gpu-prefill-fallback)
fi
profile_args+=(
  --gpu-prefill-weight-reuse "${MSLIM_PREFILL_WEIGHT_REUSE_TOKENS:-1}"
  --gpu-hot-v-tokens "${MSLIM_HOT_V_TOKENS:-0}"
  --gpu-verified-prompt-replay "${MSLIM_VERIFIED_PROMPT_REPLAY_TOKENS:-0}"
)
# The micro hot tier is valid only for explicitly approximate LOD. Exact
# epsilon-zero uses the separate exact hot-V tier above.
if [ "${MSLIM_ENABLE_LOD:-false}" = "true" ] &&
   [ "${MSLIM_LOD_EPSILON:-0}" != "0" ] &&
   [ "${MSLIM_MICRO_HOT_V_TOKENS:-0}" -gt 0 ]; then
  profile_args+=(--gpu-micro-hot-v-tokens "$MSLIM_MICRO_HOT_V_TOKENS")
else
  profile_args+=(--gpu-micro-hot-v-tokens 0)
fi

[ "${MSLIM_VERIFIED_SHARED_PREFIX:-false}" = "true" ] &&
  profile_args+=(--gpu-verified-shared-prefix) ||
  profile_args+=(--gpu-no-shared-prefix)
[ "${MSLIM_DECODE_UNROLL4:-false}" = "true" ] &&
  profile_args+=(--gpu-decode-unroll4) ||
  profile_args+=(--gpu-decode-original)
[ "${MSLIM_DEFERRED_COLD:-false}" = "true" ] &&
  profile_args+=(--gpu-deferred-cold) ||
  profile_args+=(--gpu-immediate-cold)
[ "${MSLIM_FUSED_DECODE:-false}" = "true" ] &&
  profile_args+=(--gpu-fused-decode) ||
  profile_args+=(--gpu-unfused-decode)
[ "${MSLIM_EXACT_HOT_FUSED_ATTENTION:-false}" = "true" ] &&
  profile_args+=(--gpu-exact-fused-attention) ||
  profile_args+=(--gpu-exact-original-attention)

exec "$binary" \
  --model "$model_dir" \
  --host 127.0.0.1 \
  --port "${MSLIM_WORKER_PORT:-8188}" \
  --max-context-tokens "${MSLIM_MAX_CONTEXT_TOKENS:-8192}" \
  --timeout-secs "${MSLIM_REQUEST_TIMEOUT_SECS:-300}" \
  "${profile_args[@]}"
