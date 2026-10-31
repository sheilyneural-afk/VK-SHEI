#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# Provider-neutral prefill profile selector.
# The command is supplied by the runtime, never hard-coded here.
: "${KVSLIM_BENCH_CMD:?Set KVSLIM_BENCH_CMD to the runtime benchmark command}"
timeout_s="${KVSLIM_BENCH_TIMEOUT_SECS:-120}"
blocks="${KVSLIM_PREFILL_BLOCK_CANDIDATES:-64 128 256}"
modes="${KVSLIM_WEIGHT_MODES:-stream cached resident}"
out="${KVSLIM_BENCH_OUTPUT:-deploy/local-production/runtime/validation/prefill-profile.json}"
mkdir -p "$(dirname "$out")"

best_mode=""
best_block=""
best_seconds=""
results=()

for mode in $modes; do
  for block in $blocks; do
    start=$(date +%s%N)
    if KVSLIM_WEIGHT_MODE="$mode" KVSLIM_PREFILL_BLOCK_TOKENS="$block"       timeout "$timeout_s" bash -lc "$KVSLIM_BENCH_CMD" >/tmp/kvslim-bench.out 2>/tmp/kvslim-bench.err; then
      end=$(date +%s%N)
      seconds=$(awk "BEGIN {printf \"%.6f\", ($end-$start)/1000000000}")
      results+=($(printf '{"mode":"%s","block_tokens":%s,"seconds":%s,"ok":true}' "$mode" "$block" "$seconds"))
      if [ -z "$best_seconds" ] || awk "BEGIN {exit !($seconds < $best_seconds)}"; then
        best_mode="$mode"; best_block="$block"; best_seconds="$seconds"
      fi
    else
      results+=($(printf '{"mode":"%s","block_tokens":%s,"seconds":null,"ok":false}' "$mode" "$block"))
    fi
  done
done

[ -n "$best_mode" ] || { echo "No valid prefill profile" >&2; exit 1; }
{
  printf '{"schema":"kvslim.prefill.profile/v1","selected":{"mode":"%s","block_tokens":%s,"seconds":%s},"candidates":[' "$best_mode" "$best_block" "$best_seconds"
  (IFS=,; printf '%s' "${results[*]}")
  printf ']}\n'
} > "$out"
cat "$out"
