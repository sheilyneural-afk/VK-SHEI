#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"

case "$runtime" in
  "$root"/*) ;;
  *) echo "runtime escapes project root" >&2; exit 1 ;;
esac

mkdir -p \
  "$runtime/bin" \
  "$runtime/models" \
  "$runtime/coldstore/default" \
  "$runtime/state" \
  "$runtime/logs" \
  "$runtime/receipts"

chmod 0750 "$runtime" "$runtime/bin" "$runtime/models" "$runtime/coldstore" "$runtime/state" "$runtime/logs" "$runtime/receipts"
echo "Created local runtime layout under: $runtime"
