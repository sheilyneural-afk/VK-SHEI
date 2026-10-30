#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)

# Cargo registry, git checkouts, downloads and build products stay in the
# project. The wrapper reuses them across corrections; it never recreates the
# dependency environment unless Cargo itself needs a new package.
export CARGO_HOME="$root/deploy/local-production/runtime/cargo-home"
export CARGO_TARGET_DIR="$root/deploy/local-production/runtime/build-target"
export TMPDIR="$root/deploy/local-production/runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$TMPDIR"
exec cargo "$@"
