#!/usr/bin/env bash
# deploy/local-production/scripts/cargo-local.sh
#
# Wrapper around cargo that:
#   1. Keeps Cargo home, temporaries and build artefacts inside the project.
#   2. Uses sccache as RUSTC_WRAPPER when it is installed in runtime/tool-bin
#      (compiler object cache shared by debug and release builds, 5 GB cap).
#   3. Disables incremental compilation for --release builds.
#   4. After commands that produce artefacts, prunes in the background the
#      build artefacts not touched for 7 days (cargo-sweep, when installed).
#
# Usage:  ./cargo-local.sh build --release [extra flags]
#         ./cargo-local.sh test
#         ./cargo-local.sh sweep          # prune artefacts older than 7 days now
#
# Every other command, `clean` included, is passed to cargo unchanged.
# sccache and cargo-sweep are installed with install-tools.sh.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)

# ── Project-local runtime dirs ────────────────────────────────────────────────
CARGO_HOME_LOCAL="$root/deploy/local-production/runtime/cargo-home"
BUILD_TARGET="$root/deploy/local-production/runtime/build-target"
TOOL_BIN="$root/deploy/local-production/runtime/tool-bin"
TMPDIR_LOCAL="$root/deploy/local-production/runtime/tmp"
SWEEP_DAYS=7

export CARGO_HOME="$CARGO_HOME_LOCAL"
export CARGO_TARGET_DIR="$BUILD_TARGET"
export TMPDIR="$TMPDIR_LOCAL"
export TMP="$TMPDIR_LOCAL"
export TEMP="$TMPDIR_LOCAL"

mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$TMPDIR" "$TOOL_BIN"

# cargo-sweep is a cargo subcommand binary: its first argument must be
# `sweep`. It locates the target directory through `cargo metadata`, which
# honours CARGO_TARGET_DIR exported above.
sweep_old_artefacts() {
  "$TOOL_BIN/cargo-sweep" sweep --time "$SWEEP_DAYS" "$root"
}

# ── sccache (compiler object cache) ──────────────────────────────────────────
SCCACHE_BIN="$TOOL_BIN/sccache"
if [[ -x "$SCCACHE_BIN" ]]; then
  export RUSTC_WRAPPER="$SCCACHE_BIN"
  export SCCACHE_DIR="$root/deploy/local-production/runtime/sccache-cache"
  export SCCACHE_CACHE_SIZE="5G"
  mkdir -p "$SCCACHE_DIR"
fi

# ── Release builds: never keep incremental state ─────────────────────────────
for arg in "$@"; do
  if [[ "$arg" == "--release" ]]; then
    export CARGO_INCREMENTAL=0
    break
  fi
done

# ── Synthetic command ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "sweep" ]]; then
  if [[ ! -x "$TOOL_BIN/cargo-sweep" ]]; then
    echo "[cargo-local] cargo-sweep is not installed; run install-tools.sh" >&2
    exit 1
  fi
  echo "[cargo-local] pruning artefacts not used for $SWEEP_DAYS days"
  sweep_old_artefacts
  exit 0
fi

# ── Run cargo ─────────────────────────────────────────────────────────────────
# With `set -e` a failing cargo command ends the script with cargo's status.
cargo "$@"

# ── Post-build sweep (background, non-blocking) ───────────────────────────────
case "${1:-}" in
  fmt|check|clippy|doc|metadata|locate-project|read-manifest|verify-project|clean)
    ;;
  *)
    if [[ -x "$TOOL_BIN/cargo-sweep" ]]; then
      # Only artefacts untouched for SWEEP_DAYS are removed, so a build that
      # starts while the sweep runs never loses files it is using.
      sweep_old_artefacts >/dev/null 2>&1 &
    fi
    ;;
esac
