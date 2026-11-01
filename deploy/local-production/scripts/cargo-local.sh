#!/usr/bin/env bash
# deploy/local-production/scripts/cargo-local.sh
#
# Wrapper around cargo that:
#   1. Keeps ALL build artefacts inside the project tree.
#   2. Activates sccache when it is installed in the project tool-bin
#      (compiler cache shared across dev + release builds).
#   3. After every non-clean command, runs cargo-sweep to prune artefacts
#      that are no longer referenced by any current build graph fingerprint.
#
# Usage:  ./cargo-local.sh build --release [extra flags]
#         ./cargo-local.sh test
#         ./cargo-local.sh clean          # sweep only; never removes src
#         ./cargo-local.sh sweep          # alias for above
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)

# ── Project-local runtime dirs ────────────────────────────────────────────────
CARGO_HOME_LOCAL="$root/deploy/local-production/runtime/cargo-home"
BUILD_TARGET="$root/deploy/local-production/runtime/build-target"
TOOL_BIN="$root/deploy/local-production/runtime/tool-bin"
TMPDIR_LOCAL="$root/deploy/local-production/runtime/tmp"

export CARGO_HOME="$CARGO_HOME_LOCAL"
export CARGO_TARGET_DIR="$BUILD_TARGET"
export TMPDIR="$TMPDIR_LOCAL"
export TMP="$TMPDIR_LOCAL"
export TEMP="$TMPDIR_LOCAL"

mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$TMPDIR" "$TOOL_BIN"

# ── sccache (compiler object cache) ──────────────────────────────────────────
SCCACHE_BIN="$TOOL_BIN/sccache"
if [[ -x "$SCCACHE_BIN" ]]; then
  export RUSTC_WRAPPER="$SCCACHE_BIN"
  export SCCACHE_DIR="$root/deploy/local-production/runtime/sccache-cache"
  export SCCACHE_CACHE_SIZE="5G"
  mkdir -p "$SCCACHE_DIR"
fi

# ── Release builds: never keep incremental state ─────────────────────────────
# Detect if --release is on the command line (including after subcommands).
_is_release=0
for arg in "$@"; do
  [[ "$arg" == "--release" ]] && _is_release=1 && break
done
if (( _is_release )); then
  export CARGO_INCREMENTAL=0
fi

# ── Handle synthetic commands ─────────────────────────────────────────────────
case "${1:-}" in
  sweep|clean)
    echo "[cargo-local] Running cargo-sweep (stamp + sweep)…"
    _do_sweep "$root" "$TOOL_BIN" "$BUILD_TARGET"
    exit 0
    ;;
esac

# ── Run cargo ─────────────────────────────────────────────────────────────────
cargo "$@"
_cargo_exit=$?

# ── Post-build sweep (background, non-blocking) ───────────────────────────────
# Skip on commands that don't produce artefacts (fmt, clippy --fix, …).
_skip_sweep=0
case "${1:-}" in
  fmt|check|clippy|doc|metadata|locate-project|read-manifest|verify-project)
    _skip_sweep=1 ;;
esac

if (( ! _skip_sweep )) && [[ -x "$TOOL_BIN/cargo-sweep" ]]; then
  (
    # Stamp must be set immediately before the build and used after.
    # Since we can't stamp before cargo, we use --time 7 (days) instead:
    # artefacts not touched in the last 7 days are orphans.
    "$TOOL_BIN/cargo-sweep" \
      --target-dir "$BUILD_TARGET" \
      --time 7 \
      2>/dev/null &
  )
fi

exit $_cargo_exit

# ── Helper ────────────────────────────────────────────────────────────────────
_do_sweep() {
  local _root="$1" _tool_bin="$2" _target="$3"
  local sweep_bin="$_tool_bin/cargo-sweep"
  if [[ ! -x "$sweep_bin" ]]; then
    echo "[cargo-local] cargo-sweep not installed in tool-bin; skipping." >&2
    return 0
  fi
  "$sweep_bin" --target-dir "$_target" --time 7
}
