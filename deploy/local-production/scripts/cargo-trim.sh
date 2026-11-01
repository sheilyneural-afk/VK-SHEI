#!/usr/bin/env bash
# deploy/local-production/scripts/cargo-trim.sh
#
# Manual deep-clean of the project-local Cargo target directory.
# Run this when disk usage grows unexpectedly or before a full release build.
#
# What it does (in order, each step idempotent):
#   1. cargo-sweep --time N   — remove artefacts not touched in N days
#   2. Remove .d / .rmeta files orphaned by renamed crates
#   3. Remove incremental/ directories from release profiles (safety net)
#   4. Optionally wipe the sccache object store and start fresh (--reset-sccache)
#   5. Print before/after disk usage
#
# Usage:
#   ./cargo-trim.sh                  # sweep artefacts older than 7 days
#   ./cargo-trim.sh --days 3         # tighter window
#   ./cargo-trim.sh --reset-sccache  # also clear the compiler object cache
#   ./cargo-trim.sh --full           # remove ALL build artefacts (full clean)
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)

BUILD_TARGET="$root/deploy/local-production/runtime/build-target"
SCCACHE_DIR="$root/deploy/local-production/runtime/sccache-cache"
TOOL_BIN="$root/deploy/local-production/runtime/tool-bin"

DAYS=7
RESET_SCCACHE=0
FULL_CLEAN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)        DAYS="$2"; shift 2 ;;
    --days=*)      DAYS="${1#*=}"; shift ;;
    --reset-sccache) RESET_SCCACHE=1; shift ;;
    --full)        FULL_CLEAN=1; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
du_human() { du -sh "$1" 2>/dev/null | cut -f1 || echo "0"; }

section() { echo; echo "── $* ─────────────────────────────────────────────"; }

# ── Before snapshot ───────────────────────────────────────────────────────────
section "Before"
echo "  build-target : $(du_human "$BUILD_TARGET")"
echo "  sccache-cache: $(du_human "$SCCACHE_DIR")"

if (( FULL_CLEAN )); then
  section "Full clean (removing all build artefacts)"
  echo "  !! This will delete $BUILD_TARGET"
  echo "  !! Next build will recompile everything from scratch."
  read -r -p "  Proceed? [y/N] " _confirm
  [[ "$_confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
  rm -rf "$BUILD_TARGET"
  mkdir -p "$BUILD_TARGET"
  echo "  Done."
else
  # ── Step 1: cargo-sweep ───────────────────────────────────────────────────
  section "cargo-sweep (artefacts older than ${DAYS} days)"
  SWEEP_BIN="$TOOL_BIN/cargo-sweep"
  if [[ -x "$SWEEP_BIN" ]]; then
    # cargo-sweep needs its `sweep` subcommand and finds the target directory
    # through `cargo metadata`; use the same Cargo home and target as builds.
    CARGO_HOME="$root/deploy/local-production/runtime/cargo-home" \
      CARGO_TARGET_DIR="$BUILD_TARGET" \
      "$SWEEP_BIN" sweep --time "$DAYS" "$root"
  else
    echo "  cargo-sweep not found in $TOOL_BIN — skipping."
    echo "  Run: deploy/local-production/scripts/install-tools.sh"
  fi

  # ── Step 2: Orphaned incremental/ dirs inside release profiles ────────────
  section "Removing incremental/ dirs from release profiles"
  find "$BUILD_TARGET" \
    -type d -name incremental \
    -path "*/release/*" \
    -print -exec rm -rf {} + \
    2>/dev/null || true

  # ── Step 3: Stale .rmeta files (> N days, not touched by last build) ──────
  section "Pruning stale .rmeta / .d files (older than ${DAYS} days)"
  find "$BUILD_TARGET" \
    \( -name '*.rmeta' -o -name '*.d' \) \
    -mtime "+$DAYS" \
    -print -delete \
    2>/dev/null || true
fi

# ── Step 4: sccache reset (optional) ─────────────────────────────────────────
if (( RESET_SCCACHE )); then
  section "Resetting sccache object store"
  SCCACHE_BIN="$TOOL_BIN/sccache"
  if [[ -x "$SCCACHE_BIN" ]]; then
    "$SCCACHE_BIN" --stop-server 2>/dev/null || true
    rm -rf "$SCCACHE_DIR"
    mkdir -p "$SCCACHE_DIR"
    echo "  sccache cache cleared."
  else
    rm -rf "$SCCACHE_DIR"
    mkdir -p "$SCCACHE_DIR"
    echo "  sccache cache cleared (binary not found, cleared directory only)."
  fi
fi

# ── After snapshot ────────────────────────────────────────────────────────────
section "After"
echo "  build-target : $(du_human "$BUILD_TARGET")"
echo "  sccache-cache: $(du_human "$SCCACHE_DIR")"
echo
echo "✓ cargo-trim complete."
