#!/usr/bin/env bash
# deploy/local-production/scripts/install-tools.sh
#
# Installs project-local build tools (sccache, cargo-sweep) into
# deploy/local-production/runtime/tool-bin/
#
# These tools are NOT installed system-wide. The cargo-local.sh wrapper
# picks them up automatically from tool-bin once they are installed.
#
# Usage:
#   ./install-tools.sh           # install / update all tools
#   ./install-tools.sh sccache   # only sccache
#   ./install-tools.sh sweep     # only cargo-sweep
#
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)

CARGO_HOME_LOCAL="$root/deploy/local-production/runtime/cargo-home"
TOOL_BIN="$root/deploy/local-production/runtime/tool-bin"
export CARGO_HOME="$CARGO_HOME_LOCAL"

mkdir -p "$CARGO_HOME" "$TOOL_BIN"

install_sccache() {
  echo "[install-tools] Installing sccache …"
  cargo install sccache \
    --root "$TOOL_BIN/.cargo-install-tmp-sccache" \
    --locked \
    2>&1
  cp -f "$TOOL_BIN/.cargo-install-tmp-sccache/bin/sccache" "$TOOL_BIN/sccache"
  rm -rf "$TOOL_BIN/.cargo-install-tmp-sccache"
  echo "[install-tools] sccache installed → $TOOL_BIN/sccache"
  "$TOOL_BIN/sccache" --version
}

install_sweep() {
  echo "[install-tools] Installing cargo-sweep …"
  cargo install cargo-sweep \
    --root "$TOOL_BIN/.cargo-install-tmp-sweep" \
    --locked \
    2>&1
  cp -f "$TOOL_BIN/.cargo-install-tmp-sweep/bin/cargo-sweep" "$TOOL_BIN/cargo-sweep"
  rm -rf "$TOOL_BIN/.cargo-install-tmp-sweep"
  echo "[install-tools] cargo-sweep installed → $TOOL_BIN/cargo-sweep"
  "$TOOL_BIN/cargo-sweep" --version
}

TARGET="${1:-all}"

case "$TARGET" in
  all)
    install_sccache
    install_sweep
    ;;
  sccache)
    install_sccache
    ;;
  sweep)
    install_sweep
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Usage: $0 [all|sccache|sweep]" >&2
    exit 1
    ;;
esac

echo
echo "✓ Tool installation complete."
echo "  Run 'cargo-local.sh build --release' to start using them."
