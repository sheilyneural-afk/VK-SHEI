#!/usr/bin/env bash
# Single pre-merge gate for M-SLIM. CI runs exactly this script, and the
# pre-push hook (see scripts/install-hooks.sh) runs it locally, so a change
# that passes here passes CI.
#
#   scripts/check.sh            format + clippy + tests (CPU, no-default, accelerator)
#   scripts/check.sh --torch    also lint/test the LibTorch backend (needs LIBTORCH)
#
# Physical-model checks (GPU, bundles) are never part of this gate; they run
# only on a self-hosted runner (see .github/workflows/mslim.yml).
set -euo pipefail
cd "$(dirname "$0")/.."

with_torch=0
for argument in "$@"; do
    case "$argument" in
        --torch) with_torch=1 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

# .cargo/config.toml points TMPDIR at this directory.
mkdir -p deploy/local-production/runtime/tmp

step() { printf '\n==> %s\n' "$*"; }

step "toolchain"
rustc --version
cargo --version
cargo clippy --version

step "rustfmt"
cargo fmt --all -- --check

for features in "--no-default-features" "" "--features accelerator-backend"; do
    step "clippy ${features:-default features}"
    # shellcheck disable=SC2086
    cargo clippy --locked --lib --bins --tests $features -- -D warnings
done

if [ "$with_torch" = 1 ]; then
    step "clippy torch-backend"
    cargo clippy --locked --lib --bins --tests --features torch-backend -- -D warnings
fi

for features in "--no-default-features" "" "--features accelerator-backend"; do
    step "tests ${features:-default features}"
    # shellcheck disable=SC2086
    cargo test --locked --lib --bins --tests $features
done

if [ "$with_torch" = 1 ]; then
    step "tests torch-backend"
    cargo test --locked --lib --bins --tests --features torch-backend
fi

step "adapter api crate"
cargo test --locked -p mslim-adapter-api

step "release manifest contract"
cargo test --locked --features release-installer --bin mslim-install-release

step "persistent contract migration"
python3 scripts/migrate-persistent-contracts.py --self-test

step "model bundle content digest"
python3 scripts/model-bundle-digest.py --self-test

step "source inventory regression gates"
scripts/check-source-inventory.sh

printf '\nOK: all checks passed.\n'
