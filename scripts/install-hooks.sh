#!/usr/bin/env bash
# One-time local setup: use the versioned hooks in .githooks/.
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x .githooks/pre-push scripts/check.sh
git config core.hooksPath .githooks
echo "Hooks enabled: git push now runs scripts/check.sh."
