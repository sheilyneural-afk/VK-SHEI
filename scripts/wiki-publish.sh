#!/usr/bin/env bash
# Publish an already generated and audited wiki. This script never generates
# content and requires an explicit --confirm-publish flag.
set -euo pipefail
cd "$(dirname "$0")/.."

wiki_dir=""
remote="https://github.com/sheilyneural-afk/KV-SHEI.wiki.git"
confirm=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --wiki) wiki_dir=${2:?missing value for --wiki}; shift 2 ;;
        --remote) remote=${2:?missing value for --remote}; shift 2 ;;
        --confirm-publish) confirm=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$confirm" -ne 1 ] || [ -z "$wiki_dir" ]; then
    echo "usage: scripts/wiki-publish.sh --wiki DIR [--remote URL] --confirm-publish" >&2
    exit 2
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "refusing publication from a dirty source tree" >&2
    exit 1
fi

receipt=$(mktemp)
trap 'rm -f "$receipt"' EXIT
python3 scripts/wiki_audit.py check --wiki "$wiki_dir" --receipt "$receipt"
python3 - "$receipt" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if receipt.get("status") != "passed" or not receipt.get("reusable_evidence"):
    raise SystemExit("wiki receipt is not reusable")
PY

worktree=$(mktemp -d)
trap 'rm -rf "$worktree"; rm -f "$receipt"' EXIT
git clone --quiet "$remote" "$worktree"
rsync -a --delete --exclude=.git/ "$wiki_dir"/ "$worktree"/
git -C "$worktree" add --all
if git -C "$worktree" diff --cached --quiet; then
    echo "wiki already matches generated content"
    exit 0
fi
git -C "$worktree" commit -m "Synchronize wiki with $(git rev-parse --short=12 HEAD)"
git -C "$worktree" fetch --quiet origin master
git -C "$worktree" merge-base --is-ancestor origin/master HEAD || {
    echo "refusing non-fast-forward wiki publication" >&2
    exit 1
}
git -C "$worktree" push origin master
