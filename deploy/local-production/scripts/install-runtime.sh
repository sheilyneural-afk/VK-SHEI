#!/usr/bin/env bash
set -euo pipefail
[ "$#" -ge 4 ] || { echo "uso: $0 <catalog.json> <runtime-id> <artifact-url> <sha256> [destination]" >&2; exit 2; }
catalog=$1; runtime_id=$2; url=$3; expected_sha=$4
destination=${5-"runtimes/$runtime_id/runtime.bin"}
[ -f "$catalog" ] || { echo "catálogo inexistente" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq es necesario" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl es necesario" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum es necesario" >&2; exit 1; }
jq -e --arg id "$runtime_id" '.schema == "mslim.runtime.catalog/v1" and any(.entries[]; .id == $id)' "$catalog" >/dev/null || { echo "runtime no declarado en el catálogo" >&2; exit 1; }
case "$url" in https://*) ;; *) echo "solo HTTPS" >&2; exit 1 ;; esac
printf '%s' "$expected_sha" | grep -Eq '^[0-9a-fA-F]{64}$' || { echo "SHA-256 inválido" >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"
case "$destination" in ''|/*|*'..'*|*'//'*) echo "destino inválido" >&2; exit 1 ;; esac
target="$runtime/$destination"; mkdir -p "$runtime/tmp" "$(dirname -- "$target")"
tmp=$(mktemp "$runtime/tmp/runtime.XXXXXX"); trap 'rm -f -- "$tmp"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error -o "$tmp" "$url"
actual=$(sha256sum "$tmp" | awk '{print $1}')
[ "$actual" = "$expected_sha" ] || { echo "hash no coincide" >&2; exit 1; }
chmod 0640 "$tmp"; mv -f -- "$tmp" "$target"; trap - EXIT
printf 'runtime instalado y verificado: %s (%s)\n' "$runtime_id" "$actual"
