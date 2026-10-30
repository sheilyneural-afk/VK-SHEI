#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "uso: $0 <https-url> <sha256> <runtime-relative-destination>" >&2
  exit 2
}

[ "$#" -eq 3 ] || usage
url=$1
expected_sha=$2
relative_destination=$3

case "$url" in
  https://*) ;;
  *) echo "solo se permiten URLs HTTPS" >&2; exit 1 ;;
esac

printf '%s' "$expected_sha" | grep -Eq '^[0-9a-fA-F]{64}$' || {
  echo "SHA-256 inválido" >&2
  exit 1
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"

case "$relative_destination" in
  ''|/*|*'..'*|*'//'*)
    echo "destino inválido: debe ser relativo y permanecer dentro de runtime" >&2
    exit 1
    ;;
esac

destination="$runtime/$relative_destination"
mkdir -p "$runtime/tmp" "$(dirname -- "$destination")"
temporary=$(mktemp "$runtime/tmp/artifact.XXXXXX")
cleanup() { rm -f -- "$temporary"; }
trap cleanup EXIT

command -v curl >/dev/null 2>&1 || { echo "curl es necesario" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum es necesario" >&2; exit 1; }

curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
  --output "$temporary" "$url"
actual_sha=$(sha256sum "$temporary" | awk '{print $1}')
if [ "${actual_sha,,}" != "${expected_sha,,}" ]; then
  echo "hash no coincide: esperado $expected_sha, obtenido $actual_sha" >&2
  exit 1
fi

chmod 0640 "$temporary"
mv -f -- "$temporary" "$destination"
trap - EXIT
echo "instalado y verificado: ${relative_destination} (${actual_sha})"
