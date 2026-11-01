#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "uso: $0 <runtime-id> <version> <artifact-url> <checksum-url> <destination>" >&2
  exit 2
}
[ "$#" -eq 5 ] || usage

runtime_id=$1
version=$2
artifact_url=$3
checksum_url=$4
destination=$5

command -v curl >/dev/null || { echo "curl es necesario" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum es necesario" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq es necesario" >&2; exit 1; }

case "$artifact_url" in https://*) ;; *) echo "artifact-url debe usar HTTPS" >&2; exit 1 ;; esac
case "$checksum_url" in https://*) ;; *) echo "checksum-url debe usar HTTPS" >&2; exit 1 ;; esac
case "$destination" in ''|/*|*'..'*|*'//'*) echo "destino inválido" >&2; exit 1 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
runtime="$root/deploy/local-production/runtime"
mkdir -p "$runtime/tmp" "$runtime/manifests" "$(dirname -- "$runtime/$destination")"

artifact_tmp=$(mktemp "$runtime/tmp/artifact.XXXXXX")
checksum_tmp=$(mktemp "$runtime/tmp/checksum.XXXXXX")
trap 'rm -f -- "$artifact_tmp" "$checksum_tmp"' EXIT

curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
  --output "$artifact_tmp" "$artifact_url"
curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
  --output "$checksum_tmp" "$checksum_url"

artifact_name=$(basename "$artifact_url" | sed 's/[?#].*$//')
expected_sha=$(awk -v name="$artifact_name" '
  /^[[:space:]]*[0-9a-fA-F]{64}[[:space:]]+/ {
    line=$0
    sub(/^[[:space:]]*/, "", line)
    split(line, fields, /[[:space:]]+/)
    file=fields[2]
    sub(/^\*/, "", file)
    if (file == name || file == "./" name) { print tolower(fields[1]); exit }
  }
  /^[0-9a-fA-F]{64}$/ { print tolower($0); exit }
' "$checksum_tmp")

printf '%s' "$expected_sha" | grep -Eq '^[0-9a-f]{64}$' || {
  echo "no se encontró un SHA-256 inequívoco para $artifact_name" >&2
  exit 1
}

actual_sha=$(sha256sum "$artifact_tmp" | awk '{print tolower($1)}')
[ "$actual_sha" = "$expected_sha" ] || {
  echo "SHA-256 no coincide: esperado $expected_sha, obtenido $actual_sha" >&2
  exit 1
}

target="$runtime/$destination"
chmod 0640 "$artifact_tmp"
mv -f -- "$artifact_tmp" "$target"
manifest="$runtime/manifests/$runtime_id-$version.json"
jq -n \
  --arg id "$runtime_id" \
  --arg version "$version" \
  --arg platform "$(uname -s | tr '[:upper:]' '[:lower:]')" \
  --arg arch "$(uname -m)" \
  --arg artifact_url "$artifact_url" \
  --arg checksum_url "$checksum_url" \
  --arg sha256 "$actual_sha" \
  --arg path "$destination" \
  '{schema:"mslim.runtime.manifest/v1",id:$id,version:$version,platform:$platform,architecture:$arch,artifact_url:$artifact_url,checksum_url:$checksum_url,sha256:$sha256,path:$path,verified:true}' \
  > "$manifest"
printf 'runtime verificado: %s %s\nmanifest: %s\nsha256: %s\n' "$runtime_id" "$version" "$manifest" "$actual_sha"
