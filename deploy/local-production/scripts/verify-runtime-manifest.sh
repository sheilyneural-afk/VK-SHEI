#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "uso: $0 <manifest.json>" >&2; exit 2; }
manifest=$1
[ -f "$manifest" ] || { echo "manifest inexistente" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq es necesario" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum es necesario" >&2; exit 1; }

jq -e '
  .schema == "kvslim.runtime.manifest/v1"
  and (.id | type == "string" and length > 0)
  and (.version | type == "string" and length > 0)
  and .verified == true
  and (.artifact_url | startswith("https://"))
  and (.checksum_url | startswith("https://"))
  and (.sha256 | test("^[0-9a-fA-F]{64}$"))
  and (.path | type == "string" and (startswith("/") | not) and (contains("..") | not))
' "$manifest" >/dev/null || { echo "manifest inválido o no verificado" >&2; exit 1; }

platform=$(jq -r '.platform' "$manifest")
architecture=$(jq -r '.architecture' "$manifest")
expected_platform=$(uname -s | tr '[:upper:]' '[:lower:]')
expected_arch=$(uname -m)
[ "$platform" = "$expected_platform" ] || { echo "plataforma incompatible" >&2; exit 1; }
[ "$architecture" = "$expected_arch" ] || { echo "arquitectura incompatible" >&2; exit 1; }

echo "manifest válido: $(jq -r '.id + " " + .version' "$manifest") ($platform/$architecture)"
