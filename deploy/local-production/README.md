# Entorno local de M-SLIM

```yaml
project: KV-SHEI
product: M-SLIM
technical_namespace: mslim
```

`deploy/local-production` es una ruta persistente conservada por compatibilidad
con instalaciones y recibos existentes; no es una declaración del estado del
servicio. El entorno se identifica como `local-runtime`. Este árbol aísla
modelos, binarios, estado, recibos y compilación dentro del proyecto.

## Qué demuestra cada capa

```mermaid
flowchart LR
    S[Fuente limpia] --> B[build-release]
    B --> V[verify-release]
    V --> E[Recibos y matriz]
    E --> R[/ready]
    R --> G[Gateway autenticado]
    G --> X[E2E con reinicio total]
```

El desarrollo completo de este flujo está en
[`../../docs/ARCHITECTURE_FLOWS.md`](../../docs/ARCHITECTURE_FLOWS.md).

El contenido de `runtime/` es local, mutable e ignorado por Git. Su mera
existencia no demuestra el estado del commit ni el de un proceso residente.

- `build-release.sh` demuestra que ambos binarios se construyeron sin que
  cambiaran los inputs y deja sus hashes en `mslim.local_release/v3`.
- `verify-release.sh` demuestra estructura, rutas, permisos y coincidencia de
  hashes de fuente, binarios y bundle en el instante de la comprobación.
- `mslim.validation_receipt/v2`, `mslim.validation-matrix/v2` y
  `mslim.calibration_receipt/v4` demuestran aspectos distintos del modelo; no
  son intercambiables.
- `/health` demuestra vida del proceso. `/ready` exige modelo, matriz y
  calibración compatibles con la configuración efectiva.
- solo una petición autenticada por el gateway demuestra el recorrido HTTP
  público; una restauración tras reiniciar el provider demuestra además la
  persistencia física del K/V.

La certificación de una revisión debe ejecutar todas esas capas desde un árbol
limpio y registrar por separado sus resultados. Ningún recibo histórico se
presenta como evidencia nueva por haber sido migrado de esquema.

## Layout

```text
deploy/local-production/
├── README.md
├── env.example
├── local-production.env          # configuración local; no versionar secretos
├── manifest.example.json
├── manifest.json                 # identidad esperada de la release local
├── runtime/
│   ├── bin/                      # binarios instalados; puede estar vacío
│   ├── models/<digest>/          # bundles importados
│   ├── coldstore/                # páginas K/V persistentes
│   ├── sessions/                 # SessionStore
│   ├── calibration/              # recibos de calibración
│   ├── validation/               # matrices y E2E
│   ├── receipts/
│   ├── logs/
│   ├── cargo-home/               # registro y caché local de Cargo
│   └── build-target/             # artefactos de compilación
└── scripts/
```

`install-layout.sh` crea los directorios. No instala drivers ni una release.

## Modelo local

El bundle activo debe estar dentro de:

```text
deploy/local-production/runtime/models/<digest>/
```

`mslim-import-model` rechaza enlaces simbólicos, calcula un digest determinista,
crea `bundle-manifest.json` y copia el bundle al layout local. El origen no se
modifica.

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-import --bin mslim-import-model -- \
  /ruta/autorizada/al/bundle
```

`mslim-models` enumera bundles locales; no demuestra que un adapter pueda
cargarlos:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --no-default-features --bin mslim-models
```

## Catálogo remoto

La feature `model-catalog` puede consultar candidatos y ordenarlos según el
inventario local. Una recomendación no descarga ni activa automáticamente un
modelo.

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-catalog --bin mslim-recommend -- \
  "text-generation" 20
```

`mslim-download-model --auto` se detiene si no encuentra un adapter disponible.
Después de descargar siguen siendo obligatorias la inspección del bundle, la
carga real y la validación del perfil.

Las operaciones privadas usan el token de lectura gestionado por `mslim-auth`.
El token no debe aparecer en manifests, logs ni documentación.

## Instalación de artefactos

`install-artifact.sh` acepta una URL HTTPS, un SHA-256 esperado y un destino
relativo dentro de `runtime/`. Descarga a un temporal, verifica el hash y mueve
el archivo de forma atómica. No ejecuta el artefacto ni instala drivers.

```bash
deploy/local-production/scripts/install-artifact.sh \
  https://origen-autorizado/artefacto \
  SHA256_DE_64_CARACTERES \
  runtimes/<runtime>/<archivo>
```

`resolve-runtime-manifest.sh` requiere que la release proporcione la URL del
artefacto y un fichero oficial de checksums. El catálogo de ejemplo no contiene
una release instalable por sí solo.

## Compilación local

El wrapper mantiene Cargo, temporales y `target` dentro del proyecto:

```bash
deploy/local-production/scripts/cargo-local.sh fetch --locked
deploy/local-production/scripts/cargo-local.sh check \
  --locked --no-default-features
deploy/local-production/scripts/cargo-local.sh test \
  --locked --lib --no-default-features
```

No es necesario borrar `runtime/cargo-home` ni `runtime/build-target` para una
compilación normal. Son cachés recuperables y permiten compilación incremental.

## Configuración

La plantilla se copia de forma explícita:

```bash
deploy/local-production/scripts/install-layout.sh
cp deploy/local-production/env.example \
  deploy/local-production/local-production.env
```

Hay que reemplazar todos los valores `REPLACE_WITH_*`. El constructor canónico
compila los dos binarios, instala copias atómicas, recalcula el digest de todos
los inputs de compilación y comprueba el contenido completo del modelo:

```bash
deploy/local-production/scripts/build-release.sh
```

El script escribe `mslim.local_release/v3`, actualiza únicamente los tres
digests efectivos del entorno y ejecuta `verify-release.sh`. El digest del
modelo usa el mismo dominio (ruta relativa + NUL + bytes + NUL) que los
importadores; no es un hash de la lista de nombres. `MSLIM_GATEWAY_TOKEN` es
obligatorio para el gateway.

Los contratos persistentes que preceden a la identidad obligatoria se migran
una sola vez con:

```bash
python3 scripts/migrate-persistent-contracts.py --apply
python3 scripts/migrate-persistent-contracts.py --check
```

Antes de sustituir un JSON, el migrador archiva sus bytes exactos bajo
`runtime/receipts/contract-migration/archive/<sha256>.json`. Las cadenas
`calibration-receipts.jsonl` v1 no se reescriben: conservan su dominio de hash,
son verificables y quedan en modo histórico de solo lectura; las transacciones
nuevas empiezan una raíz v2.

El entorno efectivo debe coincidir exactamente con la configuración congelada
en la matriz v2, incluidos `prefill_weight_reuse_tokens`, `hot_v_tokens`,
replay, prefijo compartido, commit diferido, unroll4 y fusiones. Ante cualquier
ausencia o diferencia, `/health` puede responder que el proceso vive, pero
`/ready` responde 503.

`MSLIM_ENABLE_LOD=true` con `MSLIM_LOD_EPSILON=0` crea la ruta exacta sin
descarte. Un valor positivo no está autorizado por la matriz actual.

## Qué comprueba `verify-release.sh`

El script comprueba:

- que existen `local-production.env` y `manifest.json`;
- que `manifest.json` usa `mslim.local_release/v3` y que sus rutas y hashes de
  binarios y modelo coinciden con la configuración efectiva;
- que la cabecera declara `KV-SHEI` / `M-SLIM` / `mslim` y que
  `build_input_digest` coincide con código, manifests, lockfiles, toolchain,
  configuración y assets de compilación actuales;
- que `MSLIM_HOST` es `127.0.0.1`;
- que `MSLIM_ENABLE_LOD` es `true` o `false`;
- que `MSLIM_COLDSTORE_FORMAT` es `TSKVPG04`;
- que las rutas configuradas permanecen dentro del proyecto;
- que existen y son ejecutables `mslim-gateway` y
  `mslim-runtime-worker`;
- que sus SHA-256 coinciden con las variables del entorno;
- que el digest de contenido del bundle coincide con el nombre de su directorio;
- que los archivos del modelo no son escribibles por grupo u otros usuarios.

No comprueba:

- todos los límites y opciones operativas de `manifest.json`;
- que el driver o el dispositivo funcionen;
- que el modelo pueda cargarse;
- que el worker esté listo;
- que la matriz corresponda al perfil efectivo;
- que la API genere una respuesta.

Por tanto, `VERIFY-OK` significa “estructura y hashes básicos válidos”, no
“servicio listo”.

## Worker y gateway

`run-local-worker.sh` inicia el worker GPU en `127.0.0.1`, construyendo sus
argumentos a partir de `local-production.env`. El worker no implementa
autenticación y no debe exponerse fuera de loopback.

`run-local-runtime.sh` llama a `run-auto-runtime.sh`. Este último:

1. ejecuta `verify-release.sh`;
2. valida un manifest externo si se configuró;
3. busca un worker local que responda a `/ready` y `/v1/capabilities`;
4. inicia el gateway autenticado.

El gateway exige un bearer token para `/v1/models`, `/v1/capabilities` y
`/v1/chat/completions`; `/health` y `/ready` no requieren token. Tanto gateway
como workers aceptados deben usar loopback.

### Contrato canónico de calibración

La ruta usa un único contrato en el código actual:

| Componente | Contrato aceptado |
|---|---|
| `mslim-calibrate` | genera `mslim.calibration_receipt/v4`, estado `measured-sustained-execution` |
| Worker | valida `/v4` mediante `calibration_contract` |
| Gateway | valida `/v4` mediante `calibration_contract` |
| `mslim-autoconfigure` | valida `/v4` mediante `calibration_contract` |
| Recibo local | cuando se genera debe ser `/v4`; una migración puede enlazar el `/v3` original por SHA-256, pero no cuenta como una medición nueva |

La calibración v4 incluye la revisión fuente en su identidad. Productores y
consumidores rechazan una revisión distinta o un árbol que estuviera sucio al
compilar; una medición anterior del mismo bundle no autoriza el binario actual.

El gateway rechaza un worker cuyo recibo o contrato de capacidades no cumpla
los contratos compartidos. Para certificar un arranque hay que regenerar la
evidencia desde la fuente actual, instalar los binarios y ejecutar un E2E
público por el gateway.

### Recibo general del bundle

`mslim-validate-model` escribe
`runtime/validation/<bundle-digest>.json` con schema
`mslim.validation_receipt/v2`. `mslim-models` y `mslim-autoconfigure` exigen ese
schema. Un recibo local debe regenerarse para certificar una revisión nueva; la
presencia de uno anterior no enlaza por sí sola el binario residente con el
`HEAD` actual.

La matriz `validation-matrix-<digest>.json` no sustituye este recibo: valida la
relación densa/ColdStore, mientras que el recibo general identifica un adapter y
registra inspect, load, prefill, decode, snapshot y restore. Antes de generar un
perfil automático certificado hay que volver a ejecutar `mslim-validate-model`
y después `mslim-calibrate` desde los binarios instalados vinculados a la fuente
actual.

## Matriz exacta

La validación física actual se ejecuta así:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked --release \
  --features accelerator-backend,native-cpu-backend \
  --bin mslim-adapter-parity -- \
  deploy/local-production/runtime/models/<digest> \
  --decode-steps 8
```

El recibo queda en `runtime/validation/validation-matrix-<digest>.json`. El gate
depende de determinismo, exactitud densa/ColdStore dentro de cada backend y
snapshot físico GPU. CPU/GPU se conserva como diagnóstico. Consulte
[`../../docs/VALIDATION.md`](../../docs/VALIDATION.md).

## E2E con proveedor externo

`scripts/e2e-worker-external-restart.sh` compila y arranca un proveedor y un
worker, genera un token, crea un snapshot, detiene ambos procesos, vuelve a
arrancar ambos y restaura la sesión desde el estado físico persistido.

```bash
MSLIM_MODEL_DIR="$PWD/deploy/local-production/runtime/models/<digest>" \
  scripts/e2e-worker-external-restart.sh
```

Cada ejecución usa un directorio identificado por fecha y PID para no mezclar
sesiones ni sobrescribir evidencia. Esos directorios locales se conservan hasta
una limpieza auditable; el script no afirma que el resultado anterior siga
vigente para otra revisión.

La restauración posterior no puede depender de memoria residente del provider:
el socket anterior desaparece y el nuevo proceso vuelve a abrir el ColdStore
conservado por la ejecución.

## Seguridad y alcance

- Worker y gateway están restringidos a loopback por sus scripts y validaciones.
- El gateway usa bearer token, pero no TLS. Loopback sigue siendo obligatorio.
- Los drivers son externos al proyecto.
- Un health positivo no sustituye readiness, calibración, matriz ni generación
  E2E.
- Los recibos locales están ignorados por Git; para reproducibilidad hay que
  conservar su hash y la identidad exacta del árbol que los produjo.
- La matriz v2 incluye `source_revision` y los consumidores exigen que coincida
  con el binario creado desde un árbol limpio.
- Los nuevos recibos deben usar únicamente el contrato `mslim.*` vigente.
