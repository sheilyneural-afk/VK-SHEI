# Entorno local de M-SLIM

El nombre del directorio es `local-production`, pero su existencia no demuestra
que haya una release instalada ni un servicio activo. Este árbol aísla modelos,
binarios, estado, recibos y compilación dentro del proyecto.

## Estado inspeccionado

A fecha de la revisión documental del 31 de octubre de 2026:

- existe un bundle local bajo `runtime/models/`;
- existe una matriz actual `mslim.validation-matrix/v1` que pasó para una
  configuración exacta concreta;
- no se conservan calibraciones, certificaciones ni E2E anteriores;
- no existe todavía un recibo general `mslim.validation_receipt/v1` para el
  bundle;
- `runtime/bin/` no contiene los binarios declarados por la release;
- no hay procesos M-SLIM activos;
- el gateway y el worker no aceptan actualmente el mismo contrato de
  calibración.

Por esas razones, este layout no debe describirse ahora como una release
instalada o arrancable de extremo a extremo.

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

Hay que reemplazar todos los valores `REPLACE_WITH_*` y registrar los digests
de los binarios y del árbol del modelo. `MSLIM_GATEWAY_TOKEN` es obligatorio
para el gateway.

El perfil incluido activa varias optimizaciones exactas. La matriz actual no
las cubre todas a la vez: su configuración congelada desactiva replay de prompt,
prefijo compartido, commit diferido y unroll4. Antes de instalar ese perfil como
release hay que repetir la validación con la combinación efectiva o reducir el
perfil a la configuración ya medida.

`MSLIM_ENABLE_LOD=true` con `MSLIM_LOD_EPSILON=0` crea la ruta exacta sin
descarte. Un valor positivo no está autorizado por la matriz actual.

## Qué comprueba `verify-release.sh`

El script comprueba:

- que existen `local-production.env` y `manifest.json`;
- que `MSLIM_HOST` es `127.0.0.1`;
- que `MSLIM_ENABLE_LOD` es `true` o `false`;
- que `MSLIM_COLDSTORE_FORMAT` es `TSKVPG04`;
- que las rutas configuradas permanecen dentro del proyecto;
- que existen y son ejecutables `mslim-server` y
  `mslim-runtime-worker`;
- que sus SHA-256 coinciden con las variables del entorno;
- que el digest del árbol del modelo coincide;
- que los archivos del modelo no son escribibles por grupo u otros usuarios.

No comprueba:

- que el contenido de `manifest.json` coincida con todas las variables;
- que el worker y el gateway acepten el mismo schema de calibración;
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

### Bloqueo actual de calibración

La ruta no está cerrada en el código actual:

| Componente | Contrato aceptado |
|---|---|
| `mslim-calibrate` | genera `mslim.calibration_receipt/v3`, estado `measured-sustained-execution` |
| Worker | `mslim.calibration_receipt/v2` o `/v3`, estado `measured-sustained-execution` |
| Gateway | `mslim.calibration_receipt/v1`, estado `measured-basic-execution` |
| `mslim-autoconfigure` | `mslim.calibration_receipt/v1`, estado `measured-basic-execution` |
| Recibo local | no existe hasta unificar el contrato |

El gateway rechaza un worker que no cumpla su contrato. Antes de documentar el
arranque como operativo hay que unificar esos schemas, generar una calibración
actual, instalar los binarios y ejecutar un E2E por el gateway.

### Recibo general del bundle

`mslim-validate-model` escribe
`runtime/validation/<bundle-digest>.json` con schema
`mslim.validation_receipt/v1`. `mslim-models` y `mslim-autoconfigure` exigen ese
schema. No se conserva ahora ningún recibo general del bundle.

La matriz `validation-matrix-<digest>.json` no sustituye este recibo: valida la
relación densa/ColdStore, mientras que el recibo general identifica un adapter y
registra inspect, load, prefill, decode, snapshot y restore. Antes de generar un
perfil automático hay que volver a ejecutar `mslim-validate-model` y después
`mslim-calibrate` con los contratos unificados.

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
worker, genera un token, crea un snapshot, reinicia únicamente el worker y
restaura la sesión contra el proveedor que permaneció activo.

```bash
MSLIM_MODEL_DIR="$PWD/deploy/local-production/runtime/models/<digest>" \
  scripts/e2e-worker-external-restart.sh
```

No se conserva ningún resultado E2E anterior. La siguiente ejecución deberá
generar evidencia nueva y sustituible, sin acumular directorios de intentos.

No existe en ese script una prueba de reinicio del proveedor. Si se necesita
recuperación tras caída completa, debe añadirse un ensayo separado que reinicie
worker y proveedor y restaure desde estado persistente.

## Seguridad y alcance

- Worker y gateway están restringidos a loopback por sus scripts y validaciones.
- El gateway usa bearer token, pero no TLS. Loopback sigue siendo obligatorio.
- Los drivers son externos al proyecto.
- Un health positivo no sustituye readiness, calibración, matriz ni generación
  E2E.
- Los recibos locales están ignorados por Git; para reproducibilidad hay que
  conservar su hash y la identidad exacta del árbol que los produjo.
- La matriz actual no incluye esa identidad del árbol; debe añadirse a futuros
  recibos o conservarse externamente junto a ellos.
- Los nuevos recibos deben usar únicamente el contrato `mslim.*` vigente.
