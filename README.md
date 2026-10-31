# KV-SHEI / M-SLIM

KV-SHEI es el repositorio de M-SLIM, un runtime local escrito en Rust para
ejecutar modelos de lenguaje, gestionar su caché K/V y conservar sesiones con
identidad verificable.

Esta documentación describe el código y las pruebas que existen en el árbol
actual. No utiliza “producción”, “exacto” o “certificado” como sinónimos de
“compila” o “tiene tests”. Cada afirmación indica el alcance de la evidencia.

## Estado actual comprobado

La evidencia más reciente del árbol de trabajo es la matriz:

```text
deploy/local-production/runtime/validation/
validation-matrix-43e8fa4641e6f06d9df96abc4ef285556a8c714af6909b13e22e01e723d3d686.json
```

Su SHA-256 es
`8afd03f87d0c842788f33e86c3e45ff8343aeee974a9049e3b849a8d2d7e3344`.
El recibo no está versionado y solo describe el bundle, el equipo y la
configuración congelada que figuran en él. No incluye el commit ni un digest del
árbol fuente, por lo que su hash protege el recibo pero no demuestra por sí solo
qué revisión exacta lo produjo.

Resultado observado:

| Comprobación | Resultado |
|---|---|
| Determinismo CPU hot-dense | pasó en dos ejecuciones |
| Determinismo CPU ColdStore exacto | pasó en dos ejecuciones |
| Determinismo GPU densa | pasó en dos ejecuciones |
| Determinismo GPU ColdStore exacto | pasó en dos ejecuciones |
| CPU hot-dense frente a CPU ColdStore exacto | logits, tokens y SHA-256 lógico de K/V iguales |
| GPU densa frente a GPU ColdStore exacto | logits, tokens y SHA-256 lógico de K/V iguales |
| Snapshot de la ruta GPU exacta | contrato físico de ColdStore reconocido |
| CPU densa frente a GPU densa | mismos top-1 y tokens en la muestra; logits y K/V no bit-exactos |

La comparación CPU/GPU es diagnóstica. En este recibo, sobre un prefill y ocho
pasos de decode forzado, el máximo absoluto entre logits fue `0.234375`, el
máximo KL fue `0.0004376959013237726` y el p95 absoluto máximo fue `0.0859375`.
Estos números no son umbrales generales ni una evaluación de calidad del
modelo.

### Lo que esa matriz permite afirmar

- En la configuración congelada del recibo, introducir ColdStore exacto no
  alteró el resultado del backend CPU ni el del backend GPU frente a su propia
  referencia densa.
- Las cuatro rutas fueron reproducibles en las dos ejecuciones realizadas.
- La ruta GPU exacta emitió un snapshot físico con autoridad
  `canonical-bf16-coldstore`.
- CPU y GPU pueden producir resultados numéricos distintos sin que eso
  demuestre una corrupción de ColdStore.

### Lo que esa matriz no permite afirmar

- No valida `epsilon > 0`; el descarte aproximado sigue sin estar autorizado.
- No demuestra calidad de conversación, perplexidad global ni generaciones
  largas. Utiliza un único bundle, el prompt de prueba del bundle y ocho pasos
  de decode con entradas fijadas.
- No certifica otros modelos, drivers, dispositivos o configuraciones.
- No certifica automáticamente el perfil de `local-production.env`: la matriz
  desactiva replay de prompt, prefijo compartido, commit diferido y unroll4,
  mientras que el perfil local puede activarlos.
- No demuestra rendimiento útil. La matriz comprueba integridad, no velocidad.
- No ejecuta restore; comprueba el contenido lógico K/V y el tipo del snapshot.
  El restore pertenece a las pruebas de conformidad y continuidad separadas.
- `eligible-for-reviewed-activation` significa que pasó los gates incluidos en
  el recibo; no significa que exista una release instalada o un servicio activo.

## Estado del despliegue local

En el estado inspeccionado el directorio
`deploy/local-production/runtime/bin/` no contiene los binarios instalados y no
hay procesos M-SLIM activos. Por tanto, el layout local no es ahora una release
ejecutable.

Además, existe una incompatibilidad de contrato que debe corregirse antes de
presentar la ruta gateway como operativa:

- `mslim-calibrate` genera `/v3` con estado
  `measured-sustained-execution`;
- el worker actual acepta recibos de calibración
  `mslim.calibration_receipt/v2` o `/v3` con estado
  `measured-sustained-execution`;
- el gateway actual exige `/v1` con estado `measured-basic-execution`;
- `mslim-autoconfigure` también exige ese contrato `/v1`;
- no se conserva ningún recibo de calibración hasta que el contrato quede
  unificado.

Con esos tres formatos distintos, el gateway rechaza el worker al seleccionar
una ruta. La guía de despliegue explica este límite y no propone el arranque
gateway como una secuencia actualmente cerrada.

No se conserva ahora `runtime/validation/<digest>.json`. La matriz de cuatro
rutas no reemplaza ese recibo porque responde a otra pregunta. Hay que ejecutar
`mslim-validate-model` con el código actual antes de que `mslim-models` y
`mslim-autoconfigure` puedan reconocer el bundle como validado.

## Cómo funciona

### Núcleo y adaptadores

`MslimEngine` ejecuta el contrato `ModelAdapter`. El contrato incluye
inspección, carga, prefill, decode, reset de sesión, snapshot, restore y un
informe opcional de integridad K/V. Las implementaciones físicas viven detrás
de ese contrato:

- `native-cpu-runtime`: ejecución Rust en CPU;
- `accelerator-runtime`: ejecución OpenCL en el equipo compatible;
- adaptadores opcionales LibTorch e HIP, compilados mediante features
  específicas.

Que un adaptador esté compilado no basta para usarlo. El catálogo diferencia
entre compilado, disponible, no disponible y pendiente de sondeo. El bundle
también debe superar su inspección de configuración, tokenizer, shards,
geometría y layout.

### Las cuatro rutas de validación

La matriz usa cuatro ejecuciones deliberadamente separadas:

| Ruta | Qué cambia |
|---|---|
| CPU hot-dense | La atención CPU puede leer todo el K/V desde el nivel caliente; ColdStore sigue recibiendo la representación canónica. |
| CPU ColdStore exacto | El presupuesto caliente se fija a cero y la atención relee el K/V BF16 de ColdStore. |
| GPU densa | K/V permanece en el almacenamiento denso del dispositivo. |
| GPU ColdStore exacto | El K/V BF16 se conserva en ColdStore y la ruta acelerada lo usa sin descarte aproximado. |

Cada ruta se ejecuta dos veces. Después se comparan CPU densa con CPU exacta y
GPU densa con GPU exacta. El gate exige, dentro del mismo backend, igualdad de
logits, tokens y SHA-256 de la serialización lógica K/V. La comparación CPU/GPU
solo registra max absoluto, media, p95, KL, top-1 y margen top-1/top-2.

La especificación completa está en
[`docs/VALIDATION.md`](docs/VALIDATION.md).

### ColdStore y continuidad

ColdStore conserva páginas K/V BF16 append-only y verifica su integridad al
leerlas. Los índices de 4 bits, LSH, regiones y micrometadatos ayudan a localizar
datos o a construir certificados; no sustituyen los bytes BF16 originales.

La continuidad tiene dos capas distintas:

- `SessionStore` conserva generaciones, identidad, recibos y referencias a
  snapshots;
- `SessionCoordinator` coordina el estado semántico y el snapshot físico cuando
  se usa como biblioteca;
- el worker con runtime externo usa `SessionStore` desde el hilo propietario de
  la sesión y envía `SnapshotKv`/`RestoreKv` por IPC.

Un restore comprueba modelo, tokenizer, prefijo, política, generación, límites
y recibo. Una identidad incompatible debe fallar de forma cerrada.

### Runtime externo e IPC

El contrato backend-neutral permite conectar un proveedor mediante un manifest
`mslim.runtime.manifest/v1` y un socket Unix. El worker exige las capacidades
`prefill`, `decode`, `reset_session`, `snapshot_kv` y `restore_kv`. Si falta una,
el arranque con runtime externo se rechaza.

El script E2E disponible reinicia solo el worker y mantiene el proveedor
residente. No existe una prueba retenida ni una prueba de reinicio simultáneo de
worker y proveedor.

### Servicio HTTP

El worker escucha en loopback y expone:

| Endpoint | Comportamiento actual |
|---|---|
| `GET /health` | estado del proceso y descripción del runtime cargado |
| `GET /ready` | informa que el worker terminó su carga |
| `GET /v1/models` | modelo activo |
| `GET /v1/capabilities` | configuración, calibración reconocida y matriz reconocida |
| `POST /v1/chat/completions` | generación greedy y operaciones de continuidad |

El subconjunto HTTP rechaza `stream: true`, tools, temperatura distinta de cero
y `top_p` distinto de uno. Acepta entre 1 y 512 tokens de salida. “Stream” en
`mslim_weight_mode=stream` significa streaming de pesos y no streaming SSE de
la respuesta HTTP.

El gateway es un proceso separado. Exige `MSLIM_GATEWAY_TOKEN`, compara el
bearer token y solo acepta workers en loopback. Su selector de calibración está
actualmente desalineado con el worker, como se indica en “Estado del despliegue
local”.

Cargo tiene `autobins = false` y registra `mslim-server` con
`src/bin/mslim-gateway.rs`. El archivo `src/bin/mslim-server.rs` no es el target
canónico de ese nombre y conserva una implementación HTTP anterior. Su mera
presencia no describe el servicio que se compila como `mslim-server`.

## Módulos principales

| Área | Implementación |
|---|---|
| Contrato de adaptadores | [`crates/mslim-adapter-api/src/lib.rs`](crates/mslim-adapter-api/src/lib.rs) |
| Motor | [`src/engine.rs`](src/engine.rs) |
| CPU nativa | [`src/native_cpu_adapter.rs`](src/native_cpu_adapter.rs) |
| Adapter acelerado | [`src/opencl_adapter.rs`](src/opencl_adapter.rs) |
| Runtime OpenCL | [`src/opencl_llm.rs`](src/opencl_llm.rs) |
| ColdStore acelerado | [`src/gpu_region_store.rs`](src/gpu_region_store.rs) |
| Métricas de distribución | [`src/conformance.rs`](src/conformance.rs) |
| Matriz de validación | [`src/bin/mslim-adapter-parity.rs`](src/bin/mslim-adapter-parity.rs) |
| Sesiones persistentes | [`src/session_store.rs`](src/session_store.rs) |
| Coordinación de sesión | [`src/session_coordinator.rs`](src/session_coordinator.rs) |
| IPC backend-neutral | [`src/generic_runtime.rs`](src/generic_runtime.rs), [`src/runtime_host.rs`](src/runtime_host.rs) |
| Worker HTTP | [`src/bin/mslim-runtime-worker.rs`](src/bin/mslim-runtime-worker.rs) |
| Gateway | [`src/bin/mslim-gateway.rs`](src/bin/mslim-gateway.rs) |
| Grafo documental | [`src/document_graph.rs`](src/document_graph.rs) |

## Features de Cargo

| Feature | Alcance |
|---|---|
| `native-cpu-backend` | adapter CPU Rust y soporte de bundle; es la feature por defecto |
| `accelerator-backend` | OpenCL, kernels, worker y proveedor acelerado |
| `torch-backend` | runtime LibTorch y binarios asociados |
| `hip-backend` | bridge hipBLASLt explícito sobre la ruta acelerada |
| `unified-router` | gateway HTTP; no carga pesos |
| `model-catalog` | consulta y recomendación de candidatos remotos |
| `model-import` | importación local por digest |

Una feature indica que el código se compiló; no garantiza que el runtime, el
driver, el hardware o un bundle concreto sean compatibles.

## Comprobaciones básicas del núcleo

Desde la raíz del repositorio:

```bash
deploy/local-production/scripts/install-layout.sh

deploy/local-production/scripts/cargo-local.sh check \
  --locked --no-default-features

deploy/local-production/scripts/cargo-local.sh test \
  --locked --lib --no-default-features
```

La matriz física requiere el bundle local y el backend acelerado:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked --release \
  --features accelerator-backend,native-cpu-backend \
  --bin mslim-adapter-parity -- \
  deploy/local-production/runtime/models/<digest> \
  --decode-steps 8
```

El comando sobrescribe el recibo del mismo bundle de forma atómica. Antes de
usar su resultado para activar un perfil hay que comprobar que la configuración
congelada coincide con la que se pretende ejecutar.

## Documentación

- [Contrato de validación](docs/VALIDATION.md)
- [Adaptadores y selección](docs/MODEL_ADAPTERS.md)
- [Entorno local](deploy/local-production/README.md)
- [Perfil de prefill](docs/PREFILL_PROFILE.md)
- [Estándar de nombres](docs/NAMING_STANDARD.md)
- [Índice de documentación](docs/README.md)
