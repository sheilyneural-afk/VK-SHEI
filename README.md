# KV-SHEI / M-SLIM

```yaml
project: KV-SHEI
product: M-SLIM
technical_namespace: mslim
```

KV-SHEI es el repositorio de M-SLIM, un runtime local escrito en Rust para
ejecutar modelos de lenguaje, gestionar su caché K/V y conservar sesiones con
identidad verificable.

Esta documentación describe el código y las pruebas que existen en el árbol
actual. No utiliza “producción”, “exacto” o “certificado” como sinónimos de
“compila” o “tiene tests”. Cada afirmación indica el alcance de la evidencia.

## Estado y evidencia

La documentación versionada define contratos y procedimientos; no congela el
estado mutable de esta máquina. Las matrices, calibraciones, recibos y E2E viven
en `deploy/local-production/runtime/`, están ignorados por Git y solo son
vigentes si enlazan la revisión fuente limpia, los binarios, el bundle y la
configuración efectiva que se están comprobando.

```mermaid
flowchart LR
    S[Fuente limpia] --> B[Release verificada]
    B --> V[Bundle + matriz + calibración]
    V --> R[Worker /ready]
    R --> G[Gateway autenticado]
    G --> E[E2E y recuperación]
```

La matriz `mslim.validation-matrix/v2` incluye `source_revision`. Los
consumidores la rechazan si no coincide con la revisión del binario o si la
compilación se hizo desde un árbol sucio. El mapa completo de capas, autoridad
y fallos cerrados está en
[`docs/ARCHITECTURE_FLOWS.md`](docs/ARCHITECTURE_FLOWS.md).

Una ejecución histórica local observó:

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

La comparación CPU/GPU es diagnóstica. En aquella muestra, sobre un prefill y ocho
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

Los binarios, modelos y recibos instalados viven bajo
`deploy/local-production/runtime/` y no forman parte del commit. Por eso el
repositorio no atribuye al `HEAD` el estado mutable de una máquina concreta.
`deploy/local-production/scripts/build-release.sh` construye los dos binarios
canónicos desde una única instantánea estable de inputs, instala copias locales
y escribe `mslim.local_release/v3`. `verify-release.sh` vuelve a calcular los
hashes de inputs, binarios y contenido del bundle antes de aceptar esa release.
La disponibilidad de un proceso se demuestra aparte mediante `/health` y la
aptitud para servir peticiones mediante `/ready` y un E2E autenticado.

El contrato de calibración está unificado en
`mslim.calibration_receipt/v4`, estado `measured-sustained-execution`, mediante
el validador compartido por `mslim-calibrate`, el worker, el gateway y
`mslim-autoconfigure`. La matriz `mslim.validation-matrix/v2` congela también
`prefill_weight_reuse_tokens`, `hot_v_tokens` y el resto del perfil efectivo;
el worker devuelve `/ready` 503 si falta la evidencia local o si cualquier
valor difiere. Los documentos anteriores se archivan por hash antes de
migrarlos; el bloque `migration` enlaza esquema, SHA-256 y copia original. Una
migración conserva la medición histórica, pero no la convierte en una medición
nueva ni sustituye una certificación ejecutada desde el commit actual.

## Cómo funciona

### Superficies HTTP

La API de producto soportada es la del gateway `mslim-gateway`:
`/v1/models`, `/v1/capabilities` y `/v1/chat/completions`. El endpoint
`/v1/generate` pertenece exclusivamente a `mslim-runtime-proxy`: es un puente
interno de tokens sobre la sesión IPC genérica para integración de runtimes y
pruebas locales. No es una API pública, no acepta mensajes de chat y no debe
exponerse fuera de loopback.

La interfaz servida por el worker es `assets/chat-runtime.html`. El cliente
histórico duplicado se retiró porque ningún entrypoint activo lo servía.

Cada manifiesto de release registra `build_input_digest`. Este digest cubre el
código, manifests y lockfiles de Cargo, toolchain, configuración y assets de
compilación. El verificador lo recalcula antes de aceptar la release; el propio
`manifest.json` se excluye para evitar una identidad autorreferencial.

### Núcleo y adaptadores

`MslimEngine` ejecuta el contrato `ModelAdapter`. El contrato incluye
inspección, carga, prefill, decode, reset de sesión, snapshot, restore y un
informe opcional de integridad K/V. Las implementaciones físicas viven detrás
de ese contrato:

- `native-cpu-runtime`: ejecución Rust en CPU;
- `accelerator-runtime`: ejecución GPU en el equipo compatible (OpenCL, HIP u otro backend detectado);
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

El propietario gobernado vive fuera de M-SLIM: el llamador autenticado conserva
la relación entre su usuario o trabajo y el handle opaco. El worker separa por
hash los namespaces de conversación y continuidad, persiste caducidad y
política, y convierte `release` en una lápida durable que impide restauraciones
posteriores. La máquina de estados, las responsabilidades y el límite actual de
propietario local único están definidos en
[`docs/CONTINUITY_LIFECYCLE.md`](docs/CONTINUITY_LIFECYCLE.md).

### Runtime externo e IPC

El contrato backend-neutral permite conectar un proveedor mediante un manifest
`mslim.runtime.manifest/v1` y un socket Unix. El worker exige las capacidades
`prefill`, `decode`, `reset_session`, `snapshot_kv` y `restore_kv`. Si falta una,
el arranque con runtime externo se rechaza.

El E2E `scripts/e2e-worker-external-restart.sh` detiene worker y proveedor,
arranca procesos nuevos y restaura el K/V desde ColdStore y SessionStore. La
evidencia queda en el runtime local; otro commit debe regenerarla.

### Servicio HTTP

El worker escucha en loopback y expone:

| Endpoint | Comportamiento actual |
|---|---|
| `GET /health` | estado del proceso y descripción del runtime cargado |
| `GET /ready` | exige modelo cargado, matriz válida para la configuración efectiva y calibración sostenida válida |
| `GET /v1/models` | modelo activo |
| `GET /v1/capabilities` | configuración, calibración reconocida y matriz reconocida |
| `POST /v1/chat/completions` | generación greedy y operaciones de continuidad |

El subconjunto HTTP rechaza `stream: true`, tools, temperatura distinta de cero
y `top_p` distinto de uno. Acepta entre 1 y 512 tokens de salida. “Stream” en
`mslim_weight_mode=stream` significa streaming de pesos y no streaming SSE de
la respuesta HTTP.

El gateway es un proceso separado. Exige `MSLIM_GATEWAY_TOKEN`, compara el
bearer token y solo acepta workers en loopback. Worker y gateway exigen la misma
matriz y calibración válidas; un manifest de runtime externo demuestra sus
capacidades IPC, pero no sustituye esas evidencias.

Cargo tiene `autobins = false` y registra `mslim-gateway` con
`src/bin/mslim-gateway.rs`. La implementación HTTP anterior que vivía en
`src/bin/mslim-gateway.rs` se eliminó porque no se compilaba; sigue en el
historial de git.

## Módulos principales

| Área | Implementación |
|---|---|
| Contrato de adaptadores | [`crates/mslim-adapter-api/src/lib.rs`](crates/mslim-adapter-api/src/lib.rs) |
| Motor | [`src/engine.rs`](src/engine.rs) |
| CPU nativa | [`src/native_cpu_adapter.rs`](src/native_cpu_adapter.rs) |
| Adapter acelerado | [`src/gpu_adapter.rs`](src/gpu_adapter.rs) |
| Runtime GPU | [`src/gpu_llm/mod.rs`](src/gpu_llm/mod.rs) |
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
| `accelerator-backend` | GPU kernels, worker y proveedor acelerado (backend detectado en runtime) |
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

- [Arquitectura y flujos verificables](docs/ARCHITECTURE_FLOWS.md)
- [Contrato de validación](docs/VALIDATION.md)
- [Continuidad y ciclo de vida](docs/CONTINUITY_LIFECYCLE.md)
- [Calibración adaptativa](docs/ADAPTIVE_CALIBRATION.md)
- [Adaptadores y selección](docs/MODEL_ADAPTERS.md)
- [Entorno local](deploy/local-production/README.md)
- [Perfil de prefill](docs/PREFILL_PROFILE.md)
- [Estándar de nombres](docs/NAMING_STANDARD.md)
- [Índice de documentación](docs/README.md)
