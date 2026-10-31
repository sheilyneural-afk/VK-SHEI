# KV-SHEI · KVSLIM

<p align="center">
  <strong>Contexto verificable, continuidad persistente y ejecución local para LLMs.</strong><br>
  Un núcleo Rust backend-neutral que conserva el estado K/V exacto, recupera evidencia original y solo activa un runtime cuando sus capacidades han sido comprobadas.
</p>

<p align="center">
  <a href="https://github.com/sheilyneural-afk/VK-SHEI/actions/workflows/kvslim.yml"><img src="https://github.com/sheilyneural-afk/VK-SHEI/actions/workflows/kvslim.yml/badge.svg" alt="CI"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-edition%202024-orange?logo=rust" alt="Rust 2024"></a>
  <img src="https://img.shields.io/badge/version-0.4.0-blue" alt="Version 0.4.0">
  <img src="https://img.shields.io/badge/licence-see%20repository-lightgrey" alt="License information">
</p>

> [!IMPORTANT]
> **Estado real verificado en el código:** el contrato de ejecución, el almacenamiento K/V, la continuidad, el grafo documental y las pruebas de IPC están implementados. La compatibilidad “universal” describe el contrato y el descubrimiento por capacidades; **no significa que cualquier LLM, driver o hardware pueda ejecutarse sin un adaptador y una validación física**. El repositorio no publica una promesa universal de rendimiento.

## Qué es

KV-SHEI es el repositorio del runtime **KVSLIM**: una base local para organizar contexto, almacenar estado K/V y conectar runtimes de inferencia mediante contratos verificables.

El núcleo está diseñado alrededor de cuatro ideas:

- **K/V autoritativo en BF16.** El estado exacto se conserva en páginas append-only de `ColdStore`; el índice de 4 bits localiza coordenadas reversibles, pero no sustituye al K/V original.
- **Continuidad con identidad.** Las sesiones semánticas y los snapshots físicos se asocian a generación, hashes, identidad del modelo y recibos explícitos.
- **Selección por capacidades.** El runtime se descubre por las operaciones que demuestra (`prefill`, `decode`, reset, snapshot y restore), no por el nombre de una marca.
- **Evidencia original.** El grafo documental enruta páginas mediante micro-metadatos y enlaces deterministas, pero devuelve el texto original con su digest; no lo reemplaza por un resumen inventado.

## Arquitectura en una vista

```mermaid
flowchart TB
    client["Cliente local"] --> gateway["Gateway HTTP (loopback)"]
    gateway --> worker["Worker de inferencia"]
    worker --> host["RuntimeHost / IPC"]
    host --> provider["Runtime provider opcional"]
    worker --> sessions["SessionStore + Coordinator"]
    sessions --> cold["ColdStore (K/V BF16)"]
    worker --> graph["DocumentGraph y evidencia"]
    provider --> bundle["Model bundle verificado"]
```

El flujo importante es deliberadamente explícito: un catálogo puede encontrar una implementación compilada, pero solo el sondeo local y la inspección del bundle pueden marcarla como utilizable. Si una capacidad no está demostrada, la selección falla de forma cerrada; no hay fallback silencioso que finja compatibilidad.

## Capacidades implementadas

| Área | Qué existe realmente | Fuente principal |
|---|---|---|
| Almacenamiento K/V | `ColdStore` append-only, páginas, lecturas selectivas, auditoría, reapertura de solo lectura y detección de corrupción | [`src/cold_store.rs`](src/cold_store.rs) |
| Continuidad semántica | `SessionStore` con generaciones, hashes de estado, `CURRENT` atómico y rechazo de replay divergente | [`src/session_store.rs`](src/session_store.rs) |
| Continuidad física | `SessionCoordinator` expone snapshot/restore de la KV del runtime y separa esa responsabilidad de la memoria semántica | [`src/session_coordinator.rs`](src/session_coordinator.rs) |
| Contrato de runtime | Manifests, capacidades, selección por probe, handles de tensor y protocolo IPC versionado | [`src/generic_runtime.rs`](src/generic_runtime.rs), [`src/runtime_host.rs`](src/runtime_host.rs) |
| Contrato de adaptadores | `ModelAdapter`, `RuntimeSession`, generación, reset, snapshot, restore y recibos verificables | [`crates/kvslim-adapter-api/src/lib.rs`](crates/kvslim-adapter-api/src/lib.rs) |
| Model bundles | Validación de `config.json`, tokenizer, shards SafeTensors, geometría, nombres, dtypes físicos y layout del decoder | [`src/model_bundle.rs`](src/model_bundle.rs) |
| Catálogo y selección | Inventario efímero del host; diferencia entre compilado, disponible, no disponible y pendiente de probe | [`src/adapter_catalog.rs`](src/adapter_catalog.rs) |
| Grafo documental | Páginas originales, `PageMicroMetadata`, spans, números, referencias explícitas, vecinos y recuperación acotada | [`src/document_graph.rs`](src/document_graph.rs) |
| Recuperación con evidencia | Modos `focused` y `exhaustive`, candidatos diferidos y marca explícita de cobertura no exhaustiva | [`src/document_graph.rs`](src/document_graph.rs) |
| LOD y certificados | Ruta densa exacta cuando `epsilon = 0`; rutas aproximadas solo cuando el certificado autoriza el descarte | [`src/gpu_canonical_lod.rs`](src/gpu_canonical_lod.rs), [`src/attention_certificate.rs`](src/attention_certificate.rs) |
| Servicio local | Worker en loopback, gateway, endpoints de salud/capacidades/modelos y subconjunto HTTP compatible con OpenAI | [`src/bin/kvslim-runtime-worker.rs`](src/bin/kvslim-runtime-worker.rs), [`src/bin/kvslim-gateway.rs`](src/bin/kvslim-gateway.rs) |

## Lo que no se debe inferir

KVSLIM **no es todavía** un “ejecuta cualquier LLM en cualquier máquina” acabado, ni una aplicación de edición de código, ni una base vectorial semántica.

- Un adapter compilado no equivale a un runtime usable: el dispositivo, sus librerías y el modelo deben pasar el probe y la inspección.
- El catálogo remoto (actualmente opcional, con integración de Hugging Face) solo propone candidatos. La autoridad de ejecución es el bundle importado y verificado dentro de `runtime/models/`.
- Los drivers del sistema no se instalan de forma universal desde el repositorio.
- El worker HTTP actual no implementa streaming de respuestas, tools ni sampling general; `/v1/capabilities` declara `stream: false`, `tools: false` y `sampling: false`.
- El worker directo debe quedar en loopback: no tiene autenticación propia. El gateway separado sí valida bearer token y selecciona un worker certificado.
- Las pruebas físicas dependen de un modelo real y de hardware concreto; el job CI físico es manual (`workflow_dispatch`) y no se ejecuta en cada push.

## Invariantes de integridad

1. **El K/V original es la autoridad.** Las estructuras de micro-regiones, LSH y LOD sirven para localizar o certificar; no son una segunda verdad de baja precisión.
2. **La identidad se ata a hashes.** Modelo, tokenizer, arquitectura, prefijo, generación y snapshot se comprueban antes de restaurar.
3. **Las operaciones incompatibles fallan cerradas.** Un backend no puede ganar por prioridad si solo está compilado o pendiente de probe.
4. **El modelo activo es inmutable durante la ejecución.** Se selecciona por digest dentro de `deploy/local-production/runtime/models/` y se abre en modo de lectura.
5. **La evidencia no se sustituye.** El router documental conserva bytes originales y sus SHA-256; si el presupuesto difiere páginas candidatas, lo comunica.

## Backends y features de Cargo

El contrato público es neutral; las implementaciones concretas se compilan de forma optativa. La siguiente tabla refleja `Cargo.toml`, no una promesa de disponibilidad en el equipo actual.

| Feature | Superficie que compila | Estado real |
|---|---|---|
| `native-cpu-backend` (por defecto) | Runtime CPU Rust portable + bundles de modelo | Disponible cuando el bundle pasa la inspección; la arquitectura concreta sigue siendo una decisión del adapter |
| `torch-backend` | Runtime CPU basado en LibTorch y bins HTTP/CLI asociados | Opcional; requiere LibTorch y sus librerías externas |
| `accelerator-backend` | Worker acelerado, kernels y adapter concreto del árbol `opencl_*` | Opcional; requiere runtime/dispositivo compatible y probe físico |
| `hip-backend` | Bridge nativo hipBLASLt cargado explícitamente | Opcional; hardware y artefacto externos |
| `unified-router` | Gateway HTTP que no carga pesos | Opcional; necesita que exista un worker certificado |
| `model-catalog` | Descubrimiento de candidatos (Hugging Face) | Opcional; no autoriza por sí solo la ejecución |

La implementación actual conserva nombres concretos en la frontera de compatibilidad (`opencl_adapter`, `torch_adapter`, bridge HIP). El **núcleo** no selecciona por esos nombres: trabaja con `AdapterDescriptor`, `AdapterStatus`, `RuntimeManifest` y capacidades probadas. Esa es la separación que permite añadir otra implementación sin cambiar el contrato de sesiones.

## Ciclo de vida de un modelo

```text
descubrir candidato (opcional)
        ↓
importar dentro de runtime/models/<digest>
        ↓
validar configuración, tokenizer, shards y dtypes físicos
        ↓
inspeccionar el adapter y sondear el host
        ↓
certificar paridad, KV y reinicio cuando corresponda
        ↓
activar solo el runtime marcado RuntimeAvailable
```

Un modelo que solo puede inspeccionarse no se anuncia como ejecutable. Para modelos privados, el token de catálogo es de lectura y se guarda dentro del layout local; el secreto nunca forma parte del repositorio ni de los diagnósticos.

## Inicio rápido: núcleo portable

Desde la raíz del repositorio:

```bash
# Mantiene Cargo, temporales y target incremental dentro del proyecto.
deploy/local-production/scripts/install-layout.sh

# Perfil local; no versionar secretos ni credenciales.
cp deploy/local-production/env.example \
   deploy/local-production/local-production.env

# Compilación y pruebas del núcleo sin un backend opcional.
deploy/local-production/scripts/cargo-local.sh check --locked --no-default-features
deploy/local-production/scripts/cargo-local.sh test --locked --lib --no-default-features

# Smoke test de la referencia CPU y del contrato de inspección.
deploy/local-production/scripts/cargo-local.sh run --locked \
  --no-default-features --bin kvslim-inspect -- --self-test
```

La guía operativa completa está en [`deploy/local-production/README.md`](deploy/local-production/README.md). El wrapper local evita que el proyecto dependa de rutas globales para `CARGO_HOME`, `CARGO_TARGET_DIR` y temporales.

## Documentar y consultar un repositorio o corpus

El binario `kvslim-document` crea un grafo lossless; no genera un resumen que sustituya al original:

```bash
# Construir desde texto paginado.
deploy/local-production/scripts/cargo-local.sh run --locked --bin kvslim-document -- \\
  build-text \\
  --document-id proyecto \\
  --version REVISION_O_DIGEST \\
  --text documento.txt \\
  --output graph.json \\
  --page-characters 4000

# Recuperación determinista con páginas originales y digests.
deploy/local-production/scripts/cargo-local.sh run --locked --bin kvslim-document -- \\
  query \\
  --graph graph.json \\
  --text "¿dónde se valida la continuidad?" \\
  --scope focused \\
  --max-pages 8 \\
  --max-characters 32000 \\
  --graph-hops 1
```

La recuperación puede ser `focused` o `exhaustive`, está limitada por presupuesto y expone páginas candidatas diferidas cuando no puede afirmar cobertura completa. La tokenización exacta sigue perteneciendo al runtime del modelo seleccionado.

## Servicio local y continuidad física

El despliegue local separa responsabilidades:

- **Worker interno:** por defecto `127.0.0.1:8188`; carga el modelo y posee la cola de inferencia.
- **Gateway:** por defecto `127.0.0.1:8088`; autentica con bearer token y habla con un worker certificado.
- **ColdStore / SessionStore:** viven bajo `deploy/local-production/runtime/` y no se mezclan con el checkout del código.


Endpoints expuestos por el worker/gateway actual:

| Endpoint | Uso | Estado |
|---|---|---|
| `GET /health` | proceso, modelo y backend activo | operativo |
| `GET /ready` | disponibilidad del worker | operativo |
| `GET /v1/models` | modelo seleccionado | operativo |
| `GET /v1/capabilities` | capacidades, calibración y políticas KV/LOD | operativo |
| `POST /v1/chat/completions` | generación y operaciones de continuidad | subconjunto compatible; sin streaming/tools/sampling general |
Para una release instalada, la secuencia documentada es:

```bash
deploy/local-production/scripts/verify-release.sh
deploy/local-production/scripts/run-local-worker.sh
deploy/local-production/scripts/run-local-runtime.sh
```

`verify-release.sh` comprueba rutas, digests del binario y del worker, digest del árbol del modelo, permisos de solo lectura, bind loopback y formato de `ColdStore`. El servidor actual no debe exponerse a una red compartida sin una capa de autenticación y transporte adecuada.

El worker puede hablar con un runtime externo mediante un manifest IPC. El manifest debe demostrar como mínimo `prefill`, `decode`, `reset_session`, `snapshot_kv` y `restore_kv`; si falta una de esas capacidades, el arranque se rechaza. El proveedor real que existe hoy se encuentra en [`src/bin/kvslim-runtime-provider.rs`](src/bin/kvslim-runtime-provider.rs) y utiliza el adapter acelerado compilado para ese perfil.

## Pruebas y qué certifican

| Prueba | Qué demuestra | Qué no demuestra |
|---|---|---|
| `cargo test --locked --lib` | Invariantes del núcleo, almacenamiento, contratos y recuperación | Rendimiento de un modelo real |
| `scripts/e2e-runtime-ipc.sh` | Framing Unix, request IDs, identidad de sesión, generación determinista y restore de KV | Inferencia de un checkpoint real |
| `scripts/e2e-worker-external-restart.sh` | Ruta proveedor → IPC → worker HTTP → snapshot → reinicio → restore físico con un bundle local | Rendimiento universal o compatibilidad con cualquier modelo |
| Job CI `physical` | Paridad multi-step y calibración en un runner self-hosted con `KVSLIM_PHYSICAL_MODEL` | Disponibilidad permanente de ese hardware |
| Job CI `load` | Tests manuales del worker de inferencia | Certificación de backend acelerado |

No se publican tokens/segundo como una propiedad del producto sin medir el modelo, la precisión, el backend y la máquina concretos. La propia guía de despliegue registra un ensayo de referencia de continuidad física con restore exacto de 351 bytes y una medición de 103,56 s/token en modo `stream`; ese dato certifica la ruta y su coste en aquella máquina, no es un objetivo de producción.

## Comandos y binarios principales

Los targets disponibles están declarados en [`Cargo.toml`](Cargo.toml). Entre los más útiles:

- `kvslim-inspect`: inspección de bundle y smoke test CPU; no ejecuta un LLM completo por sí solo.
- `kvslim-models`: enumera bundles bajo el runtime local.
- `kvslim-import-model`: importa y digiere un bundle local (`model-import`).
- `kvslim-validate-model`: valida un bundle con las capacidades de modelo compiladas.
- `kvslim-adapters`: muestra adaptadores compilados y su estado.
- `kvslim-autoconfigure`: calcula un perfil a partir de bundles y capacidades observadas.
- `kvslim-runtime-host`: host IPC genérico; su modo de prueba usa una sesión determinista.
- `kvslim-runtime-provider`: proveedor externo del perfil acelerado actual.
- `kvslim-runtime-worker`: HTTP del worker; requiere `accelerator-backend` en la configuración actual.
- `kvslim-server`: gateway HTTP; requiere `unified-router`.
- `kvslim-document`: construcción y consulta del grafo documental lossless.

## Mapa del repositorio

```text
KV-SHEI/
├── crates/kvslim-adapter-api/    contrato estable de adapters y sesiones
├── src/                          núcleo Rust, KV, continuidad, catálogo e IPC
│   ├── cold_store.rs             páginas BF16 append-only y auditoría
│   ├── document_graph.rs         micro-metadatos y evidencia original
│   ├── generic_runtime.rs        manifests, capacidades y protocolo IPC
│   ├── session_store.rs          generaciones semánticas persistentes
│   └── session_coordinator.rs    unión entre sesión y KV físico
├── src/bin/                      workers, gateway, CLI y certificación
├── deploy/local-production/      layout aislado, scripts y perfiles operativos
├── catalogs/                     ejemplos de manifests de runtime
├── scripts/                      E2E IPC y restart externo
├── docs/                         contrato de adapters y guía documental
└── .github/workflows/            CI portable + matrices opcionales + jobs físicos manuales
```

## Roadmap honesto

Para llamar a KV-SHEI “runtime universal de producción” todavía habría que cerrar, y medir en hardware real:

1. más implementaciones de `ModelAdapter` fuera de los perfiles actuales;
2. paridad física reproducible por modelo y backend, no solo compilación de features;
3. una ruta externa completa con benchmarks comparables de resident/stream;
4. streaming, sampling y autenticación/transporte adecuados si el servicio va a salir de loopback;
5. una matriz pública de modelos certificados con sus hashes, licencias y resultados de conformance.

La arquitectura ya separa esas piezas. El README no las presenta como terminadas hasta que el código y las pruebas las demuestren.

## Documentación relacionada

- [Guía de despliegue local](deploy/local-production/README.md)
- [Contrato de adaptadores](docs/MODEL_ADAPTERS.md)
- [Documentación del núcleo](docs/README.md)
- [Tarea de runtime universal](docs/UNIVERSAL_RUNTIME_TASK.md)
- [Workflow de CI](.github/workflows/kvslim.yml)

<p align="center">
  <sub>KV-SHEI / KVSLIM · documentación alineada con el código fuente y sus pruebas actuales.</sub>
</p>
