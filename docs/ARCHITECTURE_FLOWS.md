# Arquitectura y flujos verificables de M-SLIM

Este documento es el mapa de navegación del sistema. Distingue código fuente,
artefactos instalados, evidencia local y procesos residentes: ninguna de esas
capas sustituye a las demás. Los recibos de `deploy/local-production/runtime/`
son locales, mutables e ignorados por Git.

## 1. De la fuente a una petición pública

```mermaid
flowchart LR
    S[Árbol fuente limpio] -->|build.rs| B[Binarios]
    S -->|source_revision + build_input_digest| M[Manifest de release v3]
    B --> M
    M -->|verify-release.sh| V{Release íntegra}
    V -->|no| X[Rechazo cerrado]
    V -->|sí| R[Recibo de bundle v2]
    R --> Q[Matriz exacta v2]
    Q --> C[Calibración sostenida v4]
    C --> W{Worker /ready}
    W -->|503| X
    W -->|200| G[Gateway autenticado]
    G --> P[POST /v1/chat/completions]
    P --> E[Respuesta + estado de sesión]
```

`/health` solo confirma que el proceso vive. `/ready` vuelve a comprobar el
modelo, la revisión fuente, el perfil efectivo y las evidencias exigidas. La
petición autenticada por el gateway es la comprobación de la superficie
pública; no se deduce de un health positivo.

## 2. Autoridad de cada evidencia

```mermaid
flowchart TB
    A[Fuente versionada] --> B[Release instalada]
    B --> C[Bundle inspeccionado y cargado]
    C --> D[Validación inspect/load/prefill/decode/snapshot/restore]
    D --> E[Matriz de cuatro rutas]
    E --> F[Calibración del perfil efectivo]
    F --> G[Readiness del worker]
    G --> H[Gateway y recorrido público]
    H --> I[Persistencia y recuperación tras reinicio]

    A -. no demuestra .-> G
    C -. no demuestra .-> H
    G -. no demuestra .-> I
```

| Capa | Demuestra | No demuestra por sí sola |
|---|---|---|
| Fuente | implementación y contratos versionados | binario instalado o proceso residente |
| Release | identidad de inputs y binarios instalados | carga física del modelo |
| Recibo de bundle | operaciones del adapter concreto | paridad entre rutas |
| Matriz | exactitud densa/ColdStore por backend | calidad general o restore |
| Calibración | medición sostenida del perfil congelado | servicio activo |
| `/ready` | aptitud local en ese instante | recorrido público o recuperación |
| E2E | comportamiento del recorrido ejecutado | otros hosts, modelos o revisiones |

## 3. Petición de chat

```mermaid
sequenceDiagram
    actor Cliente
    participant Gateway
    participant Worker
    participant Sesiones as SessionStore
    participant Runtime as Proveedor/ModelAdapter
    participant KV as ColdStore

    Cliente->>Gateway: Bearer + POST /v1/chat/completions
    Gateway->>Worker: comprobar /ready y reenviar
    Worker->>Sesiones: resolver namespace y generación
    Worker->>Runtime: prefill/decode
    Runtime->>KV: escribir/leer K/V BF16 exacto
    KV-->>Runtime: páginas verificadas
    Runtime-->>Worker: tokens, logits y diagnóstico
    Worker->>Sesiones: persistir estado causal
    Worker-->>Gateway: respuesta del modelo
    Gateway-->>Cliente: respuesta HTTP
```

El gateway autentica la entrada; el worker valida evidencia y transiciones; el
runtime ejecuta; `ColdStore` conserva el K/V físico; `SessionStore` conserva
identidad, generaciones y referencias. Ninguna capa debe asumir la autoridad
de otra.

## 4. Continuidad tras reinicio total

```mermaid
sequenceDiagram
    actor Cliente
    participant Worker
    participant Estado as SessionStore
    participant Provider as Runtime externo
    participant KV as ColdStore

    Cliente->>Worker: snapshot(handle, generación N)
    Worker->>Provider: SnapshotKv
    Provider->>KV: sellar prefijo físico
    Provider-->>Worker: recibo, tokens y bytes
    Worker->>Estado: append generación N y CURRENT
    Note over Worker,Provider: se detienen worker y provider
    Note over Worker,Provider: arrancan procesos nuevos
    Cliente->>Worker: resume(handle, generación N)
    Worker->>Estado: validar identidad, vigencia y límites
    Worker->>Provider: RestoreKv(snapshot persistido)
    Provider->>KV: reabrir y verificar páginas
    Provider-->>Worker: restore aceptado
    Worker-->>Cliente: continuidad restaurada
```

El E2E reinicia ambos procesos. Por eso la recuperación no puede depender de la
RAM del proveedor anterior. `release` escribe una lápida durable: el mismo
handle no vuelve a ser restaurable, tampoco después de otro reinicio.

## 5. Matriz de validación

```mermaid
flowchart TB
    I[Mismo bundle, prompt y decode forzado]
    I --> CH[CPU hot-dense x2]
    I --> CC[CPU exact-coldstore x2]
    I --> GD[GPU dense x2]
    I --> GC[GPU exact-coldstore x2]
    CH <-->|gate bit-exacto| CC
    GD <-->|gate bit-exacto| GC
    CH -. métricas diagnósticas .-> GD
    GC --> PS[Snapshot físico ColdStore]
    CH --> A{Todos los gates}
    CC --> A
    GD --> A
    GC --> A
    PS --> A
    A -->|pasan| OK[eligible-for-reviewed-activation]
    A -->|falla alguno| KO[status failed / blocked-by-same-backend-or-integrity-gate]
```

Cada ruta debe ser determinista frente a su repetición. El gate de integridad
compara densa y exacta dentro del mismo backend. CPU/GPU puede diferir
numéricamente y se conserva solo como diagnóstico.

## 6. Calibración adaptativa

```mermaid
flowchart LR
    O[Configuración base] --> P[Propuesta]
    P --> S[Shadow: muestras físicas]
    S -->|paridad o rendimiento fallan| R[Rechazo y recibo]
    S -->|pasa| C[Canary con muestras nuevas]
    C -->|falla| RB[Reversión]
    C -->|pasa + autoridad explícita| A[active.json promovido]
    A --> L[Carga con identidad completa]
    L -->|fuera de cobertura| F[Fallback configurado]
```

El predictor ordena candidatos, pero no activa nada. Shadow y canary miden el
motor físico; una promoción requiere autoridad explícita y queda ligada a
modelo, tokenizer, backend, hardware, implementación y configuración.

## Lectura siguiente

- [Validación exacta](VALIDATION.md)
- [Continuidad y propiedad](CONTINUITY_LIFECYCLE.md)
- [Adaptadores](MODEL_ADAPTERS.md)
- [Calibración adaptativa](ADAPTIVE_CALIBRATION.md)
- [Instalación local](../deploy/local-production/README.md)
