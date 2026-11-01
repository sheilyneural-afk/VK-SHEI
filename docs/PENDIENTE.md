# Trabajo pendiente para producción

Lo que se decidió no hacer todavía, con el motivo. Ninguno de estos puntos está
resuelto en el árbol actual.

```mermaid
flowchart LR
    I[Necesidad confirmada] --> D{¿Contrato decidido?}
    D -->|no| P[Decisión de producto pendiente]
    D -->|sí| E{¿Evidencia y autoridad disponibles?}
    E -->|no| A[Aplazado con motivo]
    E -->|sí| T[Implementar y verificar]
    T --> C[Retirar de este documento]
```

Este archivo contiene únicamente trabajo no cerrado. Una capacidad ya
implementada se describe en su documento operativo y aquí permanece solo la
parte que todavía carece de contrato o evidencia.

## Partir los archivos gigantes

`src/gpu_llm/mod.rs` (unas 5.600 líneas) y `src/torch_runtime/kv_cache.rs`
(unas 4.700) concentran gran parte del código. Partirlos por
responsabilidades facilitaría la revisión. El `unsafe` está más repartido: los
archivos con más bloques son `src/gpu_llm/gpu_resources.rs`, `src/native_hip.rs`,
`src/gpu_llm/mod.rs`, `src/gpu_llm/prefill.rs` y `src/gpu_backend.rs`.

Motivo del aplazamiento: la refactorización debe hacerse sobre un árbol sin
cambios pendientes en esos archivos y volver a pasar la matriz física de
paridad, porque `gpu_llm/mod.rs` contiene la ruta de inferencia validada.

## Elegir el camino único hasta producción

Decisión de producto, no técnica. Propuesta: un modelo en CPU con ColdStore
exacto, empaquetado y ejecutable de principio a fin, con su validación y su
recibo. HIP, instalador, catálogo y router quedarían fuera de ese camino hasta
que el primero esté cerrado.

## Semántica real en la recuperación documental

El canal por raíces y la retroalimentación de pseudo-relevancia ya existen y
están medidos en `docs/RETRIEVAL_EVALUATION.md`. Las raíces mejoran flexiones,
pero no relacionan sinónimos reales como "tarjeta gráfica" y "GPU"; la
retroalimentación permanece desactivada porque no mejoró de forma estable.

Queda por diseñar y medir un canal semántico con un modelo propio o un
diccionario versionado, sin convertir texto derivado en autoridad ni introducir
una dependencia de red. Antes de activarlo necesita un conjunto etiquetado más
amplio, medidas de latencia y una política explícita de fallback.
