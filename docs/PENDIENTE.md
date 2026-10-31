# Trabajo pendiente para producción

Lo que se decidió no hacer todavía, con el motivo. Ninguno de estos puntos está
resuelto en el árbol actual.

## Partir los archivos gigantes

`src/gpu_llm/mod.rs` (unas 5.200 líneas) y `src/torch_runtime/kv_cache.rs`
(unas 4.700) concentran gran parte del código y del `unsafe`. Partirlos por
responsabilidades facilitaría la revisión.

Motivo del aplazamiento: `gpu_llm/mod.rs` tiene cambios sin commit de otra
sesión de trabajo, y una refactorización chocaría con ellos. Hacerlo cuando ese
archivo esté confirmado y sin cambios pendientes.

## Elegir el camino único hasta producción

Decisión de producto, no técnica. Propuesta: un modelo en CPU con ColdStore
exacto, empaquetado y ejecutable de principio a fin, con su validación y su
recibo. HIP, instalador, catálogo y router quedarían fuera de ese camino hasta
que el primero esté cerrado.

## Proteger `main` con pull requests obligatorios

Es una configuración del repositorio en GitHub, no de código. Debe exigir que
pase el job `core` del CI (que ejecuta `scripts/check.sh`) antes de fusionar.
Hasta entonces, el hook local `scripts/install-hooks.sh` es la única barrera.

## Canal de sinónimos en la recuperación documental

La medición de `docs/RETRIEVAL_EVALUATION.md` muestra que `routing` y `fused`
fallan cuando la pregunta y el documento usan palabras distintas para lo mismo.
Queda por implementar una expansión de consulta sin latencia añadida (raíces de
palabras y retroalimentación de pseudo-relevancia, derivadas solo del propio
documento) y medirla con las mismas 36 preguntas etiquetadas. Los embeddings
requerirían un modelo propio del proyecto, que hoy no existe.
