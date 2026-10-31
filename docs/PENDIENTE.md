# Trabajo pendiente para producción

Lo que se decidió no hacer todavía, con el motivo. Ninguno de estos puntos está
resuelto en el árbol actual.

## Partir los archivos gigantes

`src/gpu_llm/mod.rs` (unas 5.400 líneas) y `src/torch_runtime/kv_cache.rs`
(unas 4.700) siguen concentrando gran parte del código y del `unsafe`.
`gpu_llm` ya se separó parcialmente en `prefill.rs` y `gpu_resources.rs`;
queda partir el resto por responsabilidades para facilitar la revisión.

## Elegir el camino único hasta producción

Decisión de producto, no técnica. Propuesta: un modelo en CPU con ColdStore
exacto, empaquetado y ejecutable de principio a fin, con su validación y su
recibo. HIP, instalador, catálogo y router quedarían fuera de ese camino hasta
que el primero esté cerrado.

## Proteger `main` con pull requests obligatorios

`main` ya rechaza empujes directos: los cambios entran por pull request. Falta
confirmar en la configuración de GitHub que se exige el job `core` del CI (que
ejecuta `scripts/check.sh`) antes de fusionar. El hook local
`scripts/install-hooks.sh` sigue siendo una barrera adicional.

## Canal de sinónimos en la recuperación documental

La medición de `docs/RETRIEVAL_EVALUATION.md` muestra que `routing` y `fused`
fallan cuando la pregunta y el documento usan palabras distintas para lo mismo.
Queda por implementar una expansión de consulta sin latencia añadida (raíces de
palabras y retroalimentación de pseudo-relevancia, derivadas solo del propio
documento) y medirla con las mismas 36 preguntas etiquetadas. Los embeddings
requerirían un modelo propio del proyecto, que hoy no existe.
