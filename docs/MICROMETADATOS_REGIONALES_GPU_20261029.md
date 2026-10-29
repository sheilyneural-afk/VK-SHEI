# Terrsvusion V3 — micrometadatos dimensionales y selección de K/V en GPU

Fecha de las pruebas: 29 de octubre de 2026.
Proyecto independiente: /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3/
Respaldo completo anterior a estas mejoras:
/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-micro-metadata-20261029/

## Objetivo

Utilizar más eficazmente los **micrometadatos originales** del eje
dimensional y sus relaciones para reducir trabajo de atención GPU y
lecturas de K/V frío, manteniendo la precisión BF16 original cuando
la tolerancia es 0.

**No se ha modificado ningún módulo del motor CPU.** El sistema base
sigue siendo la fuente de las regiones y del ColdStore:

- adaptive_extent::ParallelContextAxis, eje dinámico content-defined;
- FourBitFingerprint + índices originales LSH, 16 coordenadas
  compactas, dos nibbles por byte y una escala f32;
- SemanticExtent::micro_metadata::communicating_dimensions,
  validation_4bit, relation_version y propagated_relevance;
- cold_store::ColdStore y sus registros BF16 íntegros;
- attention_certificate::certified y select_by_exact_mass;
- El servidor CPU anterior de puerto 8099 sigue independiente.

## Implementación

Se añadió src/gpu_micro_regions.rs, sin sustituir ni copiar el
algoritmo CPU. Recibe los IDs de estado definidos por
ParallelContextAxis y conserva, por región/capa, mínimo y máximo
por dimensión de K BF16. Es un índice auxiliar exacto, no valores
de atención cuantizados.

La planificación funciona así:

1. GPU genera la Q real con RoPE y BF16 original.
2. El índice 4-bit/LSH CANÓNICO propone estados prioritarios.
3. Los micrometadatos comunicantes pueden priorizar sus vecinos.
4. La GPU utiliza el índice de cajas K BF16 originales para construir
   un límite conservador a la masa softmax de regiones no exploradas.
5. Si el límite cumple la tolerancia **positiva explícita**, se
   seleccionan regiones y el kernel OpenCL
   llm_micro_sparse_qk calcula Q·K REAL solo para ellas.
6. Las probabilidades descartadas se inicializan a -infinito y la
   softmax original normaliza las restantes; los valores BF16 V
   se recuperan del MISMO ColdStore verificado y se procesan en
   el mismo kernel GPU BF16 PV por páginas agrupadas.
7. Si la cota no es suficiente, el sistema usa la ruta previa:
   GPU Q·K completa, el mismo certificado final CPU exact-mass y
   ColdStore íntegro. Ninguna firma LSH o enlace puede autorizar
   por sí sola una pérdida de datos.

**Epsilon=0**: no se omite ni una K ni una V. No se asignan las
cajas regionales auxiliares. La ruta GPU prefill optimizada,
V caliente, certificado y sellado BF16 del modelo siguen sin cambios.

**Epsilon>0**: se permiten omisiones solo bajo certificado de error.
No se garantiza paridad bit a bit del resultado aproximado con la
ruta densa; sí integridad de los pesos y de las K/V guardadas.

## Reparación del sobrecoste anterior

En gpu_canonical_lod::select, el código GPU anterior hacía lo
siguiente incluso después del certificado de masa exacta:
union(seleccion_certificada, TODOS_los_candidatos_LSH).

Eso podía aumentar el número de V frías recuperadas sin aportar
un certificado más estricto. Ahora el certificado CPU original
determina la selección mínima, y el LSH/metadatos solo guía
prioridades. Se contabilizan candidatos del índice que ya no
fuerzan lecturas extra.

La política utiliza también:
- un recorrido lineal de rangos de estados ya clasificados, en vez
  de buscar todos los estados en cada token;
- límites auxiliares de 512 regiones por capa: si el texto es
  demasiado fragmentado, se libera el índice GPU auxiliar y vuelve
  a la ruta exacta de scoring/ColdStore;
- telemetría de memoria dedicada a cajas BF16, regiones, preguntas
  dispersas y fallbacks.

## Prueba física de reducción K + V (AMD Phoenix / Rust OpenCL)

En una prueba con 32 tokens y dos regiones dimensionales contrastadas,
ambas formadas por el algoritmo original ParallelContextAxis:

- Consulta Q BF16 real sobre la GPU.
- 16 posiciones BF16 K rechazadas por cota conservadora.
- **32 operaciones Q·K no ejecutadas** (2 cabezas de consulta).
- **16 V recuperadas**, en vez de las 32 del camino denso.
- Error real de la salida del kernel GPU BF16 PV frente a GPU densa
  ≤ 0,001, límite solicitado.
- Los kernels llm_micro_clear_scores y llm_micro_sparse_qk se
  ejecutaron físicamente y compararon contra QK+softmax+PV
  originales sobre la misma Radeon.

Test reproducible:
TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked --offline \
  --features opencl-backend gpu_actual_micro_regions_skip_real_qk \
  -- --ignored --nocapture

La prueba usa entradas sintéticas deterministas con K y Q reales
en los kernels GPU: prueba la cota y el funcionamiento del shader,
no demuestra aceleración o recuperación semántica en cualquier libro.

## Prueba física MiniCPM5-1B, 375 tokens de entrada y 16 de salida

Datos medidos en la misma iGPU BF16:

| Modo | Latencia total | V omitidas acumuladas | Q·K ahorrados por regiones |
|---|---:|---:|---:|
| Exacto epsilon 0, hot V | 2,925 s | 0 | 0 |
| Certificado epsilon 0,001 | 4,612 s | 469 | 0 |
| Certificado epsilon 0,1 | 4,392 s | 56.238 | 0 |

La diferencia de latencia también incorpora que la ruta exacta
mantiene una caché V caliente, mientras que la ruta positiva
actual recupera V desde el almacén frío.

En el contexto probado el eje original con sigma=3 delimitó
una sola región. Se midió además sensibilidad GPU sin modificar CPU:

| Política GPU | Regiones detectadas | Q·K regionales ahorrados | Inferencia 375 entrada, 8 salida |
|---|---:|---:|---:|
| sigma 3.0 (original) | 1 | 0 | 3,010 s |
| sigma 1.5 | 26 | 0 | 3,057 s |
| sigma 0.75 | 28 | 0 | 3,038 s |
| sigma 0.75 + fuse 64 | 28 | 0 | 3,076 s |

El número de regiones aumentó, pero las cajas K contienen
dimensiones demasiado diversas y NO certificaron omisiones de K.
Por tanto, no se ha probado speedup real de Q·K en MiniCPM.
**No activar sigma agresiva ni epsilon positivo por defecto.**

La CPU densa todavía fue más rápida en esta tarea
(~1,46 segundos en la comparación HTTP anterior). No se
afirma que el nuevo sistema supere la CPU.

## Compatibilidad con la arquitectura CPU

Se respeta el carácter content-defined de ParallelContextAxis:
las regiones no se usan como un tokenizer alternativo.
La sensibilidad solo puede configurarse ANTES de crear el KV
GPU, y se conserva durante el reset de cada sesión.

Los ajustes experimentales son:
--gpu-lod --lod-epsilon 0.001
--gpu-micro-novelty-sigma 1.5
--gpu-micro-warmup 8
--gpu-micro-open-fuse 64

Por defecto, el servidor GPU continúa con:
--gpu-lod --lod-epsilon 0
para conservar todos los valores BF16 de atención y no correr
riesgos de pérdida o aproximación.

La lógica canónica de CPU no ha cambiado ni ha sido sobrescrita.
El lector puede desactivar la función de sparse K volviendo a
epsilon 0 o al backup anterior.

## Limitaciones y siguientes trabajos

1. Las cotas por dimensión de regiones con K muy diversa son
   flojas, así que la selección puede tener que reabrir todas.
   Un siguiente intento útil sería refinar progresivamente
   regiones y calcular QK exactas de unas pocas antes de ampliar,
   usando el certificado incremental original de CPU.
2. El indexado de cajas BF16 es memoria auxiliar de CPU para GPU,
   sin reemplazar la caché fría BF16 y sin cuantizar pesos.
3. El certificado de masa no incluye una garantía formal global
   de todos los errores de coma flotante de un driver arbitrario,
   y no garantiza exactitud de logits del Transformer completo.
4. Todavía no es el port completo del branch-and-bound regional
   CPU ni la reutilización persistente de prefijos.
5. La paridad exacta entre CPU LibTorch y GPU OpenCL no está
   demostrada, aunque la GPU epsilon 0 conserva sus logits de
   referencia en las pruebas anteriores.

## Evidencias

- docs/MICROMETADATOS_GPU_EPSILON_375_20261029.json
- docs/MICROMETADATOS_GPU_SENSIBILIDAD_375_20261029.json
- src/gpu_micro_regions.rs (cotas, relaciones y tests)
- src/gpu_canonical_lod.rs (selector, micro-costes y CPU ColdStore)
- assets/opencl_llm.cl (kernel QK real selectivo)
- src/opencl_llm.rs (kernel CPU/GPU, fallback y tests físicos)

El respaldo original está en el Escritorio y la ruta CPU original
permanece operativa e independiente.

