# Terrsvusion V3: micrometadatos y V BF16 residentes en iGPU

29/10/2026. Radeon AMD Phoenix1 gfx1103, Rust/Mesa OpenCL.
Checkpoint original MiniCPM5-1B BF16. Ninguna simulación.

Proyecto: /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3/
Respaldo verificado: /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-gpu-hot-micro-20261029/

## Implementación

- El mismo ParallelContextAxis de CPU y sus micrometadatos de
  relaciones y huellas 4-bit/LSH priorizan regiones.
- Las cotas conservadoras calculadas sobre K BF16 certifican
  omisiones solo si epsilon es explícitamente positivo.
- La selección de posiciones u32 permanece en la memoria GPU,
  y se reenvía únicamente cuando cambia el conjunto certificado.
- Las V originales BF16 permanecen en una caché GPU de capacidad
  limitada, mientras el mismo ColdStore de CPU conserva el
  registro exacto verificado y los hashes de cada K/V.
- Un nuevo kernel OpenCL fusiona QK, softmax y PV sobre las
  posiciones certificadas, sin staging V desde ColdStore.
- Si las cotas no permiten omitir K, se usa atención BF16
  completa en GPU con V caliente: ninguna omisión aproximada.
- Cuando el contexto supera la caché, se vuelve a la ruta
  original de lecturas verificadas y certificados.
- Epsilon cero no cambia: todos los K/V permanecen íntegros.
  Esta nueva ruta positiva es OPT-IN, no configuración predeterminada.

## Tiempos físicos: atención pequeña de 32 tokens BF16

96 iteraciones, 12 calentamientos, en AMD Phoenix real.
Medianas por consulta, incluyen certificado de regiones en CPU.

| Modalidad | Mediana |
|---|---:|
| GPU completa, V residente | 0.340029 ms |
| GPU micrometadatos anterior con ColdStore | 0.849687 ms |
| GPU micrometadatos con V e índice residentes | 0.151094 ms |

Aceleración frente a GPU completa: 2.25×.
Aceleración frente a la antigua ruta micro fría: 5.62×.
CPU planner: 0.016551 ms.
GPU fused kernel y espera: 0.133941 ms.
Subidas de índice: 1.
Reutilizaciones de índice: 108.
QK evitados: 32 de 64; V frías evitadas: 16 de 32.
Error BF16 real máximo: 0.000000002066 (epsilon = 0.001).

La aceleración de 2,25× está demostrada solo en esta operación
controlada, no en inferencia completa de cualquier modelo.

## MiniCPM5 completo: 375 tokens de entrada + 16 de salida

BF16 originales, greedy, epsilon=0.001, 512 tokens de capacidad.
Dos ejecuciones por modo, secuenciales; mismo prompt IDS.

| Métrica | Micro con V fría | Micro con V GPU |
|---|---:|---:|
| Inferencia | 5.075 s | 3.871 s |
| Prefill | 2.018 s | 1.988 s |
| Decode | 3.057 s | 1.884 s |

Aceleración inferencia: 1.31×.
Aceleración decode: 1.62×.
Cargas V desde ColdStore durante la inferencia: 137.411 frente a 0.

Los cuatro runs generaron los mismos IDs. El eje moderno detectó
regiones, pero ninguna pudo omitirse por completo bajo el
certificado. Por tanto la mejora global procede de la caché
BF16 caliente, NO del pruning QK en ese texto.

La prueba estricta GPU original vs micro hot comprobó identidad
de todos los logits y de 9.584.640 bytes BF16 K/V de las 24 capas,
también desde ColdStore. La comparación con LibTorch CPU solo sirve
como diagnóstico: no se requiere identidad binaria entre backends.
Véase [criterios de validación](CRITERIOS_VALIDACION_CPU_GPU.md).

## HTTP experimental

El servidor temporal GPU en puerto 8189 devolvió HTTP 200 al
prompt 375+16, prefill 1.825 s y
tiempo total 3.543 s.
Datos originales exactos conservados en ColdStore.
El puerto temporal se ha detenido tras el test.

## Uso

El servidor estable GPU en puerto 8188 continúa con epsilon=0,
y el motor CPU 8099 y selector 8288 conservan su propia ruta.

El nuevo modo requiere por ejemplo:
  --gpu-lod --lod-epsilon 0.001 --gpu-micro-hot-v-tokens 512
  --gpu-prefill-batch 128 --resident

Para volver a exactitud total, conservar:
  --gpu-lod --lod-epsilon 0 --gpu-hot-v-tokens 512

Esta opción gasta más memoria GPU para V y metadatos de índices
que la ruta V fría. No altera los pesos BF16 ni cuantiza K/V.
La caché GPU es un duplicado TEMPORAL de valores exactos: su
respaldo sigue siendo el ColdStore canónico del proyecto.

## Pruebas y evidencias

- docs/GPU_MICRO_HOT_PHYSICAL_MAIN_20261029.log
- docs/GPU_MICRO_HOT_MODELO_375_20261029.json
- docs/GPU_MICRO_HOT_HTTP375_20261029.json
- cargo test --features opencl-backend
  gpu_actual_micro_regions_skip_real_qk_and_cold_v_with_error_bound
- cargo test --features opencl-backend
  gpu_actual_micro_hot_model_375_bitwise_dense_fallback_and_cold_integrity

No se modificaron archivos CPU por esta optimización GPU.
Se detectaron modificaciones concurrentes en el sistema de CPU/eje;
no se sobrescribieron ni se restauraron desde la copia de seguridad.
