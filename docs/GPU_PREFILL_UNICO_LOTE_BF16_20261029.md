# Terrsvusion V3 — Prefill BF16 de un solo lote, Radeon Phoenix

Fecha: 29 de octubre de 2026.

Proyecto: /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3/
Respaldo anterior: /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-prefill-fused-projections-20261029/

## Objetivo

Acelerar el PREFILL de un prompt NUNCA visto, sin depender de cache
de prompts ni de reutilizacion de prefijos. Preservar TODOS los bytes
BF16 de pesos y KV, ColdStore y el eje dimensional/LSH originales.

No se han modificado los archivos fuente CPU, ni el modelo original.
La ruta exacta epsilon=0 no descarta ninguna K o V. No se ha
modificado ni introducido cuantizacion de pesos.

## Experimentos descartados

Los kernels GPU fusionados de QKV y gate/up/SwiGLU fueron añadidos
solo como una opcion experimental, separados del camino estable.

En una comparacion real de matriz de 128 tokens, 40 mediciones:
- QKV en tres kernels: 4.185 ms.
- QKV fusionado: 8.009 ms (PEOR).
- Gate/up/SwiGLU anterior: 11.668 ms.
- Gate/up/SwiGLU fusionado: 10.601 ms.

En el modelo completo el SwiGLU fusionado NO mostro mejora consistente:
mediana original 2.402 s,
mediana solo SwiGLU 2.407 s.

Conclusión: fusión QKV OFF y fusión SwiGLU OFF por defecto.
Los kernels se han validado bit a bit pero no se activan como
'optimizacion' mientras no reduzcan el tiempo de extremo a extremo.

## Mejora util: un lote BF16 de hasta 512 tokens

Antes existia un limite artificial de 256 tokens por lote, mientras
que el presupuesto real de scratch se mantenia en 64 MiB.
Se amplio el limite opcional a 512 con las mismas verificaciones
de overflow, asignaciones GPU y scratch de atencion <=64 MiB.

En un prompt de 375 tokens, solicitar un lote de 384 permite
procesar exactamente las 375 posiciones en UNA pasada layer-major.
El lote original 128 requiere TRES pasadas. No cambia la formula
BF16, el orden original FMA, las reducciones de 128 lanes,
el softmax causal ni las paginas originales de ColdStore.

Se verificaron todos los logits BF16 de salida, todos los tokens,
y el SHA LOGICO completo de K/V BF16 con los tamaños de lote
64, 96, 128, 192, 256, 375 y 512; ninguno divergió.

## Medición de rendimiento real: BF16 prefill

Hardware real Phoenix gfx1103, Rust release, mismo modelo ya
cargado en GPU, mismos tokens de entrada y salida, 3 ejecuciones
por modo en orden alternado. Tiempos de inferencia SIN el upload
de los pesos. Equipo bajo FUERTE carga de CPU concurrente:

| Modo | Prefill mediano | Inferencia total mediana |
|:--|--:|--:|
| 128 tokens/lote, 3 pasadas | 4.036 s | 8.327 s |
| 375 tokens/lote, 1 pasada | 2.600 s | 5.591 s |

Speedup observado de prefill: 1.55x.
Este resultado es significativo pero la carga fluctuó debido a
procesos ajenos, por lo que NO se debe extrapolar la misma
aceleracion a equipos descargados o a prompts diferentes.
Se mantiene disponible y facil de revertir con
--gpu-prefill-batch 128.

En las ejecuciones mas tranquilas anteriores, lote 128/192/256
tuvo medianas prefill 2.137 / 2.124 / 2.310 s; 192 no
supone un beneficio claro y 256 fue peor.

## Memoria y precisiones

- Modelo 2.16 GB BF16 original, no cambiado.
- KV original BF16 en ColdStore, mismo indice de cuatro bits/LSH.
- Scratch QK de GPU limitado a 64 MiB; un solo lote
  375 tokens sobre 375 prompt usa 375*16*375*4 = 9 MB aprox.
- En contexto configurado de 2048, solicitar 384 tokens/lote
  puede reservar 384*16*2048*4 = 50.3 MB de scratch.
- Todos los buffers activacion batch son temporales, no
  una segunda copia permanente del modelo.
- La fusión QKV/SwiGLU queda como opcion experimental,
  el motor por defecto usa los kernels GEMV reuse8
  ya probados. Aritmetica FP32/BF16 sin alteraciones.
- Replay verificado y prefijos compartidos SHA256 permanecen
  operativos sin relación con este cambio de tamaño de lote.

## Estado operativo

GPU 8188 normalmente ejecuta epsilon=0 exacto y esta
configurado con --gpu-verified-prompt-replay 512,
--gpu-verified-shared-prefix, ColdStore diferido,
V caliente 512, decode y atención BF16 fusionadas.
El lote de prefill recomendado es --gpu-prefill-batch 384.
El CPU 8099 usa en este momento un checkpoint distinto,
Llama3.1-8B, por lo que NO hay una comparacion de velocidad
CPU MiniCPM5 vs GPU MiniCPM5 en las mismas condiciones.

## Evidencias

- docs/GPU_PREFILL_PROJECTION_FUSION_SEPARATE_BENCH_20261029.log
- docs/GPU_PREFILL_FUSED_MLP_ONLY_RELEASE_375_20261029.log
- docs/GPU_PREFILL_BATCH_64_96_128_192_256_BF16_20261029.log
- docs/GPU_PREFILL_BATCH_ONE_SHOT_375_BF16_20261029.log
- docs/GPU_PREFILL_ONE_SHOT_375_SUMMARY_20261029.json
- src/opencl_llm/prefill.rs, src/opencl_llm.rs,
  assets/opencl_llm.cl (solo fuentes GPU)

## Servidor activo: prueba HTTP real despues de instalar lote384

El servidor Rust/OpenCL GPU de puerto 8188 fue reiniciado con
--gpu-prefill-batch 384, epsilon=0 exacto, V caliente 512,
ColdStore diferido, atención exacta BF16 fusionada,
--gpu-verified-prompt-replay 512 y --gpu-verified-shared-prefix.

Prueba HTTP con MiniCPM5 BF16 y 375 tokens de entrada + 16 salida,
petición determinista CEDRO-7319:

| Medición de modelo ya cargado | Petición nueva | Replay verificado |
|---|---:|---:|
| GPU prefill | 2.312674 s | 0.104811 s |
| GPU inferencia completa | 3.853525 s | 1.697595 s |
| Pasadas BF16 de prefill | 1 | 0 |
| SHA-256 verificado al restaurar KV | no aplica | True |
| BF16 KV restaurados | 0 bytes | 9477120 bytes |

El modelo devolvió el código CEDRO-7319 en ambos turnos.
Se generaron exactamente 16 tokens en cada
petición y no se omitió ninguna V, con
168 páginas en ColdStore.

Durante la comprobación HTTP el servidor CPU 8099
(checkpoint Llama-3.1-8B) y el selector CPU/GPU 8288
también respondieron HTTP 200.

Logs reproducibles:
- docs/GPU_BATCH384_HTTP_NEW_PROMPT_375_20261029.json
- docs/GPU_BATCH384_HTTP_REPLAY_375_20261029.json

El servidor se puede devolver al comportamiento anterior
reiniciándolo con --gpu-prefill-batch 128; se preservan
todos los pesos BF16, K/V y ColdStore canónicos.
