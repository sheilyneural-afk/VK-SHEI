# Terrsvusion V3 — atención GPU BF16 exacta fusionada

Fecha: 29/10/2026. Modelo MiniCPM5-1B BF16 original. Radeon Phoenix física.

Proyecto: /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3/
Respaldo anterior completo: /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-fused-exact-attn-20261029/

## Cambio físico

Se incorporó el kernel llm_exact_hot_fused_attention en assets/opencl_llm.cl,
accesible únicamente con la opción --gpu-exact-fused-attention.
Ejecuta QK BF16 reales, softmax FP32 y PV BF16 en una única operación
OpenCL para cada cabeza de consulta, con K/V BF16 originales en GPU.
No se descarta ninguna posición ni se altera el índice 4-bit/LSH.
La ruta anterior QK + softmax + PV (tres kernels) permanece disponible
con --gpu-exact-original-attention y sigue siendo el default del ejecutable.

El selector exacto permite la fusión para K/V calientes hasta 2048 tokens.
En contextos superiores se utiliza automáticamente la atención original.
Se verificó la integración en LOD epsilon=0 con ColdStore original,
en modo GPU denso y en el fallback de micrometadatos positivos cuando
el certificado no puede omitir K. Una selección realmente dispersa
con epsilon positivo continúa utilizando su propio kernel certificado.

## A. Comparación de atención física, Radeon OpenCL

72 iteraciones por tamaño, paridad BF16 completa contra los tres kernels originales:

| Tokens K/V | Anterior (ms) | Fusionado (ms) | Aceleración |
|---|---:|---:|---:|
| 32 | 0.590 | 0.269 | 2.19× |
| 375 | 1.440 | 0.756 | 1.91× |
| 1024 | 2.318 | 1.237 | 1.87× |
| 2048 | 3.117 | 1.562 | 2.00× |

También se comprobaron bit a bit contextos de 1, 2, 15, 127 y 512 tokens.
En todas estas pruebas se compararon los 4096 bytes BF16 de salida
(16 cabezas Q × 128 dimensiones × 2 bytes), sin un solo mismatch.

## B. MiniCPM5-1B real: 375 entrada + 16 de salida

Modo GPU BF16 LOD epsilon=0, staging 64, prefill por lotes 128,
pesos BF16 residentes y V caliente 512 tokens. Tres ejecuciones
secuenciales por variante, alternando el orden, compilación Rust release.
Los tiempos de inferencia NO incluyen la carga del modelo.

| Medida | Original | Fusionado |
|---|---:|---:|
| Prefill | 2.586 s | 2.606 s |
| Decode + sello ColdStore | 2.125 s | 1.739 s |
| Tiempo total | 4.705 s | 4.428 s |

Aceleración de decode: 1.222×.
Aceleración total: 1.063×.
Todos los IDs generados fueron idénticos en las seis ejecuciones.

Test integral independiente: mismos 16 pasos, logits completos
BF16 bitwise iguales y SHA lógico exacto de K/V/ColdStore idéntico.
Test adicional: GPU densa Hola 15 + 16 también produjo todos los
logits BF16 y el mismo SHA de KV. Test de fallback micro
epsilon positivo: no alteró los logits BF16 ni el ColdStore
cuando no había ninguna región K que descartar.

## C. Servidor real en ejecución

Port 8188 respondió HTTP 200 a 375+16: prefill 2.345 s,
inferencia total 3.789 s, texto:
  El código del dato verificable es **CEDRO-7319**.
ColdStore: 168 páginas, 9761280 bytes.
V omitidas: cero. V recuperadas de frío durante el decode: cero.
Los originales BF16 se sellaron obligatoriamente antes de responder.

Selector CPU/iGPU/Auto en http://127.0.0.1:8288/
Servidor GPU OpenCL exacto en http://127.0.0.1:8188/
Servidor CPU no se modificó, en http://127.0.0.1:8099/

## D. Precisión, alcance y limitaciones

- Se ha demostrado igualdad BITWISE frente a la ruta GPU original
  en las entradas indicadas. Este es el criterio pertinente para
  la optimización exacta; no se exige igualdad bit a bit CPU LibTorch
  vs GPU OpenCL. Véase
  [criterios de validación](CRITERIOS_VALIDACION_CPU_GPU.md).
- No hay ninguna cuantización nueva ni pérdida de K/V. El índice y el
  almacenamiento frío siguen siendo los módulos canónicos CPU.
- La GPU comparte memoria con la CPU; durante el benchmark el servidor
  CPU estaba ejecutando otro checkpoint Llama-3.1-8B-Instruct con alta
  presión de RAM. Las latencias actuales NO deben compararse directamente
  con la antigua CPU MiniCPM5-1B sin repetir ambos bajo igual carga.
- La mejora global fue menor que la mejora micro-kernel porque el
  prefill (matrices GEMM BF16) no se ha modificado. El siguiente gran
  objetivo es la multiplicación matricial del prefill en un único GPU backend.
- No se certifica contexto por encima de 2048 tokens con este kernel
  fusionado: vuelve a los kernels exactos originales.
- Se detectaron ediciones paralelas del CPU durante el proyecto;
  ninguna de estas modificaciones GPU escribe los archivos CPU.

## Evidencia reproducible

- docs/GPU_FUSED_EXACT_ATTENTION_20261029.log
- docs/GPU_FUSED_EXACT_LLM_PARITY_375_20261029.log
- docs/GPU_FUSED_EXACT_MICRO_FALLBACK_375_20261029.log
- docs/GPU_FUSED_EXACT_DENSE_HOLA_PARITY_20261029.log
- docs/GPU_FUSED_EXACT_LLM_BENCH_375_20261029.json
- docs/GPU_FUSED_EXACT_HTTP375_20261029.json

Comando de prueba física (iGPU Radeon debe estar disponible):

    TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked --offline
      --features opencl-backend
      gpu_actual_exact_hot_fused_attention_parity_for_many_context_lengths
      -- --ignored --nocapture

Para seleccionar fusión exacta en el servidor ya cargado se requiere
arrancarlo con --gpu-exact-fused-attention (la configuración es de arranque).
Rollback explícito: --gpu-exact-original-attention.


## Auditoría final de compatibilidad concurrente

El 29/10/2026, después de ejecutar los tests, los servicios CPU 8099,
GPU 8188 y router 8288 respondieron HTTP 200. El puerto 8188 informa
epsilon=0, hot-V=512, deferred ColdStore=true, y
gpu_exact_hot_fused_attention=true. CPU usa actualmente
Llama-3.1-8B-Instruct, distinto al checkpoint MiniCPM5-1B del benchmark GPU.

Se compararon por bytes los fuentes CPU con el respaldo anterior:
torch_runtime.rs, cold_store.rs, adaptive_extent.rs,
attention_certificate.rs, inference_worker.rs y Cargo.lock siguen
idénticos. src/bin/terrsvusion-server.rs recibió una modificación
concurrente (gestión configurable del timeout de peticiones, último
mtime 17:07) procedente de otra actividad, NO de este trabajo GPU.
No se sobrescribió ni se intentó revertir. El servicio CPU sigue
respondiendo. No interpretar la frase «CPU preservada» como
«todos los archivos CPU idénticos tras trabajo paralelo».
