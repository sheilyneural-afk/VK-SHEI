# Terrsvusion V3 — prefill paralelo AMD iGPU y ColdStore agrupado

Fecha: 29 de octubre de 2026. Fuente: mediciones reales de la Radeon Phoenix con MiniCPM5-1B BF16 original, en el único proyecto Terrsvusion V3 del Escritorio.

Respaldo intacto anterior: /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-prefill-gpu-20261029

## Resultado: prefill (segundos, menor es mejor)

| Entrada | Motor | Antes secuencial | Nuevo por lotes | Aceleración |
|:--|:--|--:|--:|--:|
| Aritmética (39) | gpu_dense | 3,021 | 0,835 | 3,619× |
| Aritmética (39) | gpu_lod | 3,831 | 0,812 | 4,720× |
| Fotosíntesis (42) | gpu_dense | 3,144 | 0,901 | 3,489× |
| Fotosíntesis (42) | gpu_lod | 3,975 | 0,901 | 4,410× |
| Documento (375) | gpu_dense | 30,826 | 7,423 | 4,153× |
| Documento (375) | gpu_lod | 51,158 | 7,179 | 7,126× |

## Tiempo total de inferencia (segundos)

| Entrada | Motor | Antes | Nuevo | Aceleración |
|:--|:--|--:|--:|--:|
| Aritmética (39) | gpu_dense | 3,693 | 1,339 | 2,759× |
| Aritmética (39) | gpu_lod | 4,588 | 1,484 | 3,093× |
| Fotosíntesis (42) | gpu_dense | 5,452 | 3,117 | 1,749× |
| Fotosíntesis (42) | gpu_lod | 6,954 | 3,732 | 1,863× |
| Documento (375) | gpu_dense | 32,363 | 8,759 | 3,695× |
| Documento (375) | gpu_lod | 54,085 | 9,684 | 5,585× |

## Cómo se reparó el cuello de botella

Se ha implementado un prefill layer-major por lotes de hasta 256 tokens, configurado inicialmente a 128 para la Radeon Phoenix. Embedding, productos de matrices BF16, RMSNorm, RoPE y atención causal usan kernels OpenCL reales; no se usa PyTorch ni LibTorch en la GPU.

Durante el prefill, las V exactas BF16 se almacenan temporalmente en GPU para evitar las lecturas en frío por token. Terminado el prefill, las K/V BF16 se sellan en páginas de hasta 64 tokens del mismo ColdStore de CPU, con checksums por token y SHA del registro original. La memoria temporal V se libera ANTES del decode. El CPU original no se ha modificado.

El GPU LOD mantiene K completas en GPU; las V exactas se recuperan del ColdStore con lecturas agrupadas por página y staging limitado. Los índices 4 bits/LSH y los certificados son los módulos canónicos de CPU; el branch-and-bound regional completo sigue pendiente en GPU.

## Operaciones frías: documento con 375 tokens y 16 de salida

| Contador | Antes | Nuevo |
|:--|--:|--:|
| Tokens-V leídos repetidamente a lo largo de las consultas de atención | 1,829,880 | 137,880 |
| ColdStore físico (bytes) | 11,232,000 | 9,815,040 |
| Páginas físicas | 9.360 (1 token/página en código anterior, 24 capas) | 504 |
| Lecturas frías (llamadas) | ~1,829,880 inferidas del código anterior | 5,040 observadas |
| Tokens por página en prefill | 1 | 64 |
| Buffer V temporal GPU (bytes) | 0 | 4,608,000 (se libera tras prefill) |
| Sellado en disco BF16 | Intercalado con las inferencias previas | 0,050 segundos |
| Capacidad GPU KV LOD estacionaria (512 tokens) | 6.332.736 | 6,332,736 |

Los tokens-V leídos son acumulados con repeticiones, NO tokens únicos. El contador de lecturas anterior es derivado de una llamada read_value_tokens por token; el posterior procede del ColdStore instrumentado.

## Validación y precisión

Las pruebas físicas AMD confirmaron:

1. Hola (15 tokens prompt) + 16 nuevos: GPU densa con prefill por lotes produce idénticos IDs, todos los logits BF16 y SHA de KV que la GPU densa token-a-token.
2. El mismo prompt en GPU LOD epsilon=0 mantiene los mismos logits BF16 que GPU LOD token-a-token, y reduce los registros ColdStore.
3. Documento (375 tokens) en 3 lotes: el prefill por lotes conserva SHA lógico KV y logits BF16 respecto al motor GPU denso anterior; GPU LOD con paginado coincide bit a bit con la GPU densa.
4. Aritmética, fotosíntesis y documento mantienen los mismos tokens generados que las antiguas rutas GPU.

No se afirma paridad numérica GPU contra CPU LibTorch BF16: el comparador anterior midió hasta 0,625 de diferencia entre logits a lo largo de 16 pasos, con SHA KV diferente. Tampoco se certifican 32K/100K en esta ruta.

## Arranque recomendado y vuelta atrás

Servidor GPU con prefill paralelo y GPU LOD BF16 exacto:

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo run --release --locked --features opencl-backend --bin terrsvusion-opencl-server -- --model /home/yo/models/MiniCPM5-1B --host 127.0.0.1 --port 8188 --max-context-tokens 2048 --resident --gpu-lod --lod-epsilon 0 --lod-stage 64 --gpu-prefill-batch 128
~~~

Con --gpu-prefill-legacy se vuelve al procesamiento anterior token por token. La configuración CPU, pesos y formatos de ColdStore no se alteran. Chat unificado en http://127.0.0.1:8288 y CPU original en http://127.0.0.1:8099.

## Límites

Las operaciones QK/softmax/PV del prefill siguen separadas, con scratch de scores FP32 limitado a 64 MiB; falta una atención fusionada y acceso paginado avanzado. El prefill GPU es más rápido que antes pero todavía más lento que CPU para el contexto de 375 tokens (CPU densa ~1,508 s, CPU LOD ~2,805 s, iGPU LOD nueva ~9,684 s total). En decode GPU sigue existiendo transferencia de V BF16 exacta del ColdStore. No se han implementado todavía los prefijos persistentes del CPU ni su branch-and-bound regional completo para GPU.

## Evidencia

Detalles medidos: docs/PREFILL_GPU_LOTES_COMPARATIVA_20261029.json.
Baseline: docs/COMPARATIVA_GPU_DENSO_VS_LOD_20261029.json.
Pruebas Rust incluidas en src/opencl_llm.rs y src/gpu_canonical_lod.rs.
