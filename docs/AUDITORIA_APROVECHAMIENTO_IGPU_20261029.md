# Terrsvusion V3 — Auditoría de aprovechamiento real de AMD iGPU

Fecha: 29 de octubre de 2026. Proyecto exclusivamente modificado:
/home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3

Copia íntegra ANTERIOR a estas modificaciones:
/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-optgpu-decode-20261029

## Respuesta cuantitativa

Sí, la Radeon AMD Phoenix gfx1103 ejecuta las operaciones neuronales
BF16 del MiniCPM5-1B real en Rust/OpenCL. No se utiliza PyTorch ni
LibTorch en su ruta OpenCL. Pero la GPU no está ocupada al máximo
durante el tiempo total de inferencia.

Antes de esta fase (medición de controlador amdgpu):
- Uso medio de GPU: 54.0 %.
- Uso mediano: 65.0 %.
- Uso máximo: 70 %.
- Respuesta 375+16: 3.267 s.
- Se detectaron 48 lecturas bloqueantes de K/V GPU -> host por
  token de decode, 24 capas, pese a estar V exacta en GPU.

Después de optimizar el sello frío y fusionar QKV y SwiGLU:
- Uso medio de GPU en una repetición HTTP: 54.1 %.
- Uso mediano: 66.0 %.
- Uso máximo: 71 %.
- El uso global no ha aumentado sustancialmente de forma repetible:
  otras limitaciones del kernel y del sistema siguen presentes.
- Todos los logits BF16 del ejemplo 375+16 y todo el KV (K GPU,
  V en ColdStore) coinciden con la ruta previa de GPU.
- La comparación diagnóstica con BF16 LibTorch CPU produjo diferencias
  numéricas. No se exige identidad bit a bit entre backends; la prueba
  de equivalencia de esta optimización es frente a la ruta GPU previa.

## Cuantificación de CPU frente a GPU con modelo cargado en memoria

Bench HTTP reales: 3 ejecuciones por backend, mismas peticiones,
375 tokens de prompt y 16 tokens de salida, modelos persistentes:

| Backend | Mediana total | Mediana prefill |
|:--|--:|--:|
| CPU densa BF16 (native=true) | 1.464 s | 0.526 s |
| AMD OpenCL BF16 LOD optimizada | 3.151 s | 1.881 s |

Relación GPU / CPU: 2.15x.
NO se ha superado todavía CPU densa.

En otra petición HTTP la GPU optimizada necesitó
3.009 s, prefill 1.828 s.
No presentar la mejor muestra aislada como mediana.

## Optimización 1: sello BF16 en lote al finalizar el turno

El prefill sigue creando los K/V ColdStore exactos y el índice 4-bit
original. Durante el decode epsilon=0 con todos los tokens V
disponibles en GPU, los nuevos K/V BF16 permanecen dentro del
dispositivo. Se evita la lectura bloqueante de 2 tensores por capa
y token. El resultado se sella ANTES de devolver la respuesta,
usando el mismo append_batch de ColdStore y los mismos hashes/index.

| Métrica con 375+16 | Inmediato (mediana) | Diferido (mediana) |
|:--|--:|--:|
| Total | 3.283 s | 3.159 s |
| Decode con sello | 1.514 s | 1.358 s |

En la ruta diferida solo se crearon 168 páginas ColdStore frente
a 504 en modo inmediato. El sello de 360 pares token-capa BF16
tardó aproximadamente 0,01 segundos. Se ha probado equivalencia
de TODOS los K y V del ColdStore y los logits de 16 tokens frente
a la ruta original. Si el contexto excede el V caliente, desactiva
el diferimiento y vuelve a escribir y leer inmediatamente por la
ruta canónica. Un fallo de sello no puede devolver una respuesta
falsamente correcta: se produce error y se marca contexto corrupto.

Esta optimización no usa masa aproximada, no cambia un solo
BF16 del modelo, no omite V y no altera los archivos CPU.

## Optimización 2: QKV y SwiGLU BF16 fusionados

Se añadieron kernels OpenCL optativos que unen Q, K y V de
cada capa y que calculan gate/up/SwiGLU en una sola reducción.
Utilizan las mismas matrices, 128 hilos/reducción, fma FP32
y los mismos puntos de redondeo BF16. Los test físicos con
375+16 demostraron igualdad de los logits completos y de K/V.

| Medida 375+16 | Sin fusiones (mediana) | Fusionado (mediana) |
|:--|--:|--:|
| Decode y sello | 1.473 s | 1.365 s |
| Tiempo total | 3.533 s | 3.393 s |

La mejora es real pero pequeña y oscila con la carga del
sistema: no se declara una nueva superioridad sobre CPU.

## Por qué todavía no se aprovecha toda la GPU

1. El prefill OpenCL todavía calcula el BF16 GEMM con un conjunto
   de kernels de 128 hilos por fila y solo reutilización de pesos
   para 8 tokens. No está usando WMMA matricial como GEMM principal.
2. El decode batch=1 es GEMV, dominado por lecturas de >2 GB de pesos
   del modelo por paso. La memoria de la iGPU se comparte con CPU.
3. Hay muchos kernels y sincronizaciones por capa, aunque QKV y
   SwiGLU ya se han fusionado parcialmente. Las frecuencias del
   controlador han superado 2,7 GHz; no es inactividad física.
4. OpenCL no reconoce directamente el builtin AMD gfx1103
   __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 en este runtime
   Mesa Rusticl. Se probó compilar realmente y fue rechazado.

## HIP native gfx1103: experimentado, NO instalado en el chat

El equipo dispone de HIP/hipBLASLt y la biblioteca incluye
kernels gfx1103. Una operación GEMM BF16 representativa alcanzó
~2,0 TFLOPS sobre la Radeon, frente al prefill OpenCL actual
que está mucho por debajo en utilización efectiva.

Se implementó un backend experimental en Rust con puente
hipBLASLt para los pesos originales BF16; las demás operaciones
quedan en OpenCL. NO puede adoptarse como ruta predeterminada:
las copias de BF16 entre ambos contextos HIP/OpenCL deterioraron
el decode y en algunas pruebas largas ocurrió el error
HIP launch failure 719. Forzar gfx1100 mediante
HSA_OVERRIDE_GFX_VERSION en gfx1103 puede ser inseguro, por
eso NO se utiliza en el servidor.

Para superar de manera creíble CPU densa con precisión y LOD:
usar UN ÚNICO contexto HIP gfx1103 para todo el Transformer:
pesos originales BF16 y activaciones GPU residentes, GEMM
prefill de hipBLASLt con selección de algoritmos calibrados,
GEMV decode nativo especializado y kernels del mismo flujo
HIP para RoPE, RMSNorm, atención, SwiGLU y argmax. La CPU
seguiría usando el mismo índice 4-bit/LSH, ColdStore BF16 y
certificados de masa; intercambiar solo metadatos necesarios
y KV BF16 exactos en fronteras del turno.

Ese backend HIP de extremo a extremo AÚN NO ESTÁ IMPLEMENTADO;
no fingir que un microbenchmark GEMM es una victoria para
el modelo completo. Exigir tests de mismos IDs, logits BF16
y KV para cada nueva integración.

## Rutas, rollback y límites

Servidor OpenCL GPU: http://127.0.0.1:8188
Motor CPU original (nunca modificado): http://127.0.0.1:8099
Selector CPU / iGPU / Auto: http://127.0.0.1:8288

Las opciones GPU --gpu-deferred-cold y --gpu-fused-decode son
optativas, con --gpu-immediate-cold y --gpu-unfused-decode para
recuperar el flujo anterior. El selector Auto prioriza CPU LOD
por estabilidad y preservación de funcionalidades CPU.

El ColdStore y el índice 4-bit siguen siendo los módulos
canónicos originales. No se ha alcanzado el rendimiento CPU
denso; tampoco se ha certificado contextos de 32K/100K.

Evidencia original JSON:
- docs/GPU_ACTUAL_UTILIZATION_BEFORE_DECODE_OPT_20261029.json
- docs/GPU_ACTUAL_UTILIZATION_AFTER_DECODE_OPT_20261029.json
- docs/GPU_DEFERRED_COLD_VS_IMMEDIATE_20261029.json
- docs/GPU_DECODE_FUSED_BF16_375_20261029.json
- docs/GPU_OPTIMIZADO_SERVIDOR_VS_CPU_DENSO_20261029.json

Fuentes técnicas externas para el próximo backend:
- AMD GPUOpen WMMA RDNA3: https://gpuopen.com/learn/wmma_on_rdna3/
- Mesa Rusticl: https://docs.mesa3d.org/rusticl.html
- AMD gfx1103 rocBLAS issues: https://github.com/ROCm/rocm-libraries/issues/6320
