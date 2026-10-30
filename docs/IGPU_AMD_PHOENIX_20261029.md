# Terrsvusion V3 — prueba de iGPU AMD Phoenix (29/10/2026)

## Alcance

Se ha probado **únicamente** el proyecto independiente en:

/home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3

La copia de seguridad original, creada antes de estas mejoras, permanece en:

/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-antes-mejoras-20261029

NO se ha modificado el módulo homónimo de SHEILY. NO se ha alterado el
servidor que estaba escuchando en 127.0.0.1:8099.

## Hardware detectado de verdad

| Componente | Valor medido |
| --- | --- |
| GPU | AMD Radeon Graphics (Phoenix), PCI 1002:15bf |
| Tipo OpenCL | GPU física, CL_DEVICE_TYPE_GPU = 4 |
| Driver DRM | amdgpu, /dev/dri/renderD128 |
| OpenCL | 3.0 vía Mesa Rusticl, driver 25.2.8 |
| Vulkan | Radeon RADV, API 1.4 |
| Unidades de cómputo | 12 |
| Memoria GPU reportada | 4.294.967.296 bytes |
| Límite de una asignación GPU | 2.147.483.647 bytes |
| Dtype de entrada | BF16, FP16 y FP32 en bytes sin pérdidas |
| Aritmética GPU del benchmark | FP32, tres kernels OpenCL propios |
| Lenguaje del host | Rust, sin PyTorch/LibTorch |

El identificador de GPU es el indicado por el driver; no se presupone
un modelo comercial Radeon 780M ni la cantidad de RAM libre para otros usos.

## Implementación

- Feature nueva: opencl-backend, optativa. La compilación Rust sin GPU
  continúa siendo la predeterminada.
- Archivo de kernels: assets/opencl_attention.cl
- Backend Rust: src/opencl_backend.rs
- Benchmark Rust: src/bin/terrsvusion-igpu-bench.rs
- Contrato: implementación real de AttentionBackend, no GPU simulada.
- En GPU se calculan QK, softmax estable y PV. Datos K/V BF16/FP16/FP32
  son transferidos sin cuantizar; lectura LE u16/u32 optimizada.
- El ejecutor comprueba que GPU sea CL_DEVICE_TYPE_GPU y little-endian
  antes de usar la ruta rápida; fallará explícitamente si no lo es.
- GPU tiene aritmética FP32 y referencia CPU acumula FP64.
  No se exige igualdad bit a bit con LibTorch; la referencia CPU es
  un control diagnóstico para medir diferencias numéricas.
- KV de distintos dtypes en un mismo lote todavía no se acepta
  en el backend OpenCL; se devuelve un error explícito.
- El backend GPU NO es el ejecutor completo de MiniCPM5-1B: no ejecuta
  el embedding, matrices de pesos de todas las capas, RoPE, RMSNorm,
  MLP ni decode autoregresivo de un LLM de principio a fin.

## Tests con GPU real

Comando ejecutado sobre hardware (no simulado):

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo test \
  --locked --features opencl-backend gpu_actual -- --ignored --nocapture
~~~

Resultado: 5 de 5 pruebas GPU correctas, 0 fallos:
- BF16 GQA
- FP16 GQA
- FP32 GQA
- FP32 MHA, 4 cabezas Q / 4 KV
- FP32 MQA, 4 cabezas Q / 1 KV

## Benchmarks reales optimizados

Datos Q/K/V deterministas sintéticos (NO pesos reales de MiniCPM).
Geometría del modelo real MiniCPM5-1B leída y validada: 16Q, 2KV,
head_dim 128. Rust release optimizado; 5 rondas GPU por configuración.
Se compara atención solamente frente a CpuReferenceBackend Rust.
Tiempos GPU incluyen transferencias Q/K/V, asignaciones, kernels,
lectura de salida, y gestión del driver; no incluyen carga de pesos
ni generación completa. Menos tiempo es mejor.

| Tokens | Formato | CPU Rust (ms) | Radeon iGPU (ms) | Relación | Error máximo |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | BF16 | 4,70 | 2,52 | 1,86x | 2,42e-8 |
| 4.096 | BF16 | 38,32 | 18,95 | 2,02x | 3,35e-8 |
| 8.192 | BF16 | 79,73 | 37,21 | 2,14x | 3,17e-8 |
| 32.768 | BF16 | 311,21 | 254,93 | 1,22x | 2,51e-8 |
| 1.024 | FP16 | 10,05 | 4,24 | 2,37x | 2,42e-8 |
| 1.024 | FP32 | 10,08 | 4,52 | 2,23x | 3,54e-8 |

La optimización de cargas alineadas mejora algunas geometrías, pero
también hay variabilidad entre corridas. No extrapolar los speedups
a todos los contextos ni al sistema LLM completo. La mayor parte del
tiempo en 32K se consume en el kernel QK (aprox. 195 ms de la última
muestra); este kernel todavía necesita optimización sustancial.

La tabla anterior se basa en:
- docs/IGPU_OPENCL_MEDICIONES_20261029.json (versión inicial)
- docs/IGPU_OPENCL_MEDICIONES_FASTPATH_20261029.json (ruta rápida)

## Prueba adicional de utilización física del hardware

Se ejecutaron 20 rondas BF16 con 32.768 tokens, mientras se leyó
/sys/class/drm/card1/device/gpu_busy_percent cada ~55 ms:

- 100 muestras observadas;
- ocupación mediana de GPU: 74 %;
- ocupación máxima: 86 %;
- mediana de ejecución GPU por ronda: 253,50 ms;
- diferencia máxima frente a referencia CPU: 2,51e-8.

Los datos completos se guardan en:
docs/IGPU_AMDGPU_BUSY_20261029.json

Esta medición, junto a la enumeración CL_DEVICE_TYPE_GPU, la carga
de kernels y los eventos de perfilado OpenCL, demuestra uso real de la
Radeon. No supone ni mide ahorro de VRAM de un LLM completo.

## Reproducir en el PC

Desde el proyecto Terravision V3 del Escritorio:

~~~bash
cargo fmt --all -- --check
cargo test --locked --offline --all-targets
cargo clippy --locked --offline --all-targets -- -D warnings

cargo test --locked --features opencl-backend gpu_actual \
  -- --ignored --nocapture

cargo run --release --locked --features opencl-backend \
  --bin terrsvusion-igpu-bench -- \
  --model /home/yo/models/MiniCPM5-1B \
  --tokens 8192 --dtype bf16 --rounds 5

cargo run --release --locked --features opencl-backend \
  --bin terrsvusion-igpu-bench -- \
  --tokens 1024 --dtype f16 --rounds 5
~~~

Para elegir dispositivo GPU distinto en sistemas con varias GPUs,
establecer la variable TERRSVUSION_OPENCL_DEVICE con parte del nombre
o fabricante anunciado por OpenCL. No se utiliza software fallback CPU.

Para conservar el comportamiento BF16 MiniCPM5-1B heredado, continuar
con --features torch-backend y el servidor antiguo; ningún usuario
debería activar el backend nuevo creyendo que transforma todo el
modelo automáticamente.

## Limitaciones y próximos hitos

1. Kernels de atención GPU optimizados para 8K, 32K, 100K y streaming,
   sin pagar por reconstrucción densa ni múltiples copias grandes.
2. Integración de atención OpenCL con modelo LLM completo y pruebas
   estrictas de logits/IDs/KV frente al mismo modelo de referencia.
3. Implementación y validación de proyecciones de pesos, RoPE,
   normas y MLP dentro del backend GPU Rust.
4. Datos de modelos reales con benchmarks de VRAM/RSS/prefill/decode,
   no solo QKV sintéticos.
5. Otros controladores: Intel/OpenCL, CUDA, ROCm, Metal y Vulkan
   permanecen sin validar. Solo Radeon Phoenix/Rusticl se ha probado.

