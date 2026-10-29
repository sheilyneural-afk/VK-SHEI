# Terrsvusion V3 — inferencia LLM completa en AMD iGPU con Rust/OpenCL

> **Actualización posterior (29/10/2026):** la ruta GPU incorpora ahora prefill paralelo por lotes y ColdStore agrupado en páginas de hasta 64 tokens. Las cifras de GPU secuencial en este informe son HISTÓRICAS. Resultados nuevos en PREFILL_GPU_OPTIMIZADO_20261029.md.


> **Informe de etapa anterior.** Posteriormente, el 29/10/2026, se
> integraron en GPU los mismos módulos originales de CPU para el índice
> 4-bit/LSH, ColdStore BF16 y certificado exact-mass, además del chat
> unificado CPU/iGPU/Auto. La jerarquía branch-and-bound regional CPU
> aún no está en GPU. Estado actual:
> docs/LOD_CPU_COMPARTIDO_GPU_20261029.md

Fecha: 29 de octubre de 2026. Implementación y pruebas sobre el
Terravision V3 independiente del Escritorio del PC:

/home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3

NO se ha modificado SHEILY. El servidor BF16 CPU histórico de puerto 8099
se ha mantenido sin sustituir.

## Respaldo y alcance

Copia completa verificada ANTES de comenzar esta implementación:

/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-igpu-llm-20261029

El respaldo contiene 33 archivos; origen y respaldo coincidían tras
la copia y se verificó el SHA-256 del manifiesto Cargo.toml.

Se añade un ejecutor experimental paralelo, sin alterar la semántica
del módulo BF16 CPU histórico.

## Qué ejecuta realmente la GPU

Nuevo módulo Rust: src/opencl_llm.rs
Kernels: assets/opencl_llm.cl

El modelo BF16 LlamaForCausalLM tipo MiniCPM5-1B corre en la Radeon
integrada con estas operaciones neuronales en kernels OpenCL:

- Embedding de tokens BF16.
- RMSNorm por capa, en GPU.
- Proyecciones Q/K/V y O: GEMV sobre pesos BF16 originales.
- Rotary Positional Embeddings (RoPE) y GQA.
- Caché K/V BF16 residente en memoria del dispositivo.
- QK, softmax estable y PV para atención causal de decodificación.
- Residuos, SwiGLU, gate/up/down del MLP.
- RMSNorm final, LM head de 130.560 tokens y argmax.

CPU se usa para tokenización, planificación de las operaciones,
transferencias, lectura de token IDs, comunicación HTTP e integridad;
no para operaciones neuronales del LLM. Las operaciones con BF16
almacenan intermedios y KV como BF16, con acumuladores FP32 en kernels.
El programa Rust no carga PyTorch ni enlaza con LibTorch.

Se ha implementado generación autoregresiva por tokens, incluida la
carga del prompt. Todavía no se aplica prefill GPU por lotes; esto
limita su velocidad para documentos grandes.

## Hardware real

- GPU: AMD Radeon Graphics Phoenix [1002:15bf].
- Driver DRM: amdgpu, OpenCL 3.0 mediante Mesa Rusticl.
- 12 unidades de cómputo, memoria GPU comunicada de 4.294.967.296 bytes.
- Peso del checkpoint: 2.161.265.664 bytes BF16.
- Se ha probado carga residente de los pesos y modalidad alternativa
  de transferencia por capa, sin ninguna cuantización destructiva.
- GPU detectada explícitamente mediante CL_DEVICE_TYPE_GPU, no
  OpenCL CPU ni emulación. La ruta falla si no existe un dispositivo GPU.

## Pesos: comprobación completa de ausencia de pérdidas

Se han leído de vuelta desde los buffers de la GPU los 219 tensores
BF16, todos sus 2.161.265.664 bytes, en bloques de 1 MiB, y se han
comparado byte a byte contra el safetensors original.

Resultado: 0 bytes divergentes y los dos SHA-256 agregados iguales:

1fb1632db083ec58baad8d7870b8bcfa4d593af112a903e6c14c2d231c7c2b5d

Evidencia completa: docs/IGPU_FULL_MODEL_WEIGHT_READBACK_20261029.json

Este resultado DEMUESTRA preservación exacta de los pesos en la GPU.
No equivale a afirmar igualdad numérica de la ejecución frente
a una biblioteca de álgebra distinta.

## Resultados de inferencia real

### Prueba del mismo prompt: «Hola»

Tokens idénticos de entrada: 15. GPU y referencia BF16 densa CPU
generaron ambos los IDs 61973 y 61 (texto «¡H»).

| Medida | GPU Rust/OpenCL | Densa CPU LibTorch |
| --- | --- | --- |
| IDs generados | [61973, 61] | [61973, 61] |
| Primer token divergente | ninguno | ninguno |
| Logits L-infinito | diferencia máxima 0,5 frente a CPU | referencia |
| SHA-256 lógico KV | distinto del de CPU | referencia |
| Prefill y dos pasos en prueba inicial | 1,27 s (sin carga) | ~0,22 s |
| Modelo cargado en GPU | sí, peso BF16 intacto | no, denso CPU |

Diagnóstico granular, incluidas diferencias por capa:
docs/IGPU_FULL_LLM_KV_DIAGNOSTICO_HOLA_20261029.json

Se detectó divergencia numérica desde la capa 0 (2 elementos K
distintos), que se acumula en capas posteriores. Hubo 156.258
elementos BF16 K/V divergentes en la comparación de los dos pasos,
con diferencia máxima KV 0,41552734375. Los dos tokens elegidos
continuaron siendo idénticos.

Por tanto, no se puede afirmar paridad exacta de logits o KV entre
el backend Rust/OpenCL y BF16 LibTorch. La precisión de almacenamiento
de pesos se ha demostrado por separado.

### Prueba de respuesta más larga

Ruta GPU, prompt «Hola», 24 tokens, texto generado:

«¡Hola! Parece que has estado esperando de mí. ¿Cuál es tu nombre
actual y cuál»

Prefill ~1,15 s, generación completa ~3,13 s (excluida carga
de ~0,75 s). Datos completos:
docs/IGPU_LLM_RESPUESTA_HOLA_20261029.json

### Chat web GPU y múltiples peticiones

Nuevo servidor independiente en 127.0.0.1:8188 (no reemplaza
al servidor BF16 CPU en puerto 8099).

GET / sirve assets/chat-gpu.html.
GET /health, /ready, /v1/models, /v1/capabilities disponibles.
POST /v1/chat/completions acepta modelo terrsvusion-opencl-bf16.

Probado:
- «Hola» con 8 tokens: HTTP 200, respuesta «¡Hola! Parece que has».
- Segunda consulta idéntica: mismo texto, sin recargar modelo.
- «¿Cuánto es 2+2? Responde con un número.»: HTTP 200,
  respuesta «2+2 = 4».
- Peticiones con SSE no implementado y modelo desconocido
  rechazadas de forma explícita (HTTP 422 y 404).
- Las respuestas consecutivas comienzan con KV reseteado: no
  mezclan conversaciones ni recuperan fragmentos antiguos.
- El ejecutable del servidor Rust/OpenCL no enlaza con LibTorch.
- El servidor BF16 original del puerto 8099 se comprobó
  operativo durante las pruebas del nuevo servidor.

Una prueba adicional real de 48 tokens respondió una pregunta
sobre estrellas. AMD amdgpu indicó 125 muestras, ocupación mediana
59 %, máxima 65 %, confirmando actividad física durante la generación.
Datos completos en docs/IGPU_LLM_COMPLETO_TELEMETRIA_20261029.json.
La respuesta GPU y densa CPU compartieron un prefijo largo pero
divergieron en la parte final; NO es una prueba de precisión
semántica universal.

## Comandos para ejecutar y repetir

Desde la carpeta del proyecto del Escritorio:

### Inferencia completa en GPU sin PyTorch

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo run \
  --release --locked --features opencl-backend \
  --bin terrsvusion-opencl-llm -- \
  --model /home/yo/models/MiniCPM5-1B \
  --prompt "Hola" \
  --max-context-tokens 128 \
  --resident --max-new-tokens 24
~~~

Para equipos con poca memoria GPU se puede sustituir --resident por
--stream-weights. La modalidad streaming conserva los pesos originales
pero los transfiere por capa, aumentando su latencia.

### Chat web GPU

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo run \
  --release --locked --features opencl-backend \
  --bin terrsvusion-opencl-server -- \
  --model /home/yo/models/MiniCPM5-1B \
  --host 127.0.0.1 --port 8188 \
  --max-context-tokens 2048 --resident
~~~

En el navegador DEL PC:

http://127.0.0.1:8188/

API local:

~~~bash
curl -sS http://127.0.0.1:8188/v1/models

curl -sS http://127.0.0.1:8188/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"terrsvusion-opencl-bf16","messages":[{"role":"user","content":"Hola"}],"max_tokens":24,"temperature":0,"stream":false}'
~~~

Solo se ofrece un subconjunto OpenAI-compatible: generación greedy,
sin SSE streaming, sin muestreo top_p, sin herramientas,
sin formato JSON forzado y sin esquema de seguridad remoto.

IMPORTANTE: el servidor no tiene autenticación. Permanece en
127.0.0.1 por defecto y rechaza enlaces no locales sin autorización
explícita. No exponerlo directamente a Internet.

### Tests GPU físicos

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked \
  --features opencl-backend gpu_actual -- --ignored --nocapture

TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked \
  --features opencl-backend gpu_actual_full_minicpm5 -- --ignored --nocapture
~~~

Resultado validado: cinco pruebas de kernels MHA/GQA/MQA BF16/FP16/FP32
en la GPU; prueba adicional de inferencia MiniCPM5 completa dos veces,
con mismos tokens y el mismo estado KV entre sesiones GPU.

### Comparar GPU con BF16 denso original

Esta herramienta de AUDITORÍA usa el backend BF16 CPU existente
como referencia, pero el ejecutor GPU NO depende de LibTorch:

~~~bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/yo/.local/lib/python3.12/site-packages/torch/lib

TERRSVUSION_OPENCL_DEVICE=AMD cargo run --locked \
  --features 'torch-backend,opencl-backend' \
  --bin terrsvusion-compare-opencl -- \
  --prompt Hola --max-tokens 2 --max-context 64
~~~

## Limitaciones aún pendientes

1. El índice dimensional de cuatro bits, la caché fría exacta,
   el LOD certificado y la omisión tolerada del Terrsvusion original
   **NO se usan todavía** en la nueva ruta GPU; se conservan en
   el backend BF16 CPU histórico. Integrarlos requiere nuevas pruebas
   de logits y KV, para no introducir pérdida de información.
2. La ruta admite uno o varios shards BF16 y RoPE simple o `llama3`.
   MiniCPM5-1B tiene pruebas amplias; Llama-3.1-8B tiene por ahora
   carga, una inferencia mínima y comparación LibTorch de una posición,
   no certificación larga. Otros modelos, FP16/FP32 end-to-end y otros
   backends GPU no están certificados.
3. Existe prefill GPU por lotes de hasta 512 posiciones, pero 30K/100K/131K
   no son aún prácticos ni están certificados en esta ruta.
4. La GPU calcula FP32 y redondea intermedios BF16: la paridad de
   logits/KV con LibTorch BF16 NO es exacta.
5. No hay aún kernels GPU para la compresión/LOD original, atención
   fusionada eficiente, SIMD avanzado, paginación de KV por contexto
   largo ni herramientas OpenAI API avanzadas.
6. No se ha probado el rendimiento en CPUs o GPUs diferentes.
   El speedup observado en atención aislada no se puede extrapolar
   a la inferencia LLM entera. En 48 tokens la CPU densa estuvo en
   torno a 3,02 s y GPU en torno a 5,91 s. No afirmar que GPU es
   más rápida globalmente.

## Regla de seguridad del proyecto

La inferencia previa y los resultados históricos siguen disponibles
con --features torch-backend. La implementación OpenCL es optativa
y separada; no ha reemplazado el servicio CPU preexistente. Todos
los datos y la copia previa permanecen recuperables.

## Llama 3.1 8B multishard y RoPE llama3: evidencia física

La ruta OpenCL ya resuelve tensores directamente entre los cuatro shards,
sin concatenar el checkpoint. Un test contra el checkpoint local encontró
291 tensores y 16.060.522.496 bytes. Los nombres duplicados entre shards
se rechazan. El modo streaming conserva mmap de solo lectura y sube una
capa cada vez.

Se implementó el escalado RoPE `llama3` declarado en `config.json`:
factor 8, factores de frecuencia 1/4 y contexto original 8192. Ocho
frecuencias entre dimensiones 0 y 126 coinciden con la referencia
`transformers 4.51.3` dentro de 2e-6 relativo. El programa OpenCL completo
compiló físicamente en Radeon Phoenix.

Prueba `--load-only`, Llama-3.1-8B-Instruct, cuatro shards, streaming,
capacidad de dos posiciones:

| Métrica | Resultado |
|---|---:|
| Inicialización física | 0,829 s |
| RSS máximo | 2.181.392 KiB |
| Pesos reconocidos | 16.060.522.496 bytes |
| Memoria global reportada por iGPU | 4.294.967.296 bytes |

Inferencia mínima real, ID 128000, una posición y una salida:

| Métrica | OpenCL iGPU | LibTorch CPU |
|---|---:|---:|
| Token greedy | 14924 | 14924 |
| Inferencia | 1,953 s | 0,419 s |
| L-infinito logits | \- | 0,0625 frente a OpenCL |

La paridad de token en este único caso NO demuestra paridad de logits.
Solo 66.165 de 128.256 logits fueron bit a bit iguales. Hubo 31.299
elementos KV BF16 diferentes y `L∞ KV=0,015625`. Dos componentes V ya
difieren en la capa 0 y posición 0 (`4,768e-7`), donde RoPE es identidad;
esto localiza la primera divergencia en el orden/redondeo de proyección,
no en el nuevo escalado de posiciones. El modo streaming tampoco es más
rápido que LibTorch en esta geometría mínima. No se autoriza todavía el
servidor Llama OpenCL ni se extrapola esta prueba a contexto largo.
