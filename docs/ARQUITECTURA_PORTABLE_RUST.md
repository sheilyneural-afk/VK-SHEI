# Terrsvusion V3 — extensión portátil en Rust

Fecha: 29 de octubre de 2026. Esta mejora se aplica EXCLUSIVAMENTE a la
carpeta Terravision V3 del Escritorio del PC. El respaldo original fue creado
ANTES de editar, en:

/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-antes-mejoras-20261029

La copia y el origen coincidían en 20 archivos y contenido (diff -qr).

## Contrato y límites

El núcleo general ahora compila en Rust sin PyTorch ni LibTorch. La ruta
end-to-end original para MiniCPM5-1B BF16 se mantiene como backend opcional
(torch-backend), porque sus pesos, atención y caché BF16 tenían pruebas de
paridad importantes. No se ha sustituido por una cuantización con pérdidas.

La universalidad se construirá por adaptadores. La inspección de formatos o
modelos NO significa que ya puedan ejecutarse. La disponibilidad de una GPU
tampoco significa que existan kernels compatibles.

| Módulo | Función | Estado |
| --- | --- | --- |
| src/model_config.rs | config.json, MHA/GQA/MQA, BF16/FP16/FP32, contexto declarado | Probado |
| src/portable.rs | KV en RAM que conserva cada bit, atención CPU y contrato de backend | Probado |
| src/portable_store.rs | KV SSD, SHA-256, commit y recuperación tras append interrumpido | Probado |
| src/torch_runtime.rs | Inferencia MiniCPM5 BF16 y comparador contra ruta densa | Conservado |
| src/bin/terrsvusion-inspect.rs | CLI de inspección sin cargar pesos | Probado |
| src/bin/terrsvusion-server.rs | Chat web, modelo denso y API OpenAI-compatible limitada | Probado |

### Precisión y memoria

- RawTensor y PortableKvStore almacenan bytes BF16/FP16/FP32 originales,
  incluyendo valores especiales, sin redondear sus bits.
- La atención de referencia CPU calcula en f64 y devuelve f32. No reproduce
  automáticamente los redondeos nativos de LibTorch BF16.
- PortableKvStore no comprime a cuatro bits: los payloads conservan 16/32 bits
  por valor, más cabeceras e integridad. El índice original de 4 bits sigue
  siendo un índice auxiliar, no una cuantización de los pesos BF16.
- La referencia CPU usa softmax online y descodifica valores individuales:
  no crea una copia f32 completa de cada frame KV.
- El KV persistente tiene SHA-256 de geometría y bytes y marcador de commit.
  Un append incompleto no se publica y un registro confirmado corrupto falla.
- El almacén nuevo NO sustituye todavía al ColdStore BF16 histórico. Ningún
  archivo persistente anterior se migra automáticamente.

## Compilación Rust sin PyTorch

Ejecutar desde la raíz del proyecto:

~~~bash
cargo fmt --all -- --check
cargo check --locked --offline
cargo test --locked --offline --all-targets
cargo clippy --locked --offline --all-targets -- -D warnings
cargo run --locked --offline --bin terrsvusion-inspect -- --self-test
cargo run --locked --offline --bin terrsvusion-inspect -- --model /home/yo/models/MiniCPM5-1B
~~~

El inspector informa de geometría y formatos, no de ejecución universal.

## Inferencia MiniCPM5 BF16 (backend heredado opcional)

~~~bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/yo/.local/lib/python3.12/site-packages/torch/lib

cargo test --locked --offline --features torch-backend --all-targets
cargo clippy --locked --offline --features torch-backend --all-targets -- -D warnings
cargo run --release --locked --offline --features torch-backend \
  --bin terrsvusion-server -- \
  --model /home/yo/models/MiniCPM5-1B --host 127.0.0.1 --port 8088
~~~

La ruta heredada conserva dimensiones MiniCPM5: 24 capas, 16 cabezas Q,
2 cabezas KV, dimensión 128, vocabulario 130560, RMSNorm 1e-6 y RoPE 5e6.
Otra configuración se rechaza antes de mapear pesos. Es una limitación
explícita para evitar afirmar compatibilidad y perder exactitud.

## Chat web y APIs

| Endpoint | Función |
| --- | --- |
| GET / | Chat principal Terrsvusion |
| GET /nativo | Chat MiniCPM BF16 de control |
| GET /health | Estado y contexto declarado, NO certificado |
| GET /ready | Estado tras arranque de workers |
| GET /v1/models | IDs terrsvusion-rust y minicpm-dense-bf16 |
| GET /v1/capabilities | Capacidades reales y límites |
| POST /v1/chat/completions | Compatibilidad parcial OpenAI, sin streaming SSE |

~~~bash
curl -sS http://127.0.0.1:8088/v1/models
curl -sS http://127.0.0.1:8088/v1/chat/completions \
 -H 'content-type: application/json' \
 -d '{"model":"terrsvusion-rust","messages":[{"role":"user","content":"Hola"}],"temperature":0,"stream":false,"max_completion_tokens":8}'
~~~

La API soporta generación greedy, una respuesta y selección de modelo.
Rechaza explícitamente streaming SSE, muestreo no implementado, herramientas,
n>1, stop personalizado, formato JSON forzado y parámetros desconocidos.
No ignora opciones que podrían alterar el contrato de inferencia.

Una API remota no expone por sí misma pesos ni KV de su proveedor.
Conectar clientes OpenAI-compatible no instala Terrsvusion dentro de un LLM
ajeno al proceso.

SEGURIDAD: actualmente no hay autenticación incorporada. Mantener el
servidor en 127.0.0.1. Exponerlo en red requiere proxy TLS, autenticación,
límites de acceso y políticas explícitas.

## Compatibilidad real

| Requisito | Estado |
| --- | --- |
| Rust puro para metadatos, KV y atención CPU de referencia | Implementado y probado |
| BF16, FP16 y FP32 preservados bit a bit en RAM y SSD | Implementado y probado |
| MiniCPM5 BF16 inferencia completa | Sí, con backend LibTorch opcional |
| Llama/Qwen/Gemma/Mistral arbitrarios end-to-end | Pendiente |
| MiniCPM5 completo sin LibTorch | Implementado experimentalmente con Rust/OpenCL en AMD Phoenix |
| Ejecutor GPU con índice 4-bit, LSH, ColdStore y certificado originales | Integrado y probado; branch-and-bound regional y prefijos persistentes CPU pendientes |
| Chat unificado CPU/iGPU/Auto | Operativo en Rust, 8288, sin tocar cada backend |
| AMD Phoenix OpenCL, atención QK/softmax/PV | Implementado y probado en iGPU real con BF16/FP16/FP32 |
| CUDA/ROCm/Metal/Intel XPU/Vulkan | Otros backends no implementados ni validados |
| FP8/INT4/GGUF ejecutables | Pendiente |
| OpenAI-compatible sin SSE ni tools | Operativo en local |
| SSE, tools, JSON Schema y autenticación | Pendiente |
| 32K/100K/131K certificados universalmente | No demostrado |
| Paridad de logits universal entre dispositivos | No demostrado |

## GPU AMD Phoenix verificada el 29/10/2026

Se añadió la feature optativa opencl-backend, sin dependencia Python.
Hardware probado: AMD Radeon Graphics Phoenix, OpenCL 3.0 Mesa Rusticl.
El backend nuevo puede ejecutar atención de referencia en GPU en las
geometrías MHA/GQA/MQA, leyendo K/V BF16/FP16/FP32 sin cuantización
destructiva. Usa FP32 para el cálculo y devuelve salidas cercanas, no
necesariamente idénticas bit a bit a las de la CPU o LibTorch BF16.

Las cinco pruebas de atención GPU aislada compararon Q/K/V sintéticos
con Rust CPU, con ratios de latencia de 1,2–2,1 según el contexto.
Más tarde, EL MISMO DÍA, se implementó además una ruta COMPLETA de
MiniCPM5 BF16 sobre GPU, con todos los operadores neuronales en OpenCL.
En esa ruta la carga BF16 es exacta bit a bit, pero la aritmética
no es bit-exacta frente a LibTorch. Después se integraron los MISMOS
módulos Rust originales de CPU para 4-bit/LSH, ColdStore BF16 y
certificado de masa final; no se ha trasladado aún el branch-and-bound
regional ni se ha demostrado una aceleración end-to-end. Véase
docs/LOD_CPU_COMPARTIDO_GPU_20261029.md.

Detalle, comandos, cifras y monitorización /sys de GPU en
docs/IGPU_AMD_PHOENIX_20261029.md.

## Reglas de avance

1. Separar adaptadores de LLM: tokenizer, pesos, atención, RoPE, MLP,
   máscaras y chat template. Cada arquitectura necesita pruebas.
2. Un dispositivo GPU solo figura como compatible si tiene kernels reales
   y se ha medido uso de VRAM y latencia.
3. Cualquier optimización que afecte logits o KV necesita comparación
   en idénticos tokens de entrada, pesos y dispositivo.
4. En modo exacto no introducir cuantización destructiva de 4 bits.
5. Auditar RSS, VRAM, tiempo de carga, prefill y decode.
6. No alterar la caché histórica sin migración explícita, respaldo y
   verificación del SHA lógico de KV.

Véase también docs/VERIFICACION_20261029.md.


## GPU BF16 con prefill paralelo (29/10/2026)
Los kernels OpenCL nuevos procesan prompts por lotes causales.
La copia BF16 V temporal se elimina del espacio GPU después del
prefill; la autoridad pasa al mismo ColdStore CPU en páginas
agrupadas (hasta 64 tokens) con hashes individuales verificables.
Se ha probado paridad exacta del GPU previo en logits BF16 y SHA
de KV para 375 tokens en tres lotes; no hay garantía de paridad
bit a bit contra LibTorch CPU. Consulte
docs/PREFILL_GPU_OPTIMIZADO_20261029.md.

