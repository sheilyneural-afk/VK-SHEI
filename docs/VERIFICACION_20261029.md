# Terrsvusion V3 — informe de verificación (29-10-2026)

**Interpretación vigente (30-10-2026):** las comparaciones CPU/GPU de
este informe son diagnósticas, no requisitos de igualdad bit a bit.
La evaluación de LOD GPU se hace contra GPU densa, y la integridad
de bytes BF16 se verifica independientemente.
Véase [criterio de validación](CRITERIOS_VALIDACION_CPU_GPU.md).

## Ubicación y respaldo

Proyecto modificado, exclusivamente:
  /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3

Copia íntegra ANTES de modificar nada:
  /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-antes-mejoras-20261029

Respaldo comprobado inicialmente con cp -a y diff -qr: 20 archivos,
3.285.359 bytes y mismo hash SHA-256 de Cargo.toml:
2fe3663f188c0a2d99911d3aa53b799993a398d24abc9ce0692c67678f496bd8

Ni SHEILY ni las otras copias de Terravision fueron modificadas
durante ESTA intervención.

## Baseline y cambios

Baseline original antes de esta fase: 55 tests Rust de biblioteca correctos
con el backend LibTorch activo. MiniCPM5-1B BF16, 24 capas, 16Q/2KV,
head_dim 128, pesos originales y ColdStore histórico.

Cambios realizados:
- Cargo.toml/lib.rs: LibTorch pasa a feature opcional torch-backend.
- model_config.rs: inspector de configuraciones HF, BF16/FP16/FP32
  y geometría MHA/GQA/MQA. La inspección NO ejecuta cualquier LLM.
- portable.rs: KV en RAM sin pérdida de bits y softmax CPU de referencia
  con acumulación f64, resultado f32, sin conversión total de frames.
- portable_store.rs: almacén nuevo de KV en disco, BF16/FP16/FP32 exactos,
  registros con SHA-256 de geometría y bytes, marcador de commit,
  lectura íntegra, recuperación de escritura incompleta y pruebas
  de corrupción. NO sustituye el almacenamiento BF16 anterior y NO
  constituye compresión de pesos a 4 bits.
- torch_runtime.rs: preflight del ejecutor MiniCPM5 anterior; falla
  claramente si la geometría no es soportada.
- terrsvusion-inspect.rs: CLI Rust nativa para introspección y self-test
  sin cargar todo el checkpoint ni requerir PyTorch/LibTorch.
- terrsvusion-server.rs: modelos y capacidades por HTTP, selección de
  ruta BF16 o Terrsvusion, validación de requests OpenAI-compatible.
- README.md y docs/ARQUITECTURA_PORTABLE_RUST.md: documentación fiel.

## Verificación final

Comandos ejecutados dentro del proyecto del Escritorio:

~~~bash
cargo fmt --all -- --check
cargo test --locked --offline --all-targets --quiet
cargo clippy --locked --offline --all-targets -- -D warnings

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/yo/.local/lib/python3.12/site-packages/torch/lib
cargo test --locked --offline --features torch-backend --all-targets --quiet
cargo clippy --locked --offline --features torch-backend --all-targets -- -D warnings

cargo run --locked --offline --bin terrsvusion-inspect -- \
  --model /home/yo/models/MiniCPM5-1B --self-test
~~~

Resultados finales:
- Ruta Rust sin LibTorch: 52 tests de biblioteca, 0 fallos.
- Ruta Torch opcional: 70 tests de biblioteca, 6 de servidor, 0 fallos.
- Clippy -D warnings y formato: sin advertencias o errores.
- Inspector con MiniCPM5-1B real: 24 capas, 16 cabezas Q, 2 KV,
  head_dim 128, BF16, GQA de referencia correcta.

## HTTP y chat web

Se compiló el servidor en release y se probó en 127.0.0.1:18188,
sin interrumpir el servicio anterior en 127.0.0.1:8099.
El proceso temporal 18188 se cerró después.

- GET / y /nativo: HTTP 200, los dos chats HTML.
- GET /health, /ready, /v1/models y /v1/capabilities: HTTP 200.
- POST /v1/chat/completions ruta Terrsvusion y ruta densa:
  HTTP 200 y generación real de un token sobre MiniCPM5.
- Solicitar stream:true: HTTP 422, funcionalidad pendiente.
- Solicitar un modelo inexistente: HTTP 404.
- Las dos interfaces utilizan plantillas de conversación distintas.
  Las respuestas obtenidas no demuestran paridad entre ellas.

## Paridad numérica verificada

Comando real:

~~~bash
cargo run --locked --offline --release --features torch-backend \
  --bin terrsvusion-native-parity -- \
  --model /home/yo/models/MiniCPM5-1B --max-tokens 2
~~~

Con exactamente el mismo prompt_ids de Hola (15 tokens), mismos pesos,
mismo tokenizer y greedy, MiniCPM BF16 denso y Terrsvusion produjeron:

- 2 tokens idénticos: [61973, 61], texto "¡H".
- Máxima diferencia absoluta de logits = 0.0.
- Sin primer paso de divergencia.
- SHA-256 del estado KV lógico idéntico:
  3eb97b98bb68b88bf47c7307edba05910e79c2f34a48a979433de4b8cede74c4

Este caso prueba ausencia de desviación en la entrada medida, no
compatibilidad universal, velocidad GPU ni paridad en todos los modelos.

## Límites pendientes

- Inferencia end-to-end de MiniCPM u otros LLM totalmente en Rust
  sin LibTorch: NO implementada todavía.
- Kernels CUDA/ROCm/Metal/Intel XPU/Vulkan: pendientes.
- Soporte de pesos FP8/INT4/GGUF y otros LLM end-to-end: pendiente.
- Streaming SSE, tools, JSON Schema y autenticación API: pendiente.
- Contextos 32K, 100K, 131K certificados en todo hardware: pendiente.
- Compresión física 4 bits sin pérdida para pesos arbitrarios: no
  está demostrada ni es garantizable sin redundancia suficiente.
- Cotas de aceleración y reducción de RSS/VRAM universales:
  no demostradas.

No afirmar que una interfaz Rust, inspección de metadatos o enumeración
de hardware implique ejecución numérica real en dicho dispositivo.



## Extensión: iGPU AMD Phoenix real con OpenCL

Se instaló como dependencia Rust optativa opencl3 0.12.4 para construir
un backend que no requiere ni invoca PyTorch ni LibTorch. El shader está
en assets/opencl_attention.cl; el host en src/opencl_backend.rs.
El benchmark se encuentra en src/bin/terrsvusion-igpu-bench.rs.

Hardware: PCI 1002:15bf AMD Phoenix, driver amdgpu, OpenCL Rusticl,
CL_DEVICE_TYPE_GPU, 12 unidades de cómputo. Cómputo QK, softmax y PV
real en iGPU; original BF16/FP16/FP32 conservado en bytes crudos.

Cinco pruebas OpenCL reales correctas: BF16/FP16/FP32 GQA y FP32 MHA/MQA.
La referencia CPU Rust y la iGPU discreparon como máximo por 3,54e-8
en las corridas de benchmark. Datos QKV sintéticos deterministas.

Como comprobación independiente, durante 20 rondas BF16 de 32768 tokens,
el driver amdgpu informó de una mediana de uso del 74 % y un máximo
del 86 % (100 muestras). Son cargas reales del dispositivo, no CPU
simulada ni ahorro de VRAM del modelo.

Resultados y reproducibilidad: docs/IGPU_AMD_PHOENIX_20261029.md,
docs/IGPU_OPENCL_MEDICIONES_20261029.json,
docs/IGPU_OPENCL_MEDICIONES_FASTPATH_20261029.json
y docs/IGPU_AMDGPU_BUSY_20261029.json.

No existe todavía inferencia MiniCPM end-to-end en GPU ni backends
CUDA/ROCm/Vulkan/Metal validados. La ruta CPU BF16 original se mantiene.


## Extensión posterior: inferencia LLM completa Rust/OpenCL

Dentro de la MISMA carpeta Terravision V3 se añadieron
src/opencl_llm.rs, assets/opencl_llm.cl,
src/bin/terrsvusion-opencl-llm.rs,
src/bin/terrsvusion-opencl-server.rs,
assets/chat-gpu.html y
src/bin/terrsvusion-compare-opencl.rs.

La GPU AMD Phoenix ejecutó los 24 bloques MiniCPM5-1B BF16,
incluidos embedding, RMSNorm, RoPE, proyecciones, atención, SwiGLU,
MLP, logits y argmax. La ruta es Rust y OpenCL sin LibTorch.
Generó el mismo par de tokens [61973,61] que el control BF16
CPU con el mismo prompt Hola. En esa comparación CPU/GPU se midió
`L∞` de logits de 0,5 y SHA-256 KV diferente. No se observó igualdad
binaria entre backends, que **no es un requisito** para OpenCL:
la conservación de los pesos BF16 se verificó por separado y la
fidelidad LOD se evalúa contra la ruta GPU densa.

Se verificaron TODOS los 2.161.265.664 bytes de los 219 tensores
BF16 subidos a GPU por readback, con 0 discrepancias.
SHA-256 agregado original y readback:
1fb1632db083ec58baad8d7870b8bcfa4d593af112a903e6c14c2d231c7c2b5d

Nuevo chat GPU en 127.0.0.1:8188, separado del servicio original
CPU del puerto 8099. Se probó HTTP 200 en las rutas, tres
consultas consecutivas, sanitización de las características API
no implementadas y reset KV independiente por sesión.

El registro exhaustivo, comandos, evidencia de precisión,
datos del driver amdgpu y limitaciones está en
docs/IGPU_FULL_LLM_RUST_OPENCL_20261029.md.


## Validación adicional: LOD CPU reutilizado en GPU y selector

- GPU OpenCL reutiliza realmente los módulos canónicos
  adaptive_extent::ParallelContextAxis, cold_store::ColdStore,
  attention_certificate::{select_by_exact_mass, certified}.
- GPU LOD epsilon=0: prueba física AMD Phoenix comparó GPU densa vs
  GPU con ColdStore durante 16 pasos greedy de «Hola»; todos los
  tokens y logits BF16 de ambos caminos GPU fueron idénticos bit a bit.
  La CPU LibTorch no es un oráculo de identidad binaria para OpenCL;
  esas diferencias entre dispositivos se conservan como diagnóstico.
- GPU KV asignado para capacidad 128 tokens con stage8:
  3.145.728 bytes denso vs 1.585.248 bytes GPU LOD.
  Ahorro de capacidad KV 49,6 %, NO VRAM global.
- GPU LOD epsilon=0.001: prueba real de entrada pequeña, sin omisiones
  V; certificados satisfechos para todas las consultas observadas.
- Chat nuevo Rust CPU/iGPU/Auto operativo en puerto 8288:
  7 tests de selector correctos, pruebas HTTP reales de los tres modos
  y fallback de Auto con CPU temporalmente inaccesible.
- Ambos servidores de backend siguen independientes; CPU 8099,
  GPU 8188. Las configuraciones y los pesos permanecen intactos.
- FALTA en GPU el branch-and-bound regional de CPU y la reutilización
  persistente de prefijos; documentados sin sobreafirmar compatibilidad.
- Véase docs/LOD_CPU_COMPARTIDO_GPU_20261029.md.
