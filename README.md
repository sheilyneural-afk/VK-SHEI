# VK-SHEI / Terrsvusion

Motor experimental de inferencia LLM en Rust (crate `terrsvusion`, edición 2024)
para contexto largo. Conserva los pesos y el K/V autoritativo en BF16, guarda el
K/V en un almacén frío de solo anexado con sumas de verificación y añade un
índice paralelo de 4 bits que solo sirve para localizar regiones: nunca
sustituye a los valores originales.

Hay dos motores independientes:

- **CPU con LibTorch** (`torch-backend`): MiniCPM5-1B y Llama 3.1 8B Instruct
  en BF16 a través de `tch`.
- **GPU con OpenCL** (`opencl-backend`): implementación propia en Rust y OpenCL,
  sin LibTorch. Validada sobre todo con MiniCPM5-1B en una iGPU AMD Radeon
  Phoenix (gfx1103).

CPU y GPU no tienen que dar resultados idénticos bit a bit. Los criterios de
validación están en [docs/CRITERIOS_VALIDACION_CPU_GPU.md](docs/CRITERIOS_VALIDACION_CPU_GPU.md).

## Estado

Proyecto experimental, sin autenticación y pensado para `127.0.0.1`. Todos los
servidores escuchan en local por defecto. El servidor OpenCL exige
`--allow-network` para escuchar fuera de la máquina.

### Demostrado

La evidencia de cada punto está en [docs/](docs/README.md).

- **Integridad de pesos en GPU:** los 2.161.265.664 bytes BF16 de MiniCPM5-1B
  (219 tensores) se releyeron de la GPU y coinciden con el checkpoint (mismo
  SHA-256).
- **LOD con epsilon=0 en GPU:** no omite nada. Todas las consultas pasan por
  el respaldo exacto sobre el K/V BF16 de la ColdStore.
- **Optimizaciones exactas de GPU:** prefill por lotes, atención fusionada,
  repetición verificada de prompt y prefijo compartido verificado. En las
  pruebas físicas registradas (MiniCPM5-1B, 375 + 16 tokens), los logits y el
  K/V coinciden con los de la referencia GPU.
- **ColdStore:** formato de solo anexado con pie de commit, SHA-256 por registro
  y sumas por componente. Desde `98d52aa` recupera las escrituras a medias y abre en
  solo lectura aunque la cola esté rota, con tests de regresión.
- **Llama 3.1 8B en CPU:** servicio denso exacto, prefijos de documento
  persistentes reutilizables sin copiar bytes y lectura parcial de V desde la
  ColdStore, bit a bit exacta en un caso sintético controlado.
- **Grafo documental:** autoridad por página con SHA-256, micrometadatos para
  enrutar y un endpoint que reutiliza el prefijo K/V del documento completo.

### No demostrado todavía

- **Omisión certificada positiva con el modelo real.** En la última ejecución
  completa con Llama 3.1 8B (30/10/2026) se aceptaron 0 omisiones. Esa medición
  es anterior a `ee6cf6a` y `7f4fe0f`, que cambian el certificador (celda BF16 centrada en el
  candidato FP32 y cota de V por token), y no se ha repetido.
- **Contexto largo físico:** no hay paridad física a 30K, 100K ni 131K tokens,
  ni prefill completo de El Faro (unos 112K tokens) con Llama.
- **Rendimiento GPU frente a CPU:** en la comparación HTTP registrada
  (MiniCPM5-1B, 375 + 16 tokens, aceleraciones GPU activadas), la GPU tardó
  unas 2,15 veces más que la CPU densa.
- **Llama en GPU:** OpenCL carga Llama 3.1 8B (cuatro shards, RoPE `llama3`) y
  hace inferencias mínimas, más lentas que la CPU. No hay servicio ni
  validación larga, y `terrsvusion-opencl-server` solo admite MiniCPM5.
- **Documentos:** no hay extracción de PDF, EPUB, DOCX ni OCR, ni garantía de
  exhaustividad semántica.
- **Carga sostenida:** no hay pruebas de carga ni de concurrencia sostenida.
- **Memoria:** los máximos de RSS de unos 30 GiB con Llama se midieron antes
  de `74a553e`, que libera las páginas del mmap al copiar cada peso, y de
  `b51e3b2`, que añade el K/V residente. No se han vuelto a medir.

## Compilación

Necesitas Rust estable con soporte para la edición 2024 (comprobado con
rustc 1.99.1).

```bash
cargo build --release                                # núcleo sin LibTorch ni GPU
cargo build --release --features opencl-backend      # motor OpenCL (necesita un ICD OpenCL)
cargo build --release --features unified-router      # selector CPU/GPU por HTTP
cargo build --release --features hip-backend         # puente HIP opcional
```

Para `torch-backend` hace falta LibTorch. Una forma es usar el de una
instalación de PyTorch:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1   # si la versión de PyTorch no es la que espera tch 0.18
export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
cargo build --release --features torch-backend
```

| Feature | Activa |
|---|---|
| (ninguna) | núcleo: ColdStore, índice/eje adaptativo, certificado, grafo documental, almacén portable |
| `torch-backend` | motor CPU LibTorch y sus binarios |
| `opencl-backend` | motor OpenCL completo y sus binarios |
| `unified-router` | `terrsvusion-unified-server` |
| `hip-backend` | incluye `opencl-backend` y añade `native_hip`, que carga en tiempo de ejecución `native/libterrsvusion_hipblaslt_bridge.so` (fuente en `native/hipblaslt_bridge.cpp`) |

## Pruebas

```bash
cargo test --lib                              # 79 tests
cargo test --lib --features torch-backend     # 109 tests
```

Las pruebas físicas de GPU (`gpu_actual_*`) y las que necesitan checkpoints
reales están marcadas `#[ignore]`. Usan `TERRSVUSION_OPENCL_DEVICE`,
`TERRSVUSION_TEST_MODEL` y `TERRSVUSION_LLAMA_MODEL`.

## Binarios

| Binario | Feature | Uso |
|---|---|---|
| `terrsvusion-inspect` | — | inspecciona `config.json`, la geometría y las capacidades compiladas (`--model`, `--self-test`) |
| `terrsvusion-document` | — | construye y consulta un grafo documental (`build`, `query`) |
| `terrsvusion-server` | `torch-backend` | servidor HTTP CPU, puerto 8088 por defecto; UI en `/` y `/nativo` |
| `terrsvusion-long-context` | `torch-backend` | pruebas de contexto largo: denso, oráculos, métricas de sombra, prefijos |
| `terrsvusion-native-parity` | `torch-backend` | compara la ruta Terrsvusion con la densa BF16 sobre los mismos IDs |
| `terrsvusion-torch-chat` | `torch-backend` | generación mínima por línea de comandos |
| `terrsvusion-document-ask` | `torch-backend` | preguntas a un grafo documental por fases (CLI experimental) |
| `terrsvusion-opencl-llm` | `opencl-backend` | inferencia completa OpenCL por línea de comandos |
| `terrsvusion-opencl-server` | `opencl-backend` | servidor HTTP GPU (MiniCPM5), puerto 8188 por defecto |
| `terrsvusion-compare-opencl` | `opencl-backend` + `torch-backend` | diagnóstico OpenCL frente a LibTorch |
| `terrsvusion-igpu-bench` | `opencl-backend` | benchmark de atención aislada |
| `terrsvusion-unified-server` | `unified-router` | selector CPU/GPU/Auto, puerto 8288 por defecto |

El selector busca la CPU en `http://127.0.0.1:8099` y la GPU en
`http://127.0.0.1:8188`. Arranca el servidor CPU con `--port 8099` o pasa
`--cpu-url`.

## Estructura

```text
src/                  biblioteca y binarios (src/bin/)
assets/               kernels OpenCL, UIs HTML incluidas en los servidores,
                      fixtures documentales y el corpus El Faro
native/               puente HIP/hipBLASLt (fuente y biblioteca compilada)
scripts/              auditoría de tokenización de El Faro
docs/                 documentación y evidencia (índice en docs/README.md)
```

## Documentación

El índice está en [docs/README.md](docs/README.md):

- [Arquitectura](docs/ARQUITECTURA.md): módulos, almacenamiento, certificado e
  índice.
- [Motor CPU (LibTorch)](docs/CPU_LIBTORCH.md): MiniCPM5, Llama 3.1 8B,
  servidor, prefijos y estado de la omisión certificada.
- [Motor GPU (OpenCL)](docs/GPU_OPENCL.md): opciones, valores por defecto y
  mediciones.
- [Documentos y El Faro](docs/DOCUMENTOS.md): grafo documental, endpoint y
  auditoría de tokenización.
- [Criterios de validación](docs/CRITERIOS_VALIDACION_CPU_GPU.md).
