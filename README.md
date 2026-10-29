# VK-SHEI / Terrsvusion V3

Runtime experimental completo en Rust para memoria KV persistente, análisis de
contexto largo, almacenamiento frío verificable y ejecución CPU/iGPU AMD. Este
README describe el proyecto que está en este repositorio; no es un resumen de
una sola prueba ni una promesa de soporte universal para cualquier LLM.

## Qué contiene el proyecto

```text
.
├── Cargo.toml                         configuración, features y binarios
├── src/                               runtime Rust y backends
│   ├── cold_store.rs                  KV append-only, mmap, hashes y auditoría
│   ├── adaptive_extent.rs             índice dimensional, LSH y metadatos
│   ├── attention_certificate.rs       cotas y selección certificada
│   ├── conversion_worker.rs           conversión acotada y backpressure
│   ├── inference_worker.rs            ejecución serializada y aislamiento
│   ├── torch_runtime.rs               ruta MiniCPM/LibTorch y runtime largo
│   ├── prefill_tch.rs                 prefill con tch
│   ├── portable.rs                    primitivas nativas sin LibTorch
│   ├── portable_store.rs              almacenamiento portátil
│   ├── opencl_backend.rs              backend OpenCL
│   ├── opencl_llm.rs                  ejecución LLM completa experimental
│   ├── gpu_canonical_lod.rs           LOD canónico en GPU
│   ├── gpu_micro_regions.rs            microrregiones GPU
│   ├── native_hip.rs                  puente HIP/hipBLASLt experimental
│   └── bin/                            servidores, chats, inspectores y benches
├── native/                            código nativo auxiliar
├── assets/                            corpus, interfaces web y kernels OpenCL
├── docs/                              auditorías, comparativas y evidencias
└── scripts/                           herramientas reproducibles de auditoría
```

## Capacidades actuales

- Almacén KV frío append-only con reapertura, lectura selectiva, mmap,
  comprobación de integridad, detección de corrupción y recuperación tras
  reinicio.
- Índice dimensional compacto de cuatro bits y LSH por cabeza KV para routing;
  el índice no sustituye a los K/V BF16 autoritativos.
- Prefijos persistentes, sellado de regiones, rollback de colas no selladas y
  workers con backpressure.
- Ruta CPU BF16 con MiniCPM5-1B mediante `tch`/LibTorch.
- Ruta GPU OpenCL experimental para iGPU AMD Phoenix, con atención BF16,
  prefill por lotes, LOD compartido y servidores HTTP separados.
- Selector HTTP unificado CPU/iGPU/Auto.
- Auditorías físicas de paridad de logits, KV, hashes, memoria y tiempos en
  los informes de `docs/`.
- Herramientas para inspeccionar modelos, comparar backends y reproducir
  benchmarks.

## Estado y límites importantes

Terrsvusion resuelve principalmente el problema de residencia y persistencia
del KV. No convierte por sí solo una secuencia de millones de tokens en una
ventana nativa equivalente para Llama 3.1, ni elimina automáticamente los
límites de RoPE, posiciones o atención del modelo.

La selección certificada es conservadora: en varias pruebas reales el
certificador recupera todos los estados. Los resultados de GPU, los ahorros de
memoria y las aceleraciones son específicos de la máquina, modelo, prompt y
configuración indicados en cada informe. El backend HIP es experimental y no
es la ruta predeterminada.

### No demostrado todavía

- `hierarchy_k_tokens_avoided > 0` en una inferencia real del modelo.
- Menor latencia general que BF16 denso en todas las geometrías.
- Reducción del coste cuadrático del *primer* prefill. El *hit* sí evita ese
  Transformer y ya no copia el backing, pero eso no demuestra una reducción
  general del primer prefill.
- Paridad universal o norma `L∞` de logits igual a cero cuando hay omisiones a
  `1e-3`. La medición disponible fue `0,171875`, con IDs idénticos, en la
  prueba `256/128/2`.
- Seguridad de promover `1e-3` al servidor. La autoridad de servicio sigue en
  `1e-6`.
- Certificación de 30.000, 100.000 o 131.072 tokens.
- Métricas completas de RSS, workspace interno de LibTorch o VRAM.
- Rendimiento, RSS total y escalado bajo carga de varias inferencias
  concurrentes sobre los pesos LibTorch compartidos.

## Compilación

La funcionalidad nativa Rust no requiere PyTorch:

```bash
cargo check
cargo test --lib
```

Para la ruta OpenCL:

```bash
cargo check --features opencl-backend
cargo build --release --features opencl-backend
```

Para la ruta MiniCPM/LibTorch:

```bash
LIBTORCH_USE_PYTORCH=1 cargo test --all-targets --features torch-backend
```

En una instalación Python de PyTorch puede ser necesario añadir su directorio
`torch/lib` a `LD_LIBRARY_PATH` al ejecutar los binarios.

## Binarios principales

```text
terrsvusion-inspect          inspección de modelos y formatos
terrsvusion-server           servidor MiniCPM/LibTorch
terrsvusion-long-context     benchmark y flujo de contexto largo
terrsvusion-torch-chat       chat de la ruta tch
terrsvusion-opencl-llm       LLM completo experimental en OpenCL
terrsvusion-opencl-server    servidor OpenCL
terrsvusion-unified-server   multiplexor CPU/iGPU/Auto
terrsvusion-igpu-bench       benchmarks de iGPU
terrsvusion-compare-opencl   comparación OpenCL/LibTorch
terrsvusion-native-parity    comprobaciones de paridad nativa
```

Ejemplos:

```bash
cargo run --release --bin terrsvusion-inspect -- modelo.safetensors
cargo run --release --features opencl-backend --bin terrsvusion-igpu-bench
cargo run --release --features torch-backend --bin terrsvusion-long-context -- --help
cargo run --release --features unified-router --bin terrsvusion-unified-server
```

## Cómo leer la evidencia

Los resultados no están agregados como una cifra única de “rendimiento”. Cada
afirmación se debe leer junto con su informe en `docs/`:

- `VERIFICACION_20261029.md`: estado general y comandos.
- `ARQUITECTURA_PORTABLE_RUST.md`: ruta sin LibTorch.
- `LOD_CPU_COMPARTIDO_GPU_20261029.md`: relación entre CPU, iGPU, LOD y
  selector.
- `PREFILL_GPU_OPTIMIZADO_20261029.md`: prefill OpenCL por lotes.
- `GPU_ATENCION_FUSIONADA_EXACTA_20261029.md`: atención GPU exacta.
- `MICROMETADATOS_REGIONALES_GPU_20261029.md`: micrometadatos y cotas.
- `AUDITORIA_APROVECHAMIENTO_IGPU_20261029.md`: telemetría y límites de la
  iGPU.
- `CORRECCION_LLAMA_V_CERTIFICADA_IDENTIFICADORES_FARO_20261029.md`: límites
  de la evidencia Llama/Faro.

Los logs y JSON de `docs/` son artefactos de medición, no afirmaciones
independientes de producción. La ruta CPU, la ruta GPU densa, la ruta GPU LOD,
el backend OpenCL completo y la integración opcional HIP deben mantenerse
separados al interpretar los resultados.

## Interfaces locales

Cuando se ejecutan los servidores correspondientes, las interfaces históricas
son:

```text
CPU original:       http://127.0.0.1:8099/
iGPU OpenCL:        http://127.0.0.1:8188/
selector unificado: http://127.0.0.1:8288/
```

Los puertos, modelos, rutas de pesos y opciones exactas se consultan con `--help`
en cada binario. No se deben exponer estos servidores directamente a Internet
sin autenticación, límites y aislamiento adicionales.

## Modelo de memoria

```text
entrada / prefill
        ↓
KV reciente y temporales acotados
        ↓
sellado por regiones
        ↓
ColdStore append-only + hash
        ↓
índice 4-bit/LSH y metadatos
        ↓
recuperación exacta o selección certificada conservadora
```

El índice sirve para localizar candidatos. La autoridad numérica continúa en
los K/V almacenados y verificados. Los resúmenes o micrometadatos no sustituyen
la evidencia original.

## Desarrollo y contribuciones

Antes de cambiar una ruta, identificar si afecta a CPU, GPU, OpenCL, LibTorch,
almacenamiento o al router. Ejecutar las pruebas de la feature correspondiente,
registrar el hardware, el modelo, el prompt, los flags y los hashes, y añadir la
evidencia a `docs/` sin mezclar resultados de backends diferentes.

La versión publicada debe conservar la distinción entre:

- almacenamiento de millones de tokens;
- ventana de posiciones del modelo;
- recuperación selectiva;
- atención densa;
- atención aproximada certificada;
- paridad numérica;
- rendimiento medido en una máquina concreta.


## Corrección de evidencia Llama 8B: KV residente, V certificada y Faro

El aproximadamente 99,4 % de ahorro de KV residente (~5,2–5,4 MiB frente
a ~894–902 MiB en Llama 8B) procede de la arquitectura ColdStore:
NO significa omisión certificada de V. El antiguo Llama 32Q/8KV ni
siquiera ejecutaba el certificado exact_mass, exclusivo de MiniCPM
16Q/2KV. La ruta 32Q/8KV nueva es experimental y no está expuesta actualmente
por la API HTTP: `applied_mass_tolerance` se rechaza hasta demostrar paridad
final de logits y fallback exacto. El diagnóstico interno requiere un valor
positivo y falla a atención BF16 completa cuando no hay prueba. La prueba física de datos controlados
evitó 15 V de 17 (30.720 bytes de V fría) sin error observado,
pero por sí sola no prueba el checkpoint completo. La ejecución real posterior
de Llama 3.1 8B sí obtuvo poda QK/PV y 602 V omitidas en eventos que llegaron a
la salida. El prefill selectivo reutiliza ahora el orden layer-major por bloques
del prefill funcional, en lugar de repetir todo el Transformer token a token:
128 tokens con bloques de 32 pasan por cuatro llamadas de prefill. La medición
aplicada bajó primero de 124,74 s a 23,15 s. Al agrupar las lecturas regionales
por página física, ordenar las coordenadas para `ColdStore` y reconstruir luego
el orden best-first, bajó a 6,03 s; las lecturas parciales K/V pasaron de
66.850/62.226 a 11.327/767. La canónica de control tardó 2,51 s, así que la
ruta selectiva continúa siendo 2,40 veces más lenta y fue rechazada por
`L∞=0,125`. Por tanto, esto demuestra omisión real, eliminación de pasadas
redundantes y reducción de I/O regional, no paridad ni disponibilidad de
servicio.

La traza BF16 por capa sitúa la primera divergencia en la capa 1 del bloque que
termina en la posición 96 (`L∞=0,0009765625`); el máximo dentro de las capas
crece hasta `0,5`. Una prueba separada confirma que SDPA causal batida y
querywise con K/V completo son bit a bit idénticas, por lo que la diferencia
procede de la omisión y su propagación, no de dividir el bloque. La aceptación
experimental exige ahora logits exactamente iguales (`L∞=0`) y omisión
positiva; coincidencia de tokens o error dentro de epsilon se informan por
separado y no autorizan servicio.

El servidor Llama anuncia llama-dense-bf16; la petición
minicpm-dense-bf16 devuelve HTTP 404. document= con nativo virgen
devuelve 422 y NO constituye la misma plantilla de chat.

Faro corto /tmp/faro-full.txt: 27.168 bytes y 7.065 tokens raw
Llama. Volumen completo assets/el-faro-volumen-completo.txt:
368.968 bytes y 111.632 tokens raw Llama (112.104 con MiniCPM).
El volumen completo NO se ha inferido ni validado de extremo a extremo.
La entrada previa de 7.143 tokens es un prompt diferente.

Evidencia y comandos: docs/CORRECCION_LLAMA_V_CERTIFICADA_IDENTIFICADORES_FARO_20261029.md
y scripts/audit_llama_faro_inputs.py.

---


## Primer prefill exacto: GPU batch384 verificado (29/10/2026)

Se ha comprobado físicamente prefill causal BF16 de hasta 512 tokens por lote, sujeto a un límite de 64 MiB de scratch QK. Un prompt de 375 tokens usa UNA pasada con --gpu-prefill-batch 384, frente a TRES con batch128. Todos los logits del modelo y SHA lógico del KV/ColdStore originales coinciden en los tamaños 64, 96, 128, 192, 256, 375 y 512. La mejora observada de prefill fue 1,55× en una prueba con CPU fuertemente contendido; no se presenta como rendimiento universal. En HTTP real, primera petición 375+16: 3.854 s total, 2.313 s prefill; replay idéntico, SHA256 validado, 1.698 s total. Las fusiones opcionales QKV y SwiGLU para prefill fueron probadas y descartadas como configuración predeterminada por ausencia de mejora global. CPU, original BF16 y la caché fría están preservados.

Informe: docs/GPU_PREFILL_UNICO_LOTE_BF16_20261029.md

---

## GPU BF16: atención exacta fusionada (29/10/2026)

La iGPU Radeon puede calcular QK BF16, softmax FP32 y PV BF16
en una única operación OpenCL sin omitir ninguna K/V. El núcleo
fue validado bitwise para contextos 1–2048 y dentro del MiniCPM5-1B
completo con el mismo SHA de KV/ColdStore. En 375+16 tokens,
el decode bajó de ~2,13 a ~1,74 s y la inferencia total de
~4,70 a ~4,43 s bajo carga del sistema actual. Se conserva
la caché fría BF16 y el índice 4-bit original. Disponible con
--gpu-exact-fused-attention; rollback --gpu-exact-original-attention.
El servidor GPU 8188 lo utiliza, CPU y router intactos.

Informe: docs/GPU_ATENCION_FUSIONADA_EXACTA_20261029.md

---

## V BF16 e índices GPU calientes con micrometadatos (29/10/2026)

GPU original: BF16 exacta + índice/ColdStore canónicos de CPU. La nueva ruta opcional epsilon>0 mantiene una copia V BF16 y la selección certificada residente en GPU, con fallback exacto. En atención física 32 tokens: 0,340 a 0,151 ms (2,25×); frente a micro frío 0,850 a 0,151 ms (5,6×). En MiniCPM5 375+16, epsilon=0,001: 5,07 a 3,87 s, cero lecturas V frías durante decode. El certificado no pudo omitir K en ese contexto real; el modo epsilon=0 permanece por defecto.

Informe: docs/GPU_MICROMETADATOS_V_HOT_OPTIMIZADOS_20261029.md

---

## Micrometadatos de regiones y certificados GPU (29/10/2026)

Los **micrometadatos canónicos** DimensionalMicroMetadata ya no son un
sello 4-bit y un grafo adyacente genérico: cada extensión lleva coherencia,
calidad de enlace medida y predecesor temporal separado de los vecinos
semánticos. El índice 4-bit/LSH va por cabeza KV. En GPU las cotas usan
teselas de 8 tokens y el mínimo entre caja AABB y esfera alrededor de la
media BF16. Una omisión nunca depende solo de una firma de cuatro bits.

En una prueba física separable de 32 posiciones, la Radeon calculó
32 Q·K menos y recuperó 16 V en vez de 32, con error de salida
dentro de epsilon 0,001. En MiniCPM5-1B con 375 tokens el límite
no pudo omitir ninguna región K, incluso ajustando la sensibilidad
del eje a 26–28 regiones; **no se ha demostrado mejor velocidad
en esa prueba real de modelo**. El modo exacto (epsilon=0) no
utiliza el índice auxiliar ni descarta valores.

La configuración CPU y el ejecutor original están intactos.
Las modificaciones son propias del backend GPU y el ajuste de
sensibilidad solo se aplica si el usuario solicita explícitamente
--lod-epsilon positivo.

**Informe y pruebas:**
[Micrometadatos regionales GPU](docs/MICROMETADATOS_REGIONALES_GPU_20261029.md).

---

## Actualización: aprovechamiento GPU y ColdStore sin bloqueos (29/10/2026)

El AMD Phoenix está calculando BF16 real, pero todavía está por debajo
de la CPU densa: comparación HTTP actual 375+16, mediana GPU OpenCL
3,15 s frente a CPU BF16 densa 1,46 s. El nuevo sello ColdStore
diferido evita 48 lecturas bloqueantes de K/V por token de decode
y conserva la misma estructura original de hashes, BF16 y 4-bit.
Los kernels QKV/SwiGLU fusionados mantienen los logits exactos
de la GPU anterior. Ambas opciones son específicas de GPU;
los archivos CPU no han cambiado. hipBLASLt gfx1103 está
investigado como alternativa de alto rendimiento, pero la
integración HIP/OpenCL tuvo fallos intermitentes y NO está
habilitada en el chat.

Detalles, telemetría de actividad y siguiente arquitectura:
docs/AUDITORIA_APROVECHAMIENTO_IGPU_20261029.md

---

## Prefill GPU paralelo y ColdStore agrupado (29/10/2026)

Se ha incorporado un **prefill causal por lotes OpenCL** a la ruta AMD
iGPU, sin cambiar el ejecutor ni los ajustes de CPU. La GPU realiza
embedding, BF16 GEMV, RMSNorm, RoPE y atención causal por lotes de hasta
256 tokens. El servidor GPU usa inicialmente lotes de 128 tokens; la
ruta previa token a token continúa disponible como fallback.

La ruta GPU LOD, que utiliza el MISMO índice de 4 bits/LSH,
ColdStore BF16 y certificado de masa que CPU, **elimina lecturas
frías repetidas DURANTE el prefill**: V exacta permanece temporalmente
en GPU, se sella en páginas ColdStore de hasta 64 tokens al acabar
y entonces se libera la caché V transitoria.

**375 tokens de entrada / 16 de salida en Radeon Phoenix:**
prefill iGPU LOD de 51,16 s a **7,18 s (7,13×)**;
inferencia total de 54,08 a **9,68 s (5,58×)**.
Las rutas GPU densas también mejoraron: prefill 30,83 a 7,42 s.

**Paridad:** los mismos IDs y logits BF16 de la ruta GPU anterior.
Se ha probado un prompt de 375 tokens en tres lotes y se comprobó
la igualdad completa de logits y SHA lógico KV contra GPU densa.
CPU LibTorch todavía difiere numéricamente de GPU, aunque puede
generar los mismos tokens.

Detalles, límites y comandos: [Prefill GPU optimizado](docs/PREFILL_GPU_OPTIMIZADO_20261029.md).
El motor CPU original está intacto en el puerto 8099; selector
CPU/iGPU/Auto en 8288; GPU nueva en 8188.

---

## Estado actualizado: LOD original de CPU reutilizado en iGPU (29/10/2026)

**El proyecto ya tiene un único chat con selector CPU / iGPU / Auto**:

- En el navegador del propio PC: http://127.0.0.1:8288/
- Motor CPU original: http://127.0.0.1:8099/ (BF16, LOD completo,
  índice 4-bit, LSH, ColdStore, branch-and-bound y prefijos persistentes).
- Motor iGPU AMD: http://127.0.0.1:8188/ (Rust/OpenCL BF16 con
  los MISMOS módulos originales de CPU para índice de cuatro bits,
  LSH, ColdStore BF16 y certificado terminal select_by_exact_mass).

**Precisión:** GPU LOD con epsilon=0 y GPU densa produjeron los mismos
tokens y logits BF16 bit a bit en la prueba real de 16 pasos con «Hola».
Esto NO significa paridad bit a bit frente a LibTorch BF16 CPU: ambas
rutas GPU aún discrepan numéricamente de esa referencia.

**Limitación concreta:** la CPU conserva además el branch-and-bound
jerárquico de cotas por regiones y prefijos cacheables entre peticiones.
Esas fases todavía no están portadas a GPU: el motor GPU utiliza el
mismo filtro de masa exacta y conserva K BF16 para scoring GPU y V
BF16 verificadas en el ColdStore original. No hay cuantización
destructiva de pesos ni de KV.

**Memoria KV:** prueba a 128 tokens con staging GPU de 8 tokens:
3.145.728 bytes de capacidad KV densa frente a 1.585.248 bytes
de K más staging/auxiliares en GPU (49,6 % menos en KV contabilizado);
no implica 49,6 % menos de VRAM total y puede ser más lento por IO.

Informe actual con comandos, certificación, memoria y límites:
[CPU LOD compartido con GPU y selector](docs/LOD_CPU_COMPARTIDO_GPU_20261029.md).

La copia de seguridad anterior a esta intervención sigue intacta:
 /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-unified-20261029/

---

Estado verificado del motor MiniCPM5-1B a 28 de octubre de 2026;
los resultados históricos de esa fecha se conservan.

**Extensión de 29 de octubre de 2026:** núcleo Rust sin LibTorch,
almacenamiento KV exacto BF16/FP16/FP32 en RAM y SSD, inspector de modelos
Hugging Face, API OpenAI-compatible con lista de modelos y capacidades,
y preservación del backend MiniCPM5 BF16 detrás de la feature torch-backend.
Consulte [arquitectura portátil](docs/ARQUITECTURA_PORTABLE_RUST.md)
y [validación](docs/VERIFICACION_20261029.md). Guardar bytes FP16/FP32
e inspeccionar modelos NO significa poder ejecutar cualquier LLM ni GPU.


Terrsvusion es un runtime experimental escrito en Rust para ejecutar
MiniCPM5-1B con pesos y cálculo BF16, un catálogo dimensional compacto de
cuatro bits y un backing K/V BF16 append-only. Los cuatro bits solo sirven para
routing: no transforman los pesos, no sustituyen los K/V y no constituyen una
segunda autoridad numérica.

El proyecto demuestra actualmente reducción fuerte de la residencia estable
del KV contabilizado y recuperación exacta desde backing frío. Todavía no
demuestra aceleración general: con la tolerancia local vigente, el selector
termina recuperando todos los estados en las pruebas realizadas.

### Inferencia LLM completa en AMD iGPU — Rust/OpenCL (29/10/2026)

Además de las pruebas anteriores de atención GPU aislada, se ha implementado
un nuevo ejecutor experimental que ejecuta las 24 capas completas de
MiniCPM5-1B BF16 en la iGPU AMD Phoenix mediante Rust y OpenCL.
Embedding, proyecciones Q/K/V/O, RMSNorm, RoPE, atención causal,
SwiGLU/MLP, LM-head y argmax se calculan en GPU; la CPU solo gestiona
tokenización y lanzamiento de operaciones. Esta ruta NO usa PyTorch ni
LibTorch y no modifica los pesos BF16 originales.

**Se han recuperado de la GPU y comparado byte a byte los 219 tensores
BF16 completos: 2.161.265.664 bytes, cero diferencias y SHA-256 idéntico.**

En «Hola», la GPU y el control BF16 CPU generaron ambos los dos IDs
[61973, 61] con el mismo prompt y pesos, pero los logits y el KV
NO son idénticos bit a bit: diferencia máxima observada de logits 0,5
y KV lógico distinto. Más tarde se integraron en GPU los mismos
módulos de índice 4-bit/LSH, ColdStore BF16 y el certificado final
que usa CPU; la selección regional branch-and-bound de CPU permanece
pendiente en GPU. No se ha demostrado aceleración end-to-end frente a
CPU: en una prueba de 48 tokens la CPU densa fue más rápida.

Chat unificado CPU/iGPU/Auto: http://127.0.0.1:8288/
Servidor GPU independiente, solo local: http://127.0.0.1:8188/
Servidor CPU BF16 histórico: http://127.0.0.1:8099/

Guía de implementación, prueba de pesos, precisión y límites:
[Informe de LLM GPU](docs/IGPU_FULL_LLM_RUST_OPENCL_20261029.md).

Iniciar el chat GPU sin PyTorch:

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo run --release --locked \
  --features opencl-backend --bin terrsvusion-opencl-server -- \
  --model /home/yo/models/MiniCPM5-1B --host 127.0.0.1 --port 8188 \
  --max-context-tokens 2048 --resident --gpu-lod --lod-epsilon 0 --lod-stage 64
~~~

Iniciar el selector Rust CPU/iGPU/Auto sin PyTorch:

~~~bash
cargo run --release --locked --features unified-router \
  --bin terrsvusion-unified-server -- \
  --host 127.0.0.1 --port 8288 \
  --cpu-url http://127.0.0.1:8099 \
  --gpu-url http://127.0.0.1:8188 --auto-prefer cpu
~~~

Auditoría previa de atención GPU en diferentes formatos:
[Pruebas OpenCL](docs/IGPU_AMD_PHOENIX_20261029.md).
La compatibilidad FP16/FP32 se ha comprobado en almacenamiento y
atención de referencia; el ejecutor completo GPU sigue siendo BF16
hasta incorporar y validar nuevos kernels.

## Estado ejecutivo

| Propiedad | Estado actual |
| --- | --- |
| Runtime | Rust 2024; CPU BF16 LibTorch opcional y GPU OpenCL experimental sin LibTorch |
| Modelo probado | MiniCPM5-1B local |
| Pesos | Originales; no se genera un modelo Q4 |
| Cálculo del modelo | CPU BF16 histórico; GPU BF16 intermedios con acumulación FP32 (experimental) |
| Autoridad K/V recuperada | BF16 exacta |
| Índice auxiliar | Fingerprints firmados de cuatro bits + SimHash LSH |
| Almacenamiento | Archivo append-only por petición + prefijos persistentes verificados |
| Integridad | Checksums rápidos, hashes por token y SHA-256 por registro |
| Prefill | Atención causal BF16 completa |
| Decode | Routing dimensional + branch-and-bound + certificado de masa |
| Omisión certificada observada | 0 V a `1e-6`; 8 V a `1e-3` (256/128/2, ids iguales) |
| Reutilización entre peticiones | Prefijo estable de mensajes anteriores; comprobada tras reinicio |
| Servidor | Axum/Tokio; una autoridad de inferencia y cola acotada |
| Tests históricos del 28/10 | 52 correctos (véase nueva verificación) |

## Arquitectura canónica

El crate contiene una biblioteca y varios ejecutables para CPU y GPU:

- `terrsvusion-server`: servidor HTTP Axum (torch-backend).
- `terrsvusion-inspect`: inspector y referencia CPU Rust, sin Torch.
- `terrsvusion-native-parity`: comparación de tokens/logits/KV con BF16 denso (torch-backend).
- terrsvusion-opencl-llm: inferencia completa MiniCPM5 BF16 en GPU mediante Rust/OpenCL.
- terrsvusion-opencl-server: chat web GPU independiente y API local.
- terrsvusion-compare-opencl: auditoría de GPU frente al control BF16 CPU
  (compila las dos features solo para comparar).
- terrsvusion-unified-server: selector CPU/iGPU/Auto mediante proxy local Rust;
  no toca pesos ni configuraciones. Feature unified-router.
- `terrsvusion-torch-chat`: inferencia local desde línea de comandos.
- `terrsvusion-long-context`: benchmark, documentos largos, estrés y control
  BF16 denso opcional.

Módulos de la biblioteca:

- `torch_runtime.rs`: carga de `safetensors`, tokenizer, MiniCPM5, atención,
  caché Terrsvusion, control denso y métricas.
- `adaptive_extent.rs`: estados dimensionales definidos por contenido,
  fingerprints de cuatro bits, micrometadatos y routing LSH.
- `attention_certificate.rs`: cotas de masa omitida y selección exacta por
  scores QK.
- `cold_store.rs`: formato append-only, recuperación, verificaciones y lecturas
  parciales K/V.
- `conversion_worker.rs`: trabajadores acotados para preparar y ordenar commits
  BF16.
- `inference_worker.rs`: hilo propietario del modelo, cola acotada y
  backpressure.
- `prefill_tch.rs`: proyección protegida de cada cabeza Q hacia su cabeza KV
  servidora para el routing. No implementa por sí solo la atención de prefill.

El backend CPU BF16 anterior sigue disponible y mantiene su caché LOD.
El backend GPU experimental es un ejecutor alternativo independiente en Rust,
sin LibTorch. La ruta CPU recurre a LibTorch del entorno local, pero el motor
GPU nuevo no lo enlaza. Ninguno requiere un intérprete Python en ejecución.

### Flujo de servicio

```text
HTTP o CLI
  -> TorchModelRuntime
  -> MiniCpmModel
  -> TerrsvusionKvCache
  -> catálogo dimensional de cuatro bits
  -> branch-and-bound y certificado de masa
  -> lecturas parciales BF16 del backing
  -> atención SDPA de LibTorch
```

`generate_token_ids_dense` permanece como oráculo de benchmark. No es una ruta
del servidor.

## Estados dimensionales y marcos de almacenamiento

Un estado dimensional es una región lógica variable detectada a partir del K
que ya produce el modelo. Sus límites responden a sorpresa respecto al
centroide del estado abierto y al cambio temporal frente al token anterior.
Los intervalos token son direcciones reversibles hacia los K/V del modelo; no
son una segunda tokenización.

Los estados lógicos y los marcos físicos son independientes:

- un estado puede atravesar varios marcos;
- un marco puede contener fragmentos de varios estados;
- selección y certificación operan sobre estados;
- las lecturas finales operan sobre coordenadas token exactas.

`--storage-frame-tokens` controla únicamente la granularidad física de IO. El
runtime acepta potencias de dos entre 16 y 512; el valor normal es 512.

## Catálogo de cuatro bits y LSH

Cada token obtiene un fingerprint compacto **por cabeza KV**, con dígitos
firmados `[-7, 7]`, dos coordenadas por byte y una escala local por cabeza.
No se mezclan cabezas ni se colapsa el K completo a 16 medias: cada cabeza
mantiene hasta 32 coordenadas. Este catálogo es metadata de búsqueda; los
K/V recuperados continúan siendo BF16.

El índice LSH es independiente por cabeza y usa cuatro tablas con 16 bits
**solo direccionales**. Las bandas de magnitud ya no se XOR-ean en la clave
del cubo, porque Q y K no comparten escala y eso convertía el índice en
ruido. Una consulta prueba vecindad Hamming 2 y después 4, ignora cabezas
sin energía (como las Q protegidas) y exige votos de al menos dos tablas.

LSH no es una autoridad de recall. Si no devuelve candidatos se incrementa
`parallel_lsh_misses` y se entregan cero semillas. El branch-and-bound canónico
continúa desde el estado reciente y amplía solo cuando su cota lo exige. No
existe un fallback que cargue el catálogo completo.

Los fingerprints por token, las firmas por extensión y el grafo de
micrometadatos son señales auxiliares que se solapan parcialmente. Su contrato
actual es único: priorizan candidatos, pero no autorizan por sí mismos una
carga u omisión. El grafo comunicante solo admite vecinos con similitud
coseno ≥ 0,80 y coherencia de extensión suficiente; un vecino temporal
ortogonal no se registra como semántica. `link_quality` y `coherence` miden
esa calidad. Las dimensiones comunicantes no se añaden todas al conjunto
inicial; solo desempatan el orden de ampliación.

## Certificado de atención

Durante decode, Terrsvusion:

1. Siembra estados candidatos mediante routing dimensional.
2. Conserva siempre el estado reciente.
3. Calcula cotas superiores QK para los estados omitidos.
4. Materializa un estado adicional cuando la cota no cierra.
5. Con scores exactos, selecciona el menor conjunto cuya cota de error por
   componente cumple la tolerancia.
6. Recupera los V seleccionados y ejecuta SDPA BF16.

La materialización K durante una atención es incremental: cada región abierta
se lee como máximo una vez del cold store para ese token, aunque el
branch-and-bound necesite varias rondas. Las rondas posteriores reutilizan el
tensor local ya verificado y lo liberan al terminar esa atención. Esto elimina
la antigua amplificación por relectura sin cambiar regiones, scores, tolerancia
ni orden causal.

Los scores exactos también se conservan por región durante esa llamada. Una
ampliación calcula únicamente los scores de la nueva región; la concatenación
tensorial final se realiza una vez. Si una pasada lineal demuestra que ni el
token individual más barato puede omitirse con la tolerancia restante, se
evitan las 16 ordenaciones por cabeza y se toma directamente el conjunto denso.
Los intervalos contiguos residentes usan vistas `narrow` y las páginas completas
usan `shallow_clone`; `index_select` queda reservado para coordenadas realmente
dispersas. En el cold store, `read_*_tokens` ya agrupa rangos contiguos por
cabeza y deriva automáticamente una selección `(0..N)` a la lectura completa.

La tolerancia actual está fijada en `1e-6`. La cota mantiene separadas la masa
softmax omitida y la magnitud `max|V|`. No existe todavía una demostración que
propague esa cota local a través de todas las capas hasta el margen de logits
final.

Las regiones K usan cajas alineadas a ejes. Son seguras, pero pueden construir
un máximo virtual combinando extremos de dimensiones pertenecientes a tokens
distintos. Esa laxitud geométrica sigue presente. Una prueba centro-radio fue
retirada porque no evitó estados y aumentó los metadatos; no permanece código
experimental de esa vía.

## Prefill y ciclo de vida K/V

El prefill continúa usando atención causal BF16 completa. Para cada capa se
mantiene temporalmente el K/V BF16 necesario mientras los marcos cerrados se
preparan en trabajadores Rust. Al terminar el prefill:

1. se sellan las colas parciales;
2. se drenan todos los trabajos aceptados;
3. se audita el backing mediante un mmap inmutable temporal;
4. se libera el conjunto BF16 temporal de prefill;
5. decode recupera fragmentos BF16 desde el backing.

Esto reduce la residencia estable posterior, pero no elimina el coste
cuadrático del primer prefill ni su working set temporal.

El servidor reutiliza ahora un prefijo estable entre peticiones. Cuando hay al
menos dos mensajes, todos salvo el último forman el prefijo cacheable; el
último mensaje es el sufijo o pregunta variable. La clave SHA-256 incorpora la
identidad criptográfica de pesos y tokenizer, los ids exactos del prefijo y la
geometría de almacenamiento. El manifiesto conserva certificados, páginas y
eje dimensional. La publicación usa staging y rename atómico. En un hit se
abre directamente el backing publicado como autoridad inmutable, se reconstruye
el catálogo y se verifican criptográficamente las páginas recuperadas antes de
procesar solo el sufijo. No se copia el archivo (`prefix_cache_copied_bytes=0`)
y el decode de una petición nunca modifica el prefijo persistido.

Esto elimina el cómputo Transformer repetido del prefijo, no toda su I/O ni la
rehidratación BF16. Tampoco acelera una petición de un solo mensaje, porque no
existe una frontera fiable entre documento y pregunta. Para reutilizar un
libro, debe enviarse como mensaje anterior (por ejemplo `system`) y la pregunta
como último mensaje `user`.

La preparación del backing sí está fuera del hilo de inferencia: cálculo de
hashes por token, checksums, SHA-256 y ensamblado completo del registro se hacen
en un worker acotado; el hilo propietario solo hace el append ordenado. Se usa
un único worker para evitar competir con los hilos de LibTorch durante el
prefill. Las métricas `prefill_*_nanos` separan eje dimensional, concatenación
caliente, certificados, handoff y commits.

La ruta canónica densa expone además dos contabilidades distintas, porque una
reducción de lecturas del `ColdStore` no prueba una reducción del Transformer:

- `dense_sdpa_scheduled_{qk,pv}_elements`: geometría rectangular entregada a
  LibTorch SDPA, incluida la zona tapada por la máscara causal explícita;
- `dense_sdpa_causal_{qk,pv}_elements`: pares de prefijo no enmascarados;
- `dense_sdpa_calls` y `dense_sdpa_nanos`: llamadas y tiempo medido alrededor
  de SDPA. Este tiempo no es un contador hardware y no revela los FLOP internos
  del kernel opaco.

Una prueba física Llama 3.1 8B BF16 de 128 tokens en dos bloques de 64 produjo
64 llamadas (2 bloques × 32 capas), 12.582.912 elementos QK programados y
8.454.144 causales, exactamente las fórmulas tensoriales. En 4.000 tokens, un
bloque único fue peor que ocho bloques de hasta 512: 111,458 s frente a
92,514 s de prefill y 500 MiB frente a 78 MiB de working set K/V conocido. La
causa es que el bloque único entrega un rectángulo causal 4.000×4.000 por capa;
menos llamadas no significa menos trabajo programado. La evidencia completa
está en `docs/EVIDENCIA_PREFILL_DENSO_4K_20261030.json`. Ninguna de estas
mediciones demuestra todavía un primer prefill subcuadrático.

La publicación del primer prefijo tampoco copia ya el backing cuando el
directorio temporal y la caché están en el mismo sistema de archivos: se crea
un enlace de staging, se publica el directorio por `rename`, se reabre el
backing como solo lectura y se crea un ColdStore distinto para el sufijo. Si el
enlace no está disponible, la copia entre sistemas de archivos es explícita y
sus bytes se publican en `prefix_cache_copied_bytes`; no se presenta como cero
copia. Una ejecución física Llama 3.1 8B con 807 tokens produjo
`copied_bytes=0` e `immutable=true` tanto en el miss como en el hit. El prefill
bajó de 17,574 s a 0,621 s en la segunda pregunta (28,31×). Esto elimina I/O de
publicación del primer documento, no el Transformer denso inicial. Evidencia:
`docs/EVIDENCIA_PUBLICACION_PREFIJO_SIN_COPIA_20261030.json`.

## Almacenamiento e integridad

`ColdStore` usa registros append-only con:

- geometría y offsets explícitos;
- checksums rápidos de K, V y registro;
- hash por token para validar lecturas parciales;
- digest SHA-256 persistido por registro;
- footer de commit;
- `page_id` monótono;
- recuperación de una cola final incompleta al reabrir.

La auditoría completa mapea temporalmente el archivo, verifica los registros en
paralelo y libera el mapa al terminar. La inferencia no conserva un mmap de K/V
como caché residente; usa lecturas parciales explícitas. Integridad de bytes no
equivale a relevancia semántica: esta última sigue dependiendo del routing y del
certificado.

## Oráculo exacto de omisibilidad

`--exact-oracle` activa además un diagnóstico cuando todas las regiones ya
fueron cargadas. `--compare-dense` solo ejecuta el control BF16 y no contamina
la comparación de latencia con este trabajo adicional. Con los scores QK
exactos, el oráculo calcula:

- oportunidades token/evento observadas;
- tokens omitibles a `1e-6`, `1e-5`, `1e-4` y `1e-3`;
- tolerancia mínima observada para omitir un token simultáneamente en todas las
  cabezas;
- margen mínimo top-1/top-2 de logits durante la generación.

El oráculo no gobierna la inferencia. Está desactivado en el servidor y en la
ruta normal porque ordenar scores a cuatro tolerancias es costoso. Si en el
futuro la geometría consigue omitir regiones, el oráculo deja de muestrear ese
evento en vez de volver a leer todo y destruir la aceleración.

Todavía no ejecuta una rama shadow con omisiones aplicadas, por lo que no mide
el error real de atención ni certifica paridad de tokens para tolerancias
alternativas. El margen de logits es observacional; no es una licencia para
relajar la cota.

## Instrumentación shadow de solo lectura

`--shadow-metrics` activa un informe separado de `TorchGeneration`. El módulo
`shadow_metrics.rs` solo agrega escalares; no posee tensores, no abre el cold
store, no selecciona estados y nunca autoriza omisiones. Las muestras se toman
únicamente cuando la ruta canónica ya cargó todas las regiones y dispone de los
scores exactos. Por eso el informe declara `additional_cold_reads: 0` y no se
activa en el servidor.

El informe contiene:

- distribución nearest-rank del umbral mínimo para omitir un token individual;
- holgura por región/cabeza entre la cota log-masa AABB y el máximo regional
  exacto más `ln(token_count)`;
- identidad del evento mínimo y máximo;
- cualquier violación donde AABB quede por debajo del score exacto;
- error local real de atención para selecciones exactas experimentales a
  `1e-4` y `1e-3`, comparado con `2 * alpha * max|V|`;
- limitaciones explícitas. Las omisiones solo se simulan en la salida local de
  atención: todavía no se propagan estados shadow hasta logits.

Ejecución:

```bash
cargo run --release --features torch-backend --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --book-tokens 512 \
  --chunk-size 128 \
  --storage-frame-tokens 128 \
  --generate-tokens 2 \
  --compare-dense \
  --shadow-metrics
```

Resultado real de esa geometría:

| Métrica shadow | Resultado |
| --- | ---: |
| Eventos de umbral individual | 24 |
| Umbral mínimo | `2,3360073602400125e-5` |
| Umbral p50 | `6,37478211878966e-5` |
| Umbral p90 | `1,2052335068237947e-4` |
| Umbral máximo | `2,0510393124897964e-4` |
| Comparaciones AABB exactas | 2.304 |
| Holgura log-masa AABB mínima | `0` |
| Holgura p50 | `18,555361760541164` |
| Holgura p90 | `27,091711965358204` |
| Holgura máxima | `42,49762832275392` |
| Violaciones AABB | 0 |
| Tokens omitidos a `1e-4` | 1 |
| Cabezas afectadas a `1e-4` | 16 de 384 |
| Error real máximo a `1e-4` | `1,0886421543357394e-5` |
| Ratio cota/error p50 a `1e-4` | `2,6748551764931374` |
| Violaciones de cota a `1e-4` | 0 |
| Tokens omitidos a `1e-3` | 21 |
| Cabezas afectadas a `1e-3` | 160 de 384 |
| Error real máximo a `1e-3` | `6,53646364726923e-5` |
| Ratio cota/error p50 a `1e-3` | `3,2957461035628026` |
| Violaciones de cota a `1e-3` | 0 |
| Lecturas frías canónicas | 240 |
| Lecturas frías shadow añadidas | 0 |
| Paridad de ids BF16 | `true` |

Esta corrida demuestra que las AABB son muy laxas, pero también que el umbral
individual exacto permanece muy por encima de `1e-6`. Mejorar solo la geometría
K no basta para crear una omisión certificable bajo la cota V actual.
La cota V sobrestimó el error local real por aproximadamente `2-5x` en las
cabezas con omisión de esta corrida. Eso justifica estudiar una cota V más fina,
pero no demuestra seguridad hasta logits ni autoriza cambiar `1e-6`.

Una línea base posterior con 1.024 tokens, ocho tokens solicitados, chunks y
marcos de 256 registró 168 eventos de capa shadow. El mínimo global de
`minimum_individual_tolerance` fue `2,783061069137239e-6`; por tanto, la
condición de bypass `1e-6 < minimum_individual_tolerance` se cumplió en el 100 %
de los eventos observados. La ruta O(N) entregó directamente el conjunto denso
sin ejecutar las 16 ordenaciones por evento. Se conservaron la respuesta,
paridad exacta, 816 ampliaciones y cero tokens evitados. El informe de referencia
se guardó durante la ejecución en `/tmp/terrsvusion-baseline-shadow-1024.json`;
es evidencia temporal local, no un artefacto versionado.

La ruta canónica posterior sustituyó el trabajo repetido de branch-and-bound
por un certificado regional incremental. Las prioridades se ordenan una vez;
cada región K se materializa y puntúa una vez; y las masas conservada y omitida
se actualizan en un marco exponencial fijo. La cota incluye además el máximo de
V de la cola caliente, que antes participaba en los scores conservados pero no
en ese máximo. Si una cota regional queda por debajo de la masa exacta observada,
la ejecución falla cerrada.

Una corrida real posterior, con la misma geometría de 1.024 tokens, ocho tokens
generados y comparación BF16 densa, produjo exactamente los mismos ocho ids y
la misma respuesta `AZULEJO-7319`. Terrsvusion conservó las 816 ampliaciones y
cero tokens omitidos, pero registró `1,269 s` de decode y `2,865 s` de latencia
total. Frente a la corrida anterior documentada (`~1,8 s` de decode y `3,389 s`
total), supone aproximadamente un 29,5 % y un 15,5 % menos, respectivamente.
Son corridas individuales, no una distribución estadística; demuestran que el
trabajo retirado era real, no una aceleración universal ni omisión certificada.

La instrumentación interna de una segunda corrida comparable atribuyó los
`1,217 s` de decode a `168 ms` en SDPA/ensamblado interno, `132 ms` en
materialización K, `123 ms` en scores exactos, `93 ms` en materialización V,
`37 ms` en metadatos, `14 ms` en ensamblado K y solo `0,114 ms` en actualizar
la cota regional. El resto, aproximadamente `649 ms`, pertenece al Transformer
y al control no cubierto por esos contadores. La cota incremental ya no es el
cuello dominante; puntuar y materializar todo antes de SDPA sí lo es.

También existe un oráculo diagnóstico `--multitoken-oracle K`. Dos cachés BF16
densas independientes ingieren el mismo prompt: una verifica K candidatos con
K forwards de un token y la otra con un único forward causal. Con 1.024 tokens
se observaron speedups de `2,07x`, `4,04x` y `7,14x` para K=2, 4 y 8, con los
mismos argmax en el ejemplo. Sin embargo, para K=4 el error máximo de logits fue
`0,25` y los SHA-256 del KV final fueron distintos, aunque tokens y tamaño
coincidían. Por tanto `canonical_commit_safe` es falso: el resultado demuestra
potencial de verificación paralela, pero prohíbe reutilizar ese KV batch bajo el
contrato actual de paridad residual exacta.

La divergencia se localizó después con precisión. La capa 0 conservó K/V
idénticos; la primera diferencia apareció en la capa 1 y alcanzó 23 de 24
capas. De 49.152 elementos BF16 candidatos, diferían 15.555 elementos K y
19.442 V. Ejecutar con un solo hilo BLAS/OpenMP produjo exactamente los mismos
digests divergentes, descartando una carrera: la causa era la geometría de la
atención multi-query de LibTorch.

El modo diagnóstico `--multitoken-oracle-querywise K` conserva las proyecciones
y MLP en bloque, pero ejecuta cada consulta de atención con la misma longitud y
forma de la ruta secuencial. Esta variante produjo error de logits 0, cero
elementos K/V distintos y SHA-256 idéntico en las 24 capas. Los speedups reales
fueron `1,70x` para K=2, `2,62x` para K=4 y `3,21x` para K=8; en los tres casos
`canonical_commit_safe` fue verdadero. Esto prueba que el backend puede
amortizar una parte importante del Transformer sin cambiar la autoridad BF16,
pero todavía no constituye generación especulativa: faltan un drafter real,
aceptación/rollback y una ruta paginada que no rehidrate todo el prefijo.

El benchmark `--reference-speculative` cubre el siguiente escalón sin introducir
otro modelo: un drafter busca continuaciones del sufijo generado dentro del
prompt, propone dinámicamente entre 2 y 8 tokens y el target BF16 querywise es
la única autoridad. El KV provisional se trunca al prefijo aceptado antes de
confirmar el token correctivo. En una ejecución real de 1.024 tokens, la salida
de ocho tokens conservó exactamente ids y SHA-256 KV. Se aceptaron 6 de 14
propuestas (`42,86 %`), hubo una ronda rechazada, los forwards objetivo bajaron
a cinco y decode pasó de `581,7 ms` a `527,2 ms`, un `1,10x`. Es una prueba corta
y una sola corrida: demuestra generación, aceptación y rollback exactos, no
una ganancia universal.

La verificación exacta ya se ha llevado también al ColdStore paginado mediante
`--reference-speculative-cold`. Es un modo explícito: un tensor de varios tokens
no se interpreta automáticamente como especulación, porque también podría ser
prefill. Durante una ronda, cada página histórica K/V se materializa una vez por
capa; las consultas conservan SDPA de longitud uno y prefijos causales sucesivos.
Los candidatos permanecen en una cola provisional no sellada, de modo que un
rechazo puede truncarla sin cancelar trabajos de conversión ni modificar el
backing frío.

Prueba real MiniCPM5-1B, 1.024 tokens, marcos/chunks de 256 y ocho tokens de
salida:

| Métrica | Terrsvusion secuencial | Verificación fría multí-token |
| --- | ---: | ---: |
| Decode | 1,281 s | 0,470 s |
| Speedup frente a Terrsvusion secuencial | referencia | 2,73x |
| Speedup frente a BF16 denso de la misma corrida | — | 1,21x |
| Eventos del certificador | 168 | 24 |
| Ampliaciones branch-and-bound | 816 | 120 |
| K puntuados / V materializados | 172.704 / 172.704 | 24.600 / 24.600 |
| Lecturas frías operativas | 1.680 | 624 |
| Paridad de ids | referencia | `true` |
| Paridad SHA-256 de todo el KV BF16 | referencia | `true` |

El drafter acertó seis de seis propuestas en dos rondas y redujo en un 85,7 %
los eventos del certificador. El digest se calcula en layout lógico
`[capa][K/V][cabeza][token][dimensión]`; sus lecturas diagnósticas se excluyen
de los contadores operativos.

También se ejecutó un caso adversarial con rechazos: 10 tokens finales, 10
propuestos, 6 aceptados y 3 rondas rechazadas. Conservó exactamente ids y
SHA-256 KV después de los rollbacks, pero fue más lento que BF16 denso
(`1,254 s` frente a `0,747 s`). Esto demuestra que el rollback es real y que el
speedup no es universal y motivó el siguiente cambio.

La generación canónica y el servidor usan ahora un drafter determinista basado
en un autómata de sufijos construido en Rust en tiempo lineal. El primer
candidato siempre es el argmax exacto que el target acaba de calcular; solo se
prolonga cuando el estado del documento tiene una única transición y no es
también terminal. La unicidad documental no se presenta como garantía: todos
los tokens continúan siendo verificados por el target BF16. Una política
adaptativa aumenta K tras aceptación total y, ante un rechazo, vuelve a K=2 y
puede pasar temporalmente por decode secuencial. Los modos `--exact-oracle` y
`--shadow-metrics` desactivan esta ruta para no contaminar sus diagnósticos.

Prueba canónica posterior con 1.024/256/8:

- Terrsvusion: `0,459 s` de decode; BF16 denso: `0,616 s`;
- 8 propuestas, 7 aceptadas, una ronda rechazada y 3 forwards objetivo;
- cero eventos del certificador y cero ampliaciones;
- 576 lecturas frías;
- ids y SHA-256 KV idénticos al control BF16.

La prueba adversarial canónica posterior produjo diez tokens con Terrsvusion en
`0,904 s` frente a `0,936 s` del BF16 denso: 14 propuestas, 8 aceptadas, una
ronda rechazada y 5 forwards objetivo. Conservó también la paridad exacta de
ids y SHA-256. Es una ejecución individual de una geometría concreta, no una
demostración de aceleración general ni una base suficiente para promover la
política a todas las peticiones del servidor.

## Evidencia real más reciente

### Línea base secuencial anterior a la integración multí-token

Configuración: MiniCPM5-1B, 1.024 tokens de contexto, chunks y marcos de 256,
8 tokens generados.

| Métrica | BF16 denso | Terrsvusion |
| --- | ---: | ---: |
| Respuesta | `AZULEJO-7319` | `AZULEJO-7319` |
| Paridad de ids | referencia | `true` |
| Prefill | 1,520 s | 1,526 s (referencia anterior) |
| Decode | 0,605 s | 1,281 s |
| KV BF16 de referencia | 25.337.856 bytes | — |
| KV estable residente | — | 333.780 bytes |
| Ahorro residente contabilizado | — | 98,68 % |

Esta tabla queda como línea base histórica de la ruta secuencial. La ruta
canónica actual es la multí-token exacta descrita arriba.

Después de corregir el antiguo miss LSH que sembraba todo el catálogo, los
estados iniciales acumulados bajaron de 1.008/1.008 a 192/1.008. El
branch-and-bound ejecutó 816 ampliaciones y terminó cargando 1.008/1.008:
`hierarchy_k_tokens_avoided` siguió siendo 0.

### Oráculo exacto

En la misma geometría, con `--compare-dense --exact-oracle`:

| Tolerancia local | Tokens omitibles | Porcentaje de 172.704 oportunidades |
| ---: | ---: | ---: |
| `1e-6` | 0 | 0 % |
| `1e-5` | 0 | 0 % |
| `1e-4` | 5 | 0,0029 % |
| `1e-3` | 441 | 0,2554 % |

Otros resultados:

- 168 eventos exactos observados.
- Primera tolerancia local observada que permite una omisión:
  `2,783061069137239e-6`.
- Margen mínimo top-1/top-2 observado: `0,4375`.
- Paridad final frente al control BF16: `true`.
- Decode con oráculo: aproximadamente 14,48 s frente a aproximadamente 1,8 s
  sin él. Ese coste pertenece al diagnóstico y no a la ruta normal.

El resultado descarta que la AABB sea el único bloqueo en esta prueba: incluso
con scores QK exactos, `1e-6` y `1e-5` permiten cero omisiones. No demuestra que
subir la tolerancia sea seguro para la respuesta final.

## Métricas principales

- `kv_bf16_reference_bytes`: tamaño del KV BF16 denso equivalente.
- `kv_resident_bytes`: backing excluido; estado estable residente contabilizado.
- `kv_physical_bytes`: backing más estado residente y metadatos contabilizados.
- `kv_prefill_working_peak_bytes`: K/V BF16 temporal conocido durante prefill.
- `kv_known_attention_scratch_peak_bytes`: temporales conocidos; excluye
  workspace opaco y padding del allocator LibTorch.
- `kv_exact_*`: IO, caché fría, integridad y lecturas parciales.
- `kv_conversion_*`: trabajos aceptados, entregados y tiempo de workers.
- `parallel_*`: estados dimensionales, fingerprints, micrometadatos y LSH.
- `hierarchy_*`: estados iniciales/cargados, ampliaciones y tokens K evitados.
- `exact_mass_*`: K puntuados, V materializados y cota terminal.
- `exact_oracle_*`: diagnóstico shadow de omisibilidad y margen de logits.
- `prefix_cache`: hit, tokens reutilizados, lookup verificado y rehidratación.
- `kv_sha256`: digest del estado KV lógico completo, independiente de cómo se
  fragmenten físicamente las páginas.

Estas métricas no equivalen a RSS total. En particular, no incluyen todo el
workspace interno de SDPA/LibTorch.

## Servidor y concurrencia

```bash
cargo run --release --features torch-backend --bin terrsvusion-server -- \
  --model /home/yo/models/MiniCPM5-1B \
  --prefix-cache-dir /ruta/estable/terrsvusion-prefix-cache \
  --host 127.0.0.1 \
  --port 8088
```

Endpoints:

- `GET /` chat web para probar el motor
- `GET /health`
- `GET /ready`
- `GET /v1/models` y `GET /v1/capabilities`
- `POST /v1/chat/completions` (greedy, sin SSE ni tools)

```bash
curl -sS http://127.0.0.1:8088/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hola"}],"max_tokens":32}'
```

El servidor usa un pool de hilos propietarios (`--workers`, 1–8). Sin opciones
adicionales, cada worker carga una autoridad BF16 independiente. Con
`--shared-authority`, el checkpoint se carga una vez y cada worker recibe
handles `Tensor::shallow_clone()` que comparten el mismo almacenamiento
LibTorch inmutable; el arranque comprueba la identidad de todos los `data_ptr`.
Las cachés KV siguen siendo privadas por petición. La cola de cada worker tiene
ocho slots. La saturación se devuelve como HTTP 429; un pool detenido se
devuelve como 503. Dos peticiones pueden ejecutarse a la vez si hay al menos
dos workers.

Límites operativos actuales:

- timeout HTTP por petición: `--request-timeout-secs` (300 s por defecto; 504);
- cancelación cooperativa al abandonar la espera (timeout o drop del future);
- sin deadline separado para prefill/decode;
- `GET /ready` distinto de `GET /health`;
- sin supervisor de proceso externo;
- `[profile.release] panic = "unwind"`: un pánico del encoder se reporta y el
  worker sigue sirviendo.

## Construcción y validación

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/yo/.local/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cargo fmt --all -- --check
cargo test --features torch-backend --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --all-features --all-targets
```

Estado actual verificado:

- 100 tests de biblioteca, 2 del ejecutable documental y 13 del servidor
  correctos (115 en total);
- 0 fallos;
- Clippy limpio con advertencias tratadas como error;
- biblioteca y todos los targets/features compilados en release.

Prueba real de prefijo persistente del 28 de octubre de 2026, con
MiniCPM5-1B BF16, un fragmento coherente en español de *La isla del tesoro* y
dos preguntas diferentes:

| Evento | Prefijo | Tokens reutilizados | Prefill | Lookup | Rehidratación |
| --- | --- | ---: | ---: | ---: | ---: |
| Primera pregunta | miss | 5.454 | 15,954 s | 0,00002 s | 0,155 s |
| Segunda pregunta | hit | 5.454 | 0,549 s | 0,654 s | 0,147 s |
| Tras reiniciar servidor | hit | 5.454 | 0,443 s | 0,701 s | 0,163 s |

El hit redujo el tiempo medido de prefill aproximadamente un 96,6 %. La prueba
tras reinicio demuestra persistencia de proceso. No demuestra que la respuesta
completa sea rápida: con 16 tokens generados, decode siguió costando unos
22,5 s y dominó la latencia total. La fila tras reinicio generó solo dos tokens
y sirve para probar recuperación, no para comparar latencia total.

Una comparación adicional con la misma pregunta y cuatro tokens generados
produjo exactamente el mismo texto (`<think>\nEsto`) en miss y hit. El prefill
medido fue 18,470 s en miss y 0,482 s en hit. Esta prueba acotada demuestra
paridad de la secuencia observada; no constituye una prueba universal para toda
entrada o longitud de generación.

Regresiones cubiertas explícitamente:

- cada máscara Hamming de radio hasta 4 se visita una sola vez sin construir
  el antiguo vector temporal de 12.951 máscaras por tabla;
- en builds con unwind, un panic del encoder produce un resultado de error
  ordenado y el worker procesa el trabajo siguiente; el perfil release usa
  `panic = "unwind"` para que esa recuperación exista también en binarios
  optimizados;
- el prefijo persistente reconstruye 24 capas, verifica el backing, conserva el
  eje dimensional y rehidrata K/V BF16 exacto;
- los identificadores de región duplicados fallan cerrados antes de construir
  cotas o materializar K.

Una corrida real posterior a la materialización incremental, con 512 tokens,
chunks y marcos de 128, dos tokens solicitados y control BF16, produjo los ids
`[359, 23373]` en ambas rutas (`exact_token_parity: true`). Durante el único
paso decode comparado se observaron 24 eventos de capa, 120 ampliaciones y 144
estados materializados en total; `kv_exact_cold_reads` fue 240. La ejecución
sigue cargando todos los estados y no demuestra todavía aceleración por
omisión.

Benchmark con control y oráculo:

```bash
cargo run --release --features torch-backend --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --book-tokens 1024 \
  --chunk-size 256 \
  --storage-frame-tokens 256 \
  --generate-tokens 8 \
  --compare-dense
```

`--book-file RUTA --question TEXTO` usa un documento real. El valor de
`--book-tokens` limita los tokens de libro incluidos; el prompt y la pregunta
se añaden aparte. `--adversarial-retrieval` y `--stress-seed` generan entradas
sintéticas distintas y son mutuamente excluyentes.

## Evidencia real del 28 de octubre de 2026 (esta sesión)

Corridas MiniCPM5-1B BF16 locales, no una distribución estadística.

### Omisión certificada aplicada hasta logits

```bash
cargo run --release --features torch-backend --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --book-tokens 256 --chunk-size 128 --storage-frame-tokens 128 \
  --generate-tokens 2 --apply-omission-tolerance 1e-3
```

| Métrica | Resultado |
| --- | ---: |
| Tokens omitidos de V con certificado local `1e-3` | 8 de 6.168 puntuados |
| `hierarchy_k_tokens_avoided` | 0 |
| Paridad de ids vs ruta canónica `1e-6` | `true` (`[359, 23373]`) |
| Pasos de logits comparados | 2 |
| L∞ de logits (canónica vs aplicada) | `0,171875` |
| SHA-256 KV | distinto (esperado: V parcial) |

La cota local `1e-3` sí omitió V y conservó los ids en esta geometría. No es
error 0 en logits: `0,171875` es real. El servidor sigue usando `1e-6` y no
activa esta rama. A `1e-6` la ruta canónica de la misma corrida materializó
los 6.168 V.

### Hits de prefijo sin copiar el backing

```bash
cargo run --release --features torch-backend --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --generate-tokens 2 --prefix-reuse-demo
```

| Evento | Hit | Tokens | Copiados | Inmutable | Prefill |
| --- | --- | ---: | ---: | --- | ---: |
| Primera pregunta | false | 888 | 0 | false | 1,551 s |
| Segunda pregunta | true | 888 | 0 | true | 0,121 s |

El hit abrió el archivo publicado en solo lectura. No hubo `fs::copy`. El
prefill del sufijo ensambló el histórico por capa y no retuvo las 24 capas
de prefijo en `prefill_hot`. Speedup de prefill ≈ `12,86x`. No elimina el
coste cuadrático del primer prefill.

### Servidor y paralelismo verificado

- El pool de dos ejecutores atiende dos jobs a la vez. En modo
  `--shared-authority`, los workers conservan ejecutores separados pero todos
  los tensores de peso apuntan al mismo almacenamiento LibTorch.
- Arranque real MiniCPM5-1B con dos workers compartidos: 219 tensores de peso
  verificados por igualdad de `data_ptr`; `/v1/capabilities` devolvió
  `libtorch_model_shared_between_workers=true`.
- Release con `panic = "unwind"`: el test de pánico recupera y sirve el job
  siguiente.
- Drop de la respuesta cancela un job aún en cola.
- Servidor: `--workers`, `--request-timeout-secs`, `GET /ready`.

## Garantías demostradas

- Los pesos no se reescriben ni se guardan en Q4.
- Los K/V recuperados para atención proceden de BF16 exacto.
- Un marco no se libera antes de confirmar su commit frío.
- Los registros comprometidos se verifican y las lecturas parciales validan
  cada token recuperado.
- Un miss LSH no activa fuerza bruta ni otro sistema.
- El catálogo dimensional no puede sustituir al certificado.
- La prueba descrita conserva exactamente los ids generados frente al control
  BF16.
- La residencia KV estable contabilizada es muy inferior al denso en esa
  prueba.

## No demostrado todavía

- `hierarchy_k_tokens_avoided > 0` en una inferencia real del modelo.
- Menor latencia general que BF16 denso en todas las geometrías.
- Reducción del coste cuadrático del *primer* prefill (el hit sí evita ese
  Transformer y ya no copia el backing).
- Paridad universal o L∞ de logits 0 con omisiones a `1e-3` (medido `0,171875`
  con ids idénticos en 256/128/2).
- Seguridad de promover `1e-3` al servidor; la autoridad de servicio sigue en
  `1e-6`.
- Certificación de 30.000, 100.000 o 131.072 tokens.
- Métricas completas de RSS, workspace interno de LibTorch o VRAM.
- Escalado de latencia, throughput y RSS con carga concurrente prolongada sobre
  la autoridad LibTorch compartida.

## Próxima decisión basada en evidencia

La siguiente prueba útil no es añadir una geometría sofisticada a ciegas. Es
ejecutar una rama shadow que aplique las omisiones candidatas, mida el error real
de atención, propague el resultado hasta logits y compare toda la secuencia
autoregresiva. Solo después podrá decidirse si procede:

1. derivar una tolerancia dinámica con garantía hasta logits;
2. implementar una envolvente K orientada de bajo rango con residual
   certificado;
3. separar las firmas LSH por cabeza KV;
4. o aceptar que determinadas consultas requieren atención completa.
