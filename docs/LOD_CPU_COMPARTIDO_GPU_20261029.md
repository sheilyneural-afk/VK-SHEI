# Terrsvusion V3 — LOD canónico de CPU reutilizado en AMD iGPU (29/10/2026)

> Actualización posterior (29/10/2026): el backend GPU incorpora
> un planificador de micrometadatos y regiones BF16 para seleccionar
> K/V con certificado en **epsilon positivo**. epsilon=0 sigue
> ejecutando la atención exacta. Véase
> docs/MICROMETADATOS_REGIONALES_GPU_20261029.md.


> **Actualización posterior (29/10/2026):** la ruta GPU incorpora ahora prefill paralelo por lotes y ColdStore agrupado en páginas de hasta 64 tokens. Las cifras de GPU secuencial en este informe son HISTÓRICAS. Resultados nuevos en PREFILL_GPU_OPTIMIZADO_20261029.md.


## Proyecto correcto y respaldo

Todos los cambios se realizaron SOLO en la carpeta independiente del Escritorio:

/home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3

La copia de seguridad íntegra antes de esta fase se conserva intacta en:

/home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-unified-20261029

No se modificó SHEILY. El servidor BF16 CPU original continúa en el
puerto 8099, con su modelo, índices, reglas, archivos, configuración y
caché separados del backend OpenCL.

## Qué se ha trasladado realmente del CPU a GPU

El adaptador OpenCL NO ha implementado otro sistema independiente de
búsqueda, cuantización o recuperación. Reutiliza las mismas funciones y
estructuras Rust del proyecto que emplea CPU:

| Módulo original | Uso en CPU | Uso en GPU |
| --- | --- | --- |
| adaptive_extent::ParallelContextAxis | Catálogo dinámico, firmas de 4 bits, LSH, regiones de tokens y consultas protegidas | El MISMO módulo Rust, observa el K real de capa cero y reutiliza su planificación entre capas |
| cold_store::ColdStore | Almacenamiento BF16 exacto, checksum y hashes por token, lectura parcial K/V | El MISMO formato y código Rust para guardar K y V exactos en disco y recuperar V seleccionadas |
| attention_certificate::select_by_exact_mass | Fallback terminal con límite de error de la masa softmax | La MISMA función Rust con scores QK auténticos obtenidos por los kernels GPU |
| attention_certificate::certified | Prueba de tolerancia por cabeza | La MISMA función Rust aplicada a los límites por cabeza |
| BF16 original | K/V exactos y pesos originales | Pesos originales BF16, K exactas en GPU, V exactas en ColdStore y staging V BF16 en GPU |

La GPU calcula los scores Q·K, softmax y PV con OpenCL real sobre la
Radeon Phoenix. No hay PyTorch, modelos reentrenados ni simulaciones.

El índice 4-bit no reemplaza los pesos ni la precisión BF16: es
metadato auxiliar para recuperar las coordenadas token correctas.
Los scores QK usados para certificar son de la consulta Q real.

### Arquitectura de datos

- Pesos BF16: residentes en iGPU (2.161.265.664 bytes del checkpoint).
- K exactas BF16: residentes en iGPU para el scoring de atención.
- K/V exactas BF16: copia recuperable en el ColdStore original de Rust
  y en el formato de registros original. No están cuantizadas.
- V de largo plazo: NO se asigna una matriz BF16 completa en GPU.
  Se leen de ColdStore y entran en una zona GPU de tamaño fijo.
- Índice 4-bit, regiones y LSH: memoria CPU, mismo código original.
- Certificado por cabeza y fallback: mismas funciones originales.
- Dos rutas GPU disponibles: GPU densa para referencia y GPU LOD
  con ColdStore e índices canónicos.

El cache K/V frío nuevo NO está compartido físicamente con un proceso
CPU que tenga otra sesión: cada proceso conserva su propia sesión y
sus ficheros. Reutilización de código y formato NO implica intercambiar
un KV entre modelos o sesiones sin verificar tokenizer, pesos y estado.

### Qué falta para equivalencia completa con el CPU

El original CPU tiene además un paso de **cota por regiones K y
branch-and-bound jerárquico** que intenta evitar leer K históricas antes
del fallback de masa exacta. En esta ruta GPU, K permanecen residentes
y la selección utiliza directamente el fallback canónico de scores
exactos. Por eso todavía NO es el pipeline completo del CPU, aunque
usa el MISMO algoritmo final para certificar V y las mismas estructuras
de índice/cold store. El sistema de prefijos persistentes verificados
entre peticiones del CPU tampoco se ha trasladado aún a GPU.

No se describe esa parte pendiente como implementada ni se declara
paridad total con el BF16 denso de CPU.

## Validación de precisión real en la AMD Phoenix

### GPU LOD epsilon cero contra GPU densa

Con el MISMO prompt_ids, pesos BF16 y generación greedy «Hola», la
prueba real comparó 16 pasos de generación:

- Identificadores de tokens idénticos en las dos rutas GPU.
- Todos los logits BF16 comparados fueron idénticos BIT A BIT para
  esos 16 pasos; la prueba falla si aparece cualquier discrepancia.
- Se aplicó la selección exacta con tolerancia 0 (ninguna V omitida).
- Se hizo el reset de sesión y se repitió la generación con los
  mismos resultados.
- Se verificó que el almacenamiento frío se utilizó en realidad
  y que el índice 4-bit físico tiene bytes residentes.
- Cargo test GPU real: correcto, 0 errores.

Comando reproducible:

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked --offline \
  --features opencl-backend gpu_actual_canonical_lod_zero -- \
  --ignored --nocapture
~~~

Importante: esta igualdad es entre rutas **GPU** con los mismos kernels.
La ruta GPU ya tenía diferencias frente a la referencia BF16 CPU:
el informe anterior encontró diferencias máximas de logits de 0,5
y SHA KV distinto. El nuevo LOD epsilon cero no corrige esa
diferencia aritmética de OpenCL respecto a LibTorch.

### Epsilon positivo

Se realizó una prueba con epsilon = 0,001 sobre GPU real.
El selector mantuvo todas las V cuando la cota no justificó
su omisión. El resultado no demuestra aceleración automática por
atención dispersa. En una prueba sintética unitaria del sistema de
certificación sí se omiten posiciones cuando la masa es
suficientemente concentrada, y los límites cumplen la tolerancia.

Los certificados están basados en scores FP32 de GPU; no cubren
automáticamente los errores de redondeo posteriores de todos los
operadores BF16 del modelo.

## Memoria reservada de KV

Con un contexto configurado para 128 tokens, 24 capas, 2 cabezas
KV y 128 dimensiones:

- GPU densa: 3.145.728 bytes de capacidad K+V completa.
- GPU LOD (K completas + staging 8 tokens + metadatos de stage):
  1.585.248 bytes.
- Ahorro de capacidad asignada a KV: aproximadamente 49,6 %.

La mejora de capacidad NO significa reducción equivalente de la RAM,
VRAM total o de los pesos del modelo. La GPU aún conserva las K
completas y los pesos pesan 2,16 GB. ColdStore ocupa espacio en SSD y
puede aumentar la latencia por las lecturas de V.

El sistema CPU original utiliza páginas de ColdStore con tokens
agrupados. Esta primera integración GPU confirma exactitud con una
página por token, que añade metadatos/IO; agruparlas en marcos
de 16–512 tokens preservando hash, offsets y certificados es una
optimización posterior necesaria.

## Configuración de la GPU

Iniciar GPU con los componentes LOD de CPU, tolerancia estricta:

~~~bash
TERRSVUSION_OPENCL_DEVICE=AMD cargo run --release --locked \
  --features opencl-backend --bin terrsvusion-opencl-server -- \
  --model /home/yo/models/MiniCPM5-1B \
  --host 127.0.0.1 --port 8188 \
  --resident --max-context-tokens 2048 \
  --gpu-lod --lod-epsilon 0 --lod-stage 64
~~~

La GPU server habilita el LOD canónico por defecto con epsilon 0.
La opción --gpu-dense-kv conserva la ruta GPU densa previa, útil
para oráculo de comparación, sin ColdStore.

Para aproximación certificada: --lod-epsilon 0.001. Solo se recomienda
tras validar por caso los scores y la calidad; no se garantizan
omisiones ni mayor velocidad.

Consultar el estado desde el PC:

~~~bash
curl -sS http://127.0.0.1:8188/health
curl -sS http://127.0.0.1:8188/v1/capabilities
~~~

La API indica módulos reutilizados, epsilon, tamaño de staging,
capacidad GPU KV y contadores reales por solicitud.

## Selector único CPU / GPU / Auto

Se creó además un router Rust sin PyTorch ni LibTorch que envía
peticiones HTTP a los dos motores AUTÓNOMOS. No modifica su
configuración, pesos, ColdStore ni estados en proceso.

Abrir en el navegador del PC:

http://127.0.0.1:8288/

Arranque del router:

~~~bash
cargo run --release --locked --features unified-router \
  --bin terrsvusion-unified-server -- \
  --host 127.0.0.1 --port 8288 \
  --cpu-url http://127.0.0.1:8099 \
  --gpu-url http://127.0.0.1:8188 \
  --auto-prefer cpu
~~~

- CPU: ruta BF16 del proyecto previo, ColdStore LOD completo,
  incluyendo selección jerárquica y prefijos persistentes.
- iGPU: modelo BF16 OpenCL con el índice/ColdStore/certificado
  canónicos descritos arriba, sin todas las optimizaciones CPU.
- Auto: prioriza CPU por defecto. Si CPU deja de responder, cambia
  a GPU solo si la solicitud es compatible. No cambia de motor
  a mitad de un turno ni transfiere KV sin verificar.
- Una solicitud explícita CPU/GPU nunca usa el otro sin aviso.
- Si el usuario envía un documento KV CPU y CPU no está disponible,
  responde error en lugar de perder el documento enviándolo a GPU.

El router no se expone públicamente ni incorpora autenticación.
Se enlaza a 127.0.0.1 y solo acepta backends loopback.

## Estado y límites honestos

- Motor CPU original: intacto, operativo en 8099.
- Motor GPU con parte del LOD CPU compartido: operativo en 8188.
- Selector CPU / iGPU / Auto: operativo en 8288.
- Certificados de mass fallback y cold store: idéntico código.
- Cota jerárquica original antes del fallback: pendiente en GPU.
- KV ColdStore de prefijos entre peticiones: pendiente en GPU.
- Compresión de pesos a 4 bits sin pérdidas: no declarada.
- Paridad GPU LOD=0 vs GPU densa: demostrada en 16 pasos de «Hola».
- Paridad total GPU vs CPU BF16: NO demostrada.
- Speedup end-to-end por LOD: NO demostrado.
- Compatibilidad con cualquier LLM/GPU: NO demostrada.

El respaldo completo antes de esta fase sigue en el Escritorio.

