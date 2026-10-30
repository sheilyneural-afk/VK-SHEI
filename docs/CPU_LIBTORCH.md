# Motor CPU (LibTorch)

Necesita la feature `torch-backend`. El código está en `torch_runtime.rs` y
`prefill_tch.rs`. Estado de `main` al 30/10/2026.

## Modelos admitidos

`model_config.rs` (`ensure_torch_runtime_compatible`) solo acepta dos modelos.
Cualquier otro checkpoint solo se puede inspeccionar.

| Modelo | Requisitos | Rutas de servicio |
|---|---|---|
| MiniCPM5-1B BF16 | checkpoint local | `terrsvusion-rust` (ColdStore + certificado) y `minicpm-dense-bf16` (denso) |
| Llama 3.1 8B Instruct BF16 | `tokenizer.json` y los shards safetensors | `llama-dense-bf16` (denso) |

Al cargar, cada peso se copia desde el mmap y después se liberan esas páginas
con `MADV_DONTNEED` (`74a553e`). Así, durante la carga, el RSS no cuenta los
pesos dos veces.

## MiniCPM5-1B

- **Ruta `terrsvusion-rust`:** el K/V se sella en páginas de la ColdStore. En
  la decodificación, la atención usa masa exacta (`exact_mass_attention`) con
  una tolerancia fija de 1e-6: calcula todos los scores K y solo deja fuera
  los V cuya contribución queda acotada.
- **Ruta `minicpm-dense-bf16`** (`native=true`): plantilla oficial de MiniCPM y
  K/V denso BF16, sin caché Terrsvusion.
- `compare_native=true` ejecuta las dos rutas sobre los mismos tokens e
  informa de las diferencias. No se puede combinar con `native=true`.
- Si no hay tolerancia explícita, oráculos ni métricas de sombra, la
  generación usa decodificación especulativa con un borrador determinista de
  sufijos del propio prompt (`DeterministicSuffixDrafter`).

## Llama 3.1 8B Instruct

- **Prefill en streaming por bloques.** Desde `b51e3b2`, cada capa mantiene en
  RAM un `ResidentCommittedKv`: cada página fría se lee y se verifica una sola
  vez, y luego solo se añaden las páginas nuevas. El presupuesto se fija con
  `TERRSVUSION_RESIDENT_KV_BYTES` (por defecto, un tercio de la RAM con un
  máximo de 24 GiB). Si una capa no cabe, vuelve a la relectura desde la
  ColdStore. El informe incluye `kv_committed_resident_{bytes,peak_bytes,pages_reused,fallbacks}`.

### Omisión certificada de V: solo diagnóstico

En Llama, la omisión de lecturas de V está desactivada en el servidor. Solo
se puede probar con `terrsvusion-long-context --apply-omission-tolerance <ε>`.
Se acepta únicamente si los logits finales coinciden bit a bit con los
canónicos (`omission_accepted` exige `exact_logit_parity`). La ruta puntúa todo
el historial K para conservar el denominador del softmax.

Cambios del 30/10/2026 en esta ruta:

- `ee6cf6a`: la celda de redondeo BF16 se comprueba alrededor del candidato
  FP32 sin redondear.
- `7f4fe0f`: la cota de omisión de V usa, para cada token, el máximo `|V|` de
  su región, de su página o de la cola, en lugar del máximo de toda la cabeza.
- `6f02a66`: el informe añade `hierarchy_k_events` y
  `hierarchy_k_tokens_skippable`, es decir, cuántos tokens K podría saltar la
  prueba regional. En Llama, `hierarchy_k_tokens_avoided` sigue en 0 porque no
  se ahorra ninguna lectura real. La cota regional usa ahora `head / (Q / KV)`,
  que vale para 32Q/8KV.

Estado medido:

- **Caso sintético físico** (test
  `llama_32q8kv_certified_v_read_from_physical_coldstore_is_partial_and_bounded`,
  32Q/8KV, 17 tokens): puntúa las 17 K pero solo materializa como mucho 2 V.
  La lectura parcial evita bytes físicos de la ColdStore, la celda de redondeo
  BF16 queda certificada y la salida de atención es idéntica bit a bit. No es
  `#[ignore]` y pasa en `main`.
- **Checkpoint real** (30/10, 128 tokens, tolerancia 1e-3): 11 eventos
  candidatos, 0 certificados por celda, 0 omisiones aceptadas. Tras el
  respaldo denso, los logits coinciden exactamente. Esta ejecución es
  **anterior** a `ee6cf6a`, `7f4fe0f` y `6f02a66`, y no se ha repetido.
  Evidencia: [EVIDENCIA_LLAMA_SDPA_V_CERO_20261030.json](evidencia/cpu/EVIDENCIA_LLAMA_SDPA_V_CERO_20261030.json).

Los mensajes de usuario y asistente se envían al tokenizer con su contenido
original, sin pasar por normalizaciones, traducciones, ampliaciones de
preguntas ni reescrituras de contenido. La ruta puede añadir un mensaje de
sistema por defecto cuando la conversación no especifica ninguno.

El digest de comparación del KV BF16 recorre todas las cabezas y dimensiones,
incluidas las 8 cabezas KV de Llama 3.1 8B. Es una métrica diagnóstica,
no una prueba automática de igualdad entre bibliotecas.

## Servidor `terrsvusion-server`

```bash
cargo run --release --features torch-backend --bin terrsvusion-server -- \
  --model /ruta/al/modelo [--port 8088] [--workers N] [--shared-authority] \
  [--prefix-cache-dir DIR] [--document-graph grafo.json] [--request-timeout-secs S]
```

| Ruta | Función |
|---|---|
| `GET /`, `GET /nativo` | UIs de chat (`assets/chat.html`, `assets/chat-native.html`) |
| `GET /health`, `GET /ready` | estado y preparación |
| `GET /v1/models`, `GET /v1/capabilities` | modelos y capacidades |
| `POST /v1/chat/completions` | chat estilo OpenAI; `max_tokens` entre 1 y 512 (128 por defecto) |
| `POST /v1/document/query` | pregunta sobre el documento completo del grafo; ver [DOCUMENTOS.md](DOCUMENTOS.md) |

- `applied_mass_tolerance` se rechaza con 422: la tolerancia no se puede
  cambiar por la API.
- **Prefijos persistentes.** Si la conversación tiene más de un mensaje, todo
  lo anterior al último se trata como prefijo y se guarda en
  `--prefix-cache-dir` (por defecto `<modelo>/.terrsvusion-prefix-cache`). Se
  publica como ColdStore inmutable mediante enlace duro, sin copiar bytes. Si
  el enlace duro no es posible, se copia y el informe indica los bytes
  copiados. `terrsvusion-long-context --prefix-reuse-demo` hace lo mismo desde
  la línea de comandos.
- `--workers N` (de 1 a 8) crea N trabajadores con un cursor de reparto
  compartido. `--shared-authority` hace que compartan los tensores de pesos.
  `--document-graph` exige `--shared-authority`.

## `terrsvusion-long-context`

Banco de pruebas de contexto largo sobre un libro sintético o un archivo
(`--book-file`).

| Opción | Qué hace |
|---|---|
| `--book-tokens`, `--chunk-size`, `--storage-frame-tokens`, `--generate-tokens` | tamaño del contexto, del bloque de prefill, del marco de almacenamiento y de la generación (1 a 512) |
| `--dense-only`, `--compare-dense` | solo la ruta densa, o la comparación con ella |
| `--exact-oracle`, `--multitoken-oracle[-querywise]` | oráculos exactos de omisión |
| `--shadow-metrics` | métricas diagnósticas del certificado |
| `--reference-speculative[-cold]` | decodificación especulativa de referencia |
| `--apply-omission-tolerance ε` | diagnóstico de omisión certificada de V |
| `--prefix-reuse-demo` | publica un prefijo y lo reutiliza |
| `--adversarial-retrieval`, `--stress-seed`, `--question` | variantes de la prueba |

## Evidencia (Llama 3.1 8B, CPU, 30/10/2026)

Las tres mediciones son anteriores a los commits `98d52aa`–`7f4fe0f` del
30/10/2026. Los tiempos y el RSS
no incluyen el K/V residente (`b51e3b2`) ni la liberación del mmap (`74a553e`).

| Medición | Resultado | Archivo |
|---|---|---|
| Prefill denso de 4000 tokens: bloques de 512 frente a un único bloque | el mismo trabajo causal; con bloques de 512, el prefill tardó 92,5 s frente a 111,5 s (un 17 % menos) y el pico de trabajo bajó un 84,4 % | [EVIDENCIA_PREFILL_DENSO_4K_20261030.json](evidencia/cpu/EVIDENCIA_PREFILL_DENSO_4K_20261030.json) |
| Publicación de prefijo sin copia (807 tokens) | 0 bytes copiados en el fallo y en el acierto; el prefill del acierto tardó 0,62 s frente a 17,6 s (28 veces menos) | [EVIDENCIA_PUBLICACION_PREFIJO_SIN_COPIA_20261030.json](evidencia/cpu/EVIDENCIA_PUBLICACION_PREFIJO_SIN_COPIA_20261030.json) |
| Omisión certificada de V | ver arriba | [EVIDENCIA_LLAMA_SDPA_V_CERO_20261030.json](evidencia/cpu/EVIDENCIA_LLAMA_SDPA_V_CERO_20261030.json) |
