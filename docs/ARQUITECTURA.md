# Arquitectura

Estado del código en `main` al 30/10/2026. Las rutas son relativas a `src/`.

## Principio

Los pesos y el K/V autoritativo se conservan en BF16. Todo lo que es
aproximado o compacto, como el índice de 4 bits, los micrometadatos o los
resúmenes de región, sirve solo para buscar y acotar. Nunca sustituye a los
valores del modelo. Una omisión de atención solo se acepta si un certificado
demuestra que el error queda dentro de la tolerancia pedida.

## Módulos

| Módulo | Feature | Función |
|---|---|---|
| `cold_store.rs` | — | Almacén frío de páginas K/V BF16, de solo anexado |
| `adaptive_extent.rs` | — | Eje de contexto paralelo (`ParallelContextAxis`): regiones definidas por contenido, huellas de 4 bits y cotas por región |
| `attention_certificate.rs` | — | Cotas de masa omitida, selección por masa exacta, selección de V por contribución y comprobación de celdas de redondeo BF16 |
| `shadow_metrics.rs` | — | Agregación diagnóstica de los certificados de contexto largo |
| `document_graph.rs` | — | Autoridad documental por página y datos de enrutado |
| `model_config.rs` | — | Inspección de `config.json` y comprobación de compatibilidad por motor |
| `portable.rs`, `portable_store.rs` | — | Marcos K/V portables con bytes BF16/F16/F32 exactos y su almacén; referencia de CPU separada de la ColdStore |
| `inference_worker.rs`, `conversion_worker.rs` | — | Hilo único propietario del modelo con cola acotada, y transferencia en segundo plano de páginas copiadas |
| `torch_runtime.rs`, `prefill_tch.rs` | `torch-backend` | Motor CPU LibTorch para MiniCPM5 y Llama |
| `opencl_backend.rs` | `opencl-backend` | Atención QK/softmax/PV aislada en GPU, sin LLM completo (la usa `terrsvusion-igpu-bench`) |
| `opencl_llm.rs` | `opencl-backend` | Motor completo BF16 en Rust y OpenCL |
| `gpu_canonical_lod.rs` | `opencl-backend` | Adaptador que reutiliza en GPU la ColdStore y el eje de CPU, sin duplicar el índice |
| `gpu_micro_regions.rs` | `opencl-backend` | Poda de regiones certificada sobre el LOD de GPU |
| `native_hip.rs` | `hip-backend` | Puente HIP BF16 opcional para gfx1103 |

Los kernels OpenCL están en `assets/opencl_llm.cl` y
`assets/opencl_attention.cl`. Las cuatro UIs HTML (`assets/chat*.html`) se
incrustan en los servidores con `include_str!`.

## ColdStore (`cold_store.rs`)

- Cada registro empieza con la cabecera `TSKVPG04` y se confirma con un pie
  `TSKVCMIT`. Si el último anexado quedó a medias, se ignora al reabrir; si se
  corrompe un registro ya confirmado, se rechaza.
- Cada página guarda sumas rápidas de K, de V y de los hashes por token, más
  un SHA-256 del registro. La corrupción se detecta aunque coincidan las sumas
  rápidas.
- `open_read_only` abre un almacén inmutable, rechaza los anexados y no trunca
  una cola rota.
- Desde `98d52aa`, una escritura fallida a medias seguida de otro anexado ya no
  impide reabrir. Lo cubren los tests
  `failed_partial_append_followed_by_append_still_reopens`,
  `external_torn_tail_before_append_is_discarded` y
  `read_only_open_ignores_torn_tail_without_truncating`.
- La lectura selectiva de V por token conserva la disposición por cabeza y
  detecta la corrupción (test
  `selected_value_tokens_keep_head_major_layout_and_detect_corruption`).

## Índice de 4 bits (`adaptive_extent.rs`)

`FourBitFingerprint` guarda dígitos con signo en `[-7, 7]`, dos por byte, con
una escala por huella. Es metadato de catálogo: sirve para puntuar regiones y
no sustituye nunca al K/V BF16.

## Certificado (`attention_certificate.rs`)

- `omitted_error_bound`: cota del error de salida de una cabeza a partir de los
  scores conservados y de cotas superiores de masa por región omitida.
- `select_by_exact_mass`: el menor prefijo por cabeza cuyo error queda acotado
  por `2 · masa omitida · max|V|`. Necesita todos los scores K, pero no todos
  los V.
- `select_v_by_token_contribution`: desde `7f4fe0f`, la cota usa el `|V|` de
  cada token, no el máximo de la cabeza, y mantiene el denominador completo.
- `bf16_rounding_cells_contain*`: desde `ee6cf6a`, la celda de redondeo BF16 se
  comprueba alrededor del candidato FP32 sin redondear, no alrededor de una
  salida ya redondeada.

En el servicio HTTP la tolerancia no es configurable: la API rechaza
`applied_mass_tolerance` con 422.

## Concurrencia y servicio

- `inference_worker.rs` mueve el ejecutor del modelo a un único hilo y los
  manejadores HTTP le envían trabajos por una cola acotada. Los pools clonados
  comparten un único cursor de reparto (test
  `cloned_pools_share_one_round_robin_cursor`).
- Los servidores escuchan en `127.0.0.1` por defecto y no tienen
  autenticación.

## Fidelidad de entrada y parada de generación

Los mensajes de usuario son inmutables hasta su serialización mediante la
plantilla. El runtime OpenCL lee los identificadores EOS del checkpoint y
no utiliza IDs exclusivos de MiniCPM5 para Llama 3.1. El servidor GPU tiene
plantilla MiniCPM5 fija, por lo que valida la geometría y rechaza otros modelos.
La decodificación GPU densa es la opción por defecto; LOD es explícita.

## Variables de entorno

| Variable | Uso |
|---|---|
| `TERRSVUSION_RESIDENT_KV_BYTES` | Presupuesto de RAM para el K/V confirmado residente en el prefill largo. Por defecto es un tercio de la memoria física con un máximo de 24 GiB; `0` lo desactiva y vuelve a releer desde la ColdStore |
| `TERRSVUSION_OPENCL_DEVICE` | Filtro del dispositivo OpenCL que usan los motores GPU y las pruebas `gpu_actual_*` |
| `TERRSVUSION_TEST_MODEL` | Checkpoint de MiniCPM5 para las pruebas ignoradas |
| `TERRSVUSION_LLAMA_MODEL` | Checkpoint de Llama para las pruebas ignoradas |
