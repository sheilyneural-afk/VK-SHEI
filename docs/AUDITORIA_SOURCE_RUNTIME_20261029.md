# Auditoría source-to-runtime de Terrsvusion V3

Fecha: 2026-10-29

Este informe sigue el código ejecutable del checkout y separa pruebas unitarias,
pruebas físicas de GPU e inferencia real del checkpoint Llama. El README y los
informes previos se tratan como afirmaciones que deben verificarse, no como
prueba de ejecución.

## Estado de validación ejecutado

```text
cargo test --lib                                  55 passed
cargo test --all-targets --features torch-backend 77 passed
cargo test --lib --features opencl-backend        63 passed, 28 ignored
```

Las ejecuciones se hicieron con `CARGO_TARGET_DIR=/tmp/terrsvusion-cert-target`
para no compartir el lock de compilación con otros procesos del equipo.

Pruebas físicas ejecutadas en la Radeon AMD Phoenix:

```text
gpu_actual_canonical_lod_zero_matches_dense_gpu_outputs                 PASS
gpu_actual_micro_full_model_falls_back_when_uncertified                 PASS
gpu_actual_micro_regions_skip_real_qk_and_cold_v_with_error_bound       PASS
gpu_actual_micro_hot_model_375_bitwise_dense_fallback_and_cold_integrity PASS
gpu_actual_prefill_64_128_192_256_bit_exact_full_logits_kv_and_speed     PASS
gpu_actual_verified_shared_prefix_375_vs_full_bf16_kv_and_logits        PASS
```

## Inferencia real Llama 3.1 8B

Se ejecutó el binario `terrsvusion-long-context` contra:

```text
/home/yo/models/Llama-3.1-8B-Instruct
book_tokens=256
chunk_size=128
storage_frame_tokens=128
generate_tokens=1
applied_mass_tolerance=0.001
```

Resultado observado:

```text
applied_mass_tolerance: 0.001
certified_v_tokens_omitted: 0
hierarchy_k_tokens_avoided: 0
exact_token_parity: true
logit_steps_compared: 1
max_abs_logit_difference: 0.0
```

La ruta Llama positiva se intentó realmente, pero no obtuvo un certificado de
omisión en esta inferencia. Por tanto esta corrida prueba que el fallback
conservador funciona; no prueba omisión positiva en Llama real.

También midió, para esta geometría concreta:

```text
KV BF16 de referencia:       33.554.432 bytes
KV físico ColdStore:         33.661.440 bytes
KV residente estable:           107.520 bytes
ahorro residente medido:              99,68 %
prefill:                    ~7,80 s / ~7,66 s
latencia total:             ~9,41 s / ~12,02 s
```

Las dos latencias corresponden a la corrida aplicada y a su control canónico;
no son suficientes para afirmar una mejora general, especialmente porque ambas
hicieron prefill completo y la corrida aplicada omitió cero V.

## Qué demuestra el código

### Llama CPU / LibTorch

`src/torch_runtime.rs` implementa `llama_certified_value_attention` para
`32Q/8KV`, `head_dim=128` y metadata completa de `max|V|` por cabeza KV. La
selección usa Q/K reales, scores FP32 y una cota local de masa softmax. Si la
geometría, metadata, scores o tolerancia no son válidos, la función devuelve
`None` y la ruta llama a SDPA BF16 completo.

La cota no demuestra igualdad de logits finales ni de tokens. El prefill
multi-token entra en `prefill_hot.context(...)` y usa atención completa; la
omisión no reduce el primer prefill.

### GPU OpenCL

`src/gpu_canonical_lod.rs` reutiliza el índice 4-bit/LSH, `ColdStore` y
`select_by_exact_mass`. No reutiliza todavía el branch-and-bound jerárquico
completo de la CPU. Con `epsilon=0` calcula la ruta exacta y conserva todos los
K/V; las microrregiones solo pueden activarse con epsilon positivo y deben
volver a QK completo cuando la cota es ambigua.

La prueba física de microrregiones controladas sí omitió `16/32` productos QK y
`16/32` valores V con `L∞≈2,07e-9` y tolerancia `1e-3`. En ese caso frío fue más
lenta que la atención completa; la variante hot medida fue `2,24x` más rápida.
Esto demuestra el mecanismo en una geometría controlada, no la omisión en el
modelo completo.

En el MiniCPM5-1B real de 375 tokens la prueba registró:

```text
GPU_micro_hot_sparse=0
GPU_micro_exact_fallback=360
logits idénticos=true
lecturas V frías durante decode=0
```

La prueba real del modelo, por tanto, verificó fallback exacto y hot-V, no
omisión positiva de regiones.

### Prefill y prefijos

La prueba GPU de lotes 128/375 mantuvo IDs, logits BF16 y SHA lógico KV. Esto
demuestra batching y paridad con la ruta GPU de referencia. No elimina el coste
cuadrático del primer prefill de un documento nuevo.

La prueba de prefijo compartido, con 355 tokens comunes de 375, obtuvo:

```text
full_prefill_s=2.597278
shared_prefill_s=1.410440
full_model_total_s=3.828671
shared_model_total_s=2.724560
full_logits_all_steps_identical=true
ColdStore_KV_SHA_identical=true
```

Esto demuestra reutilización verificada de un prefijo ya procesado, no una
solución para el primer análisis de un documento nuevo.

### Paralelismo LibTorch

`src/bin/terrsvusion-server.rs` llama a `InferencePool::spawn_many`. Cada
ejecutor carga su propio `TorchModelRuntime`. El pool prueba aislamiento y
concurrencia entre ejecutores, pero no varias inferencias compartiendo una
única autoridad LibTorch.

## No demostrado después de la auditoría

- Omisión positiva de V en una inferencia completa del checkpoint Llama bajo la
  autoridad normal del servidor.
- Igualdad de logits finales con omisiones positivas.
- Reducción general de latencia por omisión.
- Branch-and-bound jerárquico CPU completamente portado a GPU.
- Reducción del coste cuadrático del primer prefill.
- Equivalencia certificada a 30K, 100K o 131.072 tokens.
- RSS completo, workspace interno de LibTorch o VRAM total en las pruebas CPU.
- Varias inferencias sobre una única instancia LibTorch compartida.
- Equivalencia entre menos lecturas de `ColdStore` y menos operaciones de
  atención del Transformer.

## Corrección aplicada

`TerrsvusionKvCache::restore_prefix` era una envoltura usada solo por tests,
pero compilaba en producción como función muerta. Se marcó `#[cfg(test)]` para
que la implementación de runtime no conserve una ruta desconectada ni emita
un warning de código muerto. Las rutas de producción usan las variantes
geométricas/manifiesto que sí tienen callers.

