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

## Revalidación real desde la copia canónica del Escritorio

La ejecución se repitió después de añadir los contadores de trabajo lógico al
runtime actual, usando el binario construido desde este checkout:

```text
model=/home/yo/models/Llama-3.1-8B-Instruct
book_tokens=128
chunk_size=32
storage_frame_tokens=32
generate_tokens=1
applied_mass_tolerance=0.001
```

Resultado de la inferencia real:

```text
certified_v_tokens_omitted=4
attention_v_tokens_avoided=4
exact_token_parity=true
applied_token_ids=[31976]
canonical_token_ids=[31976]
max_abs_logit_difference=0.125
tolerance=0.001
applied_prefill_seconds=57.775381028
canonical_prefill_seconds=3.135374928
applied_cold_reads=18420
canonical_cold_reads=384
applied_exact_v_tokens_materialized=485
canonical_exact_v_tokens_materialized=0
attention_dense_qk_elements=0
attention_dense_pv_elements=0
hierarchy_k_tokens_avoided=0
```

Esta corrida cambia dos conclusiones anteriores: sí queda demostrada una
omisión V positiva dentro de una inferencia completa de Llama, con paridad del
token elegido en un caso real; pero no queda demostrada paridad de logits. El
`L∞=0.125` contradice la tolerancia `1e-3`, por lo que la ruta no puede
presentarse como equivalente exacta. Además, reducir lecturas y materialización
de V no redujo el trabajo del Transformer: los contadores de operaciones
densas no atribuyen ningún ahorro QK/PV y el prefill aplicado fue 18,4 veces
más lento en esta geometría. La instrumentación es contabilidad lógica del
runtime, no una prueba de kernels LibTorch fusionados.

Como corrección posterior, `generate_token_ids_with_applied_omission` ahora
rechaza el resultado cuando el `L∞` observado supera la tolerancia solicitada.
Por tanto, una omisión positiva que diverja como esta ya no se entrega como
resultado válido: queda como diagnóstico fallido y el contrato de aplicación
es fail-closed. Esto no crea paridad; impide que una falta de paridad se
publique como si la tuviera.

La barrera se verificó además ejecutando de nuevo el binario contra el mismo
checkpoint Llama 3.1 8B y los mismos parámetros. La ejecución terminó con:

```text
Error: omisión aplicada rechazada: L∞ de logits 0.125000000 supera la tolerancia 0.001000000
```

Esto confirma el comportamiento en runtime, no solo la compilación del guard.

La auditoría del caller HTTP encontró además que el servidor aceptaba antes
`applied_mass_tolerance` y llamaba directamente a la generación candidata,
sin ejecutar la comparación final de logits del binario de diagnóstico. Esa
puerta quedó cerrada: `validate_request` devuelve `422 Unprocessable Entity`
para cualquier petición con esa opción y `/v1/capabilities` la anuncia como
deshabilitada hasta disponer de paridad y fallback exactos. La ruta normal del
servidor continúa usando atención densa BF16.

La autoridad de contexto se volvió a comprobar contra el `config.json` real:
`max_position_embeddings=131072`. El binario rechazó antes de construir el
libro una petición de `book_tokens=131073` con:

```text
Error: el benchmark excede la ventana: 131073 + 1 > 131072
```

Esto demuestra que el límite operativo procede de la configuración del
checkpoint y que la comprobación de generación no conserva un `131072`
independiente oculto. No certifica una inferencia completa a 30K, 100K o
131.072 tokens; esas tres certificaciones siguen abiertas.

## Corrección aplicada

`TerrsvusionKvCache::restore_prefix` era una envoltura usada solo por tests,
pero compilaba en producción como función muerta. Se marcó `#[cfg(test)]` para
que la implementación de runtime no conserve una ruta desconectada ni emita
un warning de código muerto. Las rutas de producción usan las variantes
geométricas/manifiesto que sí tienen callers.

## Corrección posterior en la copia del Escritorio

El runtime tenía el valor `131072` repetido en las comprobaciones de generación,
verificación y atención densa. Se sustituyó por el `max_position_embeddings`
declarado por el `config.json` del modelo cargado y se exige que ese campo exista
al construir `MiniCpmModel`. Esto corrige un límite interno hardcodeado y evita
aceptar una longitud distinta de la declarada por el checkpoint.

Esta corrección no amplía el contexto ni convierte 131.072 en 3 millones: RoPE,
la memoria de atención y la capacidad del modelo siguen limitadas por el
checkpoint. Es una corrección de autoridad de configuración, no una certificación
de contexto largo.

## Prefill Llama con tolerancia positiva: primera ejecución completa

Se añadió una ruta opt-in de prefill causal token a token para Llama cuando la
petición incluye `applied_mass_tolerance`. La ruta usa la misma función de
certificación de V que el decode de un token; la ruta exacta predeterminada
continúa usando prefill por bloques.

Ejecución contra `/home/yo/models/Llama-3.1-8B-Instruct`:

```text
book_tokens=256
chunk_size=32
storage_frame_tokens=32
applied_mass_tolerance=0.001
certified_v_tokens_omitted=71
exact_token_parity=true
max_abs_logit_difference=0.125
prefill_seconds_applied=149.780652217
prefill_seconds_canonical=5.443281626
```

Esto cierra la afirmación de que la omisión positiva puede ejecutarse en el
prefill completo de Llama, pero no cierra la paridad de logits: el token fue el
mismo y los logits no fueron idénticos. También demuestra que la implementación
token a token es actualmente demasiado lenta en CPU; no se presenta como una
reducción de coste. La siguiente corrección necesaria es un certificador de
salida final o un fallback exacto antes de publicar una respuesta aproximada.
