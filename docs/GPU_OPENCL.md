# Motor GPU (OpenCL)

Necesita la feature `opencl-backend`. El motor completo está en `opencl_llm.rs`,
`gpu_canonical_lod.rs` y `gpu_micro_regions.rs`, con los kernels en
`assets/opencl_llm.cl`. No usa LibTorch. Estado de `main` al 30/10/2026.

## Qué hace

- Embeddings, RMSNorm, QKV, RoPE, atención causal, MLP SwiGLU y LM head en la
  GPU, con pesos BF16 residentes. CPU y GPU se validan con los criterios de
  [CRITERIOS_VALIDACION_CPU_GPU.md](CRITERIOS_VALIDACION_CPU_GPU.md), no por
  igualdad bit a bit.
- **Modelos:** MiniCPM5-1B es el validado. También carga checkpoints con
  shards (`model.safetensors.index.json`) y RoPE `llama3`. Con Llama 3.1 8B
  solo hay inferencias mínimas por línea de comandos, más lentas que la CPU.
- **LOD canónico** (`gpu_canonical_lod.rs`): reutiliza la ColdStore y el eje de
  regiones de la CPU. Con epsilon=0 no omite nada: todas las consultas pasan
  por el respaldo exacto.
- `TERRSVUSION_OPENCL_DEVICE` filtra el dispositivo OpenCL.

## `terrsvusion-opencl-server`

```bash
cargo run --release --features opencl-backend --bin terrsvusion-opencl-server -- \
  --model /ruta/MiniCPM5-1B [--port 8188] [--allow-network]
```

Solo admite MiniCPM5 (usa su plantilla de chat). Si no se pasa `--model`, usa
la ruta por defecto que viene en el código (`/home/yo/models/MiniCPM5-1B`).

Rutas: `GET /` (UI `assets/chat-gpu.html`), `/health`, `/ready`, `/v1/models`,
`/v1/capabilities` y `POST /v1/chat/completions`.

### Valores por defecto

| Parámetro | Valor |
|---|---|
| Pesos | residentes en la GPU (`--stream-weights` los sube por capa) |
| Modo | LOD canónico con epsilon=0 y etapas de 64 tokens (`--lod-epsilon`, `--lod-stage`, `--lod-cold-dir`); `--gpu-dense-kv` usa la ruta densa |
| Lote de prefill | 128 (`--gpu-prefill-batch`, de 1 a 512; `--gpu-prefill-legacy` lo hace token a token) |
| Contexto máximo | 2048 tokens (`--max-context-tokens`, de 1 a 32768) |
| Tiempo límite | 300 s (`--timeout-secs`) |
| Red | solo `127.0.0.1`; cualquier otra dirección exige `--allow-network` |

### Aceleraciones opcionales (desactivadas por defecto)

Todas mantienen el K/V BF16 autoritativo. Si no se pueden aplicar, vuelven a la
ruta exacta.

| Opción | Qué hace |
|---|---|
| `--gpu-exact-fused-attention` | QK, softmax y PV exactos en un solo kernel (`--gpu-exact-original-attention` vuelve al original) |
| `--gpu-hot-v-tokens N` | ventana caliente de V en la GPU, de N tokens |
| `--gpu-micro-hot-v-tokens N` | microrregiones con V caliente; se ajustan con `--gpu-micro-novelty-sigma`, `--gpu-micro-warmup` y `--gpu-micro-open-fuse` |
| `--gpu-deferred-cold` | sella la ColdStore una vez al final del turno, no token a token (`--gpu-immediate-cold` para lo contrario) |
| `--gpu-fused-decode` | QKV y SwiGLU fusionados en la decodificación (`--gpu-unfused-decode` para lo contrario) |
| `--gpu-verified-prompt-replay N` | reutiliza el K/V verificado de un prompt idéntico de hasta N tokens (`--gpu-no-prompt-replay` lo desactiva) |
| `--gpu-verified-shared-prefix` | reutiliza el prefijo común verificado de un prompt parecido (`--gpu-no-shared-prefix` lo desactiva) |
| `--gpu-prefill-weight-reuse N`, `--gpu-decode-unroll4` | variantes de los kernels de prefill y de decodificación |

`terrsvusion-opencl-llm` admite las mismas opciones y añade las fusiones de
prefill (`--gpu-prefill-fused-qkv`, `--gpu-prefill-fused-swiglu`,
`--gpu-prefill-fused-projections`), `--verify-weights`, `--load-only` y
`--gpu-native-hipblaslt`. Ninguna está activa por defecto.

## Evidencia

Todas las mediciones son de MiniCPM5-1B BF16 en la iGPU AMD Radeon de Phoenix
(gfx1103, Mesa radeonsi), del 29/10/2026. Los tests citados son `#[ignore]` y necesitan la
iGPU y el checkpoint.

| Qué | Resultado | Evidencia |
|---|---|---|
| Pesos en la GPU | 219 tensores y 2.161.265.664 bytes BF16 releidos de la GPU, con el mismo SHA-256 que el checkpoint | [IGPU_FULL_MODEL_WEIGHT_READBACK](evidencia/gpu/IGPU_FULL_MODEL_WEIGHT_READBACK_20261029.json) |
| CPU LibTorch frente a GPU (prompt "Hola") | mismos tokens y misma decodificación voraz. Los logits difieren como mucho en 0,5 y el K/V no coincide bit a bit; es lo esperado según los criterios de validación | [IGPU_FULL_LLM_KV_DIAGNOSTICO_HOLA](evidencia/gpu/IGPU_FULL_LLM_KV_DIAGNOSTICO_HOLA_20261029.json) |
| LOD con epsilon=0 | 720 consultas, 0 certificados, 0 V omitidos: todo por respaldo exacto | [IGPU_CANONICAL_LOD_HOLA_EPS0](evidencia/gpu/IGPU_CANONICAL_LOD_HOLA_EPS0_20261029.json) |
| Prefill por lotes (375 + 16 tokens) | lotes de 64, 96, 128, 192, 256, 375 y 512: logits y K/V BF16 exactos frente a la referencia GPU de la prueba | [lotes 64–256](evidencia/gpu/GPU_PREFILL_BATCH_64_96_128_192_256_BF16_20261029.log), [lotes 375/512](evidencia/gpu/GPU_PREFILL_BATCH_ONE_SHOT_375_BF16_20261029.log), [resumen](evidencia/gpu/GPU_PREFILL_ONE_SHOT_375_SUMMARY_20261029.json) |
| Atención fusionada exacta | igualdad BF16 en todas las longitudes probadas; logits y SHA del K/V iguales con 16 tokens | [kernel](evidencia/gpu/GPU_FUSED_EXACT_ATTENTION_20261029.log), [densa 16](evidencia/gpu/GPU_FUSED_EXACT_DENSE_HOLA_PARITY_20261029.log), [LOD 375](evidencia/gpu/GPU_FUSED_EXACT_LLM_PARITY_375_20261029.log), [banco](evidencia/gpu/GPU_FUSED_EXACT_LLM_BENCH_375_20261029.json) |
| Microrregiones | omisión real de QK y de V frío con cota de error; con 375 tokens y sin certificado, todo por respaldo exacto y logits idénticos | [prueba física](evidencia/gpu/GPU_MICRO_HOT_PHYSICAL_MAIN_20261029.log), [respaldo 375](evidencia/gpu/GPU_FUSED_EXACT_MICRO_FALLBACK_375_20261029.log), [modelo 375](evidencia/gpu/GPU_MICRO_HOT_MODELO_375_20261029.json) |
| Sellado diferido (375 + 16) | 3,16 s frente a 3,28 s en total: un 4 % más rápido que el sellado inmediato | [GPU_DEFERRED_COLD_VS_IMMEDIATE](evidencia/gpu/GPU_DEFERRED_COLD_VS_IMMEDIATE_20261029.json) |
| Decodificación fusionada (375 + 16) | la decodificación con sellado baja de 1,47 s a 1,36 s (unas 1,08 veces más rápida) | [GPU_DECODE_FUSED_BF16_375](evidencia/gpu/GPU_DECODE_FUSED_BF16_375_20261029.json) |
| Repetición y prefijo verificados | prompt de 376 tokens que comparte 355 con otro: el prefill baja de 2,19 s a 1,14 s, con logits y K/V exactos | [prefijo](evidencia/gpu/GPU_VERIFIED_SHARED_PREFIX_GPU_PHYSICAL_20261029.log), [repetición](evidencia/gpu/GPU_VERIFIED_EXACT_PROMPT_REPLAY_PHYSICAL_20261029.log), [HTTP](evidencia/gpu/GPU_VERIFIED_PREFILL_REPLAY_HTTP_20261029.json) |
| GPU frente a CPU densa (HTTP, 375 + 16) | la GPU, con aceleraciones activadas, tardó unas 2,15 veces más que la CPU densa | [GPU_OPTIMIZADO_SERVIDOR_VS_CPU_DENSO](evidencia/gpu/GPU_OPTIMIZADO_SERVIDOR_VS_CPU_DENSO_20261029.json) |

Tests físicos relacionados en `opencl_llm.rs`:
`gpu_actual_canonical_lod_zero_matches_dense_gpu_outputs`,
`gpu_actual_prefill_64_128_192_256_bit_exact_full_logits_kv_and_speed`,
`gpu_actual_exact_hot_fused_attention_parity_for_many_context_lengths`,
`gpu_actual_micro_hot_model_375_bitwise_dense_fallback_and_cold_integrity`,
`gpu_actual_deferred_bf16_cold_commit_matches_immediate_and_indexes_every_token`,
`gpu_actual_verified_prompt_replay_restores_exact_bf16_logits_and_kv` y
`gpu_actual_verified_shared_prefix_375_vs_full_bf16_kv_and_logits`.

## Selector unificado (`terrsvusion-unified-server`)

Necesita la feature `unified-router`. Sirve `assets/chat-unified.html` y
reenvía cada petición a la CPU (por defecto `http://127.0.0.1:8099`), a la GPU
(por defecto `http://127.0.0.1:8188`) o al backend que elija en modo Auto.
Escucha en el puerto 8288 por defecto.

## HIP (`hip-backend`)

`native_hip.rs` carga en tiempo de ejecución
`native/libterrsvusion_hipblaslt_bridge.so`, compilada desde
`native/hipblaslt_bridge.cpp`. Es un puente experimental de matrices BF16 para
gfx1103. Solo se usa con `terrsvusion-opencl-llm --gpu-native-hipblaslt`; los
servidores no lo usan.
