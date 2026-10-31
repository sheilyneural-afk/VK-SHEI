# Perfil adaptativo de prefill

`scripts/select-prefill-profile.sh` mide perfiles sin conocer el proveedor.

Variables:

- `KVSLIM_BENCH_CMD`: comando de benchmark del runtime.
- `KVSLIM_WEIGHT_MODES`: modos a probar (por defecto `stream cached resident`).
- `KVSLIM_PREFILL_BLOCK_CANDIDATES`: bloques (por defecto 64, 128 y 256).
- `KVSLIM_BENCH_TIMEOUT_SECS`: límite por candidato.
- `KVSLIM_BENCH_OUTPUT`: recibo JSON de salida.

El comando recibe:

- `KVSLIM_WEIGHT_MODE`
- `KVSLIM_PREFILL_BLOCK_TOKENS`

y debe devolver código cero solo cuando la medición sea válida. El selector no presupone OpenCL, Vulkan, CUDA, CPU ni ningún proveedor.

Ejemplo:

```bash
KVSLIM_BENCH_CMD='runtime-bench --prompt-file /tmp/prompt.txt' \
KVSLIM_PREFILL_BLOCK_CANDIDATES='64 128 256' \
scripts/select-prefill-profile.sh
```

El resultado se guarda en `deploy/local-production/runtime/validation/prefill-profile.json` y selecciona el perfil válido más rápido. Los candidatos que fallen o excedan el timeout no se activan.
