# Perfil adaptativo de prefill

La ruta integrada de medición, predictor y promoción está documentada en
[ADAPTIVE_CALIBRATION.md](ADAPTIVE_CALIBRATION.md). Lo descrito a continuación
es el selector shell manual: su salida no autoriza la activación del runtime.

`scripts/select-prefill-profile.sh` mide perfiles sin conocer el proveedor.

Variables:

- `MSLIM_BENCH_CMD`: comando de benchmark del runtime.
- `MSLIM_WEIGHT_MODES`: modos a probar (por defecto `stream cached resident`).
- `MSLIM_PREFILL_BLOCK_CANDIDATES`: bloques (por defecto 64, 128 y 256).
- `MSLIM_BENCH_TIMEOUT_SECS`: límite por candidato.
- `MSLIM_BENCH_OUTPUT`: recibo JSON de salida.

El comando recibe:

- `MSLIM_WEIGHT_MODE`
- `MSLIM_PREFILL_BLOCK_TOKENS`

y debe devolver código cero solo cuando la medición sea válida. El selector no presupone OpenCL, Vulkan, CUDA, CPU ni ningún proveedor.

Ejemplo:

```bash
MSLIM_BENCH_CMD='runtime-bench --prompt-file /tmp/prompt.txt' \
MSLIM_PREFILL_BLOCK_CANDIDATES='64 128 256' \
scripts/select-prefill-profile.sh
```

El resultado se guarda en
`deploy/local-production/runtime/validation/prefill-profile.json`. El campo
`selected` contiene el candidato válido más rápido de esa ejecución. El script
no modifica `local-production.env`, no instala binarios y no activa el perfil;
el consumidor debe leer el recibo y aplicar la selección de forma explícita.
Los candidatos que fallen o excedan el timeout quedan registrados con
`ok: false`.

La lista por defecto incluye `cached` porque el selector es neutral respecto al
proveedor. El runtime GPU actual solo acepta `stream` y `resident`; si se usa
como comando de benchmark, `cached` debe eliminarse de `MSLIM_WEIGHT_MODES` o se
registrará como candidato fallido.
