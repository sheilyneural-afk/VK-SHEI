# Documentación de M-SLIM

Empiece por [ARCHITECTURE_FLOWS.md](ARCHITECTURE_FLOWS.md): conecta fuente,
release, evidencias, runtime, petición pública y recuperación mediante diagramas
de flujo y secuencia. Después consulte el contrato específico de la capa que
necesite.

- [README principal](../README.md): funcionamiento, estado comprobado y límites.
- [ARCHITECTURE_FLOWS.md](ARCHITECTURE_FLOWS.md): mapa integral de autoridad,
  ejecución, validación, calibración y continuidad.
- [VALIDATION.md](VALIDATION.md): matriz de cuatro rutas, gates same-backend y
  diagnóstico CPU/GPU.
- [MODEL_ADAPTERS.md](MODEL_ADAPTERS.md): contrato backend-neutral, catálogo y
  requisitos para añadir un adapter.
- [PREFILL_PROFILE.md](PREFILL_PROFILE.md): selector de perfiles de prefill.
- [ADAPTIVE_CALIBRATION.md](ADAPTIVE_CALIBRATION.md): aprendizaje auxiliar,
  medición física y activación transaccional del prefill.
- [ADAPTIVE_CALIBRATION_VALIDATION.md](ADAPTIVE_CALIBRATION_VALIDATION.md):
  evidencia histórica fechada; no describe el estado actual del runtime.
- [TILED_ATTENTION.md](TILED_ATTENTION.md): kernels opcionales de atención por
  teselas, contrato numérico, activación y evidencia registrada.
- [CONTINUITY_LIFECYCLE.md](CONTINUITY_LIFECYCLE.md): propiedad, generaciones,
  restore, release y reinicio total.
- [RETRIEVAL_EVALUATION.md](RETRIEVAL_EVALUATION.md): medición de la
  recuperación documental (`routing` frente a `fused`) y cómo reproducirla.
- [PENDIENTE.md](PENDIENTE.md): trabajo aplazado hacia producción y su motivo.
- [NAMING_STANDARD.md](NAMING_STANDARD.md): identidad y vocabulario técnico.
- [Entorno local](../deploy/local-production/README.md): layout, instalación,
  verificación y arranque.
