# Documentación de VK-SHEI / Terrsvusion

Refleja el código de `main` al 30/10/2026. Cada documento distingue lo que está
implementado de lo que se ha medido, y cada medición enlaza con su archivo de
evidencia.

| Documento | Contenido |
|---|---|
| [ARQUITECTURA.md](ARQUITECTURA.md) | Módulos, ColdStore, índice de 4 bits, certificado, concurrencia y variables de entorno |
| [CPU_LIBTORCH.md](CPU_LIBTORCH.md) | Motor CPU: MiniCPM5-1B, Llama 3.1 8B, `terrsvusion-server`, `terrsvusion-long-context`, prefijos persistentes y estado de la omisión certificada |
| [GPU_OPENCL.md](GPU_OPENCL.md) | Motor OpenCL: servidor, valores por defecto, aceleraciones opcionales, mediciones en iGPU, selector unificado y HIP |
| [DOCUMENTOS.md](DOCUMENTOS.md) | Grafo documental, herramientas, `POST /v1/document/query` y El Faro |
| [CRITERIOS_VALIDACION_CPU_GPU.md](CRITERIOS_VALIDACION_CPU_GPU.md) | Cómo leer las comparaciones CPU/GPU, LOD y velocidad |

## Evidencia

Los archivos de `evidencia/` son los resultados tal como salieron de cada
ejecución y no se han editado. Pueden contener rutas locales de la máquina
donde se midieron.

- `evidencia/cpu/`: Llama 3.1 8B en CPU (omisión de V, prefill por bloques,
  publicación de prefijos).
- `evidencia/gpu/`: MiniCPM5-1B en la iGPU (pesos, LOD, prefill por lotes,
  atención fusionada, microrregiones, prefijos verificados, comparación con la
  CPU).
- `evidencia/documentos/`: grafo de El Faro, recuperación por fases, endpoint
  documental y auditoría de tokenización.

Los informes de sesión y las mediciones de rutas que ya no existen o que se
han superado se quitaron en la limpieza del 30/10/2026. Se pueden recuperar
desde la historia de git (commit `7f4fe0f` y anteriores).
