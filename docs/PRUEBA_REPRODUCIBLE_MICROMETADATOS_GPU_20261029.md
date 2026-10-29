# Prueba reproducible — Terrsvusion V3, micrometadatos GPU

Ejecutada en yo-Venus-series, AMD Radeon Phoenix1, 29/10/2026 15:25 (Europe/Madrid).

## Prueba física GPU con K/V BF16

| Métrica | Resultado comprobado |
| --- | --- |
| Tokens K totales | 32 |
| Tokens K calculados realmente en la Radeon | 16 |
| Productos QK evitados | 32 de 64 (50 %) |
| V BF16 recuperadas desde el ColdStore | 16 de 32 |
| V BF16 no recuperadas | 16 de 32 (50 %) |
| Límite certificado por cabeza | [4.214002949874172e-9, 4.214002949874172e-9] |
| Error máximo GPU real BF16 | 0.000000002066372 |
| Tolerancia configurada | 0.001 |
| Resultado del test | 1 passed; 0 failed |

Salida BF16 GPU completa: [2.066372e-9, 1.0, 0.0, 0.0, 2.066372e-9, 1.0, 0.0, 0.0]

Salida BF16 GPU micro: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

Comando desde el proyecto:

    TERRSVUSION_OPENCL_DEVICE=AMD cargo test --locked --offline --features opencl-backend gpu_actual_micro_regions_skip_real_qk_and_cold_v_with_error_bound -- --ignored --nocapture

Log de ejecución: docs/PRUEBA_REAL_MICROMETADATOS_GPU_20261029.log

Contrapruebas de región ambigua, epsilon=0 y entrada no finita: 3 tests superados.
Log: docs/PRUEBA_REGIONES_MICRO_FALLBACK_20261029.log

## Prueba con MiniCPM5-1B real

Servidor Radeon BF16, mismo modelo original. Con epsilon=0, 375 tokens de entrada + 16 de salida, HTTP 200.

Respuesta: El código del dato verificable es **CEDRO-7319**.

Prefill: 1.877 s.
Inferencia total: 3.055 s.
Micro regiones exactas: sin omisiones por definición.
ColdStore exacto: 168 páginas, 9761280 bytes.
Índice original cuatro bits: 4680 bytes.

En pruebas anteriores con epsilon=0,001 se omitieron 469 accesos V acumulados, pero cero QK regionales en este contexto. La velocidad fue 4.612 s (peor que epsilon=0). No se ha demostrado todavía un aumento de velocidad general con los micrometadatos en el modelo completo.

Evidencia: docs/PRUEBA_MODELO_REAL_MICROMETADATOS_EPS0_20261029.json y docs/MICROMETADATOS_GPU_EPSILON_375_20261029.json

## Integridad

Los archivos principales del motor CPU y Cargo.lock se verificaron idénticos respecto al backup pre-micro-metadata. Los servicios CPU (8099), GPU (8188) y router (8288) devolvieron HTTP 200.

La mejora del 50 % en QK y V es específica de la entrada BF16 sintética controlada. No implica un 50 % de aceleración del modelo ni un funcionamiento sin aproximación cuando epsilon es positivo.
