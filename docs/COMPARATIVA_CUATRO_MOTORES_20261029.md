# Comparativa de cuatro motores Terrsvusion V3 — 29/10/2026

> **Actualización posterior (29/10/2026):** la ruta GPU incorpora ahora prefill paralelo por lotes y ColdStore agrupado en páginas de hasta 64 tokens. Las cifras de GPU secuencial en este informe son HISTÓRICAS. Resultados nuevos en PREFILL_GPU_OPTIMIZADO_20261029.md.


Proyecto: /home/yo/Escritorio/terrsvusion-v3-rust-20261028-231043./terravision-v3/

Modelo: MiniCPM5-1B BF16, pesos originales. Equipo: yo-Venus-series, AMD Radeon Phoenix (OpenCL). Todas las pruebas son reales, con hardware físico.

## Advertencia: alias de la versión CPU en ejecución

La versión anterior del servidor CPU ignora el alias de modelo minicpm-dense-bf16 si no se añade native:true. La primera medición CPU densa estaba midiendo CPU LOD por error. Se detectó y repitió con native:true, comprobando que la respuesta HTTP devolvía el ID real del modelo denso y cero páginas ColdStore.

Los datos inválidos permanecen archivados en COMPARATIVA_MOTORES_HTTP_PRE_CORRECCION_ALIAS_20261029.json, y los válidos en COMPARATIVA_MOTORES_HTTP_20261029.json.

## Rendimiento de inferencia, sin incluir carga de pesos

| Prueba (tokens) | CPU densa | CPU LOD | iGPU densa | iGPU LOD |
|:--|--:|--:|--:|--:|
| Aritmética (39 entrada, 7 salida) | 0,485 s | 0,503 s | 3,693 s | 4,588 s |
| Fotosíntesis (42 entrada, 12 CPU/27 GPU) | 0,779 s | 0,941 s | 5,452 s | 6,954 s |
| Código CEDRO-7319 (375 entrada, 16 salida) | 1,508 s | 2,805 s | 32,363 s | 54,085 s |

Metodología: CPU HTTP desde el servidor residente 8099: 3 repeticiones para aritmética, 2 para fotosíntesis, 1 para el contexto. GPU en ejecutable Rust release aislado: 1 pasada por configuración y caso, primero denso y luego LOD, capacidad idéntica de 512 tokens, sin incluir carga del modelo. El mismo tokenizer local y los mismos IDs se verificaron entre las dos rutas GPU. Las entradas HTTP conservan iguales recuentos de tokens; el servidor CPU no expone los IDs completos, así que no se atribuye igualdad absoluta de IDs a HTTP solo por el recuento.

En fotosíntesis, la CPU generó 12 tokens y la GPU 27; esa comparación de latencia total no tiene igual cantidad de salida. Aritmética y CEDRO sí terminaron con la misma cantidad de salida en las cuatro rutas.

## Prefill (procesamiento de entrada)

| Prueba | CPU densa | CPU LOD | iGPU densa | iGPU LOD |
|:--|--:|--:|--:|--:|
| Aritmética | 0,090 s | 0,087 s | 3,021 s | 3,831 s |
| Fotosíntesis | 0,091 s | 0,084 s | 3,144 s | 3,975 s |
| Código CEDRO-7319 | 0,563 s | 0,478 s | 30,826 s | 51,158 s |

Diagnóstico: el prefill GPU sigue procesando tokens secuencialmente. El LOD GPU añade escritura ColdStore y lectura BF16 V a través de CPU/SSD con una página por token. Los kernels y el almacenamiento son reales, pero esa ruta no está optimizada para throughput.

## Comparación numérica con los mismos 15 IDs raw de Hola y 16 pasos greedy

- CPU LOD frente a CPU densa: 16/16 tokens idénticos, L-infinito logits = 0.0, KV SHA idéntico = True.
- iGPU LOD con epsilon=0 frente a iGPU densa: 16/16 tokens y todos los logits BF16 idénticos bit a bit en prueba física de regresión GPU.
- CPU densa frente a iGPU densa: 16/16 tokens idénticos, diferencia máxima logits = 0.625; KV SHA iguales = False.
- El primer KV diferente entre CPU y GPU aparece en la capa 0; total de elementos BF16 KV divergentes = 297,828, máxima diferencia KV = 0.41552734375.
- Coincidencia de tokens NO equivale a paridad numérica de atención ni a ausencia general de cambios semánticos entre CPU y GPU.

## Memoria de caché (no confundir con pesos ni VRAM total)

| Ámbito | Bytes | Qué significa |
|:--|--:|:--|
| Pesos BF16 MiniCPM5-1B | 2.161.265.664 | Intactos y necesarios en cada dispositivo |
| CPU densa: KV completo, caso 375 tokens | 9.584.640 | Buffer BF16 completo de referencia |
| CPU LOD: KV caliente, caso 375 tokens | 423.324 | Solo residencia de caché contabilizada, NO pesos ni memoria total |
| CPU LOD: KV almacenado en frío y metadatos | 9.638.940 | Datos recuperables en disco; sin pérdida |
| GPU densa: capacidad KV configurada 512 | 12.582.912 | Reserva K+V completa en iGPU |
| GPU LOD: capacidad KV configurada 512 | 6.332.736 | Reserva K BF16 + staging V BF16 + buffers auxiliares |
| GPU LOD: KV frío físico al finalizar contexto 375 | 11.232.000 | K/V originales y metadatos en ColdStore |

El ahorro GPU LOD es aproximadamente 49,67 % de RESERVA KV, no de la memoria total, ni de los 2,16 GB de pesos. El CPU LOD desplaza KV de RAM a disco y retiene metadatos; no afirma comprimir sin pérdidas todos los valores a 4 bits.

## Respuestas y exactitud semántica

- Aritmética: los cuatro motores respondieron 2 + 2 = 4.
- Fotosíntesis: los dos CPU respondieron una tautología; las GPU respondieron una explicación más larga con la palabra incorrecta «procesión». No es prueba de mejor calidad universal.
- Contexto 375 tokens: los cuatro motores recuperaron CEDRO-7319.

## Recomendación según prioridad

1. Velocidad: CPU densa fue la más rápida en los tres casos medidos. Su KV crece completo en RAM.
2. Memoria caliente reducida, contexto y funciones Terrsvusion completas: CPU LOD conserva branch-and-bound regional, índice 4-bit/LSH, ColdStore, certificados y prefijos persistentes; pagó latencia adicional en estas pruebas.
3. GPU con menor latencia: la GPU densa fue más rápida que GPU LOD, pero aún muy por detrás de CPU en el prefill.
4. GPU con menos reserva de KV: la GPU LOD ahorra aproximadamente la mitad, pero es más lenta por transferencias; no omitió V en las pruebas con epsilon=0.
5. Selección Auto actual: prioriza CPU LOD por política de compatibilidad, no cambia automáticamente a CPU densa en función de la velocidad. Puede pasar a GPU cuando CPU no esté disponible y la solicitud sea compatible.
6. La GPU aún no implementa el branch-and-bound regional completo ni los prefijos persistentes del CPU; comparte el índice 4-bit/LSH, el ColdStore exacto y el certificado terminal.

## Evidencias

- docs/COMPARATIVA_MOTORES_HTTP_20261029.json (medición CPU densa CORREGIDA).
- docs/COMPARATIVA_MOTORES_HTTP_PRE_CORRECCION_ALIAS_20261029.json (prueba errónea archivada).
- docs/COMPARATIVA_GPU_DENSO_VS_LOD_20261029.json.
- docs/COMPARATIVA_CPU_DENSO_GPU_DENSO_16_20261029.json.
- CPU LOD vs denso: docs/COMPARATIVA_CPU_LOD_DENSO_16_20261029.json, ejecutable terrsvusion-native-parity.

Servicios al finalizar: CPU 8099, GPU 8188, router unificado 8288. Configuración de GPU LOD restaurada: epsilon=0, staging=64, contexto máximo=2048. No se han modificado los pesos ni sustituido el servidor CPU.
