# Auditoría de corrección — Llama 3.1 8B, V, alias y Faro

Fecha: 29/10/2026. Proyecto: Terrsvusion V3.
Backup bit a bit previo: /home/yo/Escritorio/terrsvusion-v3-rust-BACKUP-pre-llama-certified-v-and-faro-audit-20261029/

## Distinción obligatoria: KV residente frente a omisión certificada

El resultado de la prueba fría anterior (Llama 3.1 8B, 7.143 IDs de entrada idénticos, 8 pasos greedy con L-infinito de logits 0,0, sin prefijo persistido) sigue limitado a esa entrada y ese modelo.

La memoria KV residente de Terrsvusion de aproximadamente 5,2–5,4 MiB frente a 894–902 MiB densos, aproximadamente 99,4 % menos, corresponde a la GEOMETRÍA DE RESIDENCIA y ColdStore; NO significa que el certificado haya omitido V.

El código anterior montaba K y V completas cuando la geometría NO era 16Q/2KV: Llama 8B usa 32Q/8KV. Por tanto v_omit=0 no significaba un rechazo del certificador: el certificado exact_mass de MiniCPM ni siquiera se ejecutaba para Llama.

## Nueva ruta Llama 32Q8KV (experimental, opt-in)

Se ha añadido una ruta en src/torch_runtime.rs que calcula scores QK de BF16 real y aplica el selector matemático select_by_exact_mass para las 32 cabezas Q. Las ocho cabezas KV tienen ahora sus propias cotas max|V| en el certificado de página. Se preserva el campo original 2KV para MiniCPM; manifiestos antiguos sin las ocho cotas fallan hacia la atención BF16 completa.

En caso de omisión certificada, recupera solo las posiciones V seleccionadas mediante ColdStore::read_value_tokens, con valores BF16 y verificaciones originales. Sin certificado, con scores no finitos, con prefijo antiguo o con epsilon=0, utiliza la atención densa BF16 original. El certificado local de error de atención NO demuestra igualdad de logits finales ni exactitud universal.

El parámetro interno de diagnóstico `applied_mass_tolerance` acepta valores entre 0 (exclusivo) y 0,01 (inclusive), pero la API HTTP lo rechaza actualmente hasta demostrar paridad final de logits y fallback exacto. Por defecto sigue siendo None; la ruta llama-dense-bf16 nunca omite V. Toda ejecución experimental desactiva lectura y escritura de la caché persistente de prefijos para evitar contaminaciones entre políticas de inferencia.

Prueba física CONTROLADA en CPU con geometría Llama 32Q/8KV/128 dimensiones, K/Q/V sintéticos, LibTorch BF16 y ColdStore auténtico:

| Métrica | Resultado |
|---|---:|
| K puntuadas | 17 |
| V materializadas | 2 |
| V omitidas realmente | 15 |
| Lecturas ColdStore de V parciales | 1 |
| Bytes BF16 de V cuya lectura se evitó | 30.720 |
| Error observado salida BF16 frente a SDPA completa | 0 |
| Tolerancia de prueba | 0,001 |

Prueba además fallo seguro con certificado de prefijo antiguo, epsilon cero y consulta difusa. Registro: docs/LLAMA_32Q8KV_CERTIFIED_PHYSICAL_V_20261029.log.

Esto prueba la lógica física de la selección 32Q/8KV, NO demuestra omisión de V positiva en inferencia con los PESOS reales de Llama 8B. El último v_omit real previo sigue siendo 0, hasta una prueba nueva del checkpoint completo.

## Identificadores de modelo contrastados

El servidor CPU que carga el checkpoint Llama-3.1-8B-Instruct anuncia terrsvusion-rust y llama-dense-bf16. Una solicitud con minicpm-dense-bf16 respondió HTTP 404. Ese identificador pertenece a un servidor que cargue MiniCPM, no al Llama actual.

El nativo Llama con document= respondió HTTP 422: un documento inyectado como conversación Terrsvusion NO constituye la plantilla nativa virgen. Las pruebas de API validaron rechazo de tolerancia en rutas nativas y aceptación opt-in en Terrsvusion.

## Faro corto y volumen completo, medidos con tokenizadores reales

| Archivo | Bytes UTF-8 | Llama 8B raw tokens | MiniCPM5 raw tokens |
|---|---:|---:|---:|
| /tmp/faro-full.txt | 27.168 | 7.065 | 6.954 |
| assets/el-faro-volumen-completo.txt | 368.968 | 111.632 | 112.104 |

Para Llama con el tokenizador instalado, en orden (Faro corto / volumen completo):
- Texto raw: 7.065 / 111.632.
- Usuario nativo virgen: 7.075 / 111.642.
- Sistema Terrsvusion y usuario: 7.138 / 111.705.
- Sistema Terrsvusion, documento, respuesta de lectura y pregunta de ejemplo: 7.221 / 111.789.

Cada plantilla tiene token IDs y SHA distintos. La histórica entrada de 7.143 tokens fue otra secuencia, no identificable de manera fiable mediante solo un número de tokens.

Fuente reproducible sin inferencia pesada: scripts/audit_llama_faro_inputs.py.
Pruebas y SHA: docs/AUDITORIA_FARO_CORPUS_TOKENIZADORES_20261029.json y docs/AUDITORIA_LLAMA_FARO_IDS_Y_PLANTILLAS_20261029.json.

## Estado, límites y despliegue

Se verificaron 76 tests torch CPU, 9 tests API y 55 tests Rust core. Las pruebas adicionales del certificado sintético, del prefijo aislado y del fallo seguro también pasaron.

El nuevo ejecutable fue recompilado en modo release después de las modificaciones del código. El proceso 8099 no fue reiniciado para no interrumpir trabajos simultáneos: el API experimental nuevo no está ACTIVO en el servidor anterior hasta un reinicio controlado. Los procesos GPU y router no se editaron.

## Verificación de paridad por bloque canónico (29/10/2026)

La ruta experimental posterior conserva el denominador softmax completo: calcula QK para toda la historia y aplica PV únicamente sobre las posiciones V seleccionadas, sin renormalizar la distribución. Para evitar una falsa autoridad numérica, el oráculo no compara contra llamadas SDPA de una sola consulta. LibTorch puede escoger para esas llamadas otro kernel u otro orden de acumulación. La autoridad es ahora una sola SDPA densa del bloque completo, con la misma geometría batched y la misma máscara causal que el prefill canónico.

Ejecución física con el checkpoint `/home/yo/models/Llama-3.1-8B-Instruct`, 128 tokens, bloques y páginas de 32 tokens, un token generado y tolerancia experimental `1e-3`:

| Evidencia | Resultado |
|---|---:|
| Salidas de capa comparadas | 128 |
| Primera divergencia | ninguna |
| L-infinito máximo de capa | 0 |
| L-infinito de logits finales | 0 |
| Eventos con candidato de omisión | 11 |
| Posiciones V candidatas | 20 |
| Eventos de atención exactamente iguales | 0 |
| Posiciones V rechazadas por el oráculo | 20 |
| Posiciones V autorizadas como omitidas | 0 |

El resultado corrige el fallback y demuestra paridad exacta del diagnóstico, pero refuta que la cota local `1e-3` baste para una omisión positiva exacta en esta entrada: las 20 posiciones propuestas cambiaban al menos un bit de la salida de atención. El oráculo devuelve el bloque denso canónico y restaura todos los contadores de ahorro. Por ello `omission_accepted=false`, aunque las lecturas parciales exploratorias sigan visibles como trabajo diagnóstico realmente ejecutado.

El coste también queda separado: la ruta canónica necesitó 384 lecturas ColdStore y 2,29 s de prefill; el diagnóstico exhaustivo realizó 6.560 lecturas y 7,11 s. Esa diferencia no es una aceleración ni una regresión del servidor normal: corresponde a puntuar candidatos consulta por consulta y verificar una vez cada bloque contra la autoridad densa. Antes de escalarlo debe sustituirse por una garantía suficiente por capa —por ejemplo, una celda de redondeo BF16 con cotas de error aritmético— que pueda rechazar o autorizar sin leer V omitida ni ejecutar después el bloque denso.

La revisión posterior elimina la reconstrucción de K por consulta: cada capa ensambla K una vez por bloque, calcula QK batched una vez y entrega a cada consulta únicamente su fila causal. En la misma prueba física, las lecturas del diagnóstico bajaron de 6.560 a 416 y los bytes leídos de aproximadamente 430 MB a 27,2 MB; la autoridad canónica registró 384 lecturas y 25,2 MB. El prefill diagnóstico bajó a 6,69 s frente a 2,62 s canónicos. El sobrecoste residual ya no corresponde a I/O repetido ni a QK repetido: son las decisiones, softmax y PV pequeños por consulta, además del oráculo denso.

También se incorporó una caja de error por componente y una prueba estricta de celda de redondeo BF16. Las páginas nuevas guardan `max|V|` por las 8 cabezas KV y sus 128 dimensiones; prefijos antiguos sin ese campo fallan hacia denso. En esta ejecución se comprobaron 11 candidatas y cero cajas quedaron íntegramente dentro de las 4.096 celdas BF16, coincidiendo con el oráculo, que también rechazó las 11. Esta condición permanece diagnóstica: todavía falta cerrar una cota formal que cubra todos los errores del score, softmax y PV del kernel SDPA fusionado antes de convertirla en autoridad del servidor.

Continúa sin demostrarse: omisión positiva exacta usando el checkpoint real Llama 8B; una garantía previa sin lectura de V omitida; paridad física 30K/100K/131K; paridad 111k sobre Faro completo; ni equivalencia automática entre chats con `document=`/`system` y el nativo virgen. Los SHA de representaciones KV diferentes tampoco deben compararse como hashes binarios iguales sin una canonización lógica definida.
