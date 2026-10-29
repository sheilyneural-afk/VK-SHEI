# Terrsvusion v3

Estado verificado del código a 28 de octubre de 2026.

Terrsvusion es un runtime experimental escrito en Rust para ejecutar
MiniCPM5-1B con pesos y cálculo BF16, un catálogo dimensional compacto de
cuatro bits y un backing K/V BF16 append-only. Los cuatro bits solo sirven para
routing: no transforman los pesos, no sustituyen los K/V y no constituyen una
segunda autoridad numérica.

El proyecto demuestra actualmente reducción fuerte de la residencia estable
del KV contabilizado y recuperación exacta desde backing frío. Todavía no
demuestra aceleración general: con la tolerancia local vigente, el selector
termina recuperando todos los estados en las pruebas realizadas.

## Estado ejecutivo

| Propiedad | Estado actual |
| --- | --- |
| Runtime | Rust 2024 + `tch`/LibTorch CPU |
| Modelo probado | MiniCPM5-1B local |
| Pesos | Originales; no se genera un modelo Q4 |
| Cálculo del modelo | BF16 |
| Autoridad K/V recuperada | BF16 exacta |
| Índice auxiliar | Fingerprints firmados de cuatro bits + SimHash LSH |
| Almacenamiento | Archivo append-only por petición + prefijos persistentes verificados |
| Integridad | Checksums rápidos, hashes por token y SHA-256 por registro |
| Prefill | Atención causal BF16 completa |
| Decode | Routing dimensional + branch-and-bound + certificado de masa |
| Omisión certificada observada | 0 tokens en la prueba real descrita abajo |
| Reutilización entre peticiones | Prefijo estable de mensajes anteriores; comprobada tras reinicio |
| Servidor | Axum/Tokio; una autoridad de inferencia y cola acotada |
| Tests actuales | 41 correctos |

## Arquitectura canónica

El crate contiene una biblioteca y tres ejecutables:

- `terrsvusion-server`: servidor HTTP Axum.
- `terrsvusion-torch-chat`: inferencia local desde línea de comandos.
- `terrsvusion-long-context`: benchmark, documentos largos, estrés y control
  BF16 denso opcional.

Módulos de la biblioteca:

- `torch_runtime.rs`: carga de `safetensors`, tokenizer, MiniCPM5, atención,
  caché Terrsvusion, control denso y métricas.
- `adaptive_extent.rs`: estados dimensionales definidos por contenido,
  fingerprints de cuatro bits, micrometadatos y routing LSH.
- `attention_certificate.rs`: cotas de masa omitida y selección exacta por
  scores QK.
- `cold_store.rs`: formato append-only, recuperación, verificaciones y lecturas
  parciales K/V.
- `conversion_worker.rs`: trabajadores acotados para preparar y ordenar commits
  BF16.
- `inference_worker.rs`: hilo propietario del modelo, cola acotada y
  backpressure.
- `prefill_tch.rs`: proyección protegida de cada cabeza Q hacia su cabeza KV
  servidora para el routing. No implementa por sí solo la atención de prefill.

No hay runtime Python, Candle, servidor alternativo ni selector heredado en el
árbol actual. LibTorch procede de la instalación local de PyTorch, pero el
proceso de Terrsvusion no ejecuta un intérprete Python.

### Flujo de servicio

```text
HTTP o CLI
  -> TorchModelRuntime
  -> MiniCpmModel
  -> TerrsvusionKvCache
  -> catálogo dimensional de cuatro bits
  -> branch-and-bound y certificado de masa
  -> lecturas parciales BF16 del backing
  -> atención SDPA de LibTorch
```

`generate_token_ids_dense` permanece como oráculo de benchmark. No es una ruta
del servidor.

## Estados dimensionales y marcos de almacenamiento

Un estado dimensional es una región lógica variable detectada a partir del K
que ya produce el modelo. Sus límites responden a sorpresa respecto al
centroide del estado abierto y al cambio temporal frente al token anterior.
Los intervalos token son direcciones reversibles hacia los K/V del modelo; no
son una segunda tokenización.

Los estados lógicos y los marcos físicos son independientes:

- un estado puede atravesar varios marcos;
- un marco puede contener fragmentos de varios estados;
- selección y certificación operan sobre estados;
- las lecturas finales operan sobre coordenadas token exactas.

`--storage-frame-tokens` controla únicamente la granularidad física de IO. El
runtime acepta potencias de dos entre 16 y 512; el valor normal es 512.

## Catálogo de cuatro bits y LSH

Cada token obtiene un fingerprint compacto con dígitos firmados `[-7, 7]`, dos
coordenadas por byte y una escala local. Este catálogo es metadata de búsqueda;
los K/V recuperados continúan siendo BF16.

El índice LSH usa cuatro tablas, proyecciones deterministas y firmas compuestas
por 16 bits direccionales más bandas de magnitud y dispersión. Una consulta
prueba vecindad Hamming 2 y después 4, y exige votos de al menos dos tablas.

LSH no es una autoridad de recall. Si no devuelve candidatos se incrementa
`parallel_lsh_misses` y se entregan cero semillas. El branch-and-bound canónico
continúa desde el estado reciente y amplía solo cuando su cota lo exige. No
existe un fallback que cargue el catálogo completo.

Los fingerprints por token, las firmas por extensión y el grafo de
micrometadatos son señales auxiliares que se solapan parcialmente. Su contrato
actual es único: priorizan candidatos, pero no autorizan por sí mismos una
carga u omisión. En particular, las dimensiones comunicantes ya no se añaden
todas al conjunto inicial; solo desempatan el orden de ampliación. La versión
anterior podía convertir relaciones densas en una carga inicial completa.

La prueba actual mostró 109 misses en 112 consultas. Separar el índice por
cabeza KV y revisar la asimetría de escala Q/K siguen pendientes; reducir esos
misses puede mejorar el orden de exploración, pero no autoriza omisiones por sí
solo.

## Certificado de atención

Durante decode, Terrsvusion:

1. Siembra estados candidatos mediante routing dimensional.
2. Conserva siempre el estado reciente.
3. Calcula cotas superiores QK para los estados omitidos.
4. Materializa un estado adicional cuando la cota no cierra.
5. Con scores exactos, selecciona el menor conjunto cuya cota de error por
   componente cumple la tolerancia.
6. Recupera los V seleccionados y ejecuta SDPA BF16.

La materialización K durante una atención es incremental: cada región abierta
se lee como máximo una vez del cold store para ese token, aunque el
branch-and-bound necesite varias rondas. Las rondas posteriores reutilizan el
tensor local ya verificado y lo liberan al terminar esa atención. Esto elimina
la antigua amplificación por relectura sin cambiar regiones, scores, tolerancia
ni orden causal.

Los scores exactos también se conservan por región durante esa llamada. Una
ampliación calcula únicamente los scores de la nueva región; la concatenación
tensorial final se realiza una vez. Si una pasada lineal demuestra que ni el
token individual más barato puede omitirse con la tolerancia restante, se
evitan las 16 ordenaciones por cabeza y se toma directamente el conjunto denso.
Los intervalos contiguos residentes usan vistas `narrow` y las páginas completas
usan `shallow_clone`; `index_select` queda reservado para coordenadas realmente
dispersas. En el cold store, `read_*_tokens` ya agrupa rangos contiguos por
cabeza y deriva automáticamente una selección `(0..N)` a la lectura completa.

La tolerancia actual está fijada en `1e-6`. La cota mantiene separadas la masa
softmax omitida y la magnitud `max|V|`. No existe todavía una demostración que
propague esa cota local a través de todas las capas hasta el margen de logits
final.

Las regiones K usan cajas alineadas a ejes. Son seguras, pero pueden construir
un máximo virtual combinando extremos de dimensiones pertenecientes a tokens
distintos. Esa laxitud geométrica sigue presente. Una prueba centro-radio fue
retirada porque no evitó estados y aumentó los metadatos; no permanece código
experimental de esa vía.

## Prefill y ciclo de vida K/V

El prefill continúa usando atención causal BF16 completa. Para cada capa se
mantiene temporalmente el K/V BF16 necesario mientras los marcos cerrados se
preparan en trabajadores Rust. Al terminar el prefill:

1. se sellan las colas parciales;
2. se drenan todos los trabajos aceptados;
3. se audita el backing mediante un mmap inmutable temporal;
4. se libera el conjunto BF16 temporal de prefill;
5. decode recupera fragmentos BF16 desde el backing.

Esto reduce la residencia estable posterior, pero no elimina el coste
cuadrático del primer prefill ni su working set temporal.

El servidor reutiliza ahora un prefijo estable entre peticiones. Cuando hay al
menos dos mensajes, todos salvo el último forman el prefijo cacheable; el
último mensaje es el sufijo o pregunta variable. La clave SHA-256 incorpora la
identidad criptográfica de pesos y tokenizer, los ids exactos del prefijo y la
geometría de almacenamiento. El manifiesto conserva certificados, páginas y
eje dimensional. La publicación usa staging y rename atómico. En un hit se
copia el backing inmutable a una caché de trabajo, se reconstruye el catálogo,
se verifica criptográficamente y se rehidrata K/V BF16 antes de procesar solo
el sufijo. El decode de una petición nunca modifica el prefijo persistido.

Esto elimina el cómputo Transformer repetido del prefijo, no toda su I/O ni la
rehidratación BF16. Tampoco acelera una petición de un solo mensaje, porque no
existe una frontera fiable entre documento y pregunta. Para reutilizar un
libro, debe enviarse como mensaje anterior (por ejemplo `system`) y la pregunta
como último mensaje `user`.

La preparación del backing sí está fuera del hilo de inferencia: cálculo de
hashes por token, checksums, SHA-256 y ensamblado completo del registro se hacen
en un worker acotado; el hilo propietario solo hace el append ordenado. Se usa
un único worker para evitar competir con los hilos de LibTorch durante el
prefill. Las métricas `prefill_*_nanos` separan eje dimensional, concatenación
caliente, certificados, handoff y commits.

## Almacenamiento e integridad

`ColdStore` usa registros append-only con:

- geometría y offsets explícitos;
- checksums rápidos de K, V y registro;
- hash por token para validar lecturas parciales;
- digest SHA-256 persistido por registro;
- footer de commit;
- `page_id` monótono;
- recuperación de una cola final incompleta al reabrir.

La auditoría completa mapea temporalmente el archivo, verifica los registros en
paralelo y libera el mapa al terminar. La inferencia no conserva un mmap de K/V
como caché residente; usa lecturas parciales explícitas. Integridad de bytes no
equivale a relevancia semántica: esta última sigue dependiendo del routing y del
certificado.

## Oráculo exacto de omisibilidad

`--exact-oracle` activa además un diagnóstico cuando todas las regiones ya
fueron cargadas. `--compare-dense` solo ejecuta el control BF16 y no contamina
la comparación de latencia con este trabajo adicional. Con los scores QK
exactos, el oráculo calcula:

- oportunidades token/evento observadas;
- tokens omitibles a `1e-6`, `1e-5`, `1e-4` y `1e-3`;
- tolerancia mínima observada para omitir un token simultáneamente en todas las
  cabezas;
- margen mínimo top-1/top-2 de logits durante la generación.

El oráculo no gobierna la inferencia. Está desactivado en el servidor y en la
ruta normal porque ordenar scores a cuatro tolerancias es costoso. Si en el
futuro la geometría consigue omitir regiones, el oráculo deja de muestrear ese
evento en vez de volver a leer todo y destruir la aceleración.

Todavía no ejecuta una rama shadow con omisiones aplicadas, por lo que no mide
el error real de atención ni certifica paridad de tokens para tolerancias
alternativas. El margen de logits es observacional; no es una licencia para
relajar la cota.

## Instrumentación shadow de solo lectura

`--shadow-metrics` activa un informe separado de `TorchGeneration`. El módulo
`shadow_metrics.rs` solo agrega escalares; no posee tensores, no abre el cold
store, no selecciona estados y nunca autoriza omisiones. Las muestras se toman
únicamente cuando la ruta canónica ya cargó todas las regiones y dispone de los
scores exactos. Por eso el informe declara `additional_cold_reads: 0` y no se
activa en el servidor.

El informe contiene:

- distribución nearest-rank del umbral mínimo para omitir un token individual;
- holgura por región/cabeza entre la cota log-masa AABB y el máximo regional
  exacto más `ln(token_count)`;
- identidad del evento mínimo y máximo;
- cualquier violación donde AABB quede por debajo del score exacto;
- error local real de atención para selecciones exactas experimentales a
  `1e-4` y `1e-3`, comparado con `2 * alpha * max|V|`;
- limitaciones explícitas. Las omisiones solo se simulan en la salida local de
  atención: todavía no se propagan estados shadow hasta logits.

Ejecución:

```bash
cargo run --release --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --book-tokens 512 \
  --chunk-size 128 \
  --storage-frame-tokens 128 \
  --generate-tokens 2 \
  --compare-dense \
  --shadow-metrics
```

Resultado real de esa geometría:

| Métrica shadow | Resultado |
| --- | ---: |
| Eventos de umbral individual | 24 |
| Umbral mínimo | `2,3360073602400125e-5` |
| Umbral p50 | `6,37478211878966e-5` |
| Umbral p90 | `1,2052335068237947e-4` |
| Umbral máximo | `2,0510393124897964e-4` |
| Comparaciones AABB exactas | 2.304 |
| Holgura log-masa AABB mínima | `0` |
| Holgura p50 | `18,555361760541164` |
| Holgura p90 | `27,091711965358204` |
| Holgura máxima | `42,49762832275392` |
| Violaciones AABB | 0 |
| Tokens omitidos a `1e-4` | 1 |
| Cabezas afectadas a `1e-4` | 16 de 384 |
| Error real máximo a `1e-4` | `1,0886421543357394e-5` |
| Ratio cota/error p50 a `1e-4` | `2,6748551764931374` |
| Violaciones de cota a `1e-4` | 0 |
| Tokens omitidos a `1e-3` | 21 |
| Cabezas afectadas a `1e-3` | 160 de 384 |
| Error real máximo a `1e-3` | `6,53646364726923e-5` |
| Ratio cota/error p50 a `1e-3` | `3,2957461035628026` |
| Violaciones de cota a `1e-3` | 0 |
| Lecturas frías canónicas | 240 |
| Lecturas frías shadow añadidas | 0 |
| Paridad de ids BF16 | `true` |

Esta corrida demuestra que las AABB son muy laxas, pero también que el umbral
individual exacto permanece muy por encima de `1e-6`. Mejorar solo la geometría
K no basta para crear una omisión certificable bajo la cota V actual.
La cota V sobrestimó el error local real por aproximadamente `2-5x` en las
cabezas con omisión de esta corrida. Eso justifica estudiar una cota V más fina,
pero no demuestra seguridad hasta logits ni autoriza cambiar `1e-6`.

Una línea base posterior con 1.024 tokens, ocho tokens solicitados, chunks y
marcos de 256 registró 168 eventos de capa shadow. El mínimo global de
`minimum_individual_tolerance` fue `2,783061069137239e-6`; por tanto, la
condición de bypass `1e-6 < minimum_individual_tolerance` se cumplió en el 100 %
de los eventos observados. La ruta O(N) entregó directamente el conjunto denso
sin ejecutar las 16 ordenaciones por evento. Se conservaron la respuesta,
paridad exacta, 816 ampliaciones y cero tokens evitados. El informe de referencia
se guardó durante la ejecución en `/tmp/terrsvusion-baseline-shadow-1024.json`;
es evidencia temporal local, no un artefacto versionado.

La ruta canónica posterior sustituyó el trabajo repetido de branch-and-bound
por un certificado regional incremental. Las prioridades se ordenan una vez;
cada región K se materializa y puntúa una vez; y las masas conservada y omitida
se actualizan en un marco exponencial fijo. La cota incluye además el máximo de
V de la cola caliente, que antes participaba en los scores conservados pero no
en ese máximo. Si una cota regional queda por debajo de la masa exacta observada,
la ejecución falla cerrada.

Una corrida real posterior, con la misma geometría de 1.024 tokens, ocho tokens
generados y comparación BF16 densa, produjo exactamente los mismos ocho ids y
la misma respuesta `AZULEJO-7319`. Terrsvusion conservó las 816 ampliaciones y
cero tokens omitidos, pero registró `1,269 s` de decode y `2,865 s` de latencia
total. Frente a la corrida anterior documentada (`~1,8 s` de decode y `3,389 s`
total), supone aproximadamente un 29,5 % y un 15,5 % menos, respectivamente.
Son corridas individuales, no una distribución estadística; demuestran que el
trabajo retirado era real, no una aceleración universal ni omisión certificada.

La instrumentación interna de una segunda corrida comparable atribuyó los
`1,217 s` de decode a `168 ms` en SDPA/ensamblado interno, `132 ms` en
materialización K, `123 ms` en scores exactos, `93 ms` en materialización V,
`37 ms` en metadatos, `14 ms` en ensamblado K y solo `0,114 ms` en actualizar
la cota regional. El resto, aproximadamente `649 ms`, pertenece al Transformer
y al control no cubierto por esos contadores. La cota incremental ya no es el
cuello dominante; puntuar y materializar todo antes de SDPA sí lo es.

También existe un oráculo diagnóstico `--multitoken-oracle K`. Dos cachés BF16
densas independientes ingieren el mismo prompt: una verifica K candidatos con
K forwards de un token y la otra con un único forward causal. Con 1.024 tokens
se observaron speedups de `2,07x`, `4,04x` y `7,14x` para K=2, 4 y 8, con los
mismos argmax en el ejemplo. Sin embargo, para K=4 el error máximo de logits fue
`0,25` y los SHA-256 del KV final fueron distintos, aunque tokens y tamaño
coincidían. Por tanto `canonical_commit_safe` es falso: el resultado demuestra
potencial de verificación paralela, pero prohíbe reutilizar ese KV batch bajo el
contrato actual de paridad residual exacta.

La divergencia se localizó después con precisión. La capa 0 conservó K/V
idénticos; la primera diferencia apareció en la capa 1 y alcanzó 23 de 24
capas. De 49.152 elementos BF16 candidatos, diferían 15.555 elementos K y
19.442 V. Ejecutar con un solo hilo BLAS/OpenMP produjo exactamente los mismos
digests divergentes, descartando una carrera: la causa era la geometría de la
atención multi-query de LibTorch.

El modo diagnóstico `--multitoken-oracle-querywise K` conserva las proyecciones
y MLP en bloque, pero ejecuta cada consulta de atención con la misma longitud y
forma de la ruta secuencial. Esta variante produjo error de logits 0, cero
elementos K/V distintos y SHA-256 idéntico en las 24 capas. Los speedups reales
fueron `1,70x` para K=2, `2,62x` para K=4 y `3,21x` para K=8; en los tres casos
`canonical_commit_safe` fue verdadero. Esto prueba que el backend puede
amortizar una parte importante del Transformer sin cambiar la autoridad BF16,
pero todavía no constituye generación especulativa: faltan un drafter real,
aceptación/rollback y una ruta paginada que no rehidrate todo el prefijo.

El benchmark `--reference-speculative` cubre el siguiente escalón sin introducir
otro modelo: un drafter busca continuaciones del sufijo generado dentro del
prompt, propone dinámicamente entre 2 y 8 tokens y el target BF16 querywise es
la única autoridad. El KV provisional se trunca al prefijo aceptado antes de
confirmar el token correctivo. En una ejecución real de 1.024 tokens, la salida
de ocho tokens conservó exactamente ids y SHA-256 KV. Se aceptaron 6 de 14
propuestas (`42,86 %`), hubo una ronda rechazada, los forwards objetivo bajaron
a cinco y decode pasó de `581,7 ms` a `527,2 ms`, un `1,10x`. Es una prueba corta
y una sola corrida: demuestra generación, aceptación y rollback exactos, no
una ganancia universal.

La verificación exacta ya se ha llevado también al ColdStore paginado mediante
`--reference-speculative-cold`. Es un modo explícito: un tensor de varios tokens
no se interpreta automáticamente como especulación, porque también podría ser
prefill. Durante una ronda, cada página histórica K/V se materializa una vez por
capa; las consultas conservan SDPA de longitud uno y prefijos causales sucesivos.
Los candidatos permanecen en una cola provisional no sellada, de modo que un
rechazo puede truncarla sin cancelar trabajos de conversión ni modificar el
backing frío.

Prueba real MiniCPM5-1B, 1.024 tokens, marcos/chunks de 256 y ocho tokens de
salida:

| Métrica | Terrsvusion secuencial | Verificación fría multí-token |
| --- | ---: | ---: |
| Decode | 1,281 s | 0,470 s |
| Speedup frente a Terrsvusion secuencial | referencia | 2,73x |
| Speedup frente a BF16 denso de la misma corrida | — | 1,21x |
| Eventos del certificador | 168 | 24 |
| Ampliaciones branch-and-bound | 816 | 120 |
| K puntuados / V materializados | 172.704 / 172.704 | 24.600 / 24.600 |
| Lecturas frías operativas | 1.680 | 624 |
| Paridad de ids | referencia | `true` |
| Paridad SHA-256 de todo el KV BF16 | referencia | `true` |

El drafter acertó seis de seis propuestas en dos rondas y redujo en un 85,7 %
los eventos del certificador. El digest se calcula en layout lógico
`[capa][K/V][cabeza][token][dimensión]`; sus lecturas diagnósticas se excluyen
de los contadores operativos.

También se ejecutó un caso adversarial con rechazos: 10 tokens finales, 10
propuestos, 6 aceptados y 3 rondas rechazadas. Conservó exactamente ids y
SHA-256 KV después de los rollbacks, pero fue más lento que BF16 denso
(`1,254 s` frente a `0,747 s`). Esto demuestra que el rollback es real y que el
speedup no es universal y motivó el siguiente cambio.

La generación canónica y el servidor usan ahora un drafter determinista basado
en un autómata de sufijos construido en Rust en tiempo lineal. El primer
candidato siempre es el argmax exacto que el target acaba de calcular; solo se
prolonga cuando el estado del documento tiene una única transición y no es
también terminal. La unicidad documental no se presenta como garantía: todos
los tokens continúan siendo verificados por el target BF16. Una política
adaptativa aumenta K tras aceptación total y, ante un rechazo, vuelve a K=2 y
puede pasar temporalmente por decode secuencial. Los modos `--exact-oracle` y
`--shadow-metrics` desactivan esta ruta para no contaminar sus diagnósticos.

Prueba canónica posterior con 1.024/256/8:

- Terrsvusion: `0,459 s` de decode; BF16 denso: `0,616 s`;
- 8 propuestas, 7 aceptadas, una ronda rechazada y 3 forwards objetivo;
- cero eventos del certificador y cero ampliaciones;
- 576 lecturas frías;
- ids y SHA-256 KV idénticos al control BF16.

La prueba adversarial canónica posterior produjo diez tokens con Terrsvusion en
`0,904 s` frente a `0,936 s` del BF16 denso: 14 propuestas, 8 aceptadas, una
ronda rechazada y 5 forwards objetivo. Conservó también la paridad exacta de
ids y SHA-256. Son ejecuciones individuales, no una distribución estadística,
pero la aceleración ya gobierna la ruta del servidor y no es solo un benchmark.

## Evidencia real más reciente

### Línea base secuencial anterior a la integración multí-token

Configuración: MiniCPM5-1B, 1.024 tokens de contexto, chunks y marcos de 256,
8 tokens generados.

| Métrica | BF16 denso | Terrsvusion |
| --- | ---: | ---: |
| Respuesta | `AZULEJO-7319` | `AZULEJO-7319` |
| Paridad de ids | referencia | `true` |
| Prefill | 1,520 s | 1,526 s (referencia anterior) |
| Decode | 0,605 s | 1,281 s |
| KV BF16 de referencia | 25.337.856 bytes | — |
| KV estable residente | — | 333.780 bytes |
| Ahorro residente contabilizado | — | 98,68 % |

Esta tabla queda como línea base histórica de la ruta secuencial. La ruta
canónica actual es la multí-token exacta descrita arriba.

Después de corregir el antiguo miss LSH que sembraba todo el catálogo, los
estados iniciales acumulados bajaron de 1.008/1.008 a 192/1.008. El
branch-and-bound ejecutó 816 ampliaciones y terminó cargando 1.008/1.008:
`hierarchy_k_tokens_avoided` siguió siendo 0.

### Oráculo exacto

En la misma geometría, con `--compare-dense --exact-oracle`:

| Tolerancia local | Tokens omitibles | Porcentaje de 172.704 oportunidades |
| ---: | ---: | ---: |
| `1e-6` | 0 | 0 % |
| `1e-5` | 0 | 0 % |
| `1e-4` | 5 | 0,0029 % |
| `1e-3` | 441 | 0,2554 % |

Otros resultados:

- 168 eventos exactos observados.
- Primera tolerancia local observada que permite una omisión:
  `2,783061069137239e-6`.
- Margen mínimo top-1/top-2 observado: `0,4375`.
- Paridad final frente al control BF16: `true`.
- Decode con oráculo: aproximadamente 14,48 s frente a aproximadamente 1,8 s
  sin él. Ese coste pertenece al diagnóstico y no a la ruta normal.

El resultado descarta que la AABB sea el único bloqueo en esta prueba: incluso
con scores QK exactos, `1e-6` y `1e-5` permiten cero omisiones. No demuestra que
subir la tolerancia sea seguro para la respuesta final.

## Métricas principales

- `kv_bf16_reference_bytes`: tamaño del KV BF16 denso equivalente.
- `kv_resident_bytes`: backing excluido; estado estable residente contabilizado.
- `kv_physical_bytes`: backing más estado residente y metadatos contabilizados.
- `kv_prefill_working_peak_bytes`: K/V BF16 temporal conocido durante prefill.
- `kv_known_attention_scratch_peak_bytes`: temporales conocidos; excluye
  workspace opaco y padding del allocator LibTorch.
- `kv_exact_*`: IO, caché fría, integridad y lecturas parciales.
- `kv_conversion_*`: trabajos aceptados, entregados y tiempo de workers.
- `parallel_*`: estados dimensionales, fingerprints, micrometadatos y LSH.
- `hierarchy_*`: estados iniciales/cargados, ampliaciones y tokens K evitados.
- `exact_mass_*`: K puntuados, V materializados y cota terminal.
- `exact_oracle_*`: diagnóstico shadow de omisibilidad y margen de logits.
- `prefix_cache`: hit, tokens reutilizados, lookup verificado y rehidratación.
- `kv_sha256`: digest del estado KV lógico completo, independiente de cómo se
  fragmenten físicamente las páginas.

Estas métricas no equivalen a RSS total. En particular, no incluyen todo el
workspace interno de SDPA/LibTorch.

## Servidor y concurrencia

```bash
cargo run --release --bin terrsvusion-server -- \
  --model /home/yo/models/MiniCPM5-1B \
  --prefix-cache-dir /ruta/estable/terrsvusion-prefix-cache \
  --host 127.0.0.1 \
  --port 8088
```

Endpoints:

- `GET /health`
- `POST /v1/chat/completions`

```bash
curl -sS http://127.0.0.1:8088/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hola"}],"max_tokens":32}'
```

El servidor usa un hilo propietario de inferencia y `sync_channel` con ocho
slots de espera. La saturación se devuelve como HTTP 429; un worker detenido se
devuelve como 503. Las inferencias no se ejecutan en paralelo sobre una misma
autoridad del modelo.

Límites operativos actuales:

- sin timeout HTTP por petición;
- sin cancelación cooperativa si el cliente desconecta;
- sin deadline separado para prefill/decode;
- sin endpoint `/ready` distinto de `/health`;
- sin supervisor que reconstruya el modelo tras terminar el hilo;
- `[profile.release] panic = "abort"`: en release, un pánico aborta el proceso y
  no puede ser recuperado por `catch_unwind`.

## Construcción y validación

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/yo/.local/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --all-targets
```

Estado actual verificado:

- 48 tests correctos;
- 0 fallos;
- Clippy limpio con advertencias tratadas como error;
- biblioteca y tres ejecutables compilados en release.

Prueba real de prefijo persistente del 28 de octubre de 2026, con
MiniCPM5-1B BF16, un fragmento coherente en español de *La isla del tesoro* y
dos preguntas diferentes:

| Evento | Prefijo | Tokens reutilizados | Prefill | Lookup | Rehidratación |
| --- | --- | ---: | ---: | ---: | ---: |
| Primera pregunta | miss | 5.454 | 15,954 s | 0,00002 s | 0,155 s |
| Segunda pregunta | hit | 5.454 | 0,549 s | 0,654 s | 0,147 s |
| Tras reiniciar servidor | hit | 5.454 | 0,443 s | 0,701 s | 0,163 s |

El hit redujo el tiempo medido de prefill aproximadamente un 96,6 %. La prueba
tras reinicio demuestra persistencia de proceso. No demuestra que la respuesta
completa sea rápida: con 16 tokens generados, decode siguió costando unos
22,5 s y dominó la latencia total. La fila tras reinicio generó solo dos tokens
y sirve para probar recuperación, no para comparar latencia total.

Una comparación adicional con la misma pregunta y cuatro tokens generados
produjo exactamente el mismo texto (`<think>\nEsto`) en miss y hit. El prefill
medido fue 18,470 s en miss y 0,482 s en hit. Esta prueba acotada demuestra
paridad de la secuencia observada; no constituye una prueba universal para toda
entrada o longitud de generación.

Regresiones cubiertas explícitamente:

- cada máscara Hamming de radio hasta 4 se visita una sola vez sin construir
  el antiguo vector temporal de 12.951 máscaras por tabla;
- en builds con unwind, un panic del encoder produce un resultado de error
  ordenado y el worker procesa el trabajo siguiente; el perfil release conserva
  `panic = "abort"` y no promete recuperación dentro del proceso;
- el prefijo persistente reconstruye 24 capas, verifica el backing, conserva el
  eje dimensional y rehidrata K/V BF16 exacto;
- los identificadores de región duplicados fallan cerrados antes de construir
  cotas o materializar K.

Una corrida real posterior a la materialización incremental, con 512 tokens,
chunks y marcos de 128, dos tokens solicitados y control BF16, produjo los ids
`[359, 23373]` en ambas rutas (`exact_token_parity: true`). Durante el único
paso decode comparado se observaron 24 eventos de capa, 120 ampliaciones y 144
estados materializados en total; `kv_exact_cold_reads` fue 240. La ejecución
sigue cargando todos los estados y no demuestra todavía aceleración por
omisión.

Benchmark con control y oráculo:

```bash
cargo run --release --bin terrsvusion-long-context -- \
  --model /home/yo/models/MiniCPM5-1B \
  --book-tokens 1024 \
  --chunk-size 256 \
  --storage-frame-tokens 256 \
  --generate-tokens 8 \
  --compare-dense
```

`--book-file RUTA --question TEXTO` usa un documento real. El valor de
`--book-tokens` limita los tokens de libro incluidos; el prompt y la pregunta
se añaden aparte. `--adversarial-retrieval` y `--stress-seed` generan entradas
sintéticas distintas y son mutuamente excluyentes.

## Garantías demostradas

- Los pesos no se reescriben ni se guardan en Q4.
- Los K/V recuperados para atención proceden de BF16 exacto.
- Un marco no se libera antes de confirmar su commit frío.
- Los registros comprometidos se verifican y las lecturas parciales validan
  cada token recuperado.
- Un miss LSH no activa fuerza bruta ni otro sistema.
- El catálogo dimensional no puede sustituir al certificado.
- La prueba descrita conserva exactamente los ids generados frente al control
  BF16.
- La residencia KV estable contabilizada es muy inferior al denso en esa
  prueba.

## No demostrado todavía

- `hierarchy_k_tokens_avoided > 0` en una inferencia real del modelo.
- Menor latencia general que BF16 denso.
- Reducción del coste cuadrático del prefill.
- Paridad universal, conservación semántica universal o garantía end-to-end de
  logits.
- Seguridad de tolerancias superiores a `1e-6`.
- Error real y paridad de una rama shadow con omisiones aplicadas.
- Eliminación de I/O/rehidratación del prefijo en hits; hoy se evita su cómputo
  Transformer, pero el backing todavía se copia y se rehidrata por petición.
- Concurrencia paralela de múltiples inferencias sobre el modelo.
- Certificación de 30.000, 100.000 o 131.072 tokens.
- Métricas completas de RSS, workspace interno de LibTorch o VRAM.
- Preparación operacional de producción.

## Próxima decisión basada en evidencia

La siguiente prueba útil no es añadir una geometría sofisticada a ciegas. Es
ejecutar una rama shadow que aplique las omisiones candidatas, mida el error real
de atención, propague el resultado hasta logits y compare toda la secuencia
autoregresiva. Solo después podrá decidirse si procede:

1. derivar una tolerancia dinámica con garantía hasta logits;
2. implementar una envolvente K orientada de bajo rango con residual
   certificado;
3. separar las firmas LSH por cabeza KV;
4. o aceptar que determinadas consultas requieren atención completa.
