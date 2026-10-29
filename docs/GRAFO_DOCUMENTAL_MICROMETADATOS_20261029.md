# Grafo documental y micrometadatos verificables

Estado: núcleo Rust funcional y rutas de consulta conectadas a Llama mediante
`terrsvusion-document-ask` y `POST /v1/document/query`; todavía no conectado a
un extractor PDF/OCR.

## Autoridad

`src/document_graph.rs` conserva por página:

- texto original;
- número y sección;
- SHA-256 del texto;
- términos y números deterministas;
- referencias explícitas a páginas;
- aristas de vecindad, sección y referencia.

Los micrometadatos solo enrutan. Cada evidencia devuelta contiene el texto
original y su digest. Antes de consultar se verifican todos los hashes.
Al reabrir un archivo persistido, los índices y aristas se reconstruyen desde
las páginas verificadas: un índice alterado no puede ocultar una página.

## Micrometadata grafal

Cada frase o línea no vacía se representa como una hiperarista compacta
`GraphMicroSpan`. No copia el texto ni crea todas las parejas de palabras:
conecta implícitamente los términos y números que coexisten en ese tramo y
guarda únicamente:

- índice del tramo;
- offsets UTF-8 inicial y final;
- términos y números normalizados;
- SHA-256 del fragmento original.

El ranking focal usa la mayor coincidencia conjunta dentro de una hiperarista.
Por tanto, dos términos presentes en lugares inconexos de una misma página ya
no reciben la misma autoridad relacional que dos términos presentes en una
misma frase. Antes de recuperar se comprueban límites UTF-8 y SHA-256 de cada
ancla contra la página original.

Esta es micrometadata grafal textual, no una afirmación de comprensión causal:
todavía no extrae de forma certificada entidades, hechos, situaciones,
temporalidad o causalidad. Esas capas deberán añadir aristas tipadas que apunten
a estas mismas anclas, marcando por separado relaciones explícitas e inferidas.

## Recuperación por fases

Una fase tiene límites separados de páginas, caracteres y saltos de grafo.
Si no caben todos los candidatos, devuelve `deferred_candidate_pages` y
`complete_for_candidate_set=false`. Si ningún micrometadato coincide, falla
abierto a todas las páginas y expresa la cobertura parcial según el presupuesto.

Hay dos alcances explícitos:

- `focused`: conserva las páginas con la máxima coincidencia conjunta de
  términos deterministas, coincidencias numéricas o frase exacta, y después
  expande el grafo. Su indicador de completitud cubre únicamente ese conjunto
  focal.
- `exhaustive`: conserva todas las páginas que coincidan con cualquier señal
  determinista y expone como diferidas las que no quepan.

Ninguno certifica exhaustividad semántica. Esta separación evita que una
entidad muy frecuente convierta todas las páginas en semillas sin ocultar las
coincidencias amplias: `broad_match_pages` conserva esa contabilidad.

Esto todavía NO demuestra que cualquier pregunta semántica encuentre su
respuesta. La implementación actual cubre coincidencia determinista, números,
frases normalizadas, vecindad y referencias explícitas. Entidades, embeddings,
OCR, tablas, hechos inferidos y planificación iterativa permanecen pendientes.

## CLI reproducible

Entrada `pages.json`:

```json
[
  {"page_number":1,"section":"Contrato","original_text":"Pago de 38 millones."},
  {"page_number":2,"section":"Excepciones","original_text":"Véase la página 1."}
]
```

Construcción y consulta:

```bash
cargo run --release --bin terrsvusion-document -- build \
  --document-id contrato --version sha-documento \
  --pages pages.json --output contrato.graph.json

cargo run --release --bin terrsvusion-document -- query \
  --graph contrato.graph.json --text '38 millones' \
  --scope focused --max-pages 4 --max-characters 16000 --graph-hops 1
```

La ruta física de inferencia requiere `torch-backend`:

```bash
cargo run --release --features torch-backend \
  --bin terrsvusion-document-ask -- \
  --graph contrato.graph.json --question '¿Cuál era el pago?' \
  --scope focused --max-pages 4 --max-characters 16000 --graph-hops 1
```

El servidor carga el grafo una vez y exige la autoridad LibTorch compartida:

```bash
terrsvusion-server \
  --model /ruta/Llama-3.1-8B-Instruct \
  --shared-authority \
  --document-graph documento.graph.json

curl -X POST http://127.0.0.1:8088/v1/document/query \
  -H 'content-type: application/json' \
  -d '{"question":"¿Cuántos habitantes tenía Puerto Gris?","scope":"focused"}'
```

El arranque rechaza `--document-graph` sin `--shared-authority`. Las fases se
preparan desde el grafo residente y se envían secuencialmente al pool; ninguna
petición documental carga un checkpoint. El prefill de evidencia usa la ruta
densa BF16 para no confundir selección documental con omisión V experimental.

Para recorrer candidatos sin recargar el modelo, `--max-phases` permite varias
fases sobre un único ranking inmutable. Cada página se materializa una sola vez;
la inferencia continúa únicamente cuando la salida normalizada es exactamente
`EVIDENCIA_INSUFICIENTE`:

```bash
cargo run --release --features torch-backend \
  --bin terrsvusion-document-ask -- \
  --graph contrato.graph.json --question '¿Cuál es el valor?' \
  --scope exhaustive --max-pages 1 --max-phases 8
```

## Medición física con El Faro

Prueba ejecutada el 29 de octubre de 2026 con
`assets/el-faro-volumen-completo.txt`, Llama 3.1 8B Instruct y una autoridad
LibTorch real:

- documento: 368.968 bytes, 95 páginas documentales;
- pregunta: `¿Cuántos habitantes tenía Puerto Gris?`;
- coincidencias deterministas amplias: 94 páginas;
- semillas focales sin expansión: página 1;
- tokens del prompt documental completo: 116.978;
- tokens entregados realmente al Transformer: 1.392;
- posiciones de prompt no procesadas: 115.586 (98,81 %);
- prefill selectivo: 15,933815094 s;
- respuesta: `[Página 1] Puerto Gris era una población de dos mil habitantes.`;
- RSS máximo del proceso: 31.472.752 KiB;
- `semantic_exhaustiveness_certified=false`.

La reconstrucción con micrometadata grafal produjo además:

- 4.091 hiperaristas sobre 95 páginas;
- máximo de 81 hiperaristas por página;
- índice anterior sin hiperaristas: 638.504 bytes;
- índice grafal: 1.622.246 bytes;
- incremento: 983.742 bytes;
- construcción: 0,60 s y 10.560 KiB de RSS máximo;
- consulta poblacional: 94 coincidencias amplias, pero solo la página 1 como
  candidata final. La página 7 también relaciona `habitantes` y `Puerto Gris`,
  pero la pregunta cuantitativa prioriza la hiperarista de la página 1 que
  contiene `dos mil`. Los artículos ambiguos `un/una/uno` no se clasifican
  como cantidades sin una regla contextual.

La inferencia física posterior procesó 1.413 tokens frente a 116.999 del
documento completo, evitó 115.586 posiciones y respondió
`Dos mil habitantes. [Página 1]`. El controlador también fue probado contra
una salida de solo cita (`[Página 1]`) y ya no la acepta como respuesta
sustantiva.

## Prueba HTTP sobre autoridad compartida

Se arrancó el servidor real con Llama 3.1 8B, el grafo de 95 páginas,
`--shared-authority`, un worker y el endpoint documental. Evidencia observada:

- `/ready`: `authority_mode=shared_libtorch_weight_storage`;
- 291 tensores de peso verificados por `data_ptr`;
- 95 páginas y 4.091 hiperaristas residentes;
- cargas de modelo durante la petición: 0;
- coincidencias amplias: 94; candidatas: 1; evidencia: página 1;
- prompt físico del worker: 1.350 tokens;
- respuesta: `La respuesta es: 2.000 habitantes. [Página 1]`;
- tiempo HTTP: 22,28 s;
- RSS residente después de la petición: 16.384.728 KiB;
- parada limpia del servidor tras la medición.

Esto demuestra la conexión grafo -> endpoint -> pool -> LibTorch compartido.
No demuestra exhaustividad semántica.

## Concurrencia documental multiworker

La primera prueba con dos peticiones simultáneas descubrió un defecto real de
reparto: Axum clona el estado de aplicación para cada handler y
`InferencePool::clone` copiaba el valor de su cursor atómico. Dos handlers
podían empezar ambos en el worker 0 y serializarse aunque el servidor anunciara
dos workers. El cursor es ahora un `Arc<AtomicUsize>` compartido; la regresión
`cloned_pools_share_one_round_robin_cursor` verifica el reparto 0 -> 1 -> 0 a
través de clones del pool.

Con el arreglo, dos preguntas distintas se ejecutaron simultáneamente sobre el
servidor real de dos workers, 291 tensores LibTorch compartidos y cero cargas de
modelo por petición:

- tiempo total concurrente: 50,2334 s;
- tiempo total de las mismas peticiones en secuencia y sobre el mismo binario:
  61,8076 s;
- reducción del tiempo agregado: 18,73 %;
- throughput: de 0,03236 a 0,03981 peticiones/s, una mejora del 23,04 %;
- RSS máximo concurrente: 16.760.692 KiB;
- RSS máximo secuencial: 16.782.648 KiB;
- latencia de la consulta A: de 28,1696 a 44,7298 s (+58,79 %);
- latencia de la consulta B: de 33,6380 a 50,2334 s (+49,34 %).

Las latencias internas de cada inferencia concurrente, 44,7112 y 50,216 s,
coinciden aproximadamente con sus terminaciones de pared: las consultas se
solaparon de verdad y el tiempo total no es su suma. La mejora es de throughput
agregado, no de latencia individual. La contención de CPU y memoria hace cada
inferencia aproximadamente un 49-59 % más lenta. La diferencia de RSS de
21.956 KiB está dentro del ruido de esta medición y no se presenta como ahorro.

Esta prueba cierra el caso acotado de dos consultas documentales activas. No es
una certificación de carga sostenida, de más de dos consultas simultáneas ni de
escalado universal.

Esta medición sí demuestra que la ruta selectiva reduce el número efectivo de
posiciones procesadas por el Transformer en esta consulta. No demuestra que
toda pregunta se enrute correctamente ni que el coste baje en la misma
proporción para cualquier geometría.

## Medición física de continuación por fases

La fixture `assets/document-phases-example.json` fuerza una respuesta ausente
en la primera página y presente en la segunda. Con una sola autoridad LibTorch:

- fases disponibles y ejecutadas: 2;
- cargas del modelo: 1;
- fase 1, página 1: 308 tokens y `EVIDENCIA_INSUFICIENTE`;
- fase 2, página 2: 278 tokens y `COBALTO-73. [Página 2]`;
- páginas repetidas: 0;
- RSS máximo: 31.468.252 KiB;
- tokens acumulados por fases: 586;
- prompt denso completo equivalente: 350 tokens;
- delta frente al denso: -236 tokens, es decir, 236 tokens adicionales.

La prueba demuestra continuación, no una ganancia universal. En este documento
de dos páginas las instrucciones repetidas cuestan más que el denso completo.
Por eso el recibo publica simultáneamente
`transformer_prompt_tokens_avoided`, `transformer_prompt_tokens_extra` y un
`transformer_prompt_token_delta_vs_full` con signo. Un router posterior debe
escoger denso para documentos pequeños y fases para documentos cuyo contexto
evitado supere el coste de las instrucciones repetidas.

## Límites no cerrados

- No extrae directamente PDF, EPUB, DOCX ni imágenes.
- No genera entidades o relaciones mediante Llama.
- No enlaza todavía páginas documentales con prefijos/KV de ColdStore.
- Ensambla el prompt final y continúa por el ranking cuando Llama devuelve
  exactamente evidencia insuficiente. Todavía no cambia automáticamente de
  alcance focal a exhaustivo ni usa una síntesis acumulativa entre fases.
- No certifica exhaustividad semántica.
- La medición anterior demuestra menos posiciones QK/PV para una consulta; no
  constituye todavía una equivalencia general entre esa contabilidad y tiempo
  de kernel en todas las longitudes, lotes y dispositivos.
