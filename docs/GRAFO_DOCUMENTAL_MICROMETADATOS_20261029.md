# Grafo documental y micrometadatos verificables

Estado: núcleo Rust funcional y ruta de consulta conectada a Llama mediante
`terrsvusion-document-ask`; todavía no conectado al servidor HTTP ni a un
extractor PDF/OCR.

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
