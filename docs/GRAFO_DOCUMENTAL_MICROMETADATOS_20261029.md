# Grafo documental y micrometadatos verificables

Estado: núcleo Rust funcional y aislado; todavía no conectado al servidor
Llama ni a un extractor PDF/OCR.

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
  --max-pages 4 --max-characters 16000 --graph-hops 1
```

## Límites no cerrados

- No extrae directamente PDF, EPUB, DOCX ni imágenes.
- No genera entidades o relaciones mediante Llama.
- No enlaza todavía páginas documentales con prefijos/KV de ColdStore.
- No ensambla todavía el prompt final ni ejecuta una segunda búsqueda.
- No certifica exhaustividad semántica.
- No reduce por sí solo el coste de los kernels; reduce el contexto únicamente
  cuando está integrado delante del prefill.
