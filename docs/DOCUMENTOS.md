# Documentos y El Faro

Código: `document_graph.rs`, `terrsvusion-document`,
`terrsvusion-document-ask` y `POST /v1/document/query` en
`terrsvusion-server`. Estado de `main` al 30/10/2026.

## Grafo documental (`document_graph.rs`)

- **Autoridad sin pérdida:** cada página guarda el texto original y su
  SHA-256. `load_verified` y `verify_original_authority` rechazan un grafo
  alterado. `save_atomic` escribe de forma atómica.
- **Micrometadatos para enrutar:** términos, números, referencias explícitas
  a páginas y tramos por frase o línea (`GraphMicroSpan`, cada uno con su
  SHA-256). Sirven para elegir páginas candidatas, no para responder.
- **Aristas:** página anterior y siguiente, referencia explícita y misma
  sección.
- **Recuperación por fases** con presupuesto (`max_pages`,
  `max_characters`, `graph_hops`) y dos ámbitos: `focused` (las páginas con
  más evidencia conjunta) y `exhaustive` (todas las que coinciden con alguna
  señal). Ninguno certifica que la cobertura semántica sea completa. Si un
  presupuesto no basta, la fase indica qué candidatos quedan pendientes.

## Herramientas

```bash
# Grafo a partir de texto plano (paginado por caracteres) o de páginas JSON
terrsvusion-document build-text --document-id ID --version V --text archivo.txt --output grafo.json [--page-characters N]
terrsvusion-document build --document-id ID --version V --pages paginas.json --output grafo.json
terrsvusion-document query --graph grafo.json --text "pregunta" [--max-pages N] [--max-characters N] [--graph-hops N] [--scope focused|exhaustive]

# Preguntas por fases con el modelo (torch-backend)
terrsvusion-document-ask --model DIR --graph grafo.json --question "..." [--max-phases N] [--scope ...] [--max-new-tokens N]
```

Fixtures: `assets/document-pages-example.json` y
`assets/document-phases-example.json`.

## Endpoint `POST /v1/document/query`

Se activa con `terrsvusion-server --shared-authority --document-graph grafo.json`. El cuerpo es
`{"question": "...", "max_tokens": 64}`; `max_tokens` va de 1 a 512.

El endpoint mete el documento completo del grafo como mensaje de referencia y
lo trata como prefijo persistente. La primera consulta hace el prefill denso
del documento. Las siguientes reutilizan su K/V BF16 inmutable y solo hacen el
prefill de la pregunta. La respuesta incluye `persistent_prefix` (`hit`,
`tokens`, `copied_bytes`, `immutable`) y el SHA-256 del documento ensamblado.
Sin `--document-graph` devuelve 404.

## El Faro (`assets/el-faro-volumen-completo.txt`)

Es el corpus de prueba del proyecto: 368.968 bytes, SHA-256
`11bc81eb83080e59462fdfc5fdb3f8f78c35ae6be650cdefbca2cc188b5e6ef5`.
`scripts/audit_llama_faro_inputs.py` cuenta los tokens con los tokenizadores
reales y con las plantillas de chat que extrae de `torch_runtime.rs`.

| Tokenizador | Tokens del texto | Con plantilla nativa de Llama |
|---|---|---|
| Llama 3.1 8B Instruct | 111.632 | 111.642 |
| MiniCPM5-1B | 112.104 | — |

El volumen completo no se ha procesado nunca de una vez con el transformer.

## Evidencia (29/10/2026)

| Medición | Resultado | Archivo |
|---|---|---|
| Tokens de El Faro | tabla anterior; también el fragmento de unos 7K tokens y tres plantillas de chat | [AUDITORIA_LLAMA_FARO_IDS_Y_PLANTILLAS](evidencia/documentos/AUDITORIA_LLAMA_FARO_IDS_Y_PLANTILLAS_20261029.json) |
| Grafo de El Faro | 95 páginas y 4091 microtramos, construido en 0,6 s. Para "¿Cuántos habitantes tenía Puerto Gris?" se eligió 1 página, con un prompt de 1413 tokens frente a 116.999 del documento entero. Respuesta: "Dos mil habitantes" | [EVIDENCIA_MICROMETADATA_GRAFAL_FARO](evidencia/documentos/EVIDENCIA_MICROMETADATA_GRAFAL_FARO_20261029.json) |
| Fases (fixture de 2 páginas) | la fase 1 da evidencia insuficiente y la fase 2 responde "COBALTO-73", con una sola carga del modelo. No es más rápido que el prefill denso: usa 236 tokens más | [EVIDENCIA_PREFILL_POR_FASES](evidencia/documentos/EVIDENCIA_PREFILL_POR_FASES_20261029.json) |
| Endpoint con prefijo persistente (Llama 3.1 8B) | la primera consulta falla en la caché y la segunda acierta: prefijo inmutable de 152 tokens, 0 bytes copiados, prefill de 0,93 s frente a 3,37 s | [EVIDENCIA_MEMORIA_DOCUMENTAL_KV_PERSISTENTE](evidencia/documentos/EVIDENCIA_MEMORIA_DOCUMENTAL_KV_PERSISTENTE_20261029.json) |

Las cifras de RSS de estas mediciones son de antes de `74a553e` y no se
reproducen aquí.
