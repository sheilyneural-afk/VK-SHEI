# Evaluación de la recuperación documental

`mslim-document query` tiene dos modos de ordenar páginas:

- `--rank routing` (por defecto): el enrutado determinista de `document_graph`.
- `--rank fused`: lo fusiona con un canal BM25 mediante RRF ponderado
  (`fused_retrieval`, `page_fusion`). Cada página admitida se verifica contra el
  SHA-256 de su texto original.

## Resultado medido

Se midió con el binario real. El modo por defecto no cambia: no hay evidencia de
que `fused` mejore la recuperación, así que sigue siendo opt-in.

| Prueba | Preguntas | `routing` | `fused` |
|---|---|---|---|
| Sintética, palabras copiadas del texto, 3 páginas | 300 | 98,7 % | 99,7 % |
| Sintética, páginas de 800 caracteres, 1 página | 600 | 99,0 % | 98,8 % |
| 36 preguntas escritas a mano con sinónimos, 1 página | 36 | 50 % | 50 % |
| ídem, 2 páginas | 36 | 58 % | 64 % |
| ídem, 3 páginas | 36 | 67 % | 67 % |

Las diferencias entre modos son de 0 a 3 preguntas y no son significativas. Lo
que sí muestran las preguntas a mano es el límite real de ambos canales:
fallan cuando la pregunta y el documento usan palabras distintas para lo mismo
(por ejemplo "tarjeta gráfica" frente a GPU). Eso requiere expansión semántica
o embeddings, que no están implementados.

## Reproducir

```bash
BIN=deploy/local-production/runtime/build-target/debug/mslim-document
python3 scripts/eval-retrieval-labeled.py "$BIN" scripts/retrieval-labeled-questions.json
python3 scripts/eval-retrieval.py "$BIN" docs/VALIDATION.md README.md --queries 60
```

Limitaciones: las 36 preguntas las escribió quien conocía el buscador, sobre tres
documentos de este repositorio; las sintéticas miden solo coincidencia léxica.
Ninguna sustituye un conjunto de relevancia etiquetado por personas.
