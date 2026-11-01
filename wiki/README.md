# Wiki verificable de KV-SHEI

La wiki de GitHub se trata como una vista derivada y auditable del repositorio,
no como una segunda fuente de verdad. Las páginas declaradas en
`wiki-manifest.json` se regeneran desde sus documentos canónicos; las páginas
editoriales se conservan, pero pasan por los mismos controles de enlaces,
diagramas y secretos.

## Flujo seguro

```bash
# Clonar la wiki publicada para conservar las páginas editoriales.
git clone https://github.com/sheilyneural-afk/KV-SHEI.wiki.git /tmp/kv-shei-wiki

# Regenerar una propuesta sin modificar el clon.
python3 scripts/wiki_audit.py generate \
  --existing-wiki /tmp/kv-shei-wiki \
  --output deploy/local-production/runtime/wiki/generated

# Verificar y emitir un recibo reproducible.
python3 scripts/wiki_audit.py check \
  --wiki deploy/local-production/runtime/wiki/generated \
  --receipt deploy/local-production/runtime/wiki/wiki-audit-receipt.json

# Comparar la propuesta con lo publicado.
python3 scripts/wiki_audit.py diff \
  --existing-wiki /tmp/kv-shei-wiki \
  --wiki deploy/local-production/runtime/wiki/generated
```

`check` no publica. En pull requests, el workflow `documentation-truth` genera
la propuesta y el recibo como artefactos de CI y conserva permisos de solo
lectura. Después de integrar un cambio aprobado en `main`, un segundo job puede
publicar exactamente ese artefacto auditado. La misma operación puede invocarse
manualmente con `scripts/wiki-publish.sh`; en ambos casos se rechazan árboles
sucios, historiales divergentes y resultados que no hayan superado el auditor.

## Autoridad

- Los documentos canónicos mandan sobre las páginas derivadas.
- Una página editorial puede interpretar resultados, pero no elevar una
  afirmación por encima de la evidencia registrada.
- Un recibo producido desde un árbol sucio queda marcado como no reutilizable.
- Un dato físico solo es vigente para las identidades que enlaza su propio
  contrato; el auditor documental no convierte una existencia de archivo en
  evidencia física.
