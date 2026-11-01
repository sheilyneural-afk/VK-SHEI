## Objetivo y alcance

Explique qué contrato cambia, qué queda fuera y qué Issue/Milestone gobierna el trabajo.

Closes #

## Identidad y riesgo

- Fuente/base revisada:
- Interfaces, formatos o estado persistente afectados:
- Riesgos de concurrencia, autorización, migración y rollback:

## Evidencia ejecutada

| Escalón | Resultado | Evidencia o enlace |
|---|---|---|
| Prueba focal/regresión | | |
| `scripts/check.sh` | | |
| Build/artefacto | | |
| Configuración efectiva | | |
| Instalado/residente | No medido / Resultado | |
| E2E autenticado y efecto | No medido / Resultado | |
| Persistencia/readback | No medido / Resultado | |
| Reinicio aplicable | No medido / Resultado | |

## Checklist de integridad

- [ ] El diff contiene solo cambios pertenecientes a este objetivo.
- [ ] Añadí una regresión que reproduce la causa correcta o expliqué por qué no aplica.
- [ ] No presento mocks, fixtures, manifests, health o transporte como prueba de producto E2E.
- [ ] Distingo fuente, build, instalación, proceso residente y publicación remota.
- [ ] Los hashes/recibos están ligados al commit, configuración, bundle y backend evaluados.
- [ ] Documenté evidencia omitida, no disponible o fallida; no la convertí en éxito.
- [ ] Revisé compatibilidad de esquemas, variables, flags, rutas y consumidores persistentes.
- [ ] No incluí secretos, tokens, modelos, caches ni estado runtime.
- [ ] Actualicé documentación, operación, rollback y trabajo pendiente cuando corresponde.
