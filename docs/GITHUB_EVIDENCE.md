# Gobierno y evidencia en GitHub

GitHub organiza el trabajo y conserva evidencia; no sustituye la certificación
física del runtime. Esta política mantiene separados los estados de fuente,
build, instalación, proceso residente, efecto autorizado, persistencia y
recuperación posterior al reinicio.

## Cadena de autoridad

```mermaid
flowchart LR
    M[Meta / Milestone] --> I[Issue con aceptación]
    I --> P[Pull Request]
    P --> C[CI: core, seguridad, dependencias]
    C --> A[Artefacto + manifest de hashes]
    A --> T[Atestación GitHub/Sigstore]
    T --> R[GitHub Release]
    R --> V[Verificación de instalación]
    V --> E[E2E autorizado + efecto]
    E --> D[Persistencia + readback]
    D --> X[Reinicio + recuperación]
```

Cada enlace demuestra algo distinto. Una atestación de build vincula un
artefacto con repositorio, workflow y commit, pero no demuestra que ese
artefacto esté instalado o residente, ni que el modelo, backend o estado
persistente sean correctos.

## Controles versionados

| Superficie | Finalidad | Límite |
|---|---|---|
| `M-SLIM / core` | Gate reproducible de formato, lint, tests, contratos e inventario | No usa modelo o hardware físico |
| `Security evidence / CodeQL` | Análisis de Rust y workflows de Actions | Análisis estático, no explotación runtime |
| `Dependency review` | Bloquea nuevas dependencias vulnerables desde severidad moderada | Depende del grafo que GitHub pueda resolver |
| `OpenSSF Scorecard` | Señales de postura de cadena de suministro | No certifica ausencia de vulnerabilidades |
| `github-evidence.py` | Hashes, fuente, contexto y límites de cada artefacto | Registra hechos; no amplía el nivel de prueba |
| `Release provenance` | Paquete ligado a fuente, SHA-256, manifest y atestaciones | Build provenance, no readiness productiva |
| Job físico manual | Paridad, restauración física y calibración sobre runner real | Solo el backend/modelo/configuración ejecutados |

Las Actions externas están fijadas por SHA completo. El comentario conserva la
versión humana; el SHA es la autoridad inmutable.

## Inventario reproducible de avisos de dependencias

El workflow `Dependency advisory evidence` conserva durante 90 días cuatro
piezas ligadas al commit analizado:

1. la salida JSON completa de `cargo audit` y la identidad de su base RustSec;
2. las alertas Dependabot abiertas obtenidas desde la API de GitHub;
3. el árbol Cargo con todas las características y objetivos;
4. un recibo normalizado `mslim.dependency_advisory_evidence/v1`.

La consulta de Dependabot usa exclusivamente el `GITHUB_TOKEN` efímero con
`vulnerability-alerts: read`; no persiste un PAT ni concede escritura sobre
alertas. `security-events` corresponde a otras superficies de seguridad y no
autoriza este endpoint.

El recibo compara esos datos con
`.github/security/dependency-advisory-baseline.json`. El gate falla si aparece
un identificador no registrado, cambia paquete/versión/clase o `Cargo.lock` ya
no coincide con el hash revisado. Un gate verde significa **sin deriva nueva**;
no significa “sin vulnerabilidades”. Los hallazgos conocidos continúan
figurando como `known_findings_open` hasta desaparecer de las fuentes.

La revisión del commit `8648ff2` descompone los 15 identificadores agregados
por Scorecard de esta forma:

| Grupo | Cantidad | Cadena / superficie | Alcanzabilidad observada | Estado |
|---|---:|---|---|---|
| `zip` / RustSec extracción | 3 | `tch 0.18.1 → zip 0.6.6`; solo `torch-backend` | Las APIs afectadas existen, pero no se encontró una llamada a `ZipArchive::extract` o `ZipStreamReader::extract` en KV-SHEI ni en las rutas observadas de `tch`/`torch-sys` | Riesgo transitivo abierto; #32 |
| `zip` / GHSA | 2 | Misma cadena opcional | No se encontró extracción streaming ni lectura AES con contraseña; esto reduce la alcanzabilidad observada, no corrige el paquete | Riesgo transitivo abierto; #32 |
| `zstd-safe` | 9 | `tch → zip → zstd 0.11.2 → zstd-safe 5.0.2`; `torch-backend` | `zip` activa Zstandard, pero las funciones de bajo nivel señaladas no aparecen en la ruta observada; `RUSTSEC-2026-0403` afecta el decoder de escritura y `zip` usa decoder de lectura / encoder de escritura | Riesgo transitivo abierto; #34 |
| `paste` sin mantenimiento | 1 | `tokenizers 0.22.2 → paste 1.0.15`; `model-bundle`, incluido el build por defecto | Macro de procedimiento usada al compilar; el aviso es de mantenimiento, no una vulnerabilidad runtime declarada | Sustitución pendiente; #34 |

La clasificación “no se encontró una llamada” está limitada a la fuente y a
las dependencias inspeccionadas en esa revisión. No equivale a una prueba
universal de inalcanzabilidad. La migración automática de `tch 0.18.1` a
`0.26.0` tampoco se acepta solo porque CI portátil compile: cambia el contrato
de PyTorch/LibTorch y necesita build, carga de modelo, paridad K/V y validación
física específicos antes de integrarse.

## Project, campos y vistas

El Project `KV-SHEI — Evidencia y preparación para producción` utiliza Issues y
Pull Requests como elementos. Los campos recomendados son:

- **Status:** Backlog, Ready, In progress, Validation, Blocked, Done.
- **Priority:** P0, P1, P2, P3.
- **Area:** Runtime, Gateway, Model, Adapters, K/V, ColdStore, Continuity,
  Retrieval, Validation, Release, Documentation, Repository.
- **Evidence:** Proposed, Source, Focused test, CI gate, Built, Installed,
  Resident, E2E, Persistence, Restart, Attested release.
- **Result:** Not measured, Failed, Partial, Passed.
- **Backend:** General, Native CPU, Accelerator, LibTorch, HIP.
- **Start date**, **Target date**, **Iteration** y **Effort**.

Vistas mínimas:

1. **Executive board:** agrupado por Status y ordenado por Priority.
2. **Evidence matrix:** tabla agrupada por Evidence y Backend.
3. **Production roadmap:** Roadmap por Start/Target y Milestone.
4. **Blocked P0/P1:** filtro de prioridades críticas bloqueadas.
5. **Validation queue:** Status=Validation, agrupado por Area.
6. **Release scope:** filtro por Milestone de la release activa.

Insights mínimos:

- Burn-up histórico: abiertos frente a completados y alcance total.
- Elementos actuales por Status, agrupados por Priority.
- Elementos por Evidence, agrupados por Backend.
- Resultados por Area.
- Esfuerzo por Iteration.

GitHub no expone todas las vistas y gráficos de Projects mediante su API. Los
campos y elementos pueden automatizarse; la disposición visual y determinados
Insights se verifican en la interfaz del Project.

## Milestones y condiciones de cierre

Un Milestone representa una meta o versión, no una simple fecha. Su descripción
debe incluir:

- objetivo y exclusiones;
- configuración/backends cubiertos;
- gates requeridos;
- artefactos y recibos esperados;
- política de rollback;
- evidencia que continuará sin medir.

Cerrar todos sus Issues solo demuestra que el registro de trabajo está cerrado.
La release exige además que sus condiciones observables y recibos sean válidos.

## Verificar una release

Descargue el archivo, su `.sha256` y el manifest `.evidence.json` desde la
release. Compruebe primero el contenido y después su procedencia:

```bash
sha256sum --check mslim-vX.Y.Z-linux-x86_64.tar.gz.sha256
gh attestation verify mslim-vX.Y.Z-linux-x86_64.tar.gz \
  --repo sheilyneural-afk/KV-SHEI
tar -xzf mslim-vX.Y.Z-linux-x86_64.tar.gz
python3 scripts/github-evidence.py verify \
  mslim-vX.Y.Z-linux-x86_64.evidence.json \
  --artifact-root mslim-vX.Y.Z-linux-x86_64
```

La instalación y el runtime requieren después `verify-release.sh`, identidad
del binario residente, matriz/calibración aplicables, petición autenticada,
efecto, persistencia/readback y el reinicio que corresponda.

## Retención y secretos

- Los artefactos ordinarios de CI se retienen 30 días; candidatos de release,
  90 días. Una GitHub Release conserva los entregables publicados.
- Nunca se suben modelos, tokens, secretos, URLs privadas, caches, estado de
  usuario ni el árbol `deploy/local-production/runtime/` completo.
- Los logs físicos se limitan a salidas explícitas y recibos seleccionados.
- Una evidencia fallida se conserva como fallo; no se transforma en éxito ni
  se reemplaza por documentación.
