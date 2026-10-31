# Capacidades candidatas: extracción nativa para M-SLIM

Fecha: 31 de octubre de 2026. Estado: inventario y evaluación de transferencia, **no implementación ni certificación de las capacidades en M-SLIM**.

## Alcance y trazabilidad

Se analizó el árbol de trabajo local, incluidos cambios previos sin commit. Los hashes individuales del inventario son más precisos que citar solo HEAD. Se realizó inventario estructural del alcance completo y lectura dirigida de los núcleos seleccionados; no se afirma haber revisado semánticamente cada línea.

Python se indexó con AST. Rust usa un extractor léxico para el inventario, complementado con lecturas de código; no sustituye a rust-analyzer. Las pruebas contadas no se ejecutaron en esta tarea. La única incidencia de parseo Python encontrada pertenece a una fixture deliberadamente inválida: cognition/tests/tools/audit/deadcode_guard/fixtures/pkg/bad.py.

| Proyecto | HEAD observado | Archivos fuente | Líneas inventariadas |
|---|---|---:|---:|
| SHEILY-COGNITIVA | `4005207c8ee3` | 2828 | 1151786 |
| INICIO | `0c051a3fc024` | 230 | 103957 |
| KV-SHEI | `9cb24635fe0b` | 86 | 43583 |

Se excluyen caches de compilación, dependencias vendorizadas bajo vendor/vendor-patches, datos de runtime y enlaces simbólicos. El código de aplicaciones externas presente en el árbol, cuando se inventaría, queda marcado aparte. En SHEILY hay 821 archivos clasificados como componentes externos/vendorizados dentro de los 2.828 inventariados; no son 2.828 capacidades propias.

El inventario inicial v1 se conserva. La versión v2 corrige la clasificación de inicio/evidence como código de producción, no como evidencia histórica, y la posición léxica de símbolos Rust. Los hashes de las fuentes no cambiaron.

## Regla de independencia

Los donantes solo sirven como origen de código y pruebas. No se propone instalar sus runtimes, invocar sus APIs, importar sus paquetes Python ni añadir dependencias Cargo que apunten fuera de M-SLIM. La prueba final de cualquier futuro port debe poder retirar ambos árboles donantes y seguir compilando/funcionando. Las referencias a esos árboles en esta carpeta son evidencia de auditoría, no dependencias del producto.

## Decisiones resumidas

| ID | Capacidad | Decisión | Destino propuesto, no creado |
|---|---|---|---|
| S01 | Índice de proyectos y símbolos | ADAPTAR P1 | `project_index` |
| S02 | Grafo de dependencias e impacto | ADAPTAR P1 | `project_index::dependency_graph` |
| S03 | Auditoría e integridad del proyecto | EXTRAER P1 | `project_audit` |
| S04 | Puertas de cierre y documentación verificable | ADAPTAR P1 | `validation::project_closure` |
| S05 | Recuperación híbrida con citas verificadas | EXTRAER P1 | `evidence_retrieval` |
| S06 | Memoria de trabajo acotada | EXTRAER P1 | `experience_memory::working_set` |
| S07 | Hechos, hipótesis, dudas y contradicciones | ADAPTAR P1 | `experience_memory::epistemic_state` |
| S08 | Consolidación estructurada de experiencias | ADAPTAR P2 | `experience_memory::consolidation` |
| S09 | Memoria de fallos y patrones | ADAPTAR P2 | `experience_memory::failure_patterns` |
| S10 | Minimum Space: historial y reducción de estado | ADAPTAR P2 | `experience_memory::event_reducer` |
| S11 | Selección de candidatos medidos | EXTRAER P1 | `adaptive_policy::candidate_ranker` |
| S12 | Búsqueda acotada de secuencias | ADAPTAR P2 | `adaptive_policy::bounded_planner` |
| S13 | Análisis documental y relaciones derivadas | FUSIONAR P2 | `document_graph + derived_relations` |
| S14 | Registro de capacidades y contratos | FUSIONAR P2 | `runtime_registry + typed_contracts` |
| S15 | Continuidad contextual con ámbito | FUSIONAR P2 | `session_store + continuity_contract` |
| S16 | Asesor sin autoridad implícita | EXTRAER P1 | `adaptive_policy::advisory_contract` |
| S17 | Memoria procedimental de intentos verificados | ADAPTAR P2 | `experience_memory::procedural_replay` |
| S18 | Portafolio de solvers y crédito causal | EXPERIMENTAL P3 | `calibration::numerical_models` |
| S19 | Geometría Pitágoras de TIDEX | EXPERIMENTAL P3 | `diagnostics::geometry` |
| S20 | Generación de corpus y entrenamiento auxiliar | EXPERIMENTAL P3 | `offline_validation::retrieval_corpus` |
| I01 | Admisión de aprendizaje | EXTRAER P1 | `adaptive_policy::learning_admission` |
| I02 | Evaluación en sombra, canary y revocación | FUSIONAR P1 | `calibration_transaction` |
| I03 | Vigilancia secuencial de degradación | EXTRAER P1 | `calibration::degradation_watch` |
| I04 | Estadística y confianza explícita | EXTRAER P1 | `validation::statistical_evidence` |
| I05 | Modelo continuo del coste observado | ADAPTAR P2 | `adaptive_policy::cost_model` |
| I06 | Rutas según régimen de carga | ADAPTAR P2 | `adaptive_policy::workload_router` |
| I07 | Detección de insuficiencia y cambio de dinámica | ADAPTAR P2 | `adaptive_policy::sufficiency_monitor` |
| I08 | Recuperación de experiencias operativas | ADAPTAR P1 | `experience_memory::profile_retrieval` |
| I09 | Reorganización del índice de experiencias | ADAPTAR P2 | `experience_memory::derived_index` |
| I10 | Registro transaccional de experiencia y aprendizaje | ADAPTAR P2 | `experience_memory::event_log` |
| I11 | Checkpoints de datos sin ejecución arbitraria | FUSIONAR P2 | `session_store + auxiliary_checkpoint` |
| I12 | Comparación causal y prevención de contaminación | ADAPTAR P2 | `validation::paired_evaluation` |
| I13 | Restricciones y decisiones admisibles | EXTRAER P1 | `adaptive_policy::constraints` |
| I14 | Estrategias de búsqueda reemplazables | ADAPTAR P2 | `adaptive_policy::search_strategy` |
| I15 | Aprendizaje auxiliar en cuarentena | ADAPTAR P2 | `adaptive_policy::shadow_tracker` |
| I16 | Evaluación de alternativas y composición | EXPERIMENTAL P3 | `adaptive_policy::proposal_lab` |
| I17 | Red modular y adaptadores entrenables | EXPERIMENTAL P3 | `optional_auxiliary_predictor` |
| I18 | Dinámica ondulatoria y geometría | EXPERIMENTAL P3 | `isolated_research_only` |
| I19 | Federación de evidencias y malla de eventos | NO PORTAR | `none` |
| I20 | Runtime de organismo y puente HTTP | NO PORTAR | `none` |

## Fichas de extracción y límites

### S01 · Índice de proyectos y símbolos

**Decisión:** ADAPTAR P1. **Destino propuesto:** `project_index`.

Localizar archivos, funciones, tipos y módulos del código que M-SLIM deba estudiar.

**Límite:** resource-index solo indexa .py y extrae nombres de alto nivel mediante texto. Para Rust hay que extraer el analizador de la consola y sus dependencias; no basta renombrar el índice.

**Aceptación exigida al port:** Comparar símbolos y rangos con un corpus Rust/Python conocido; resolver cambios y borrados sin devolver índices antiguos.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-resource-index-rs/src/lib.rs:280` · `build_module_index` · SHA-256 `d3f48a668c8126a0…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-cognitive-console-rs/src/inventory.rs:1` · SHA-256 `597b8c03b0f8a902…`.
- `SHEILY-COGNITIVA/auto-doc/autodoc/analyzer.py:1` · SHA-256 `dad56e384e8e021b…`.

### S02 · Grafo de dependencias e impacto

**Decisión:** ADAPTAR P1. **Destino propuesto:** `project_index::dependency_graph`.

Relacionar módulos y localizar qué pruebas o consumidores podrían afectar los cambios.

**Límite:** El grafo del auditor usa imports Python por regex; no es un grafo semántico multilenguaje ni demuestra llamadas ejecutadas.

**Aceptación exigida al port:** Distinguir imports, definiciones, llamadas y aristas inferidas; validar módulos relativos, macros y límites del análisis.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-repo-auditor-rs/src/graph.rs:15` · `build_dependency_graph` · SHA-256 `531cec8e2c891e4e…`.
- `SHEILY-COGNITIVA/auto-doc/autodoc/indexer.py:122` · `build_graph` · SHA-256 `140366ca6ba6a67e…`.

### S03 · Auditoría e integridad del proyecto

**Decisión:** EXTRAER P1. **Destino propuesto:** `project_audit`.

Inventariar código, registrar anomalías y sellar conjuntos de archivos mediante hashes.

**Límite:** Un certificado de integridad no certifica corrección funcional. Ejecutar escaneos fuera de la ruta de inferencia; no copiar rutas o exclusiones sin revisarlas.

**Aceptación exigida al port:** Detectar modificación, pérdida y duplicación de archivos; producir resultados deterministas sin leer secretos.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-repo-auditor-rs/src/inventory.rs:25` · `build_inventory` · SHA-256 `b85221c479214374…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-repo-auditor-rs/src/certify.rs:135` · `certify_state` · SHA-256 `46fdb80d2fc2aee1…`.

### S04 · Puertas de cierre y documentación verificable

**Decisión:** ADAPTAR P1. **Destino propuesto:** `validation::project_closure`.

Separar código presente, cableado, compilación, pruebas y evidencia de ejecución.

**Límite:** No copiar el generador como autoridad infalible; sus búsquedas son parcialmente heurísticas. Convertir controles útiles a herramientas propias o Rust sin dependencia de la instalación donante.

**Aceptación exigida al port:** Un módulo desconectado o una prueba omitida debe impedir un estado de validación completa.

**Fuentes locales:**
- `SHEILY-COGNITIVA/auto-doc/autodoc/closure.py:850` · `assess_closure` · SHA-256 `27d85f9a22da9db5…`.
- `SHEILY-COGNITIVA/auto-doc/autodoc/verify.py:1` · SHA-256 `104ad9f64df2ba9e…`.

### S05 · Recuperación híbrida con citas verificadas

**Decisión:** EXTRAER P1. **Destino propuesto:** `evidence_retrieval`.

Fusionar varios rankings, deduplicar por identidad/revisión y aceptar solo resultados cuya fuente canónica, ámbito y hash coincidan.

**Límite:** La fusión no crea embeddings ni un buscador semántico. Aporta ranking y validación de candidatos; debe extender document_graph, no reemplazar originales por resúmenes.

**Aceptación exigida al port:** Conservar rechazos de índice obsoleto, ámbito cruzado, duplicados, hash alterado y citas con límites UTF-8 inválidos.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/hybrid_retrieval.rs:337` · `weighted_rrf` · SHA-256 `7a20f52b88f0b448…`.

### S06 · Memoria de trabajo acotada

**Decisión:** EXTRAER P1. **Destino propuesto:** `experience_memory::working_set`.

Mantener entidades, ejemplos, notas e hipótesis de la operación actual con capacidad explícita.

**Límite:** Es un búfer lógico, no una KV-cache exacta ni un mecanismo de compresión de pesos.

**Aceptación exigida al port:** Probar límites, orden de expulsión, aislamiento de sesión y reconstrucción desde fuentes duraderas.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/working_memory.rs:26` · `WorkingMemoryBuffer` · SHA-256 `1b3adb985ca9ec43…`.

### S07 · Hechos, hipótesis, dudas y contradicciones

**Decisión:** ADAPTAR P1. **Destino propuesto:** `experience_memory::epistemic_state`.

Registrar qué está respaldado, qué sigue siendo hipótesis y qué preguntas impiden cerrar un trabajo.

**Límite:** Las etiquetas y reglas de confianza no verifican automáticamente la verdad ni eliminan alucinaciones. El código de revisión contiene reglas de texto y requiere validación del dominio.

**Aceptación exigida al port:** Una contradicción no debe aumentar confianza; evitar que un recuerdo se corrobore a sí mismo.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/epistemic.rs:76` · `EpistemicWorkingState` · SHA-256 `2244a91e2ea8a6c3…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-cortex-rs/src/belief_revision.rs:54` · `revise_beliefs` · SHA-256 `9fd6604d73eafb52…`.

### S08 · Consolidación estructurada de experiencias

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::consolidation`.

Derivar hechos y procedimientos de episodios conservando referencias a las fuentes.

**Límite:** El módulo está acoplado a escritores, ámbitos e inferencia de SHEILY. Extraer contratos y lógica, no importar la memoria canónica ni sus servicios. La síntesis semántica no es preservación bit a bit.

**Aceptación exigida al port:** Separar preparación sin escrituras de persistencia; revalidar fuentes al confirmar; impedir referencias inexistentes.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/structured_consolidation.rs:457` · `prepare_structured_consolidation` · SHA-256 `1e382a642d5bacd9…`.

### S09 · Memoria de fallos y patrones

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::failure_patterns`.

Reconocer configuraciones o procedimientos asociados a fallos anteriores y detectar candidatos conflictivos.

**Límite:** Hay coincidencias textuales y kernels sobre vectores aportados por el llamador; no diagnostican por sí mismos todas las causas de un fallo.

**Aceptación exigida al port:** Una coincidencia no puede bloquear un perfil correcto sin evidencia contextual; separar memoria histórica de autoridad de ejecución.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/failure_memory.rs:33` · `FailureMemory` · SHA-256 `8f17b1e7a313a644…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/pattern_memory.rs:1` · SHA-256 `2a93c520a7527e60…`.

### S10 · Minimum Space: historial y reducción de estado

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::event_reducer`.

Reconstruir estado derivado a partir de eventos y detectar/quarantinar sufijos corruptos de una cadena.

**Límite:** Es un subsistema de experiencia, evidencia y estado cognitivo, muy ligado al motor de turnos. No es un compresor lossless de tensores, KV ni pesos de 2 bits.

**Aceptación exigida al port:** Mismos eventos deben producir mismo estado; una corrupción no puede reescribir silenciosamente historia válida ni alterar KV canónica.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-turn-engine-rs/src/minimum_space_state_transition.rs:1` · SHA-256 `a86278476774c3fb…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-turn-engine-rs/src/minimum_space_persistence.rs:1` · SHA-256 `33ad26fb5d9ae8e5…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-turn-engine-rs/src/minimum_space_chain_integrity.rs:780` · `verify_experience_chain` · SHA-256 `55be779c851ed128…`.

### S11 · Selección de candidatos medidos

**Decisión:** EXTRAER P1. **Destino propuesto:** `adaptive_policy::candidate_ranker`.

Ordenar perfiles y estrategias candidatos con señales, evidencias y límites aportados por M-SLIM.

**Límite:** La puntuación combina señales; no demuestra rendimiento físico. Mantener recomendaciones separadas de permisos y usar solo métricas observadas cuando se declare una mejora real.

**Aceptación exigida al port:** Rechazar valores no finitos, candidatos sin procedencia y promociones que no hayan pasado las puertas existentes.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-cognitive-mechanisms-rs/src/hypothesis.rs:1` · SHA-256 `1755203803cb1577…`.

### S12 · Búsqueda acotada de secuencias

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::bounded_planner`.

Explorar alternativas de operaciones con límites de pasos y una cinta de decisiones reproducible.

**Límite:** Necesita un generador y un evaluador propios. No hace más inteligente al LLM automáticamente ni justifica aproximar la atención en modo exacto.

**Aceptación exigida al port:** Límite verificable de trabajo y repetibilidad; distinguir puntuaciones de modelo de medidas reales.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-cognitive-mechanisms-rs/src/trajectory_search.rs:710` · `search_trajectory` · SHA-256 `d1328854453b3abe…`.

### S13 · Análisis documental y relaciones derivadas

**Decisión:** FUSIONAR P2. **Destino propuesto:** `document_graph + derived_relations`.

Añadir entidades, relaciones e hipótesis al grafo de páginas original de M-SLIM.

**Límite:** No copiar un segundo almacén de documentos; relaciones extraídas son derivadas y deben apuntar a páginas/rangos verificables.

**Aceptación exigida al port:** Toda relación debe llevar fuente y estado de confianza; una relación derivada no puede sustituir al texto original.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-cortex-rs/src/document_analysis.rs:45` · `DocumentKnowledgeGraph` · SHA-256 `2b546933acacde3c…`.

### S14 · Registro de capacidades y contratos

**Decisión:** FUSIONAR P2. **Destino propuesto:** `runtime_registry + typed_contracts`.

Describir qué funciones ofrece cada componente y vincularlas con implementación, versión y validación.

**Límite:** M-SLIM ya tiene registro de runtimes y adaptadores. Extraer mecanismos de verificación; no copiar identidades, permisos o contratos de SHEILY como dependencia.

**Aceptación exigida al port:** Detectar capacidades declaradas sin consumidor o sin implementación; contratos versionados y fallos explícitos.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-runtime-registry-rs/src/lib.rs:1` · SHA-256 `6717830f8b147173…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-turn-contracts-rs/src/lib.rs:1` · SHA-256 `97f0b9b338a84374…`.

### S15 · Continuidad contextual con ámbito

**Decisión:** FUSIONAR P2. **Destino propuesto:** `session_store + continuity_contract`.

Conservar vínculos entre turnos, documentos y referencias dentro de una sesión.

**Límite:** No copiar la autoridad de SHEILY ni confundir continuidad semántica con restaurar exactamente el estado K/V de otro modelo.

**Aceptación exigida al port:** Separación entre proyectos/sesiones, revisiones de documento y modelo; rechazo de snapshots incompatibles.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-context-authority-rs/src/lib.rs:133` · `TurnContextEnvelope` · SHA-256 `767ffd68be0b2c92…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/session_continuity.rs:1` · SHA-256 `57d1ca0c87a1b225…`.

### S16 · Asesor sin autoridad implícita

**Decisión:** EXTRAER P1. **Destino propuesto:** `adaptive_policy::advisory_contract`.

Hacer que el optimizador recomiende o pida revisión sin poder ejecutar, cambiar código o promover perfiles por sí mismo.

**Límite:** Esta proyección GPEM no implementa reparto de RAM/VRAM ni un gestor físico de recursos. Esa afirmación anterior no queda respaldada por este módulo.

**Aceptación exigida al port:** Una recomendación no puede convertirse en permiso ni modificar la memoria canónica.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-memory-engine-rs/src/gpem_authority.rs:20` · `gpem_authority_projection` · SHA-256 `3d7e8941d3ff1279…`.

### S17 · Memoria procedimental de intentos verificados

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::procedural_replay`.

Recordar qué procedimientos funcionaron a partir de recibos auténticos, sin inventar un segundo historial.

**Límite:** Los esquemas actuales corresponden a solvers TIDEX. Hay que definir recibos propios de M-SLIM y mantener límites del búfer.

**Aceptación exigida al port:** No reconstruir éxitos a partir de stdout no autenticado; replay estable y rechazo de esquemas desconocidos.

**Fuentes locales:**
- `SHEILY-COGNITIVA/tidex/src/learning/procedural_memory.rs:1` · SHA-256 `8a625efc2cc6ffce…`.
- `SHEILY-COGNITIVA/tidex/src/learning/procedural_replay.rs:127` · `rebuild_from_authenticated_receipt` · SHA-256 `c74fbf174ca6cb72…`.

### S18 · Portafolio de solvers y crédito causal

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `calibration::numerical_models`.

Comparar solucionadores o ajustes numéricos con residuos recomputados y atribución de resultados.

**Límite:** Es un portafolio de mínimos cuadrados, no un selector directo CPU/GPU/LLM. Solo compensa extraerlo si el problema de calibración necesita esos solvers.

**Aceptación exigida al port:** Residuos independientes, cotas de recursos y evaluación fuera de muestra antes de atribuir una mejora.

**Fuentes locales:**
- `SHEILY-COGNITIVA/tidex/src/learning/solver_portfolio.rs:1` · SHA-256 `8132a2c0f8869e53…`.
- `SHEILY-COGNITIVA/tidex/src/learning/causal_credit.rs:201` · `estimate_causal_credit` · SHA-256 `658aa4544394a5d5…`.

### S19 · Geometría Pitágoras de TIDEX

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `diagnostics::geometry`.

Comparar normas L1/L2 y analizar perturbaciones en experimentos numéricos.

**Límite:** evaluate_and_correct calcula normas y razones; project_to_geodesic reescala el vector. No comprime sin pérdida; la reescala modifica sus valores y no debe tocar pesos/KV exactos.

**Aceptación exigida al port:** Separar diagnóstico de transformación; cualquier transformación debe declararse aproximada y validarse, nunca presentarse como paridad exacta.

**Fuentes locales:**
- `SHEILY-COGNITIVA/tidex/src/analysis/pythagoras_topology.rs:53` · `PythagorasStaircaseMetric` · SHA-256 `9777760c14de8b1c…`.

### S20 · Generación de corpus y entrenamiento auxiliar

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `offline_validation::retrieval_corpus`.

Preparar corpus y entrenar un recuperador auxiliar cuando existan datos y una evaluación adecuada.

**Límite:** No importar el stack entero de entrenamiento/inferencia. Entrenar cambia parámetros; no es compresión transparente ni mejora garantizada en el hardware actual.

**Aceptación exigida al port:** Datos separados de evaluación, trazabilidad y comparación con búsqueda exacta antes de incorporar un recuperador entrenado.

**Fuentes locales:**
- `SHEILY-COGNITIVA/cognition/native/sheily-dataset-compiler-rs/src/retrieval_curriculum.rs:1` · SHA-256 `0e967159499c187a…`.
- `SHEILY-COGNITIVA/cognition/native/sheily-training-kernels-rs/src/retrieval_encoder_training.rs:1` · SHA-256 `98bf0ed3f18123b9…`.

### I01 · Admisión de aprendizaje

**Decisión:** EXTRAER P1. **Destino propuesto:** `adaptive_policy::learning_admission`.

Decidir qué observaciones pueden actualizar el modelo auxiliar de rendimiento y cuáles quedan bloqueadas.

**Límite:** Es un contrato de aprendizaje de dinámica, no autoridad de ejecución. Portar las reglas a tipos propios Rust y a métricas reales de M-SLIM.

**Aceptación exigida al port:** Datos no finitos, observaciones ambiguas y muestras de evaluación deben quedar fuera del aprendizaje activo.

**Fuentes locales:**
- `INICIO/inicio/adaptation/learning_admission.py:147` · `learning_admission_for_evidence` · SHA-256 `1dd7900e95601839…`.

### I02 · Evaluación en sombra, canary y revocación

**Decisión:** FUSIONAR P1. **Destino propuesto:** `calibration_transaction`.

Probar perfiles candidatos, activarlos de forma acotada y restaurar el anterior cuando fallen.

**Límite:** M-SLIM ya tiene CalibrationTransaction. La búsqueda estática no encontró usos de ese tipo fuera de su propio archivo/pruebas; aprovechar y cablear ese núcleo antes de duplicarlo.

**Aceptación exigida al port:** Ensayo real con fallo de canary y reinicio; restauración verificada y ninguna promoción a partir de métricas inventadas.

**Fuentes locales:**
- `INICIO/inicio/adaptation/structural_shadow.py:1` · SHA-256 `883457a0630ec702…`.
- `INICIO/inicio/adaptation/structural_canary.py:1` · SHA-256 `3626b632fce184ad…`.
- `INICIO/inicio/adaptation/structural_revoke.py:1` · SHA-256 `444c8ad88f753805…`.
- `INICIO/inicio/governance/promotion_manager.py:1` · SHA-256 `a2d116a155a309bc…`.

### I03 · Vigilancia secuencial de degradación

**Decisión:** EXTRAER P1. **Destino propuesto:** `calibration::degradation_watch`.

Vigilar si una configuración nueva empeora una pérdida medida respecto a su referencia.

**Límite:** El módulo combina un proceso de capital y CUSUM; su docstring limita la garantía y lo declara asesor. Adaptar el significado de pérdida a latencia/error con supuestos explícitos.

**Aceptación exigida al port:** Paridad Python/Rust sobre secuencias registradas y rechazo de NaN, infinito y pérdidas inválidas; una alarma no autoriza una mutación.

**Fuentes locales:**
- `INICIO/inicio/adaptation/degradation_eprocess.py:23` · `SequentialDegradationWatch` · SHA-256 `4048e91ae44a5359…`.

### I04 · Estadística y confianza explícita

**Decisión:** EXTRAER P1. **Destino propuesto:** `validation::statistical_evidence`.

Separar insuficiencia de datos, alarmas secuenciales y tipos de cobertura estadística.

**Límite:** Las garantías requieren hipótesis sobre los datos; la cobertura conformal adaptativa no equivale a una garantía exacta por petición.

**Aceptación exigida al port:** Pruebas de fronteras, warmup y datos dependientes; registrar los supuestos y no renombrar estimaciones como certezas.

**Fuentes locales:**
- `INICIO/inicio/evidence/statistics.py:1` · SHA-256 `acf4a63a59323461…`.
- `INICIO/inicio/evidence/reliability.py:24` · `BettingEProcess` · SHA-256 `55ed5bb6b1dc063e…`.

### I05 · Modelo continuo del coste observado

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::cost_model`.

Aprender un predictor pequeño del coste de operaciones a partir de longitud de contexto, lote y telemetría.

**Límite:** La aplicación a coste de inferencia es una propuesta de adaptación, no una capacidad medida ya en M-SLIM. El modelo continuo no es un LLM ni almacena documentos sin pérdida.

**Aceptación exigida al port:** Predicción contrastada con medidas reales y conjuntos reservados; límites de recursos y detección de cambio de distribución.

**Fuentes locales:**
- `INICIO/inicio/cognition/cognitive_engines/m_continuous.py:1245` · `apply_rls_theta_update` · SHA-256 `0bc8d1b7781e8eef…`.
- `INICIO/inicio/cognition/cognitive_engines/m_continuous.py:1295` · `GaussianGenerativeModel` · SHA-256 `0bc8d1b7781e8eef…`.

### I06 · Rutas según régimen de carga

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::workload_router`.

Distinguir regímenes recurrentes de trabajo y reutilizar modelos auxiliares especializados.

**Límite:** Adaptar identidades y contratos a hardware/carga. Cambiar de LLM no conserva necesariamente respuestas ni KV: debe ser una decisión explícita, no una promesa de paridad.

**Aceptación exigida al port:** No aplicar una ruta conocida a hardware/modelo incompatible; probar regímenes nuevos, agotamiento de capacidad y retorno a uno anterior.

**Fuentes locales:**
- `INICIO/inicio/adaptation/regime_composition.py:350` · `RegimeRoutedGenerativeModel` · SHA-256 `385c4009f51f8c20…`.
- `INICIO/inicio/substrates/routing_substrate.py:1` · SHA-256 `68de3cba5d350015…`.

### I07 · Detección de insuficiencia y cambio de dinámica

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::sufficiency_monitor`.

Saber cuándo el predictor auxiliar dejó de describir las medidas y conviene abstenerse o recalibrar.

**Límite:** No diagnostica por sí solo todas las causas físicas. Sus umbrales y observables deben rediseñarse para rendimiento y exactitud de inferencia.

**Aceptación exigida al port:** Cambios de carga, temperatura o ruta deben evaluarse con medidas de referencia y no confundirse con ganancias persistentes.

**Fuentes locales:**
- `INICIO/inicio/cognition/cognitive_engines/m_continuous.py:119` · `DynamicSufficiencyMonitor` · SHA-256 `0bc8d1b7781e8eef…`.

### I08 · Recuperación de experiencias operativas

**Decisión:** ADAPTAR P1. **Destino propuesto:** `experience_memory::profile_retrieval`.

Recuperar experiencias compatibles con ambiente, régimen y esquema para sugerir perfiles ya probados.

**Límite:** La recuperación es asesora y puede abstenerse. No es la recuperación de tokens K/V ni un permiso para activar una estrategia.

**Aceptación exigida al port:** Identidad ausente, evidencia insegura o índice obsoleto deben causar abstención; nunca crear candidatos que no existían.

**Fuentes locales:**
- `INICIO/inicio/memory/memory_retrieval.py:75` · `ContextualMemoryRetriever` · SHA-256 `0004f235bb0186e6…`.

### I09 · Reorganización del índice de experiencias

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::derived_index`.

Reorganizar proyecciones para recuperar historia útil con menos trabajo.

**Límite:** Tratar el índice como derivado y reconstruible. No extrapolar reorganización semántica a borrar información canónica o comprimir arbitrariamente K/V.

**Aceptación exigida al port:** Índice obsoleto detectable, reconstrucción y mismas identidades de evidencia tras reorganizar.

**Fuentes locales:**
- `INICIO/inicio/memory/plastic_memory_reorganizer.py:1` · SHA-256 `ebdf9bd46fe53953…`.

### I10 · Registro transaccional de experiencia y aprendizaje

**Decisión:** ADAPTAR P2. **Destino propuesto:** `experience_memory::event_log`.

Relacionar observaciones, decisiones de admisión y mutaciones con integridad y prevención de fugas de evaluación.

**Límite:** No copiar un segundo almacén de KV ni sus identidades de organismo. Definir un único registro de evidencia operacional propio.

**Aceptación exigida al port:** Atomicidad, caída entre eventos, duplicados e identificación de muestras reservadas.

**Fuentes locales:**
- `INICIO/inicio/persistence/event_store.py:86` · `EventStore` · SHA-256 `538dc7f269cd54e0…`.

### I11 · Checkpoints de datos sin ejecución arbitraria

**Decisión:** FUSIONAR P2. **Destino propuesto:** `session_store + auxiliary_checkpoint`.

Validar versiones y configuración antes de restaurar modelos auxiliares y estados de optimización.

**Límite:** El formato donante usa árboles JSON y arrays NPZ, no pickle. Conservar el principio, no añadir Python al runtime ni sustituir el formato de KV exacta.

**Aceptación exigida al port:** Rechazar formas, tipos, versiones y configuraciones incompatibles; ninguna carga ejecutable desde un checkpoint.

**Fuentes locales:**
- `INICIO/inicio/persistence/neural_checkpoint.py:110` · `prepare_neural_restore` · SHA-256 `7a1d8d05e755eb2b…`.
- `INICIO/inicio/persistence/state_store.py:1` · SHA-256 `a6c980b5f3fc904a…`.

### I12 · Comparación causal y prevención de contaminación

**Decisión:** ADAPTAR P2. **Destino propuesto:** `validation::paired_evaluation`.

Comparar políticas sin mezclar entrenamiento y evaluación; agrupar episodios o bloques temporales.

**Límite:** La evaluación off-policy necesita cobertura y supuestos de identificación. No convertir historial sesgado en una mejora causal demostrada.

**Aceptación exigida al port:** Abstención cuando falta soporte, evaluación retenida y resultados reproducibles frente a comparación pareada directa.

**Fuentes locales:**
- `INICIO/inicio/evidence/causal.py:162` · `OffPolicyEvaluator` · SHA-256 `d754e96c3ebc8733…`.
- `INICIO/inicio/evidence/boundary.py:1` · SHA-256 `cfb26f73ed27a1bc…`.

### I13 · Restricciones y decisiones admisibles

**Decisión:** EXTRAER P1. **Destino propuesto:** `adaptive_policy::constraints`.

Filtrar decisiones por límites explícitos de memoria, exactitud, coste y permisos.

**Límite:** Copiar la separación entre objetivos y restricciones; no la autoridad humana específica ni un nuevo sistema de permisos paralelo.

**Aceptación exigida al port:** Un optimizador no puede relajar el modo exacto, el presupuesto o una prohibición para mejorar su puntuación.

**Fuentes locales:**
- `INICIO/inicio/cognition/cognitive_engines/m4_governance.py:380` · `GovernanceGateway` · SHA-256 `0c72b5c2df37f2a7…`.

### I14 · Estrategias de búsqueda reemplazables

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::search_strategy`.

Elegir cómo buscar perfiles o planes dentro de un presupuesto computacional.

**Límite:** Los certificados distinguen búsqueda exacta de aproximada. Esa distinción del planificador no autoriza atención aproximada ni pérdida de tokens.

**Aceptación exigida al port:** Referencia exhaustiva en problemas pequeños, límites de exploración y rotulado correcto de resultados incompletos.

**Fuentes locales:**
- `INICIO/inicio/cognition/cognitive_engines/policy_search.py:207` · `ExactPolicySearchStrategy` · SHA-256 `031489f86ea6108c…`.
- `INICIO/inicio/cognition/cognitive_engines/policy_search.py:221` · `BeamPolicySearchStrategy` · SHA-256 `031489f86ea6108c…`.
- `INICIO/inicio/cognition/cognitive_engines/planner_plasticity.py:1` · SHA-256 `0404e1fe354871ee…`.

### I15 · Aprendizaje auxiliar en cuarentena

**Decisión:** ADAPTAR P2. **Destino propuesto:** `adaptive_policy::shadow_tracker`.

Seguir ajustando candidatos auxiliares aislados cuando la actualización activa no es admisible.

**Límite:** No aprende el LLM canónico en secreto ni concede autoridad. Separar sus datos y parámetros del perfil activo.

**Aceptación exigida al port:** Ningún cambio observable en el perfil vivo durante la cuarentena; activación solo tras evaluación independiente.

**Fuentes locales:**
- `INICIO/inicio/adaptation/quarantine_tracker.py:152` · `QuarantineSafeTrackerBank` · SHA-256 `26230fe959d1118b…`.

### I16 · Evaluación de alternativas y composición

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `adaptive_policy::proposal_lab`.

Proponer y comparar alternativas antes de un ensayo físico acotado.

**Límite:** Los contrafactuales son predicciones, no mediciones. No cumplen por sí solos el requisito de resultados reales; no deben entrar como benchmarks.

**Aceptación exigida al port:** Etiquetar predicción/observación y exigir ensayo real separado para cualquier afirmación de mejora.

**Fuentes locales:**
- `INICIO/inicio/cognition/parallel_counterfactual_engine.py:33` · `ParallelCounterfactualEngine` · SHA-256 `4e25c407815cc9a7…`.
- `INICIO/inicio/adaptation/composition_evaluator.py:1` · SHA-256 `2b3e5ceac9e90f59…`.

### I17 · Red modular y adaptadores entrenables

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `optional_auxiliary_predictor`.

Explorar un predictor auxiliar modular con capacidad y parámetros aislados.

**Límite:** El propio módulo advierte que preservar parámetros no garantiza predicciones invariantes cuando cambia el router. No es ampliación automática de inteligencia ni compresión sin pérdida.

**Aceptación exigida al port:** Capacidad máxima, aislamiento, restauración y evaluación de cambios de salida; nunca modificar pesos de inferencia exacta implícitamente.

**Fuentes locales:**
- `INICIO/inicio/substrates/continual_modular_network.py:210` · `ContinualModularNetwork` · SHA-256 `14d56890d6d57513…`.

### I18 · Dinámica ondulatoria y geometría

**Decisión:** EXPERIMENTAL P3. **Destino propuesto:** `isolated_research_only`.

Conservar como líneas de investigación numérica, separadas del producto exacto.

**Límite:** La existencia de estos módulos no demuestra una compresión o atención superior en M-SLIM. No incorporar al núcleo hasta medir utilidad, recursos y desviación.

**Aceptación exigida al port:** Benchmark independiente con datos reales, ablation y comparación exacta; sin promesas basadas en el nombre del algoritmo.

**Fuentes locales:**
- `INICIO/inicio/substrates/wave_substrate/weight_soliton.py:1` · SHA-256 `45f815a363b7b909…`.
- `INICIO/inicio/substrates/wave_substrate/attention_bridge.py:1` · SHA-256 `2440bd7a65a7bb64…`.
- `INICIO/inicio/cognition/lie_invariant_geometry.py:1` · SHA-256 `19deb884c736bfef…`.

### I19 · Federación de evidencias y malla de eventos

**Decisión:** NO PORTAR. **Destino propuesto:** `none`.

No aporta una necesidad inmediata al motor local independiente solicitado.

**Límite:** Importar una malla distribuida añade identidades, red y complejidad ajenas al objetivo. No se propone conexión alguna con los proyectos donantes.

**Aceptación exigida al port:** Mantener fuera de dependencias, servicios y despliegue de M-SLIM.

**Fuentes locales:**
- `INICIO/inicio/federation/federated_evidence.py:1` · SHA-256 `3225818007babb38…`.
- `INICIO/inicio/federation/swarm_event_mesh.py:1` · SHA-256 `251d9800a86d6db2…`.

### I20 · Runtime de organismo y puente HTTP

**Decisión:** NO PORTAR. **Destino propuesto:** `none`.

Extraer algoritmos pequeños de sus módulos; no trasladar el organismo ni usar su servidor.

**Límite:** Copiar o conectar el runtime completo contradice la independencia pedida. El runtime propio de M-SLIM conserva todo el control.

**Aceptación exigida al port:** El resultado debe compilar y funcionar con las carpetas SHEILY-COGNITIVA e INICIO ausentes y sin sus endpoints.

**Fuentes locales:**
- `INICIO/inicio/runtime/organism.py:1` · SHA-256 `7ce250f5796ccd13…`.
- `INICIO/inicio/integration/external_transition_gateway.py:1` · SHA-256 `a43f234a3b51a077…`.
- `INICIO/inicio/integration/http_service.py:1` · SHA-256 `c0b3b0780750e463…`.

## Qué M-SLIM ya tiene y no conviene duplicar

| Fuente M-SLIM | Evidencia estática observada | Acción recomendada |
|---|---|---|
| `src/document_graph.rs` | Páginas originales, hashes, presupuesto y candidatos diferidos; referencias en CLI documental. | Extender con fusión de candidatos y relaciones derivadas. |
| `src/calibration_transaction.rs` | Propuesta, sombra, canary, promoción, rollback; las 4 referencias a `CalibrationTransaction` encontradas están en este mismo archivo. | Verificar/cablear su uso por el producto antes de importar otra máquina de estados. |
| `src/shadow_metrics.rs` | Referenciado por `torch_runtime.rs`. | Añadir observaciones y alarmas sin darles autoridad sobre exactitud. |
| `src/adaptive_extent.rs` | Eje reversible utilizado por varios componentes, incluidas rutas CPU/GPU. | No sustituirlo por memoria semántica o geometría experimental. |
| `src/session_store.rs` y `src/continuity_contract.rs` | Referencias en coordinador y runtime-worker. | Añadir estado auxiliar sin duplicar sesiones ni alterar KV. |
| `runtime_registry`, catálogos, adaptadores y host | Implementaciones/exportaciones presentes. | Reutilizar el plano de ejecución propio. Su ejecución completa no se certificó aquí. |

Una referencia estática es evidencia de cableado textual, no prueba de ejecución. La búsqueda de CalibrationTransaction no descarta todos los usos indirectos posibles, pero no aporta evidencia de un consumidor fuera del módulo.

## Orden de adaptación recomendado

1. **Recuperación verificable y conocimiento del proyecto.** S01-S05. Reutilizar el grafo documental existente; añadir extracción real de símbolos Rust, revisiones de fuente y fusión con citas. No empezar por un Cortex general.
2. **Experiencias y estado de tarea.** S06-S09, S15, S17, I08-I10. Separar datos originales, índices derivados y estado epistemológico. Mantener un único propietario para cada dato.
3. **Control adaptativo medido.** S11, S16, I01-I07, I13-I15. Empezar con consejos sobre perfiles existentes. Primero medición en sombra, después canary y promoción explícita; no reescribir prompts ni pesos.
4. **Pruebas de utilidad y regresión.** S03-S04, I04, I12. Medir memoria pico, tiempos de prefill/decode, tráfico de disco y calidad/recuperación por separado. Exigir pruebas de paridad cuando se declare exactitud.
5. **Investigación opcional aislada.** S12, S18-S20, I16-I18. Solo avanzar cuando una necesidad concreta y medidas reales justifiquen el coste. Predicción o simulación no cuenta como benchmark físico.

## Condiciones que no deben perderse

- Datos canónicos, pesos y K/V exactos no se sustituyen por una síntesis semántica. Las proyecciones auxiliares deben poder reconstruirse o declararse incompletas.
- No copiar credenciales, estados personales, servicios, rutas absolutas o dependencias transitivas del donante. Registrar procedencia del código y revisar los metadatos de licencia antes de distribuirlo; esta revisión no emite una conclusión jurídica.
- Las herramientas de auditoría y aprendizaje auxiliar no deben introducir recorridos completos, hashes totales o entrenamiento por token en el camino crítico.
- Los cambios de modelo, precisión y algoritmo que alteren salidas no se presentan como optimizaciones exactas transparentes.
- El nombre de un algoritmo no demuestra una capacidad: Minimum Space no es compresión universal; Pitágoras en TIDEX es geometría L1/L2; la dinámica continua de INICIO no es un LLM.

## Estado de esta entrega

Se han creado únicamente artefactos de auditoría dentro de esta carpeta de KV-SHEI. No se han conectado proyectos, portado módulos funcionales, alterado el runtime, ejecutado suites de los donantes, hecho commit ni push. No se modificaron intencionadamente los árboles originales. La comprobación final de hashes está en `source_preservation_check.json`.

El registro `COMPONENTES.md` asigna una disposición a los 38 crates nativos principales de SHEILY y a los 85 módulos de implementación Python de INICIO. `source_inventory.v2.csv` y `.json` mantienen el detalle por archivo, incluidos otros subsistemas, herramientas, tests y componentes externos. `capability_matrix.json` contiene las 40 fichas y sus hashes completos.
