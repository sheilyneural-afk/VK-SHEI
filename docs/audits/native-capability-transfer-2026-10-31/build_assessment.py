#!/usr/bin/env python3
"""Build the reviewed transfer matrix and complete structural component register.
This produces audit documents only. It does not port or execute donor code.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'source_inventory.v2.json').read_text())
LOOKUP = {(project, item['path']): item for project, snap in DATA['projects'].items() for item in snap['sources']}
ROWS = []
S = 'SHEILY-COGNITIVA'
I = 'INICIO'
N = 'cognition/native/'

def add(identifier, name, project, sources, decision, target, benefit, limit, gate):
    evidence = []
    for source in sources:
        path, _, symbol = source.partition('|')
        item = LOOKUP[(project, path)]
        found = [entry for entry in item['symbols'] if entry['name'] == symbol] if symbol else []
        if symbol and not found:
            raise ValueError(f'Missing symbol: {project}/{source}')
        evidence.append({'project': project, 'path': path, 'symbol': symbol or None,
                         'line': found[0]['line'] if found else 1, 'sha256': item['sha256'],
                         'static_tests_in_file': item['static_test_count']})
    ROWS.append({'id': identifier, 'name': name, 'decision': decision, 'proposed_target': target,
                 'benefit': benefit, 'limit': limit, 'acceptance_gate': gate, 'sources': evidence,
                 'status': 'source_reviewed_or_structurally_indexed; not_ported; not_runtime_certified'})

add('S01','Índice de proyectos y símbolos',S,[N+'sheily-resource-index-rs/src/lib.rs|build_module_index',N+'sheily-cognitive-console-rs/src/inventory.rs','auto-doc/autodoc/analyzer.py'],
    'ADAPTAR P1','project_index','Localizar archivos, funciones, tipos y módulos del código que M-SLIM deba estudiar.',
    'resource-index solo indexa .py y extrae nombres de alto nivel mediante texto. Para Rust hay que extraer el analizador de la consola y sus dependencias; no basta renombrar el índice.',
    'Comparar símbolos y rangos con un corpus Rust/Python conocido; resolver cambios y borrados sin devolver índices antiguos.')
add('S02','Grafo de dependencias e impacto',S,[N+'sheily-repo-auditor-rs/src/graph.rs|build_dependency_graph','auto-doc/autodoc/indexer.py|build_graph'],
    'ADAPTAR P1','project_index::dependency_graph','Relacionar módulos y localizar qué pruebas o consumidores podrían afectar los cambios.',
    'El grafo del auditor usa imports Python por regex; no es un grafo semántico multilenguaje ni demuestra llamadas ejecutadas.',
    'Distinguir imports, definiciones, llamadas y aristas inferidas; validar módulos relativos, macros y límites del análisis.')
add('S03','Auditoría e integridad del proyecto',S,[N+'sheily-repo-auditor-rs/src/inventory.rs|build_inventory',N+'sheily-repo-auditor-rs/src/certify.rs|certify_state'],
    'EXTRAER P1','project_audit','Inventariar código, registrar anomalías y sellar conjuntos de archivos mediante hashes.',
    'Un certificado de integridad no certifica corrección funcional. Ejecutar escaneos fuera de la ruta de inferencia; no copiar rutas o exclusiones sin revisarlas.',
    'Detectar modificación, pérdida y duplicación de archivos; producir resultados deterministas sin leer secretos.')
add('S04','Puertas de cierre y documentación verificable',S,['auto-doc/autodoc/closure.py|assess_closure','auto-doc/autodoc/verify.py'],
    'ADAPTAR P1','validation::project_closure','Separar código presente, cableado, compilación, pruebas y evidencia de ejecución.',
    'No copiar el generador como autoridad infalible; sus búsquedas son parcialmente heurísticas. Convertir controles útiles a herramientas propias o Rust sin dependencia de la instalación donante.',
    'Un módulo desconectado o una prueba omitida debe impedir un estado de validación completa.')
add('S05','Recuperación híbrida con citas verificadas',S,[N+'sheily-memory-engine-rs/src/hybrid_retrieval.rs|weighted_rrf'],
    'EXTRAER P1','evidence_retrieval','Fusionar varios rankings, deduplicar por identidad/revisión y aceptar solo resultados cuya fuente canónica, ámbito y hash coincidan.',
    'La fusión no crea embeddings ni un buscador semántico. Aporta ranking y validación de candidatos; debe extender document_graph, no reemplazar originales por resúmenes.',
    'Conservar rechazos de índice obsoleto, ámbito cruzado, duplicados, hash alterado y citas con límites UTF-8 inválidos.')
add('S06','Memoria de trabajo acotada',S,[N+'sheily-memory-engine-rs/src/working_memory.rs|WorkingMemoryBuffer'],
    'EXTRAER P1','experience_memory::working_set','Mantener entidades, ejemplos, notas e hipótesis de la operación actual con capacidad explícita.',
    'Es un búfer lógico, no una KV-cache exacta ni un mecanismo de compresión de pesos.',
    'Probar límites, orden de expulsión, aislamiento de sesión y reconstrucción desde fuentes duraderas.')
add('S07','Hechos, hipótesis, dudas y contradicciones',S,[N+'sheily-memory-engine-rs/src/epistemic.rs|EpistemicWorkingState',N+'sheily-cortex-rs/src/belief_revision.rs|revise_beliefs'],
    'ADAPTAR P1','experience_memory::epistemic_state','Registrar qué está respaldado, qué sigue siendo hipótesis y qué preguntas impiden cerrar un trabajo.',
    'Las etiquetas y reglas de confianza no verifican automáticamente la verdad ni eliminan alucinaciones. El código de revisión contiene reglas de texto y requiere validación del dominio.',
    'Una contradicción no debe aumentar confianza; evitar que un recuerdo se corrobore a sí mismo.')
add('S08','Consolidación estructurada de experiencias',S,[N+'sheily-memory-engine-rs/src/structured_consolidation.rs|prepare_structured_consolidation'],
    'ADAPTAR P2','experience_memory::consolidation','Derivar hechos y procedimientos de episodios conservando referencias a las fuentes.',
    'El módulo está acoplado a escritores, ámbitos e inferencia de SHEILY. Extraer contratos y lógica, no importar la memoria canónica ni sus servicios. La síntesis semántica no es preservación bit a bit.',
    'Separar preparación sin escrituras de persistencia; revalidar fuentes al confirmar; impedir referencias inexistentes.')
add('S09','Memoria de fallos y patrones',S,[N+'sheily-memory-engine-rs/src/failure_memory.rs|FailureMemory',N+'sheily-memory-engine-rs/src/pattern_memory.rs'],
    'ADAPTAR P2','experience_memory::failure_patterns','Reconocer configuraciones o procedimientos asociados a fallos anteriores y detectar candidatos conflictivos.',
    'Hay coincidencias textuales y kernels sobre vectores aportados por el llamador; no diagnostican por sí mismos todas las causas de un fallo.',
    'Una coincidencia no puede bloquear un perfil correcto sin evidencia contextual; separar memoria histórica de autoridad de ejecución.')
add('S10','Minimum Space: historial y reducción de estado',S,[N+'sheily-turn-engine-rs/src/minimum_space_state_transition.rs',N+'sheily-turn-engine-rs/src/minimum_space_persistence.rs',N+'sheily-turn-engine-rs/src/minimum_space_chain_integrity.rs|verify_experience_chain'],
    'ADAPTAR P2','experience_memory::event_reducer','Reconstruir estado derivado a partir de eventos y detectar/quarantinar sufijos corruptos de una cadena.',
    'Es un subsistema de experiencia, evidencia y estado cognitivo, muy ligado al motor de turnos. No es un compresor lossless de tensores, KV ni pesos de 2 bits.',
    'Mismos eventos deben producir mismo estado; una corrupción no puede reescribir silenciosamente historia válida ni alterar KV canónica.')
add('S11','Selección de candidatos medidos',S,[N+'sheily-cognitive-mechanisms-rs/src/hypothesis.rs'],
    'EXTRAER P1','adaptive_policy::candidate_ranker','Ordenar perfiles y estrategias candidatos con señales, evidencias y límites aportados por M-SLIM.',
    'La puntuación combina señales; no demuestra rendimiento físico. Mantener recomendaciones separadas de permisos y usar solo métricas observadas cuando se declare una mejora real.',
    'Rechazar valores no finitos, candidatos sin procedencia y promociones que no hayan pasado las puertas existentes.')
add('S12','Búsqueda acotada de secuencias',S,[N+'sheily-cognitive-mechanisms-rs/src/trajectory_search.rs|search_trajectory'],
    'ADAPTAR P2','adaptive_policy::bounded_planner','Explorar alternativas de operaciones con límites de pasos y una cinta de decisiones reproducible.',
    'Necesita un generador y un evaluador propios. No hace más inteligente al LLM automáticamente ni justifica aproximar la atención en modo exacto.',
    'Límite verificable de trabajo y repetibilidad; distinguir puntuaciones de modelo de medidas reales.')
add('S13','Análisis documental y relaciones derivadas',S,[N+'sheily-cortex-rs/src/document_analysis.rs|DocumentKnowledgeGraph'],
    'FUSIONAR P2','document_graph + derived_relations','Añadir entidades, relaciones e hipótesis al grafo de páginas original de M-SLIM.',
    'No copiar un segundo almacén de documentos; relaciones extraídas son derivadas y deben apuntar a páginas/rangos verificables.',
    'Toda relación debe llevar fuente y estado de confianza; una relación derivada no puede sustituir al texto original.')
add('S14','Registro de capacidades y contratos',S,[N+'sheily-runtime-registry-rs/src/lib.rs',N+'sheily-turn-contracts-rs/src/lib.rs'],
    'FUSIONAR P2','runtime_registry + typed_contracts','Describir qué funciones ofrece cada componente y vincularlas con implementación, versión y validación.',
    'M-SLIM ya tiene registro de runtimes y adaptadores. Extraer mecanismos de verificación; no copiar identidades, permisos o contratos de SHEILY como dependencia.',
    'Detectar capacidades declaradas sin consumidor o sin implementación; contratos versionados y fallos explícitos.')
add('S15','Continuidad contextual con ámbito',S,[N+'sheily-context-authority-rs/src/lib.rs|TurnContextEnvelope',N+'sheily-memory-engine-rs/src/session_continuity.rs'],
    'FUSIONAR P2','session_store + continuity_contract','Conservar vínculos entre turnos, documentos y referencias dentro de una sesión.',
    'No copiar la autoridad de SHEILY ni confundir continuidad semántica con restaurar exactamente el estado K/V de otro modelo.',
    'Separación entre proyectos/sesiones, revisiones de documento y modelo; rechazo de snapshots incompatibles.')
add('S16','Asesor sin autoridad implícita',S,[N+'sheily-memory-engine-rs/src/gpem_authority.rs|gpem_authority_projection'],
    'EXTRAER P1','adaptive_policy::advisory_contract','Hacer que el optimizador recomiende o pida revisión sin poder ejecutar, cambiar código o promover perfiles por sí mismo.',
    'Esta proyección GPEM no implementa reparto de RAM/VRAM ni un gestor físico de recursos. Esa afirmación anterior no queda respaldada por este módulo.',
    'Una recomendación no puede convertirse en permiso ni modificar la memoria canónica.')
add('S17','Memoria procedimental de intentos verificados',S,['tidex/src/learning/procedural_memory.rs','tidex/src/learning/procedural_replay.rs|rebuild_from_authenticated_receipt'],
    'ADAPTAR P2','experience_memory::procedural_replay','Recordar qué procedimientos funcionaron a partir de recibos auténticos, sin inventar un segundo historial.',
    'Los esquemas actuales corresponden a solvers TIDEX. Hay que definir recibos propios de M-SLIM y mantener límites del búfer.',
    'No reconstruir éxitos a partir de stdout no autenticado; replay estable y rechazo de esquemas desconocidos.')
add('S18','Portafolio de solvers y crédito causal',S,['tidex/src/learning/solver_portfolio.rs','tidex/src/learning/causal_credit.rs|estimate_causal_credit'],
    'EXPERIMENTAL P3','calibration::numerical_models','Comparar solucionadores o ajustes numéricos con residuos recomputados y atribución de resultados.',
    'Es un portafolio de mínimos cuadrados, no un selector directo CPU/GPU/LLM. Solo compensa extraerlo si el problema de calibración necesita esos solvers.',
    'Residuos independientes, cotas de recursos y evaluación fuera de muestra antes de atribuir una mejora.')
add('S19','Geometría Pitágoras de TIDEX',S,['tidex/src/analysis/pythagoras_topology.rs|PythagorasStaircaseMetric'],
    'EXPERIMENTAL P3','diagnostics::geometry','Comparar normas L1/L2 y analizar perturbaciones en experimentos numéricos.',
    'evaluate_and_correct calcula normas y razones; project_to_geodesic reescala el vector. No comprime sin pérdida; la reescala modifica sus valores y no debe tocar pesos/KV exactos.',
    'Separar diagnóstico de transformación; cualquier transformación debe declararse aproximada y validarse, nunca presentarse como paridad exacta.')
add('S20','Generación de corpus y entrenamiento auxiliar',S,[N+'sheily-dataset-compiler-rs/src/retrieval_curriculum.rs',N+'sheily-training-kernels-rs/src/retrieval_encoder_training.rs'],
    'EXPERIMENTAL P3','offline_validation::retrieval_corpus','Preparar corpus y entrenar un recuperador auxiliar cuando existan datos y una evaluación adecuada.',
    'No importar el stack entero de entrenamiento/inferencia. Entrenar cambia parámetros; no es compresión transparente ni mejora garantizada en el hardware actual.',
    'Datos separados de evaluación, trazabilidad y comparación con búsqueda exacta antes de incorporar un recuperador entrenado.')

add('I01','Admisión de aprendizaje',I,['inicio/adaptation/learning_admission.py|learning_admission_for_evidence'],
    'EXTRAER P1','adaptive_policy::learning_admission','Decidir qué observaciones pueden actualizar el modelo auxiliar de rendimiento y cuáles quedan bloqueadas.',
    'Es un contrato de aprendizaje de dinámica, no autoridad de ejecución. Portar las reglas a tipos propios Rust y a métricas reales de M-SLIM.',
    'Datos no finitos, observaciones ambiguas y muestras de evaluación deben quedar fuera del aprendizaje activo.')
add('I02','Evaluación en sombra, canary y revocación',I,['inicio/adaptation/structural_shadow.py','inicio/adaptation/structural_canary.py','inicio/adaptation/structural_revoke.py','inicio/governance/promotion_manager.py'],
    'FUSIONAR P1','calibration_transaction','Probar perfiles candidatos, activarlos de forma acotada y restaurar el anterior cuando fallen.',
    'M-SLIM ya tiene CalibrationTransaction. La búsqueda estática no encontró usos de ese tipo fuera de su propio archivo/pruebas; aprovechar y cablear ese núcleo antes de duplicarlo.',
    'Ensayo real con fallo de canary y reinicio; restauración verificada y ninguna promoción a partir de métricas inventadas.')
add('I03','Vigilancia secuencial de degradación',I,['inicio/adaptation/degradation_eprocess.py|SequentialDegradationWatch'],
    'EXTRAER P1','calibration::degradation_watch','Vigilar si una configuración nueva empeora una pérdida medida respecto a su referencia.',
    'El módulo combina un proceso de capital y CUSUM; su docstring limita la garantía y lo declara asesor. Adaptar el significado de pérdida a latencia/error con supuestos explícitos.',
    'Paridad Python/Rust sobre secuencias registradas y rechazo de NaN, infinito y pérdidas inválidas; una alarma no autoriza una mutación.')
add('I04','Estadística y confianza explícita',I,['inicio/evidence/statistics.py','inicio/evidence/reliability.py|BettingEProcess'],
    'EXTRAER P1','validation::statistical_evidence','Separar insuficiencia de datos, alarmas secuenciales y tipos de cobertura estadística.',
    'Las garantías requieren hipótesis sobre los datos; la cobertura conformal adaptativa no equivale a una garantía exacta por petición.',
    'Pruebas de fronteras, warmup y datos dependientes; registrar los supuestos y no renombrar estimaciones como certezas.')
add('I05','Modelo continuo del coste observado',I,['inicio/cognition/cognitive_engines/m_continuous.py|apply_rls_theta_update','inicio/cognition/cognitive_engines/m_continuous.py|GaussianGenerativeModel'],
    'ADAPTAR P2','adaptive_policy::cost_model','Aprender un predictor pequeño del coste de operaciones a partir de longitud de contexto, lote y telemetría.',
    'La aplicación a coste de inferencia es una propuesta de adaptación, no una capacidad medida ya en M-SLIM. El modelo continuo no es un LLM ni almacena documentos sin pérdida.',
    'Predicción contrastada con medidas reales y conjuntos reservados; límites de recursos y detección de cambio de distribución.')
add('I06','Rutas según régimen de carga',I,['inicio/adaptation/regime_composition.py|RegimeRoutedGenerativeModel','inicio/substrates/routing_substrate.py'],
    'ADAPTAR P2','adaptive_policy::workload_router','Distinguir regímenes recurrentes de trabajo y reutilizar modelos auxiliares especializados.',
    'Adaptar identidades y contratos a hardware/carga. Cambiar de LLM no conserva necesariamente respuestas ni KV: debe ser una decisión explícita, no una promesa de paridad.',
    'No aplicar una ruta conocida a hardware/modelo incompatible; probar regímenes nuevos, agotamiento de capacidad y retorno a uno anterior.')
add('I07','Detección de insuficiencia y cambio de dinámica',I,['inicio/cognition/cognitive_engines/m_continuous.py|DynamicSufficiencyMonitor'],
    'ADAPTAR P2','adaptive_policy::sufficiency_monitor','Saber cuándo el predictor auxiliar dejó de describir las medidas y conviene abstenerse o recalibrar.',
    'No diagnostica por sí solo todas las causas físicas. Sus umbrales y observables deben rediseñarse para rendimiento y exactitud de inferencia.',
    'Cambios de carga, temperatura o ruta deben evaluarse con medidas de referencia y no confundirse con ganancias persistentes.')
add('I08','Recuperación de experiencias operativas',I,['inicio/memory/memory_retrieval.py|ContextualMemoryRetriever'],
    'ADAPTAR P1','experience_memory::profile_retrieval','Recuperar experiencias compatibles con ambiente, régimen y esquema para sugerir perfiles ya probados.',
    'La recuperación es asesora y puede abstenerse. No es la recuperación de tokens K/V ni un permiso para activar una estrategia.',
    'Identidad ausente, evidencia insegura o índice obsoleto deben causar abstención; nunca crear candidatos que no existían.')
add('I09','Reorganización del índice de experiencias',I,['inicio/memory/plastic_memory_reorganizer.py'],
    'ADAPTAR P2','experience_memory::derived_index','Reorganizar proyecciones para recuperar historia útil con menos trabajo.',
    'Tratar el índice como derivado y reconstruible. No extrapolar reorganización semántica a borrar información canónica o comprimir arbitrariamente K/V.',
    'Índice obsoleto detectable, reconstrucción y mismas identidades de evidencia tras reorganizar.')
add('I10','Registro transaccional de experiencia y aprendizaje',I,['inicio/persistence/event_store.py|EventStore'],
    'ADAPTAR P2','experience_memory::event_log','Relacionar observaciones, decisiones de admisión y mutaciones con integridad y prevención de fugas de evaluación.',
    'No copiar un segundo almacén de KV ni sus identidades de organismo. Definir un único registro de evidencia operacional propio.',
    'Atomicidad, caída entre eventos, duplicados e identificación de muestras reservadas.')
add('I11','Checkpoints de datos sin ejecución arbitraria',I,['inicio/persistence/neural_checkpoint.py|prepare_neural_restore','inicio/persistence/state_store.py'],
    'FUSIONAR P2','session_store + auxiliary_checkpoint','Validar versiones y configuración antes de restaurar modelos auxiliares y estados de optimización.',
    'El formato donante usa árboles JSON y arrays NPZ, no pickle. Conservar el principio, no añadir Python al runtime ni sustituir el formato de KV exacta.',
    'Rechazar formas, tipos, versiones y configuraciones incompatibles; ninguna carga ejecutable desde un checkpoint.')
add('I12','Comparación causal y prevención de contaminación',I,['inicio/evidence/causal.py|OffPolicyEvaluator','inicio/evidence/boundary.py'],
    'ADAPTAR P2','validation::paired_evaluation','Comparar políticas sin mezclar entrenamiento y evaluación; agrupar episodios o bloques temporales.',
    'La evaluación off-policy necesita cobertura y supuestos de identificación. No convertir historial sesgado en una mejora causal demostrada.',
    'Abstención cuando falta soporte, evaluación retenida y resultados reproducibles frente a comparación pareada directa.')
add('I13','Restricciones y decisiones admisibles',I,['inicio/cognition/cognitive_engines/m4_governance.py|GovernanceGateway'],
    'EXTRAER P1','adaptive_policy::constraints','Filtrar decisiones por límites explícitos de memoria, exactitud, coste y permisos.',
    'Copiar la separación entre objetivos y restricciones; no la autoridad humana específica ni un nuevo sistema de permisos paralelo.',
    'Un optimizador no puede relajar el modo exacto, el presupuesto o una prohibición para mejorar su puntuación.')
add('I14','Estrategias de búsqueda reemplazables',I,['inicio/cognition/cognitive_engines/policy_search.py|ExactPolicySearchStrategy','inicio/cognition/cognitive_engines/policy_search.py|BeamPolicySearchStrategy','inicio/cognition/cognitive_engines/planner_plasticity.py'],
    'ADAPTAR P2','adaptive_policy::search_strategy','Elegir cómo buscar perfiles o planes dentro de un presupuesto computacional.',
    'Los certificados distinguen búsqueda exacta de aproximada. Esa distinción del planificador no autoriza atención aproximada ni pérdida de tokens.',
    'Referencia exhaustiva en problemas pequeños, límites de exploración y rotulado correcto de resultados incompletos.')
add('I15','Aprendizaje auxiliar en cuarentena',I,['inicio/adaptation/quarantine_tracker.py|QuarantineSafeTrackerBank'],
    'ADAPTAR P2','adaptive_policy::shadow_tracker','Seguir ajustando candidatos auxiliares aislados cuando la actualización activa no es admisible.',
    'No aprende el LLM canónico en secreto ni concede autoridad. Separar sus datos y parámetros del perfil activo.',
    'Ningún cambio observable en el perfil vivo durante la cuarentena; activación solo tras evaluación independiente.')
add('I16','Evaluación de alternativas y composición',I,['inicio/cognition/parallel_counterfactual_engine.py|ParallelCounterfactualEngine','inicio/adaptation/composition_evaluator.py'],
    'EXPERIMENTAL P3','adaptive_policy::proposal_lab','Proponer y comparar alternativas antes de un ensayo físico acotado.',
    'Los contrafactuales son predicciones, no mediciones. No cumplen por sí solos el requisito de resultados reales; no deben entrar como benchmarks.',
    'Etiquetar predicción/observación y exigir ensayo real separado para cualquier afirmación de mejora.')
add('I17','Red modular y adaptadores entrenables',I,['inicio/substrates/continual_modular_network.py|ContinualModularNetwork'],
    'EXPERIMENTAL P3','optional_auxiliary_predictor','Explorar un predictor auxiliar modular con capacidad y parámetros aislados.',
    'El propio módulo advierte que preservar parámetros no garantiza predicciones invariantes cuando cambia el router. No es ampliación automática de inteligencia ni compresión sin pérdida.',
    'Capacidad máxima, aislamiento, restauración y evaluación de cambios de salida; nunca modificar pesos de inferencia exacta implícitamente.')
add('I18','Dinámica ondulatoria y geometría',I,['inicio/substrates/wave_substrate/weight_soliton.py','inicio/substrates/wave_substrate/attention_bridge.py','inicio/cognition/lie_invariant_geometry.py'],
    'EXPERIMENTAL P3','isolated_research_only','Conservar como líneas de investigación numérica, separadas del producto exacto.',
    'La existencia de estos módulos no demuestra una compresión o atención superior en M-SLIM. No incorporar al núcleo hasta medir utilidad, recursos y desviación.',
    'Benchmark independiente con datos reales, ablation y comparación exacta; sin promesas basadas en el nombre del algoritmo.')
add('I19','Federación de evidencias y malla de eventos',I,['inicio/federation/federated_evidence.py','inicio/federation/swarm_event_mesh.py'],
    'NO PORTAR','none','No aporta una necesidad inmediata al motor local independiente solicitado.',
    'Importar una malla distribuida añade identidades, red y complejidad ajenas al objetivo. No se propone conexión alguna con los proyectos donantes.',
    'Mantener fuera de dependencias, servicios y despliegue de M-SLIM.')
add('I20','Runtime de organismo y puente HTTP',I,['inicio/runtime/organism.py','inicio/integration/external_transition_gateway.py','inicio/integration/http_service.py'],
    'NO PORTAR','none','Extraer algoritmos pequeños de sus módulos; no trasladar el organismo ni usar su servidor.',
    'Copiar o conectar el runtime completo contradice la independencia pedida. El runtime propio de M-SLIM conserva todo el control.',
    'El resultado debe compilar y funcionar con las carpetas SHEILY-COGNITIVA e INICIO ausentes y sin sus endpoints.')

NATIVE_DECISIONS = {
 'sheily-agent-continuity': 'No copiar el agente completo; solo patrones de continuidad justificados por S15.',
 'sheily-agent-neural': 'No copiar ejecutores/agentes ni servicios; fuera del motor de memoria.',
 'sheily-agent-virtual': 'No copiar orquestación de agentes; no requerida para la transferencia.',
 'sheily-api': 'No copiar; conservar API y runtime propios de M-SLIM.',
 'sheily-cognitive-console': 'Extraer analizador/inventario para S01; no la consola ni sus servicios.',
 'sheily-cognitive-mechanisms': 'Núcleos reutilizables S11-S12; retirar dominios y contratos propios de SHEILY.',
 'sheily-cognitive-root': 'No trasplantar raíz de identidad, claves ni autoridad.',
 'sheily-cognitive-runtime': 'No copiar ciclo cognitivo completo; selección de núcleos en fichas S/I.',
 'sheily-config': 'Solo patrón de configuración tipada si falta; no rutas de instalación del donante.',
 'sheily-content-guard': 'Revisión opcional de validadores; no reescritura silenciosa de prompts.',
 'sheily-context-authority': 'Adaptar contratos y ámbito en S15; no importar autoridad externa.',
 'sheily-context-continuity': 'Fusionar principios con contratos/sesiones existentes; no puente.',
 'sheily-cortex': 'Extraer revisión/selección/análisis; no copiar Cortex entero.',
 'sheily-dataset-compiler': 'Opcional offline, S20; no requisito del runtime.',
 'sheily-errors': 'No duplicar jerarquía completa; convertir errores al dominio M-SLIM.',
 'sheily-evolution': 'Solo controles/evidencias útiles; fuera de alcance automodificación de código.',
 'sheily-execution-kernel': 'No copiar ejecutor general ni todas sus dependencias.',
 'sheily-governance-kernel': 'Extraer validadores puntuales; no segundo sistema de autoridad.',
 'sheily-governance': 'No copiar plano completo de gobernanza ni conectarlo.',
 'sheily-inference': 'No sustituir el motor de inferencia propio por el de SHEILY.',
 'sheily-interfaces': 'Redefinir contratos mínimos propios; sin dependencia path al donante.',
 'sheily-joint-intent': 'Opcional análisis de tarea; no necesario para los primeros núcleos.',
 'sheily-ledger': 'Adaptar patrones de recibos/encadenado; evitar almacenes canónicos duplicados.',
 'sheily-memory-engine': 'Extraer S05-S09, S15-S16; no arrastrar Postgres, inferencia ni todos los escritores.',
 'sheily-parity-support': 'Aprovechar corpora/criterios de paridad como pruebas, no como dependencia runtime.',
 'sheily-policy': 'Extraer validaciones necesarias; no copiar políticas de otro producto.',
 'sheily-port-auditor': 'Herramienta opcional de migración; no parte del hot path.',
 'sheily-repo-auditor': 'S02-S03; revisar limitaciones Python y coste de escaneo.',
 'sheily-resource-index': 'S01; adaptación obligatoria para indexar Rust realmente.',
 'sheily-runtime-paths': 'No copiar rutas absolutas ni descubrimiento del árbol SHEILY.',
 'sheily-runtime-registry': 'S14; fusionar con registro existente.',
 'sheily-sandbox-supervisor': 'No copiar supervisión externa para un motor local de memoria.',
 'sheily-secrets': 'No copiar credenciales; una eventual utilidad debe diseñarse para el producto propio.',
 'sheily-shl': 'No introducir el lenguaje/orquestación de SHEILY como requisito.',
 'sheily-task-compiler': 'Opcional futuro; primero planificador tipado mínimo propio.',
 'sheily-training-kernels': 'S20, opcional; no importar entrenamiento completo al runtime.',
 'sheily-turn-contracts': 'S14, contratos mínimos propios con versión.',
 'sheily-turn-engine': 'Solo reductores/evidencia S10; no copiar motor de turnos completo.',
}

def inicio_disposition(path):
    if '/integration/' in path or '/runtime/' in path: return 'NO PORTAR runtime/puente completo; extraer únicamente núcleos de otras fichas.'
    if '/federation/' in path: return 'NO PORTAR para este objetivo local independiente.'
    if '/wave_substrate/' in path or path.endswith(('continual_modular_network.py','substrate_discovery.py','lie_invariant_geometry.py','symbolic_operator_evaluator.py','v4_runtime_integration.py','meta_learning_orchestrator.py','prototype_registry.py','parallel_counterfactual_engine.py')): return 'EXPERIMENTAL P3; beneficio no demostrado en M-SLIM.'
    if '/persistence/' in path: return 'ADAPTAR/FUSIONAR contratos de evidencia/checkpoints; no duplicar KV.'
    if '/memory/' in path: return 'ADAPTAR recuperación/índices de experiencias (I08-I09).'
    if '/evidence/' in path: return 'ADAPTAR primitivas estadísticas, trazabilidad y evaluación (I04/I12).'
    if '/governance/' in path: return 'ADAPTAR puertas/revocación; no trasplantar autoridades o claves.'
    if '/contracts/' in path: return 'ADAPTAR tipos mínimos versionados; no adoptar identidad de organismo.'
    if '/adaptation/' in path: return 'ADAPTAR módulos de aprendizaje/validación; composición compleja queda P3.'
    if '/cognitive_engines/' in path: return 'EXTRAER o ADAPTAR núcleos según I05/I07/I13/I14; no importar ciclo completo.'
    if '/lab/' in path: return 'No runtime: conservar solo como herramienta experimental aislada si hace falta.'
    if path.endswith('paths.py'): return 'No copiar rutas de INICIO; usar rutas internas de M-SLIM.'
    return 'REVISIÓN ESTRUCTURAL; sin prioridad de traslado antes de definir un caso útil y pruebas.'


def main():
    out = {'schema':'mslim.native_transfer.assessment/v1','date':'2026-10-31','scope':'native independent adaptation; not integration',
           'source_inventory':'source_inventory.v2.json','runtime_changes_made':False,'donor_tests_executed':False,
           'capabilities':ROWS}
    with (HERE/'capability_matrix.json').open('x') as f: json.dump(out,f,ensure_ascii=False,indent=2); f.write('\n')
    lines=['# Capacidades candidatas: extracción nativa para M-SLIM','',
           'Fecha: 31 de octubre de 2026. Estado: inventario y evaluación de transferencia, **no implementación ni certificación de las capacidades en M-SLIM**.','',
           '## Alcance y trazabilidad','',
           'Se analizó el árbol de trabajo local, incluidos cambios previos sin commit. Los hashes individuales del inventario son más precisos que citar solo HEAD. Se realizó inventario estructural del alcance completo y lectura dirigida de los núcleos seleccionados; no se afirma haber revisado semánticamente cada línea.','',
           'Python se indexó con AST. Rust usa un extractor léxico para el inventario, complementado con lecturas de código; no sustituye a rust-analyzer. Las pruebas contadas no se ejecutaron en esta tarea. La única incidencia de parseo Python encontrada pertenece a una fixture deliberadamente inválida: cognition/tests/tools/audit/deadcode_guard/fixtures/pkg/bad.py.','',
           '| Proyecto | HEAD observado | Archivos fuente | Líneas inventariadas |','|---|---|---:|---:|']
    for project,snap in DATA['projects'].items(): lines.append(f"| {project} | `{snap['head'][:12]}` | {snap['source_count']} | {snap['source_lines']} |")
    lines += ['', 'Se excluyen caches de compilación, dependencias vendorizadas bajo vendor/vendor-patches, datos de runtime y enlaces simbólicos. El código de aplicaciones externas presente en el árbol, cuando se inventaría, queda marcado aparte. En SHEILY hay 821 archivos clasificados como componentes externos/vendorizados dentro de los 2.828 inventariados; no son 2.828 capacidades propias.', '',
              'El inventario inicial v1 se conserva. La versión v2 corrige la clasificación de inicio/evidence como código de producción, no como evidencia histórica, y la posición léxica de símbolos Rust. Los hashes de las fuentes no cambiaron.', '',
              '## Regla de independencia', '',
              'Los donantes solo sirven como origen de código y pruebas. No se propone instalar sus runtimes, invocar sus APIs, importar sus paquetes Python ni añadir dependencias Cargo que apunten fuera de M-SLIM. La prueba final de cualquier futuro port debe poder retirar ambos árboles donantes y seguir compilando/funcionando. Las referencias a esos árboles en esta carpeta son evidencia de auditoría, no dependencias del producto.', '',
              '## Decisiones resumidas', '', '| ID | Capacidad | Decisión | Destino propuesto, no creado |','|---|---|---|---|']
    for row in ROWS: lines.append(f"| {row['id']} | {row['name']} | {row['decision']} | `{row['proposed_target']}` |")
    lines += ['', '## Fichas de extracción y límites', '']
    for row in ROWS:
        lines += [f"### {row['id']} · {row['name']}", '',f"**Decisión:** {row['decision']}. **Destino propuesto:** `{row['proposed_target']}`.", '',row['benefit'],'',f"**Límite:** {row['limit']}",'',f"**Aceptación exigida al port:** {row['acceptance_gate']}",'','**Fuentes locales:**']
        for source in row['sources']:
            lines.append(f"- `{source['project']}/{source['path']}:{source['line']}`" + (f" · `{source['symbol']}`" if source['symbol'] else '') + f" · SHA-256 `{source['sha256'][:16]}…`.")
        lines.append('')
    lines += ['## Qué M-SLIM ya tiene y no conviene duplicar','',
              '| Fuente M-SLIM | Evidencia estática observada | Acción recomendada |','|---|---|---|',
              '| `src/document_graph.rs` | Páginas originales, hashes, presupuesto y candidatos diferidos; referencias en CLI documental. | Extender con fusión de candidatos y relaciones derivadas. |',
              '| `src/calibration_transaction.rs` | Propuesta, sombra, canary, promoción, rollback; las 4 referencias a `CalibrationTransaction` encontradas están en este mismo archivo. | Verificar/cablear su uso por el producto antes de importar otra máquina de estados. |',
              '| `src/shadow_metrics.rs` | Referenciado por `torch_runtime.rs`. | Añadir observaciones y alarmas sin darles autoridad sobre exactitud. |',
              '| `src/adaptive_extent.rs` | Eje reversible utilizado por varios componentes, incluidas rutas CPU/GPU. | No sustituirlo por memoria semántica o geometría experimental. |',
              '| `src/session_store.rs` y `src/continuity_contract.rs` | Referencias en coordinador y runtime-worker. | Añadir estado auxiliar sin duplicar sesiones ni alterar KV. |',
              '| `runtime_registry`, catálogos, adaptadores y host | Implementaciones/exportaciones presentes. | Reutilizar el plano de ejecución propio. Su ejecución completa no se certificó aquí. |','',
              'Una referencia estática es evidencia de cableado textual, no prueba de ejecución. La búsqueda de CalibrationTransaction no descarta todos los usos indirectos posibles, pero no aporta evidencia de un consumidor fuera del módulo.', '',
              '## Orden de adaptación recomendado', '',
              '1. **Recuperación verificable y conocimiento del proyecto.** S01-S05. Reutilizar el grafo documental existente; añadir extracción real de símbolos Rust, revisiones de fuente y fusión con citas. No empezar por un Cortex general.',
              '2. **Experiencias y estado de tarea.** S06-S09, S15, S17, I08-I10. Separar datos originales, índices derivados y estado epistemológico. Mantener un único propietario para cada dato.',
              '3. **Control adaptativo medido.** S11, S16, I01-I07, I13-I15. Empezar con consejos sobre perfiles existentes. Primero medición en sombra, después canary y promoción explícita; no reescribir prompts ni pesos.',
              '4. **Pruebas de utilidad y regresión.** S03-S04, I04, I12. Medir memoria pico, tiempos de prefill/decode, tráfico de disco y calidad/recuperación por separado. Exigir pruebas de paridad cuando se declare exactitud.',
              '5. **Investigación opcional aislada.** S12, S18-S20, I16-I18. Solo avanzar cuando una necesidad concreta y medidas reales justifiquen el coste. Predicción o simulación no cuenta como benchmark físico.', '',
              '## Condiciones que no deben perderse', '',
              '- Datos canónicos, pesos y K/V exactos no se sustituyen por una síntesis semántica. Las proyecciones auxiliares deben poder reconstruirse o declararse incompletas.',
              '- No copiar credenciales, estados personales, servicios, rutas absolutas o dependencias transitivas del donante. Registrar procedencia del código y revisar los metadatos de licencia antes de distribuirlo; esta revisión no emite una conclusión jurídica.',
              '- Las herramientas de auditoría y aprendizaje auxiliar no deben introducir recorridos completos, hashes totales o entrenamiento por token en el camino crítico.',
              '- Los cambios de modelo, precisión y algoritmo que alteren salidas no se presentan como optimizaciones exactas transparentes.',
              '- El nombre de un algoritmo no demuestra una capacidad: Minimum Space no es compresión universal; Pitágoras en TIDEX es geometría L1/L2; la dinámica continua de INICIO no es un LLM.', '',
              '## Estado de esta entrega', '',
              'Se han creado únicamente artefactos de auditoría dentro de esta carpeta de KV-SHEI. No se han conectado proyectos, portado módulos funcionales, alterado el runtime, ejecutado suites de los donantes, hecho commit ni push. No se modificaron intencionadamente los árboles originales. La comprobación final de hashes está en `source_preservation_check.json`.', '',
              'El registro `COMPONENTES.md` asigna una disposición a los 38 crates nativos principales de SHEILY y a los 85 módulos de implementación Python de INICIO. `source_inventory.v2.csv` y `.json` mantienen el detalle por archivo, incluidos otros subsistemas, herramientas, tests y componentes externos. `capability_matrix.json` contiene las 40 fichas y sus hashes completos.']
    with (HERE/'INFORME.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    components=['# Registro estructural de componentes','',
        'Esta tabla cubre unidades principales; no representa una auditoría semántica línea por línea. Las fichas de INFORME.md justifican las prioridades. El inventario JSON conserva todos los símbolos/imports detectados.','',
        '## SHEILY · 38 crates nativos principales','', '| Crate | Dependencias locales declaradas | Disposición |','|---|---|---|']
    natives=[m for m in DATA['projects'][S]['manifests'] if m['path'].startswith(N) and m['path'].count('/')==3 and m.get('name')]
    assert {m['name'] for m in natives} == set(NATIVE_DECISIONS), 'Unclassified native crate'
    for manifest in natives:
        deps=', '.join(k for k,v in manifest['dependencies'].items() if isinstance(v,dict) and 'path' in v) or 'Ninguna declarada directamente'
        components.append(f"| `{manifest['name']}` | {deps} | {NATIVE_DECISIONS[manifest['name']]} |")
    components += ['', '## INICIO · 85 módulos de implementación','', '| Fuente | Símbolos representativos, no exhaustivos | Disposición |','|---|---|---|']
    modules=[item for item in DATA['projects'][I]['sources'] if item['path'].startswith('inicio/') and item['path'].endswith('.py') and not item['path'].endswith(('/__init__.py','/__main__.py'))]
    for module in modules:
        symbols=', '.join(f"{s['name']}:{s['line']}" for s in module['symbols'] if not s['name'].startswith('_'))[:260]
        components.append(f"| `{module['path']}` | `{symbols}` | {inicio_disposition(module['path'])} |")
    components += ['', '## Otros subsistemas SHEILY inventariados','',
        '| Subsistema | Decisión |','|---|---|',
        '| `tidex` | S17-S19: extracción muy selectiva de procedimientos y diagnóstico; no cirugía de pesos ni solver completo por defecto. |',
        '| `auto-doc` | S01-S04: herramientas de inventario/cierre; no incorporar su servidor o interfaz como runtime. |',
        '| `globus/native` | Tiene fusión RRF, pero S05 ofrece la validación canónica más directamente relevante; evitar duplicar fusores. |',
        '| `globus/live-plane` y SDKs | Aplicaciones/dependencias externas; no se presentan como algoritmos propios ni como prioridad de port. |',
        '| `data-plane` | Esquemas de memoria/evidencia pueden informar contratos; no trasladar el despliegue de bases de datos. |',
        '| `src`, `runtime`, `ops`, `deploy`, `tooling`, `cognition/tools` | Orquestación y herramientas: referencia de operación/auditoría, no instalar los proyectos donantes. |',
        '| `tests`, `benchmarks`, `evidence`, `research`, `docs` | Separar pruebas/corpora y evidencias históricas de código de producto. No contar un resultado histórico como prueba nueva. |', '',
        'Todos los archivos fuente restantes dentro del alcance están registrados individualmente en source_inventory.v2.csv/json; no se ocultan como capacidades omitidas. Su clasificación estructural no significa recomendación de copiarlos.']
    with (HERE/'COMPONENTES.md').open('x') as f: f.write('\n'.join(components)+'\n')
    validation={'capability_records':len(ROWS),'source_references_checked':sum(len(r['sources']) for r in ROWS),
                'native_crates_classified':len(natives),'inicio_modules_classified':len(modules),
                'all_source_paths_exist_in_snapshot':True,'all_named_symbols_found':True,
                'runtime_code_changed':False,'donor_test_results':'not_run',
                'nature':'documentary_and_structural_validation_only'}
    with (HERE/'assessment_validation.json').open('x') as f: json.dump(validation,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(validation,ensure_ascii=False,indent=2))
    print('REPORT',HERE/'INFORME.md')

if __name__=='__main__': main()
