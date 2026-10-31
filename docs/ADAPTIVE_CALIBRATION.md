# Calibración adaptativa del prefill

## Alcance y propietarios

Esta ampliación ayuda al calibrador existente; no introduce otro planificador,
almacén de memoria, sistema de sesiones ni autoridad de activación.

| Componente | Responsabilidad |
|---|---|
| `calibration_predictor.rs` | Estimación RLS del tiempo de prefill y orden de candidatos; sin autoridad. |
| `calibration_profile.rs` | Contrato estricto de ajustes, identidad y lectura de perfiles promovidos. |
| `prefill_calibration.rs` | Mediciones físicas mediante `OpenClModelAdapter` y uso de la transacción existente. |
| `calibration_transaction.rs` | Propuesta, shadow, canary, promoción explícita, recibos y reversión de archivos. |
| `OpenClLlm` | Aplicación de los dos ajustes sobre el motor real, antes del prefill. |

El predictor es una implementación nueva y pequeña del mecanismo de mínimos
cuadrados recursivos considerado en la revisión de INICIO. No es una copia del
organismo ni un port completo de sus modelos continuos. No importa paquetes,
servicios o datos de SHEILY/INICIO y no necesita que esos proyectos existan.

## Qué puede cambiar

Solo `batch_tokens` (1..512 o el modo secuencial ya existente) y
`weight_reuse_tokens` (1, 8, 16 o 32). El runtime conserva sus restricciones de
fusión, registros y virtual lanes. Las combinaciones incompatibles se rechazan
antes de medir. Un perfil no puede cambiar pesos, precisión, tokenizer, modelo,
backend, epsilon, selección de atención, KV canónico, documentos ni prompts.

La primera implementación física usa OpenCL BF16 exacto. La biblioteca del
predictor y los contratos compilan sin GPU. La ruta de CPU no recibe ajustes
OpenCL y `--tune-prefill` sin `accelerator-backend` falla explícitamente.

## Predictor acotado

RLS usa tres características: `1`, `tokens/1024` y su cuadrado. La covarianza
inicial es diagonal con valor 1000 y el factor de olvido es 1. El objetivo son
segundos de prefill medidos. Se validan valores finitos, covarianza simétrica
positiva y actualización atómica. No predice exactitud ni memoria disponible.

El presupuesto es de 32 perfiles y 4096 observaciones identificadas. Un mismo
ID de evidencia no puede entrenarse dos veces. Para predecir se necesitan al
menos seis observaciones y tres longitudes distintas. Fuera del intervalo
observado, con otra identidad o con error empírico excesivo, se abstiene.
El error secuencial registrado NO es un intervalo de confianza ni una garantía
estadística de rendimiento futuro.

El modelo se reconstruye desde informes medidos: no añade una base de datos ni
un checkpoint ejecutable. Una actualización numérica rechazada no activa nada
ni impide que continúe la medición física.

**La versión actual mide todos los candidatos.** El predictor cambia su orden;
no reduce todavía el número de pruebas ni demuestra por sí mismo un ahorro de
tiempo de calibración. La decisión final se toma con mediciones, no predicciones.

## Compilar y medir

Desde la raíz de KV-SHEI:

```bash
deploy/local-production/scripts/cargo-local.sh build --offline --release \
  --features accelerator-backend \
  --bin mslim-calibrate --bin mslim-runtime-cli --bin mslim-runtime-worker

MODEL=/ruta/al/bundle-validado
BIN=deploy/local-production/runtime/build-target/release

"$BIN/mslim-calibrate" "$MODEL" --tune-prefill \
  --contexts 128,256,512 \
  --candidates 32:8,64:8,256:8 \
  --max-context-tokens 2048 \
  --warmups 1 --repeats 3 --decode-probes 2 \
  --output deploy/local-production/runtime/calibration/mi-campana
```

El directorio de salida debe ser nuevo. La campaña no sobrescribe evidencias
anteriores, no instala binarios, no modifica `local-production.env`, no reinicia
servicios ni cambia sesiones de producción. Carga un motor aislado y usa
ColdStore temporal dentro de `deploy/local-production/runtime/tmp`.

Sin `--promote` un candidato aceptado queda en shadow, **no activado**. Para
solicitar la promoción en el directorio aislado de la campaña:

```bash
"$BIN/mslim-calibrate" "$MODEL" --tune-prefill \
  --contexts 128,256,512 --candidates 32:8,64:8,256:8 \
  --max-context-tokens 2048 \
  --output deploy/local-production/runtime/calibration/mi-campana-promocion \
  --promote operador:revision-local
```

El valor de autoridad documenta una petición explícita del operador; NO es una
firma criptográfica ni acredita su identidad frente a un atacante local.

`--history INFORME.json` permite reconstruir el predictor de una campaña
anterior. Se verifica la integridad del informe y la identidad completa. Un
entorno incompatible se registra como ignorado. Un informe corrupto se rechaza.
Los valores previstos nunca se reintroducen como tiempos medidos.

### Configuración de partida

`--settings ajustes.json` acepta únicamente claves existentes del adaptador:

```json
{
  "mslim_weight_mode": "resident",
  "mslim_prefill_batch_tokens": 128,
  "mslim_prefill_weight_reuse_tokens": 8,
  "mslim_prefill_fuse_projections": true,
  "mslim_hot_v_tokens": 512,
  "mslim_prompt_replay_tokens": 0,
  "mslim_shared_prefix": false
}
```

Los tipos se comprueban: las cadenas no se convierten silenciosamente a números.
El resto de opciones admitidas aparece en `runtime_config` de
`src/prefill_calibration.rs`. La campaña fuerza ColdStore exacto y epsilon cero.
Elimina cualquier `mslim_prefill_profile` heredado de la configuración: un perfil
anterior nunca puede sustituir silenciosamente al candidato durante la medida.

El modo de pesos permanece fijo entre candidatos. El cambio de residencia de
pesos NO es una variable aprendida en esta versión. Los demás ajustes deben
coincidir con la configuración donde se vaya a usar el perfil.

## Evidencia y puertas

Cada contexto tiene calentamiento y varias repeticiones. Se comparan contra la
referencia los bits de los logits del prefill y las continuaciones de prueba,
los tokens argmax y los SHA-256 lógicos de K/V antes y después de la
continuación. Las entradas de continuación son las mismas para todas las rutas.
La referencia también se comprueba entre repeticiones.

Se registra throughput de prefill, p95 de latencia y máximo de error. La
promoción requiere exactitud, las puertas existentes de rendimiento y una
segunda evaluación física (canary) con muestras nuevas. No se reciclan las
muestras shadow como prueba canary. Un candidato más rápido pero distinto se
rechaza. Las comprobaciones pesadas de K/V están en la campaña, no se añaden al
camino normal de generación.

La memoria registrada es `VmHWM` del proceso Linux, acumulada en la campaña;
**no es VRAM, ni un pico aislado por candidato, ni prueba de ahorro de memoria**.
Las lecturas por token no se instrumentan en esta versión; el campo heredado
cero está etiquetado como no medido en el informe. No se aprueba una afirmación
de ahorro de memoria a partir de esos valores.

Los prompts de prueba son texto determinista tokenizado por el modelo real.
Son pruebas físicas de carga y paridad, NO un benchmark de calidad lingüística
ni una certificación de cualquier prompt o longitud.

## Consumir un perfil en el runtime

La activación es opt-in. Mantenga `active.json`, sus recibos y su directorio
juntos; el archivo solo no basta.

- CLI local: `--prefill-profile /ruta/campana/active.json`.
- Worker local: la misma opción, después de elegir ajustes compatibles.
- ModelAdapter: `"mslim_prefill_profile": "/ruta/campana/active.json"`.

Todos llaman al mismo método de `OpenClLlm`; no hay tres implementaciones de
activación. La opción del worker se rechaza en modo de runtime externo por IPC,
en lugar de ignorarse silenciosamente.

Se exige coincidencia de modelo, tokenizer, backend, hardware/controlador,
implementación, modo de pesos y configuración no ajustable. Cambiar fuentes
relevantes, compilación debug/release o parámetros exige recalibrar. No se
comparan magnitudes volátiles como RAM libre para crear una identidad estable.

El perfil solo selecciona sus ajustes para las **longitudes medidas**. Para
otras longitudes recupera los ajustes originales del consumidor; no extrapola
una garantía de paridad. Cambiar ajustes necesita una sesión vacía. El replay
de una sesión ya activa puede continuar si los ajustes no cambian. Los cambios
incompatibles posteriores de configuración fallan de forma explícita.

Las respuestas diagnósticas de CLI y worker exponen el perfil, cobertura,
ajustes efectivos y configuración de respaldo. No se aplica un perfil por
mera presencia de un archivo en disco.

## Transacción y fallos

La transacción existente ahora exige registrar la propuesta y aprobar shadow
antes del canary. Valida identidades de mediciones, impide dos escritores sobre
el mismo perfil y no promueve con un archivo modificado fuera de la transacción.
Conserva checkpoint y cadena de recibos. Un marcador `active.pending.json`
bloquea consumidores mientras una activación está incompleta.

Un error canary revierte el archivo y el motor aislado vuelve a sus ajustes
originales cuando sigue sano. Un fallo que envenene el contexto GPU hace que se
destruya el motor aislado; no se finge haberlo recuperado. El Drop de una
transacción canary intenta restaurar su archivo si no fue promovida.

**Límite de recuperación:** SIGKILL, apagado o fallo de disco pueden dejar el
marcador o una reserva `active.transaction.lock`. Se bloquea la carga, no se
asume una recuperación automática. Hay que conservar los recibos/checkpoint y
hacer una recuperación explícita después de comprobar que no existe un escritor
vivo; alternativamente iniciar una campaña nueva en otro directorio. No borre
estos archivos para forzar la aceptación de un candidato sin verificarlo.

Los hashes detectan cambios accidentales, no sustituyen permisos de archivos o
firmas de una autoridad separada. Un propietario con control completo del
filesystem también puede reescribir la evidencia.

## Validación reproducible

```bash
deploy/local-production/scripts/cargo-local.sh test --offline --no-default-features --workspace
deploy/local-production/scripts/cargo-local.sh test --offline --workspace
deploy/local-production/scripts/cargo-local.sh test --offline --features accelerator-backend --workspace
deploy/local-production/scripts/cargo-local.sh clippy --offline --features accelerator-backend \
  --lib --bin mslim-calibrate --bin mslim-runtime-cli --bin mslim-runtime-worker \
  --test prefill_profile_lifecycle -- -D warnings
```

Las pruebas unitarias usan datos matemáticos/contratos controlados y las de
integración usan archivos reales; ninguna se presenta como medida de GPU.
Las campañas físicas y la carga del perfil en CLI se registran por separado.

El selector shell histórico sigue siendo una utilidad manual de benchmark.
Su JSON no es un perfil promovido y no puede entrar por esta ruta de activación.
No se han implementado nuevos presupuestos automáticos de RAM/VRAM, selección
de modelos, control distribuido ni adaptación de la atención.
