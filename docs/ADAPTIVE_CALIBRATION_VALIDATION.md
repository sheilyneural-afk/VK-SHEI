# Validación de calibración adaptativa — 31 de octubre de 2026

> Evidencia histórica: este documento registra una ejecución fechada y no
> certifica el `HEAD`, los binarios instalados ni el runtime residente actual.

```mermaid
flowchart LR
    S[Commit congelado] --> T[Suites automáticas]
    T --> M[Campaña física shadow]
    M --> C[Canary con muestras nuevas]
    C --> P[Promoción aislada]
    P --> H[Registro histórico fechado]
    H -. no certifica .-> A[Runtime actual]
```

## Fuente y alcance

Código probado: `eb1a51d5a990511afdb9ea18ee00f752c46f7d7f`. Rama: `feat/adaptive-calibration-rls`.
La validación se ejecutó sobre un archivo de ese commit extraído dentro del
runtime local, con un directorio de compilación separado. Esto excluye cambios
concurrentes no incluidos en el commit. No se modifica el motor de producción.

El commit `0f5c4fb` conserva el trabajo que ya estaba sin commit al empezar;
`9611077` implementa la calibración y `eb1a51d` corrige el roundtrip de decimales.
`main` se conserva en `9cb24635fe0bd7d3355c669b19da610a16d4680f`.

## Pruebas automáticas

| Configuración | Aprobadas | Fallidas | Omitidas |
|---|---:|---:|---:|
| portable | 144 | 0 | 0 |
| native-cpu | 149 | 0 | 0 |
| accelerator | 165 | 0 | 0 |

Son conteos por configuración, no pruebas únicas sumables. Incluyen las pruebas
del workspace, integración de perfiles y doctests disponibles. Clippy con
`-D warnings` pasó y los tres binarios de calibrador, CLI y worker compilaron en
release. Se usó Cargo offline, con cachés y temporales dentro del proyecto.

## Ejecución física

Modelo real: Llama 3 8B Instruct, pesos BF16. Dispositivo: AMD Radeon integrado
Phoenix, runtime Rust/OpenCL. Sin sustituir la inferencia por un mock.

Longitudes 8, 16 y 24 tokens, tres repeticiones por longitud y dos continuaciones
con entradas fijadas. Se midieron 27 muestras shadow y 18 muestras canary nuevas,
además de calentamientos. Los candidatos conservaron los bits de logits,
argmax y hashes lógicos de K/V de la referencia; máximo error observado: 0.0.

El perfil `batch-16-reuse-8` superó las puertas y se promovió únicamente
en el directorio aislado de la campaña. El motor de la campaña recuperó su
configuración original. La referencia explícita fue batch 32/reuse 1, no una
comparación contra todas las configuraciones previamente optimizadas del producto.

No se publica un factor general de aceleración: son contextos breves de carga,
no un benchmark de contextos largos, calidad lingüística o ahorro de memoria.
La memoria registrada es VmHWM acumulada del proceso, no VRAM aislada.

## Consumo de perfiles y continuidad de evidencia

La CLI cargó el perfil promovido y usó batch 16/reuse 8 con un prompt de ocho
tokens. Conservó el token de salida y el resumen de logits frente a la
configuración de referencia. Los logits completos se compararon en la campaña;
la salida de esta CLI solo publica resúmenes.

Con nueve tokens, fuera de la cobertura medida, la CLI volvió a batch 32/reuse 1.
Con una capacidad de contexto distinta (256 en lugar de 128) rechazó el perfil.

Una nueva campaña leyó el informe anterior, reconstruyó el predictor y cambió
el orden solicitado de candidatos de `32:8,16:8` a `16:8,32:8`. Todos se midieron.
Sin `--promote` terminó en `shadow_passed_not_activated`; no hubo activación
implícita. El error empírico secuencial registrado alrededor de 0.36 no se
presenta como precisión alta ni como intervalo de confianza.

El worker local cargó el perfil y respondió HTTP 200 a una petición real de chat.
Su prompt de diez tokens usó correctamente el respaldo por estar fuera de
cobertura. El diagnóstico incluyó el perfil y los ajustes efectivos. El proceso
se detuvo al terminar. La opción de perfil en modo de runtime externo por IPC
fue rechazada explícitamente, sin intentar abrir ese runtime.

## Fallo encontrado y corregido durante la validación

Una campaña previa mantuvo la paridad física pero falló al verificar un recibo:
el decimal `1464.1582419999997` se releía como `1464.1582419999995`, cambiando un bit
antes de recalcular SHA-256. Se reprodujo el fallo con una prueba aislada.

La dependencia existente serde_json usa ahora `float_roundtrip`. Una regresión
recorre toda la transacción con ese valor real y otra prueba 4096 patrones de
bits finitos. La campaña se repitió después de corregirlo y terminó promovida.
Los artefactos de la campaña fallida se conservan; no se reescribieron para
presentarlos como aprobados. El perfil fallido volvió a la referencia y quedó
bloqueado por su marcador pendiente.

## Evidencias locales

Directorio, relativo a la raíz del proyecto:

```text
deploy/local-production/runtime/receipts/adaptive-calibration-20261031T180050Z/
```

El manifiesto `validation-summary.json` contiene los conteos, commit de fuente,
hashes de informes y binarios, perfil seleccionado y límites. Entre los recibos:

- `verified-test-summary.json`: pruebas del commit aislado.
- `physical-prefill-verified/report.json`: mediciones y canary físicos.
- `physical-history-verified/report.json`: reutilización de evidencia por RLS.
- `physical-consumer-summary.json`: CLI, respaldo y rechazo de entorno.
- `worker-profile-summary.json`: petición HTTP y cierre del proceso de prueba.
- `decimal-roundtrip-reproduction.log`: reproducción negativa previa a la corrección.

La copia de seguridad inicial (`source-baseline.tar.gz`, hashes y diff) está en
el mismo directorio. Los binarios probados se conservan en
`deploy/local-production/runtime/build-target-adaptive-validation/release/`;
no sustituyen los binarios de producción.

## Límites no certificados

No se certifica rendimiento con contextos largos ni otros modelos, backends o
configuraciones. No hay presupuesto automático de RAM/VRAM nuevo. El predictor
no elimina pruebas. Un cierre forzado puede requerir recuperación explícita de
marcadores/reservas; no se inventa una recuperación automática.

Los cambios concurrentes de otras funciones permanecen en el árbol de trabajo,
fuera de estos commits. No se han mezclado, eliminado ni validado como parte de
esta entrega. No se hizo push ni merge en main.

Instrucciones de uso: [ADAPTIVE_CALIBRATION.md](ADAPTIVE_CALIBRATION.md).
