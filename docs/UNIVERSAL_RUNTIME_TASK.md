# Tarea maestra: KV-SHEI universal con runtime externo

## Objetivo

Completar la separación entre KV-SHEI y cualquier implementación concreta. El núcleo debe seleccionar por manifiestos y capacidades; el proceso externo debe ejecutar el modelo. No se acepta declarar universalidad solo porque exista un `Probe/OpenSession`.

## Banderas de trabajo

- `generic-contract`: contratos, handles, recibos y errores fail-closed.
- `runtime-host`: servidor IPC ejecutable sobre socket local.
- `external-routing`: el worker enruta Prefill/Decode/Snapshot/Restore al host externo.
- `session-recovery`: persistencia, reinicio y restauración de sesión.
- `e2e-real`: prueba con bundle real y generación real.
- `benchmark`: comparación densa, KV reutilizada y KVSLIM.
- `production-gate`: solo se activa cuando todas las anteriores pasan.

Las banderas son gates de validación, no sustitutos de funcionalidad. Una bandera no puede marcarse como completada por un mock.

## Trabajo paralelo

### Agente A — host IPC

Implementar un binario host que:

1. abra el endpoint declarado por el manifiesto;
2. valide el esquema;
3. responda a Probe y OpenSession;
4. ejecute Prefill y Decode mediante una RuntimeSession;
5. implemente SnapshotKv, RestoreKv y Release;
6. cierre con error ante esquema, recibo, identidad o sesión inválidos.

### Agente B — worker

Cuando exista `--runtime-manifest` o `KVSLIM_RUNTIME_MANIFEST`:

1. validar el manifiesto;
2. conectar al endpoint;
3. hacer Probe y OpenSession;
4. enrutar realmente la inferencia al host externo;
5. no cargar ni ejecutar un motor local como sustituto silencioso;
6. exponer en health si la sesión externa está realmente conectada.

### Agente C — sesiones

Conectar SessionStore/SessionCoordinator con:

- digest de modelo;
- digest de tokenizer;
- digest de arquitectura;
- digest de prefijo;
- generación monotónica;
- recibo exacto;
- restauración tras reinicio;
- rechazo de replay divergente.

### Agente D — validación y rendimiento

Medir con el mismo modelo y prompt:

- carga;
- prefill;
- decode;
- tokens por segundo;
- bytes leídos;
- bytes evitados;
- tokens KV reutilizados;
- tiempo de snapshot;
- tiempo de restore;
- memoria residente;
- fallback denso.

## Reglas inmutables

No tocar ni eliminar:

- micrometadatos dimensionales;
- LOD;
- certificados de atención;
- ColdStore;
- fallback exacto epsilon=0;
- recibos verificables;
- pruebas de determinismo.

No se permite que el núcleo dependa de nombres de proveedores o APIs concretas. Las implementaciones concretas solo pueden vivir en procesos o paquetes externos registrados por manifest.

## Criterios de aceptación

La tarea no se considera terminada hasta que:

- `cargo test --offline --locked --lib` pasa;
- compilan las features portables y aceleradas;
- existe un host IPC ejecutable;
- el worker usa realmente el host externo;
- una sesión genera tokens a través de IPC;
- snapshot y restore funcionan tras reiniciar el host;
- una identidad incompatible falla cerrado;
- se ejecuta una prueba con bundle real;
- existe benchmark reproducible;
- el health no afirma capacidades no demostradas;
- el árbol de trabajo queda limpio y el commit queda identificado.

## Estado actual

- Contratos genéricos: implementados.
- Dispatcher RuntimeHost: implementado.
- Transporte IPC: implementado.
- Handshake del worker: implementado.
- Enrutamiento real de inferencia externa: pendiente.
- Host IPC ejecutable: pendiente.
- E2E con modelo real: pendiente.
- Benchmark real: pendiente.
