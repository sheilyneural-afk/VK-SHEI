# Tarea: KV-SHEI runtime universal de extremo a extremo

## Objetivo

Completar la ruta universal real:

manifest -> descubrimiento -> host IPC -> sesión -> Prefill/Decode -> snapshot/restore KV -> worker HTTP.

El núcleo no debe decidir ni nombrar proveedores concretos. Los runtimes físicos se instalan externamente y se conectan mediante el contrato genérico.

## Banderas de activación

- `MSLIM_RUNTIME_MANIFEST`: manifiesto del runtime externo.
- `MSLIM_RUNTIME_REQUIRED=1`: obliga a usar runtime externo; si no existe o falla, arranque fail-closed.
- `MSLIM_RUNTIME_E2E=1`: activa la prueba completa de socket y generación.
- `MSLIM_RUNTIME_ALLOW_LOCAL_FALLBACK=0`: por defecto no se permite sustituir silenciosamente el runtime externo.
- `MSLIM_RUNTIME_SOCKET`: endpoint Unix del host.
- `MSLIM_TEST_MODEL_DIR`: bundle de prueba autorizado.

## Entregables obligatorios

1. Host IPC ejecutable con protocolo versionado.
2. Worker que enrute realmente Prefill/Decode/Snapshot/Restore al host externo.
3. Ningún handshake que luego ignore la sesión.
4. Fail-closed ante manifest inválido, capacidades insuficientes, identidad distinta o recibo KV inválido.
5. Prueba E2E con socket real y sesión determinista.
6. Prueba de reinicio: snapshot, apagar host, reiniciar, restore y continuar.
7. Prueba de fallback explícito: solo funciona si la bandera lo permite.
8. Tests del núcleo sin backend físico.
9. Comprobación de que micrometadatos, LOD, ColdStore y certificados siguen intactos.
10. Benchmark separado de carga, prefill, decode, restauración y bytes evitados.
11. Health debe declarar si la sesión externa está realmente conectada y siendo usada.
12. No afirmar paridad ni rendimiento sin medición.

## Criterios de aceptación

- `cargo fmt --check` correcto.
- `cargo check --locked` correcto.
- `cargo test --locked --lib` correcto.
- `cargo test --locked --features accelerator-backend --lib` correcto cuando el entorno lo permita.
- La prueba E2E usa un socket real, no solo un mock en memoria.
- Una petición Prefill/Decode llega al host y vuelve con el mismo request_id.
- Snapshot/Restore valida identidad, digest, generación y bytes.
- Si el host no responde, el worker devuelve error explícito y no ejecuta otra ruta.
- El sistema sigue compilando sin ninguna implementación física.
- Los cambios se separan en commits pequeños y revisables.

## Política de implementación

No borrar ni degradar:

- PageMicroMetadata.
- DimensionalMicroMetadata.
- MicroRegionPlan.
- LOD exacto y aproximado.
- fallback denso exacto.
- ColdStore.
- certificados de atención.
- continuidad y SessionStore.

Los nombres de proveedores o APIs físicas solo pueden aparecer en módulos/procesos externos opcionales. El contrato público debe expresarse en capacidades, manifiestos, handles y recibos.

## Orden de trabajo

1. Host IPC ejecutable.
2. Cliente y worker externo.
3. Snapshot/restore y reinicio.
4. Tests E2E.
5. Limpieza de claims y nombres.
6. Benchmark.
7. Commit final y revisión de diff.
