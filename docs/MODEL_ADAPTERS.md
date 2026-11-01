# Adaptadores de modelo de M-SLIM

## Contrato

El contrato ejecutable vive en `crates/mslim-adapter-api`. `ModelAdapter`
separa el núcleo de una implementación física y define estas operaciones:

- declarar capacidades;
- inspeccionar un bundle sin activarlo;
- cargar el modelo;
- ejecutar prefill y decode;
- obtener los últimos logits cuando el adapter los expone;
- reiniciar la sesión;
- emitir y restaurar un snapshot K/V;
- emitir, de forma opcional, un informe de integridad K/V.

`MslimEngine` conserva el estado del ciclo de vida y no permite prefill, decode
o restore antes de cargar el adapter.

## Catálogo y disponibilidad

El catálogo distingue cuatro estados:

- `Compiled`: el código existe en el binario;
- `RuntimeAvailable`: el sondeo local encontró sus requisitos;
- `RuntimeUnavailable`: el sondeo comprobó que faltan requisitos;
- `RequiresProbe`: no hay evidencia suficiente para decidir.

Una feature de Cargo o una entrada de catálogo no basta para ejecutar un modelo.
El adapter debe estar disponible en ese host y el bundle debe superar su propia
inspección.

El inventario del host se calcula localmente. No convierte un modelo remoto en
autoridad ni instala drivers. Los candidatos de un catálogo remoto solo son
propuestas hasta que se importan por digest dentro de
`deploy/local-production/runtime/models/`.

## Implementaciones actuales

### CPU Rust

`native-cpu-runtime` carga un bundle soportado por `ModelBundle`. Su ruta exacta
conserva K/V BF16 en ColdStore. El parámetro `native_hot_kv_bytes` controla qué
cantidad puede atenderse desde el nivel caliente:

- un presupuesto suficiente produce la referencia `cpu-hot-dense`;
- presupuesto cero obliga a releer desde ColdStore y produce
  `cpu-exact-coldstore`.

Las dos rutas usan el mismo escalado de atención y la matriz exige logits,
tokens y digest K/V idénticos.

### Acelerador GPU

`accelerator-runtime` acepta dos modos explícitos mediante `mslim_kv_mode`:

- `dense`: K/V denso en el dispositivo;
- `exact-coldstore`: ColdStore BF16 exacto y snapshot físico.

Otro valor se rechaza. El modo por defecto del adapter es
`exact-coldstore`. El modo denso no puede declarar un epsilon LOD positivo.

`mslim_weight_mode` admite `stream` y `resident`. Este “stream” describe cómo se
cargan los pesos y no habilita streaming HTTP.

### Otras features

El repositorio también contiene rutas opcionales LibTorch e HIP. Su presencia
en el código no las convierte en la referencia de la matriz actual ni demuestra
disponibilidad en el host.

## Inspección del bundle

La inspección valida, según el adapter, configuración, tokenizer, shards
SafeTensors, geometría de atención, dtypes y layout del decoder. Un bundle que
solo puede inspeccionarse no debe anunciarse como ejecutable.

El bundle activo se identifica por digest. El runtime no debe seleccionar un
modelo por un nombre ambiguo ni modificar sus archivos durante la inferencia.

`mslim-validate-model` emite `mslim.validation_receipt/v2`. Ese recibo declara
que un adapter concreto completó inspect, load, prefill, decode, snapshot y
restore. Es distinto de `mslim.validation-matrix/v2`, que compara las cuatro
rutas exactas. Los consumidores que piden uno no deben aceptar el otro por
similitud de nombre.

## Requisitos para aceptar una ruta exacta

La validación se hace contra la referencia densa del mismo backend:

1. repetir la ruta densa para comprobar determinismo;
2. repetir la ruta exacta para comprobar determinismo;
3. comparar densa y exacta con los mismos inputs;
4. exigir logits y tokens idénticos;
5. exigir SHA-256 lógico K/V idéntico;
6. para GPU exacta, exigir un snapshot físico de ColdStore.

CPU y GPU se comparan por métricas diagnósticas; la igualdad bit a bit entre
ellos no es el gate de integridad. Véase [VALIDATION.md](VALIDATION.md).

## Requisitos para añadir un adapter o una arquitectura

Como mínimo:

- inspección real del bundle y rechazo de layouts no soportados;
- probe de disponibilidad del runtime y del dispositivo;
- prefill, decode, reset, snapshot y restore;
- determinismo caracterizado;
- referencia densa y ruta exacta comparadas dentro del mismo backend;
- digest lógico K/V;
- rechazo de identidad, geometría o recibo incompatibles;
- prueba de reinicio que indique con precisión qué proceso se reinicia;
- prueba HTTP o IPC por la ruta pública que se pretende usar;
- medición de rendimiento separada de la validación de integridad.

Para una ruta aproximada hacen falta además métricas de calidad y memoria sobre
un corpus representativo. La matriz exacta con `epsilon = 0` no autoriza
`epsilon > 0`.

## Credenciales y catálogo remoto

La feature `model-catalog` puede consultar candidatos remotos. `mslim-auth`
gestiona un token de lectura. El catálogo no decide compatibilidad, no activa un
modelo y no sustituye el digest del bundle local.

La compatibilidad efectiva siempre depende de tres elementos: adapter
disponible, bundle validado y evidencia del perfil concreto.
