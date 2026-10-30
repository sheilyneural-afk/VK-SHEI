# Entorno local de producción de KVSLIM

Este directorio define el entorno local aislado para ejecutar `KV-SHEI` como un
servicio controlado dentro del propio proyecto. No usa `/opt`, `/var`, el home
del usuario, un repositorio externo ni rutas de modelos descubiertas por
defecto.

## Principios

- Los artefactos activos viven bajo `deploy/local-production/runtime/`.
- Los modelos se instalan por digest y se leen en modo solo lectura.
- El binario, el modelo, el tokenizer, la configuración y el formato de
  `ColdStore` forman una generación compatible.
- El servicio debe escuchar únicamente en `127.0.0.1`.
- El endpoint no se considera productivo para red compartida: el servidor actual
  de KVSLIM no implementa autenticación.
- `health` significa proceso vivo; `ready` solo se acepta después de validar el
  binario, el modelo, la configuración y el almacenamiento local.
- El LOD positivo queda desactivado en el perfil seguro local hasta certificarlo
  con el modelo y el hardware concretos.
- No se mezclan modelos, K/V, estado de sesión, logs ni artefactos de build.

## Layout

```text
deploy/local-production/
├── README.md
├── env.example
├── local-production.env          # generado localmente; no versionar secretos
├── manifest.example.json
├── runtime/
│   ├── bin/                      # binario release validado
│   ├── models/<model-digest>/    # pesos y tokenizer, solo lectura
│   ├── coldstore/<worker-id>/    # K/V activo del worker
│   ├── state/                    # estado operativo local no cognitivo
│   ├── logs/
│   └── receipts/
└── scripts/
    ├── install-layout.sh
    ├── install-artifact.sh
    ├── verify-release.sh
    ├── run-local-runtime.sh
    └── cargo-local.sh
```

El script `install-layout.sh` prepara automáticamente todo el layout dentro de
este árbol. No instala drivers en el sistema ni usa rutas externas.

La instalación de una release debe seguir este orden:

1. preparar el layout local;
2. obtener el manifiesto de artefactos autorizado para la release;
3. descargar cada binario, runtime o modelo únicamente dentro de `runtime/`;
4. verificar SHA-256 y, cuando exista, la firma del manifiesto;
5. inspeccionar el modelo y sondear las capacidades del equipo;
6. activar solo un adaptador que haya pasado esas comprobaciones.

El usuario no debe instalar manualmente dependencias del proyecto. Cargo las
mantiene en `runtime/cargo-home` y el build en `runtime/build-target`. Los
drivers del sistema son la única excepción: no se pueden empaquetar de forma
universal dentro de KVSLIM y deben existir ya en el sistema operativo. Si un
driver o runtime no está disponible, KVSLIM lo informa y deja el núcleo de
contexto/KV utilizable, sin fingir que el modelo puede ejecutarse.

Los modelos y runtimes con licencia o distribución restringida requieren una
acción explícita del usuario. KVSLIM puede descargar y verificar el artefacto
cuando el manifiesto autorizado contiene su origen, versión y SHA-256, pero no
debe descubrir URLs arbitrarias ni aceptar archivos sin hash.

## Modelo local

El modelo lo aporta el usuario o la empresa que tenga derecho a utilizarlo.
KVSLIM no impone una arquitectura, una marca, un proveedor ni un único formato
de pesos. Cada adaptador declara los formatos que entiende y rechaza el bundle
si no puede demostrar compatibilidad.

El bundle activo debe vivir dentro del proyecto:

```text
deploy/local-production/runtime/models/<sha256-del-bundle>/
├── config.json
├── tokenizer.json o tokenizer.model
├── pesos del modelo en el formato declarado por el adaptador
└── manifiesto de integridad
```

El usuario puede introducirlo mediante una importación local o mediante una
descarga declarada en un manifiesto autorizado. En ambos casos KVSLIM calcula y
comprueba el digest antes de activarlo, lo abre en modo solo lectura y registra
la referencia por digest, nunca por un nombre ambiguo o una ruta externa.

No se copia un modelo a una carpeta global del sistema. No se usa una ruta de
usuario como autoridad de producción y no se modifica el bundle durante la
ejecución. Si el modelo necesita conversión, se genera otro bundle versionado y
verificable; el original se conserva intacto.

Para consultar qué bundles existen y qué puede inspeccionarse localmente:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --no-default-features --bin kvslim-models
```

La salida solo enumera bundles bajo `runtime/models/`. No escanea el sistema,
no envía información a una red y no recomienda ejecución hasta que el sondeo
del runtime y la validación del adaptador hayan pasado.

Este comando no consulta catálogos remotos. Si en el futuro se habilita un
proveedor de descubrimiento, podrá proponer modelos desde un registro externo,
pero la descarga requerirá autorización y el modelo solo será válido después
de importarse y verificarse localmente. Un registro remoto nunca será la
autoridad de ejecución.

La capa de catálogo permite que KVSLIM encuentre candidatos por el usuario:
detecta primero las capacidades locales y después filtra por formato,
arquitectura, memoria y adaptador. Una recomendación remota no se activa por sí
sola; el resolvedor debe importar y probar el bundle dentro del proyecto antes
de declararlo utilizable.

Para consultar y ordenar candidatos en función del equipo actual:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-catalog --bin kvslim-recommend -- \
  "text-generation" 20
```

La salida contiene el inventario efímero del host, la puntuación, los motivos y
las exclusiones de cada candidato. No descarga modelos ni marca ninguno como
ejecutable.

Cuando exista un adaptador local sondeado, la descarga puede seleccionar
automáticamente el mejor candidato:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-catalog --bin kvslim-download-model -- \
  --auto "text-generation"
```

El modo automático se detiene si no hay adaptador local disponible. Descargar
un candidato no equivale todavía a certificarlo: la carga real y la prueba de
KV siguen siendo obligatorias antes de activarlo.

Las operaciones privadas usan un token de lectura, nunca una contraseña:

```bash
printf '%s\n' 'hf_TU_TOKEN' | deploy/local-production/scripts/cargo-local.sh run \
  --locked --features model-catalog --bin kvslim-auth -- store
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-catalog --bin kvslim-auth -- validate
```

Para importar un modelo que ya posee el usuario o la empresa, sin dejarlo
fuera del proyecto:

```bash
deploy/local-production/scripts/cargo-local.sh run --locked \
  --features model-import --bin kvslim-import-model -- \
  /ruta/autorizada/al/bundle
```

El importador inspecciona la configuración antes de copiar, rechaza enlaces
simbólicos, calcula un digest determinista sobre las rutas y bytes, genera
`bundle-manifest.json` y activa el resultado como
`runtime/models/<digest>`. El origen nunca se modifica y el bundle activo no se
selecciona por nombre.

Para instalar un artefacto autorizado, la release debe declararlo primero en
`manifest.example.json` (copiado como manifiesto de la release) con `id`,
`kind`, `source`, `destination`, `version`, `sha256`, `required` y la política
de licencia. No se deben pasar destinos o URLs improvisados a una release.

El instalador de un artefacto individual se usa así:

```bash
deploy/local-production/scripts/install-artifact.sh \
  https://origen-autorizado/artefacto \
  SHA256_DE_64_CARACTERES \
  runtimes/<runtime>/<archivo>
```

El instalador solo acepta HTTPS, rechaza destinos absolutos o que escapen de
`runtime/`, descarga a un temporal local, verifica SHA-256 y mueve el archivo
de forma atómica. No instala drivers, no ejecuta el archivo descargado y no
acepta una descarga cuyo hash no coincida.

## Cargo incremental y recuperable

Usa siempre el wrapper local para que el registro de crates, descargas,
checkouts, temporales y `target` permanezcan dentro del proyecto:

```bash
deploy/local-production/scripts/cargo-local.sh fetch --locked
deploy/local-production/scripts/cargo-local.sh check --locked --no-default-features
deploy/local-production/scripts/cargo-local.sh test --locked --lib --no-default-features
```

El `Cargo.lock` fija las versiones. Las correcciones normales reutilizan el
registro y el target incremental; Cargo recompila únicamente las cajas
afectadas y sus dependientes. No se debe borrar `runtime/cargo-home` ni
`runtime/build-target` para corregir código. La limpieza, si fuese necesaria,
debe hacerse por perfil y conservar el registro de crates y el target de
desarrollo.

## Perfil inicial recomendado

El primer perfil local es automático y portable:

```text
backend              = auto / runtime compatible validado
lod                  = desactivado
host                 = 127.0.0.1
port                 = 8088
max_context_tokens   = explícito por entorno
max_new_tokens       = explícito por entorno
model                = ruta absoluta dentro de runtime/models
```

Los aceleradores se incorporan como workers separados cuando el dispositivo,
el runtime, el modelo y las pruebas físicas del nodo local hayan sido
verificados. Ningún runtime concreto forma parte del contrato público ni es
requisito del perfil local.
No se permite un fallback silencioso CPU/GPU.

## Preparación

Desde la raíz de `KV-SHEI`:

```bash
deploy/local-production/scripts/install-layout.sh
cp deploy/local-production/env.example deploy/local-production/local-production.env
```

El instalador debe colocar el modelo dentro de
`deploy/local-production/runtime/models/<digest>` y registrar sus hashes en el
manifiesto antes de arrancar. No se aceptan rutas externas ni modelos elegidos
por nombre sin verificación.

Después construye el binario con el perfil que corresponda y colócalo en:

```text
deploy/local-production/runtime/bin/kvslim-server
```

La verificación es obligatoria:

```bash
deploy/local-production/scripts/verify-release.sh
```

El script falla si falta el manifiesto, si las rutas escapan del proyecto, si
el binario o el modelo no coinciden con sus hashes, si el modelo es escribible,
si el bind no es loopback o si el formato de ColdStore no está declarado.

## Estado de producción

Este entorno es local y cerrado, pero todavía no es una certificación de
producción general. Antes de habilitar tráfico real deben pasar los gates de
compilación, tests, backend físico, exactitud, carga, recuperación y seguridad.
La ausencia de autenticación en el servidor actual mantiene el alcance en
loopback.
