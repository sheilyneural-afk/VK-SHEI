# KVSLIM

KVSLIM es un runtime local de almacenamiento, integridad y reutilización de
estado K/V para modelos de lenguaje. El núcleo no pertenece a un proveedor, un
modelo, un tokenizer ni un hardware concreto.

## Contrato universal

Un modelo se incorpora mediante una configuración y un adaptador. El adaptador
debe declarar tokenizer, arquitectura, precisión, operadores, contexto,
formato de pesos y capacidades de generación. KVSLIM inspecciona esa
declaración y rechaza cualquier combinación no demostrada; no selecciona un
modelo por nombre ni hace fallback silencioso.

El núcleo universal cubre:

- ColdStore K/V append-only con integridad por registro.
- Lectura, persistencia y recuperación de páginas K/V.
- Validación de geometría y configuración desde `config.json`.
- Servidor local y contrato de petición independiente del proveedor.
- Adaptación explícita a backends de inferencia.

Las rutas de inferencia acelerada son plugins opcionales. No forman parte del
contrato universal hasta que un adaptador específico demuestre compatibilidad
completa, paridad, persistencia, reinicio y servicio.

## Construcción y pruebas

```bash
cargo check --offline --no-default-features
cargo test --offline --lib --no-default-features
cargo build --offline --release --no-default-features
```

Los adaptadores de aceleración son opcionales y se habilitan únicamente cuando
el runtime correspondiente está instalado y validado en el equipo. El núcleo
no depende de una API, fabricante ni proveedor concretos.

El entorno de compilación, modelos, estado, logs y recibos se mantiene bajo
`deploy/local-production/`. No se usan rutas del sistema ni modelos externos.

## Entorno local

La guía operativa única está en
[deploy/local-production/README.md](deploy/local-production/README.md).

El contrato de adaptadores está en
[docs/MODEL_ADAPTERS.md](docs/MODEL_ADAPTERS.md).

La documentación histórica, benchmarks y corpus de prueba no forman parte del
árbol de producción y se han eliminado.

## Alcance actual

El núcleo y su contrato son universales. Las implementaciones opcionales de
inferencia existentes siguen necesitando adaptadores concretos y no deben
anunciar compatibilidad con cualquier LLM. Esta distinción es deliberada:
KVSLIM solo marca un modelo como ejecutable cuando la configuración y el
adaptador lo autorizan de forma verificable.
