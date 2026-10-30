# Revisión de literatura de 2006 aplicada a KVSLIM

Esta revisión no trata un paper como prueba de que KVSLIM sea correcto. Cada
idea se convierte en una restricción de diseño y debe conservar una prueba o
una medición local reproducible.

## Resultados aplicados

### Localidad y heterogeneidad

Chung, Graham, Bhagwan, Savage y Voelker estudian la asignación conjunta de
datos y tareas en sistemas heterogéneos y muestran que la localidad debe
formar parte de la decisión de colocación, no ser una consecuencia accidental.
Su resultado de aproximación se aplica a KVSLIM como una regla: el resolvedor
debe preferir una ejecución que conserve modelo, KV y ruta de acceso dentro del
mismo backend local, y no migrar silenciosamente a otro backend.

Referencia: https://www.sciencedirect.com/science/article/pii/S0022000006000791

### Presupuesto de memoria en índices de alta dimensión

El trabajo *Filter ranking in high-dimensional space* exige un umbral de
memoria principal y continúa funcionando cuando ese presupuesto es estrecho.
KVSLIM adopta ahora un presupuesto conservador para los pesos: un candidato no
puede entrar en selección automática si su tamaño declarado supera el 70 % de
la memoria disponible o si el tamaño no está disponible. El 70 % no es una
garantía universal: es un filtro preventivo que debe sustituirse por una
medición de carga real y debe dejar margen para runtime, KV y sistema.

Referencia: https://www.sciencedirect.com/science/article/abs/pii/S0169023X05000388

### Vecinos de alta dimensión y límites de aproximación

Andoni e Indyk estudian hashing aproximado para vecinos cercanos en espacios de
alta dimensión; Liu, Moore y Gray muestran que se puede acelerar la búsqueda
no paramétrica sin convertir automáticamente el resultado en una respuesta
aproximada no declarada. En KVSLIM esto refuerza dos reglas: el índice puede
proponer candidatos, pero la autoridad exacta del bundle y del KV no se
aproxima; y toda recomendación debe incluir su estado y sus límites.

Referencias:

- https://explore.openaire.eu/search/publication?pid=10.1145/1327452.1327494
- https://www.jmlr.org/papers/volume7/liu06a.html

### Contexto en recuperación de información

El trabajo de 2006 sobre uso de contexto en consultas personales defiende que
el contexto puede completar una petición incompleta y mejorar la recuperación,
pero no autoriza a inventar preferencias o identidad. KVSLIM conserva esa
separación: el contexto/KV puede orientar la selección, pero no almacena
nombres de usuario ni convierte señales externas en autoridad de ejecución.

Referencia: https://www.iadisportal.org/digital-library/usage-of-context-in-personal-queries

## Cambios implementados

- Cada adaptador declara `supported_model_types`.
- Un candidato sin arquitectura confirmada queda en `IncompatibleWithLocalAdapters`.
- Un candidato con arquitectura no declarada por ningún adaptador disponible se
  descarta de la selección automática.
- La ficha detallada de Hugging Face se consulta antes de puntuar, sin inferir
  la arquitectura a partir del nombre del repositorio.
- El catálogo recoge el tamaño declarado de los pesos cuando la API lo expone.
- La selección automática exige tamaño conocido y presupuesto de memoria.
- El runtime disponible, la carga real y la restauración de KV siguen siendo
  gates posteriores; la puntuación del catálogo no los sustituye.

## Limitaciones abiertas

La API de un catálogo no basta para demostrar que el modelo cabe con un
contexto concreto. La certificación final debe medir carga, prefill, decode,
uso de memoria y snapshot/restore de KV en el equipo actual. Ningún resultado
de esta revisión convierte OpenCL, Torch o cualquier otro backend en una
dependencia universal.
