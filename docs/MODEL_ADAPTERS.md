# M-SLIM: núcleo backend-neutral y adaptadores de ejecución

M-SLIM es el nombre único del producto y de su API local. El núcleo trabaja con
metadatos de checkpoint, geometría K/V, serialización, integridad y almacenamiento
sin fijar un proveedor, una marca o un modelo como identidad del sistema.

El contrato ejecutable vive en el crate independiente
`crates/mslim-adapter-api`. Cada implementación se registra con backend,
tipo de cómputo, prioridad y política de activación. El núcleo no presupone
ninguna API, fabricante, proveedor ni runtime concreto. Un backend que dependa
de hardware o bibliotecas externas debe quedar en modo explícito hasta
que exista una detección real de disponibilidad y capacidades en ese equipo.
Si no existe una coincidencia probada, la selección falla; no hay fallback
silencioso ni se presenta un backend opcional como compatible sin pruebas.

El catálogo distingue entre backend compilado y runtime disponible en el equipo:
`Compiled`, `RuntimeAvailable`, `RuntimeUnavailable` y `RequiresProbe`. Que una
opción exista en el binario no demuestra que sea utilizable en otro ordenador.
La detección solo observa el estado local; no instala drivers ni fuerza una API.

El inventario del host se calcula en memoria al iniciar o consultar el catálogo.
Incluye únicamente capacidades operativas —por ejemplo, disponibilidad de CPU,
paralelismo y runtimes que responden— y no conserva nombres de usuario, rutas
personales, identificadores de máquina ni datos de identidad.

El inventario local no consulta automáticamente Hugging Face ni ningún catálogo
remoto. Puede existir un proveedor de catálogo opcional —Hugging Face u otro—
para descubrir modelos, versiones, licencias y formatos, pero ese proveedor no
forma parte del núcleo ni decide la compatibilidad. La autoridad es siempre el
bundle que el usuario o la empresa ha importado dentro de `runtime/models/` y
que M-SLIM ha verificado por contenido.

La dirección de producto es que el usuario no tenga que conocer de antemano el
modelo correcto: M-SLIM debe detectar capacidades locales, consultar uno o
varios catálogos, descartar candidatos incompatibles y presentar una
recomendación explicada. La recomendación no convierte un candidato remoto en
autoridad. Antes de activarlo, el resolvedor debe descargarlo dentro del
proyecto, verificar licencia, tamaño, formato, digest, arquitectura,
tokenizer, memoria y una carga real con el adaptador seleccionado. Si falla
cualquier comprobación, el candidato queda en `inspect-only`.

Las credenciales privadas se gestionan como tokens de lectura de Hugging Face,
nunca como contraseñas. El gestor opcional de credenciales vive en
`mslim-auth`; valida el token contra el servicio antes de una operación
autenticada, no lo incluye en diagnósticos y permite eliminarlo.

La ejecución de un modelo concreto se organiza como un adaptador. Un adaptador
puede declarar qué arquitectura, precisión, tokenizer, RoPE, operadores y formato
de pesos admite. Si el adaptador no puede demostrar compatibilidad completa,
M-SLIM debe rechazar la ejecución y permitir únicamente inspección; no debe
simular compatibilidad ni seleccionar una arquitectura por nombre.

Los perfiles concretos que aparecen en las pruebas de referencia son datos de
validación, no contratos públicos ni nombres de M-SLIM. Los checkpoints y sus
tokenizers deben residir exclusivamente bajo
`deploy/local-production/runtime/models/<digest>` y se seleccionan mediante
configuración y digest, nunca mediante rutas personales o absolutas.
El bundle lo aporta el usuario o la empresa que tenga derecho a utilizarlo; el
adaptador declara el formato aceptado y M-SLIM no convierte ni modifica el
original durante la ejecución.

Para incorporar una nueva arquitectura o hardware se requiere un adaptador explícito,
pruebas de geometría y tokenizer, paridad de logits, persistencia/relectura de
K/V, reinicio y prueba de servicio. Hasta completar esas pruebas, el perfil se
considera `inspect-only`.
