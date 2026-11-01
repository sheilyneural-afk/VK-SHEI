# Propiedad y ciclo de vida de la continuidad

Este documento define quién puede pedir un snapshot o una restauración, a qué
sesión pertenece y cuándo deja de ser utilizable. El contrato cubre la
continuidad física K/V del worker conectado a un runtime externo; no convierte
M-SLIM en un servicio multiusuario.

## Límite de autoridad

La instalación local tiene un único límite de acceso: el gateway autenticado.
Quien posee su credencial puede enviar una operación de continuidad. El sistema
que llama al gateway es el propietario de la relación entre su identidad
gobernada (usuario, agente, conversación o trabajo) y
`opaque_session_handle`. M-SLIM no recibe ni infiere esa identidad gobernada.

Por tanto:

- el llamador crea un identificador opaco nuevo y conserva su asociación con el
  propietario real;
- el llamador decide cuándo solicitar `snapshot`, `resume` o `release`;
- el gateway decide si la petición puede entrar según su autenticación;
- el worker decide si la transición es válida según el estado durable, la
  generación, los cuatro digests, la caducidad y los límites;
- el runtime externo ejecuta el snapshot, restore o reset físico, pero no decide
  a quién pertenece la sesión.

El `policy_digest` vincula una política ya decidida por el llamador; no es una
credencial ni reemplaza la autenticación. Para ofrecer aislamiento entre varios
usuarios sería necesario un plano de identidad y autorización por principal que
no existe en esta instalación. Hasta entonces, el contrato demostrado es de
propietario local único.

## Identidad durable

Una sesión explícita queda vinculada, desde su primer snapshot, a:

- `opaque_session_handle`;
- `model_digest`;
- `tokenizer_digest`;
- `prefix_digest`;
- `policy_digest`;
- `expires_at_unix_s`;
- la generación actual;
- los tokens y bytes físicos medidos en el snapshot.

El identificador externo no se usa como nombre de directorio. El worker lo
deriva mediante SHA-256 con separación de dominio. El estado durable usa el
schema cerrado `mslim.worker.continuity-state/v1`; otra versión o campos
desconocidos se rechazan. Las conversaciones
automáticas usan el dominio `conversation` y la continuidad explícita usa el
dominio `continuity`; el mismo texto externo no puede hacer que ambos estados
colisionen.

## Máquina de estados

| Estado actual | Operación aceptada | Condición principal | Estado resultante |
|---|---|---|---|
| ausente | `snapshot` | generación esperada `0` | activo, generación `0` |
| activo `N` | `snapshot` | generación `N`, misma identidad y sesión no caducada | activo, generación `N+1` |
| activo `N` | `resume` | generación `N`, misma identidad, política, vigencia y límites | activo, generación `N` |
| activo `N` | `release` | generación `N` y misma identidad | liberado, generación `N+1` |
| liberado `N` | cualquiera | ninguna | rechazo |

`release` resetea el estado causal del runtime y escribe una lápida durable. No
borra la historia ni permite reutilizar el handle: para una sesión nueva el
llamador debe crear un identificador nuevo. La lápida hace que un `resume`
posterior falle también después de reiniciar el worker.

## Reglas de restauración

Antes de enviar `RestoreKv`, el worker exige todo lo siguiente:

1. existe una generación durable y coincide exactamente con
   `expected_generation`;
2. el estado no está liberado;
3. modelo, tokenizer, prefijo y política coinciden;
4. el snapshot almacenado no ha caducado;
5. la capacidad presentada no amplía la caducidad almacenada;
6. los tokens reutilizados no superan `max_reused_tokens`;
7. los bytes del prefijo físico ColdStore no superan
   `max_restored_bytes`;
8. el runtime valida su propio recibo y la integridad K/V antes de aceptar el
   restore.

El recibo de continuidad informa los tokens del historial físico y los bytes
del prefijo ColdStore. No usa el tamaño del JSON que transporta el handle como
sustituto de los bytes K/V.

## Persistencia y conservación

`SessionStore` conserva una cadena append-only de generaciones y un puntero
`CURRENT`. Los snapshots físicos están fijados por su hash y por el tamaño del
prefijo ColdStore. El arranque del worker crea un namespace físico nuevo: una
sesión anterior solo vuelve a ser causal mediante un `resume` que supera todas
las comprobaciones anteriores.

No hay recolección automática de snapshots caducados o liberados. Se conservan
para auditoría y recuperación forense; borrarlos sin inventario rompería esa
propiedad. La limpieza física requiere una herramienta separada con inventario,
retención y borrado auditable. Esa herramienta todavía no forma parte del
producto y, por tanto, la liberación lógica no debe interpretarse como borrado
físico.

## Evidencia ejecutable

`scripts/e2e-worker-external-restart.sh` demuestra la ruta real:

1. genera K/V en un runtime externo;
2. crea el snapshot explícito;
3. reinicia el worker sin reiniciar el proveedor;
4. restaura la generación durable y comprueba tokens y bytes físicos;
5. libera la sesión y crea una generación nueva con lápida;
6. prueba que esa generación liberada ya no puede restaurarse.

Las pruebas unitarias demuestran además que los namespaces automáticos y
explícitos son deterministas, internos y separados por dominio, y que estados
durables con campos desconocidos se rechazan.
