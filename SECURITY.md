# Seguridad

## Informar de una vulnerabilidad

No abra un Issue público con una vulnerabilidad explotable, credenciales,
secretos, rutas privadas ni datos de usuarios. Utilice **Security → Report a
vulnerability** en GitHub para enviar un informe privado al mantenedor:
[reporte privado de seguridad](https://github.com/sheilyneural-afk/KV-SHEI/security/advisories/new).

La política versionada está en este archivo. La postura automatizada se publica
en [Security](https://github.com/sheilyneural-afk/KV-SHEI/security), y el método
para interpretar CodeQL, Dependabot, Scorecard y los recibos de CI está en
[`docs/GITHUB_EVIDENCE.md`](docs/GITHUB_EVIDENCE.md). Los avisos públicos de
dependencias se gestionan como deuda sin publicar detalles de explotación no
coordinados.

Incluya, cuando sea seguro:

- componente, commit y configuración afectados;
- modelo de amenaza y frontera de confianza;
- reproducción mínima y efecto observable;
- impacto sobre autorización, continuidad, K/V, ColdStore o persistencia;
- mitigaciones conocidas y material que deba redactarse.

## Alcance de las afirmaciones

Un test unitario, un `HTTP 200`, un endpoint de salud o un manifest no prueban
por sí solos seguridad ni preparación productiva. La evaluación debe enlazar
fuente, configuración efectiva, build, artefacto instalado, proceso residente,
petición autorizada, efecto, persistencia, readback y reinicio cuando aplique.

No se garantiza un plazo de resolución. El mantenedor confirmará recepción,
validará el alcance y coordinará publicación y corrección de forma proporcional
al riesgo.

## Versiones mantenidas y divulgación

No existe todavía una release estable declarada como camino productivo
soportado. La rama `main`, una build de CI o un artefacto atestado no implican
soporte de seguridad ni readiness productiva. Cuando exista una release
soportada, esta sección deberá enumerar explícitamente sus versiones y fecha de
fin de soporte.

No publique el informe antes de que el mantenedor confirme una ventana de
divulgación coordinada. Si el reporte es válido, la respuesta distinguirá
recepción, reproducción, corrección en fuente, build, publicación e instalación;
ninguno de esos estados sustituye a los demás.
