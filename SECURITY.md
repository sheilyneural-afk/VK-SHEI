# Seguridad

## Informar de una vulnerabilidad

No abra un Issue público con una vulnerabilidad explotable, credenciales,
secretos, rutas privadas ni datos de usuarios. Utilice **Security → Report a
vulnerability** en GitHub para enviar un informe privado al mantenedor.

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
