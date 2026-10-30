# Documentación mínima de KVSLIM

KVSLIM mantiene una documentación deliberadamente pequeña para que el árbol de
producción no dependa de informes históricos, corpus ni benchmarks concretos.

- [MODEL_ADAPTERS.md](MODEL_ADAPTERS.md): contrato universal de configuración y
  adaptadores.
- [deploy/local-production/README.md](../deploy/local-production/README.md):
  ejecución local, layout, hashes y verificación.

Los detalles de un modelo concreto deben vivir en su adaptador y en su
manifiesto de release, no en el núcleo ni en esta documentación.
