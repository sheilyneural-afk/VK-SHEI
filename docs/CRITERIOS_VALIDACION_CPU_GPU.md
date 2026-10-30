# Criterios de validación numérica: CPU, GPU y LOD

**Vigente desde el 30/10/2026.** Política de interpretación para todos los
informes, benchmarks, mensajes de interfaz y pruebas de VK-SHEI / Terrsvusion.
Esta política aclara la lectura de las mediciones históricas sin reescribirlas.

## 1. No exigir identidad binaria CPU/GPU

La igualdad bit a bit de los logits, activaciones o K/V calculados por
**CPU LibTorch** y **GPU Rust/OpenCL** **no es un requisito funcional** de la
ruta GPU ni una condición de aceptación de LOD. Dos backends pueden utilizar
diferentes órdenes de reducción, operaciones fusionadas, acumulaciones FP32,
redondeos BF16 y estrategias de ejecución. Pueden producir resultados
numéricamente diferentes aun usando los mismos pesos y tokens de entrada.

Las comparaciones CPU/GPU siguen siendo **diagnósticos útiles**: con los
mismos pesos, IDs de entrada, máscara, RoPE, parámetros y política greedy
deben registrarse las diferencias de logits, K/V, tokens y rendimiento.
Una diferencia no demuestra automáticamente un bug; tampoco demuestra
por sí sola que la GPU sea correcta. Las discrepancias grandes, los
cambios inesperados en tokens y la degradación de calidad deben investigarse.

## 2. Comprobar LOD contra la referencia del mismo backend

- **GPU LOD con epsilon=0 vs GPU densa:** comprobar la equivalencia de
  resultados bajo los mismos kernels y condiciones. En las pruebas GPU
  citadas hubo igualdad bit a bit para las entradas medidas; no se
  extrapola automáticamente a todas las entradas o plataformas.
- **GPU LOD con epsilon>0 vs GPU densa:** exigir que las omisiones estén
  respaldadas por certificados válidos y verificar que las diferencias
  reales respeten el contrato de error que se haya definido. Una cota
  local de atención **no garantiza por sí sola** identidad de logits ni
  de tokens a nivel de Transformer completo. Sin garantías suficientes,
  usar fallback seguro y no declarar la ruta apta para servicio.
- **CPU LOD vs CPU densa:** conservar las condiciones de exactitud y
  aceptación específicas de ese backend. Un gate de `L∞=0` o de SHA
  dentro de CPU no debe confundirse con una exigencia GPU/CPU.

## 3. Independencia de la integridad de almacenamiento

La verificación de los bytes originales BF16 (pesos y K/V persistidos)
mediante SHA-256, readback y comparaciones exactas **sí** debe ser
estricta cuando el contrato promete almacenamiento sin pérdida. Esto
no implica que CPU y GPU deban calcular activaciones idénticas bit a bit.

## 4. Interpretación de los datos históricos

Los valores publicados, como `L∞ logits=0,5` y SHA lógico KV diferente
en una comparación CPU/GPU, **se mantienen como mediciones**. No se
interpretan ni como un requisito fallido de paridad binaria ni como prueba
de precisión universal. Las pruebas de paridad GPU densa vs GPU LOD
`epsilon=0` miden una propiedad diferente y son las relevantes para
esa optimización. Las comparaciones de velocidad necesitan misma
carga útil, generación solicitada y condiciones experimentales.

Los JSON, logs y pruebas históricas se preservan sin modificar los
resultados medidos. Ante cualquier texto anterior ambiguo, prevalecen
estos criterios de interpretación.
