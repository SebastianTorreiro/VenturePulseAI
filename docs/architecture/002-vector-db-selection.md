# 002: Selección de Base de Datos Vectorial: Qdrant vs PGVector

## Decisión: Qdrant
Se selecciona Qdrant sobre PGVector por las siguientes razones técnicas:

1. **Payload Filtering & Indexing**: Qdrant permite indexar campos del payload (ej. `amount_usd`). En PGVector, los filtros complejos sobre JSONB pueden degradar el rendimiento de la búsqueda ANN.
2. **Hardware Optimization**: Escrito en Rust, permite optimizaciones a nivel de SIMD para el cálculo de distancias.
3. **Hybrid Search Out-of-the-box**: Facilita la implementación de filtrado por facetas y búsqueda semántica en una sola solicitud atómica.
4. **HNSW Management**: Permite un control granular sobre la construcción del grafo de navegación (m, ef_construct), crítico para balancear latencia y recall.