# Spec: Embedding Pipeline & Chunking Strategy

## Modelo Seleccionado
`text-embedding-3-small` (OpenAI) debido a su capacidad de **Matryoshka Embeddings**, que permite reducir dimensiones sin pérdida lineal de información.

## Álgebra y Geometría
- **Métrica de Similitud**: Producto Punto (Dot Product) si los vectores están normalizados, o Coseno.
- **Estrategia de Chunking**: 
    - **Señales**: 1 solo chunk por noticia (resumen por LLM).
    - **CV/Perfil**: Chunking semántico basado en secciones (Experiencia, Skills, Proyectos).
- **Normalización**: Todos los vectores serán normalizados a L2 para asegurar consistencia en la métrica de distancia.