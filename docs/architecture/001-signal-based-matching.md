# 001: Signal-Based Matching Architecture

## Contexto
El mercado laboral de IA es hipersensible a los flujos de capital. Las vacantes suelen publicarse 2-4 semanas después de una ronda de inversión. 

## Objetivos del Sistema
- **Proactividad**: Identificar empresas con "Buying Power" antes de que saturen los job boards.
- **Precisión Semántica**: Superar el keyword-matching tradicional usando representaciones vectoriales de alta densidad.

## Flujo de Datos (Data Flow)
1. **Ingestion Layer**: Scrapers asíncronos recolectan noticias de Crunchbase/TechCrunch.
2. **Intelligence Layer**: LLM extrae entidades (Monto, Serie, Tesis) y genera un "Vector de Crecimiento".
3. **Storage Layer**: Qdrant almacena el vector con su payload (señales financieras).
4. **Matching Layer**: API de FastAPI recibe el CV del usuario y ejecuta una búsqueda híbrida.