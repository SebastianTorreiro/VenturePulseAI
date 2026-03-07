# Spec: Ingestión de Datos y Evasión de Anti-Bots

## Fuentes de Datos
- **RSS/News Aggregators**: Para detectar eventos de inversión.
- **Web Scraping**: Extracción de contenido de blogs corporativos.

## Estrategia de Evasión
1. **User-Agent Rotation**: Uso de encabezados dinámicos.
2. **Asincronía**: Uso de `httpx` con semáforos para evitar rate-limiting.
3. **Proxy Layer**: Preparado para rotación de IPs (Middleware pattern).

## Pipeline de Procesamiento
- **Limpieza**: Remoción de boilerplate HTML (usando Selectolax por velocidad).
- **Validación**: Pydantic garantiza que la "señal" tenga monto, moneda y empresa antes de pasar al motor de embeddings.