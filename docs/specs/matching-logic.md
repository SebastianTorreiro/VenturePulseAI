# Spec: Matching Logic & Re-ranking

## El Algoritmo
El matching no es binario. Es un score compuesto:
`Final_Score = (alpha * Semantic_Similarity) + (beta * Signal_Strength)`

- **Signal_Strength**: Función logarítmica sobre el monto de inversión. Una Serie B (+$20M) pesa más que una Seed ($500k).
- **Reciprocal Rank Fusion (RRF)**: Si se usan múltiples modelos, se combinan los rankings para evitar el sesgo de un solo modelo.

## Evaluación (Benchmarking)
Se implementará un set de pruebas "Golden Dataset" para medir el **Hit Rate @ K** y asegurar que el sistema realmente prioriza empresas relevantes.