# CLAUDE.md

Antes de cualquier cambio en este repositorio, leé en este orden:

1. `ARCHITECTURE.md` — capas, puertos, ADRs y reglas de dependencia.
2. `CONVENTIONS.md` — nomenclatura, plantilla de casos de uso, manejo de errores y tests.

Reglas no negociables:
- Ninguna importación de SDKs o frameworks dentro de `app/domain/` o `app/application/`.
- Toda excepción cruda de un SDK se traduce a `InfrastructureError` o subclase, con `raise ... from e`.
- Cada puerto debe tener al menos una implementación FREE y una PAID, intercambiables por `.env`.
- Tests unitarios sin red, sin Docker, sin sleep.

Antes de proponer código:
- Identificá la capa donde vive el cambio.
- Si requiere modificar un caso de uso para soportar una nueva fuente o adaptador, parate: probablemente la abstracción está mal.

Idioma: código en inglés, documentación en español.