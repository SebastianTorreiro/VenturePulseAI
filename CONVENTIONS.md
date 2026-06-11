# CONVENTIONS.md — VenturePulseAI

> Guía operativa para cada sesión de desarrollo (humana o con IA). Si una regla de aquí contradice al código existente, gana esta guía: corregí el código o abrí una discusión, no propagues la inconsistencia.

**Prerequisito:** leé [ARCHITECTURE.md](ARCHITECTURE.md) primero. Este documento asume sus 4 capas (`domain` → `application` → `infrastructure` → `api/cli`) y sus puertos (`ISignalRepository`, `IEmbeddingService`, `ISignalScraper`, `ILLMService`, `ICVGenerator`).

**Regla de oro (DIP):** ninguna clase de `domain/` o `application/` importa SDKs, frameworks ni nada de `infrastructure/`. Si estás escribiendo `from qdrant_client import ...` fuera de `infrastructure/`, parate: estás en la capa equivocada.

---

## 1. Nomenclatura

### 1.1 Archivos y paquetes

| Elemento | Convención | Ejemplo |
|---|---|---|
| Módulos | `snake_case.py`, un concepto por archivo | `funding_round.py`, `qdrant_signal_repository.py` |
| Entidades | singular | `signal.py`, no `signals.py` |
| Casos de uso | verbo_objeto | `ingest_signals.py`, `generate_tailored_cv.py` |
| Tests | espejo exacto de la ruta del módulo + prefijo `test_` | `app/domain/entities/signal.py` → `tests/unit/domain/entities/test_signal.py` |
| ADRs | `NNN-titulo-kebab.md` incremental en `docs/architecture/` | `007-nueva-decision.md` |

### 1.2 Clases

| Elemento | Convención | Ejemplo | Anti-ejemplo |
|---|---|---|---|
| Entidad | `PascalCase`, sustantivo de negocio | `FundingRound` | `FundingRoundModel`, `FundingRoundData` |
| Value object | `PascalCase`, `@dataclass(frozen=True)` | `Money`, `Embedding` | `MoneyVO` |
| Puerto | prefijo `I` + capacidad | `ICVGenerator` | `AbstractCVGenerator`, `CVGeneratorBase` |
| Adaptador | `<Tecnología><Puerto sin I>` | `ClaudeLLMService`, `RSSFundingScraper` | `LLMServiceImpl`, `MyScraper` |
| Caso de uso | `<VerboObjeto>UseCase` | `MatchOpportunitiesUseCase` | `MatchingService`, `MatchManager` |
| DTO de caso de uso | `<CasoDeUso>Input` / `<CasoDeUso>Output` | `MatchOpportunitiesInput` | `MatchRequest` (eso es de la API) |
| Schema de API | `<Acción>Request` / `<Acción>Response` | `GenerateCVRequest` | `GenerateCVInput` (eso es del caso de uso) |
| Excepción | sufijo `Error`, hereda de `VenturePulseError` | `ScrapingError` | `ScrapingException`, `ScrapingFailure` |
| Fake de test | `Fake<Puerto sin I>` | `FakeSignalRepository` | `MockSignalRepository` (mock es otra cosa, ver §4) |

### 1.3 Métodos y variables

- Métodos de puertos: verbos de dominio (`save`, `search`, `fetch`, `generate`), **nunca** jerga del proveedor (`upsert_points`, `create_message`).
- Caso de uso: un único método público `execute(input) -> output`. Helpers privados con `_`.
- Booleanos: prefijo `is_`/`has_`/`should_` (`is_fresh`, `has_salary`).
- Variables de entorno: `SCREAMING_SNAKE` con prefijo de área (`LLM_PROVIDER`, `QDRANT_URL`, `SCRAPER_RSS_FEEDS`).
- Idioma: **inglés en todo el código**; español solo en docs/ADRs.

---

## 2. Estructura de un caso de uso

Plantilla canónica — todo caso de uso nuevo se copia de aquí:

```python
# app/application/use_cases/match_opportunities.py
from dataclasses import dataclass

from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import ISignalRepository


@dataclass(frozen=True)
class MatchOpportunitiesInput:
    profile: DeveloperProfile
    min_signal_strength: float = 0.0
    limit: int = 10


@dataclass(frozen=True)
class MatchOpportunitiesOutput:
    matches: list[ScoredSignal]


class MatchOpportunitiesUseCase:
    def __init__(
        self,
        signal_repository: ISignalRepository,
        embedding_service: IEmbeddingService,
    ) -> None:
        self._signal_repository = signal_repository
        self._embedding_service = embedding_service

    async def execute(
        self, input: MatchOpportunitiesInput
    ) -> MatchOpportunitiesOutput:
        ...
```

**Reglas:**

1. **Dependencias por constructor**, tipadas como puertos (interfaces), nunca como adaptadores concretos. El cableado real ocurre solo en `infrastructure/config/container.py` (ADR-006).
2. **Un solo método público `execute()`**. Si necesitás un segundo método público, son dos casos de uso.
3. **Input/Output son dataclasses congeladas** propias del caso de uso. No recibe ni devuelve schemas de FastAPI, ni tipos de SDK, ni dicts sueltos. Entidades del dominio sí pueden viajar dentro del Output.
4. **`async def execute`** siempre — toda la I/O del sistema es async.
5. **Sin estado mutable**: el caso de uso no guarda nada entre ejecuciones; debe ser seguro instanciarlo una vez y reusarlo.
6. **No atrapa excepciones de infraestructura** para silenciarlas (ver §3): o las traduce a un resultado de negocio significativo, o las deja subir.
7. La lógica reutilizable entre casos de uso va a `application/services/` (ej. `scoring.py`), no a un "utils".

---

## 3. Manejo de errores

### 3.1 Jerarquía

Todas las excepciones propias heredan de una raíz, definida en `app/domain/exceptions.py`:

```python
class VenturePulseError(Exception):
    """Raíz de todas las excepciones del proyecto."""

# ── Dominio: violaciones de reglas de negocio ──
class DomainError(VenturePulseError): ...
class SignalValidationError(DomainError): ...      # señal sin monto/empresa
class CVHallucinationError(DomainError): ...       # CV con hechos no presentes en el perfil
class ProfileIncompleteError(DomainError): ...

# ── Infraestructura: fallas del mundo exterior ──
class InfrastructureError(VenturePulseError): ...
class ScrapingError(InfrastructureError): ...      # fuente caída, HTML cambió, 429
class EmbeddingError(InfrastructureError): ...
class LLMError(InfrastructureError): ...
class RepositoryError(InfrastructureError): ...
```

### 3.2 Reglas por capa

| Capa | Lanza | Atrapa |
|---|---|---|
| `domain/` | `DomainError` y subclases | nada |
| `infrastructure/` | `InfrastructureError` y subclases (**traduciendo** la del SDK) | excepciones del SDK, solo para traducirlas |
| `application/` | deja subir ambas; puede lanzar `DomainError` propia | `InfrastructureError` solo si puede degradar con sentido de negocio |
| `api/` | nada propio | todo: traduce a HTTP en un exception handler global |

**Traducción obligatoria en adaptadores.** Un adaptador nunca deja escapar la excepción cruda del SDK — eso filtraría el proveedor hacia adentro y rompería la intercambiabilidad free/paid (ADR-004). Siempre con `raise ... from e` para conservar la causa:

```python
# en QdrantSignalRepository
try:
    await self._client.upsert(...)
except UnexpectedResponse as e:
    raise RepositoryError(f"Failed to save signal {signal.id}") from e
```

**Degradación con sentido de negocio (la única captura permitida en application).** Ejemplo: en `IngestSignalsUseCase`, si un scraper falla, se loguea y se continúa con las demás fuentes — una fuente caída no aborta la ingesta. En cambio, `GenerateTailoredCVUseCase` deja subir `LLMError`: sin LLM no hay CV, no hay degradación posible.

**Mapeo a HTTP** (un solo handler global en `api/`, nunca try/except por endpoint):

| Excepción | HTTP |
|---|---|
| `DomainError` (validación) | 422 |
| `CVHallucinationError` (tras reintentos) | 502 con mensaje explicativo |
| `InfrastructureError` | 503 |
| no contemplada | 500 + log con stack trace |

**Prohibido:** `except Exception: pass`, atrapar para solo re-lanzar sin traducir, y usar excepciones para control de flujo normal (un search sin resultados devuelve lista vacía, no lanza).

### 3.3 Reintentos

- Los reintentos por fallas transitorias (429, timeout) viven en el **adaptador**, no en el caso de uso.
- Excepción: el loop anti-alucinación del CV (máx. 2 reintentos con feedback) es regla de negocio → vive en el caso de uso / `ICVGenerator`.

---

## 4. Tests

### 4.1 Estructura y qué se testea dónde

```
tests/
├── unit/            # domain + application. Sin red, sin Docker, sin sleep. <1s total deseable.
├── integration/     # adaptadores contra servicios reales (Qdrant en Docker, Ollama).
├── fixtures/        # Golden Dataset (Hit Rate @ K) y fakes compartidos.
└── conftest.py
```

| Qué | Tipo de test | Dependencias |
|---|---|---|
| Invariantes de entidades (`CV.validate_against`, `Signal.is_fresh`) | unit | ninguna |
| Casos de uso (orquestación, scoring, degradación por scraper caído) | unit | **fakes** de los puertos |
| Adaptadores (`QdrantSignalRepository`, `FastembedService`) | integration | stack FREE en Docker |
| Calidad del matching (Hit Rate @ K contra Golden Dataset) | integration | Qdrant + embeddings reales |
| Rutas de la API (status codes, serialización) | unit | casos de uso fakeados vía `Depends` override |

### 4.2 Fakes, no mocks

Para los puertos se escriben **fakes**: implementaciones en memoria con comportamiento real simplificado, en `tests/fixtures/fakes.py`. Se prefieren sobre `MagicMock` porque verifican comportamiento, no llamadas, y sobreviven a refactors de la firma interna.

```python
class FakeSignalRepository(ISignalRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[Signal, Embedding]] = []

    async def save(self, signal: Signal, embedding: Embedding) -> None:
        self.saved.append((signal, embedding))

    async def search(self, query, filters, limit=10) -> list[ScoredSignal]:
        return [...]  # determinista, ordenado por similitud coseno real

    async def exists(self, content_hash: str) -> bool:
        return any(s.content_hash == content_hash for s, _ in self.saved)
```

`unittest.mock` se reserva para verificar efectos colaterales puntuales (ej. "se logueó el scraper caído").

### 4.3 Reglas

1. **Nomenclatura:** `test_<comportamiento>_<condición>` en inglés: `test_ingest_skips_duplicate_signals`, `test_cv_generation_retries_on_hallucination`.
2. **Patrón AAA** (arrange/act/assert) con líneas en blanco entre bloques; sin comentarios `# arrange`.
3. **Un comportamiento por test.** Parametrizá variantes con `@pytest.mark.parametrize`, no con loops dentro del test.
4. Tests async con `pytest.mark.asyncio` (configurado global en `conftest.py`, no por test).
5. Los tests de integración llevan `@pytest.mark.integration` y se excluyen del run default: `pytest` corre solo unit; `pytest -m integration` requiere `docker compose up`.
6. **Ningún test unitario toca la red.** Si un test unit necesita un servicio externo, está mal clasificado o el código está mal capeado.
7. Los tests de integración **siempre corren contra el stack FREE** (fastembed + Ollama + Qdrant local). Las implementaciones PAID se cubren con contract tests mínimos que se ejecutan manualmente (`@pytest.mark.paid`), nunca en CI.
8. La frontera de capas se verifica en CI con `import-linter`: es un test más, no una convención de buena voluntad.

### 4.4 Definition of done

Un cambio está terminado cuando: (a) tests unit nuevos para el comportamiento agregado, (b) `pytest` verde, (c) `pytest -m integration` verde si tocaste un adaptador, (d) `import-linter` verde.

---

## 5. Checklist: agregar una nueva fuente de datos

> Escenario: querés ingerir señales desde una fuente nueva (ej. un job board, un agregador de noticias, una API).

**Antes de escribir código, decidí:**
- [ ] ¿Qué tipo de señal produce? (`FundingRound`, `JobOffer`, ¿o un tipo nuevo de `Signal`? Un tipo nuevo requiere ADR).
- [ ] ¿Es FREE o PAID? Si es PAID, ¿cuál es su contraparte FREE existente o nueva? (ADR-004: ningún puerto puede quedar solo con implementación paga).

**Pasos:**

1. [ ] **Adaptador** en `app/infrastructure/scrapers/<fuente>_scraper.py`, clase `<Fuente>Scraper(ISignalScraper)`. Implementa `source_name()` (slug estable, ej. `"remoteok-jobs"`) y `fetch(since) -> AsyncIterator[RawSignal]`.
2. [ ] **Sin tipos del proveedor hacia afuera**: `fetch` produce `RawSignal`; el parseo de HTML/JSON queda encapsulado en el adaptador.
3. [ ] **Errores traducidos**: toda falla (HTTP, parseo, rate limit) sale como `ScrapingError` con `from e`. Los reintentos/backoff van dentro del adaptador.
4. [ ] **Respetá la spec de ingestión** ([data-ingestion.md](docs/specs/data-ingestion.md)): httpx async con semáforo, user-agent rotation si aplica, limpieza de boilerplate.
5. [ ] **Config**: agregá las variables necesarias a `settings.py` y `.env.example` (prefijo `SCRAPER_`, ej. `SCRAPER_REMOTEOK_ENABLED=true`). Nunca hardcodees URLs ni API keys.
6. [ ] **Registro en el container**: en `container.py`, agregá el scraper a la lista que recibe `IngestSignalsUseCase`, condicionado a su flag de config. **No toques el caso de uso** — si necesitás modificar `IngestSignalsUseCase` para soportar la fuente, algo está mal abstraído: frená y revisá.
7. [ ] **Tests**:
   - unit: parseo de respuestas con HTML/JSON de ejemplo guardado en `tests/fixtures/<fuente>/` (sin red).
   - unit: la falla de la fuente produce `ScrapingError`, y el caso de uso continúa con las demás fuentes.
   - integration (`@pytest.mark.integration`): un fetch real acotado, si la fuente lo permite.
8. [ ] **Docs**: una línea en la spec de ingestión (fuente, tipo de señal, FREE/PAID, particularidades anti-bot).

**No hace falta tocar:** `domain/`, `application/`, `api/`. Si te ves tocándolos, releé el paso 6.

---

## 6. Checklist: agregar un nuevo generador de contenido

> Escenario: además del CV querés generar otro artefacto derivado del perfil + una señal (ej. carta de presentación, mensaje de outreach, respuestas a screening questions).

**Antes de escribir código, decidí:**
- [ ] ¿Es una variante de CV (otro formato/plantilla) o un artefacto nuevo? Una variante es solo otra implementación de `ICVGenerator`; un artefacto nuevo sigue todos los pasos de abajo.
- [ ] ¿Cuál es su invariante anti-alucinación? Todo contenido generado debe ser verificable contra el perfil. Si no podés definir el invariante, no está listo para implementarse.

**Pasos (artefacto nuevo, ej. `CoverLetter`):**

1. [ ] **Entidad de dominio** en `app/domain/entities/cover_letter.py`: dataclass pura, inmutable, con `profile_id`, `target_signal_id`, `generated_at`, y su método de validación (`validate_against(profile)`). Sin Pydantic, sin imports de otras capas.
2. [ ] **Puerto** en `app/domain/ports/cover_letter_generator.py`: `ICoverLetterGenerator(ABC)` con un método `generate(profile, target) -> CoverLetter`. Firma con tipos del dominio únicamente.
3. [ ] **Adaptador(es)** en `app/infrastructure/generation/`: `LLMCoverLetterGenerator(ICoverLetterGenerator)` que compone `ILLMService` (no un SDK directo — así hereda gratis el dual FREE/PAID de Ollama/Claude). Prompts como constantes/templates en el módulo del adaptador, no inline en métodos.
4. [ ] **Guardrail**: el adaptador valida el output con `entidad.validate_against(profile)` y reintenta con feedback (máx. 2) ante `*HallucinationError`, igual que el flujo de CV (ARCHITECTURE.md §5.3).
5. [ ] **Caso de uso** en `app/application/use_cases/generate_cover_letter.py`: `GenerateCoverLetterUseCase` siguiendo la plantilla de §2 (Input/Output congelados, `execute()` único, dependencias por constructor: `ISignalRepository` + `ICoverLetterGenerator`).
6. [ ] **Excepciones**: si necesitás una nueva (ej. `CoverLetterHallucinationError`), heredá de `DomainError` y agregala al mapeo HTTP de §3.2.
7. [ ] **Container + API**: cableá en `container.py`; ruta nueva en `api/routes/` con schemas `Request`/`Response` propios (no reuses los DTOs del caso de uso como schemas).
8. [ ] **Tests**:
   - unit de la entidad: `validate_against` detecta contenido inventado (caso positivo y negativo).
   - unit del caso de uso con `Fake<Artefacto>Generator`.
   - unit del adaptador con `FakeLLMService`: verifica el loop de reintento y que tras 2 fallos lanza.
   - integration: generación real contra Ollama (`@pytest.mark.integration`).
9. [ ] **Docs**: agregá el flujo al ARCHITECTURE.md (§5) si el artefacto es permanente, y un ADR si introdujo una decisión nueva (ej. plantillas de prompt versionadas).

**Señal de mal diseño:** si el nuevo generador necesita conocer Qdrant, FastAPI o el proveedor de LLM concreto, está violando DIP — debe componer puertos, nunca adaptadores.

---

## 7. Resumen para arrancar una sesión

1. Leé ARCHITECTURE.md + este archivo.
2. Ubicá en qué capa va tu cambio (tabla §1 de ARCHITECTURE.md). Duda razonable → es la capa más interna posible.
3. Si agregás fuente de datos → checklist §5. Si agregás generador → checklist §6. Si es otra cosa → plantilla §2 y reglas §3.
4. Tests según §4; `pytest` + `import-linter` verdes antes de commitear.
5. Toda decisión no obvia y permanente → ADR nuevo en `docs/architecture/`.
