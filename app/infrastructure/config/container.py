"""Composition root: build and wire every application component.

This is the only module where concrete adapter classes are named
(ADR-006). Everything it hands out is typed as a domain port, so
consumers depend on interfaces, never on implementations.
"""

from dataclasses import dataclass

from app.domain.ports import (
    ICVGenerator,
    IEmbeddingService,
    ILLMService,
    ISignalRepository,
    ISignalScraper,
)
from app.infrastructure.config.settings import Settings, get_settings
from app.infrastructure.embedding.fastembed_service import (
    FastembedEmbeddingService,
)
from app.infrastructure.llm.llm_cv_generator import LLMCVGenerator
from app.infrastructure.llm.ollama_llm_service import OllamaLLMService
from app.infrastructure.persistence.qdrant_signal_repository import (
    QdrantSignalRepository,
)
from app.infrastructure.scraping.rss_signal_scraper import RSSSignalScraper


@dataclass(frozen=True)
class AppContainer:
    """Assembled application components.

    This is the composition root — the only place where concrete
    adapter classes are named. All consumers receive interfaces.
    """

    settings: Settings
    embedder: IEmbeddingService
    repository: ISignalRepository
    llm_service: ILLMService
    scraper: ISignalScraper
    cv_generator: ICVGenerator


async def build_embedder(settings: Settings | None = None) -> IEmbeddingService:
    """Build the embedding service adapter.

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
    """
    if settings is None:
        settings = get_settings()

    return FastembedEmbeddingService(settings.embedding)


async def build_repository(
    settings: Settings | None = None,
    embedder: IEmbeddingService | None = None,
) -> ISignalRepository:
    """Build the signal repository adapter.

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
        embedder: Optional pre-built embedder to reuse. If None, builds one.
    """
    if settings is None:
        settings = get_settings()
    if embedder is None:
        embedder = await build_embedder(settings)

    return await QdrantSignalRepository.create(settings.qdrant, embedder)


async def build_llm_service(settings: Settings | None = None) -> ILLMService:
    """Build the LLM service adapter.

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
    """
    if settings is None:
        settings = get_settings()

    return await OllamaLLMService.create(settings.llm)


async def build_scraper(settings: Settings | None = None) -> ISignalScraper:
    """Build the signal scraper adapter.

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
    """
    if settings is None:
        settings = get_settings()

    return RSSSignalScraper(settings.scraper)


async def build_cv_generator(
    settings: Settings | None = None,
    llm_service: ILLMService | None = None,
) -> ICVGenerator:
    """Build the CV generator adapter.

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
        llm_service: Optional pre-built LLM service to reuse. If None,
            builds one.
    """
    if settings is None:
        settings = get_settings()
    if llm_service is None:
        llm_service = await build_llm_service(settings)

    return LLMCVGenerator(llm_service)


async def build_container(settings: Settings | None = None) -> AppContainer:
    """Build and wire all application components.

    This is an async factory because several adapters require I/O at
    construction time (Qdrant collection check, Ollama model
    availability check).

    Args:
        settings: Optional Settings override (useful in tests). If None,
            calls get_settings().
    """
    if settings is None:
        settings = get_settings()

    embedder = await build_embedder(settings)
    repository = await build_repository(settings, embedder=embedder)
    llm_service = await build_llm_service(settings)
    scraper = await build_scraper(settings)
    cv_generator = await build_cv_generator(settings, llm_service=llm_service)

    return AppContainer(
        settings=settings,
        embedder=embedder,
        repository=repository,
        llm_service=llm_service,
        scraper=scraper,
        cv_generator=cv_generator,
    )


if __name__ == "__main__":
    import asyncio

    async def smoke():
        container = await build_container()
        print("OK: container built")
        print(f"  embedder dimensions: {container.embedder.dimensions}")
        print(f"  collection: {container.repository.collection_name}")
        print(f"  llm model: {container.settings.llm.ollama_model}")

    asyncio.run(smoke())
