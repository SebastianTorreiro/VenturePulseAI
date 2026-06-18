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

    embedder = FastembedEmbeddingService(settings.embedding)

    repository = await QdrantSignalRepository.create(settings.qdrant, embedder)

    llm_service = await OllamaLLMService.create(settings.llm)

    scraper = RSSSignalScraper(settings.scraper)

    cv_generator = LLMCVGenerator(llm_service)

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
