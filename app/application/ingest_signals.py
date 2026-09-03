"""Use case: ingest funding signals from a scraper into the repository.

Imports only from app.domain.* — no infrastructure. The concrete
adapters are wired in the composition root and injected as ports.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from app.domain.entities.signal import FundingRound
from app.domain.exceptions import EmbeddingError, LLMError, RepositoryError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.llm_service import ILLMService
from app.domain.ports.signal_repository import ISignalRepository
from app.domain.ports.signal_scraper import ISignalScraper
from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.identifiers import new_signal_id

logger = logging.getLogger(__name__)

_MAX_SUMMARY = 500


@dataclass(frozen=True)
class IngestResult:
    scraped: int = 0
    ingested: int = 0
    skipped_duplicate: int = 0
    skipped_no_entities: int = 0
    errors: int = 0


class IngestSignalsUseCase:
    def __init__(
        self,
        scrapers: list[ISignalScraper],
        llm_service: ILLMService,
        embedder: IEmbeddingService,
        repository: ISignalRepository,
    ) -> None:
        self._scrapers = scrapers
        self._llm = llm_service
        self._embedder = embedder
        self._repo = repository

    async def execute(self, since: datetime) -> IngestResult:
        partial_results = await asyncio.gather(
            *(self._process_scraper(scraper, since) for scraper in self._scrapers)
        )

        return IngestResult(
            scraped=sum(r.scraped for r in partial_results),
            ingested=sum(r.ingested for r in partial_results),
            skipped_duplicate=sum(r.skipped_duplicate for r in partial_results),
            skipped_no_entities=sum(r.skipped_no_entities for r in partial_results),
            errors=sum(r.errors for r in partial_results),
        )

    async def _process_scraper(
        self, scraper: ISignalScraper, since: datetime
    ) -> IngestResult:
        scraped = 0
        ingested = 0
        skipped_duplicate = 0
        skipped_no_entities = 0
        errors = 0

        start = time.perf_counter()
        async for raw in scraper.fetch(since):
            scraped += 1
            try:
                entities = await self._llm.extract_funding_entities(raw.content)

                # Quality filter: drop signals lacking a source or amount.
                if not raw.source or not entities.amount:
                    logger.info(
                        "skipped (no source/amount) from %s: %s",
                        raw.source,
                        raw.content[:80],
                    )
                    skipped_no_entities += 1
                    continue

                # Default an unknown series rather than dropping the
                # signal: keeps more data in the system (and a valid
                # FundingSeries avoids a None reaching the persistence
                # codec).
                series = entities.series
                if series is None:
                    logger.debug("series unknown, defaulting to SEED")
                    series = FundingSeries.SEED

                signal = FundingRound.from_extraction(
                    id=new_signal_id(),
                    source=raw.source,
                    raw_content=raw.content,
                    summary=raw.content[:_MAX_SUMMARY],
                    detected_at=raw.fetched_at,
                    entities=entities,
                    series=series,
                )

                if await self._repo.exists(signal.content_hash):
                    skipped_duplicate += 1
                    continue

                embedding = await self._embedder.embed(
                    f"{signal.company_name} {signal.summary}"
                )
                await self._repo.save(signal, embedding)
                ingested += 1

            except (LLMError, EmbeddingError, RepositoryError) as e:
                logger.warning("signal skipped: %s", e)
                errors += 1
                continue
        elapsed = time.perf_counter() - start
        logger.info(
            "source %s finished in %.2fs (scraped=%d, ingested=%d)",
            scraper.source_name(),
            elapsed,
            scraped,
            ingested,
        )

        return IngestResult(
            scraped=scraped,
            ingested=ingested,
            skipped_duplicate=skipped_duplicate,
            skipped_no_entities=skipped_no_entities,
            errors=errors,
        )
