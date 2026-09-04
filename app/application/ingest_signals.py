"""Use case: ingest funding signals from a scraper into the repository.

Imports only from app.domain.* — no infrastructure. The concrete
adapters are wired in the composition root and injected as ports.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from app.domain.entities.signal import FundingRound, JobOffer, RawSignal, Signal
from app.domain.exceptions import EmbeddingError, LLMError, RepositoryError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.llm_service import ILLMService
from app.domain.ports.signal_repository import ISignalRepository
from app.domain.ports.signal_scraper import ISignalScraper, SignalKind
from app.domain.value_objects.enums import FundingSeries, Seniority
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
                signal = await self._build_signal(scraper.signal_type, raw)
                if signal is None:
                    logger.info(
                        "skipped (missing required fields) from %s: %s",
                        raw.source,
                        raw.content[:80],
                    )
                    skipped_no_entities += 1
                    continue

                if await self._repo.exists(signal.content_hash):
                    skipped_duplicate += 1
                    continue

                embedding = await self._embedder.embed(signal.embedding_text)
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

    async def _build_signal(
        self, signal_type: SignalKind, raw: RawSignal
    ) -> Signal | None:
        """Extract entities and build the right Signal for this source.

        Dispatches on signal_type rather than inspecting `raw` itself:
        one scraper produces exactly one signal kind (ADR / signal_scraper
        port), so the raw text's content never needs sniffing.
        """
        if signal_type == "funding_round":
            entities = await self._llm.extract_funding_entities(raw.content)
            if not raw.source or not entities.amount:
                return None

            series = entities.series
            if series is None:
                logger.debug("series unknown, defaulting to SEED")
                series = FundingSeries.SEED

            return FundingRound.from_extraction(
                id=new_signal_id(),
                source=raw.source,
                raw_content=raw.content,
                summary=raw.content[:_MAX_SUMMARY],
                detected_at=raw.fetched_at,
                entities=entities,
                series=series,
            )

        if signal_type == "job_offer":
            entities = await self._llm.extract_job_entities(raw.content)
            if not raw.source or not entities.title:
                return None

            seniority = entities.seniority
            if seniority is None:
                logger.info("seniority unknown, defaulting to UNKNOWN")
                seniority = Seniority.UNKNOWN

            return JobOffer.from_extraction(
                id=new_signal_id(),
                source=raw.source,
                raw_content=raw.content,
                summary=raw.content[:_MAX_SUMMARY],
                detected_at=raw.fetched_at,
                entities=entities,
                seniority=seniority,
                url=raw.url,
            )

        raise AssertionError(f"unhandled signal_type: {signal_type!r}")
