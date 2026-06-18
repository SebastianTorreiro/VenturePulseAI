"""Use case: ingest funding signals from a scraper into the repository.

Imports only from app.domain.* — no infrastructure. The concrete
adapters are wired in the composition root and injected as ports.
"""

import logging
import re
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
# Signal strength normalizes the funding amount: $1B caps the score at 1.0.
_STRENGTH_DENOMINATOR = 1_000_000_000
# Company name = text before the first funding verb (naive MVP heuristic).
_COMPANY_RE = re.compile(
    r"(.+?)\s+(?:raised|raises|announced|announces|secured|secures|"
    r"closed|closes)\b",
    re.IGNORECASE,
)


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
        scraper: ISignalScraper,
        llm_service: ILLMService,
        embedder: IEmbeddingService,
        repository: ISignalRepository,
    ) -> None:
        self._scraper = scraper
        self._llm = llm_service
        self._embedder = embedder
        self._repo = repository

    async def execute(self, since: datetime) -> IngestResult:
        scraped = 0
        ingested = 0
        skipped_duplicate = 0
        skipped_no_entities = 0
        errors = 0

        async for raw in self._scraper.fetch(since):
            scraped += 1
            try:
                entities = await self._llm.extract_funding_entities(raw.content)

                # Quality filter: drop signals lacking a source or amount.
                if not raw.source or not entities.amount:
                    skipped_no_entities += 1
                    continue

                # Default an unknown series rather than dropping the signal:
                # keeps more data in the system (and a valid FundingSeries
                # avoids a None reaching the persistence codec).
                series = entities.series
                if series is None:
                    logger.debug("series unknown, defaulting to SEED")
                    series = FundingSeries.SEED

                signal = FundingRound(
                    id=new_signal_id(),
                    source=raw.source,
                    company_name=_extract_company_name(raw.content),
                    summary=raw.content[:_MAX_SUMMARY],
                    detected_at=raw.fetched_at,
                    signal_strength=min(
                        1.0,
                        float(entities.amount.amount) / _STRENGTH_DENOMINATOR,
                    ),
                    amount=entities.amount,
                    series=series,
                    investors=list(entities.investors),
                    investment_thesis=entities.investment_thesis or "",
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

        return IngestResult(
            scraped=scraped,
            ingested=ingested,
            skipped_duplicate=skipped_duplicate,
            skipped_no_entities=skipped_no_entities,
            errors=errors,
        )


def _extract_company_name(text: str) -> str:
    """Best-effort company name: text before the first funding verb.

    MVP heuristic, no LLM. Returns 'Unknown' when nothing matches.
    """
    match = _COMPANY_RE.search(text)
    if match:
        name = match.group(1).strip()
        if name:
            return name
    return "Unknown"
