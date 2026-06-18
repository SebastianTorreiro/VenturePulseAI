"""Unit tests for IngestSignalsUseCase — fakes for every port, no I/O."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from app.application.ingest_signals import IngestSignalsUseCase
from app.domain.entities.signal import FundingRound, RawSignal, Signal
from app.domain.exceptions import LLMError, RepositoryError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.llm_service import FundingEntities, ILLMService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.ports.signal_scraper import ISignalScraper
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.identifiers import SignalId
from app.domain.value_objects.money import Money


# --- fakes ------------------------------------------------------------


class FakeScraper(ISignalScraper):
    def __init__(self, raws: list[RawSignal]) -> None:
        self._raws = list(raws)

    def source_name(self) -> str:
        return "fake-rss"

    async def fetch(self, since):
        for raw in self._raws:
            yield raw


class FakeLLM(ILLMService):
    """Returns canned results in order; an Exception entry is raised."""

    def __init__(self, results: list) -> None:
        self._results = list(results)
        self._i = 0

    async def extract_funding_entities(self, raw_text: str) -> FundingEntities:
        result = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        if isinstance(result, Exception):
            raise result
        return result

    async def complete(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class FakeEmbedder(IEmbeddingService):
    async def embed(self, text: str) -> Embedding:
        return Embedding(vector=(0.1, 0.2, 0.3), model_id="fake")

    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return [await self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_id(self) -> str:
        return "fake"


class FakeRepo(ISignalRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[Signal, Embedding]] = []
        self._hashes: set[str] = set()

    async def save(self, signal: Signal, embedding: Embedding) -> None:
        self.saved.append((signal, embedding))
        self._hashes.add(signal.content_hash)

    async def search(
        self, query: Embedding, filters: SignalFilter, limit: int = 10
    ) -> list[ScoredSignal]:
        return []

    async def exists(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    async def get_by_id(self, signal_id: SignalId) -> Signal:
        for signal, _ in self.saved:
            if signal.id == signal_id:
                return signal
        raise RepositoryError(f"Signal {signal_id} not found")


# --- builders ---------------------------------------------------------


def _raw(content: str = "Acme Corp raised $10M in Series A.") -> RawSignal:
    return RawSignal(
        source="fake-rss",
        url="https://example.com/a",
        content=content,
        fetched_at=datetime.now(timezone.utc),
    )


def _entities(
    amount: str | None = "10000000",
    series: FundingSeries | None = FundingSeries.A,
    investors: tuple[str, ...] = ("Sequoia Capital",),
    thesis: str | None = "fintech payments",
) -> FundingEntities:
    return FundingEntities(
        amount=(
            Money(amount=Decimal(amount), currency="USD")
            if amount is not None
            else None
        ),
        series=series,
        investors=investors,
        investment_thesis=thesis,
    )


def _use_case(scraper, llm, embedder=None, repo=None) -> IngestSignalsUseCase:
    return IngestSignalsUseCase(
        scraper, llm, embedder or FakeEmbedder(), repo or FakeRepo()
    )


_SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- tests ------------------------------------------------------------


def test_execute_ingests_valid_signals():
    repo = FakeRepo()
    scraper = FakeScraper(
        [_raw(), _raw(content="Beta Inc raised $5M in Series A.")]
    )
    llm = FakeLLM([_entities(), _entities(amount="5000000")])

    result = asyncio.run(
        _use_case(scraper, llm, repo=repo).execute(_SINCE)
    )

    assert result.ingested == 2
    assert result.scraped == 2
    assert len(repo.saved) == 2
    assert all(isinstance(s, FundingRound) for s, _ in repo.saved)


def test_execute_skips_signals_without_amount():
    scraper = FakeScraper([_raw()])
    llm = FakeLLM([_entities(amount=None)])

    result = asyncio.run(_use_case(scraper, llm).execute(_SINCE))

    assert result.skipped_no_entities == 1
    assert result.ingested == 0


def test_execute_skips_duplicate_signals():
    repo = FakeRepo()
    # Two signals with identical content produce the same content_hash.
    scraper = FakeScraper([_raw(), _raw()])
    llm = FakeLLM([_entities(), _entities()])

    result = asyncio.run(
        _use_case(scraper, llm, repo=repo).execute(_SINCE)
    )

    assert result.ingested == 1
    assert result.skipped_duplicate == 1
    assert len(repo.saved) == 1


def test_execute_counts_errors_without_raising():
    scraper = FakeScraper([_raw(), _raw(content="Beta Inc raised $5M.")])
    # First extraction raises; the loop must continue, not abort.
    llm = FakeLLM([LLMError("boom"), _entities(amount="5000000")])

    result = asyncio.run(_use_case(scraper, llm).execute(_SINCE))

    assert result.errors == 1
    assert result.ingested == 1
    assert result.scraped == 2


def test_execute_returns_correct_counts():
    raws = [
        _raw(content="Acme Corp raised $10M in Series A."),  # ingested
        _raw(content="Acme Corp raised $10M in Series A."),  # duplicate
        _raw(content="Gamma raised more funds this year."),  # no amount
        _raw(content="Delta Inc raised $7M in Series B."),   # error
        _raw(content="Beta Inc raised $5M in Series A."),    # ingested
    ]
    llm = FakeLLM(
        [
            _entities(amount="10000000"),
            _entities(amount="10000000"),
            _entities(amount=None),
            LLMError("boom"),
            _entities(amount="5000000"),
        ]
    )

    result = asyncio.run(_use_case(FakeScraper(raws), llm).execute(_SINCE))

    assert result.scraped == 5
    assert result.ingested == 2
    assert result.skipped_duplicate == 1
    assert result.skipped_no_entities == 1
    assert result.errors == 1
