"""Unit tests for IngestSignalsUseCase — fakes for every port, no I/O."""

import asyncio
from datetime import datetime, timezone

from app.application.ingest_signals import IngestSignalsUseCase
from app.domain.entities.signal import FundingRound, JobOffer, RawSignal, Signal
from app.domain.exceptions import LLMError, RepositoryError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.llm_service import FundingEntities, ILLMService, JobEntities
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.ports.signal_scraper import ISignalScraper, SignalKind
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.enums import Seniority
from app.domain.value_objects.identifiers import SignalId
from tests.fixtures.factories import make_funding_entities, make_job_entities

# --- fakes ------------------------------------------------------------


class FakeScraper(ISignalScraper):
    def __init__(
        self, raws: list[RawSignal], signal_type: SignalKind = "funding_round"
    ) -> None:
        self._raws = list(raws)
        self._signal_type = signal_type

    @property
    def signal_type(self) -> SignalKind:
        return self._signal_type

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
        return self._next_result()

    async def extract_job_entities(self, raw_text: str) -> JobEntities:
        return self._next_result()

    def _next_result(self):
        result = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        if isinstance(result, Exception):
            raise result
        return result

    async def complete(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class FakeEmbedder(IEmbeddingService):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> Embedding:
        self.calls.append(text)
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


def _use_case(scraper, llm, embedder=None, repo=None) -> IngestSignalsUseCase:
    return IngestSignalsUseCase(
        [scraper], llm, embedder or FakeEmbedder(), repo or FakeRepo()
    )


_SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- tests ------------------------------------------------------------


def test_execute_ingests_valid_signals():
    repo = FakeRepo()
    scraper = FakeScraper(
        [_raw(), _raw(content="Beta Inc raised $5M in Series A.")]
    )
    llm = FakeLLM([make_funding_entities(), make_funding_entities(amount="5000000")])

    result = asyncio.run(
        _use_case(scraper, llm, repo=repo).execute(_SINCE)
    )

    assert result.ingested == 2
    assert result.scraped == 2
    assert len(repo.saved) == 2
    assert all(isinstance(s, FundingRound) for s, _ in repo.saved)


def test_execute_skips_signals_without_amount():
    scraper = FakeScraper([_raw()])
    llm = FakeLLM([make_funding_entities(amount=None)])

    result = asyncio.run(_use_case(scraper, llm).execute(_SINCE))

    assert result.skipped_no_entities == 1
    assert result.ingested == 0


def test_execute_skips_duplicate_signals():
    repo = FakeRepo()
    # Two signals with identical content produce the same content_hash.
    scraper = FakeScraper([_raw(), _raw()])
    llm = FakeLLM([make_funding_entities(), make_funding_entities()])

    result = asyncio.run(
        _use_case(scraper, llm, repo=repo).execute(_SINCE)
    )

    assert result.ingested == 1
    assert result.skipped_duplicate == 1
    assert len(repo.saved) == 1


def test_execute_counts_errors_without_raising():
    scraper = FakeScraper([_raw(), _raw(content="Beta Inc raised $5M.")])
    # First extraction raises; the loop must continue, not abort.
    llm = FakeLLM([LLMError("boom"), make_funding_entities(amount="5000000")])

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
            make_funding_entities(amount="10000000"),
            make_funding_entities(amount="10000000"),
            make_funding_entities(amount=None),
            LLMError("boom"),
            make_funding_entities(amount="5000000"),
        ]
    )

    result = asyncio.run(_use_case(FakeScraper(raws), llm).execute(_SINCE))

    assert result.scraped == 5
    assert result.ingested == 2
    assert result.skipped_duplicate == 1
    assert result.skipped_no_entities == 1
    assert result.errors == 1


def test_execute_aggregates_results_across_multiple_scrapers():
    repo = FakeRepo()
    scraper_a = FakeScraper([_raw(content="Acme Corp raised $10M in Series A.")])
    scraper_b = FakeScraper(
        [
            _raw(content="Beta Inc raised $5M in Series A."),
            _raw(content="Gamma raised more funds this year."),  # no amount
        ]
    )
    llm = FakeLLM(
        [
            make_funding_entities(amount="10000000"),
            make_funding_entities(amount="5000000"),
            make_funding_entities(amount=None),
        ]
    )

    use_case = IngestSignalsUseCase([scraper_a, scraper_b], llm, FakeEmbedder(), repo)
    result = asyncio.run(use_case.execute(_SINCE))

    assert result.scraped == 3
    assert result.ingested == 2
    assert result.skipped_no_entities == 1
    assert len(repo.saved) == 2


def test_execute_ingests_job_offer_via_dispatch():
    repo = FakeRepo()
    scraper = FakeScraper(
        [_raw(content="Acme Corp is hiring a Senior Backend Engineer.")],
        signal_type="job_offer",
    )
    llm = FakeLLM([make_job_entities()])

    result = asyncio.run(_use_case(scraper, llm, repo=repo).execute(_SINCE))

    assert result.ingested == 1
    assert result.scraped == 1
    assert len(repo.saved) == 1
    signal, _ = repo.saved[0]
    assert isinstance(signal, JobOffer)
    assert signal.title == "Senior Backend Engineer"
    assert signal.required_skills == ["python", "fastapi"]


def test_execute_embeds_job_offer_using_its_embedding_text():
    repo = FakeRepo()
    embedder = FakeEmbedder()
    scraper = FakeScraper(
        [_raw(content="Acme Corp is hiring a Senior Backend Engineer.")],
        signal_type="job_offer",
    )
    llm = FakeLLM([make_job_entities()])

    asyncio.run(_use_case(scraper, llm, embedder=embedder, repo=repo).execute(_SINCE))

    signal, _ = repo.saved[0]
    assert embedder.calls == [signal.embedding_text]
    # title/skills must actually be in the embedded text, not just the
    # generic "{company_name} {summary}" the use case used to send.
    assert "Senior Backend Engineer" in embedder.calls[0]
    assert "python" in embedder.calls[0]
    assert "fastapi" in embedder.calls[0]


def test_execute_skips_job_offer_without_title():
    scraper = FakeScraper(
        [_raw(content="Some generic announcement, not a job posting.")],
        signal_type="job_offer",
    )
    llm = FakeLLM([make_job_entities(title=None)])

    result = asyncio.run(_use_case(scraper, llm).execute(_SINCE))

    assert result.skipped_no_entities == 1
    assert result.ingested == 0


def test_execute_defaults_unresolved_seniority_to_unknown():
    repo = FakeRepo()
    scraper = FakeScraper(
        [_raw(content="Acme Corp is hiring, level not specified.")],
        signal_type="job_offer",
    )
    llm = FakeLLM([make_job_entities(seniority=None)])

    asyncio.run(_use_case(scraper, llm, repo=repo).execute(_SINCE))

    signal, _ = repo.saved[0]
    assert signal.seniority == Seniority.UNKNOWN
