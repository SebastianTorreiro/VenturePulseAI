"""Unit tests for SearchSignalsUseCase — fakes for every port, no I/O."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from app.application.search_signals import SearchSignalsUseCase
from app.domain.entities.signal import FundingRound, Signal
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.identifiers import SignalId, new_signal_id
from app.domain.value_objects.money import Money

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


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


class FakeSearchRepo(ISignalRepository):
    """Returns canned results, honoring min_amount_usd, and records filters."""

    def __init__(self, results: list[ScoredSignal]) -> None:
        self._results = list(results)
        self.last_filters: SignalFilter | None = None
        self.last_limit: int | None = None

    async def save(self, signal, embedding) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(self, query, filters, limit=10) -> list[ScoredSignal]:
        self.last_filters = filters
        self.last_limit = limit
        results = self._results
        if filters.min_amount_usd is not None:
            results = [
                r
                for r in results
                if isinstance(r.signal, FundingRound)
                and r.signal.amount.amount >= filters.min_amount_usd
            ]
        return list(results)

    async def exists(self, content_hash) -> bool:  # pragma: no cover
        return False

    async def get_by_id(self, signal_id: SignalId) -> Signal:  # pragma: no cover
        raise NotImplementedError


def _scored(company: str, amount: str, score: float = 0.9) -> ScoredSignal:
    signal = FundingRound(
        id=new_signal_id(),
        source="fake-rss",
        company_name=company,
        summary=f"{company} raised funds",
        detected_at=_NOW,
        signal_strength=0.5,
        amount=Money(amount=Decimal(amount), currency="USD"),
        series=FundingSeries.A,
    )
    return ScoredSignal(signal=signal, semantic_score=score)


def test_search_returns_results_for_valid_query():
    repo = FakeSearchRepo([_scored("Acme", "10000000"), _scored("Beta", "5000000")])
    use_case = SearchSignalsUseCase(FakeEmbedder(), repo)

    result = asyncio.run(use_case.execute("fintech startups"))

    assert result.total == 2
    assert result.query == "fintech startups"
    assert isinstance(result.signals, tuple)
    assert len(result.signals) == 2


def test_search_returns_empty_for_no_matches():
    repo = FakeSearchRepo([])
    use_case = SearchSignalsUseCase(FakeEmbedder(), repo)

    result = asyncio.run(use_case.execute("no matches here"))

    assert result.total == 0
    assert result.signals == ()


def test_search_applies_amount_filter():
    repo = FakeSearchRepo(
        [_scored("Big", "50000000"), _scored("Small", "1000000")]
    )
    use_case = SearchSignalsUseCase(FakeEmbedder(), repo)

    result = asyncio.run(
        use_case.execute(
            "startups",
            min_amount_usd=Decimal("10000000"),
            signal_type="funding_round",
        )
    )

    assert result.total == 1
    assert result.signals[0].signal.company_name == "Big"
    # the use case mapped its params onto the SignalFilter
    assert repo.last_filters.min_amount_usd == Decimal("10000000")
    assert repo.last_filters.signal_type == "funding_round"
