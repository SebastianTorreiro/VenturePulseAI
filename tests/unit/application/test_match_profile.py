"""Unit tests for MatchProfileUseCase — fakes for every port, no I/O."""

import asyncio
from datetime import datetime, timezone

from app.application.match_profile import MatchProfileUseCase
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.entities.signal import Signal
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.identifiers import SignalId, new_profile_id


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
    """Records the filters and limit it was searched with."""

    def __init__(self) -> None:
        self.last_filters: SignalFilter | None = None
        self.last_limit: int | None = None

    async def save(self, signal, embedding) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(self, query, filters, limit=10) -> list[ScoredSignal]:
        self.last_filters = filters
        self.last_limit = limit
        return []

    async def exists(self, content_hash) -> bool:  # pragma: no cover
        return False

    async def get_by_id(self, signal_id: SignalId) -> Signal:  # pragma: no cover
        raise NotImplementedError

    async def find_funding_rounds_since(self, since):  # pragma: no cover
        raise NotImplementedError


def _profile() -> DeveloperProfile:
    return DeveloperProfile(
        id=new_profile_id(),
        full_name="Jane Developer",
        headline="Backend engineer",
        contact="jane@example.com",
    )


def test_match_passes_since_to_the_repository_as_detected_after():
    repo = FakeRepo()
    use_case = MatchProfileUseCase(FakeEmbedder(), repo)
    since = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    asyncio.run(use_case.execute(_profile(), since=since))

    assert repo.last_filters.detected_after == since


def test_match_defaults_to_no_date_filter():
    repo = FakeRepo()
    use_case = MatchProfileUseCase(FakeEmbedder(), repo)

    asyncio.run(use_case.execute(_profile()))

    assert repo.last_filters.detected_after is None


def test_match_still_restricts_to_job_offers_and_forwards_limit():
    repo = FakeRepo()
    use_case = MatchProfileUseCase(FakeEmbedder(), repo)

    asyncio.run(
        use_case.execute(
            _profile(), limit=5, since=datetime(2026, 10, 1, tzinfo=timezone.utc)
        )
    )

    assert repo.last_filters.signal_type == "job_offer"
    assert repo.last_limit == 5
