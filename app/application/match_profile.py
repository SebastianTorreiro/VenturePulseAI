"""Use case: match a developer profile against open job offers.

Imports only from app.domain.* — no infrastructure.
"""

from app.application.search_signals import SearchResult
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import ISignalRepository, SignalFilter


class MatchProfileUseCase:
    def __init__(
        self, embedder: IEmbeddingService, repository: ISignalRepository
    ) -> None:
        self._embedder = embedder
        self._repo = repository

    async def execute(
        self, profile: DeveloperProfile, limit: int = 10
    ) -> SearchResult:
        embedding = await self._embedder.embed(profile.embedding_text)
        filters = SignalFilter(signal_type="job_offer")
        results = await self._repo.search(embedding, filters, limit)
        return SearchResult(
            signals=tuple(results),
            query=profile.headline,
            total=len(results),
        )
