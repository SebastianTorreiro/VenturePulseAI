"""Integration tests for QdrantSignalRepository (real Qdrant, no mocks).

Requires a Qdrant instance reachable at QDRANT__URL (default
http://localhost:6333; `docker compose up qdrant`). Marked `integration`
and excluded from the default `pytest` run; run with `pytest -m integration`.

Each test uses a unique collection prefix and an async context manager
that deletes the collection on teardown, so tests are independent. The
whole scenario (build repo + operations + teardown) runs inside a single
asyncio.run() so the AsyncQdrantClient lives on one event loop — no
pytest-asyncio plugin needed.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.exceptions import RepositoryError
from app.domain.entities.signal import FundingRound, JobOffer
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import SignalFilter
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import new_signal_id
from app.domain.value_objects.money import Money
from app.infrastructure.config.settings import EmbeddingSettings, QdrantSettings
from app.infrastructure.embedding.fastembed_service import (
    FastembedEmbeddingService,
)
from app.infrastructure.persistence.qdrant_signal_repository import (
    QdrantSignalRepository,
)

pytestmark = pytest.mark.integration


# --- builders ---------------------------------------------------------


def _make_funding_round(
    *,
    company: str = "Acme AI",
    summary: str = "Acme AI raised a $20M Series B for its ML platform.",
    amount: str = "20000000",
    series: FundingSeries = FundingSeries.B,
) -> FundingRound:
    return FundingRound(
        id=new_signal_id(),
        source="techcrunch-rss",
        company_name=company,
        summary=summary,
        detected_at=datetime.now(timezone.utc),
        signal_strength=0.8,
        amount=Money(amount=Decimal(amount), currency="USD"),
        series=series,
    )


def _make_job_offer(
    *,
    company: str = "Acme AI",
    summary: str = "Senior ML engineer role at Acme AI platform team.",
    title: str = "Senior ML Engineer",
) -> JobOffer:
    return JobOffer(
        id=new_signal_id(),
        source="remoteok-jobs",
        company_name=company,
        summary=summary,
        detected_at=datetime.now(timezone.utc),
        signal_strength=0.6,
        title=title,
        required_skills=["python", "qdrant"],
        seniority=Seniority.SENIOR,
        url="https://example.com/job/1",
    )


# --- test stub: an embedding service reporting a fixed dimension ------
# Stubbing the *embedding* service is allowed; the rule forbids faking the
# repository under test. Used only for the dimension-mismatch scenario.


class _FixedDimEmbeddingService(IEmbeddingService):
    def __init__(self, model_id: str, dimensions: int) -> None:
        self._model_id = model_id
        self._dimensions = dimensions

    async def embed(self, text):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def embed_batch(self, texts):  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return self._model_id


# --- fixtures and helpers ---------------------------------------------


@pytest.fixture(scope="module")
def embedding_service() -> FastembedEmbeddingService:
    return FastembedEmbeddingService(EmbeddingSettings())


@pytest.fixture
def qdrant_settings() -> QdrantSettings:
    return QdrantSettings(collection_prefix=f"test_{uuid4().hex[:8]}")


@asynccontextmanager
async def _repository(settings, embedding_service):
    repo = await QdrantSignalRepository.create(settings, embedding_service)
    try:
        yield repo
    finally:
        await repo._client.delete_collection(repo.collection_name)
        await repo._client.close()


# --- tests ------------------------------------------------------------


def test_repository_creates_collection_on_first_instantiation(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            assert await repo._client.collection_exists(repo.collection_name)

    asyncio.run(scenario())


def test_repository_rejects_dimension_mismatch_against_existing_collection(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            mismatched = _FixedDimEmbeddingService(
                model_id=embedding_service.model_id, dimensions=999
            )
            with pytest.raises(RepositoryError, match="dimension"):
                await QdrantSignalRepository.create(qdrant_settings, mismatched)

    asyncio.run(scenario())


def test_create_closes_client_on_ensure_collection_failure(
    qdrant_settings, embedding_service
):
    async def scenario():
        # Create the collection with the real (384-dim) service first, so a
        # second create() with a mismatched dimension fails in
        # _ensure_collection and must close its own client.
        async with _repository(qdrant_settings, embedding_service) as repo:
            mismatched = _FixedDimEmbeddingService(
                model_id=embedding_service.model_id, dimensions=999
            )
            closed: list[AsyncQdrantClient] = []
            real_close = AsyncQdrantClient.close

            async def spy_close(self, *args, **kwargs):
                closed.append(self)
                return await real_close(self, *args, **kwargs)

            with patch.object(AsyncQdrantClient, "close", spy_close):
                with pytest.raises(RepositoryError, match="dimension"):
                    await QdrantSignalRepository.create(
                        qdrant_settings, mismatched
                    )

            # Exactly the failed repository's client was closed (the outer
            # repo's teardown close happens later, outside the patch).
            assert len(closed) == 1

    asyncio.run(scenario())


def test_save_persists_funding_round_with_embedding(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            signal = _make_funding_round()
            embedding = await embedding_service.embed(signal.summary)

            await repo.save(signal, embedding)

            results = await repo.search(embedding, SignalFilter(), limit=5)
            match = next(r for r in results if r.signal.id == signal.id)
            assert isinstance(match.signal, FundingRound)
            assert match.signal.amount.amount == Decimal("20000000")
            assert match.signal.series is FundingSeries.B

    asyncio.run(scenario())


def test_save_persists_job_offer_with_embedding(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            signal = _make_job_offer()
            embedding = await embedding_service.embed(signal.summary)

            await repo.save(signal, embedding)

            results = await repo.search(embedding, SignalFilter(), limit=5)
            match = next(r for r in results if r.signal.id == signal.id)
            assert isinstance(match.signal, JobOffer)
            assert match.signal.title == "Senior ML Engineer"
            assert match.signal.seniority is Seniority.SENIOR

    asyncio.run(scenario())


def test_exists_returns_true_for_saved_content_hash(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            signal = _make_funding_round()
            embedding = await embedding_service.embed(signal.summary)
            await repo.save(signal, embedding)

            assert await repo.exists(signal.content_hash) is True

    asyncio.run(scenario())


def test_exists_returns_false_for_unknown_content_hash(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            assert await repo.exists("0" * 64) is False

    asyncio.run(scenario())


def test_search_returns_empty_list_when_collection_is_empty(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            query = await embedding_service.embed("anything at all")

            results = await repo.search(query, SignalFilter(), limit=10)

            assert results == []

    asyncio.run(scenario())


def test_search_returns_signals_ordered_by_similarity(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            fintech = _make_funding_round(
                company="FinPay",
                summary="fintech B2B payments SaaS platform for banks",
            )
            cooking = _make_funding_round(
                company="ChefCo",
                summary="italian pasta cooking recipes and food blog",
            )
            for signal in (fintech, cooking):
                embedding = await embedding_service.embed(signal.summary)
                await repo.save(signal, embedding)

            query = await embedding_service.embed(
                "fintech enterprise payment software"
            )
            results = await repo.search(query, SignalFilter(), limit=10)

            assert len(results) == 2
            assert results[0].signal.company_name == "FinPay"
            assert results[0].semantic_score >= results[1].semantic_score

    asyncio.run(scenario())


def test_search_applies_signal_type_filter(qdrant_settings, embedding_service):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            funding = _make_funding_round(company="FundCo")
            job = _make_job_offer(company="HireCo")
            for signal in (funding, job):
                embedding = await embedding_service.embed(signal.summary)
                await repo.save(signal, embedding)

            query = await embedding_service.embed("machine learning company")
            results = await repo.search(
                query, SignalFilter(signal_type="job_offer"), limit=10
            )

            assert len(results) == 1
            assert all(isinstance(r.signal, JobOffer) for r in results)
            assert results[0].signal.id == job.id

    asyncio.run(scenario())


def test_search_applies_min_amount_filter(qdrant_settings, embedding_service):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            small = _make_funding_round(
                company="SmallCo",
                summary="SmallCo raised a small seed round.",
                amount="1000000",
                series=FundingSeries.SEED,
            )
            big = _make_funding_round(
                company="BigCo",
                summary="BigCo raised a large growth round.",
                amount="50000000",
                series=FundingSeries.GROWTH,
            )
            for signal in (small, big):
                embedding = await embedding_service.embed(signal.summary)
                await repo.save(signal, embedding)

            query = await embedding_service.embed("startup funding round")
            results = await repo.search(
                query,
                SignalFilter(min_amount_usd=Decimal("10000000")),
                limit=10,
            )

            assert len(results) == 1
            assert results[0].signal.id == big.id

    asyncio.run(scenario())


def test_get_by_id_returns_signal_when_exists(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            signal = _make_funding_round()
            embedding = await embedding_service.embed(signal.summary)
            await repo.save(signal, embedding)

            retrieved = await repo.get_by_id(signal.id)

            assert isinstance(retrieved, FundingRound)
            assert retrieved.id == signal.id
            assert retrieved.company_name == signal.company_name
            assert retrieved.amount.amount == signal.amount.amount
            assert retrieved.series is signal.series

    asyncio.run(scenario())


def test_get_by_id_raises_repository_error_when_not_found(
    qdrant_settings, embedding_service
):
    async def scenario():
        async with _repository(qdrant_settings, embedding_service) as repo:
            with pytest.raises(RepositoryError):
                await repo.get_by_id(new_signal_id())

    asyncio.run(scenario())
