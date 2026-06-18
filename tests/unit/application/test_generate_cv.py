"""Unit tests for GenerateCVUseCase — fakes for every port, no I/O."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.application.generate_cv import GenerateCVUseCase
from app.domain.entities.cv import CV, CVSection
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.entities.signal import FundingRound, Signal
from app.domain.exceptions import CVHallucinationError, RepositoryError
from app.domain.ports.cv_generator import ICVGenerator
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.identifiers import (
    SignalId,
    new_cv_id,
    new_profile_id,
    new_signal_id,
)
from app.domain.value_objects.money import Money

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeRepo(ISignalRepository):
    def __init__(self, signal: Signal | None) -> None:
        self._signal = signal

    async def save(self, signal, embedding) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(self, query, filters, limit=10):  # pragma: no cover
        return []

    async def exists(self, content_hash) -> bool:  # pragma: no cover
        return False

    async def get_by_id(self, signal_id: SignalId) -> Signal:
        if self._signal is None:
            raise RepositoryError(f"Signal {signal_id} not found")
        return self._signal


class FakeCVGenerator(ICVGenerator):
    def __init__(self, cv: CV | None = None, error: Exception | None = None) -> None:
        self._cv = cv
        self._error = error
        self.calls = 0

    async def generate(self, profile, target) -> CV:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._cv


def _profile() -> DeveloperProfile:
    return DeveloperProfile(
        id=new_profile_id(),
        full_name="Jane Developer",
        headline="Backend engineer",
        contact="jane@example.com",
    )


def _signal() -> FundingRound:
    return FundingRound(
        id=new_signal_id(),
        source="fake-rss",
        company_name="Acme AI",
        summary="Acme AI raised $20M Series B.",
        detected_at=_NOW,
        signal_strength=0.8,
        amount=Money(amount=Decimal("20000000"), currency="USD"),
        series=FundingSeries.B,
    )


def _cv(profile: DeveloperProfile, signal: FundingRound) -> CV:
    return CV(
        id=new_cv_id(),
        profile_id=profile.id,
        target_signal_id=signal.id,
        sections=(CVSection(title="Headline", content="Backend engineer"),),
        emphasis_rationale="r",
        generated_at=_NOW,
    )


def test_generate_returns_cv_for_valid_signal():
    profile = _profile()
    signal = _signal()
    cv = _cv(profile, signal)
    generator = FakeCVGenerator(cv=cv)
    use_case = GenerateCVUseCase(FakeRepo(signal=signal), generator)

    result = asyncio.run(use_case.execute(signal.id, profile))

    assert result is cv
    assert generator.calls == 1


def test_generate_propagates_repository_error_when_not_found():
    generator = FakeCVGenerator(cv=_cv(_profile(), _signal()))
    use_case = GenerateCVUseCase(FakeRepo(signal=None), generator)

    with pytest.raises(RepositoryError):
        asyncio.run(use_case.execute(new_signal_id(), _profile()))

    assert generator.calls == 0  # never reached the generator


def test_generate_propagates_hallucination_error():
    signal = _signal()
    generator = FakeCVGenerator(error=CVHallucinationError("hallucinated"))
    use_case = GenerateCVUseCase(FakeRepo(signal=signal), generator)

    with pytest.raises(CVHallucinationError):
        asyncio.run(use_case.execute(signal.id, _profile()))
