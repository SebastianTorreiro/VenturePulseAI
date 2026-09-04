"""Shared builders for domain port DTOs used across unit tests.

Consolidates what used to be two near-duplicate `_entities()` helpers
in tests/unit/application/test_ingest_signals.py and
tests/unit/domain/entities/test_signal.py. CONVENTIONS.md §4.2
documents tests/fixtures/ as the home for fakes/factories shared
across test modules.
"""

from decimal import Decimal

from app.domain.ports.llm_service import FundingEntities, JobEntities
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.money import Money


def make_funding_entities(
    amount: str | None = "10000000", **overrides
) -> FundingEntities:
    kwargs = dict(
        series=FundingSeries.A,
        investors=("Sequoia Capital",),
        investment_thesis="fintech payments",
        company_name=None,
    )
    kwargs.update(overrides)
    return FundingEntities(
        amount=(
            Money(amount=Decimal(amount), currency="USD")
            if amount is not None
            else None
        ),
        **kwargs,
    )


def make_job_entities(**overrides) -> JobEntities:
    kwargs = dict(
        company_name="Acme Corp",
        title="Senior Backend Engineer",
        required_skills=("python", "fastapi"),
        seniority=Seniority.SENIOR,
    )
    kwargs.update(overrides)
    return JobEntities(**kwargs)
