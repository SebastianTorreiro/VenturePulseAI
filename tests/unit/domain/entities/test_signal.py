from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest

from app.domain.entities.signal import FundingRound, JobOffer, Signal
from app.domain.exceptions import SignalValidationError
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import new_signal_id
from app.domain.value_objects.money import Money


def make_signal(**overrides) -> Signal:
    kwargs = dict(
        id=new_signal_id(),
        source="techcrunch-rss",
        company_name="Acme AI",
        summary="Acme AI raised a $20M Series B to expand its ML platform.",
        detected_at=datetime.now(timezone.utc),
        signal_strength=0.8,
    )
    kwargs.update(overrides)
    return Signal(**kwargs)


def make_funding_round(**overrides) -> FundingRound:
    kwargs = dict(
        id=new_signal_id(),
        source="techcrunch-rss",
        company_name="Acme AI",
        summary="Acme AI raised a $20M Series B to expand its ML platform.",
        detected_at=datetime.now(timezone.utc),
        signal_strength=0.8,
        amount=Money(amount=Decimal("20000000"), currency="USD"),
        series=FundingSeries.B,
    )
    kwargs.update(overrides)
    return FundingRound(**kwargs)


def make_job_offer(**overrides) -> JobOffer:
    kwargs = dict(
        id=new_signal_id(),
        source="remoteok-jobs",
        company_name="Acme AI",
        summary="Senior ML engineer for the platform team.",
        detected_at=datetime.now(timezone.utc),
        signal_strength=0.6,
        title="Senior ML Engineer",
        required_skills=["Python", "Qdrant"],
        seniority=Seniority.SENIOR,
        url="https://example.com/job/1",
    )
    kwargs.update(overrides)
    return JobOffer(**kwargs)


def test_signal_is_fresh_within_ttl():
    signal = make_signal(
        detected_at=datetime.now(timezone.utc) - timedelta(days=10)
    )

    assert signal.is_fresh(ttl_days=30)


def test_signal_is_stale_past_ttl():
    signal = make_signal(
        detected_at=datetime.now(timezone.utc) - timedelta(days=31)
    )

    assert not signal.is_fresh(ttl_days=30)


def test_signal_rejects_empty_company_name():
    with pytest.raises(SignalValidationError, match="company_name"):
        make_signal(company_name="   ")


def test_signal_rejects_blank_summary():
    with pytest.raises(SignalValidationError, match="summary"):
        make_signal(summary="   ")


def test_signal_rejects_naive_detected_at():
    with pytest.raises(SignalValidationError, match="timezone-aware"):
        make_signal(detected_at=datetime(2026, 6, 1, 12, 0, 0))


def test_signal_rejects_datetime_with_none_utcoffset():
    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

    aware_looking = datetime(2026, 6, 1, 12, 0, 0, tzinfo=NoOffset())

    with pytest.raises(SignalValidationError, match="timezone-aware"):
        make_signal(detected_at=aware_looking)


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_signal_rejects_strength_outside_unit_interval(strength):
    with pytest.raises(SignalValidationError, match="signal_strength"):
        make_signal(signal_strength=strength)


def test_content_hash_is_stable_for_identical_content():
    first = make_signal()
    second = make_signal()

    assert first.id != second.id
    assert first.content_hash == second.content_hash


def test_content_hash_differs_when_summary_changes():
    base = make_signal()
    changed = make_signal(summary="Acme AI acquired BetaCorp for $5M.")

    assert base.content_hash != changed.content_hash


def test_funding_round_rejects_non_positive_amount():
    # Money's own invariant makes a non-positive amount unbuildable through
    # the public API, so bypass it to exercise the entity's defensive rule.
    tampered = Money(amount=Decimal("1"), currency="USD")
    object.__setattr__(tampered, "amount", Decimal("0"))

    with pytest.raises(SignalValidationError, match="must be positive"):
        make_funding_round(amount=tampered)


def test_job_offer_rejects_blank_title():
    with pytest.raises(SignalValidationError, match="title"):
        make_job_offer(title="   ")


def test_job_offer_normalizes_skills_to_lowercase_and_drops_blanks():
    offer = make_job_offer(required_skills=["  Python ", "QDRANT", "", "  "])

    assert offer.required_skills == ["python", "qdrant"]
