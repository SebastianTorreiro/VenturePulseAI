from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest

from app.domain.entities.signal import FundingRound, JobOffer, Signal
from app.domain.exceptions import SignalValidationError
from app.domain.ports.llm_service import FundingEntities
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


def _entities(amount: str = "10000000", **overrides) -> FundingEntities:
    kwargs = dict(
        amount=Money(amount=Decimal(amount), currency="USD"),
        series=FundingSeries.A,
        investors=("Sequoia Capital",),
        investment_thesis="fintech payments",
    )
    kwargs.update(overrides)
    return FundingEntities(**kwargs)


def _from_extraction(**overrides) -> FundingRound:
    kwargs = dict(
        id=new_signal_id(),
        source="techcrunch-rss",
        raw_content="Acme Corp raised $10M in Series A.",
        summary="Acme Corp raised $10M in Series A.",
        detected_at=datetime.now(timezone.utc),
        entities=_entities(),
        series=FundingSeries.A,
    )
    kwargs.update(overrides)
    return FundingRound.from_extraction(**kwargs)


@pytest.mark.parametrize(
    "raw_content,expected_name",
    [
        ("Acme Corp raised $10M in Series A.", "Acme Corp"),
        ("Beta Inc secures $5M seed round.", "Beta Inc"),
        ("Gamma Ltd closed a $2M round.", "Gamma Ltd"),
        ("Delta Co. announced its Series B.", "Delta Co."),
        ("Acme Corp levantó $10M en una Serie A.", "Acme Corp"),
        ("Beta Inc recaudó $5M en una ronda semilla.", "Beta Inc"),
        ("Gamma Ltd cerró una ronda de $2M.", "Gamma Ltd"),
        ("Delta Co. captó fondos en su Serie B.", "Delta Co."),
    ],
)
def test_from_extraction_resolves_company_name_from_funding_verb(
    raw_content, expected_name
):
    signal = _from_extraction(raw_content=raw_content, summary=raw_content)

    assert signal.company_name == expected_name


def test_from_extraction_defaults_company_name_to_unknown_without_match():
    signal = _from_extraction(
        raw_content="No funding verb here.", summary="No funding verb here."
    )

    assert signal.company_name == "Unknown"


def test_from_extraction_prefers_llm_company_name_over_regex():
    signal = _from_extraction(
        entities=_entities(company_name="Acme AI"),
        raw_content="Totally Different Name raised $10M in Series A.",
        summary="Totally Different Name raised $10M in Series A.",
    )

    assert signal.company_name == "Acme AI"


def test_from_extraction_falls_back_to_regex_when_llm_company_name_is_blank():
    signal = _from_extraction(
        entities=_entities(company_name="   "),
        raw_content="Acme Corp raised $10M in Series A.",
        summary="Acme Corp raised $10M in Series A.",
    )

    assert signal.company_name == "Acme Corp"


def test_from_extraction_computes_signal_strength_from_amount():
    signal = _from_extraction(entities=_entities(amount="500000000"))  # $500M

    assert signal.signal_strength == pytest.approx(0.5)


def test_from_extraction_caps_signal_strength_at_one():
    signal = _from_extraction(entities=_entities(amount="5000000000"))  # $5B

    assert signal.signal_strength == 1.0


def test_from_extraction_delegates_to_constructor_validation():
    with pytest.raises(SignalValidationError, match="timezone-aware"):
        _from_extraction(detected_at=datetime(2026, 6, 1, 12, 0, 0))


def test_from_extraction_passes_through_entities_fields():
    signal = _from_extraction(
        entities=_entities(
            investors=("Sequoia Capital", "a16z"), investment_thesis="AI infra"
        )
    )

    assert signal.investors == ["Sequoia Capital", "a16z"]
    assert signal.investment_thesis == "AI infra"


def test_from_extraction_defaults_investment_thesis_to_empty_string():
    signal = _from_extraction(entities=_entities(investment_thesis=None))

    assert signal.investment_thesis == ""
