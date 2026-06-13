"""Market signals: the unit that gets embedded and stored (ADR-001)."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.domain.exceptions import SignalValidationError
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import SignalId
from app.domain.value_objects.money import Money


@dataclass(kw_only=True)
class RawSignal:
    """What an ISignalScraper produces, before LLM enrichment.

    Carries no business invariants on purpose: structural validation of
    scraped data happens at the boundary (Pydantic, per ADR-005 and
    docs/specs/data-ingestion.md), before a RawSignal is promoted to a
    domain Signal.
    """

    source: str  # stable scraper slug, e.g. "techcrunch-rss"
    url: str
    content: str  # cleaned text, HTML boilerplate already removed
    fetched_at: datetime


@dataclass(kw_only=True)
class Signal:
    """Base of every market observation with predictive value.

    `summary` is the canonical text that gets embedded (one chunk per
    signal, see docs/specs/embedding-pipeline.md).
    """

    id: SignalId
    source: str
    company_name: str
    summary: str
    detected_at: datetime
    signal_strength: float  # normalized [0, 1], computed by application scoring

    def __post_init__(self) -> None:
        if not self.company_name.strip():
            raise SignalValidationError("Signal.company_name must not be empty")
        if not self.summary.strip():
            raise SignalValidationError("Signal.summary must not be empty")
        if not 0.0 <= self.signal_strength <= 1.0:
            raise SignalValidationError(
                f"Signal.signal_strength must be within [0, 1], "
                f"got {self.signal_strength}"
            )
        if (
            self.detected_at.tzinfo is None
            or self.detected_at.tzinfo.utcoffset(self.detected_at) is None
        ):
            raise SignalValidationError(
                "Signal.detected_at must be timezone-aware"
            )

    def is_fresh(self, ttl_days: int = 30) -> bool:
        return datetime.now(timezone.utc) - self.detected_at <= timedelta(
            days=ttl_days
        )

    @property
    def content_hash(self) -> str:
        """Stable hash for deduplication (ISignalRepository.exists)."""
        canonical = f"{self.source}|{self.company_name.lower()}|{self.summary}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(kw_only=True)
class FundingRound(Signal):
    """Anticipatory signal: precedes job openings by 2-4 weeks (ADR-001)."""

    amount: Money
    series: FundingSeries
    investors: list[str] = field(default_factory=list)
    investment_thesis: str = ""  # extracted by ILLMService

    def __post_init__(self) -> None:
        super().__post_init__()  # company_name / summary / strength checks
        # Money already rejects non-positive amounts with ValueError at
        # construction; this re-check states the business rule in domain
        # terms in case the value object invariant ever relaxes.
        if self.amount.amount <= 0:
            raise SignalValidationError(
                f"FundingRound.amount must be positive, got {self.amount.amount}"
            )


@dataclass(kw_only=True)
class JobOffer(Signal):
    """Confirmatory signal: a published vacancy."""

    title: str
    required_skills: list[str]
    seniority: Seniority
    url: str
    salary_range: Money | None = None  # most offers do not publish it

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.title.strip():
            raise SignalValidationError("JobOffer.title must not be empty")
        # Skills are matched case-insensitively across the whole system.
        self.required_skills = [
            skill.strip().lower() for skill in self.required_skills if skill.strip()
        ]
