"""Market signals: the unit that gets embedded and stored (ADR-001)."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from app.domain.exceptions import SignalValidationError
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import SignalId
from app.domain.value_objects.money import Money

if TYPE_CHECKING:
    # Deferred to break the import cycle: app.domain.ports.__init__ pulls in
    # cv_generator, which imports Signal from this module.
    from app.domain.ports.llm_service import FundingEntities, JobEntities

# Signal strength normalizes the funding amount: $1B caps the score at 1.0.
_STRENGTH_DENOMINATOR = 1_000_000_000
# Company name = text before the first funding verb (naive MVP heuristic).
_COMPANY_RE = re.compile(
    r"(.+?)\s+(?:raised|raises|announced|announces|secured|secures|"
    r"closed|closes|"
    r"levanta|levantó|recauda|recaudó|capta|captó|"
    r"consigue|consiguió|cierra|cerró)\b",
    re.IGNORECASE,
)


def _extract_company_name(text: str) -> str:
    """Best-effort company name: text before the first funding verb.

    MVP heuristic, no LLM. Returns 'Unknown' when nothing matches.
    """
    match = _COMPANY_RE.search(text)
    if match:
        name = match.group(1).strip()
        if name:
            return name
    return "Unknown"


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

    @classmethod
    def from_extraction(
        cls,
        *,
        id: SignalId,
        source: str,
        raw_content: str,
        summary: str,
        detected_at: datetime,
        entities: "FundingEntities",
        series: FundingSeries,
    ) -> "FundingRound":
        """Build a FundingRound from raw text and LLM-extracted entities.

        Resolves company_name (prefers the LLM-extracted entities.company_name;
        falls back to a regex over raw_content when the LLM didn't find one)
        and signal_strength (normalized amount) before delegating to the
        normal constructor, so __post_init__ validation still runs
        unmodified.
        """
        return cls(
            id=id,
            source=source,
            company_name=(
                entities.company_name.strip()
                if entities.company_name and entities.company_name.strip()
                else _extract_company_name(raw_content)
            ),
            summary=summary,
            detected_at=detected_at,
            signal_strength=min(
                1.0, float(entities.amount.amount) / _STRENGTH_DENOMINATOR
            ),
            amount=entities.amount,
            series=series,
            investors=list(entities.investors),
            investment_thesis=entities.investment_thesis or "",
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

    @classmethod
    def from_extraction(
        cls,
        *,
        id: SignalId,
        source: str,
        raw_content: str,
        summary: str,
        detected_at: datetime,
        entities: "JobEntities",
        seniority: Seniority,
        url: str,
    ) -> "JobOffer":
        """Build a JobOffer from raw text and LLM-extracted entities.

        Resolves company_name the same way FundingRound.from_extraction()
        does: prefers entities.company_name, falls back to a regex over
        raw_content when the LLM didn't find one. That regex only
        matches funding verbs, so it rarely helps for job postings —
        "Unknown" is the honest outcome most of the time until a
        job-posting-specific fallback is worth writing.

        seniority is a separate parameter rather than read from
        `entities` directly, mirroring how from_extraction() takes
        `series` explicitly for FundingRound: the caller resolves
        missing/unclear values (e.g. defaulting to Seniority.UNKNOWN) before
        calling this, so that policy lives in the application layer, not
        here.

        signal_strength is a constant 1.0: unlike FundingRound (an
        anticipatory signal scored by funding amount), a JobOffer is
        confirmatory — it already happened, so there's no magnitude to
        normalize against.
        """
        return cls(
            id=id,
            source=source,
            company_name=(
                entities.company_name.strip()
                if entities.company_name and entities.company_name.strip()
                else _extract_company_name(raw_content)
            ),
            summary=summary,
            detected_at=detected_at,
            signal_strength=1.0,
            title=entities.title or "",
            required_skills=list(entities.required_skills),
            seniority=seniority,
            url=url,
        )
