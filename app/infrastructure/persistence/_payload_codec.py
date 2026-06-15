"""Serialization between domain Signal and Qdrant payload dict.

The embedding vector is NOT part of the payload — it travels as the
point's vector. Money is normalized to USD; non-USD is rejected for now
(TODO post-MVP: convert via FX rates). The payload carries a `type`
discriminator so payload_to_signal can rebuild the right subclass.
"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from app.domain.entities.signal import FundingRound, JobOffer, Signal
from app.domain.exceptions import RepositoryError
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import SignalId
from app.domain.value_objects.money import Money

FUNDING_ROUND = "funding_round"
JOB_OFFER = "job_offer"


def _amount_in_usd(money: Money) -> Decimal:
    if money.currency != "USD":
        # TODO(post-MVP): support non-USD amounts via an FX rate source.
        raise RepositoryError(
            f"Only USD amounts are supported for now, got {money.currency!r}"
        )
    return money.to_usd(Decimal("1")).amount


def signal_to_payload(signal: Signal) -> dict:
    """Map a domain Signal to a JSON-serializable Qdrant payload."""
    payload: dict = {
        "id": str(signal.id),
        "source": signal.source,
        "company_name": signal.company_name,
        "summary": signal.summary,
        "detected_at": signal.detected_at.isoformat(),
        "signal_strength": signal.signal_strength,
        "content_hash": signal.content_hash,
    }

    if isinstance(signal, FundingRound):
        payload["type"] = FUNDING_ROUND
        # Stored as float so Qdrant range filters work; precision is
        # acceptable for MVP filtering (TODO post-MVP: revisit).
        payload["amount_usd"] = float(_amount_in_usd(signal.amount))
        payload["currency"] = "USD"
        payload["series"] = signal.series.value
        payload["investors"] = list(signal.investors)
        payload["investment_thesis"] = signal.investment_thesis
        return payload

    if isinstance(signal, JobOffer):
        payload["type"] = JOB_OFFER
        payload["title"] = signal.title
        payload["required_skills"] = list(signal.required_skills)
        payload["seniority"] = signal.seniority.value
        payload["url"] = signal.url
        if signal.salary_range is not None:
            payload["salary_amount_usd"] = float(_amount_in_usd(signal.salary_range))
            payload["salary_currency"] = "USD"
        return payload

    raise RepositoryError(
        f"Cannot serialize signal of type {type(signal).__name__}"
    )


def payload_to_signal(payload: dict) -> Signal:
    """Rebuild a domain Signal from a Qdrant payload."""
    common = {
        "id": SignalId(UUID(payload["id"])),
        "source": payload["source"],
        "company_name": payload["company_name"],
        "summary": payload["summary"],
        "detected_at": datetime.fromisoformat(payload["detected_at"]),
        "signal_strength": payload["signal_strength"],
    }
    signal_type = payload.get("type")

    if signal_type == FUNDING_ROUND:
        return FundingRound(
            **common,
            amount=Money(
                amount=Decimal(str(payload["amount_usd"])),
                currency=payload["currency"],
            ),
            series=FundingSeries(payload["series"]),
            investors=list(payload.get("investors", [])),
            investment_thesis=payload.get("investment_thesis", ""),
        )

    if signal_type == JOB_OFFER:
        salary_range = None
        if payload.get("salary_amount_usd") is not None:
            salary_range = Money(
                amount=Decimal(str(payload["salary_amount_usd"])),
                currency=payload["salary_currency"],
            )
        return JobOffer(
            **common,
            title=payload["title"],
            required_skills=list(payload.get("required_skills", [])),
            seniority=Seniority(payload["seniority"]),
            url=payload["url"],
            salary_range=salary_range,
        )

    raise RepositoryError(f"Unknown signal type in payload: {signal_type!r}")
