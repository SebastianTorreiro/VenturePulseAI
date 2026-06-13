"""Immutable value objects of the domain. Pure stdlib — no frameworks."""

from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.identifiers import (
    CVId,
    ProfileId,
    SignalId,
    new_cv_id,
    new_profile_id,
    new_signal_id,
)
from app.domain.value_objects.match_score import MatchScore
from app.domain.value_objects.money import Money

__all__ = [
    "CVId",
    "Embedding",
    "FundingSeries",
    "MatchScore",
    "Money",
    "ProfileId",
    "Seniority",
    "SignalId",
    "new_cv_id",
    "new_profile_id",
    "new_signal_id",
]
