"""Composite score of a profile/signal match (see docs/specs/matching-logic.md)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Final_Score = alpha * semantic + beta * signal_strength.

    Invariants: every component is within [0, 1]. The weighting itself is
    application logic (application/services/scoring.py); this value object
    only guarantees the components are normalized.
    """

    semantic: float
    signal_strength: float
    final: float

    def __post_init__(self) -> None:
        for name in ("semantic", "signal_strength", "final"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"MatchScore.{name} must be within [0, 1], got {value}"
                )
