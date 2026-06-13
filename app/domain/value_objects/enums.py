"""Closed vocabularies of the domain."""

from enum import StrEnum


class FundingSeries(StrEnum):
    """Stage of a funding round, ordered by company maturity."""

    SEED = "seed"
    A = "series_a"
    B = "series_b"
    C = "series_c"
    GROWTH = "growth"


class Seniority(StrEnum):
    """Seniority level required by a job offer."""

    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
