"""Monetary amount with currency."""

import re
from dataclasses import dataclass
from decimal import Decimal

_ISO_4217 = re.compile(r"[A-Z]{3}")


@dataclass(frozen=True, slots=True)
class Money:
    """An amount of money in a single currency.

    Invariants: amount is finite and strictly positive; currency is an
    ISO 4217 code (three uppercase ASCII letters).
    """

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise ValueError(
                f"Money.amount must be a Decimal, got {type(self.amount).__name__}"
            )
        if not self.amount.is_finite():
            raise ValueError(f"Money.amount must be finite, got {self.amount}")
        if self.amount <= 0:
            raise ValueError(f"Money.amount must be positive, got {self.amount}")
        if not _ISO_4217.fullmatch(self.currency):
            raise ValueError(
                "Money.currency must be an ISO 4217 code "
                f"(3 uppercase letters), got {self.currency!r}"
            )

    def to_usd(self, rate: Decimal) -> "Money":
        """Return a new Money converted to USD at the given rate.

        `rate` is units of USD per unit of this currency. Converting USD
        returns an equal Money without applying the rate.
        """
        if self.currency == "USD":
            return Money(amount=self.amount, currency="USD")
        if not rate.is_finite() or rate <= 0:
            raise ValueError(f"Conversion rate must be positive, got {rate}")
        return Money(amount=self.amount * rate, currency="USD")
