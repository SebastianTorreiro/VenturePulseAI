from decimal import Decimal

import pytest

from app.domain.value_objects.money import Money


def test_money_rejects_negative_amount():
    with pytest.raises(ValueError, match="must be positive"):
        Money(amount=Decimal("-5"), currency="USD")


def test_money_rejects_nan_amount():
    with pytest.raises(ValueError, match="must be finite"):
        Money(amount=Decimal("NaN"), currency="USD")


def test_money_rejects_non_decimal_amount():
    with pytest.raises(ValueError, match="must be a Decimal"):
        Money(amount=100.0, currency="USD")


def test_money_rejects_lowercase_currency():
    with pytest.raises(ValueError, match="ISO 4217"):
        Money(amount=Decimal("100"), currency="usd")


def test_money_to_usd_applies_rate_and_returns_new_instance():
    original = Money(amount=Decimal("100"), currency="EUR")

    converted = original.to_usd(Decimal("1.08"))

    assert converted == Money(amount=Decimal("108.00"), currency="USD")
    assert converted is not original


def test_money_to_usd_is_identity_when_already_usd():
    original = Money(amount=Decimal("100"), currency="USD")

    converted = original.to_usd(Decimal("2"))

    assert converted == original


@pytest.mark.parametrize("rate", [Decimal("0"), Decimal("-1.5")])
def test_money_to_usd_rejects_non_positive_rate(rate):
    money = Money(amount=Decimal("100"), currency="EUR")

    with pytest.raises(ValueError, match="rate must be positive"):
        money.to_usd(rate)
