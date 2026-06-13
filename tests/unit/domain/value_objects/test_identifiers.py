import pytest

from app.domain.value_objects.identifiers import (
    new_cv_id,
    new_profile_id,
    new_signal_id,
)


@pytest.mark.parametrize("factory", [new_signal_id, new_profile_id, new_cv_id])
def test_factory_ids_are_unique_across_calls(factory):
    first = factory()
    second = factory()

    assert first != second
