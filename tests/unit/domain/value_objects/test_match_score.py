import pytest

from app.domain.value_objects.match_score import MatchScore


@pytest.mark.parametrize("component", ["semantic", "signal_strength", "final"])
@pytest.mark.parametrize("out_of_range", [-0.01, 1.01])
def test_match_score_rejects_components_outside_unit_interval(
    component, out_of_range
):
    kwargs = {"semantic": 0.5, "signal_strength": 0.5, "final": 0.5}
    kwargs[component] = out_of_range

    with pytest.raises(ValueError, match=component):
        MatchScore(**kwargs)
