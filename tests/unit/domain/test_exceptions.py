import pytest

from app.domain.exceptions import (
    CVHallucinationError,
    DomainError,
    EmbeddingError,
    InfrastructureError,
    LLMError,
    ProfileIncompleteError,
    RepositoryError,
    ScrapingError,
    SignalValidationError,
    VenturePulseError,
)


@pytest.mark.parametrize(
    "error_class",
    [
        DomainError,
        SignalValidationError,
        CVHallucinationError,
        ProfileIncompleteError,
        InfrastructureError,
        ScrapingError,
        EmbeddingError,
        LLMError,
        RepositoryError,
    ],
)
def test_all_errors_inherit_from_venturepulse_root(error_class):
    assert issubclass(error_class, VenturePulseError)


def test_domain_and_infrastructure_branches_do_not_overlap():
    assert not issubclass(DomainError, InfrastructureError)
    assert not issubclass(InfrastructureError, DomainError)
