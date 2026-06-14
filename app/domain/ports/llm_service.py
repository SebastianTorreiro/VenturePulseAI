"""Port: reasoning (entity extraction and free-form completion)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.money import Money


@dataclass(frozen=True, slots=True)
class FundingEntities:
    """Structured funding facts extracted from raw text.

    Every field is optional: the LLM may fail to find a given fact in a
    noisy source. `investors` is an empty tuple when none were found.
    Promotion to a FundingRound entity (which enforces business
    invariants) happens in the application layer, not here.
    """

    amount: Money | None = None
    series: FundingSeries | None = None
    investors: tuple[str, ...] = ()
    investment_thesis: str | None = None


class ILLMService(ABC):
    """Language-model reasoning behind a domain-typed interface."""

    @abstractmethod
    async def extract_funding_entities(self, raw_text: str) -> FundingEntities:
        """Extract amount, series, thesis and investors from raw text.

        Returns a FundingEntities with whatever could be identified;
        fields the model could not determine are left as None / empty.

        Raises:
            LLMError: the provider failed or returned an unparseable
                response after exhausting adapter-level retries.
        """
        ...

    @abstractmethod
    async def complete(self, prompt: str, system: str | None = None) -> str:
        """Free-form completion for unstructured cases.

        Returns the model's text completion verbatim.

        Raises:
            LLMError: the provider failed (API error, timeout).
        """
        ...
