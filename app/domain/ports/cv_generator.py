"""Port: profile + opportunity -> tailored CV."""

from abc import ABC, abstractmethod

from app.domain.entities.cv import CV
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.entities.signal import Signal


class ICVGenerator(ABC):
    """Projects a DeveloperProfile onto an opportunity as a tailored CV.

    Note: this interface does not mention ILLMService. An implementation
    may compose one internally (composition, not inheritance, per
    ARCHITECTURE.md §4) — but the use case that depends on this port
    must not know an LLM is involved.
    """

    @abstractmethod
    async def generate(self, profile: DeveloperProfile, target: Signal) -> CV:
        """Generate a CV from `profile` tailored to `target`.

        Guarantee: the implementation calls CV.validate_against(profile)
        before returning, so a CV that comes back is grounded in the
        profile — it never contains invented facts.

        Raises:
            CVHallucinationError: the generated CV still contained
                ungrounded claims after exhausting retries.
            LLMError: the underlying language-model service failed.
        """
        ...
