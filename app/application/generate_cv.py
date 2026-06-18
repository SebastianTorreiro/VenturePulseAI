"""Use case: generate a tailored CV for a stored signal.

Imports only from app.domain.* — no infrastructure.
"""

from app.domain.entities.cv import CV
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.ports.cv_generator import ICVGenerator
from app.domain.ports.signal_repository import ISignalRepository
from app.domain.value_objects.identifiers import SignalId


class GenerateCVUseCase:
    def __init__(
        self, repository: ISignalRepository, cv_generator: ICVGenerator
    ) -> None:
        self._repo = repository
        self._cv_generator = cv_generator

    async def execute(
        self, signal_id: SignalId, profile: DeveloperProfile
    ) -> CV:
        # RepositoryError (not found / I/O) and CVHallucinationError /
        # LLMError (generation) propagate to the caller deliberately.
        signal = await self._repo.get_by_id(signal_id)
        return await self._cv_generator.generate(profile, signal)
