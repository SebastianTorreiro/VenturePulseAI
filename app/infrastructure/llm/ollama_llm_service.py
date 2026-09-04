"""FREE implementation of ILLMService using a local Ollama server."""

import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ollama import AsyncClient

from app.domain.exceptions import LLMError
from app.domain.ports.llm_service import FundingEntities, ILLMService, JobEntities
from app.domain.value_objects.enums import FundingSeries, Seniority
from app.domain.value_objects.money import Money
from app.infrastructure.config.settings import LLMSettings
from app.infrastructure.llm.schemas import (
    FundingExtractionSchema,
    JobExtractionSchema,
    render_fields_section,
    to_ollama_schema,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "funding_extraction.txt"
_JOB_PROMPT_PATH = Path(__file__).parent / "prompts" / "job_extraction.txt"

# JSON schemas handed to Ollama's `format` for structured output, derived
# from the Pydantic models in schemas.py — single source of truth shared
# with each prompt's "Fields:" section (see render_fields_section below).
_FUNDING_SCHEMA = to_ollama_schema(FundingExtractionSchema)
_JOB_SCHEMA = to_ollama_schema(JobExtractionSchema)


class OllamaLLMService(ILLMService):
    """Runs entity extraction and completion against a local Ollama model.

    Build with the async factory `create()`, which checks the server is
    reachable and the configured model is pulled — that probe is async and
    cannot live in __init__.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._model = settings.ollama_model
        self._num_ctx = settings.num_ctx
        # HttpUrl renders a trailing slash; the client wants a bare host.
        self._client = AsyncClient(host=str(settings.ollama_host).rstrip("/"))
        self._funding_prompt_template = _PROMPT_PATH.read_text(
            encoding="utf-8"
        ).replace("{fields_section}", render_fields_section(FundingExtractionSchema))
        self._job_prompt_template = _JOB_PROMPT_PATH.read_text(
            encoding="utf-8"
        ).replace("{fields_section}", render_fields_section(JobExtractionSchema))

    @classmethod
    async def create(cls, settings: LLMSettings) -> "OllamaLLMService":
        service = cls(settings)
        try:
            response = await service._client.list()
        except Exception as e:
            raise LLMError(
                f"Cannot reach Ollama at {settings.ollama_host}"
            ) from e

        available = {model.model for model in response.models}
        if not service._is_model_available(available):
            raise LLMError(
                f"Model {service._model!r} is not available in Ollama. "
                f"Run `ollama pull {service._model}` first "
                f"(available: {sorted(available)})"
            )
        logger.info("Ollama ready with model %s", service._model)
        return service

    def _is_model_available(self, available: set[str]) -> bool:
        if self._model in available:
            return True
        # Tolerate an omitted ":latest"-style tag in the configured name.
        if ":" not in self._model:
            return any(name.split(":")[0] == self._model for name in available)
        return False

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                stream=False,
                options={"temperature": 0, "num_ctx": self._num_ctx},
            )
        except Exception as e:
            raise LLMError("Ollama completion failed") from e
        return response.message.content or ""

    async def extract_funding_entities(self, raw_text: str) -> FundingEntities:
        prompt = self._funding_prompt_template.format(raw_text=raw_text)
        try:
            response = await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format=_FUNDING_SCHEMA,
                stream=False,
                options={"temperature": 0, "num_ctx": self._num_ctx},
            )
            data = json.loads(response.message.content)
        except Exception as e:
            raise LLMError("Ollama funding extraction failed") from e
        return _to_funding_entities(data)

    async def extract_job_entities(self, raw_text: str) -> JobEntities:
        prompt = self._job_prompt_template.format(raw_text=raw_text)
        try:
            response = await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format=_JOB_SCHEMA,
                stream=False,
                options={"temperature": 0, "num_ctx": self._num_ctx},
            )
            data = json.loads(response.message.content)
        except Exception as e:
            raise LLMError("Ollama job extraction failed") from e
        return _to_job_entities(data)


def _to_funding_entities(data: dict) -> FundingEntities:
    return FundingEntities(
        amount=_parse_money(data.get("amount_usd")),
        series=_parse_series(data.get("series")),
        investors=tuple(data.get("investors") or ()),
        investment_thesis=data.get("investment_thesis"),
        company_name=_parse_company_name(data.get("company_name")),
    )


def _to_job_entities(data: dict) -> JobEntities:
    return JobEntities(
        company_name=_parse_company_name(data.get("company_name")),
        title=_parse_title(data.get("title")),
        required_skills=tuple(data.get("required_skills") or ()),
        seniority=_parse_seniority(data.get("seniority")),
    )


def _parse_money(amount_usd: object) -> Money | None:
    # The schema forces a number (no null); 0 is the "not found" sentinel
    # the adapter translates back to None for the domain. amount_usd is
    # already in USD by definition of the field.
    try:
        amount = Decimal(str(amount_usd))
    except (InvalidOperation, ValueError):
        # Defensive: a noisy model could still emit a non-numeric value.
        return None
    if amount <= 0:
        return None
    return Money(amount=amount, currency="USD")


def _parse_series(series: object) -> FundingSeries | None:
    if isinstance(series, str) and series.upper() in FundingSeries.__members__:
        return FundingSeries[series.upper()]
    return None


def _parse_company_name(company_name: object) -> str | None:
    if isinstance(company_name, str) and company_name.strip():
        return company_name.strip()
    return None


def _parse_title(title: object) -> str | None:
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _parse_seniority(seniority: object) -> Seniority | None:
    if isinstance(seniority, str) and seniority.upper() in Seniority.__members__:
        return Seniority[seniority.upper()]
    return None
