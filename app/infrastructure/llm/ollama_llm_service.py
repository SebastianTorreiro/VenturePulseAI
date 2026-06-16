"""FREE implementation of ILLMService using a local Ollama server."""

import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ollama import AsyncClient

from app.domain.exceptions import LLMError
from app.domain.ports.llm_service import FundingEntities, ILLMService
from app.domain.value_objects.enums import FundingSeries
from app.domain.value_objects.money import Money
from app.infrastructure.config.settings import LLMSettings

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "funding_extraction.txt"

# Strict JSON schema handed to Ollama's `format` for structured output.
# This is data shape, not a prompt, so it stays in code; the prompt text
# lives in prompts/funding_extraction.txt.
_FUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "amount_usd": {"type": "number", "minimum": 0},
        "currency": {"type": ["string", "null"]},
        "series": {
            "type": ["string", "null"],
            "enum": ["SEED", "A", "B", "C", "GROWTH", None],
        },
        "investors": {"type": "array", "items": {"type": "string"}},
        "investment_thesis": {"type": ["string", "null"]},
    },
    "required": [
        "amount_usd",
        "currency",
        "series",
        "investors",
        "investment_thesis",
    ],
}


class OllamaLLMService(ILLMService):
    """Runs entity extraction and completion against a local Ollama model.

    Build with the async factory `create()`, which checks the server is
    reachable and the configured model is pulled — that probe is async and
    cannot live in __init__.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._model = settings.ollama_model
        # HttpUrl renders a trailing slash; the client wants a bare host.
        self._client = AsyncClient(host=str(settings.ollama_host).rstrip("/"))

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
                options={"temperature": 0},
            )
        except Exception as e:
            raise LLMError("Ollama completion failed") from e
        return response.message.content or ""

    async def extract_funding_entities(self, raw_text: str) -> FundingEntities:
        prompt = _load_funding_prompt(raw_text)
        try:
            response = await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format=_FUNDING_SCHEMA,
                stream=False,
                options={"temperature": 0},
            )
            data = json.loads(response.message.content)
        except Exception as e:
            raise LLMError("Ollama funding extraction failed") from e
        return _to_funding_entities(data)


def _load_funding_prompt(raw_text: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(raw_text=raw_text)


def _to_funding_entities(data: dict) -> FundingEntities:
    return FundingEntities(
        amount=_parse_money(data.get("amount_usd")),
        series=_parse_series(data.get("series")),
        investors=tuple(data.get("investors") or ()),
        investment_thesis=data.get("investment_thesis"),
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
