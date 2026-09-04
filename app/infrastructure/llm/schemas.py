"""Pydantic schemas for LLM structured-output extraction (infra-only).

These models exist purely to give Ollama's `format` parameter a JSON
schema and to generate the "Fields:" section of each extraction prompt
from a single source of truth (each Field's description). They are
never imported by app.domain or app.application — the domain-facing
shapes are FundingEntities/JobEntities (plain dataclasses in
app.domain.ports.llm_service), built by hand-written adapter code from
whatever JSON these schemas' structured output produces (ADR-005:
Pydantic only at the boundaries, never inside the domain).
"""

from typing import Literal

from pydantic import BaseModel, Field


class FundingExtractionSchema(BaseModel):
    """Structured-output contract for OllamaLLMService.extract_funding_entities."""

    company_name: str | None = Field(
        None,
        description=(
            'the name of the company/startup raising the funding, exactly '
            'as it appears in the text (e.g., "Acme Corp"); null if unclear'
        ),
    )
    amount_usd: float = Field(
        0,
        description=(
            'numeric amount in USD as an integer (e.g., "$10M" -> 10000000). '
            "Output 0 only if no monetary amount is mentioned at all."
        ),
    )
    currency: str | None = Field(
        None, description="original currency code (USD/EUR/GBP); null if unknown"
    )
    series: Literal["SEED", "A", "B", "C", "GROWTH"] | None = Field(
        None, description="one of SEED, A, B, C, GROWTH, or null"
    )
    investors: list[str] = Field(
        default_factory=list,
        description="array of investor names (empty array if none mentioned)",
    )
    investment_thesis: str | None = Field(
        None,
        description=(
            "short phrase describing the company's sector or business "
            "(null if unclear)"
        ),
    )


class JobExtractionSchema(BaseModel):
    """Structured-output contract for OllamaLLMService.extract_job_entities."""

    company_name: str | None = Field(
        None,
        description=(
            'the name of the company/organization hiring, exactly as it '
            'appears in the text (e.g., "Acme Corp"); null if unclear'
        ),
    )
    title: str | None = Field(
        None,
        description=(
            'the job title being advertised (e.g., "Senior Backend '
            'Engineer"); null if unclear'
        ),
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "array of required technical skills or technologies mentioned "
            "(empty array if none listed)"
        ),
    )
    seniority: Literal["JUNIOR", "MID", "SENIOR", "STAFF"] | None = Field(
        None,
        description=(
            "one of JUNIOR, MID, SENIOR, STAFF, or null if the seniority "
            "level is not clear from the text"
        ),
    )


def to_ollama_schema(model: type[BaseModel]) -> dict:
    """Derive Ollama's `format` JSON schema from a Pydantic model.

    model_json_schema() omits "required" here because every field has a
    default (each fact is optional-to-the-LLM by design) — left as-is,
    Ollama would let the model omit fields instead of stating them as
    null. Every field is forced into "required" so the model always
    states a value (or null), matching the discipline the hand-written
    schema used to enforce. Verified empirically: Ollama accepts the
    rest of the generated schema (anyOf nullable unions, default, title,
    description) unmodified via chat(format=...).
    """
    schema = model.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


def render_fields_section(model: type[BaseModel]) -> str:
    """Render the "Fields:" prompt block from a schema's Field descriptions.

    Single source of truth: the same descriptions that document each
    field also become the model's extraction instructions, so the
    prompt text and the schema can never drift apart.
    """
    lines = ["Fields:"]
    for name, field_info in model.model_fields.items():
        lines.append(f"- {name}: {field_info.description}")
    return "\n".join(lines)
