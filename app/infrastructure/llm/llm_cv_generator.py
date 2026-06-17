"""ICVGenerator implementation that composes an ILLMService (ADR-006)."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from app.domain.entities.cv import CV, CVSection
from app.domain.entities.developer_profile import DeveloperProfile
from app.domain.entities.signal import FundingRound, JobOffer, Signal
from app.domain.exceptions import CVHallucinationError, LLMError
from app.domain.ports.cv_generator import ICVGenerator
from app.domain.ports.llm_service import ILLMService
from app.domain.value_objects.identifiers import new_cv_id

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "cv_generation.txt"

_MAX_ATTEMPTS = 3
_RETRY_SUFFIX = (
    "\n\nBE MORE CONSERVATIVE: only state facts explicitly present in the "
    "profile below. Do NOT extrapolate."
)
# Small local models wrap JSON in a ```json ... ``` fence and often add
# prose before and after it, so we search for the fenced block anywhere in
# the response rather than stripping fences anchored to the start/end.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class LLMCVGenerator(ICVGenerator):
    """Generates a tailored CV by prompting an injected ILLMService.

    Composition, not inheritance (ARCHITECTURE.md §4): it depends on the
    ILLMService interface and never knows which concrete LLM runs behind
    it. No async factory — construction only reads the local prompt file.
    """

    def __init__(self, llm_service: ILLMService) -> None:
        self._llm = llm_service
        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    async def generate(self, profile: DeveloperProfile, target: Signal) -> CV:
        base_prompt = self._build_prompt(profile, target)
        last_error: CVHallucinationError | None = None

        for attempt in range(_MAX_ATTEMPTS):
            prompt = base_prompt if attempt == 0 else base_prompt + _RETRY_SUFFIX
            raw = await self._llm.complete(prompt)  # LLMError propagates
            data = _parse_response(raw)
            cv = self._build_cv(profile, target, data)
            try:
                cv.validate_against(profile)
                return cv
            except CVHallucinationError as e:
                last_error = e
                logger.warning(
                    "CV hallucinated on attempt %d/%d: %s",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    e,
                )

        # Loop always runs at least once, so last_error is set.
        assert last_error is not None
        raise last_error

    def _build_prompt(self, profile: DeveloperProfile, target: Signal) -> str:
        signal_type, thesis_or_description = _describe_target(target)
        return self._prompt_template.format(
            company_name=target.company_name,
            signal_type=signal_type,
            target_summary=target.summary,
            thesis_or_description=thesis_or_description,
            profile_yaml=_profile_to_text(profile),
        )

    def _build_cv(
        self, profile: DeveloperProfile, target: Signal, data: dict
    ) -> CV:
        try:
            sections = tuple(
                CVSection(title=section["title"], content=section["content"])
                for section in data["sections"]
            )
        except (KeyError, TypeError) as e:
            raise LLMError("LLM returned malformed CV sections") from e
        return CV(
            id=new_cv_id(),
            profile_id=profile.id,
            target_signal_id=target.id,
            sections=sections,
            emphasis_rationale=data.get("emphasis_rationale", ""),
            generated_at=datetime.now(timezone.utc),
        )


def _extract_json(raw_response: str) -> str:
    """Extract the JSON object from an LLM response.

    Handles three cases:
    1. JSON wrapped in markdown code fences (with or without 'json' lang)
    2. JSON with preamble/postamble prose around the fences
    3. Raw JSON without fences (fallback)
    """
    match = _JSON_BLOCK_RE.search(raw_response)
    if match:
        return match.group(1)
    # Fallback for when the LLM omits code fences entirely.
    # Greedy \{.*\} spans from the first brace to the last, capturing a
    # single top-level object (our responses contain exactly one).
    brace_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    raise LLMError("No JSON object found in LLM response")


def _parse_response(raw: str) -> dict:
    json_str = _extract_json(raw)
    try:
        # strict=False tolerates control chars (literal \n) inside JSON strings.
        # Small LLMs generating multi-line Markdown content embed real newlines
        # in 'content' fields. The result is technically non-conformant JSON
        # but semantically correct; the resulting Python strings are intact.
        data = json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        raise LLMError("LLM returned invalid JSON") from e
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise LLMError("LLM returned malformed CV structure")
    return data


def _describe_target(target: Signal) -> tuple[str, str]:
    if isinstance(target, FundingRound):
        return "funding_round", target.investment_thesis or "(not specified)"
    if isinstance(target, JobOffer):
        skills = ", ".join(target.required_skills)
        return "job_offer", f"{target.title}; required skills: {skills}"
    return "signal", "(not specified)"


def _profile_to_text(profile: DeveloperProfile) -> str:
    """Render the profile as plain structured text (no pyyaml dependency)."""
    lines = [
        f"full_name: {profile.full_name}",
        f"headline: {profile.headline}",
        f"contact: {profile.contact}",
    ]
    if profile.experiences:
        lines.append("experiences:")
        for exp in profile.experiences:
            lines.append(f"  - role: {exp.role}")
            lines.append(f"    company: {exp.company}")
            if exp.achievements:
                lines.append("    achievements:")
                lines.extend(f"      - {item}" for item in exp.achievements)
            if exp.skills_used:
                lines.append(f"    skills_used: {', '.join(exp.skills_used)}")
    if profile.skills:
        lines.append("skills:")
        lines.extend(
            f"  - {skill.name} ({skill.years}y, {skill.level})"
            for skill in profile.skills
        )
    if profile.projects:
        lines.append("projects:")
        for project in profile.projects:
            lines.append(f"  - name: {project.name}")
            lines.append(f"    description: {project.description}")
            if project.technologies:
                lines.append(
                    f"    technologies: {', '.join(project.technologies)}"
                )
    return "\n".join(lines)
