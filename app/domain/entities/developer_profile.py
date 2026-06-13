"""The developer's master profile: source of truth for every generated CV."""

import re
from dataclasses import dataclass, field

from app.domain.exceptions import ProfileIncompleteError
from app.domain.value_objects.identifiers import ProfileId

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace, for fact matching across entities."""
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass(kw_only=True)
class Experience:
    role: str
    company: str
    achievements: list[str] = field(default_factory=list)  # quantified, e.g. "cut p99 latency 40%"
    skills_used: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class Skill:
    name: str
    years: float
    level: str  # free text for now ("advanced", "expert"); closed vocab pending


@dataclass(kw_only=True)
class Project:
    name: str
    description: str
    technologies: list[str] = field(default_factory=list)
    url: str | None = None


@dataclass(kw_only=True)
class DeveloperProfile:
    """Never mutated by generation: CVs are projections derived from it."""

    id: ProfileId
    full_name: str
    headline: str
    contact: str
    experiences: list[Experience] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.full_name.strip():
            raise ProfileIncompleteError(
                "DeveloperProfile.full_name must not be empty"
            )

    def corpus(self) -> str:
        """Normalized text of every fact in the profile.

        This is the ground truth that generated artifacts are verified
        against (CV.validate_against): a claim not found here is treated
        as a hallucination.
        """
        parts: list[str] = [self.full_name, self.headline, self.contact]
        for experience in self.experiences:
            parts.append(experience.role)
            parts.append(experience.company)
            parts.extend(experience.achievements)
            parts.extend(experience.skills_used)
        for skill in self.skills:
            parts.append(skill.name)
        for project in self.projects:
            parts.append(project.name)
            parts.append(project.description)
            parts.extend(project.technologies)
        return normalize_text("\n".join(parts))
