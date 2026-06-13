"""CV: an immutable artifact derived from a DeveloperProfile for one signal."""

from dataclasses import dataclass
from datetime import datetime

from app.domain.entities.developer_profile import DeveloperProfile, normalize_text
from app.domain.exceptions import CVHallucinationError
from app.domain.value_objects.identifiers import CVId, ProfileId, SignalId

_BULLET_PREFIXES = ("-", "*", "•")


@dataclass(kw_only=True, frozen=True)
class CVSection:
    title: str
    content: str  # one claim per line; bullet markers allowed


@dataclass(kw_only=True, frozen=True)
class CV:
    """A projection of a DeveloperProfile onto one opportunity.

    Frozen on purpose: a CV is a derived, auditable artifact
    (ARCHITECTURE.md §3.5) — to change it, generate a new one.
    Key invariant: a CV must not contain facts absent from the profile;
    enforce it by calling validate_against() after generation.
    """

    id: CVId
    profile_id: ProfileId
    target_signal_id: SignalId
    sections: tuple[CVSection, ...]
    emphasis_rationale: str  # why the generator prioritized what it did
    generated_at: datetime

    def validate_against(self, profile: DeveloperProfile) -> bool:
        """Check every claim in this CV is grounded in the profile.

        Returns True when every claim is found. On failure it raises
        CVHallucinationError detailing the unsupported claims — it never
        returns False; the bool return type keeps call sites readable
        (e.g. `assert cv.validate_against(profile)`).

        TODO(domain): naive substring check. It produces false positives
        (a reformulated-but-true claim will be flagged) — replace with
        claim extraction + per-fact verification before relying on it
        beyond the generation retry loop.
        """
        corpus = profile.corpus()
        unsupported: list[str] = []
        for section in self.sections:
            for claim in _claims(section.content):
                if normalize_text(claim) not in corpus:
                    unsupported.append(f"[{section.title}] {claim}")
        if unsupported:
            raise CVHallucinationError(
                "CV contains claims not found in the profile: "
                + "; ".join(unsupported)
            )
        return True


def _claims(content: str) -> list[str]:
    """Split section content into individual claims (one per line)."""
    claims = []
    for line in content.splitlines():
        stripped = line.strip()
        while stripped[:1] in _BULLET_PREFIXES:
            stripped = stripped[1:].lstrip()
        if stripped:
            claims.append(stripped)
    return claims
