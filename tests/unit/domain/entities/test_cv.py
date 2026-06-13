from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from app.domain.entities.cv import CV, CVSection
from app.domain.entities.developer_profile import DeveloperProfile, Experience
from app.domain.exceptions import CVHallucinationError
from app.domain.value_objects.identifiers import (
    new_cv_id,
    new_profile_id,
    new_signal_id,
)


def make_profile() -> DeveloperProfile:
    return DeveloperProfile(
        id=new_profile_id(),
        full_name="Sebastian Torreiro",
        headline="Backend developer focused on AI systems",
        contact="dev@example.com",
        experiences=[
            Experience(
                role="Backend Developer",
                company="StartupX",
                achievements=["Reduced p99 latency by 40% migrating to async I/O"],
                skills_used=["Python", "FastAPI"],
            )
        ],
    )


def make_cv(profile: DeveloperProfile, *sections: CVSection) -> CV:
    return CV(
        id=new_cv_id(),
        profile_id=profile.id,
        target_signal_id=new_signal_id(),
        sections=sections,
        emphasis_rationale="Latency work matches the platform-team thesis.",
        generated_at=datetime.now(timezone.utc),
    )


def test_validate_passes_when_every_claim_is_in_profile():
    profile = make_profile()
    cv = make_cv(
        profile,
        CVSection(
            title="Experience",
            content=(
                "Backend Developer\n"
                "Reduced p99 latency by 40% migrating to async I/O"
            ),
        ),
    )

    assert cv.validate_against(profile) is True


def test_validate_raises_hallucination_naming_the_offending_claim():
    profile = make_profile()
    cv = make_cv(
        profile,
        CVSection(title="Experience", content="Led a team of 50 engineers"),
    )

    with pytest.raises(
        CVHallucinationError,
        match=r"\[Experience\] Led a team of 50 engineers",
    ):
        cv.validate_against(profile)


def test_validate_ignores_bullet_markers_and_blank_lines():
    profile = make_profile()
    cv = make_cv(
        profile,
        CVSection(
            title="Experience",
            content=(
                "- Backend Developer\n"
                "\n"
                "  • Reduced p99 latency by 40% migrating to async I/O\n"
                "* StartupX\n"
                "   \n"
            ),
        ),
    )

    assert cv.validate_against(profile) is True


def test_cv_is_immutable():
    profile = make_profile()
    cv = make_cv(profile, CVSection(title="Experience", content="StartupX"))

    with pytest.raises(FrozenInstanceError):
        cv.emphasis_rationale = "tampered"
