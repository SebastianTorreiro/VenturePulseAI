import pytest

from app.domain.entities.developer_profile import (
    DeveloperProfile,
    Experience,
    Project,
    Skill,
)
from app.domain.exceptions import ProfileIncompleteError
from app.domain.value_objects.identifiers import new_profile_id


def test_profile_rejects_blank_full_name():
    with pytest.raises(ProfileIncompleteError, match="full_name"):
        DeveloperProfile(
            id=new_profile_id(),
            full_name="   ",
            headline="Backend developer",
            contact="dev@example.com",
        )


def test_corpus_contains_every_achievement_and_skill():
    profile = DeveloperProfile(
        id=new_profile_id(),
        full_name="Sebastian Torreiro",
        headline="Backend developer focused on AI systems",
        contact="dev@example.com",
        experiences=[
            Experience(
                role="Backend Developer",
                company="StartupX",
                achievements=["Reduced p99 latency by 40% migrating to async I/O"],
                skills_used=["FastAPI"],
            )
        ],
        skills=[Skill(name="Python", years=5, level="advanced")],
        projects=[
            Project(
                name="VenturePulse",
                description="Job market signal collector",
                technologies=["Qdrant"],
            )
        ],
    )

    corpus = profile.corpus()

    assert "reduced p99 latency by 40% migrating to async i/o" in corpus
    assert "fastapi" in corpus
    assert "python" in corpus
    assert "venturepulse" in corpus
    assert "qdrant" in corpus
