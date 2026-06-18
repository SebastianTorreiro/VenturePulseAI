"""Command-line entry point for VenturePulseAI.

Presentation layer: wires the composition root to the use cases. Each
command builds the container and runs its use case inside a single
asyncio.run() so the adapters' async clients (Qdrant, Ollama) live on one
event loop — building the container in one loop and running the use case
in another would use clients bound to an already-closed loop.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
import yaml

from app.application.generate_cv import GenerateCVUseCase
from app.application.ingest_signals import IngestSignalsUseCase
from app.application.search_signals import SearchSignalsUseCase
from app.domain.entities.developer_profile import (
    DeveloperProfile,
    Experience,
    Project,
    Skill,
)
from app.domain.exceptions import ProfileIncompleteError
from app.domain.value_objects.identifiers import SignalId, new_profile_id
from app.infrastructure.config.container import build_container

app = typer.Typer(help="VenturePulseAI — job-market signal intelligence.")


@app.command()
def collect(
    days: int = typer.Option(7, help="Collect signals from last N days"),
):
    """Scrape, extract, deduplicate, embed and persist funding signals."""

    async def _run():
        container = await build_container()
        use_case = IngestSignalsUseCase(
            container.scraper,
            container.llm_service,
            container.embedder,
            container.repository,
        )
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return await use_case.execute(since)

    result = asyncio.run(_run())
    typer.echo(f"Scraped:              {result.scraped}")
    typer.echo(f"Ingested:             {result.ingested}")
    typer.echo(f"Skipped (duplicate):  {result.skipped_duplicate}")
    typer.echo(f"Skipped (no data):    {result.skipped_no_entities}")
    typer.echo(f"Errors:               {result.errors}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(10, help="Max results"),
    min_amount: Optional[float] = typer.Option(None, help="Min amount USD"),
):
    """Semantic search over stored signals, with optional amount filter."""

    async def _run():
        container = await build_container()
        use_case = SearchSignalsUseCase(
            container.embedder, container.repository
        )
        return await use_case.execute(
            query=query,
            limit=limit,
            min_amount_usd=Decimal(str(min_amount)) if min_amount else None,
        )

    result = asyncio.run(_run())
    if not result.signals:
        typer.echo("No signals found.")
        raise typer.Exit(0)

    typer.echo(f"Found {result.total} signal(s) for '{result.query}':\n")
    for i, scored in enumerate(result.signals, 1):
        s = scored.signal
        amount_str = (
            f"${s.amount.amount / 1_000_000:.1f}M {s.amount.currency}"
            if hasattr(s, "amount") and s.amount
            else "amount unknown"
        )
        typer.echo(f"  {i}. [{s.id}] {s.company_name}")
        typer.echo(f"     Score: {scored.semantic_score:.2f} | {amount_str}")
        typer.echo(f"     {s.summary[:100]}...")
        typer.echo("")


@app.command()
def apply(
    signal_id: str = typer.Argument(..., help="Signal ID from search output"),
    profile_path: Path = typer.Option(
        Path("profile/developer_profile.yaml"),
        help="Path to developer profile YAML",
    ),
    output: Path = typer.Option(
        Path("cv_output.md"),
        help="Output markdown file",
    ),
):
    """Generate a tailored CV for a signal and write it to a markdown file."""
    if not profile_path.exists():
        typer.echo(f"Error: profile not found at {profile_path}", err=True)
        typer.echo(
            "Copy profile/developer_profile.example.yaml to "
            "profile/developer_profile.yaml and fill in your data.",
            err=True,
        )
        raise typer.Exit(1)

    profile = _load_profile(profile_path)

    try:
        sid = SignalId(uuid.UUID(signal_id))
    except ValueError:
        typer.echo(f"Error: invalid signal id {signal_id!r}", err=True)
        raise typer.Exit(1)

    async def _run():
        container = await build_container()
        use_case = GenerateCVUseCase(
            container.repository, container.cv_generator
        )
        return await use_case.execute(sid, profile)

    cv = asyncio.run(_run())

    lines = [
        f"# CV — {cv.target_signal_id}\n",
        f"*{cv.emphasis_rationale}*\n",
    ]
    for section in cv.sections:
        lines.append(f"## {section.title}\n\n{section.content}\n")
    output.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"✓ CV written to {output}")


def _load_profile(path: Path) -> DeveloperProfile:
    """Build a DeveloperProfile from a YAML file.

    User errors (missing required fields, malformed structure) exit with
    code 1 and a clear message — no stacktrace.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    missing = [
        field
        for field in ("full_name", "headline", "contact")
        if not data.get(field)
    ]
    if missing:
        typer.echo(
            f"Error: profile missing required field(s): {', '.join(missing)}",
            err=True,
        )
        raise typer.Exit(1)

    try:
        experiences = [
            Experience(
                role=exp.get("role", ""),
                company=exp.get("company", ""),
                achievements=list(exp.get("achievements") or []),
                skills_used=list(exp.get("skills_used") or []),
            )
            for exp in (data.get("experiences") or [])
        ]
        skills = [
            Skill(
                name=skill["name"],
                years=float(skill.get("years", 0)),
                level=skill.get("level", ""),
            )
            for skill in (data.get("skills") or [])
        ]
        projects = [
            Project(
                name=proj.get("name", ""),
                description=proj.get("description", ""),
                technologies=list(proj.get("technologies") or []),
                url=proj.get("url"),
            )
            for proj in (data.get("projects") or [])
        ]
        return DeveloperProfile(
            id=new_profile_id(),
            full_name=data["full_name"],
            headline=data["headline"],
            contact=data["contact"],
            experiences=experiences,
            skills=skills,
            projects=projects,
        )
    except (KeyError, ValueError, TypeError, ProfileIncompleteError) as e:
        typer.echo(f"Error: could not parse profile: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
