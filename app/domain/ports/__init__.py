"""Abstract ports of the domain. Pure stdlib + app.domain — no SDKs."""

from app.domain.ports.cv_generator import ICVGenerator
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.llm_service import FundingEntities, ILLMService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.ports.signal_scraper import ISignalScraper

__all__ = [
    "ISignalRepository",
    "SignalFilter",
    "ScoredSignal",
    "IEmbeddingService",
    "ISignalScraper",
    "ILLMService",
    "FundingEntities",
    "ICVGenerator",
]
