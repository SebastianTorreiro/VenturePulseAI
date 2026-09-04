"""Port: external source -> raw signals. One implementation per source."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal

from app.domain.entities.signal import RawSignal

SignalKind = Literal["funding_round", "job_offer"]


class ISignalScraper(ABC):
    """Fetches raw, un-enriched signals from a single external source."""

    @property
    @abstractmethod
    def signal_type(self) -> SignalKind:
        """Which Signal subtype this source's raw output should become.

        IngestSignalsUseCase reads this to pick the right ILLMService
        extraction method and Signal.from_extraction() classmethod for
        everything this scraper yields — one scraper, one signal kind.
        """
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Stable slug identifying the source (e.g. "techcrunch-rss").

        Not async: this is the adapter's static identity, not I/O. Used
        to tag every RawSignal and for per-source config/logging.
        """
        ...

    @abstractmethod
    def fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        """Stream raw signals detected at or after `since`.

        Yields RawSignal items lazily so the use case can process and
        deduplicate without holding the whole source in memory. The
        consumer decides what to do with each item. Implemented as an
        `async def` with `yield` (an async generator); declared here as
        a plain method returning AsyncIterator because async generators
        are not compatible with @abstractmethod's coroutine detection.

        Raises:
            ScrapingError: the source failed (unreachable host, 429,
                markup no longer matching the parser).
        """
        ...
