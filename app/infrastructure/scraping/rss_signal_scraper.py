"""ISignalScraper implementation against RSS feeds (httpx + feedparser)."""

import calendar
import html
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from urllib.parse import urlsplit

import feedparser
import httpx

from app.domain.entities.signal import RawSignal
from app.domain.exceptions import ScrapingError
from app.domain.ports.signal_scraper import ISignalScraper
from app.infrastructure.config.settings import ScraperSettings

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


class RSSSignalScraper(ISignalScraper):
    """Fetches raw signals from a single RSS feed.

    No async factory: construction only stores settings; all I/O happens
    in fetch(). RawSignal carries no business invariants — structural
    validation belongs to the application boundary (ADR-005).
    """

    def __init__(self, settings: ScraperSettings) -> None:
        self._settings = settings

    def source_name(self) -> str:
        host = urlsplit(str(self._settings.rss_feed_url)).hostname or ""
        label = host.split(".")[0] if host else "feed"
        return f"{label}-rss"

    def fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        # Plain method returning an async iterator (the port contract):
        # calling it hands back the async generator without an await.
        return self._fetch_impl(since)

    async def _fetch_impl(self, since: datetime) -> AsyncIterator[RawSignal]:
        timeout = self._settings.fetch_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(str(self._settings.rss_feed_url))
                response.raise_for_status()
            except Exception as e:
                raise ScrapingError(
                    f"Failed to fetch RSS feed {self._settings.rss_feed_url}"
                ) from e

            # feedparser is sync but fast and does not raise on bad markup
            # (it sets a bozo flag); no to_thread needed.
            parsed = feedparser.parse(response.content)

            for entry in parsed.entries:
                published = self._parse_pubdate(entry)
                if published is None or published < since:
                    continue
                yield RawSignal(
                    source=self.source_name(),
                    url=entry.get("link", ""),
                    content=self._extract_content(entry),
                    fetched_at=datetime.now(timezone.utc),
                )

    @staticmethod
    def _parse_pubdate(entry) -> datetime | None:
        struct = entry.get("published_parsed")
        if struct is None:
            return None
        try:
            # feedparser normalizes published_parsed to UTC; timegm reads
            # the struct as UTC (mktime would wrongly apply local time).
            epoch = calendar.timegm(struct)
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        no_tags = _TAG_RE.sub(" ", text)
        return re.sub(r"\s+", " ", no_tags).strip()

    @staticmethod
    def _extract_content(entry) -> str:
        title = RSSSignalScraper._strip_html(
            html.unescape(entry.get("title", "") or "")
        )
        raw_body = (
            entry.get("summary")
            or entry.get("description")
            or entry.get("title")
            or ""
        )
        body = RSSSignalScraper._strip_html(html.unescape(raw_body))
        parts = [part for part in (title, body) if part]
        # Avoid duplicating when the body fell back to the title.
        if len(parts) == 2 and parts[0] == parts[1]:
            parts = parts[:1]
        return "\n\n".join(parts)
