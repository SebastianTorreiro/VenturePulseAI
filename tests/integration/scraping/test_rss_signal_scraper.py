"""Tests for RSSSignalScraper.

source_name() is pure (no network) and runs in the default suite. The
fetch() tests hit the real TechCrunch feed, so they are marked
`integration` and excluded from the default `pytest` run; run with
`pytest -m integration`. If the network is down they fail — that is the
expected behavior of real integration tests, not a bug.

Async fetch is driven with asyncio.run() so the httpx AsyncClient lives
on one event loop — no pytest-asyncio plugin needed.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.exceptions import ScrapingError
from app.infrastructure.config.settings import ScraperSettings
from app.infrastructure.scraping.rss_signal_scraper import RSSSignalScraper


def test_source_name_returns_expected_slug():
    scraper = RSSSignalScraper(ScraperSettings())

    assert scraper.source_name() == "techcrunch-rss"


@pytest.mark.integration
def test_fetch_returns_at_least_one_raw_signal():
    async def scenario():
        scraper = RSSSignalScraper(ScraperSettings())
        since = datetime.now(timezone.utc) - timedelta(days=30)

        signals = [signal async for signal in scraper.fetch(since)]

        assert len(signals) >= 1
        assert all(signal.content.strip() for signal in signals)

    asyncio.run(scenario())


@pytest.mark.integration
def test_fetch_filters_signals_older_than_since():
    async def scenario():
        scraper = RSSSignalScraper(ScraperSettings())
        a_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
        in_the_future = datetime.now(timezone.utc) + timedelta(days=1)

        from_last_year = [s async for s in scraper.fetch(a_year_ago)]
        from_future = [s async for s in scraper.fetch(in_the_future)]

        assert len(from_last_year) >= len(from_future)
        assert len(from_future) == 0

    asyncio.run(scenario())


@pytest.mark.integration
def test_fetch_returns_signals_with_required_fields():
    async def scenario():
        scraper = RSSSignalScraper(ScraperSettings())
        since = datetime.now(timezone.utc) - timedelta(days=30)

        signals = [signal async for signal in scraper.fetch(since)]

        assert signals
        for signal in signals:
            assert signal.source == "techcrunch-rss"
            assert signal.url.strip() != ""
            assert signal.content.strip() != ""
            assert signal.fetched_at.tzinfo is not None
            assert signal.fetched_at.utcoffset() is not None

    asyncio.run(scenario())


@pytest.mark.integration
def test_fetch_raises_scraping_error_on_network_failure():
    async def scenario():
        settings = ScraperSettings(
            rss_feed_url="https://this-domain-does-not-exist-12345.invalid/feed"
        )
        scraper = RSSSignalScraper(settings)
        since = datetime.now(timezone.utc) - timedelta(days=30)

        with pytest.raises(ScrapingError):
            async for _ in scraper.fetch(since):
                pass

    asyncio.run(scenario())
