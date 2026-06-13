"""Exception hierarchy for VenturePulseAI (see CONVENTIONS.md §3).

Two branches under a single root:

- DomainError: a business rule was violated. The input or state is wrong
  in a way that retrying will not fix.
- InfrastructureError: the outside world failed. Adapters translate raw
  SDK exceptions into these (`raise ... from e`); retrying may succeed.
"""


class VenturePulseError(Exception):
    """Root of every exception raised by this project.

    Never raised directly; catch it only at the outermost boundary
    (the API exception handler or a CLI entry point).
    """


# ── Domain: business rule violations ──────────────────────────────────


class DomainError(VenturePulseError):
    """A business invariant was violated. Maps to HTTP 422 at the API."""


class SignalValidationError(DomainError):
    """A signal lacks required business data (e.g. a FundingRound without
    a positive amount or a company name). Raised by entity invariants."""


class CVHallucinationError(DomainError):
    """A generated artifact contains claims that cannot be traced back to
    the DeveloperProfile. Raised by CV.validate_against(); triggers the
    retry-with-feedback loop (max 2) and maps to HTTP 502 if exhausted."""


class ProfileIncompleteError(DomainError):
    """The DeveloperProfile lacks the sections needed for the requested
    operation (e.g. generating a CV from a profile with no experience)."""


# ── Infrastructure: failures of the outside world ─────────────────────


class InfrastructureError(VenturePulseError):
    """An external dependency failed. Maps to HTTP 503 at the API.
    Adapters must raise a subclass of this, never the raw SDK exception."""


class ScrapingError(InfrastructureError):
    """A signal source failed: unreachable host, rate limiting (429),
    or markup that no longer matches the parser."""


class EmbeddingError(InfrastructureError):
    """The embedding provider failed to vectorize a text (API error,
    local model unavailable, input over the model limit)."""


class LLMError(InfrastructureError):
    """The LLM provider failed: API error, timeout, or an unparseable
    completion after exhausting adapter-level retries."""


class RepositoryError(InfrastructureError):
    """The vector store failed to persist or search (connection refused,
    missing collection, dimension mismatch with the collection schema)."""
