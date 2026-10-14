"""Errors and retry classification for the OpenRouter LLM method."""

from __future__ import annotations


class LlmConfigError(RuntimeError):
    """Missing API key, model, or LangChain packages — fix env / requirements."""


class LlmInvokeError(RuntimeError):
    """The agent call failed after retries or returned nothing usable."""


def is_retryable(exc: BaseException) -> bool:
    """Return True for transient network / rate-limit style failures."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "timeout",
        "timed out",
        "rate limit",
        "rate_limit",
        "429",
        "503",
        "502",
        "504",
        "connection",
        "connect",
        "temporarily unavailable",
        "overloaded",
        "retry",
    )
    if any(marker in name for marker in ("timeout", "connection", "ratelimit")):
        return True
    return any(marker in text for marker in markers)
