"""Load LLM settings from the environment (chat or systemone providers)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from classifier.methods.llm.errors import LlmConfigError

# Repo root: classifier/methods/llm/config.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Used only when LLM_BASE_URL is unset. Override per provider in .env.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

ApiMode = Literal["chat", "systemone"]


def load_dotenv_files() -> None:
    """Load `.env` from the repo root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def parse_api_mode(raw: str | None = None) -> ApiMode:
    value = (raw if raw is not None else _env("LLM_API_MODE", "chat")).lower()
    # Aliases for the TypeSafe / Jev / Tev1 decision API.
    if value in {"systemone", "jev", "tev", "tev1", "typesafe", "decision"}:
        return "systemone"
    if value in {"chat", "openai", "openai_chat"}:
        return "chat"
    raise LlmConfigError(
        f"unknown LLM_API_MODE={value!r}; use 'chat' or 'systemone'"
    )


@dataclass(frozen=True)
class LlmSettings:
    api_key: str
    model: str
    base_url: str
    api_mode: ApiMode
    app_url: str | None
    app_title: str | None
    max_retries: int
    allow_fallback: bool
    strict: bool


def read_settings() -> LlmSettings:
    """Read settings after dotenv load. Does not validate completeness."""
    load_dotenv_files()
    retries_raw = _env("LLM_MAX_RETRIES", "3") or "3"
    try:
        max_retries = max(1, int(retries_raw))
    except ValueError:
        max_retries = 3
    base_url = _env("LLM_BASE_URL") or DEFAULT_BASE_URL
    return LlmSettings(
        api_key=_env("LLM_API_KEY"),
        model=_env("LLM_MODEL"),
        base_url=base_url.rstrip("/"),
        api_mode=parse_api_mode(),
        app_url=_env("LLM_APP_URL") or None,
        app_title=_env("LLM_APP_TITLE") or None,
        max_retries=max_retries,
        allow_fallback=env_flag("LLM_FALLBACK", True),
        strict=env_flag("LLM_STRICT", False),
    )


def require_settings(settings: LlmSettings | None = None) -> LlmSettings:
    """Raise LlmConfigError if key/model are missing."""
    settings = settings or read_settings()
    missing: list[str] = []
    if not settings.api_key:
        missing.append("LLM_API_KEY")
    if not settings.model:
        missing.append("LLM_MODEL")
    if missing:
        raise LlmConfigError(
            "missing " + ", ".join(missing) + " (set them in .env; see .env.example)"
        )
    return settings


def _silence_python314_langchain_noise() -> None:
    """Filter known upstream warnings on Python 3.14 (filters stay process-wide)."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*Pydantic V1 functionality isn't compatible with Python 3\.14.*",
        category=UserWarning,
        module=r"langchain_core\.utils\.pydantic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*asyncio\.iscoroutinefunction.*",
        category=DeprecationWarning,
        module=r"langsmith\._internal\._beta_decorator",
    )


def require_langchain() -> None:
    """Raise LlmConfigError if LangChain / OpenAI chat deps are missing (chat mode)."""
    _silence_python314_langchain_noise()
    try:
        import langchain  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError as exc:
        raise LlmConfigError(
            "LangChain packages missing; run: pip install -r requirements.txt "
            f"({exc})"
        ) from exc
