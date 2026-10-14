"""Load and validate OpenRouter settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from classifier.methods.llm.errors import LlmConfigError

# Repo root: classifier/methods/llm/config.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


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


@dataclass(frozen=True)
class LlmSettings:
    api_key: str
    model: str
    app_url: str | None
    app_title: str | None
    max_retries: int
    allow_fallback: bool
    strict: bool


def read_settings() -> LlmSettings:
    """Read settings after dotenv load. Does not validate completeness."""
    load_dotenv_files()
    retries_raw = os.getenv("LLM_MAX_RETRIES", "3").strip() or "3"
    try:
        max_retries = max(1, int(retries_raw))
    except ValueError:
        max_retries = 3
    return LlmSettings(
        api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        model=(os.getenv("OPENROUTER_MODEL") or "").strip(),
        app_url=(os.getenv("OPENROUTER_APP_URL") or "").strip() or None,
        app_title=(os.getenv("OPENROUTER_APP_TITLE") or "").strip() or None,
        max_retries=max_retries,
        allow_fallback=env_flag("LLM_FALLBACK", True),
        strict=env_flag("LLM_STRICT", False),
    )


def require_settings(settings: LlmSettings | None = None) -> LlmSettings:
    """Raise LlmConfigError if key/model are missing."""
    settings = settings or read_settings()
    missing: list[str] = []
    if not settings.api_key:
        missing.append("OPENROUTER_API_KEY")
    if not settings.model:
        missing.append("OPENROUTER_MODEL")
    if missing:
        raise LlmConfigError(
            "missing " + ", ".join(missing) + " (set them in .env; see .env.example)"
        )
    return settings


def require_langchain() -> None:
    """Raise LlmConfigError if LangChain / OpenRouter packages are not installed."""
    import warnings

    # langchain_core still imports pydantic.v1 for legacy models. On Python 3.14+
    # that emits a UserWarning; it is harmless for our Pydantic v2 schemas.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Pydantic V1 functionality isn't compatible with Python 3\.14.*",
            category=UserWarning,
            module=r"langchain_core\.utils\.pydantic",
        )
        try:
            import langchain  # noqa: F401
            import langchain_openrouter  # noqa: F401
        except ImportError as exc:
            raise LlmConfigError(
                "LangChain packages missing; run: pip install -r requirements.txt "
                f"({exc})"
            ) from exc
