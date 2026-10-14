"""LLM classifier (method name: "llm"): OpenAI chat agent or systemone decisions."""

from __future__ import annotations

import random
import time
from typing import Any

from classifier.log import log
from classifier.methods.llm.config import (
    LlmSettings,
    read_settings,
    require_langchain,
    require_settings,
)
from classifier.methods.llm.errors import LlmConfigError, LlmInvokeError, is_retryable
from classifier.methods.llm.fallback import rules_fallback
from classifier.types import Classification, Email


class LlmClassifier:
    """Remote model classifier. Falls back to rules on config / network failures."""

    name = "llm"

    def __init__(self, settings: LlmSettings | None = None) -> None:
        self._settings = settings or read_settings()
        self._agent: Any | None = None
        self._ready = False
        self._setup_error: str | None = None
        self.llm_successes = 0
        self.fallback_count = 0
        self._prepare()

    def _prepare(self) -> None:
        """Validate config; build chat agent only when LLM_API_MODE=chat."""
        try:
            self._settings = require_settings(self._settings)
            if self._settings.api_mode == "systemone":
                self._ready = True
                self._setup_error = None
                log(
                    f"[llm] systemone mode model={self._settings.model} "
                    f"base_url={self._settings.base_url}"
                )
                return

            require_langchain()
            log(
                f"[llm] building chat agent model={self._settings.model} "
                f"base_url={self._settings.base_url} "
                f"retries={self._settings.max_retries}"
            )
            from classifier.methods.llm.agent import build_agent

            self._agent = build_agent(self._settings)
            self._ready = True
            self._setup_error = None
            log("[llm] chat agent ready")
        except LlmConfigError as exc:
            self._ready = False
            self._agent = None
            self._setup_error = str(exc)
            log(f"[llm] not available: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._ready = False
            self._agent = None
            self._setup_error = f"setup failed: {exc}"
            log(f"[llm] setup failed: {exc}")

    def classify(self, email: Email) -> Classification:
        email_id = email.get("id", "?")
        if not self._ready:
            return self._handle_failure(email, self._setup_error or "llm unavailable")

        last_error: str | None = None
        attempts = self._settings.max_retries
        for attempt in range(1, attempts + 1):
            try:
                mode = self._settings.api_mode
                log(f"[llm] invoke id={email_id} mode={mode} attempt={attempt}/{attempts} …")
                started = time.monotonic()
                result = self._invoke_once(email)
                elapsed = time.monotonic() - started
                self.llm_successes += 1
                log(
                    f"[llm] ok id={email_id} "
                    f"source={result.source} topic={result.topic} ({elapsed:.1f}s)"
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                log(f"[llm] error id={email_id} attempt={attempt}/{attempts}: {last_error}")
                if attempt < attempts and is_retryable(exc):
                    delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    log(f"[llm] retrying in {delay:.1f}s …")
                    time.sleep(delay)
                    continue
                return self._handle_failure(email, last_error)

        return self._handle_failure(email, last_error or "unknown error")

    def _invoke_once(self, email: Email) -> Classification:
        if self._settings.api_mode == "systemone":
            from classifier.methods.llm.systemone import classify_via_systemone

            return classify_via_systemone(email, self._settings)

        from classifier.methods.llm.agent import extract_structured_result, invoke_agent

        assert self._agent is not None
        state = invoke_agent(self._agent, email)
        structured = extract_structured_result(state)
        if structured is None:
            keys = list(state.keys()) if isinstance(state, dict) else type(state)
            preview = ""
            if isinstance(state, dict):
                preview = f" structured_response={state.get('structured_response')!r}"[:300]
                messages = state.get("messages") or []
                if messages:
                    last = messages[-1]
                    preview += (
                        f" last_message={type(last).__name__}"
                        f" content={getattr(last, 'content', None)!r}"[:200]
                    )
            raise LlmInvokeError(
                f"agent returned no structured ClassificationResult (keys={keys}){preview}"
            )
        reason = (structured.reason or "").strip() or (
            f"llm labels source={structured.source}, topic={structured.topic}"
        )
        return Classification(
            id=email["id"],
            source=structured.source,  # type: ignore[arg-type]
            topic=structured.topic,  # type: ignore[arg-type]
            reason=reason,
            method=self.name,
        )

    def _handle_failure(self, email: Email, cause: str) -> Classification:
        if self._settings.strict or not self._settings.allow_fallback:
            raise LlmConfigError(cause) if "missing" in cause or "packages" in cause else LlmInvokeError(
                cause
            )
        self.fallback_count += 1
        log(f"[llm] fallback id={email.get('id', '?')}: {cause}")
        return rules_fallback(email, cause)

    def print_summary(self) -> None:
        total = self.llm_successes + self.fallback_count
        if total == 0:
            return
        log(
            f"[llm] summary: {self.llm_successes} llm, "
            f"{self.fallback_count} rules-fallback (of {total})"
        )


def classify_email(email: Email) -> Classification:
    return LlmClassifier().classify(email)
