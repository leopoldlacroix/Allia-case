"""OpenRouter + LangChain agent classifier (method name: "llm")."""

from __future__ import annotations

import random
import sys
import time
from typing import Any

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
    """Agent over OpenRouter. Falls back to rules on config / network failures."""

    name = "llm"

    def __init__(self, settings: LlmSettings | None = None) -> None:
        self._settings = settings or read_settings()
        self._agent: Any | None = None
        self._agent_error: str | None = None
        self.llm_successes = 0
        self.fallback_count = 0
        self._prepare_agent()

    def _prepare_agent(self) -> None:
        """Validate config and build the agent once. Defer failures until classify()."""
        try:
            # Settings first so a missing key falls back without requiring LangChain.
            self._settings = require_settings(self._settings)
            require_langchain()
            from classifier.methods.llm.agent import build_agent

            self._agent = build_agent(self._settings)
            self._agent_error = None
        except LlmConfigError as exc:
            self._agent = None
            self._agent_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — surface unexpected build errors
            self._agent = None
            self._agent_error = f"agent build failed: {exc}"

    def classify(self, email: Email) -> Classification:
        if self._agent is None:
            return self._handle_failure(email, self._agent_error or "agent unavailable")

        last_error: str | None = None
        attempts = self._settings.max_retries
        for attempt in range(1, attempts + 1):
            try:
                result = self._invoke_once(email)
                self.llm_successes += 1
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts and is_retryable(exc):
                    self._sleep_backoff(attempt)
                    continue
                if is_retryable(exc) or attempt >= attempts:
                    return self._handle_failure(email, last_error)
                return self._handle_failure(email, last_error)

        return self._handle_failure(email, last_error or "unknown error")

    def _invoke_once(self, email: Email) -> Classification:
        from classifier.methods.llm.agent import email_user_message, extract_structured_result

        assert self._agent is not None
        state = self._agent.invoke(
            {"messages": [{"role": "user", "content": email_user_message(email)}]}
        )
        structured = extract_structured_result(state)
        if structured is None:
            raise LlmInvokeError("agent returned no structured ClassificationResult")
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
        print(
            f"[llm] fallback for {email.get('id', '?')}: {cause}",
            file=sys.stderr,
        )
        return rules_fallback(email, cause)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        # attempt 1 → ~1s, attempt 2 → ~2s, plus jitter
        delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
        time.sleep(delay)

    def print_summary(self) -> None:
        total = self.llm_successes + self.fallback_count
        if total == 0:
            return
        print(
            f"[llm] done: {self.llm_successes} llm, {self.fallback_count} rules-fallback "
            f"(of {total})",
            file=sys.stderr,
        )


def classify_email(email: Email) -> Classification:
    return LlmClassifier().classify(email)
