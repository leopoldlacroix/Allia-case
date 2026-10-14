"""Fail-safe tests for the LLM method (no live OpenRouter calls)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from classifier.methods.llm import LlmClassifier
from classifier.methods.llm.config import LlmSettings
from classifier.methods.llm.errors import LlmInvokeError, is_retryable
from classifier.types import Email


def _email() -> Email:
    return {
        "id": "fallback-test",
        "subject": "Q4 Financial Statements",
        "body": "Please review depreciation and the P&L.",
        "participants": [
            {"email": "michael.chen@smithmanufacturing.com", "type": "from"}
        ],
        "attachments": [],
    }


def _settings(**overrides: object) -> LlmSettings:
    base = dict(
        api_key="test-key",
        model="test/model",
        app_url=None,
        app_title=None,
        max_retries=2,
        allow_fallback=True,
        strict=False,
    )
    base.update(overrides)
    return LlmSettings(**base)  # type: ignore[arg-type]


class RetryableTests(unittest.TestCase):
    def test_rate_limit_is_retryable(self) -> None:
        self.assertTrue(is_retryable(RuntimeError("Error 429 rate limit")))
        self.assertFalse(is_retryable(ValueError("bad schema")))


class FallbackTests(unittest.TestCase):
    def test_missing_api_key_falls_back_to_rules(self) -> None:
        classifier = LlmClassifier(
            _settings(api_key="", model="dots-studio/dots-3-note-preview:free")
        )
        result = classifier.classify(_email())
        self.assertEqual(result.method, "rules")
        self.assertIn("llm unavailable", result.reason)
        self.assertEqual(result.source, "client")
        self.assertEqual(result.topic, "accounting")
        self.assertEqual(classifier.fallback_count, 1)

    def test_missing_key_raises_when_fallback_disabled(self) -> None:
        classifier = LlmClassifier(
            _settings(api_key="", allow_fallback=False)
        )
        with self.assertRaises(Exception):
            classifier.classify(_email())

    def test_invoke_error_retries_then_falls_back(self) -> None:
        classifier = LlmClassifier(_settings(max_retries=2))
        classifier._agent = object()  # pretend agent is ready
        classifier._agent_error = None

        with patch.object(
            classifier,
            "_invoke_once",
            side_effect=ConnectionError("connection reset"),
        ), patch.object(classifier, "_sleep_backoff"):
            result = classifier.classify(_email())

        self.assertEqual(result.method, "rules")
        self.assertIn("ConnectionError", result.reason)
        self.assertEqual(classifier.fallback_count, 1)

    def test_non_retryable_falls_back_without_looping_forever(self) -> None:
        classifier = LlmClassifier(_settings(max_retries=5))
        classifier._agent = object()
        classifier._agent_error = None

        with patch.object(
            classifier,
            "_invoke_once",
            side_effect=LlmInvokeError("agent returned no structured ClassificationResult"),
        ) as invoke, patch.object(classifier, "_sleep_backoff") as sleep:
            result = classifier.classify(_email())

        self.assertEqual(result.method, "rules")
        self.assertEqual(invoke.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
