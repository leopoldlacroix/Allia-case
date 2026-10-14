"""Unit tests for the systemone / Tev1 / TypeSafe decision path."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from classifier.methods.llm.config import LlmSettings, parse_api_mode
from classifier.methods.llm.errors import LlmConfigError, LlmInvokeError
from classifier.methods.llm.systemone import (
    classify_via_systemone,
    parse_systemone_answers,
    systemone_url,
)
from classifier.types import Email


def _email() -> Email:
    return {
        "id": "sys1",
        "subject": "Q4 Financial Statements",
        "body": "Please review depreciation.",
        "participants": [
            {"email": "michael.chen@smithmanufacturing.com", "type": "from"}
        ],
        "attachments": [],
    }


def _settings(**overrides: object) -> LlmSettings:
    base = dict(
        api_key="ollama",
        model="tev1",
        base_url="http://localhost:11434/v1",
        api_mode="systemone",
        app_url=None,
        app_title=None,
        max_retries=1,
        allow_fallback=True,
        strict=False,
    )
    base.update(overrides)
    return LlmSettings(**base)  # type: ignore[arg-type]


class SystemoneUnitTests(unittest.TestCase):
    def test_url_appends_systemone(self) -> None:
        self.assertEqual(
            systemone_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1/systemone",
        )
        self.assertEqual(
            systemone_url("http://localhost:11434/v1/systemone"),
            "http://localhost:11434/v1/systemone",
        )

    def test_parse_api_mode_aliases(self) -> None:
        self.assertEqual(parse_api_mode("chat"), "chat")
        self.assertEqual(parse_api_mode("systemone"), "systemone")
        self.assertEqual(parse_api_mode("tev1"), "systemone")
        self.assertEqual(parse_api_mode("jev"), "systemone")
        with self.assertRaises(LlmConfigError):
            parse_api_mode("nope")

    def test_parse_answers(self) -> None:
        source, topic, reason = parse_systemone_answers(
            {
                "answers": {
                    "source": {"choice": "client", "confidence": 0.9},
                    "topic": {"choice": "accounting", "confidence": 0.8},
                }
            }
        )
        self.assertEqual(source, "client")
        self.assertEqual(topic, "accounting")
        self.assertIn("confidence", reason)

    def test_parse_answers_missing_raises(self) -> None:
        with self.assertRaises(LlmInvokeError):
            parse_systemone_answers({"answers": {"source": {"choice": "client"}}})

    def test_classify_posts_and_maps(self) -> None:
        fake = MagicMock()
        fake.raise_for_status = MagicMock()
        fake.json.return_value = {
            "answers": {
                "source": {"choice": "client"},
                "topic": {"choice": "accounting"},
            }
        }
        with patch("httpx.post", return_value=fake) as post:
            result = classify_via_systemone(_email(), _settings())
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.source, "client")
        self.assertEqual(result.topic, "accounting")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://localhost:11434/v1/systemone")
        body = kwargs["json"]
        self.assertEqual(body["model"], "tev1")
        self.assertIn("source", body["questions"])
        self.assertIn("topic", body["questions"])
        self.assertEqual(body["state"]["from"], "michael.chen@smithmanufacturing.com")


if __name__ == "__main__":
    unittest.main()
