"""Unit tests for LLM helper tools (no API key required)."""

from __future__ import annotations

import json
import unittest

from classifier.methods.llm.tools import (
    check_bulk_or_spam_signals,
    get_sender_info,
    score_topic_keywords,
)


class HelperToolTests(unittest.TestCase):
    def test_get_sender_info_detects_firm(self) -> None:
        payload = json.loads(
            get_sender_info("sarah.johnson@accountingfirm.com")
        )
        self.assertTrue(payload["is_firm"])
        self.assertEqual(payload["domain"], "accountingfirm.com")

    def test_bulk_and_spam_signals(self) -> None:
        payload = json.loads(
            check_bulk_or_spam_signals(
                "noreply@vendor.com",
                "Hello",
                "Claim your prize today",
            )
        )
        self.assertTrue(payload["is_bulk_local_part"])
        self.assertIn("claim your prize", payload["spam_phrase_hits"])
        self.assertTrue(payload["looks_external_bulk_or_spam"])

    def test_topic_keyword_scores(self) -> None:
        payload = json.loads(
            score_topic_keywords(
                "Q4 Financial Statements",
                "Please review the depreciation schedule.",
                "Depreciation_Schedule.xlsx",
            )
        )
        self.assertGreaterEqual(payload["scores"]["accounting"], 2)
        self.assertIn("accounting", payload["above_threshold"])


if __name__ == "__main__":
    unittest.main()
