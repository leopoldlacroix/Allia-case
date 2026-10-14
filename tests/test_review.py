"""Smoke tests for the HTML review page."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from classifier.review import write_review

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_accountant_emails.json"
EVAL_LABELS = ROOT / "data" / "eval_labels.json"


class ReviewTests(unittest.TestCase):
    def test_writes_html_with_match_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "review.html"
            write_review(
                SAMPLE,
                out,
                method="rules",
                labels_path=EVAL_LABELS,
                save=False,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", text)
            self.assertIn("eval match 20/20", text)
            self.assertIn("Q4 Financial Statements", text)
            self.assertIn("class=\"badge ok\"", text)


if __name__ == "__main__":
    unittest.main()
