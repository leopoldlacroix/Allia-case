"""Unit tests for rules, registry, and the path-flexible executor.

Eval labels live in data/eval_labels.json so the original sample is never edited
to match a method. Rule edge cases use tiny synthetic emails built in-memory.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from classifier.executor import load_emails, load_labels, main, run
from classifier.methods.rules import classify_email
from classifier.registry import available_methods, get_classifier
from classifier.types import Email

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_accountant_emails.json"
EVAL_LABELS = ROOT / "data" / "eval_labels.json"


def _email(subject: str, body: str, sender: str) -> Email:
    """Minimal Email that still satisfies the typed input contract."""
    return {
        "id": "synthetic",
        "subject": subject,
        "body": body,
        "participants": [{"email": sender, "type": "from"}],
        "attachments": [],
    }


class EvalSetTests(unittest.TestCase):
    def test_rules_match_eval_labels(self) -> None:
        emails = load_emails(SAMPLE)
        labels = load_labels(EVAL_LABELS)
        self.assertEqual(len(emails), len(labels))
        for email in emails:
            expected = labels[email["id"]]
            result = classify_email(email)
            self.assertEqual(
                (result.source, result.topic),
                expected,
                email["subject"],
            )
            self.assertEqual(result.method, "rules")
            self.assertTrue(result.reason)

    def test_executor_accepts_any_emails_path(self) -> None:
        """Synthetic datasets use the same {"emails": [...]} shape."""
        payload = {
            "emails": [
                _email(
                    "March payroll",
                    "Please process salaries.",
                    "cfo@client.com",
                )
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            results = run(path, method="rules", save=False)
        finally:
            path.unlink()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "client")
        self.assertEqual(results[0].topic, "payroll")

    def test_cli_prints_classifications(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([str(SAMPLE), "--method", "rules", "--no-save"])
        payload = json.loads(buffer.getvalue())
        labels = load_labels(EVAL_LABELS)
        self.assertEqual(len(payload), 20)
        self.assertEqual(
            set(payload[0]),
            {"id", "source", "topic", "reason", "method"},
        )
        for row in payload:
            self.assertEqual(
                (row["source"], row["topic"]),
                labels[row["id"]],
                row["id"],
            )
            self.assertEqual(row["method"], "rules")


class RegistryTests(unittest.TestCase):
    def test_rules_is_registered(self) -> None:
        self.assertIn("rules", available_methods())
        classifier = get_classifier("rules")
        self.assertEqual(classifier.name, "rules")

    def test_llm_is_registered(self) -> None:
        self.assertIn("llm", available_methods())

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_classifier("does-not-exist")


class RuleTests(unittest.TestCase):
    def test_repeated_audit_in_the_body_stays_other(self) -> None:
        # Same keyword twice in the body still scores 1 → below threshold.
        email = _email(
            "Thanks",
            "The audit went well. Really, the audit went well.",
            "david.rodriguez@accountingfirm.com",
        )
        result = classify_email(email)
        self.assertEqual((result.source, result.topic), ("internal", "other"))

    def test_spam_phrase_from_a_normal_address_is_external(self) -> None:
        email = _email(
            "Hello",
            "Claim your prize today.",
            "friend@gmail.com",
        )
        result = classify_email(email)
        self.assertEqual(result.source, "external")
        self.assertIn("claim your prize", result.reason)

    def test_payroll_and_legal_keywords(self) -> None:
        payroll = classify_email(
            _email("March payroll", "See attached.", "cfo@client.com")
        )
        legal = classify_email(
            _email(
                "Lawsuit filed yesterday",
                "Our attorney will call.",
                "lisa.thompson@accountingfirm.com",
            )
        )
        self.assertEqual((payroll.source, payroll.topic), ("client", "payroll"))
        self.assertEqual((legal.source, legal.topic), ("internal", "legal"))

    def test_tie_goes_to_the_more_specific_topic(self) -> None:
        # Subject hits both payroll and taxes; payroll is listed first → wins.
        email = _email(
            "Payroll tax notice",
            "Please read.",
            "cfo@client.com",
        )
        result = classify_email(email)
        self.assertEqual(result.topic, "payroll")


if __name__ == "__main__":
    unittest.main()
