import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from classifier.classify import classify_email, classify_file, load_emails, main

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_accountant_emails.json"

# Hand labels for the 20 sample emails, in file order.
# Source follows the From address. Topic needs a subject keyword or two
# different message keywords; one message keyword stays "other".
EXPECTED = [
    ("client", "accounting"),
    ("internal", "taxes"),
    ("internal", "accounting"),
    ("external", "other"),
    ("external", "other"),
    ("internal", "accounting"),
    ("external", "taxes"),
    ("client", "taxes"),
    ("external", "accounting"),
    ("internal", "other"),
    ("internal", "accounting"),
    ("internal", "taxes"),
    ("internal", "accounting"),
    ("client", "other"),
    ("internal", "taxes"),
    ("external", "taxes"),
    ("internal", "other"),
    ("external", "accounting"),
    ("internal", "other"),
    ("internal", "other"),
]


def _email(subject: str, body: str, sender: str) -> dict:
    return {
        "id": "synthetic",
        "subject": subject,
        "body": body,
        "participants": [{"email": sender, "type": "from"}],
        "attachments": [],
    }


class SampleTests(unittest.TestCase):
    def test_sample_labels(self):
        emails = load_emails(SAMPLE)
        self.assertEqual(len(emails), len(EXPECTED))
        for email, expected in zip(emails, EXPECTED):
            result = classify_email(email)
            self.assertEqual(
                (result.source, result.topic),
                expected,
                email["subject"],
            )
            self.assertTrue(result.reason)

    def test_cli_prints_classifications(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([str(SAMPLE)])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(len(payload), 20)
        self.assertEqual(set(payload[0]), {"id", "source", "topic", "reason"})
        self.assertEqual(
            [(row["source"], row["topic"]) for row in payload],
            EXPECTED,
        )
        self.assertEqual(classify_file(SAMPLE), payload)


class RuleTests(unittest.TestCase):
    def test_repeated_audit_in_the_body_stays_other(self):
        email = _email(
            "Thanks",
            "The audit went well. Really, the audit went well.",
            "david.rodriguez@accountingfirm.com",
        )
        result = classify_email(email)
        self.assertEqual((result.source, result.topic), ("internal", "other"))

    def test_spam_phrase_from_a_normal_address_is_external(self):
        email = _email(
            "Hello",
            "Claim your prize today.",
            "friend@gmail.com",
        )
        result = classify_email(email)
        self.assertEqual(result.source, "external")
        self.assertIn("claim your prize", result.reason)

    def test_payroll_and_legal_keywords(self):
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

    def test_tie_goes_to_the_more_specific_topic(self):
        email = _email(
            "Payroll tax notice",
            "Please read.",
            "cfo@client.com",
        )
        result = classify_email(email)
        self.assertEqual(result.topic, "payroll")


if __name__ == "__main__":
    unittest.main()
