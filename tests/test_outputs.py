"""Tests for run artifacts and the compare dashboard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from classifier.compare import consensus_vs_label, render_compare, score_run, write_compare
from classifier.executor import run
from classifier.outputs import RunArtifact, load_run, save_run
from classifier.types import Classification

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_accountant_emails.json"


def _result(email_id: str, source: str, topic: str) -> Classification:
    return Classification(
        id=email_id,
        source=source,  # type: ignore[arg-type]
        topic=topic,  # type: ignore[arg-type]
        reason="test",
        method="rules",
    )


class OutputSaveTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results = run(SAMPLE, method="rules", save=False)
            path = save_run(
                results,
                method="rules",
                emails_path=SAMPLE,
                tag="unit",
                settings={},
                outputs_dir=out_dir,
            )
            self.assertTrue(path.is_file())
            loaded = load_run(path)
            self.assertEqual(loaded.method, "rules")
            self.assertEqual(loaded.tag, "unit")
            self.assertEqual(len(loaded.results), len(results))
            self.assertEqual(loaded.results[0].id, results[0].id)


class CompareTests(unittest.TestCase):
    def test_consensus_vs_label(self) -> None:
        self.assertEqual(
            consensus_vs_label(
                [("client", "other"), ("client", "other"), ("client", "accounting")],
                ("client", "accounting"),
            ),
            ("client", "other"),
        )
        self.assertIsNone(
            consensus_vs_label(
                [("client", "accounting"), ("client", "accounting")],
                ("client", "accounting"),
            )
        )

    def test_scores_and_html_flag_label_check(self) -> None:
        emails = [
            {
                "id": "e1",
                "subject": "Hello",
                "body": "Body text for inspection.",
                "participants": [{"email": "a@b.com", "type": "from"}],
                "attachments": [],
            }
        ]
        labels = {"e1": ("client", "accounting")}
        runs = [
            RunArtifact(
                id="run_a",
                created_at="t",
                method="llm",
                emails_path="x",
                results=[_result("e1", "client", "other")],
                tag="a",
                settings={"api_mode": "chat", "model": "m1", "base_url": "http://x"},
            ),
            RunArtifact(
                id="run_b",
                created_at="t",
                method="llm",
                emails_path="x",
                results=[_result("e1", "client", "other")],
                tag="b",
                settings={
                    "api_mode": "systemone",
                    "model": "tev1",
                    "base_url": "http://x",
                },
            ),
        ]
        score = score_run(runs[0], labels)
        self.assertEqual(score["both"], 0)
        self.assertEqual(score["source"], 1)

        html_doc = render_compare(emails, runs, labels, "emails.json")
        self.assertIn("label check?", html_doc)
        self.assertIn("Body text for inspection.", html_doc)
        self.assertIn("Both", html_doc)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for run_art in runs:
                save_run(
                    run_art.results,
                    method=run_art.method,
                    emails_path=SAMPLE,
                    tag=run_art.tag,
                    settings=run_art.settings,
                    outputs_dir=out_dir,
                )
            emails_path = out_dir / "emails.json"
            emails_path.write_text(json.dumps({"emails": emails}), encoding="utf-8")
            labels_path = out_dir / "labels.json"
            labels_path.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "id": "e1",
                                "source": "client",
                                "topic": "accounting",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            html_path = out_dir / "compare.html"
            write_compare(
                emails_path,
                html_path,
                labels_path=labels_path,
                outputs_dir=out_dir,
            )
            text = html_path.read_text(encoding="utf-8")
            self.assertIn("label check?", text)


if __name__ == "__main__":
    unittest.main()
