"""Run any registered classifier over an emails JSON file.

The emails path is an argument on purpose: the same executor can score the
sample, a hand-made synthetic set, or a future larger dump — as long as the
file is {"emails": [ Email, ... ]}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from classifier.log import log
from classifier.registry import available_methods, get_classifier
from classifier.types import Classification, Classifier, Email, Source, Topic


def load_emails(path: str | Path) -> list[Email]:
    """Load emails from any dataset that uses {"emails": [...]}."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "emails" not in payload:
        raise ValueError(f"{path} must be a JSON object with an 'emails' list")
    emails = payload["emails"]
    if not isinstance(emails, list):
        raise ValueError(f"{path}: 'emails' must be a list")
    return emails


def load_labels(path: str | Path) -> dict[str, tuple[Source, Topic]]:
    """Load provisional gold labels keyed by email id.

    Labels live in a separate file so the original sample stays untouched.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    labels: dict[str, tuple[Source, Topic]] = {}
    for row in payload["labels"]:
        labels[row["id"]] = (row["source"], row["topic"])
    return labels


def run(
    emails_path: str | Path,
    method: str = "rules",
    *,
    allow_fallback: bool | None = None,
) -> list[Classification]:
    """Classify every email in emails_path with the chosen method."""
    if allow_fallback is False:
        os.environ["LLM_FALLBACK"] = "0"
    emails = load_emails(emails_path)
    total = len(emails)
    log(f"[classifier] method={method} emails={total} path={emails_path}")
    classifier = get_classifier(method)
    results: list[Classification] = []
    for index, email in enumerate(emails, start=1):
        subject = (email.get("subject") or "(no subject)").replace("\n", " ")
        if len(subject) > 70:
            subject = subject[:67] + "..."
        log(f"[classifier] {index}/{total} start  id={email.get('id', '?')}  {subject}")
        result = classifier.classify(email)
        results.append(result)
        log(
            f"[classifier] {index}/{total} done   "
            f"method={result.method} source={result.source} topic={result.topic}"
        )
    _maybe_print_summary(classifier)
    return results


def run_as_dicts(
    emails_path: str | Path,
    method: str = "rules",
    *,
    allow_fallback: bool | None = None,
) -> list[dict[str, str]]:
    return [
        result.to_dict()
        for result in run(emails_path, method=method, allow_fallback=allow_fallback)
    ]


def _maybe_print_summary(classifier: Classifier) -> None:
    summary = getattr(classifier, "print_summary", None)
    if callable(summary):
        summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify accountant emails with a selectable method.",
    )
    parser.add_argument(
        "emails",
        type=Path,
        help="Path to a JSON file with an 'emails' list (sample or synthetic).",
    )
    parser.add_argument(
        "--method",
        default="rules",
        choices=available_methods(),
        help="Classification method to run (default: rules).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="For --method llm: fail instead of falling back to rules.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        results = run_as_dicts(
            args.emails,
            method=args.method,
            allow_fallback=False if args.no_fallback else None,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
