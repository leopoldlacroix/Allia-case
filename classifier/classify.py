"""Classify emails from a JSON file and print the result."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from classifier.source import classify_source
from classifier.topic import classify_topic


@dataclass(frozen=True)
class Classification:
    id: str
    source: str
    topic: str
    reason: str


def classify_email(email: dict) -> Classification:
    source, source_reason = classify_source(email)
    topic, topic_reason = classify_topic(email)
    return Classification(
        id=email["id"],
        source=source,
        topic=topic,
        reason=f"source: {source_reason}; topic: {topic_reason}",
    )


def load_emails(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["emails"]


def classify_file(path: str | Path) -> list[dict]:
    return [asdict(classify_email(email)) for email in load_emails(path)]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m classifier <emails.json>", file=sys.stderr)
        raise SystemExit(2)
    json.dump(classify_file(args[0]), sys.stdout, indent=2)
    sys.stdout.write("\n")
