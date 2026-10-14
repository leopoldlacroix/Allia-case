"""Deterministic address + keyword rules (method name: "rules")."""

from __future__ import annotations

from classifier.methods.rules.source import classify_source
from classifier.methods.rules.topic import classify_topic
from classifier.types import Classification, Email


class RulesClassifier:
    """Inspectable rules: no model, no API key, every decision has a reason string."""

    name = "rules"

    def classify(self, email: Email) -> Classification:
        source, source_reason = classify_source(email)
        topic, topic_reason = classify_topic(email)
        return Classification(
            id=email["id"],
            source=source,
            topic=topic,
            # Both axes in one string so the CLI output is self-explanatory.
            reason=f"source: {source_reason}; topic: {topic_reason}",
            method=self.name,
        )


def classify_email(email: Email) -> Classification:
    """Convenience for unit tests that exercise the rules method directly."""
    return RulesClassifier().classify(email)
