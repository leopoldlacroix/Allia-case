"""Fall back to the deterministic rules classifier when the LLM path fails."""

from __future__ import annotations

from classifier.methods.rules import RulesClassifier
from classifier.types import Classification, Email


def rules_fallback(email: Email, cause: str) -> Classification:
    """Run rules and prefix the reason so the review page shows why LLM was skipped."""
    result = RulesClassifier().classify(email)
    return Classification(
        id=result.id,
        source=result.source,
        topic=result.topic,
        reason=f"llm unavailable ({cause}); used rules: {result.reason}",
        method="rules",
    )
