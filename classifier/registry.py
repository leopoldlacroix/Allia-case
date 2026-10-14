"""Map a method name to a Classifier instance.

To add a method later:
  1. Implement Classifier under classifier/methods/
  2. Register it in _load_classifier below
  3. Leave the executor and CLI unchanged
"""

from __future__ import annotations

from classifier.methods.rules import RulesClassifier
from classifier.types import Classifier

# Keep names here so `available_methods()` works even if optional deps are missing.
_METHOD_NAMES = ("rules", "llm")


def available_methods() -> list[str]:
    return sorted(_METHOD_NAMES)


def get_classifier(method: str) -> Classifier:
    try:
        return _load_classifier(method)
    except KeyError as exc:
        known = ", ".join(available_methods()) or "(none)"
        raise ValueError(f"unknown method {method!r}; choose one of: {known}") from exc


def _load_classifier(method: str) -> Classifier:
    if method == RulesClassifier.name:
        return RulesClassifier()
    if method == "llm":
        # Lazy import: rules-only users never need LangChain installed.
        from classifier.methods.llm import LlmClassifier

        return LlmClassifier()
    raise KeyError(method)
