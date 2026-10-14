"""Map a method name to a Classifier instance.

To add a method later (LLM, trained model):
  1. Implement Classifier under classifier/methods/
  2. Register it in _CLASSIFIERS below
  3. Leave the executor and CLI unchanged
"""

from __future__ import annotations

from classifier.methods.rules import RulesClassifier
from classifier.types import Classifier

# name → class. Instantiated on demand by get_classifier().
_CLASSIFIERS: dict[str, type] = {
    RulesClassifier.name: RulesClassifier,
    # "llm": LlmClassifier,
    # "ml": MlClassifier,
}


def available_methods() -> list[str]:
    return sorted(_CLASSIFIERS)


def get_classifier(method: str) -> Classifier:
    try:
        return _CLASSIFIERS[method]()
    except KeyError as exc:
        known = ", ".join(available_methods()) or "(none)"
        raise ValueError(f"unknown method {method!r}; choose one of: {known}") from exc
