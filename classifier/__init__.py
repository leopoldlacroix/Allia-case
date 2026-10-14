"""Email classification with swappable methods.

Public surface used by the CLI and tests. Method implementations live under
`classifier.methods`.
"""

from classifier.executor import load_emails, load_labels, run
from classifier.methods.rules import RulesClassifier, classify_email
from classifier.registry import available_methods, get_classifier
from classifier.types import Classification, Classifier, Email, Source, Topic

__all__ = [
    "Classification",
    "Classifier",
    "Email",
    "RulesClassifier",
    "Source",
    "Topic",
    "available_methods",
    "classify_email",
    "get_classifier",
    "load_emails",
    "load_labels",
    "run",
]
