"""Shared input and output shapes for every classification method.

Any new method (rules, LLM, small ML model) must consume `Email` and
produce `Classification`. The executor and tests depend only on this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, NotRequired, Protocol, TypedDict

# Closed label sets. Keep these in sync with the assignment brief.
Source = Literal["client", "internal", "external"]
Topic = Literal["accounting", "taxes", "payroll", "legal", "other"]

SOURCES: tuple[Source, ...] = ("client", "internal", "external")
TOPICS: tuple[Topic, ...] = ("accounting", "taxes", "payroll", "legal", "other")


class Participant(TypedDict):
    email: str
    type: str  # "from" | "to" | "cc" | "bcc"
    id: NotRequired[str]
    is_read: NotRequired[bool]


class Attachment(TypedDict):
    file_name: str
    id: NotRequired[str]
    file_type: NotRequired[str]
    file_size: NotRequired[str]


class Email(TypedDict):
    """Expected email shape from a dataset.

    Datasets may carry extra keys (timestamps, thread_id, …). Those are ignored
    by classifiers that do not need them; TypedDict does not forbid extras at runtime.
    """

    id: str
    subject: str
    body: str
    participants: list[Participant]
    attachments: list[Attachment]
    sent_at: NotRequired[str]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    thread_id: NotRequired[str]
    status: NotRequired[str]


@dataclass(frozen=True)
class Classification:
    """One email's labels, plus enough context to audit the decision."""

    id: str
    source: Source
    topic: Topic
    reason: str  # human-readable explanation (rules fired, model rationale, …)
    method: str  # registry name, e.g. "rules"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Classifier(Protocol):
    """Swap implementations via the registry without changing the executor."""

    name: str

    def classify(self, email: Email) -> Classification:
        """Return source and topic for one email."""
...
