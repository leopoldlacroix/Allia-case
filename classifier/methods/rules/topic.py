"""Topic label: what is the email about?

Independent of source: a tax newsletter is still topic "taxes".
A weak keyword hit stays "other" to keep false positives low.
"""

from __future__ import annotations

import re

from classifier.types import Email, Topic

# Insertion order is the tie-break when two topics score equally:
# more specific labels (payroll, legal) beat broader ones (taxes, accounting).
TOPIC_KEYWORDS: dict[Topic, tuple[str, ...]] = {
    "payroll": ("payroll", "salary", "w-2", "withholding"),
    "legal": ("lawsuit", "attorney", "contract dispute"),
    "taxes": ("tax", "deduction", "filing", "irs", "r&d credit"),
    "accounting": (
        "financial statement",
        "invoice",
        "audit",
        "reconciliation",
        "bookkeeping",
        "depreciation",
        "p&l",
        "bank statement",
        # Covers "credit card statement" in the sample; "bank statement" alone would miss it.
        "card statement",
    ),
}

SUBJECT_WEIGHT = 2
BODY_WEIGHT = 1
# Subject hit (2) is enough. A single body hit (1) is not — e.g. "audit" in a thank-you.
THRESHOLD = 2

# Precompiled once: each keyword matches as a word, with an optional trailing
# letter so "tax" also catches "taxes" / "taxation".
_PATTERNS = {
    keyword: re.compile(rf"\b{re.escape(keyword)}\w*\b", re.IGNORECASE)
    for keywords in TOPIC_KEYWORDS.values()
    for keyword in keywords
}


def classify_topic(email: Email) -> tuple[Topic, str]:
    """Return (topic label, short reason). Each keyword counts once per field."""
    subject = email.get("subject", "")
    message = _message_text(email)

    scores: dict[Topic, int] = {}
    evidence: dict[Topic, list[str]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        subject_hits = _hits(subject, keywords)
        message_hits = _hits(message, keywords)
        scores[topic] = (
            SUBJECT_WEIGHT * len(subject_hits) + BODY_WEIGHT * len(message_hits)
        )
        evidence[topic] = [f"{hit} in subject" for hit in subject_hits] + [
            f"{hit} in message" for hit in message_hits
        ]

    winner = _winner(scores)
    if winner is None:
        # Surface weak evidence in the reason so reviewers can see why we abstained.
        weak = [
            f"{topic} ({', '.join(evidence[topic])})"
            for topic in TOPIC_KEYWORDS
            if scores[topic]
        ]
        if not weak:
            return "other", "no topic keyword matched"
        return "other", "below threshold: " + "; ".join(weak)

    return winner, ", ".join(evidence[winner])


def _message_text(email: Email) -> str:
    """Body plus attachment names — filenames often carry the real topic signal."""
    names = " ".join(
        attachment.get("file_name", "") for attachment in email.get("attachments", [])
    )
    return f"{email.get('body', '')}\n{names}"


def _hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    # Filenames use underscores ("Bank_Reconciliation"); treat them as spaces.
    normalized = text.replace("_", " ")
    return [keyword for keyword in keywords if _PATTERNS[keyword].search(normalized)]


def _winner(scores: dict[Topic, int]) -> Topic | None:
    """Highest score above the threshold; ties prefer earlier TOPIC_KEYWORDS entries."""
    best_topic: Topic | None = None
    best_key: tuple[int, int] | None = None
    for index, topic in enumerate(TOPIC_KEYWORDS):
        if scores[topic] < THRESHOLD:
            continue
        # Sort key: higher score first; on a tie, smaller index (more specific) wins.
        key = (scores[topic], -index)
        if best_key is None or key > best_key:
            best_key = key
            best_topic = topic
    return best_topic
