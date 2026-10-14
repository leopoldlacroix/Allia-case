"""Decide what the email is about. A weak match stays 'other'."""

import re

# Insertion order is the tie-break: earlier topics win equal scores.
TOPIC_KEYWORDS = {
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
        # "bank statement" misses "credit card statement"; the sample has one.
        "card statement",
    ),
}

SUBJECT_WEIGHT = 2
BODY_WEIGHT = 1
# One body keyword scores 1, which is not enough. A subject keyword scores 2.
THRESHOLD = 2

_PATTERNS = {
    keyword: re.compile(rf"\b{re.escape(keyword)}\w*\b", re.IGNORECASE)
    for keywords in TOPIC_KEYWORDS.values()
    for keyword in keywords
}


def classify_topic(email: dict) -> tuple[str, str]:
    """Return (label, reason). Each keyword counts once per field."""
    subject = email.get("subject", "")
    message = _message_text(email)

    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        subject_hits = _hits(subject, keywords)
        message_hits = _hits(message, keywords)
        scores[topic] = SUBJECT_WEIGHT * len(subject_hits) + BODY_WEIGHT * len(message_hits)
        evidence[topic] = [f"{hit} in subject" for hit in subject_hits] + [
            f"{hit} in message" for hit in message_hits
        ]

    winner = _winner(scores)
    if winner is None:
        weak = [
            f"{topic} ({', '.join(evidence[topic])})"
            for topic in TOPIC_KEYWORDS
            if scores[topic]
        ]
        if not weak:
            return "other", "no topic keyword matched"
        return "other", "below threshold: " + "; ".join(weak)
    return winner, ", ".join(evidence[winner])


def _message_text(email: dict) -> str:
    names = " ".join(
        attachment.get("file_name", "") for attachment in email.get("attachments", [])
    )
    return f"{email.get('body', '')}\n{names}"


def _hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    # Filenames use underscores: "Bank_Reconciliation" should match "reconciliation".
    normalized = text.replace("_", " ")
    return [keyword for keyword in keywords if _PATTERNS[keyword].search(normalized)]


def _winner(scores: dict[str, int]) -> str | None:
    best_topic = None
    best_key = None
    for index, topic in enumerate(TOPIC_KEYWORDS):
        if scores[topic] < THRESHOLD:
            continue
        # Higher score wins. On a tie, the earlier topic in TOPIC_KEYWORDS wins.
        key = (scores[topic], -index)
        if best_key is None or key > best_key:
            best_key = key
            best_topic = topic
    return best_topic
