"""Helper tools that ground the LLM with deterministic signals.

Important: free models often invent bad tool args (e.g. from_email=': ').
Per LangChain docs, tools read the current email from ToolRuntime.context so
those fields are NOT in the schema the model fills in.
"""

from __future__ import annotations

import json
import os
from typing import Any

from classifier.methods.llm.context import EmailContext
from classifier.methods.rules.source import (
    BULK_LOCAL_PARTS,
    FIRM_DOMAIN as DEFAULT_FIRM_DOMAIN,
    SPAM_PHRASES,
)
from classifier.methods.rules.topic import (
    BODY_WEIGHT,
    SUBJECT_WEIGHT,
    THRESHOLD,
    TOPIC_KEYWORDS,
    _hits,
)


def _firm_domain() -> str:
    return os.getenv("FIRM_DOMAIN", DEFAULT_FIRM_DOMAIN).strip() or DEFAULT_FIRM_DOMAIN


# --- Pure helpers (unit-tested; no LangChain dependency) ---


def sender_info_payload(from_email: str) -> str:
    """Parse the From address into local-part, domain, and whether it is the firm."""
    address = from_email.strip().lower()
    local_part, _, domain = address.rpartition("@")
    if not local_part or not domain:
        return json.dumps({"error": f"invalid from_email: {from_email!r}"})
    firm = _firm_domain()
    payload: dict[str, Any] = {
        "local_part": local_part,
        "domain": domain,
        "is_firm": domain == firm,
        "firm_domain": firm,
    }
    return json.dumps(payload)


def bulk_or_spam_payload(from_email: str, subject: str, body: str) -> str:
    """Check bulk local-parts and tight spam phrases (marketing/lottery style)."""
    address = from_email.strip().lower()
    local_part, _, _domain = address.rpartition("@")
    text = f"{subject}\n{body}".lower()
    spam_hits = [phrase for phrase in SPAM_PHRASES if phrase in text]
    payload = {
        "local_part": local_part,
        "is_bulk_local_part": local_part in BULK_LOCAL_PARTS,
        "spam_phrase_hits": spam_hits,
        "looks_external_bulk_or_spam": local_part in BULK_LOCAL_PARTS or bool(spam_hits),
    }
    return json.dumps(payload)


def topic_keyword_payload(subject: str, body: str, attachment_names: str = "") -> str:
    """Score topic keywords (subject weight 2, body/attachments weight 1). Threshold is 2."""
    message = f"{body}\n{attachment_names}"
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        subject_hits = _hits(subject, keywords)
        message_hits = _hits(message, keywords)
        scores[topic] = SUBJECT_WEIGHT * len(subject_hits) + BODY_WEIGHT * len(message_hits)
        evidence[topic] = [f"{hit} in subject" for hit in subject_hits] + [
            f"{hit} in message" for hit in message_hits
        ]
    above = {topic: score for topic, score in scores.items() if score >= THRESHOLD}
    payload = {
        "scores": scores,
        "evidence": {k: v for k, v in evidence.items() if v},
        "above_threshold": above,
        "threshold": THRESHOLD,
        "hint": (
            "Prefer topics in above_threshold; if empty, topic is often 'other'."
        ),
    }
    return json.dumps(payload)


# Backwards-compatible aliases used by existing unit tests.
get_sender_info = sender_info_payload
check_bulk_or_spam_signals = bulk_or_spam_payload
score_topic_keywords = topic_keyword_payload


def build_helper_tools() -> list[Any]:
    """LangChain tools: no model-visible args; email comes from runtime.context."""
    # Import into this module's globals so @tool can resolve annotations under
    # `from __future__ import annotations` (nested defs share module __globals__).
    from langchain.tools import ToolRuntime, tool

    globals()["ToolRuntime"] = ToolRuntime

    @tool
    def get_sender_info(runtime: ToolRuntime[EmailContext]) -> str:  # noqa: F811
        """Inspect the From address of the email currently being classified.

        No arguments needed — the current email is injected by the runtime.
        Returns local-part, domain, and whether it is the firm domain.
        """
        return sender_info_payload(runtime.context.from_email)

    @tool
    def check_bulk_or_spam_signals(  # noqa: F811
        runtime: ToolRuntime[EmailContext],
    ) -> str:
        """Check whether the current email looks like bulk mail or spam.

        No arguments needed — uses the current email from runtime context.
        """
        ctx = runtime.context
        return bulk_or_spam_payload(ctx.from_email, ctx.subject, ctx.body)

    @tool
    def score_topic_keywords(runtime: ToolRuntime[EmailContext]) -> str:  # noqa: F811
        """Score accounting/tax/payroll/legal keywords on the current email.

        No arguments needed — uses subject, body, and attachment names from context.
        """
        ctx = runtime.context
        return topic_keyword_payload(ctx.subject, ctx.body, ctx.attachment_names)

    return [get_sender_info, check_bulk_or_spam_signals, score_topic_keywords]
