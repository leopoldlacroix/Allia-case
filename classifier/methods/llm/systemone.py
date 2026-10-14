"""TypeSafe / Jev / Tev1 decision API (/v1/systemone).

Not OpenAI chat: POST state + choice questions, read answers.*.choice.
Works with Ollama tev1 and TypeSafe-compatible gateways (e.g. Beat).
"""

from __future__ import annotations

import json
from typing import Any

from classifier.methods.llm.config import LlmSettings
from classifier.methods.llm.errors import LlmInvokeError
from classifier.methods.llm.prompts import firm_domain
from classifier.types import SOURCES, TOPICS, Classification, Email, Source, Topic


def systemone_url(base_url: str) -> str:
    """Resolve the systemone endpoint from LLM_BASE_URL."""
    base = base_url.rstrip("/")
    if base.endswith("/systemone"):
        return base
    return f"{base}/systemone"


def email_state(email: Email) -> dict[str, Any]:
    """Structured text for the decision model (same fields as the chat user message)."""
    from_address = ""
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() == "from":
            from_address = participant.get("email", "")
            break
    attachments = [
        name
        for name in (
            attachment.get("file_name", "")
            for attachment in email.get("attachments", [])
        )
        if name
    ]
    return {
        "id": email.get("id"),
        "from": from_address,
        "subject": email.get("subject", ""),
        "body": email.get("body", ""),
        "attachment_names": attachments,
    }


def classification_questions() -> dict[str, Any]:
    """Two choice questions: source and topic (label glossary for the model)."""
    domain = firm_domain()
    return {
        "source": {
            "type": "choice",
            "instructions": (
                "Who sent this email to an accountant's inbox? "
                f"Firm domain is {domain}. Prefer low false positives: "
                "do not label a named person as external."
            ),
            "criteria": {
                "client": (
                    "A named person at another company, including first-contact prospects."
                ),
                "internal": (
                    f"Sender is from the firm domain {domain} "
                    "(colleagues or the accountant's own replies)."
                ),
                "external": (
                    "Marketing, newsletter, spam, bulk/noreply, or other non-person mail."
                ),
            },
        },
        "topic": {
            "type": "choice",
            "instructions": (
                "What is the main topic of the email content? "
                "Independent of who sent it. If only weakly mentioned, prefer other."
            ),
            "criteria": {
                "accounting": (
                    "Financial statements, bookkeeping, audit, invoices, reconciliations, "
                    "depreciation, P&L, bank/card statements."
                ),
                "taxes": "Tax planning, filings, deductions, credits, tax law, CPE tax topics.",
                "payroll": "Payroll, salary, W-2, withholding.",
                "legal": "Lawsuits, attorney matters, contract disputes.",
                "other": "Does not clearly fit accounting, taxes, payroll, or legal.",
            },
        },
    }


def classify_via_systemone(email: Email, settings: LlmSettings) -> Classification:
    """One HTTP call; map choice answers to Classification."""
    import httpx

    url = systemone_url(settings.base_url)
    payload = {
        "model": settings.model,
        "state": email_state(email),
        "questions": classification_questions(),
    }
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    if settings.app_url:
        headers["HTTP-Referer"] = settings.app_url
    if settings.app_title:
        headers["X-Title"] = settings.app_title

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=120.0)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        raise LlmInvokeError(f"systemone HTTP error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LlmInvokeError(f"systemone returned non-JSON: {exc}") from exc

    source, topic, reason = parse_systemone_answers(body)
    return Classification(
        id=email["id"],
        source=source,
        topic=topic,
        reason=reason,
        method="llm",
    )


def parse_systemone_answers(
    body: dict[str, Any],
) -> tuple[Source, Topic, str]:
    """Extract source/topic choices from a systemone JSON response."""
    answers = body.get("answers")
    if not isinstance(answers, dict):
        # Some gateways may return the map at the top level.
        answers = body

    source = _choice(answers, "source", SOURCES)
    topic = _choice(answers, "topic", TOPICS)
    if source is None or topic is None:
        raise LlmInvokeError(
            "systemone response missing source/topic choices: "
            f"{json.dumps(body, default=str)[:500]}"
        )

    bits = [f"systemone source={source}", f"topic={topic}"]
    for key in ("source", "topic"):
        entry = answers.get(key)
        if isinstance(entry, dict) and entry.get("confidence") is not None:
            bits.append(f"{key}_confidence={entry['confidence']}")
    return source, topic, "; ".join(bits)


def _choice(
    answers: dict[str, Any],
    key: str,
    allowed: tuple[str, ...],
) -> str | None:
    entry = answers.get(key)
    if entry is None:
        return None
    if isinstance(entry, str):
        value = entry.strip().lower()
    elif isinstance(entry, dict):
        raw = entry.get("choice", entry.get("answer", entry.get("value")))
        if raw is None:
            return None
        value = str(raw).strip().lower()
    else:
        return None
    return value if value in allowed else None
