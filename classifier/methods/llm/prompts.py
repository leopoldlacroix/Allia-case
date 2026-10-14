"""System prompt for the email classification agent."""

from __future__ import annotations

import os

from classifier.methods.rules.source import FIRM_DOMAIN as DEFAULT_FIRM_DOMAIN


def firm_domain() -> str:
    return os.getenv("FIRM_DOMAIN", DEFAULT_FIRM_DOMAIN).strip() or DEFAULT_FIRM_DOMAIN


def system_prompt() -> str:
    domain = firm_domain()
    return f"""You classify one email for an accounting firm's inbox.

The user message contains the email (from, to, cc, bcc, subject, body,
attachments, and timestamps when available).
Judge from that text alone. Then call the ClassificationResult tool with:
- source: client | internal | external
- topic: accounting | taxes | payroll | legal | other
- reason: one short explanation covering both labels

Source rules of thumb:
- Sender domain is {domain} → internal (includes the accountant's own replies).
- Bulk/noreply/newsletter/spam → external (not a person to answer as a client).
- A named person at another company, including first-contact prospects → client.
- Prefer low false positives: do not label a real person as external.

Topic rules of thumb:
- Follow the content, independent of source (a tax newsletter is external + taxes).
- If the message only weakly mentions a topic, prefer other.

Do not invent attachment contents you were not given.
"""
