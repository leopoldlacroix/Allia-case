"""Compact email dict shared by chat agent and systemone decision calls."""

from __future__ import annotations

from typing import Any

from classifier.types import Email


def email_payload(email: Email) -> dict[str, Any]:
    """All classification-relevant fields: participants, body, attachments, metadata."""
    by_role: dict[str, list[str]] = {
        "from": [],
        "to": [],
        "cc": [],
        "bcc": [],
    }
    for participant in email.get("participants", []):
        role = participant.get("type", "").lower()
        address = (participant.get("email") or "").strip()
        if not address:
            continue
        if role in by_role:
            by_role[role].append(address)
        else:
            by_role.setdefault(role, []).append(address)

    attachments: list[dict[str, str]] = []
    for attachment in email.get("attachments", []):
        name = (attachment.get("file_name") or "").strip()
        if not name:
            continue
        item: dict[str, str] = {"file_name": name}
        file_type = (attachment.get("file_type") or "").strip()
        file_size = (attachment.get("file_size") or "").strip()
        if file_type:
            item["file_type"] = file_type
        if file_size:
            item["file_size"] = file_size
        attachments.append(item)

    payload: dict[str, Any] = {
        "id": email.get("id"),
        "from": by_role["from"],
        "to": by_role["to"],
        "cc": by_role["cc"],
        "bcc": by_role["bcc"],
        "subject": email.get("subject", ""),
        "body": email.get("body", ""),
        "attachments": attachments,
    }

    for key in ("sent_at", "thread_id", "status"):
        value = email.get(key)  # type: ignore[literal-required]
        if value:
            payload[key] = value

    return payload
