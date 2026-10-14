"""Per-email context injected into tools via ToolRuntime (hidden from the model)."""

from __future__ import annotations

from dataclasses import dataclass

from classifier.types import Email


@dataclass(frozen=True)
class EmailContext:
    """Immutable fields for the email currently being classified."""

    email_id: str
    from_email: str
    subject: str
    body: str
    attachment_names: str


def email_context_from_email(email: Email) -> EmailContext:
    from_address = ""
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() == "from":
            from_address = participant.get("email", "")
            break
    names = " ".join(
        attachment.get("file_name", "")
        for attachment in email.get("attachments", [])
        if attachment.get("file_name")
    )
    return EmailContext(
        email_id=str(email.get("id", "")),
        from_email=from_address,
        subject=email.get("subject", "") or "",
        body=email.get("body", "") or "",
        attachment_names=names,
    )
