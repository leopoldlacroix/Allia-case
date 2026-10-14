"""Decide who sent the email: a client, the firm, or everyone else."""

FIRM_DOMAIN = "accountingfirm.com"

# Machine or broadcast senders. A named person at a company is not in this set.
BULK_LOCAL_PARTS = frozenset({"noreply", "statements", "updates", "events", "winner"})

# Kept tight on purpose. A client saying "congratulations" must stay a client.
SPAM_PHRASES = ("lottery", "you've won", "claim your prize")


def classify_source(email: dict) -> tuple[str, str]:
    """Return (label, reason). First match wins."""
    local_part, domain = _sender(email)
    if domain == FIRM_DOMAIN:
        return "internal", f"sender domain is {domain}"
    if local_part in BULK_LOCAL_PARTS:
        return "external", f"bulk local-part {local_part}"

    text = f"{email.get('subject', '')}\n{email.get('body', '')}".lower()
    for phrase in SPAM_PHRASES:
        if phrase in text:
            return "external", f"spam phrase {phrase!r}"
    return "client", "person at an outside domain"


def _sender(email: dict) -> tuple[str, str]:
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() != "from":
            continue
        address = participant["email"].strip().lower()
        local_part, _, domain = address.rpartition("@")
        if not local_part or not domain:
            raise ValueError(f"sender address {address!r} has no domain")
        return local_part, domain
    raise ValueError(f"email {email.get('id', '?')} has no sender")
