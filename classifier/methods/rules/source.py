"""Source label: who sent this message?

Rules are ordered. The first match wins so a firm address never falls through
to the bulk/spam checks.
"""

from __future__ import annotations

from classifier.types import Email, Source

# Domain of the accountant's firm in the sample data. In a real product this
# would be configuration; for the test it is part of the dataset contract.
FIRM_DOMAIN = "accountingfirm.com"

# Local-parts that signal automated / broadcast mail, not a named person.
# Deliberately small: a real client named "events@acme.com" would be rare, and
# false-labelling a client as external is the costly mistake.
BULK_LOCAL_PARTS = frozenset({"noreply", "statements", "updates", "events", "winner"})

# Exact phrases. Kept tight so "congratulations on the audit" stays a client.
SPAM_PHRASES = ("lottery", "you've won", "claim your prize")


def classify_source(email: Email) -> tuple[Source, str]:
    """Return (source label, short reason). First matching rule wins."""
    local_part, domain = _sender(email)

    # 1. Same firm → internal (includes Sarah's own outbound replies).
    if domain == FIRM_DOMAIN:
        return "internal", f"sender domain is {domain}"

    # 2. Known bulk sender or spam wording → external (not a client to answer).
    if local_part in BULK_LOCAL_PARTS:
        return "external", f"bulk local-part {local_part}"

    text = f"{email.get('subject', '')}\n{email.get('body', '')}".lower()
    for phrase in SPAM_PHRASES:
        if phrase in text:
            return "external", f"spam phrase {phrase!r}"

    # 3. Named person at another company, including first-contact prospects.
    return "client", "person at an outside domain"


def _sender(email: Email) -> tuple[str, str]:
    """Split the From address into (local_part, domain), both lowercased."""
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() != "from":
            continue
        address = participant["email"].strip().lower()
        local_part, _, domain = address.rpartition("@")
        if not local_part or not domain:
            raise ValueError(f"sender address {address!r} has no domain")
        return local_part, domain
    raise ValueError(f"email {email.get('id', '?')} has no sender")
