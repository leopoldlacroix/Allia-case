"""Structured final answer the agent must submit via ToolStrategy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceLabel = Literal["client", "internal", "external"]
TopicLabel = Literal["accounting", "taxes", "payroll", "legal", "other"]


class ClassificationResult(BaseModel):
    """Final classification for one email. Call this tool when ready to finish."""

    source: SourceLabel = Field(
        description="Who sent it: client (person outside the firm), "
        "internal (firm domain), or external (marketing/newsletter/spam/bulk)."
    )
    topic: TopicLabel = Field(
        description="Main subject: accounting, taxes, payroll, legal, or other."
    )
    reason: str = Field(
        description="Short explanation of both labels (what evidence you used)."
    )
