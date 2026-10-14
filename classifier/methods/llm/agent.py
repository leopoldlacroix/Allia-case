"""Build the LangChain agent: helper tools + ToolStrategy final schema."""

from __future__ import annotations

import json
from typing import Any

from classifier.methods.llm.config import LlmSettings
from classifier.methods.llm.prompts import system_prompt
from classifier.methods.llm.schemas import ClassificationResult
from classifier.methods.llm.tools import build_helper_tools
from classifier.types import Email


def build_chat_model(settings: LlmSettings) -> Any:
    from langchain_openrouter import ChatOpenRouter

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "temperature": 0,
        "max_retries": 1,  # we own the outer retry loop
    }
    if settings.app_url:
        kwargs["app_url"] = settings.app_url
    if settings.app_title:
        kwargs["app_title"] = settings.app_title
    return ChatOpenRouter(**kwargs)


def build_agent(settings: LlmSettings) -> Any:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    model = build_chat_model(settings)
    return create_agent(
        model=model,
        tools=build_helper_tools(),
        system_prompt=system_prompt(),
        response_format=ToolStrategy(ClassificationResult),
    )


def email_user_message(email: Email) -> str:
    """Compact payload for the model (avoid dumping unused metadata)."""
    from_address = ""
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() == "from":
            from_address = participant.get("email", "")
            break
    attachments = [
        attachment.get("file_name", "")
        for attachment in email.get("attachments", [])
        if attachment.get("file_name")
    ]
    payload = {
        "id": email.get("id"),
        "from": from_address,
        "subject": email.get("subject", ""),
        "body": email.get("body", ""),
        "attachment_names": attachments,
    }
    return (
        "Classify this email. Use helper tools if helpful, then submit "
        "ClassificationResult.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def extract_structured_result(state: Any) -> ClassificationResult | None:
    """Pull ClassificationResult from create_agent's final state."""
    if not isinstance(state, dict):
        return None
    structured = state.get("structured_response")
    if isinstance(structured, ClassificationResult):
        return structured
    if isinstance(structured, dict):
        try:
            return ClassificationResult.model_validate(structured)
        except Exception:
            return None
    return None
