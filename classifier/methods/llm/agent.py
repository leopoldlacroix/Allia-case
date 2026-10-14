"""Build the LangChain agent: helper tools + ToolStrategy final schema."""

from __future__ import annotations

import json
import os
from typing import Any

from classifier.methods.llm.config import LlmSettings
from classifier.methods.llm.context import EmailContext, email_context_from_email
from classifier.methods.llm.prompts import system_prompt
from classifier.methods.llm.schemas import ClassificationResult
from classifier.methods.llm.tools import build_helper_tools
from classifier.types import Email

# Free models can spin on bad tool calls; keep the graph shallow.
DEFAULT_RECURSION_LIMIT = 12


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
        # Lets tools read EmailContext via ToolRuntime (args hidden from the model).
        context_schema=EmailContext,
    )


def recursion_limit() -> int:
    raw = (os.getenv("LLM_RECURSION_LIMIT") or "").strip()
    if not raw:
        return DEFAULT_RECURSION_LIMIT
    try:
        return max(4, int(raw))
    except ValueError:
        return DEFAULT_RECURSION_LIMIT


def invoke_agent(agent: Any, email: Email) -> Any:
    """Run the agent with injected email context and a hard step limit."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": email_user_message(email)}]},
        context=email_context_from_email(email),
        config={"recursion_limit": recursion_limit()},
    )


def email_user_message(email: Email) -> str:
    """Compact payload for the model (avoid dumping unused metadata)."""
    ctx = email_context_from_email(email)
    payload = {
        "id": ctx.email_id,
        "from": ctx.from_email,
        "subject": ctx.subject,
        "body": ctx.body,
        "attachment_names": (
            ctx.attachment_names.split() if ctx.attachment_names else []
        ),
    }
    # Prefer a single string for attachment names in the JSON for readability.
    payload["attachment_names"] = [
        name
        for name in (
            attachment.get("file_name", "")
            for attachment in email.get("attachments", [])
        )
        if name
    ]
    return (
        "Classify this email. Helper tools need no arguments — call them if useful, "
        "then submit ClassificationResult.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def extract_structured_result(state: Any) -> ClassificationResult | None:
    """Pull ClassificationResult from create_agent's final state.

    Also recovers from common free-model shapes: dict payload, or a final
    AIMessage tool_call whose args match ClassificationResult.
    """
    if not isinstance(state, dict):
        return None

    structured = state.get("structured_response")
    parsed = _coerce_result(structured)
    if parsed is not None:
        return parsed

    messages = state.get("messages") or []
    for message in reversed(list(messages)):
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if not name:
                continue
            if "ClassificationResult" not in str(name) and name != "ClassificationResult":
                continue
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            parsed = _coerce_result(args)
            if parsed is not None:
                return parsed
    return None


def _coerce_result(value: Any) -> ClassificationResult | None:
    if isinstance(value, ClassificationResult):
        return value
    if isinstance(value, dict):
        try:
            return ClassificationResult.model_validate(value)
        except Exception:
            return None
    if isinstance(value, str) and value.strip():
        try:
            return ClassificationResult.model_validate_json(value)
        except Exception:
            try:
                return ClassificationResult.model_validate(json.loads(value))
            except Exception:
                return None
    return None
