"""Debug helper: classify one sample email with the llm method.

Run under the Cursor/VS Code debugger (see .vscode/launch.json), or:

  .venv/bin/python scripts/debug_one_email.py
  .venv/bin/python scripts/debug_one_email.py 0
  .venv/bin/python scripts/debug_one_email.py 550e8400-e29b-41d4-a716-446655440001

Set breakpoints in classifier/methods/llm/__init__.py (_invoke_once),
classifier/methods/llm/systemone.py (classify_via_systemone), or
classifier/methods/llm/agent.py (extract_structured_result).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.executor import load_emails
from classifier.log import log
from classifier.methods.llm import LlmClassifier
from classifier.methods.llm.config import read_settings


def _pick_email(emails: list, selector: str):
    if selector.isdigit():
        index = int(selector)
        return emails[index]
    for email in emails:
        if email.get("id") == selector:
            return email
    raise SystemExit(f"email not found: {selector!r}")


def _debug_chat(classifier: LlmClassifier, email: dict) -> dict:
    """Chat-mode path: invoke LangChain agent and log intermediate state."""
    from classifier.methods.llm.agent import extract_structured_result, invoke_agent

    if classifier._agent is None:
        raise SystemExit(f"agent unavailable: {classifier._setup_error}")

    log("[debug] invoking chat agent…")
    state = invoke_agent(classifier._agent, email)

    log(f"[debug] state keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
    structured = state.get("structured_response") if isinstance(state, dict) else None
    log(f"[debug] structured_response type={type(structured).__name__} value={structured!r}")

    if isinstance(state, dict) and "messages" in state:
        log(f"[debug] message count: {len(state['messages'])}")
        for index, msg in enumerate(state["messages"]):
            kind = type(msg).__name__
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            preview = repr(content)[:200] if content is not None else ""
            log(f"[debug]   [{index}] {kind} content={preview}")
            if tool_calls:
                log(f"[debug]        tool_calls={tool_calls!r}"[:500])

    result = extract_structured_result(state)
    if result is None:
        log("[debug] extract_structured_result returned None — inspect `state` in the debugger")
        raise SystemExit(1)
    return result.model_dump()


def _debug_systemone(classifier: LlmClassifier, email: dict) -> dict:
    """Systemone / Tev1 path: one HTTP decision call (no LangChain agent)."""
    from classifier.methods.llm.systemone import (
        classify_via_systemone,
        systemone_url,
    )

    url = systemone_url(classifier._settings.base_url)
    log(f"[debug] systemone POST {url} model={classifier._settings.model}")
    result = classify_via_systemone(email, classifier._settings)
    return result.to_dict()


def main() -> None:
    selector = sys.argv[1] if len(sys.argv) > 1 else "0"
    # Fail fast while debugging — no silent rules fallback.
    os.environ.setdefault("LLM_FALLBACK", "0")
    os.environ.setdefault("LLM_STRICT", "1")

    emails = load_emails(ROOT / "data" / "sample_accountant_emails.json")
    email = _pick_email(emails, selector)
    settings = read_settings()
    log(
        f"[debug] mode={settings.api_mode} model={settings.model} "
        f"id={email.get('id')} subject={email.get('subject')}"
    )

    classifier = LlmClassifier(settings)
    if not classifier._ready:
        raise SystemExit(f"llm unavailable: {classifier._setup_error}")

    if settings.api_mode == "systemone":
        payload = _debug_systemone(classifier, email)
    else:
        payload = _debug_chat(classifier, email)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
