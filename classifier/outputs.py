"""Persist classification runs under runs/results/ for later comparison."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classifier.types import Classification

DEFAULT_OUTPUTS_DIR = Path("runs/results")


@dataclass(frozen=True)
class RunArtifact:
    id: str
    created_at: str
    method: str
    emails_path: str
    results: list[Classification]
    tag: str | None = None
    settings: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "tag": self.tag,
            "method": self.method,
            "emails_path": self.emails_path,
            "settings": self.settings or {},
            "results": [result.to_dict() for result in self.results],
        }


def _slug(value: str, *, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (cleaned or "run")[:max_len]


def make_run_id(
    method: str,
    *,
    model: str | None = None,
    tag: str | None = None,
    when: datetime | None = None,
) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, _slug(method)]
    if model:
        parts.append(_slug(model))
    if tag:
        parts.append(_slug(tag))
    return "_".join(parts)


def save_run(
    results: list[Classification],
    *,
    method: str,
    emails_path: str | Path,
    tag: str | None = None,
    settings: dict[str, str] | None = None,
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> Path:
    """Write one run JSON; return the path."""
    model = (settings or {}).get("model")
    created = datetime.now(timezone.utc)
    run_id = make_run_id(method, model=model, tag=tag, when=created)
    artifact = RunArtifact(
        id=run_id,
        created_at=created.isoformat().replace("+00:00", "Z"),
        method=method,
        emails_path=str(emails_path),
        results=results,
        tag=tag,
        settings=settings,
    )
    directory = Path(outputs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _is_run_payload(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("results"), list)
        and isinstance(payload.get("id"), str)
        and isinstance(payload.get("method"), str)
    )


def load_run(path: str | Path) -> RunArtifact:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not _is_run_payload(payload):
        raise ValueError(f"{path} is not a classification run artifact")
    results = [
        Classification(
            id=row["id"],
            source=row["source"],
            topic=row["topic"],
            reason=row.get("reason", ""),
            method=row.get("method", payload.get("method", "unknown")),
        )
        for row in payload["results"]
    ]
    return RunArtifact(
        id=payload["id"],
        created_at=payload.get("created_at", ""),
        method=payload.get("method", "unknown"),
        emails_path=payload.get("emails_path", ""),
        results=results,
        tag=payload.get("tag"),
        settings=payload.get("settings") or {},
    )


def list_runs(outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR) -> list[Path]:
    directory = Path(outputs_dir)
    if not directory.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if _is_run_payload(payload):
            paths.append(path)
    return paths


def load_runs(
    paths: list[str | Path] | None = None,
    *,
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> list[RunArtifact]:
    files = [Path(p) for p in paths] if paths else list_runs(outputs_dir)
    return [load_run(path) for path in files]
