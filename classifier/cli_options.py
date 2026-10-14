"""Shared argparse flags for executor / review CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from classifier.methods.llm.config import LlmOverrides


def add_llm_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional LLM profile env file (overrides repo .env).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM_MODEL for this run.",
    )
    parser.add_argument(
        "--api-mode",
        default=None,
        help="Override LLM_API_MODE (chat or systemone).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override LLM_BASE_URL for this run.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override LLM_API_KEY for this run.",
    )


def add_run_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label stored in runs/results/ filename and metadata.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a run artifact under runs/results/.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("runs/results"),
        help="Directory for run JSON files (default: runs/results). Any path is fine.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Classify this many emails in parallel (default: 1). Helps when the "
        "provider can overlap requests; local Ollama often still serializes on GPU.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only classify the first N emails (smoke / faster compare).",
    )


def overrides_from_args(args: argparse.Namespace) -> LlmOverrides:
    return LlmOverrides(
        api_key=getattr(args, "api_key", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        api_mode=getattr(args, "api_mode", None),
    )
