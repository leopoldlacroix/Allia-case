"""Tiny stderr progress logger (flushed so free-model waits don't look frozen)."""

from __future__ import annotations

import sys


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
