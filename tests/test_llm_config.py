"""Tests for LLM env-file and CLI override loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classifier.methods.llm.config import LlmOverrides, read_settings


_LLM_KEYS = (
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_API_MODE",
    "LLM_APP_URL",
    "LLM_APP_TITLE",
    "LLM_MAX_RETRIES",
    "LLM_FALLBACK",
    "LLM_STRICT",
)


class LlmConfigTests(unittest.TestCase):
    def test_cli_overrides_win(self) -> None:
        env = {key: "" for key in _LLM_KEYS}
        env.update(
            {
                "LLM_API_KEY": "env-key",
                "LLM_MODEL": "env-model",
                "LLM_BASE_URL": "http://env.example/v1",
                "LLM_API_MODE": "chat",
            }
        )
        with patch.dict(os.environ, env, clear=False):
            settings = read_settings(
                overrides=LlmOverrides(
                    model="cli-model",
                    api_mode="systemone",
                    base_url="http://cli.example/v1",
                )
            )
            self.assertEqual(settings.model, "cli-model")
            self.assertEqual(settings.api_mode, "systemone")
            self.assertEqual(settings.base_url, "http://cli.example/v1")
            self.assertEqual(settings.api_key, "env-key")

    def test_env_file_overrides_ambient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.env"
            path.write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=file-key",
                        "LLM_MODEL=file-model",
                        "LLM_BASE_URL=http://file.example/v1",
                        "LLM_API_MODE=systemone",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            env = {key: "" for key in _LLM_KEYS}
            env.update(
                {
                    "LLM_API_KEY": "ambient-key",
                    "LLM_MODEL": "ambient-model",
                    "LLM_BASE_URL": "http://ambient.example/v1",
                    "LLM_API_MODE": "chat",
                }
            )
            with patch.dict(os.environ, env, clear=False):
                settings = read_settings(env_file=path)
                self.assertEqual(settings.api_key, "file-key")
                self.assertEqual(settings.model, "file-model")
                self.assertEqual(settings.api_mode, "systemone")
                self.assertEqual(settings.base_url, "http://file.example/v1")


if __name__ == "__main__":
    unittest.main()
