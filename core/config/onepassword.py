# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Thin wrapper around the 1Password ``op`` CLI.

The client never puts secret values on the command line: new API credential
items receive their value through stdin.  Error messages contain only the
first stderr line, never the value or the reference contents beyond what the
CLI itself prints.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)


class OnePasswordError(ConfigError):
    """Raised when a 1Password CLI invocation fails."""


class OnePasswordClient:
    """Invoke ``op`` with a configurable command and timeout."""

    def __init__(
        self,
        command: list[str],
        timeout_s: float,
        *,
        write_command: list[str] | None = None,
    ) -> None:
        if not command:
            raise OnePasswordError("1Password command must not be empty")
        self.command = list(command)
        self.write_command = list(write_command) if write_command else list(command)
        self.timeout_s = timeout_s
        self._request_count = 0

    @property
    def request_count(self) -> int:
        """Number of CLI invocations made by this client."""
        return self._request_count

    def available(self) -> bool:
        """Return whether the CLI executable can be found on PATH."""
        return shutil.which(self.command[0]) is not None

    def _run(self, argv: list[str], *, stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
        self._request_count += 1
        try:
            return subprocess.run(
                argv,
                input=stdin_text,
                stdin=subprocess.DEVNULL if stdin_text is None else None,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OnePasswordError(f"1Password CLI timed out after {self.timeout_s:g}s") from exc
        except OSError as exc:
            raise OnePasswordError(f"Failed to run 1Password CLI: {exc}") from exc

    @staticmethod
    def _failure_message(completed: subprocess.CompletedProcess[str]) -> str:
        stderr = (completed.stderr or "").strip()
        first_line = stderr.splitlines()[0] if stderr else ""
        if not first_line:
            first_line = f"exit code {completed.returncode}"
        return f"1Password CLI failed: {first_line}"

    def read(self, ref: str) -> str:
        """Read a secret value for *ref*."""
        completed = self._run([*self.command, "read", "--no-newline", ref])
        if completed.returncode != 0:
            raise OnePasswordError(self._failure_message(completed))
        return completed.stdout

    def item_exists(self, vault: str, title: str) -> bool:
        """Return whether an item with *title* exists in *vault*."""
        completed = self._run([*self.command, "item", "get", title, "--vault", vault, "--format", "json"])
        return completed.returncode == 0

    def create_api_credential(self, vault: str, title: str, field: str, value: str) -> str:
        """Create an API credential item and return its item id.

        The value is passed through stdin, never through argv.
        """
        template = {
            "title": title,
            "category": "API_CREDENTIAL",
            "fields": [
                {
                    "id": field,
                    "type": "CONCEALED",
                    "label": field,
                    "value": value,
                }
            ],
        }
        completed = self._run(
            [*self.write_command, "item", "create", "--vault", vault, "--format", "json", "-"],
            stdin_text=json.dumps(template, ensure_ascii=False),
        )
        if completed.returncode != 0:
            raise OnePasswordError(self._failure_message(completed))
        try:
            payload: Any = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise OnePasswordError("1Password CLI returned invalid JSON for item create") from exc
        item_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(item_id, str) or not item_id:
            raise OnePasswordError("1Password CLI returned no item id for item create")
        return item_id
