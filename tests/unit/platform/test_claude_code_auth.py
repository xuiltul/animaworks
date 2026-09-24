"""Tests for Claude Code CLI authentication detection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.platform.claude_code import get_claude_auth_status


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (
            0,
            '{"loggedIn": true, "authMethod": "claude.ai", "subscriptionType": "max"}',
            {
                "installed": True,
                "logged_in": True,
                "subscription_type": "max",
                "auth_method": "claude.ai",
            },
        ),
        (
            0,
            '{"loggedIn": false}',
            {
                "installed": True,
                "logged_in": False,
                "subscription_type": None,
                "auth_method": None,
            },
        ),
        (1, "", {"installed": True, "logged_in": False, "subscription_type": None}),
        (0, "not json", {"installed": True, "logged_in": False, "subscription_type": None}),
    ],
)
def test_get_claude_auth_status(returncode, stdout, expected):
    result = SimpleNamespace(returncode=returncode, stdout=stdout)
    with (
        patch("core.platform.claude_code.get_claude_executable", return_value="/usr/bin/claude"),
        patch("core.platform.claude_code.subprocess.run", return_value=result) as run,
    ):
        assert get_claude_auth_status() == expected

    run.assert_called_once()
    assert run.call_args.args[0] == ["/usr/bin/claude", "auth", "status", "--json"]
    assert run.call_args.kwargs["env"] is not None
    assert run.call_args.kwargs["timeout"] == 15.0


def test_get_claude_auth_status_without_executable():
    with (
        patch("core.platform.claude_code.get_claude_executable", return_value=None),
        patch("core.platform.claude_code.subprocess.run") as run,
    ):
        assert get_claude_auth_status() == {
            "installed": False,
            "logged_in": False,
            "subscription_type": None,
        }
    run.assert_not_called()
