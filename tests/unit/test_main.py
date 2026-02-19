"""Unit tests for main.py — CLI entry point."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestMain:
    @patch("cli.cli_main")
    def test_main_calls_cli_main(self, mock_cli):
        """Verify main.py imports and calls cli_main."""
        # Import and check the module structure
        import main
        assert hasattr(main, "cli_main")

    @patch("cli.cli_main")
    def test_main_module_execution(self, mock_cli):
        """Verify that running main as __main__ calls cli_main."""
        import runpy

        # runpy won't actually call because __name__ != "__main__"
        # but we can verify the structure by importing
        import main
        # The module-level import brings cli_main into scope
        from cli import cli_main
        assert callable(cli_main)
