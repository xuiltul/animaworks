# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Global test fixtures for AnimaWorks E2E tests.

Provides filesystem isolation, config cache management, and
mock/live switching for all test modules.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from tests.helpers.filesystem import (
    create_anima_dir,
    create_test_data_dir,
)

# Load .env at module level so API keys are available before fixtures run.
# main.py calls load_dotenv() for CLI usage; tests need it here.
load_dotenv()

# ── Optional dependency detection ────────────────────────

# Check if ChromaDB is available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ── CLI options ───────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mock",
        action="store_true",
        default=False,
        help="Force mock mode for all API calls",
    )
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run @pytest.mark.live tests (skipped by default)",
    )


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_load_auth():
    """Restore server.app.load_auth after tests that monkey-patch it.

    Several E2E test helpers persist ``_sa.load_auth = lambda: _auth`` beyond
    their ``with patch(...)`` blocks so that the auth-guard middleware returns
    ``local_trust`` at request time.  Without this teardown the monkey-patch
    leaks into subsequent tests that expect the real ``load_auth``.
    """
    import server.app as sa
    original = sa.load_auth
    yield
    sa.load_auth = original


@pytest.fixture
def use_mock(request: pytest.FixtureRequest) -> bool:
    """Determine whether to use mocks or real API calls.

    Returns True when:
      - ``--mock`` flag is passed, OR
      - ``ANTHROPIC_API_KEY`` is not set in the environment
    """
    if request.config.getoption("--mock"):
        return True
    return not os.environ.get("ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def _skip_live_without_key(request: pytest.FixtureRequest, use_mock: bool) -> None:
    """Auto-skip ``@pytest.mark.live`` tests unless explicitly enabled.

    Live tests are skipped by default unless:
      - ``--run-live`` flag is passed, AND
      - Required API keys are available

    Additional skip rules:
      - ``@pytest.mark.azure`` requires ``AZURE_API_KEY`` in environment
      - ``@pytest.mark.ollama`` requires ``OLLAMA_API_BASE`` in environment
    """
    run_live = request.config.getoption("--run-live", default=False)
    if request.node.get_closest_marker("live"):
        if not run_live:
            pytest.skip("Skipping live test: use --run-live to enable")
        if use_mock:
            pytest.skip("Skipping live test: no API key or --mock flag set")
    if request.node.get_closest_marker("azure"):
        if not os.environ.get("AZURE_API_KEY"):
            pytest.skip("Skipping Azure test: AZURE_API_KEY not set")
    if request.node.get_closest_marker("ollama"):
        if not os.environ.get("OLLAMA_API_BASE"):
            pytest.skip("Skipping Ollama test: OLLAMA_API_BASE not set")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated AnimaWorks runtime data directory.

    - Redirects ``ANIMAWORKS_DATA_DIR`` to a temp directory
    - Invalidates config and prompt caches before and after the test
    """
    from core.config import invalidate_cache
    from core.paths import _prompt_cache

    # Create the data directory structure
    d = create_test_data_dir(tmp_path)

    # Redirect all path resolution to the temp directory
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))

    # Invalidate caches to pick up the new data dir
    invalidate_cache()
    _prompt_cache.clear()

    yield d

    # Teardown: kill any supervisor.runner child processes spawned during
    # this test.  Matches processes whose command line references the
    # test's tmp data directory so we don't affect production processes.
    _kill_orphan_runners(str(d))

    # Cleanup: invalidate caches again to avoid leaking between tests
    invalidate_cache()
    _prompt_cache.clear()


def _kill_orphan_runners(data_dir_str: str) -> None:
    """Terminate supervisor.runner processes whose cmdline references *data_dir_str*.

    Scans ``/proc`` on Linux to find child processes; falls back to
    ``pgrep`` if ``/proc`` is unavailable.
    """
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        # Fallback for non-Linux: use pgrep
        _kill_orphan_runners_pgrep(data_dir_str)
        return

    for pid_dir in proc_dir.iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmdline_file = pid_dir / "cmdline"
        try:
            cmdline = cmdline_file.read_text().replace("\x00", " ")
            if "core.supervisor.runner" in cmdline and data_dir_str in cmdline:
                pid = int(pid_dir.name)
                logger.info("Killing orphan runner process PID=%s", pid)
                os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError, PermissionError):
            pass


def _kill_orphan_runners_pgrep(data_dir_str: str) -> None:
    """Fallback: use pgrep + kill for non-Linux platforms."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "core.supervisor.runner"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            # Verify cmdline contains our data_dir before killing
            try:
                cmdline_path = Path(f"/proc/{pid}/cmdline")
                if cmdline_path.exists():
                    cmdline = cmdline_path.read_text().replace("\x00", " ")
                    if data_dir_str not in cmdline:
                        continue
            except OSError:
                continue
            logger.info("Killing orphan runner process PID=%s", pid)
            os.kill(pid, signal.SIGTERM)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass


@pytest.fixture
def make_anima(data_dir: Path):
    """Factory fixture to create anima directories within the test data_dir.

    Returns a callable that creates an anima directory and updates config.json.
    """
    from core.config import invalidate_cache

    def _make(
        name: str = "test-anima",
        **kwargs: Any,
    ) -> Path:
        anima_dir = create_anima_dir(data_dir, name, **kwargs)
        # Invalidate config cache after changing config.json
        invalidate_cache()
        return anima_dir

    return _make
