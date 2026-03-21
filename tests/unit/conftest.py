# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit-test fixtures: global permissions cache for ToolHandler security tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.config.global_permissions import GlobalPermissionsCache
from core.paths import TEMPLATES_DIR


@pytest.fixture(autouse=True)
def _global_permissions_for_unit_tests(tmp_path: Path) -> None:
    """Load ``permissions.global.json`` template so command block patterns match production.

    Uses a dedicated subdirectory so the hash file (``run/``) does not
    pollute the test's ``tmp_path`` root.
    """
    GlobalPermissionsCache.reset()
    gp_dir = tmp_path / "_global_perms"
    gp_dir.mkdir(exist_ok=True)
    src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
    dst = gp_dir / "permissions.global.json"
    shutil.copy(src, dst)
    GlobalPermissionsCache.get().load(dst, interactive=False)
    yield
    GlobalPermissionsCache.reset()
