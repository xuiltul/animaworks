# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for activity swimlane page (structure + pure-helper node suite)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SWIMLANE_JS = REPO_ROOT / "server" / "static" / "pages" / "activity" / "swimlane.js"
SWIMLANE_LAYOUT_JS = REPO_ROOT / "server" / "static" / "pages" / "activity" / "swimlane-layout.js"
GROUP_DETAIL_JS = REPO_ROOT / "server" / "static" / "pages" / "activity" / "group-detail.js"
ACTIVITY_JS = REPO_ROOT / "server" / "static" / "pages" / "activity.js"
ACTIVITY_TIMELINE_JS = REPO_ROOT / "server" / "static" / "pages" / "activity-timeline.js"
ACTIVITY_TYPES_JS = REPO_ROOT / "server" / "static" / "shared" / "activity-types.js"
ACTIVITY_CSS = REPO_ROOT / "server" / "static" / "styles" / "activity.css"
I18N_DIR = REPO_ROOT / "server" / "static" / "i18n"
NODE_TEST = REPO_ROOT / "tests" / "unit" / "frontend" / "test_activity_swimlane.mjs"

SWIMLANE_I18N_KEYS = (
    "activity.swimlane_range_label",
    "activity.swimlane_range_1h",
    "activity.swimlane_range_3h",
    "activity.swimlane_range_6h",
    "activity.swimlane_range_24h",
    "activity.swimlane_range_48h",
    "activity.swimlane_now",
    "activity.swimlane_tooltip_events",
    "activity.swimlane_load_earlier",
    "activity.swimlane_detail_close",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSwimlaneModuleStructure:
    def test_files_exist(self) -> None:
        assert SWIMLANE_JS.is_file()
        assert SWIMLANE_LAYOUT_JS.is_file()
        assert GROUP_DETAIL_JS.is_file()
        assert ACTIVITY_JS.is_file()
        assert ACTIVITY_TIMELINE_JS.is_file()

    def test_exports_pure_helpers(self) -> None:
        layout = _read(SWIMLANE_LAYOUT_JS)
        for name in (
            "export function isGroupInProgress",
            "export function groupHasError",
            "export function assignOverlapRows",
            "export function buildLanes",
            "export function computeBarGeometry",
        ):
            assert name in layout, f"missing export: {name}"
        assert "export function renderSwimlane" in _read(SWIMLANE_JS)

    def test_in_progress_uses_five_minute_window(self) -> None:
        src = _read(SWIMLANE_LAYOUT_JS)
        assert "OPEN_WINDOW_MS" in src
        assert "5 * 60 * 1000" in src
        assert "is_open" in src

    def test_ambient_visual_constants(self) -> None:
        src = _read(SWIMLANE_LAYOUT_JS)
        assert "AMBIENT_BAR_H = 8" in src
        assert "SIGNAL_BAR_H = 18" in src
        assert "0.35" in src

    def test_group_detail_exports_render(self) -> None:
        src = _read(GROUP_DETAIL_JS)
        assert "export function renderGroupDetail" in src
        assert "tool_result" in src
        assert "activity-group-events" in src

    def test_activity_page_uses_swimlane_and_ranges(self) -> None:
        src = _read(ACTIVITY_TIMELINE_JS)
        assert "renderSwimlane" in src
        assert "renderGroupDetail" in src
        assert "RANGE_OPTIONS" in src
        assert "export function render" in src
        assert "export function destroy" in src
        assert "group_limit" in src
        assert "30000" in src

    def test_activity_host_loads_timeline_tab(self) -> None:
        src = _read(ACTIVITY_JS)
        assert "activity-timeline.js" in src
        assert "activityTabLoader" in src

    def test_group_type_colors_map(self) -> None:
        src = _read(ACTIVITY_TYPES_JS)
        assert "export const GROUP_TYPE_COLORS" in src
        assert "AMBIENT_GROUP_TYPES" in src

    def test_css_has_swimlane_classes(self) -> None:
        css = _read(ACTIVITY_CSS)
        for cls in (
            ".swimlane-wrap",
            ".swimlane-bar",
            ".swimlane-now-line",
            ".swimlane-detail",
            ".activity-range-chip",
            "swimlanePulse",
        ):
            assert cls in css, f"missing css: {cls}"


class TestSwimlaneI18n:
    @pytest.mark.parametrize("locale", ["ja", "en", "ko"])
    def test_keys_present(self, locale: str) -> None:
        data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        for key in SWIMLANE_I18N_KEYS:
            assert key in data, f"{locale}: missing {key}"
            assert data[key], f"{locale}: empty {key}"


class TestSwimlanePureHelpersNode:
    def test_node_suite_passes(self) -> None:
        node = shutil.which("node")
        if not node:
            pytest.skip("node not available")
        result = subprocess.run(
            [node, "--test", str(NODE_TEST)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"node swimlane suite failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
