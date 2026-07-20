"""Static integration checks for workspace timeline parallel lanes."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMELINE_DOM = REPO_ROOT / "server/static/workspace/modules/timeline-dom.js"
TIMELINE_STYLE = REPO_ROOT / "server/static/workspace/style.css"


def test_timeline_rebuilds_parallel_groups_from_all_loaded_events() -> None:
    source = TIMELINE_DOM.read_text(encoding="utf-8")

    assert "buildParallelGroups" in source
    assert "buildParallelGroups(events)" in source
    assert "getParallelTaskCounts(events)" not in source


def test_workspace_parallel_badge_is_removed() -> None:
    source = TIMELINE_DOM.read_text(encoding="utf-8")

    assert "activity-parallel-badge" not in source
    assert "parallelCount" not in source


def test_timeline_avatar_uses_small_cached_thumbnail() -> None:
    source = TIMELINE_DOM.read_text(encoding="utf-8")

    assert 'document.createElement("img")' in source
    assert 'resolveCachedAvatar(anima, candidates, "S")' in source
    assert "avatar.className = className" in source
    assert 'avatar.addEventListener("error"' in source


def test_workspace_styles_define_nested_lanes_and_round_avatars() -> None:
    source = TIMELINE_STYLE.read_text(encoding="utf-8")

    for selector in (
        ".tl-event-avatar",
        ".tl-parallel-group",
        ".tl-parallel-anima-header",
        ".tl-task-lanes",
        ".tl-task-lane-header::before",
    ):
        assert selector in source
    assert "border-radius: 50%" in source
