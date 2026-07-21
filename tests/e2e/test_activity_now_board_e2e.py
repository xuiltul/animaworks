from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVITY_TIMELINE = REPO_ROOT / "server" / "static" / "pages" / "activity-timeline.js"


def test_activity_now_board_coexists_with_swimlane_timeline() -> None:
    source = ACTIVITY_TIMELINE.read_text(encoding="utf-8")
    assert 'import { renderNowBoard } from "./activity/now-board.js"' in source
    assert 'import { renderSwimlane } from "./activity/swimlane.js"' in source
    assert 'id="activityNowBoard"' in source
    assert "_nowBoard?.destroy()" in source


def test_activity_now_board_websocket_to_card_flow() -> None:
    """Run the browserless DOM flow from CustomEvent injection to card update."""
    result = subprocess.run(
        ["node", "--test", "tests/unit/frontend/test_now_board.mjs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
