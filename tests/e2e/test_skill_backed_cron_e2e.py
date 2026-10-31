from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pytest

from core.runtime.schedule_parser import parse_cron_md
from core.skills.cron_context import build_cron_skill_context
from core.skills.usage import SkillUsageTracker


def _write_skill(path: Path, name: str, body: str, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: e2e skill\n{extra}---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.mark.e2e
def test_skill_backed_cron_resolves_from_cron_md_to_prompt_context(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    _write_skill(anima_dir / "skills" / "local-review" / "SKILL.md", "local-review", "Local review procedure.")
    _write_skill(
        common_dir / "community" / "common-review" / "SKILL.md",
        "common-review",
        "Common review procedure.",
    )
    _write_skill(
        anima_dir / "skills" / "caution-review" / "SKILL.md",
        "caution-review",
        "Caution review procedure.",
        "security:\n  verdict: caution\n",
    )

    cron_md = """\
## PR Review
schedule: 0 9 * * *
type: llm
skills:
  - local-review
  - common_skills/community/common-review/SKILL.md
  - caution-review
Review open pull requests.
"""
    task = parse_cron_md(cron_md)[0]

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        context = build_cron_skill_context(anima_dir, task.skills)

    rendered = context.render()
    assert "Local review procedure." in rendered
    assert "Common review procedure." in rendered
    assert "caution-review: security_caution" in rendered
    assert SkillUsageTracker(anima_dir).get_stats("local-review").use_count == 1
    assert SkillUsageTracker(anima_dir).get_stats("common-review").use_count == 1
    assert SkillUsageTracker(anima_dir).get_stats("caution-review").use_count == 0
