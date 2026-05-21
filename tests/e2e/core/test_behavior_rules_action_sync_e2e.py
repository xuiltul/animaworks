from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_v063_migration_syncs_runtime_prompt_surfaces_end_to_end(tmp_path: Path) -> None:
    """Run the aggregate sync against a stale runtime and verify prompt/DB surfaces."""
    from core.migrations.steps import step_v063_behavior_rules_action_rules_skill_sync
    from core.tooling.prompt_db import ToolPromptStore

    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    (data_dir / "animas").mkdir()
    prompts_dir = data_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "behavior_rules.md").write_text("stale behavior rules", encoding="utf-8")

    store = ToolPromptStore(data_dir / "tool_prompts.sqlite3")
    store.set_section("behavior_rules", "stale system section", None)
    store.set_guide("non_s", "stale guide")

    result = step_v063_behavior_rules_action_rules_skill_sync(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    behavior_rules = (data_dir / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
    action_guide = (
        data_dir / "common_knowledge" / "operations" / "action-rules-guide.md"
    ).read_text(encoding="utf-8")
    skill_creator = (data_dir / "common_skills" / "skill-creator" / "SKILL.md").read_text(encoding="utf-8")
    store = ToolPromptStore(data_dir / "tool_prompts.sqlite3")

    assert "[ACTION-RULE]" in behavior_rules
    assert "create_skill" in behavior_rules
    assert "gmail_draft" in action_guide
    assert "slack_post" not in action_guide
    assert "routing_examples" in skill_creator
    assert "[ACTION-RULE]" in (store.get_section("behavior_rules") or "")
    assert "common_knowledge/operations/action-rules-guide.md" in (store.get_guide("non_s") or "")


def test_cli_migrate_fresh_process_resyncs_prompt_surfaces(tmp_path: Path) -> None:
    """CLI migration succeeds from a fresh process and leaves no stale behavior rules."""
    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    (data_dir / "animas").mkdir()
    prompts_dir = data_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "behavior_rules.md").write_text("stale behavior rules", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["ANIMAWORKS_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-m", "cli", "migrate"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "ERROR:" not in output
    assert "cannot import name" not in output

    behavior_rules = (data_dir / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
    assert "[ACTION-RULE]" in behavior_rules
    assert "通常チャットでは `submit_tasks` を使わない" in behavior_rules
    assert "人間からの指示・依頼は必ず `submit_tasks`" not in behavior_rules
