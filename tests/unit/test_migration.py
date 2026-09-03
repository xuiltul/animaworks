from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified migration framework."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.migrations.registry import MigrationReport, MigrationRunner, MigrationStep, StepResult
from core.migrations.tracker import MigrationState, MigrationTracker

# ── Tracker tests ───────────────────────────────────────────


class TestMigrationTracker:
    def test_load_empty(self, tmp_path: Path) -> None:
        tracker = MigrationTracker(tmp_path)
        state = tracker.load()
        assert state.applied_version == ""
        assert state.steps_applied == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        tracker = MigrationTracker(tmp_path)
        state = MigrationState(
            applied_version="0.5.4",
            steps_applied={"step_a": "2026-03-18T10:00:00"},
            last_migrated_at="2026-03-18T10:00:00",
        )
        tracker.save(state)

        tracker2 = MigrationTracker(tmp_path)
        loaded = tracker2.load()
        assert loaded.applied_version == "0.5.4"
        assert "step_a" in loaded.steps_applied

    def test_is_step_applied(self, tmp_path: Path) -> None:
        tracker = MigrationTracker(tmp_path)
        assert not tracker.is_step_applied("step_x")
        tracker.mark_applied("step_x")
        assert tracker.is_step_applied("step_x")

    def test_corrupt_state_file(self, tmp_path: Path) -> None:
        (tmp_path / "migration_state.json").write_text("not json", encoding="utf-8")
        tracker = MigrationTracker(tmp_path)
        state = tracker.load()
        assert state.applied_version == ""

    def test_mark_applied_updates_version(self, tmp_path: Path) -> None:
        tracker = MigrationTracker(tmp_path)
        with patch("core.migrations.tracker._get_package_version", return_value="1.2.3"):
            tracker.mark_applied("test_step")
        state = tracker.load()
        assert state.applied_version == "1.2.3"


# ── Runner tests ────────────────────────────────────────────


def _ok_step(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    return StepResult(changed=1, skipped=0, details=["did something"])


def _skip_step(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    return StepResult(changed=0, skipped=1, details=["nothing to do"])


def _error_step(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    return StepResult(changed=0, skipped=0, details=[], error="something broke")


def _crash_step(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    raise RuntimeError("unhandled crash")


def _dry_aware_step(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    if dry_run:
        return StepResult(changed=1, skipped=0, details=["would change"])
    (data_dir / "test_marker.txt").write_text("changed", encoding="utf-8")
    return StepResult(changed=1, skipped=0, details=["changed"])


class TestMigrationRunner:
    def _make_runner(self, tmp_path: Path) -> MigrationRunner:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        runner = MigrationRunner(tmp_path)
        runner.register(MigrationStep("s1", "Step 1", "structural", _ok_step))
        runner.register(MigrationStep("s2", "Step 2", "per_anima", _skip_step))
        runner.register(MigrationStep("s3", "Step 3", "db_sync", _error_step))
        return runner

    def test_run_all(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        report = runner.run_all()
        assert isinstance(report, MigrationReport)
        assert len(report.steps) == 3
        assert report.total_changed == 1
        assert report.total_skipped == 1
        assert len(report.errors) == 1

    def test_run_all_skips_applied(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner.tracker.mark_applied("s1")
        report = runner.run_all()
        step_results = {s.id: r for s, r in report.steps}
        assert step_results["s1"].skipped == 1
        assert step_results["s1"].changed == 0

    def test_run_all_force_reapplies(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner.tracker.mark_applied("s1")
        report = runner.run_all(force=True)
        step_results = {s.id: r for s, r in report.steps}
        assert step_results["s1"].changed == 1

    def test_run_resync_db(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        report = runner.run_resync_db()
        assert len(report.steps) == 1
        assert report.steps[0][0].id == "s3"

    def test_dry_run_no_side_effects(self, tmp_path: Path) -> None:
        runner = MigrationRunner(tmp_path)
        runner.register(MigrationStep("dry", "Dry test", "structural", _dry_aware_step))
        report = runner.run_all(dry_run=True)
        assert report.total_changed == 1
        assert not (tmp_path / "test_marker.txt").exists()
        assert not runner.tracker.is_step_applied("dry")

    def test_crash_step_handled(self, tmp_path: Path) -> None:
        runner = MigrationRunner(tmp_path)
        runner.register(MigrationStep("crash", "Crash step", "structural", _crash_step))
        report = runner.run_all()
        assert len(report.errors) == 1
        assert "unhandled crash" in report.errors[0]

    def test_list_steps(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner.tracker.mark_applied("s1")
        steps = runner.list_steps()
        assert len(steps) == 3
        assert steps[0]["applied"]
        assert not steps[1]["applied"]


# ── Step function tests ─────────────────────────────────────


class TestMigrationSteps:
    @pytest.fixture()
    def data_dir(self, tmp_path: Path) -> Path:
        dd = tmp_path / ".animaworks"
        dd.mkdir()
        (dd / "config.json").write_text("{}", encoding="utf-8")
        (dd / "animas").mkdir()
        return dd

    def _make_anima(self, data_dir: Path, name: str) -> Path:
        d = data_dir / "animas" / name
        d.mkdir(parents=True)
        (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
        (d / "state").mkdir()
        return d

    def test_step_vault_reencrypt_generates_key_and_encrypts_plaintext(self, data_dir: Path) -> None:
        from core.config.vault import VaultManager
        from core.migrations.steps import step_vault_reencrypt

        original = {
            "shared": {"API_TOKEN": "plain-token"},
            "sakura": {"SERVICE_KEY": "plain-service-key"},
        }
        (data_dir / "vault.json").write_text(json.dumps(original), encoding="utf-8")

        result = step_vault_reencrypt(data_dir, dry_run=False, verbose=True)

        assert result.error is None
        assert result.changed == 1
        assert (data_dir / "vault.key").is_file()
        encrypted = json.loads((data_dir / "vault.json").read_text(encoding="utf-8"))
        assert encrypted["shared"]["API_TOKEN"] != original["shared"]["API_TOKEN"]
        assert encrypted["sakura"]["SERVICE_KEY"] != original["sakura"]["SERVICE_KEY"]
        vault = VaultManager(data_dir)
        assert vault.get("shared", "API_TOKEN") == original["shared"]["API_TOKEN"]
        assert vault.get("sakura", "SERVICE_KEY") == original["sakura"]["SERVICE_KEY"]
        assert len(list(data_dir.glob("vault.json.bak-*"))) == 1

    def test_step_vault_reencrypt_rolls_back_on_verification_failure(self, data_dir: Path) -> None:
        from core.config.vault import VaultManager
        from core.migrations.steps import step_vault_reencrypt

        original_text = json.dumps({"shared": {"API_TOKEN": "plain-token"}})
        (data_dir / "vault.json").write_text(original_text, encoding="utf-8")

        with patch.object(VaultManager, "decrypt", return_value="corrupted"):
            result = step_vault_reencrypt(data_dir, dry_run=False, verbose=True)

        assert result.error is not None
        assert "Round-trip verification failed" in result.error
        assert (data_dir / "vault.json").read_text(encoding="utf-8") == original_text
        assert not (data_dir / "vault.key").exists()

    def test_step_vault_reencrypt_backs_up_existing_key(self, data_dir: Path) -> None:
        from core.config.vault import VaultManager
        from core.migrations.steps import step_vault_reencrypt

        vault = VaultManager(data_dir)
        vault.generate_key()
        original_key = vault.key_path.read_bytes()
        vault.save_vault({"shared": {"API_TOKEN": "plain-token"}})

        result = step_vault_reencrypt(data_dir, dry_run=False, verbose=True)

        assert result.error is None
        key_backups = list(data_dir.glob("vault.key.bak-*"))
        assert len(key_backups) == 1
        assert key_backups[0].read_bytes() == original_key

    def test_step_current_task_rename(self, data_dir: Path) -> None:
        from core.migrations.steps import step_current_task_rename

        anima = self._make_anima(data_dir, "alice")
        (anima / "state" / "current_task.md").write_text("tasks here", encoding="utf-8")
        result = step_current_task_rename(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1
        assert (anima / "state" / "current_state.md").exists()
        assert not (anima / "state" / "current_task.md").exists()

    def test_step_current_task_rename_dry_run(self, data_dir: Path) -> None:
        from core.migrations.steps import step_current_task_rename

        anima = self._make_anima(data_dir, "alice")
        (anima / "state" / "current_task.md").write_text("tasks here", encoding="utf-8")
        result = step_current_task_rename(data_dir, dry_run=True, verbose=True)
        assert result.changed == 0 or result.details
        assert (anima / "state" / "current_task.md").exists()

    def test_step_current_task_rename_skip_if_state_exists(self, data_dir: Path) -> None:
        from core.migrations.steps import step_current_task_rename

        anima = self._make_anima(data_dir, "alice")
        (anima / "state" / "current_task.md").write_text("old", encoding="utf-8")
        (anima / "state" / "current_state.md").write_text("new", encoding="utf-8")
        result = step_current_task_rename(data_dir, dry_run=False, verbose=True)
        assert result.changed == 0

    def test_step_pending_merge(self, data_dir: Path) -> None:
        from core.migrations.steps import step_pending_merge

        anima = self._make_anima(data_dir, "bob")
        (anima / "state" / "current_state.md").write_text("# State\n", encoding="utf-8")
        (anima / "state" / "pending.md").write_text("urgent task", encoding="utf-8")
        result = step_pending_merge(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1
        content = (anima / "state" / "current_state.md").read_text(encoding="utf-8")
        assert "urgent task" in content
        assert not (anima / "state" / "pending.md").exists()

    def test_step_pending_merge_empty(self, data_dir: Path) -> None:
        from core.migrations.steps import step_pending_merge

        anima = self._make_anima(data_dir, "bob")
        (anima / "state" / "pending.md").write_text("", encoding="utf-8")
        result = step_pending_merge(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1
        assert not (anima / "state" / "pending.md").exists()

    def test_step_current_task_references(self, data_dir: Path) -> None:
        from core.migrations.steps import step_current_task_references

        anima = self._make_anima(data_dir, "carol")
        (anima / "heartbeat.md").write_text("Check current_task.md for status\nReview current_task", encoding="utf-8")
        result = step_current_task_references(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1
        content = (anima / "heartbeat.md").read_text(encoding="utf-8")
        assert "current_state.md" in content
        assert "current_task" not in content

    def test_step_current_task_references_no_match(self, data_dir: Path) -> None:
        from core.migrations.steps import step_current_task_references

        anima = self._make_anima(data_dir, "carol")
        (anima / "heartbeat.md").write_text("No references here", encoding="utf-8")
        result = step_current_task_references(data_dir, dry_run=False, verbose=True)
        assert result.changed == 0

    def test_step_person_to_anima_skip(self, data_dir: Path) -> None:
        from core.migrations.steps import step_person_to_anima

        result = step_person_to_anima(data_dir, dry_run=False, verbose=True)
        assert result.skipped == 1

    def test_step_enable_skill_catalog_router_updates_existing_config(self, data_dir: Path) -> None:
        from core.migrations.steps import step_enable_skill_catalog_router

        config_path = data_dir / "config.json"
        config_path.write_text(
            json.dumps({"prompt": {"skill_catalog_router_enabled": False}}),
            encoding="utf-8",
        )

        result = step_enable_skill_catalog_router(data_dir, dry_run=False, verbose=True)

        assert result.changed == 1
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        assert raw["prompt"]["skill_catalog_router_enabled"] is True
        assert raw["prompt"]["skill_catalog_router_top_k"] == 5
        assert raw["prompt"]["skill_catalog_router_min_score"] == 1.15
        assert raw["prompt"]["skill_catalog_router_include_body"] is True

    def test_step_enable_skill_catalog_router_dry_run_keeps_config(self, data_dir: Path) -> None:
        from core.migrations.steps import step_enable_skill_catalog_router

        config_path = data_dir / "config.json"
        config_path.write_text('{"prompt": {"skill_catalog_router_enabled": false}}\n', encoding="utf-8")

        result = step_enable_skill_catalog_router(data_dir, dry_run=True, verbose=True)

        assert result.changed == 1
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        assert raw["prompt"]["skill_catalog_router_enabled"] is False

    def test_step_enable_skill_catalog_router_preserves_tuned_values(self, data_dir: Path) -> None:
        from core.migrations.steps import step_enable_skill_catalog_router

        config_path = data_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "prompt": {
                        "skill_catalog_router_enabled": False,
                        "skill_catalog_router_top_k": 9,
                        "skill_catalog_router_min_score": 2.0,
                        "skill_catalog_router_include_body": False,
                    }
                }
            ),
            encoding="utf-8",
        )

        result = step_enable_skill_catalog_router(data_dir, dry_run=False, verbose=True)

        assert result.changed == 1
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        assert raw["prompt"]["skill_catalog_router_enabled"] is True
        assert raw["prompt"]["skill_catalog_router_top_k"] == 9
        assert raw["prompt"]["skill_catalog_router_min_score"] == 2.0
        assert raw["prompt"]["skill_catalog_router_include_body"] is False

    def test_step_models_json_create_skip_existing(self, data_dir: Path) -> None:
        from core.migrations.steps import step_models_json_create

        (data_dir / "models.json").write_text("{}", encoding="utf-8")
        result = step_models_json_create(data_dir, dry_run=False, verbose=True)
        assert result.skipped == 1

    def test_step_grok_models_json_adds_entries_and_preserves_existing(self, data_dir: Path) -> None:
        from core.migrations.steps import step_grok_models_json

        models_path = data_dir / "models.json"
        existing = {"custom/model": {"mode": "A", "context_window": 12345}}
        models_path.write_text(json.dumps(existing), encoding="utf-8")

        result = step_grok_models_json(data_dir, dry_run=False, verbose=True)

        assert result.changed == 2
        raw = json.loads(models_path.read_text(encoding="utf-8"))
        assert raw["custom/model"] == existing["custom/model"]
        assert raw["grok/grok-4.5"] == {"mode": "X", "context_window": 500000}
        assert raw["grok/*"] == {"mode": "X", "context_window": 500000}

    def test_step_grok_models_json_preserves_existing_grok_entry(self, data_dir: Path) -> None:
        from core.migrations.steps import step_grok_models_json

        models_path = data_dir / "models.json"
        custom_grok = {"mode": "A", "context_window": 999999}
        models_path.write_text(json.dumps({"grok/*": custom_grok}), encoding="utf-8")

        result = step_grok_models_json(data_dir, dry_run=False, verbose=True)

        assert result.changed == 1
        raw = json.loads(models_path.read_text(encoding="utf-8"))
        assert raw["grok/*"] == custom_grok
        assert raw["grok/grok-4.5"] == {"mode": "X", "context_window": 500000}

    def test_step_shortterm_layout(self, data_dir: Path) -> None:
        from core.migrations.steps import step_shortterm_layout

        anima = self._make_anima(data_dir, "dave")
        shortterm = anima / "shortterm"
        shortterm.mkdir(exist_ok=True)
        (shortterm / "session_state.json").write_text("{}", encoding="utf-8")
        result = step_shortterm_layout(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1
        assert (shortterm / "chat" / "session_state.json").exists()
        assert not (shortterm / "session_state.json").exists()

    def test_step_update_version(self, data_dir: Path) -> None:
        from core.migrations.steps import step_update_version

        result = step_update_version(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1

    def test_remove_turn_limit_backs_up_and_updates_persisted_config(self, tmp_path: Path) -> None:
        from core.migrations.steps import step_remove_turn_limit

        legacy_key = "max_turns"
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"anima_defaults": {"model": "test-model", legacy_key: 10000}}),
            encoding="utf-8",
        )
        status_path = tmp_path / "animas" / "alice" / "status.json"
        status_path.parent.mkdir(parents=True)
        status_path.write_text(
            json.dumps({"model": "test-model", legacy_key: 30}),
            encoding="utf-8",
        )

        result = step_remove_turn_limit(tmp_path, dry_run=False, verbose=True)

        assert result.error is None
        assert result.changed == 2
        assert legacy_key not in json.loads(config_path.read_text(encoding="utf-8"))["anima_defaults"]
        assert legacy_key not in json.loads(status_path.read_text(encoding="utf-8"))
        assert list(tmp_path.glob("config.json.bak-*"))
        assert list(status_path.parent.glob("status.json.bak-*"))

        second = step_remove_turn_limit(tmp_path, dry_run=False, verbose=True)
        assert second.error is None
        assert second.changed == 0
        assert second.skipped == 1

    def test_remove_turn_limit_dry_run_does_not_modify_or_back_up(self, tmp_path: Path) -> None:
        from core.migrations.steps import step_remove_turn_limit

        legacy_key = "max_turns"
        status_path = tmp_path / "animas" / "alice" / "status.json"
        status_path.parent.mkdir(parents=True)
        original = json.dumps({legacy_key: 30})
        status_path.write_text(original, encoding="utf-8")

        result = step_remove_turn_limit(tmp_path, dry_run=True, verbose=True)

        assert result.changed == 1
        assert status_path.read_text(encoding="utf-8") == original
        assert not list(status_path.parent.glob("status.json.bak-*"))

    def test_remove_machine_config_removes_key_and_is_idempotent(self, tmp_path: Path) -> None:
        from core.migrations.steps import step_remove_machine_config

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model": "x", "machine": {"engine_priority": ["claude"]}}),
            encoding="utf-8",
        )

        result = step_remove_machine_config(tmp_path, dry_run=False, verbose=True)
        assert result.error is None
        assert result.changed == 1
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "machine" not in data
        assert "model" in data  # other keys preserved
        assert list(tmp_path.glob("config.json.bak-*"))

        # idempotent: no-op on second run
        second = step_remove_machine_config(tmp_path, dry_run=False, verbose=True)
        assert second.error is None
        assert second.changed == 0
        assert second.skipped == 1

    def test_remove_machine_config_noop_when_no_key(self, tmp_path: Path) -> None:
        from core.migrations.steps import step_remove_machine_config

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")

        result = step_remove_machine_config(tmp_path, dry_run=False, verbose=True)
        assert result.error is None
        assert result.changed == 0
        assert "machine" not in json.loads(config_path.read_text(encoding="utf-8"))

    def test_remove_machine_config_dry_run_does_not_modify(self, tmp_path: Path) -> None:
        from core.migrations.steps import step_remove_machine_config

        config_path = tmp_path / "config.json"
        original = json.dumps({"machine": {"engine_priority": ["claude"]}})
        config_path.write_text(original, encoding="utf-8")

        result = step_remove_machine_config(tmp_path, dry_run=True, verbose=True)
        assert result.error is None
        assert result.changed == 1
        assert config_path.read_text(encoding="utf-8") == original
        assert not list(tmp_path.glob("config.json.bak-*"))

    def test_remove_machine_config_step_registered_before_version(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        ids = [item["id"] for item in runner.list_steps()]

        assert "remove_machine_config_20260903" in ids
        assert ids.index("remove_machine_config_20260903") < ids.index("update_version")

    def test_v063_registered_after_v062(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        ids = [item["id"] for item in runner.list_steps()]

        assert "v062_skill_removal_and_activity_log" in ids
        assert "v063_behavior_rules_action_rules_skill_sync" in ids
        assert ids.index("v063_behavior_rules_action_rules_skill_sync") > ids.index(
            "v062_skill_removal_and_activity_log"
        )
        assert ids.index("v063_behavior_rules_action_rules_skill_sync") < ids.index("update_version")

    def test_remove_turn_limit_step_registered_before_version(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        ids = [item["id"] for item in runner.list_steps()]

        assert ids.index("remove_turn_limit_20260802") < ids.index("update_version")

    def test_step_v063_resyncs_stale_runtime_prompts(self, data_dir: Path) -> None:
        from core.migrations.steps import step_v063_behavior_rules_action_rules_skill_sync

        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "behavior_rules.md").write_text(
            "stale: human instructions must be registered with `submit_tasks`",
            encoding="utf-8",
        )
        result = step_v063_behavior_rules_action_rules_skill_sync(data_dir, dry_run=False, verbose=True)

        assert result.error is None
        behavior_rules = (data_dir / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
        action_guide = (data_dir / "common_knowledge" / "operations" / "action-rules-guide.md").read_text(
            encoding="utf-8"
        )
        skill_creator = (data_dir / "common_skills" / "skill-creator" / "SKILL.md").read_text(encoding="utf-8")
        assert "[ACTION-RULE]" in behavior_rules
        assert "common_skills/skill-creator/SKILL.md" in behavior_rules
        assert "gmail_draft" in action_guide
        assert "slack_post" not in action_guide
        assert "trust_level" in skill_creator

    def test_step_task_delegation_to_common_knowledge(self, data_dir: Path) -> None:
        from core.migrations.steps import step_task_delegation_to_common_knowledge

        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        stale = prompts_dir / "task_delegation_rules.md"
        stale.write_text("old content", encoding="utf-8")

        result = step_task_delegation_to_common_knowledge(data_dir, dry_run=False, verbose=True)
        assert result.changed >= 1
        assert not stale.exists(), "stale prompts/task_delegation_rules.md should be removed"

    def test_step_task_delegation_to_common_knowledge_dry_run(self, data_dir: Path) -> None:
        from core.migrations.steps import step_task_delegation_to_common_knowledge

        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        stale = prompts_dir / "task_delegation_rules.md"
        stale.write_text("old content", encoding="utf-8")

        result = step_task_delegation_to_common_knowledge(data_dir, dry_run=True, verbose=True)
        assert result.changed >= 1
        assert stale.exists(), "dry_run should not remove the file"

    def test_step_task_delegation_no_stale_file(self, data_dir: Path) -> None:
        from core.migrations.steps import step_task_delegation_to_common_knowledge

        result = step_task_delegation_to_common_knowledge(data_dir, dry_run=False, verbose=True)
        assert result.changed >= 0

    def test_step_v0120_resyncs_prompts_and_removes_stale(self, data_dir: Path) -> None:
        from core.migrations.steps import step_v0120_prompt_deadline_engine_neutral_resync

        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "behavior_rules.md").write_text("stale: find files with `Glob` before reading", encoding="utf-8")
        (prompts_dir / "task_delegation_rules.md").write_text("old content", encoding="utf-8")

        result = step_v0120_prompt_deadline_engine_neutral_resync(data_dir, dry_run=False, verbose=True)

        assert result.error is None
        behavior_rules = (prompts_dir / "behavior_rules.md").read_text(encoding="utf-8")
        environment = (prompts_dir / "environment.md").read_text(encoding="utf-8")
        assert "Glob" not in behavior_rules
        assert "検索ツール" in behavior_rules
        # Task deadlines were torn out of environment.md (A1 task-model teardown);
        # resync should propagate the current template, not the retired section.
        assert "タスク期限" not in environment
        assert "行動の基本原則" in environment
        assert "AI-speed" not in environment
        assert not (prompts_dir / "task_delegation_rules.md").exists(), (
            "stale prompts/task_delegation_rules.md should be removed"
        )

    def test_step_v0120_dry_run_keeps_files(self, data_dir: Path) -> None:
        from core.migrations.steps import step_v0120_prompt_deadline_engine_neutral_resync

        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "behavior_rules.md").write_text(b"stale: `Glob`".decode("utf-8"), encoding="utf-8")
        (prompts_dir / "task_delegation_rules.md").write_text("old content", encoding="utf-8")

        result = step_v0120_prompt_deadline_engine_neutral_resync(data_dir, dry_run=True, verbose=True)

        assert (prompts_dir / "behavior_rules.md").exists()
        assert (prompts_dir / "task_delegation_rules.md").exists()
        assert any("Would remove stale prompts/task_delegation_rules.md" in d for d in result.details)

    def test_step_v0120_registered_before_update_version(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        ids = [item["id"] for item in runner.list_steps()]

        assert "v0120_prompt_deadline_engine_neutral_resync" in ids
        assert ids.index("v0120_prompt_deadline_engine_neutral_resync") < ids.index("update_version")

    def test_step_common_knowledge_team_design_removes_stale_machine_docs(self, data_dir: Path) -> None:
        from core.migrations.registry import StepResult
        from core.migrations.steps import step_common_knowledge_team_design_resync

        ops = data_dir / "common_knowledge" / "operations"
        ops.mkdir(parents=True)
        stale = ops / "machine-tool-usage.md"
        stale.write_text("legacy", encoding="utf-8")

        with patch(
            "core.migrations.steps.step_common_knowledge_resync",
            return_value=StepResult(changed=0, skipped=0, details=[]),
        ):
            result = step_common_knowledge_team_design_resync(data_dir, dry_run=False, verbose=True)
        assert not stale.exists()
        assert result.changed >= 1
        assert any("Removed stale" in d for d in result.details)

    def test_step_common_knowledge_team_design_dry_run_keeps_stale(self, data_dir: Path) -> None:
        from core.migrations.registry import StepResult
        from core.migrations.steps import step_common_knowledge_team_design_resync

        ops = data_dir / "common_knowledge" / "operations"
        ops.mkdir(parents=True)
        stale = ops / "machine-workflow-tester.md"
        stale.write_text("legacy", encoding="utf-8")

        with patch(
            "core.migrations.steps.step_common_knowledge_resync",
            return_value=StepResult(changed=0, skipped=0, details=[]),
        ):
            result = step_common_knowledge_team_design_resync(data_dir, dry_run=True, verbose=True)
        assert stale.exists()
        assert any("Would remove stale" in d for d in result.details)

    def test_step_remove_team_design_removes_deployed_trees(self, data_dir: Path) -> None:
        from core.migrations.registry import StepResult
        from core.migrations.steps import step_remove_team_design

        top = data_dir / "common_knowledge" / "team-design" / "legal"
        top.mkdir(parents=True)
        (top / "team.md").write_text("legacy", encoding="utf-8")
        per_anima = data_dir / "animas" / "mei" / "common_knowledge" / "team-design"
        per_anima.mkdir(parents=True)
        (per_anima / "guide.md").write_text("legacy", encoding="utf-8")
        keep = data_dir / "animas" / "mei" / "common_knowledge" / "operations"
        keep.mkdir(parents=True)
        (keep / "keep.md").write_text("keep", encoding="utf-8")

        with patch(
            "core.migrations.steps.step_common_knowledge_resync",
            return_value=StepResult(changed=0, skipped=0, details=[]),
        ):
            result = step_remove_team_design(data_dir, dry_run=False, verbose=True)

        assert not top.parent.exists()
        assert not per_anima.exists()
        assert (keep / "keep.md").exists()
        assert result.changed >= 2

    def test_step_remove_team_design_dry_run_keeps_trees(self, data_dir: Path) -> None:
        from core.migrations.registry import StepResult
        from core.migrations.steps import step_remove_team_design

        tree = data_dir / "common_knowledge" / "team-design"
        tree.mkdir(parents=True)
        (tree / "guide.md").write_text("legacy", encoding="utf-8")

        with patch(
            "core.migrations.steps.step_common_knowledge_resync",
            return_value=StepResult(changed=0, skipped=0, details=[]),
        ):
            result = step_remove_team_design(data_dir, dry_run=True, verbose=True)

        assert tree.exists()
        assert any("Would remove" in d for d in result.details)


# ── CLI tests ───────────────────────────────────────────────


class TestMigrateCLI:
    def test_register_command(self) -> None:
        import argparse

        from cli.commands.migrate_cmd import register_migrate_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_migrate_command(sub)
        args = parser.parse_args(["migrate", "--list"])
        assert args.list is True

    def test_register_command_dry_run(self) -> None:
        import argparse

        from cli.commands.migrate_cmd import register_migrate_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_migrate_command(sub)
        args = parser.parse_args(["migrate", "--dry-run", "--verbose"])
        assert args.dry_run is True
        assert args.verbose is True


# ── Integration: register_all_steps ─────────────────────────


class TestRegisterAllSteps:
    def test_register_all_steps_count(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        steps = runner.list_steps()
        assert len(steps) >= 20

    def test_all_step_ids_unique(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        ids = [s["id"] for s in runner.list_steps()]
        assert len(ids) == len(set(ids))

    def test_memory_hygiene_prompt_resync_has_new_migration_id(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps, step_prompt_resync

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        runner.tracker.mark_applied("prompt_resync")

        step = next(s for s in runner._steps if s.id == "memory_hygiene_prompt_resync_20260718")
        assert step.category == "template_sync"
        assert step.fn is step_prompt_resync
        listed = {item["id"]: item for item in runner.list_steps()}
        assert listed["prompt_resync"]["applied"] is True
        assert listed["memory_hygiene_prompt_resync_20260718"]["applied"] is False

    def test_all_categories_present(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        categories = {s["category"] for s in runner.list_steps()}
        assert "structural" in categories
        assert "per_anima" in categories
        assert "template_sync" in categories
        assert "version" in categories
