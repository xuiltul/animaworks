from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified migration framework."""

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

    def test_step_models_json_create_skip_existing(self, data_dir: Path) -> None:
        from core.migrations.steps import step_models_json_create

        (data_dir / "models.json").write_text("{}", encoding="utf-8")
        result = step_models_json_create(data_dir, dry_run=False, verbose=True)
        assert result.skipped == 1

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

    def test_step_stale_sections_cleanup_no_db(self, data_dir: Path) -> None:
        from core.migrations.steps import step_stale_sections_cleanup

        result = step_stale_sections_cleanup(data_dir, dry_run=False, verbose=True)
        assert result.skipped == 1

    def test_step_update_version(self, data_dir: Path) -> None:
        from core.migrations.steps import step_update_version

        result = step_update_version(data_dir, dry_run=False, verbose=True)
        assert result.changed == 1

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

    def test_all_categories_present(self, tmp_path: Path) -> None:
        from core.migrations.steps import register_all_steps

        runner = MigrationRunner(tmp_path)
        register_all_steps(runner)
        categories = {s["category"] for s in runner.list_steps()}
        assert "structural" in categories
        assert "per_anima" in categories
        assert "template_sync" in categories
        assert "db_sync" in categories
        assert "version" in categories
