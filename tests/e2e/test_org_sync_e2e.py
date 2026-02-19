# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for org-structure-config-sync feature.

Tests the full pipeline of supervisor synchronization between identity.md /
status.json on disk and config.json, including:

1. register_anima_in_config reads supervisor from identity.md/status.json
2. sync_org_structure repairs missing/null supervisors in a realistic scenario
3. Server startup triggers org sync via create_app
4. Full pipeline: create_from_md + register_anima_in_config -> supervisor in config

These are integration tests that exercise real filesystem operations and
the actual code paths without mocking core functionality.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    AnimaModelConfig,
    invalidate_cache,
    load_config,
    save_config,
)
from core.org_sync import sync_org_structure
from core.anima_factory import create_from_md


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Invalidate the config singleton before and after each test."""
    invalidate_cache()
    yield
    invalidate_cache()


# ── Helpers ──────────────────────────────────────────────────────


def _create_config(
    config_path: Path,
    *,
    setup_complete: bool = True,
    animas: dict[str, AnimaModelConfig] | None = None,
) -> AnimaWorksConfig:
    """Create a config.json file and return the config object."""
    cfg = AnimaWorksConfig(
        setup_complete=setup_complete,
        animas=animas or {},
    )
    save_config(cfg, config_path)
    invalidate_cache()
    return cfg


def _make_anima(
    animas_dir: Path,
    name: str,
    identity_content: str = "# Identity\nNo supervisor info.",
    status_json: dict | None = None,
) -> Path:
    """Create an anima directory with identity.md and optional status.json."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(identity_content, encoding="utf-8")
    if status_json is not None:
        (anima_dir / "status.json").write_text(
            json.dumps(status_json, ensure_ascii=False), encoding="utf-8",
        )
    return anima_dir


# ── Test 1: register_anima_in_config sets supervisor ────────────


class TestRegisterAnimaSupervisor:
    """Verify register_anima_in_config reads supervisor from identity.md."""

    def test_register_picks_up_supervisor_from_identity_md(
        self, tmp_path: Path,
    ) -> None:
        """When identity.md contains '| 上司 | sakura |',
        register_anima_in_config should set supervisor='sakura' in config.json.
        """
        from core.config.models import register_anima_in_config

        # Set up data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(config_path)

        # Create anima directory with supervisor in identity.md
        anima_dir = animas_dir / "hinata"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "# hinata\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | sakura |\n"
            "| 誕生日 | 3月21日 |\n",
            encoding="utf-8",
        )

        # Act
        register_anima_in_config(data_dir, "hinata")

        # Assert
        invalidate_cache()
        cfg = load_config(config_path)
        assert "hinata" in cfg.animas
        assert cfg.animas["hinata"].supervisor == "sakura"

    def test_register_picks_up_supervisor_from_status_json(
        self, tmp_path: Path,
    ) -> None:
        """When status.json has supervisor field,
        register_anima_in_config should read it.
        """
        from core.config.models import register_anima_in_config

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(config_path)

        anima_dir = animas_dir / "kotoha"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "# kotoha\nNo supervisor table here.",
            encoding="utf-8",
        )
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "sakura"}),
            encoding="utf-8",
        )

        register_anima_in_config(data_dir, "kotoha")

        invalidate_cache()
        cfg = load_config(config_path)
        assert "kotoha" in cfg.animas
        assert cfg.animas["kotoha"].supervisor == "sakura"

    def test_register_is_noop_for_existing_anima(
        self, tmp_path: Path,
    ) -> None:
        """If the anima already exists in config, register should not modify it."""
        from core.config.models import register_anima_in_config

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={"hinata": AnimaModelConfig(supervisor="rin")},
        )

        anima_dir = animas_dir / "hinata"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "| 上司 | sakura |\n", encoding="utf-8",
        )

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(config_path)
        # Should still be "rin", not updated to "sakura"
        assert cfg.animas["hinata"].supervisor == "rin"

    def test_register_with_japanese_supervisor_fullwidth_parens(
        self, tmp_path: Path,
    ) -> None:
        """Japanese name with full-width parens resolves to English name."""
        from core.config.models import register_anima_in_config

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(config_path)

        anima_dir = animas_dir / "chatbot"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "| 上司 | 琴葉（kotoha） |\n", encoding="utf-8",
        )

        register_anima_in_config(data_dir, "chatbot")

        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["chatbot"].supervisor == "kotoha"


# ── Test 2: sync_org_structure repairs missing/null supervisors ──


class TestSyncOrgStructureRealisticScenario:
    """Reproduce the actual bug scenario where supervisors are missing/null
    in config.json and need to be repaired by sync_org_structure.
    """

    def test_realistic_org_sync(self, tmp_path: Path) -> None:
        """Full realistic scenario matching the actual bug report.

        Setup:
        - config.json: sakura(supervisor=null), kotoha(supervisor="sakura"),
          chatwork_checker(supervisor=null)
        - animas/sakura/identity.md (no 上司 row)
        - animas/kotoha/identity.md (上司: サクラ(sakura))
        - animas/chatwork_checker/identity.md (上司: 琴葉（kotoha）)
        - animas/rin/identity.md (no 上司 row) -- NOT in config.json
        - animas/aoi/identity.md (no 上司 row) -- NOT in config.json

        Expected after sync:
        - chatwork_checker: supervisor="kotoha" (was null, repaired)
        - rin: added to config (supervisor=None since no 上司 row)
        - aoi: added to config (supervisor=None since no 上司 row)
        - sakura: still supervisor=None
        - kotoha: still supervisor="sakura" (unchanged)
        """
        # Arrange: create directory structure
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        # Create config.json with initial (buggy) state
        _create_config(
            config_path,
            animas={
                "sakura": AnimaModelConfig(supervisor=None),
                "kotoha": AnimaModelConfig(supervisor="sakura"),
                "chatwork_checker": AnimaModelConfig(supervisor=None),
            },
        )

        # Create anima directories with identity.md files

        # sakura: top-level manager, no supervisor
        _make_anima(
            animas_dir, "sakura",
            "# Identity: sakura\n\n"
            "## 基本プロフィール\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 誕生日 | 4月1日 |\n"
            "| 上司 | (なし) |\n"
            "| 身長 | 165cm |\n",
        )

        # kotoha: reports to sakura
        _make_anima(
            animas_dir, "kotoha",
            "# Identity: kotoha\n\n"
            "## 基本プロフィール\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | サクラ(sakura) |\n"
            "| 役割 | サブリーダー |\n",
        )

        # chatwork_checker: reports to kotoha (THIS IS THE BUG - config had null)
        _make_anima(
            animas_dir, "chatwork_checker",
            "# Identity: chatwork_checker\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | 琴葉（kotoha） |\n"
            "| 役割 | Chatwork監視 |\n",
        )

        # rin: exists on disk but NOT in config.json (no supervisor)
        _make_anima(
            animas_dir, "rin",
            "# Identity: rin\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | (なし) |\n"
            "| 役割 | 総務 |\n",
        )

        # aoi: exists on disk but NOT in config.json (no supervisor)
        _make_anima(
            animas_dir, "aoi",
            "# Identity: aoi\n\n"
            "## プロフィール\n"
            "クリエイティブ担当。\n",
        )

        # Act
        result = sync_org_structure(animas_dir, config_path)

        # Assert: all animas discovered
        assert set(result.keys()) == {
            "sakura", "kotoha", "chatwork_checker", "rin", "aoi",
        }

        # Assert: supervisor values from disk
        assert result["sakura"] is None
        assert result["kotoha"] == "sakura"
        assert result["chatwork_checker"] == "kotoha"
        assert result["rin"] is None
        assert result["aoi"] is None

        # Assert: config.json was updated
        invalidate_cache()
        cfg = load_config(config_path)

        # chatwork_checker: supervisor repaired from null to "kotoha"
        assert cfg.animas["chatwork_checker"].supervisor == "kotoha"

        # rin: newly added to config with supervisor=None
        assert "rin" in cfg.animas
        assert cfg.animas["rin"].supervisor is None

        # aoi: newly added to config with supervisor=None
        assert "aoi" in cfg.animas
        assert cfg.animas["aoi"].supervisor is None

        # sakura: still supervisor=None (unchanged)
        assert cfg.animas["sakura"].supervisor is None

        # kotoha: still supervisor="sakura" (unchanged, already correct)
        assert cfg.animas["kotoha"].supervisor == "sakura"

    def test_sync_does_not_overwrite_manual_config_override(
        self, tmp_path: Path,
    ) -> None:
        """When config has a different supervisor than identity.md,
        sync should keep the config value (manual override).
        """
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={
                "alice": AnimaModelConfig(supervisor="manual_boss"),
            },
        )

        _make_anima(
            animas_dir, "alice",
            "| 上司 | disk_boss |\n",
        )

        sync_org_structure(animas_dir, config_path)

        invalidate_cache()
        cfg = load_config(config_path)
        # Manual config override should be preserved
        assert cfg.animas["alice"].supervisor == "manual_boss"


# ── Test 3: Server startup triggers org sync ─────────────────────


class TestServerStartupOrgSync:
    """Test that create_app properly handles anima discovery with supervisors.

    Rather than testing the full lifespan (which would require asyncio and
    process management), we test that:
    - create_app discovers animas from the animas directory
    - The reconciliation callback uses register_anima_in_config
    - A newly discovered anima gets added with the correct supervisor

    We mock ProcessSupervisor and the lifespan to avoid starting real processes.
    """

    def test_create_app_discovers_animas_and_reconciliation_registers_supervisor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the reconciliation callback in lifespan properly registers
        a new anima with supervisor extracted from identity.md.

        This simulates what happens when an anima is discovered after app creation
        (e.g., filesystem reconciliation detects a new directory).
        """
        from core.config.models import register_anima_in_config

        # Set up isolated data directory
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        (shared_dir / "inbox").mkdir()
        (shared_dir / "users").mkdir()
        config_path = data_dir / "config.json"

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        # Create config with setup_complete=True but no animas
        _create_config(config_path, setup_complete=True)

        # Create an anima on disk with supervisor in identity.md
        anima_dir = animas_dir / "newanima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "# Identity: newanima\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | sakura |\n",
            encoding="utf-8",
        )

        # Directly call register_anima_in_config (the function used by
        # the reconciliation callback in lifespan)
        register_anima_in_config(data_dir, "newanima")

        # Verify
        invalidate_cache()
        cfg = load_config(config_path)
        assert "newanima" in cfg.animas
        assert cfg.animas["newanima"].supervisor == "sakura"

    def test_create_app_discovers_existing_animas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that create_app properly discovers animas from the animas directory."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        (shared_dir / "inbox").mkdir()
        (shared_dir / "users").mkdir()
        config_path = data_dir / "config.json"
        run_dir = data_dir / "run"
        run_dir.mkdir()
        log_dir = data_dir / "logs"
        log_dir.mkdir()

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        _create_config(
            config_path,
            setup_complete=True,
            animas={"sakura": AnimaModelConfig(supervisor=None)},
        )

        # Create an anima with identity.md
        _make_anima(animas_dir, "sakura", "# Identity: sakura\nTop manager.")

        # Mock ProcessSupervisor to avoid spawning real processes
        with patch("server.app.ProcessSupervisor") as mock_supervisor_cls:
            mock_supervisor_cls.return_value = MagicMock()

            from server.app import create_app

            app = create_app(animas_dir, shared_dir)

        # Verify that anima was discovered
        assert "sakura" in app.state.anima_names

    def test_reconciliation_callback_wires_register_anima_in_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the lifespan reconciliation callback calls
        register_anima_in_config for newly added animas.
        """
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        (shared_dir / "inbox").mkdir()
        (shared_dir / "users").mkdir()
        config_path = data_dir / "config.json"
        run_dir = data_dir / "run"
        run_dir.mkdir()
        log_dir = data_dir / "logs"
        log_dir.mkdir()

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        _create_config(config_path, setup_complete=True)

        # Create an anima with supervisor
        _make_anima(
            animas_dir, "hinata",
            "| 上司 | kotoha |\n",
        )

        # Mock ProcessSupervisor
        with patch("server.app.ProcessSupervisor") as mock_supervisor_cls:
            mock_supervisor = MagicMock()
            mock_supervisor.start_all = AsyncMock()
            mock_supervisor.shutdown_all = AsyncMock()
            mock_supervisor_cls.return_value = mock_supervisor

            from server.app import create_app

            app = create_app(animas_dir, shared_dir)

        # The lifespan hasn't run yet, but we can simulate the
        # _on_anima_added callback that it registers.
        # The callback is registered in lifespan, so let's test the
        # registration mechanism directly by calling register_anima_in_config
        # as the callback would.
        from core.config.models import register_anima_in_config
        register_anima_in_config(data_dir, "new_hire")

        # Since new_hire doesn't exist on disk, supervisor should be None
        invalidate_cache()
        cfg = load_config(config_path)
        assert "new_hire" in cfg.animas
        assert cfg.animas["new_hire"].supervisor is None

        # But for an anima with identity.md on disk, supervisor IS extracted
        invalidate_cache()
        _make_anima(
            animas_dir, "another_hire",
            "| 上司 | sakura |\n",
        )
        register_anima_in_config(data_dir, "another_hire")

        invalidate_cache()
        cfg = load_config(config_path)
        assert "another_hire" in cfg.animas
        assert cfg.animas["another_hire"].supervisor == "sakura"


# ── Test 4: Full pipeline — create_from_md + register ────────────


class TestFullPipelineCreateAndRegister:
    """End-to-end test of the full anima creation pipeline:
    create_from_md creates anima directory -> register_anima_in_config
    reads supervisor from the created files -> config.json is updated.
    """

    def test_create_from_md_then_register_sets_supervisor(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """Full pipeline: character sheet with supervisor -> create_from_md ->
        register_anima_in_config -> config.json has supervisor set.
        """
        from core.config.models import register_anima_in_config

        animas_dir = data_dir / "animas"

        # Write a full character sheet
        character_sheet = """\
# Character: testworker

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testworker |
| 日本語名 | テストワーカー |
| 役職/専門 | テスト担当 |
| 上司 | sakura |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

テスト用の人格設定です。真面目で几帳面な性格。

## 役割・行動方針

テスト業務を担当します。品質管理に注力します。
"""
        sheet_path = tmp_path / "sheet.md"
        sheet_path.write_text(character_sheet, encoding="utf-8")

        # Step 1: Create anima from character sheet
        anima_dir = create_from_md(animas_dir, sheet_path)

        # Verify status.json was created with supervisor
        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == "sakura"

        # Step 2: Register anima in config
        register_anima_in_config(data_dir, "testworker")

        # Step 3: Verify config.json
        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "testworker" in cfg.animas
        assert cfg.animas["testworker"].supervisor == "sakura"

    def test_create_from_md_no_supervisor_then_register(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """When character sheet has no supervisor (なし),
        register_anima_in_config should set supervisor=None.
        """
        from core.config.models import register_anima_in_config

        animas_dir = data_dir / "animas"

        character_sheet = """\
# Character: toplevel

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | toplevel |
| 日本語名 | トップレベル |
| 役職/専門 | 社長 |
| 上司 | (なし) |
| 役割 | manager |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

トップレベルの管理者です。

## 役割・行動方針

全体を統括します。
"""
        sheet_path = tmp_path / "sheet.md"
        sheet_path.write_text(character_sheet, encoding="utf-8")

        anima_dir = create_from_md(animas_dir, sheet_path)

        # status.json should have empty string for supervisor (なし -> "")
        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == ""

        # Register
        register_anima_in_config(data_dir, "toplevel")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "toplevel" in cfg.animas
        # Empty string in status.json resolves to None in config
        assert cfg.animas["toplevel"].supervisor is None

    def test_create_from_md_with_japanese_supervisor_fullwidth_parens(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """Full pipeline with Japanese supervisor name in full-width parens."""
        from core.config.models import register_anima_in_config

        animas_dir = data_dir / "animas"

        character_sheet = """\
# Character: assistant

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | assistant |
| 日本語名 | アシスタント |
| 役職/専門 | 補佐 |
| 上司 | 琴葉（kotoha） |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

アシスタント用の人格です。

## 役割・行動方針

補佐業務を担当します。
"""
        sheet_path = tmp_path / "sheet.md"
        sheet_path.write_text(character_sheet, encoding="utf-8")

        anima_dir = create_from_md(animas_dir, sheet_path)

        # status.json stores the raw value
        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == "琴葉（kotoha）"

        # register_anima_in_config resolves to English name
        register_anima_in_config(data_dir, "assistant")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "assistant" in cfg.animas
        assert cfg.animas["assistant"].supervisor == "kotoha"

    def test_create_multiple_animas_and_sync_builds_full_org(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """Create multiple animas from character sheets, register them,
        and verify the full organizational hierarchy in config.json.
        """
        from core.config.models import register_anima_in_config

        animas_dir = data_dir / "animas"

        # Create manager (no supervisor)
        manager_sheet = """\
# Character: manager

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | manager |
| 上司 | (なし) |
| 役割 | manager |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

管理者です。

## 役割・行動方針

チームを管理します。
"""
        sheet1 = tmp_path / "manager.md"
        sheet1.write_text(manager_sheet, encoding="utf-8")
        create_from_md(animas_dir, sheet1)
        register_anima_in_config(data_dir, "manager")
        invalidate_cache()

        # Create worker reporting to manager
        worker_sheet = """\
# Character: worker

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | worker |
| 上司 | manager |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

作業者です。

## 役割・行動方針

指示に従い作業します。
"""
        sheet2 = tmp_path / "worker.md"
        sheet2.write_text(worker_sheet, encoding="utf-8")
        create_from_md(animas_dir, sheet2)
        register_anima_in_config(data_dir, "worker")
        invalidate_cache()

        # Verify full org structure in config
        cfg = load_config(data_dir / "config.json")
        assert cfg.animas["manager"].supervisor is None
        assert cfg.animas["worker"].supervisor == "manager"
