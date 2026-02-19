"""Unit tests for core/org_sync.py — org structure synchronization."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.models import (
    AnimaWorksConfig,
    AnimaModelConfig,
    invalidate_cache,
    load_config,
    save_config,
)
from core.org_sync import (
    _detect_circular_references,
    sync_org_structure,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Invalidate the config singleton before and after each test."""
    invalidate_cache()
    yield  # type: ignore[misc]
    invalidate_cache()


@pytest.fixture
def animas_dir(tmp_path: Path) -> Path:
    """Create an empty animas directory."""
    d = tmp_path / "animas"
    d.mkdir()
    return d


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Create a minimal config.json and return its path."""
    p = tmp_path / "config.json"
    cfg = AnimaWorksConfig(setup_complete=True)
    save_config(cfg, p)
    invalidate_cache()
    return p


def _make_anima(
    animas_dir: Path,
    name: str,
    identity_content: str = "# Identity\nNo supervisor info.",
    status_json: dict | None = None,
) -> Path:
    """Helper to create an anima directory with identity.md and optional status.json."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(identity_content, encoding="utf-8")
    if status_json is not None:
        (anima_dir / "status.json").write_text(
            json.dumps(status_json, ensure_ascii=False), encoding="utf-8",
        )
    return anima_dir


# ── _detect_circular_references ──────────────────────────────────


class TestDetectCircularReferences:
    """Tests for circular supervisor reference detection."""

    def test_no_cycle(self) -> None:
        """No cycles in a simple chain."""
        rels = {"a": "b", "b": "c", "c": None}
        assert _detect_circular_references(rels) == []

    def test_simple_cycle(self) -> None:
        """A->B->A cycle is detected."""
        rels = {"a": "b", "b": "a"}
        cycles = _detect_circular_references(rels)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}

    def test_three_node_cycle(self) -> None:
        """A->B->C->A cycle is detected."""
        rels = {"a": "b", "b": "c", "c": "a"}
        cycles = _detect_circular_references(rels)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_all_none(self) -> None:
        """No cycles when all supervisors are None."""
        rels = {"a": None, "b": None, "c": None}
        assert _detect_circular_references(rels) == []

    def test_self_reference(self) -> None:
        """A->A self-reference is detected."""
        rels = {"a": "a"}
        cycles = _detect_circular_references(rels)
        assert len(cycles) == 1
        assert cycles[0] == ("a",)

    def test_cycle_with_tail(self) -> None:
        """Chain leading into a cycle: D->A->B->A."""
        rels = {"d": "a", "a": "b", "b": "a"}
        cycles = _detect_circular_references(rels)
        assert len(cycles) >= 1
        # The cycle itself should contain a and b
        cycle_members = set()
        for c in cycles:
            cycle_members.update(c)
        assert {"a", "b"}.issubset(cycle_members)

    def test_disconnected_graph_no_cycle(self) -> None:
        """Multiple disconnected chains without cycles."""
        rels = {"a": "b", "b": None, "c": "d", "d": None}
        assert _detect_circular_references(rels) == []

    def test_empty_relationships(self) -> None:
        """Empty input returns no cycles."""
        assert _detect_circular_references({}) == []


# ── sync_org_structure ───────────────────────────────────────────


class TestSyncOrgStructure:
    """Tests for the main sync_org_structure function."""

    def test_empty_animas_dir(self, animas_dir: Path, config_path: Path) -> None:
        """Returns empty dict for empty animas directory."""
        result = sync_org_structure(animas_dir, config_path)
        assert result == {}

    def test_nonexistent_dir(self, tmp_path: Path, config_path: Path) -> None:
        """Returns empty dict when animas dir doesn't exist."""
        result = sync_org_structure(tmp_path / "nonexistent", config_path)
        assert result == {}

    def test_adds_new_anima_to_config(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Anima on disk but not in config gets added."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        result = sync_org_structure(animas_dir, config_path)

        assert result == {"alice": "bob"}
        invalidate_cache()
        cfg = load_config(config_path)
        assert "alice" in cfg.animas
        assert cfg.animas["alice"].supervisor == "bob"

    def test_fills_none_supervisor(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Config entry with supervisor=None gets filled from identity.md."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        # Pre-populate config with supervisor=None
        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(supervisor=None)
        save_config(cfg, config_path)
        invalidate_cache()

        result = sync_org_structure(animas_dir, config_path)

        assert result["alice"] == "bob"
        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "bob"

    def test_does_not_overwrite_existing_supervisor(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Config entry with existing supervisor is NOT overwritten."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        # Pre-populate config with a different supervisor
        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(supervisor="charlie")
        save_config(cfg, config_path)
        invalidate_cache()

        sync_org_structure(animas_dir, config_path)

        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "charlie"  # Unchanged

    def test_no_change_when_already_matched(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """No config write when config already matches disk."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(supervisor="bob")
        save_config(cfg, config_path)
        invalidate_cache()

        # Record mtime
        mtime_before = config_path.stat().st_mtime_ns

        sync_org_structure(animas_dir, config_path)

        # Config should not have been rewritten
        mtime_after = config_path.stat().st_mtime_ns
        assert mtime_before == mtime_after

    def test_multiple_animas(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Multiple animas are all discovered and synced."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")
        _make_anima(animas_dir, "bob", "| 上司 | (なし) |\n")
        _make_anima(animas_dir, "charlie", "| 上司 | alice |\n")

        result = sync_org_structure(animas_dir, config_path)

        assert result == {"alice": "bob", "bob": None, "charlie": "alice"}
        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "bob"
        assert cfg.animas["bob"].supervisor is None
        assert cfg.animas["charlie"].supervisor == "alice"

    def test_fallback_to_status_json(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Falls back to status.json when identity.md has no supervisor."""
        _make_anima(
            animas_dir,
            "alice",
            "# Identity: alice\nNo table here.",
            status_json={"supervisor": "bob"},
        )

        result = sync_org_structure(animas_dir, config_path)

        assert result["alice"] == "bob"
        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "bob"

    def test_identity_takes_priority_over_status(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """status.json supervisor is checked first by read_anima_supervisor."""
        # Note: read_anima_supervisor checks status.json first, then identity.md.
        # If status.json has a supervisor, that wins.
        _make_anima(
            animas_dir,
            "alice",
            "| 上司 | charlie |\n",
            status_json={"supervisor": "bob"},
        )

        result = sync_org_structure(animas_dir, config_path)

        # status.json takes priority in read_anima_supervisor
        assert result["alice"] == "bob"

    def test_circular_reference_skipped(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Animas involved in circular references are not synced."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")
        _make_anima(animas_dir, "bob", "| 上司 | alice |\n")
        _make_anima(animas_dir, "charlie", "| 上司 | (なし) |\n")

        sync_org_structure(animas_dir, config_path)

        invalidate_cache()
        cfg = load_config(config_path)
        # alice and bob should not be in config (circular)
        assert "alice" not in cfg.animas
        assert "bob" not in cfg.animas
        # charlie should be added normally
        assert "charlie" in cfg.animas

    def test_skips_non_anima_directories(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Directories without identity.md are skipped."""
        (animas_dir / "not-an-anima").mkdir()
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        result = sync_org_structure(animas_dir, config_path)

        assert "not-an-anima" not in result
        assert "alice" in result

    def test_skips_files_in_animas_dir(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Regular files in animas_dir are ignored."""
        (animas_dir / "README.md").write_text("ignore me", encoding="utf-8")
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        result = sync_org_structure(animas_dir, config_path)

        assert len(result) == 1
        assert "alice" in result

    def test_fullwidth_paren_name_resolution(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Japanese name with full-width parens is resolved to English."""
        _make_anima(
            animas_dir,
            "hinata",
            "| 上司 | 琴葉（kotoha） |\n",
        )

        result = sync_org_structure(animas_dir, config_path)

        assert result["hinata"] == "kotoha"
        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["hinata"].supervisor == "kotoha"

    def test_halfwidth_paren_name_resolution(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Japanese name with half-width parens is resolved to English."""
        _make_anima(
            animas_dir,
            "hinata",
            "| 上司 | 琴葉(kotoha) |\n",
        )

        result = sync_org_structure(animas_dir, config_path)

        assert result["hinata"] == "kotoha"

    def test_anima_with_no_supervisor_added_as_none(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Anima without supervisor info gets added with supervisor=None."""
        _make_anima(
            animas_dir,
            "alice",
            "# Identity: alice\nNo supervisor table.\n",
        )

        result = sync_org_structure(animas_dir, config_path)

        assert result["alice"] is None
        invalidate_cache()
        cfg = load_config(config_path)
        assert "alice" in cfg.animas
        assert cfg.animas["alice"].supervisor is None

    def test_preserves_existing_anima_config_fields(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Syncing supervisor preserves other AnimaModelConfig fields."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        # Pre-populate with extra fields
        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(
            model="openai/gpt-4o",
            max_tokens=8192,
            supervisor=None,
        )
        save_config(cfg, config_path)
        invalidate_cache()

        sync_org_structure(animas_dir, config_path)

        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "bob"
        assert cfg.animas["alice"].model == "openai/gpt-4o"
        assert cfg.animas["alice"].max_tokens == 8192

    def test_mismatch_logs_warning(
        self, animas_dir: Path, config_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Mismatched supervisors log a warning but keep config value."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")

        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(supervisor="charlie")
        save_config(cfg, config_path)
        invalidate_cache()

        with caplog.at_level("WARNING"):
            sync_org_structure(animas_dir, config_path)

        assert "supervisor mismatch" in caplog.text.lower()
        assert "alice" in caplog.text

    def test_circular_reference_logs_warning(
        self, animas_dir: Path, config_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Circular references are logged as warnings."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")
        _make_anima(animas_dir, "bob", "| 上司 | alice |\n")

        with caplog.at_level("WARNING"):
            sync_org_structure(animas_dir, config_path)

        assert "circular" in caplog.text.lower()

    def test_nashi_values_produce_none_supervisor(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Various 'none' values in identity.md produce supervisor=None."""
        _make_anima(animas_dir, "a", "| 上司 | なし |\n")
        _make_anima(animas_dir, "b", "| 上司 | (なし) |\n")
        _make_anima(animas_dir, "c", "| 上司 | - |\n")

        result = sync_org_structure(animas_dir, config_path)

        assert result["a"] is None
        assert result["b"] is None
        assert result["c"] is None

    def test_realistic_identity_md(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Supervisor extracted from a realistic multi-section identity.md."""
        identity = (
            "# Identity: hinata\n\n"
            "あなたの名前は日向 ひなた。\n\n"
            "## 基本プロフィール\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 誕生日 | 3月21日 |\n"
            "| 星座 | 牡羊座 |\n"
            "| 上司 | 凛堂 凛（rin） |\n"
            "| 身長 | 155cm |\n\n"
            "## 性格特性\n\n元気で前向き。\n"
        )
        _make_anima(animas_dir, "hinata", identity)

        result = sync_org_structure(animas_dir, config_path)

        assert result["hinata"] == "rin"

    def test_return_value_includes_all_animas(
        self, animas_dir: Path, config_path: Path,
    ) -> None:
        """Return value includes every discovered anima, not just changed ones."""
        _make_anima(animas_dir, "alice", "| 上司 | bob |\n")
        _make_anima(animas_dir, "bob", "# No supervisor.\n")

        # Pre-populate alice so she won't be changed
        cfg = load_config(config_path)
        cfg.animas["alice"] = AnimaModelConfig(supervisor="bob")
        save_config(cfg, config_path)
        invalidate_cache()

        result = sync_org_structure(animas_dir, config_path)

        # Both animas returned even though only bob was newly added
        assert "alice" in result
        assert "bob" in result
