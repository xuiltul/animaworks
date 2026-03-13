"""Tests for CJK-Latin fuzzy edit matching in handler_files.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler_files import (
    _build_fuzzy_cjk_latin_pattern,
    _is_cjk,
    _is_latin_or_digit,
)


# ── _is_cjk ──────────────────────────────────────────────


class TestIsCjk:
    def test_kanji(self) -> None:
        assert _is_cjk("漢")
        assert _is_cjk("字")

    def test_hiragana(self) -> None:
        assert _is_cjk("あ")
        assert _is_cjk("の")

    def test_katakana(self) -> None:
        assert _is_cjk("ア")
        assert _is_cjk("ン")

    def test_cjk_punctuation(self) -> None:
        assert _is_cjk("。")
        assert _is_cjk("、")

    def test_fullwidth(self) -> None:
        assert _is_cjk("Ａ")
        assert _is_cjk("１")

    def test_ascii_not_cjk(self) -> None:
        assert not _is_cjk("A")
        assert not _is_cjk("1")
        assert not _is_cjk(" ")
        assert not _is_cjk("-")


# ── _is_latin_or_digit ───────────────────────────────────


class TestIsLatinOrDigit:
    def test_letters(self) -> None:
        assert _is_latin_or_digit("a")
        assert _is_latin_or_digit("Z")

    def test_digits(self) -> None:
        assert _is_latin_or_digit("0")
        assert _is_latin_or_digit("9")

    def test_hyphen_underscore(self) -> None:
        assert _is_latin_or_digit("-")
        assert _is_latin_or_digit("_")

    def test_cjk_not_latin(self) -> None:
        assert not _is_latin_or_digit("あ")
        assert not _is_latin_or_digit("漢")

    def test_space_not_latin(self) -> None:
        assert not _is_latin_or_digit(" ")

    def test_special_chars(self) -> None:
        assert not _is_latin_or_digit("(")
        assert not _is_latin_or_digit(".")


# ── _build_fuzzy_cjk_latin_pattern ───────────────────────


class TestBuildFuzzyPattern:
    def test_empty_returns_none(self) -> None:
        assert _build_fuzzy_cjk_latin_pattern("") is None

    def test_pure_ascii_returns_none(self) -> None:
        assert _build_fuzzy_cjk_latin_pattern("hello world") is None

    def test_pure_cjk_returns_none(self) -> None:
        assert _build_fuzzy_cjk_latin_pattern("こんにちは世界") is None

    def test_model_adds_space_around_latin(self) -> None:
        """Model: 'shino の分析' → file: 'shinoの分析'"""
        pat = _build_fuzzy_cjk_latin_pattern("shino の分析")
        assert pat is not None
        assert pat.search("shinoの分析")
        assert pat.search("shino の分析")

    def test_model_removes_space(self) -> None:
        """Model: 'shinoの分析' → file: 'shino の分析'"""
        pat = _build_fuzzy_cjk_latin_pattern("shinoの分析")
        assert pat is not None
        assert pat.search("shino の分析")
        assert pat.search("shinoの分析")

    def test_dry_run_boundary(self) -> None:
        """Model: 'dry-run を経て' → file: 'dry-runを経て'"""
        pat = _build_fuzzy_cjk_latin_pattern("dry-run を経て")
        assert pat is not None
        assert pat.search("dry-runを経て")
        assert pat.search("dry-run を経て")

    def test_done_boundary(self) -> None:
        """Model: 'タスクを done に更新' → file: 'タスクをdoneに更新'"""
        pat = _build_fuzzy_cjk_latin_pattern("タスクを done に更新")
        assert pat is not None
        assert pat.search("タスクをdoneに更新")
        assert pat.search("タスクを done に更新")

    def test_ai_boundary(self) -> None:
        """Model: 'AI グリッチ' → file: 'AIグリッチ'"""
        pat = _build_fuzzy_cjk_latin_pattern("AI グリッチ")
        assert pat is not None
        assert pat.search("AIグリッチ")
        assert pat.search("AI グリッチ")

    def test_no_match_for_non_boundary_space_diff(self) -> None:
        """Spaces between two Latin words should be literal, not fuzzy."""
        pat = _build_fuzzy_cjk_latin_pattern("hello world")
        assert pat is None

    def test_regex_special_chars_escaped(self) -> None:
        """Regex metacharacters in old_string must be safely escaped."""
        pat = _build_fuzzy_cjk_latin_pattern("値がtest.pyの場合")
        assert pat is not None
        assert pat.search("値がtest.pyの場合")
        assert pat.search("値が test.py の場合")

    def test_multiple_boundaries(self) -> None:
        """Multiple CJK-Latin boundaries in one string."""
        pat = _build_fuzzy_cjk_latin_pattern("runaのheartbeatで確認")
        assert pat is not None
        assert pat.search("runaのheartbeatで確認")
        assert pat.search("runa のheartbeat で確認")
        assert pat.search("runa の heartbeat で確認")

    def test_fullmatch_exact_substring(self) -> None:
        """Pattern should match the exact substring, not a superset."""
        pat = _build_fuzzy_cjk_latin_pattern("testの値")
        assert pat is not None
        m = pat.search("abcのtestの値xyz")
        assert m is not None
        assert m.group() in ("testの値", "test の値")


# ── Edit fuzzy fallback integration ──────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    return m


@pytest.fixture
def handler(anima_dir: Path, memory: MagicMock) -> MagicMock:
    from core.tooling.handler_files import FileToolsMixin

    class TestHandler(FileToolsMixin):
        def __init__(self) -> None:
            self._anima_dir = anima_dir
            self._context_window = 128_000
            self._state_file_lock = None

        def _check_file_permission(self, path_str: str, write: bool = False) -> str:
            return ""

        def _is_state_file(self, path: Path) -> bool:
            return False

    return TestHandler()


class TestEditFuzzyFallback:
    def test_exact_match_preferred(self, handler: MagicMock, anima_dir: Path) -> None:
        """When exact match works, fuzzy is not used."""
        f = anima_dir / "test.md"
        f.write_text("shinoの分析を行う", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "shinoの分析", "new_string": "shinoの調査"}
        )
        assert "Edited" in result
        assert f.read_text(encoding="utf-8") == "shinoの調査を行う"

    def test_fuzzy_match_with_model_added_space(self, handler: MagicMock, anima_dir: Path) -> None:
        """File has 'shinoの分析', model sends 'shino の分析'."""
        f = anima_dir / "test.md"
        f.write_text("shinoの分析を行う", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "shino の分析", "new_string": "shino の調査"}
        )
        assert "Edited" in result
        assert "shino の調査を行う" in f.read_text(encoding="utf-8")

    def test_fuzzy_match_dry_run(self, handler: MagicMock, anima_dir: Path) -> None:
        """File has 'dry-runを経て', model sends 'dry-run を経て'."""
        f = anima_dir / "test.md"
        f.write_text("必ずdry-runを経てから", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "dry-run を経て", "new_string": "dry-run を実行して"}
        )
        assert "Edited" in result

    def test_fuzzy_match_ai_boundary(self, handler: MagicMock, anima_dir: Path) -> None:
        """File has 'AIグリッチ', model sends 'AI グリッチ'."""
        f = anima_dir / "test.md"
        f.write_text("AIグリッチハンティング戦略", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "AI グリッチハンティング", "new_string": "AI anomaly hunting"}
        )
        assert "Edited" in result

    def test_fuzzy_ambiguous_match(self, handler: MagicMock, anima_dir: Path) -> None:
        """File has 'testの値' twice, model sends 'test の値' → AmbiguousMatch."""
        f = anima_dir / "test.md"
        f.write_text("testの値を確認\ntestの値を再確認", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "test の値", "new_string": "test result"}
        )
        data = json.loads(result)
        assert data["error_type"] == "AmbiguousMatch"
        assert data["context"]["match_count"] == 2

    def test_no_fuzzy_for_non_boundary_space(self, handler: MagicMock, anima_dir: Path) -> None:
        """Non-CJK-Latin space difference → StringNotFound, no fuzzy."""
        f = anima_dir / "test.md"
        f.write_text("hello world", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "helloworld", "new_string": "hi"}
        )
        data = json.loads(result)
        assert data["error_type"] == "StringNotFound"

    def test_fuzzy_no_match(self, handler: MagicMock, anima_dir: Path) -> None:
        """CJK-Latin boundary exists but content doesn't match → StringNotFound."""
        f = anima_dir / "test.md"
        f.write_text("全く別の内容", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "shino の分析", "new_string": "replacement"}
        )
        data = json.loads(result)
        assert data["error_type"] == "StringNotFound"

    def test_multiline_fuzzy(self, handler: MagicMock, anima_dir: Path) -> None:
        """Fuzzy match across CJK-Latin boundaries in multiline context."""
        f = anima_dir / "test.md"
        content = "## 行動規範\n- shinoの分析仕様を実装する\n- コードレビュー\n"
        f.write_text(content, encoding="utf-8")
        result = handler._handle_edit_file(
            {
                "path": str(f),
                "old_string": "shino の分析仕様を実装する",
                "new_string": "shino の分析仕様を正確に実装する",
            }
        )
        assert "Edited" in result
        assert "正確に" in f.read_text(encoding="utf-8")

    def test_exact_match_ambiguous_still_works(self, handler: MagicMock, anima_dir: Path) -> None:
        """Exact match ambiguity is still caught (existing behavior)."""
        f = anima_dir / "test.md"
        f.write_text("hello\nhello\n", encoding="utf-8")
        result = handler._handle_edit_file(
            {"path": str(f), "old_string": "hello", "new_string": "world"}
        )
        data = json.loads(result)
        assert data["error_type"] == "AmbiguousMatch"
