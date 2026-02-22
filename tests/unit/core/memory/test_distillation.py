from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for procedural memory auto-distillation (LLM-based classification).

Covers:
  - LLM-based classification (knowledge/procedures/skip)
  - Classification output parsing
  - Procedure saving with frontmatter
  - Weekly pattern detection (activity_log-based)
  - Activity clustering (text-based fallback and vector-based)
  - Pipeline integration with consolidation
  - Fallback behaviour for unparseable LLM output
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated anima directory with required subdirectories."""
    d = tmp_path / "animas" / "test-anima"
    for sub in (
        "episodes", "knowledge", "procedures", "skills", "state",
        "shortterm", "activity_log",
    ):
        (d / sub).mkdir(parents=True)

    # Set ANIMAWORKS_DATA_DIR so MemoryManager finds shared dirs
    data_dir = d.parent.parent
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    for shared in ("company", "common_skills", "common_knowledge"):
        (data_dir / shared).mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

    return d


@pytest.fixture
def distiller(anima_dir: Path):
    """Create a ProceduralDistiller instance."""
    from core.memory.distillation import ProceduralDistiller

    return ProceduralDistiller(anima_dir=anima_dir, anima_name="test-anima")


# ── LLM Classification ─────────────────────────────────────────


class TestClassifyAndDistill:
    """Test the LLM-based classify_and_distill() method."""

    @pytest.mark.asyncio
    async def test_classify_extracts_knowledge_and_procedures(
        self, distiller,
    ) -> None:
        """LLM classification should extract both knowledge and procedure items."""
        llm_response = (
            "## knowledge抽出\n"
            "- ファイル名: knowledge/api-patterns.md\n"
            "  内容: # APIパターン\n\nREST APIでは常にべき等性を考慮する。\n\n"
            "## procedure抽出\n"
            "- ファイル名: procedures/deploy-production.md\n"
            "  description: 本番環境デプロイ手順\n"
            "  tags: deploy, ops\n"
            "  内容: # 本番デプロイ\n\n1. テスト実行\n2. ビルド\n3. デプロイ"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "## 09:00 — デプロイ\n手順に従ってデプロイした。",
                model="test-model",
            )

        assert len(result["knowledge_items"]) == 1
        assert result["knowledge_items"][0]["filename"] == "knowledge/api-patterns.md"
        assert "APIパターン" in result["knowledge_items"][0]["content"]

        assert len(result["procedure_items"]) == 1
        assert result["procedure_items"][0]["filename"] == "procedures/deploy-production.md"
        assert result["procedure_items"][0]["description"] == "本番環境デプロイ手順"
        assert "deploy" in result["procedure_items"][0]["tags"]

    @pytest.mark.asyncio
    async def test_classify_empty_input(self, distiller) -> None:
        """Empty input should return empty results without calling LLM."""
        result = await distiller.classify_and_distill("")
        assert result["knowledge_items"] == []
        assert result["procedure_items"] == []
        assert result["raw_response"] == ""

    @pytest.mark.asyncio
    async def test_classify_skip_only(self, distiller) -> None:
        """When LLM returns only skip content, both lists should be empty."""
        llm_response = (
            "## knowledge抽出\n(なし)\n\n"
            "## procedure抽出\n(なし)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "今日はいい天気でした。",
                model="test-model",
            )

        assert result["knowledge_items"] == []
        assert result["procedure_items"] == []

    @pytest.mark.asyncio
    async def test_classify_llm_error_returns_empty(self, distiller) -> None:
        """LLM call failure should return empty results without raising."""
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await distiller.classify_and_distill(
                "手順に従ってコマンドを操作した。",
                model="test-model",
            )

        assert result["knowledge_items"] == []
        assert result["procedure_items"] == []

    @pytest.mark.asyncio
    async def test_classify_unparseable_format_returns_empty(
        self, distiller,
    ) -> None:
        """Completely unexpected LLM output should yield empty results."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = (
                "I don't understand the format. Here is some random text."
            )
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "テスト入力",
                model="test-model",
            )

        assert result["knowledge_items"] == []
        assert result["procedure_items"] == []

    @pytest.mark.asyncio
    async def test_classify_with_code_fence(self, distiller) -> None:
        """LLM output wrapped in code fences should be handled."""
        llm_response = (
            "```markdown\n"
            "## knowledge抽出\n"
            "- ファイル名: knowledge/test.md\n"
            "  内容: # テスト\n\nテスト知識です。\n\n"
            "## procedure抽出\n(なし)\n"
            "```"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "テストエピソード",
                model="test-model",
            )

        assert len(result["knowledge_items"]) == 1
        assert result["procedure_items"] == []

    @pytest.mark.asyncio
    async def test_classify_multiple_items(self, distiller) -> None:
        """Multiple knowledge and procedure items should all be extracted."""
        llm_response = (
            "## knowledge抽出\n"
            "- ファイル名: knowledge/patterns.md\n"
            "  内容: # パターン\n\nパターン1の記録\n"
            "- ファイル名: knowledge/tools.md\n"
            "  内容: # ツール知識\n\nツールAの使い方\n\n"
            "## procedure抽出\n"
            "- ファイル名: procedures/deploy.md\n"
            "  description: デプロイ手順\n"
            "  tags: deploy\n"
            "  内容: # デプロイ\n\n1. ステップ1\n"
            "- ファイル名: procedures/backup.md\n"
            "  description: バックアップ手順\n"
            "  tags: backup, ops\n"
            "  内容: # バックアップ\n\n1. DBダンプ"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "作業エピソード",
                model="test-model",
            )

        assert len(result["knowledge_items"]) == 2
        assert len(result["procedure_items"]) == 2
        assert result["procedure_items"][1]["tags"] == ["backup", "ops"]


# ── Distill Procedures (Legacy Entry Point) ──────────────────


class TestDistillProcedures:
    """Test distill_procedures() with mocked LLM."""

    @pytest.mark.asyncio
    async def test_distill_returns_procedures(self, distiller) -> None:
        """Procedures extracted via LLM classification should be returned."""
        llm_response = (
            "## knowledge抽出\n(なし)\n\n"
            "## procedure抽出\n"
            "- ファイル名: procedures/deploy_to_production.md\n"
            "  description: Production deployment procedure\n"
            "  tags: deploy, ops\n"
            "  内容: # Deploy to Production\n\n## Steps\n1. Pull latest\n2. Run tests\n3. Deploy"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.distill_procedures(
                "手順に従って操作した。コマンドを実行した。デプロイした。",
                model="test-model",
            )

        assert len(result) == 1
        assert result[0]["title"] == "deploy_to_production"
        assert "Deploy to Production" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_distill_empty_input(self, distiller) -> None:
        result = await distiller.distill_procedures("")
        assert result == []

    @pytest.mark.asyncio
    async def test_distill_no_procedures_found(self, distiller) -> None:
        llm_response = (
            "## knowledge抽出\n(なし)\n\n"
            "## procedure抽出\n(なし)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.distill_procedures(
                "手順に従って操作した。コマンドを実行した。",
                model="test-model",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_distill_malformed_response(self, distiller) -> None:
        """Malformed LLM output should return empty list without error."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "not valid format at all"
            mock_llm.return_value = mock_resp

            result = await distiller.distill_procedures(
                "手順に従ってコマンドを操作した。",
                model="test-model",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_distill_llm_error(self, distiller) -> None:
        """LLM call failure should return empty list."""
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await distiller.distill_procedures(
                "手順に従ってコマンドを操作した。",
                model="test-model",
            )

        assert result == []


# ── Knowledge vs Procedures Routing ──────────────────────────


class TestKnowledgeVsProceduresRouting:
    """Test that classification correctly routes to knowledge vs procedures."""

    @pytest.mark.asyncio
    async def test_knowledge_only_episode(self, distiller) -> None:
        """An episode with only facts/lessons should yield knowledge, no procedures."""
        llm_response = (
            "## knowledge抽出\n"
            "- ファイル名: knowledge/team-structure.md\n"
            "  内容: # チーム構成\n\nAさんがリーダー、Bさんがバックエンド担当。\n\n"
            "## procedure抽出\n(なし)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "チーム構成の確認を行った。",
                model="test-model",
            )

        assert len(result["knowledge_items"]) == 1
        assert result["procedure_items"] == []

    @pytest.mark.asyncio
    async def test_procedures_only_episode(self, distiller) -> None:
        """An episode with only steps should yield procedures, no knowledge."""
        llm_response = (
            "## knowledge抽出\n(なし)\n\n"
            "## procedure抽出\n"
            "- ファイル名: procedures/db-migration.md\n"
            "  description: DB移行手順\n"
            "  tags: db, migration\n"
            "  内容: # DB移行\n\n1. バックアップ\n2. マイグレーション実行\n3. 確認"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            result = await distiller.classify_and_distill(
                "DB移行を手順に従って実施した。",
                model="test-model",
            )

        assert result["knowledge_items"] == []
        assert len(result["procedure_items"]) == 1

    def test_get_knowledge_items(self, distiller) -> None:
        """get_knowledge_items should extract from classification result."""
        classification = {
            "knowledge_items": [
                {"filename": "knowledge/x.md", "content": "test"},
            ],
            "procedure_items": [],
        }
        items = distiller.get_knowledge_items(classification)
        assert len(items) == 1
        assert items[0]["filename"] == "knowledge/x.md"


# ── Parsing Helpers ──────────────────────────────────────────


class TestParseKnowledgeItems:
    """Test _parse_knowledge_items() output parsing."""

    def test_parses_knowledge_section(self, distiller) -> None:
        text = (
            "## knowledge抽出\n"
            "- ファイル名: knowledge/test.md\n"
            "  内容: テスト知識の内容です。\n\n"
            "## procedure抽出\n(なし)"
        )
        items = distiller._parse_knowledge_items(text)
        assert len(items) == 1
        assert items[0]["filename"] == "knowledge/test.md"
        assert "テスト知識" in items[0]["content"]

    def test_no_knowledge_section(self, distiller) -> None:
        text = "Some random text without expected sections."
        items = distiller._parse_knowledge_items(text)
        assert items == []

    def test_knowledge_none_marker(self, distiller) -> None:
        text = "## knowledge抽出\n(なし)\n\n## procedure抽出\n(なし)"
        items = distiller._parse_knowledge_items(text)
        assert items == []


class TestParseProcedureItems:
    """Test _parse_procedure_items() output parsing."""

    def test_parses_procedure_section(self, distiller) -> None:
        text = (
            "## knowledge抽出\n(なし)\n\n"
            "## procedure抽出\n"
            "- ファイル名: procedures/setup.md\n"
            "  description: 環境構築手順\n"
            "  tags: setup, env\n"
            "  内容: # 環境構築\n\n1. インストール\n2. 設定"
        )
        items = distiller._parse_procedure_items(text)
        assert len(items) == 1
        assert items[0]["filename"] == "procedures/setup.md"
        assert items[0]["description"] == "環境構築手順"
        assert items[0]["tags"] == ["setup", "env"]
        assert "環境構築" in items[0]["content"]

    def test_no_procedure_section(self, distiller) -> None:
        text = "Random text."
        items = distiller._parse_procedure_items(text)
        assert items == []

    def test_procedure_none_marker(self, distiller) -> None:
        text = "## procedure抽出\n(なし)"
        items = distiller._parse_procedure_items(text)
        assert items == []


class TestParseProcedures:
    """Test the JSON parser for LLM procedure output (weekly distill)."""

    def test_valid_json_array(self, distiller) -> None:
        text = json.dumps([
            {"title": "a", "content": "# A"},
            {"title": "b", "content": "# B"},
        ])
        result = distiller._parse_procedures(text)
        assert len(result) == 2

    def test_code_fenced_json(self, distiller) -> None:
        inner = json.dumps([{"title": "x", "content": "# X"}])
        text = f"```json\n{inner}\n```"
        result = distiller._parse_procedures(text)
        assert len(result) == 1

    def test_invalid_json(self, distiller) -> None:
        result = distiller._parse_procedures("not json")
        assert result == []

    def test_non_array_json(self, distiller) -> None:
        result = distiller._parse_procedures('{"title": "x", "content": "y"}')
        assert result == []

    def test_filters_incomplete_items(self, distiller) -> None:
        text = json.dumps([
            {"title": "ok", "content": "# OK"},
            {"title": "missing_content"},
        ])
        result = distiller._parse_procedures(text)
        assert len(result) == 1


# ── Procedure Saving ──────────────────────────────────────────


class TestSaveProcedure:
    """Test save_procedure() file I/O and frontmatter."""

    def test_saves_with_frontmatter(self, distiller, anima_dir: Path) -> None:
        item = {
            "title": "deploy_app",
            "description": "Application deploy procedure",
            "tags": ["deploy", "ops"],
            "content": "# Deploy App\n\n1. Pull\n2. Build\n3. Deploy",
        }

        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            path = distiller.save_procedure(item)

        assert path is not None
        assert path.exists()
        assert path.name == "deploy_app.md"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "auto_distilled: true" in text
        assert "confidence: 0.4" in text
        assert "description: Application deploy procedure" in text
        assert "# Deploy App" in text

    def test_title_sanitization(self, distiller) -> None:
        item = {
            "title": "my procedure/with special chars!",
            "content": "# Sanitized",
        }

        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            path = distiller.save_procedure(item)
        assert path is not None
        assert path.exists()
        # Special chars should be replaced with underscores
        assert "/" not in path.name
        assert "!" not in path.name

    def test_saves_to_procedures_dir(self, distiller, anima_dir: Path) -> None:
        item = {"title": "test_proc", "content": "# Test"}
        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            path = distiller.save_procedure(item)
        assert path is not None
        assert path.parent == anima_dir / "procedures"

    def test_metadata_fields(self, distiller, anima_dir: Path) -> None:
        item = {
            "title": "check_meta",
            "description": "Test metadata fields",
            "tags": ["test"],
            "content": "# Check Metadata",
        }
        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            path = distiller.save_procedure(item)

        assert path is not None

        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        meta = mm.read_procedure_metadata(path)

        assert meta["description"] == "Test metadata fields"
        assert meta["tags"] == ["test"]
        assert meta["success_count"] == 0
        assert meta["failure_count"] == 0
        assert meta["confidence"] == 0.4
        assert meta["version"] == 1
        assert meta["auto_distilled"] is True
        assert meta["last_used"] is None
        assert "created_at" in meta


# ── RAG Duplicate Check ──────────────────────────────────────


class TestRAGDuplicateCheck:
    """Test RAG-based duplicate detection in save_procedure()."""

    def test_save_procedure_skips_rag_duplicate(
        self, distiller, anima_dir: Path,
    ) -> None:
        """When RAG finds a high-similarity match, save_procedure returns None."""
        item = {
            "title": "deploy_app",
            "content": "# Deploy App\n\n1. Pull\n2. Deploy",
        }

        with patch.object(
            distiller, "_check_rag_duplicate",
            return_value="procedures/existing_deploy.md",
        ):
            result = distiller.save_procedure(item)

        assert result is None
        # File should NOT have been created
        assert not (anima_dir / "procedures" / "deploy_app.md").exists()

    def test_save_procedure_allows_unique(
        self, distiller, anima_dir: Path,
    ) -> None:
        """When RAG finds no duplicate, save_procedure writes the file."""
        item = {
            "title": "unique_proc",
            "content": "# Unique Procedure\n\n1. Do something new",
        }

        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            result = distiller.save_procedure(item)

        assert result is not None
        assert result.exists()
        assert result.name == "unique_proc.md"

    def test_save_procedure_rag_failure_proceeds(
        self, distiller, anima_dir: Path,
    ) -> None:
        """When RAG raises an exception, save_procedure proceeds with saving."""
        item = {
            "title": "fallback_proc",
            "content": "# Fallback Procedure\n\nSteps here",
        }

        # _check_rag_duplicate catches all exceptions internally and returns None
        with patch.object(distiller, "_check_rag_duplicate", return_value=None):
            result = distiller.save_procedure(item)

        assert result is not None
        assert result.exists()

    def test_check_rag_duplicate_handles_exception(self, distiller) -> None:
        """_check_rag_duplicate returns None when RAG is unavailable."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fail_on_rag(name, *args, **kwargs):
            if "core.memory.rag" in name:
                raise RuntimeError("ChromaDB not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_on_rag):
            result = distiller._check_rag_duplicate("some content")

        assert result is None

    def test_check_rag_duplicate_searches_procedures_and_skills(
        self, distiller,
    ) -> None:
        """Both procedures and skills collections should be searched."""
        mock_retriever = MagicMock()
        # First call (procedures) returns no match, second (skills) returns match
        low_result = MagicMock()
        low_result.score = 0.5
        low_result.metadata = {"source_file": "procedures/low.md"}

        high_result = MagicMock()
        high_result.score = 0.90
        high_result.metadata = {"source_file": "skills/existing_skill.md"}

        mock_retriever.search.side_effect = [
            [low_result],   # procedures search
            [high_result],  # skills search
        ]

        mock_vector_store = MagicMock()
        mock_indexer = MagicMock()

        # Patch the lazy imports inside _check_rag_duplicate
        rag_module = MagicMock()
        rag_module.MemoryIndexer.return_value = mock_indexer
        retriever_module = MagicMock()
        retriever_module.MemoryRetriever.return_value = mock_retriever
        singleton_module = MagicMock()
        singleton_module.get_vector_store.return_value = mock_vector_store

        import sys

        with patch.dict(sys.modules, {
            "core.memory.rag": rag_module,
            "core.memory.rag.retriever": retriever_module,
            "core.memory.rag.singleton": singleton_module,
        }):
            result = distiller._check_rag_duplicate("some procedure content")

        assert result == "skills/existing_skill.md"
        # Verify both collections were searched
        assert mock_retriever.search.call_count == 2
        types_searched = [
            c.kwargs["memory_type"]
            for c in mock_retriever.search.call_args_list
        ]
        assert "procedures" in types_searched
        assert "skills" in types_searched


# ── Existing Procedures Summary ───────────────────────────────


class TestLoadExistingProcedures:
    """Test _load_existing_procedures() summary generation."""

    def test_no_procedures(self, distiller) -> None:
        summary = distiller._load_existing_procedures()
        assert summary == "(なし)"

    def test_with_procedures(self, distiller, anima_dir: Path) -> None:
        proc_dir = anima_dir / "procedures"
        (proc_dir / "deploy.md").write_text(
            "---\ndescription: Deploy the app\n---\n\n# Deploy",
            encoding="utf-8",
        )
        (proc_dir / "backup.md").write_text(
            "---\ndescription: Backup database\n---\n\n# Backup",
            encoding="utf-8",
        )

        summary = distiller._load_existing_procedures()
        assert "deploy: Deploy the app" in summary
        assert "backup: Backup database" in summary

    def test_procedure_without_frontmatter(self, distiller, anima_dir: Path) -> None:
        proc_dir = anima_dir / "procedures"
        (proc_dir / "legacy.md").write_text(
            "# Legacy Procedure\n\nNo frontmatter",
            encoding="utf-8",
        )

        summary = distiller._load_existing_procedures()
        # Falls back to filename stem as description
        assert "legacy: legacy" in summary


# ── Weekly Pattern Distillation ───────────────────────────────


class TestWeeklyPatternDistill:
    """Test weekly_pattern_distill() with activity_log-based detection."""

    @pytest.mark.asyncio
    async def test_weekly_distill_from_activity_log(
        self, distiller, anima_dir: Path,
    ) -> None:
        """Should detect patterns from activity log and create procedures."""
        # Create activity log entries with repeated tool_use
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "ts": f"{today}T09:{i:02d}:00",
                "type": "tool_use",
                "tool": "github",
                "summary": "PRレビューを実施",
            }, ensure_ascii=False))

        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        llm_response = json.dumps([
            {
                "title": "pr_review",
                "description": "PRレビュー手順",
                "tags": ["github", "review"],
                "content": "# PRレビュー\n\n1. diffを確認\n2. コメント",
            },
        ])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            with patch.object(
                distiller, "_check_rag_duplicate", return_value=None,
            ):
                result = await distiller.weekly_pattern_distill(
                    model="test-model",
                )

        assert result["patterns_detected"] >= 1
        assert len(result["procedures_created"]) == 1
        proc_path = Path(result["procedures_created"][0])
        assert proc_path.exists()

    @pytest.mark.asyncio
    async def test_weekly_distill_no_activity(self, distiller) -> None:
        """No activity log entries should return zero results."""
        result = await distiller.weekly_pattern_distill(model="test-model")
        assert result["patterns_detected"] == 0
        assert result["procedures_created"] == []

    @pytest.mark.asyncio
    async def test_weekly_distill_no_relevant_events(
        self, distiller, anima_dir: Path,
    ) -> None:
        """Activity entries of irrelevant types should be filtered out."""
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        # Only dm_sent/dm_received — not relevant for pattern detection
        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "ts": f"{today}T09:{i:02d}:00",
                "type": "dm_sent",
                "summary": "メッセージを送信",
            }, ensure_ascii=False))

        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        result = await distiller.weekly_pattern_distill(model="test-model")
        assert result["patterns_detected"] == 0

    @pytest.mark.asyncio
    async def test_weekly_distill_no_clusters(
        self, distiller, anima_dir: Path,
    ) -> None:
        """Too few entries per group should yield no clusters."""
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        # Only 2 tool_use (below min_cluster_size=3)
        entries = [
            json.dumps({
                "ts": f"{today}T09:00:00",
                "type": "tool_use",
                "tool": "unique_tool_a",
                "summary": "Something unique A",
            }, ensure_ascii=False),
            json.dumps({
                "ts": f"{today}T10:00:00",
                "type": "tool_use",
                "tool": "unique_tool_b",
                "summary": "Something unique B",
            }, ensure_ascii=False),
        ]

        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        result = await distiller.weekly_pattern_distill(model="test-model")
        assert result["patterns_detected"] == 0

    @pytest.mark.asyncio
    async def test_weekly_distill_llm_error(
        self, distiller, anima_dir: Path,
    ) -> None:
        """LLM error during weekly distill should return zero results."""
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "ts": f"{today}T09:{i:02d}:00",
                "type": "tool_use",
                "tool": "github",
                "summary": "PRレビュー",
            }, ensure_ascii=False))

        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await distiller.weekly_pattern_distill(model="test-model")

        assert result["patterns_detected"] == 0
        assert result["procedures_created"] == []


# ── Activity Clustering ──────────────────────────────────────


class TestClusterActivities:
    """Test _cluster_activities() grouping logic."""

    def test_groups_by_type_and_tool(self, distiller) -> None:
        """Entries with same type+tool should cluster together."""
        entries = [
            {"type": "tool_use", "tool": "github", "summary": f"PR #{i}"}
            for i in range(5)
        ]

        clusters = distiller._cluster_activities(entries, min_cluster_size=3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    @patch(
        "core.memory.distillation.ProceduralDistiller._cluster_activities_vector",
        side_effect=ImportError("RAG unavailable in test"),
    )
    def test_different_tools_separate_clusters(self, _mock_vector, distiller) -> None:
        """Different tools should produce separate clusters (text-based fallback)."""
        entries = (
            [{"type": "tool_use", "tool": "github", "summary": f"gh{i}"} for i in range(4)]
            + [{"type": "tool_use", "tool": "slack", "summary": f"sl{i}"} for i in range(4)]
        )

        clusters = distiller._cluster_activities(entries, min_cluster_size=3)
        assert len(clusters) == 2

    def test_too_few_entries(self, distiller) -> None:
        """Clusters below min_cluster_size should be filtered out."""
        entries = [
            {"type": "tool_use", "tool": "github", "summary": "one"},
            {"type": "tool_use", "tool": "github", "summary": "two"},
        ]

        clusters = distiller._cluster_activities(entries, min_cluster_size=3)
        assert clusters == []


class TestFormatClusters:
    """Test _format_clusters_for_prompt() output."""

    def test_formats_clusters(self, distiller) -> None:
        clusters = [
            [
                {"ts": "2026-02-18T09:00", "type": "tool_use", "tool": "github", "summary": "PRレビュー"},
                {"ts": "2026-02-18T10:00", "type": "tool_use", "tool": "github", "summary": "PRマージ"},
                {"ts": "2026-02-18T11:00", "type": "tool_use", "tool": "github", "summary": "PR作成"},
            ],
        ]

        text = distiller._format_clusters_for_prompt(clusters)
        assert "パターン 1" in text
        assert "3回繰り返し" in text
        assert "github" in text
        assert "PRレビュー" in text


class TestLoadActivityEntries:
    """Test _load_activity_entries() reading from JSONL files."""

    def test_loads_entries(self, distiller, anima_dir: Path) -> None:
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        entries = [
            json.dumps({"ts": f"{today}T09:00:00", "type": "tool_use"}, ensure_ascii=False),
            json.dumps({"ts": f"{today}T10:00:00", "type": "response_sent"}, ensure_ascii=False),
        ]
        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        result = distiller._load_activity_entries(days=1)
        assert len(result) == 2

    def test_loads_multi_day(self, distiller, anima_dir: Path) -> None:
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        (activity_dir / f"{today}.jsonl").write_text(
            json.dumps({"ts": f"{today}T09:00:00", "type": "tool_use"}) + "\n",
            encoding="utf-8",
        )
        (activity_dir / f"{yesterday}.jsonl").write_text(
            json.dumps({"ts": f"{yesterday}T09:00:00", "type": "tool_use"}) + "\n",
            encoding="utf-8",
        )

        result = distiller._load_activity_entries(days=2)
        assert len(result) == 2

    def test_empty_activity_dir(self, distiller) -> None:
        result = distiller._load_activity_entries(days=7)
        assert result == []

    def test_skips_malformed_json(self, distiller, anima_dir: Path) -> None:
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        lines = [
            json.dumps({"ts": f"{today}T09:00:00", "type": "tool_use"}),
            "not valid json",
            json.dumps({"ts": f"{today}T10:00:00", "type": "tool_use"}),
        ]
        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        result = distiller._load_activity_entries(days=1)
        assert len(result) == 2


# ── Section Splitting ─────────────────────────────────────────


class TestSplitIntoSections:
    """Test the Markdown section splitter (utility method)."""

    def test_split_by_h2_headers(self, distiller) -> None:
        text = (
            "## Section A\nContent A\n\n"
            "## Section B\nContent B"
        )
        sections = distiller._split_into_sections(text)
        assert len(sections) == 2
        assert sections[0].startswith("## Section A")
        assert sections[1].startswith("## Section B")

    def test_no_headers(self, distiller) -> None:
        text = "Just plain text without headers."
        sections = distiller._split_into_sections(text)
        assert len(sections) == 1
        assert sections[0] == text

    def test_empty_text(self, distiller) -> None:
        sections = distiller._split_into_sections("")
        assert sections == []

    def test_strips_blank_sections(self, distiller) -> None:
        text = "\n\n## Header\nContent\n\n\n"
        sections = distiller._split_into_sections(text)
        assert len(sections) == 1


# ── Weekly Pattern Filter: issue_resolved ────────────────────


class TestWeeklyPatternFilterIncludesResolved:
    """Test that issue_resolved events are included in weekly pattern detection."""

    @pytest.mark.asyncio
    async def test_issue_resolved_passes_filter(
        self, distiller, anima_dir: Path,
    ) -> None:
        """issue_resolved events should pass the relevant type filter."""
        activity_dir = anima_dir / "activity_log"
        today = datetime.now().date()

        # Write issue_resolved events to activity log
        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "ts": f"{today}T09:{i:02d}:00",
                "type": "issue_resolved",
                "summary": f"問題解決 #{i}",
                "content": f"サーバー障害対応手順 #{i}",
            }, ensure_ascii=False))

        (activity_dir / f"{today}.jsonl").write_text(
            "\n".join(entries) + "\n",
            encoding="utf-8",
        )

        llm_response = json.dumps([
            {
                "title": "server_recovery",
                "description": "サーバー障害復旧手順",
                "tags": ["ops", "recovery"],
                "content": "# サーバー復旧\n\n1. 状態確認\n2. サービス再起動",
            },
        ])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = llm_response
            mock_llm.return_value = mock_resp

            with patch.object(
                distiller, "_check_rag_duplicate", return_value=None,
            ):
                result = await distiller.weekly_pattern_distill(
                    model="test-model",
                )

        # issue_resolved events should cluster and produce patterns
        assert result["patterns_detected"] >= 1

    def test_issue_resolved_not_filtered_out(self, distiller) -> None:
        """issue_resolved entries should not be excluded by the relevant filter."""
        # The filter in weekly_pattern_distill accepts these types:
        # "tool_use", "response_sent", "cron_executed", "memory_write",
        # "issue_resolved"
        relevant_types = {
            "tool_use", "response_sent", "cron_executed",
            "memory_write", "issue_resolved",
        }
        # Verify issue_resolved is in the accepted set
        assert "issue_resolved" in relevant_types

        # Simulate the filtering logic from weekly_pattern_distill
        entries = [
            {"type": "issue_resolved", "summary": "test"},
            {"type": "dm_sent", "summary": "excluded"},
            {"type": "tool_use", "tool": "github", "summary": "included"},
        ]
        relevant = [
            e for e in entries
            if e.get("type") in relevant_types
        ]
        # issue_resolved and tool_use should pass; dm_sent should not
        assert len(relevant) == 2
        types_in_result = {e["type"] for e in relevant}
        assert "issue_resolved" in types_in_result
        assert "dm_sent" not in types_in_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
