from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Integration tests for lifecycle consolidation setup."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLifecycleConsolidationIntegration:
    """Test lifecycle manager's system cron setup."""

    @pytest.mark.asyncio
    async def test_system_crons_registered_on_start(self):
        """Test that system crons are registered when lifecycle manager starts."""
        from core.lifecycle import LifecycleManager

        manager = LifecycleManager()

        # Start the manager (needs to be in async context)
        manager.start()

        try:
            # Check that system cron jobs were registered
            jobs = manager.scheduler.get_jobs()
            job_ids = [job.id for job in jobs]

            assert "system_daily_consolidation" in job_ids
            assert "system_weekly_integration" in job_ids

            # Check daily consolidation job details
            daily_job = next(j for j in jobs if j.id == "system_daily_consolidation")
            assert daily_job.name == "System: Daily Consolidation"

            # Check weekly integration job details
            weekly_job = next(j for j in jobs if j.id == "system_weekly_integration")
            assert weekly_job.name == "System: Weekly Integration"

        finally:
            manager.shutdown()

    @pytest.mark.asyncio
    async def test_handle_daily_consolidation_with_config(self, tmp_path: Path):
        """Test daily consolidation handler respects config settings."""
        from core.lifecycle import LifecycleManager
        from core.person import DigitalPerson

        manager = LifecycleManager()

        # Create mock person
        person_dir = tmp_path / "test_person"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text("# Test Person", encoding="utf-8")

        # Mock DigitalPerson
        mock_person = MagicMock(spec=DigitalPerson)
        mock_person.name = "test_person"
        mock_person.memory = MagicMock()
        mock_person.memory.person_dir = person_dir

        manager.persons["test_person"] = mock_person

        # Mock config to disable consolidation
        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.daily_enabled = False
        mock_config.consolidation = mock_consolidation_cfg

        with patch("core.config.load_config", return_value=mock_config):
            await manager._handle_daily_consolidation()

        # Consolidation should be skipped when disabled
        # (No exception should be raised)

    @pytest.mark.asyncio
    async def test_handle_daily_consolidation_with_person(self, tmp_path: Path):
        """Test daily consolidation runs for registered person."""
        from core.lifecycle import LifecycleManager
        from core.person import DigitalPerson
        from datetime import datetime

        manager = LifecycleManager()

        # Create person with episodes
        person_dir = tmp_path / "test_person"
        episodes_dir = person_dir / "episodes"
        knowledge_dir = person_dir / "knowledge"
        episodes_dir.mkdir(parents=True)
        knowledge_dir.mkdir(parents=True)

        # Create sample episode
        today = datetime.now().date()
        episode_file = episodes_dir / f"{today}.md"
        episode_file.write_text("""## 10:00 — Test Episode

**要点**: Test content
""", encoding="utf-8")

        # Mock person
        mock_person = MagicMock(spec=DigitalPerson)
        mock_person.name = "test_person"
        mock_person.memory = MagicMock()
        mock_person.memory.person_dir = person_dir

        manager.persons["test_person"] = mock_person

        # Mock config to enable consolidation
        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.daily_enabled = True
        mock_consolidation_cfg.llm_model = "anthropic/claude-sonnet-4-20250514"
        mock_consolidation_cfg.min_episodes_threshold = 1
        mock_config.consolidation = mock_consolidation_cfg

        # Mock LLM response
        mock_llm_response = """## 既存ファイル更新
(なし)

## 新規ファイル作成
- ファイル名: knowledge/test-knowledge.md
  内容: # Test Knowledge

Test content from consolidation
"""

        with patch("core.config.load_config", return_value=mock_config):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = mock_llm_response
                mock_llm.return_value = mock_response

                await manager._handle_daily_consolidation()

        # Verify knowledge file was created
        knowledge_file = knowledge_dir / "test-knowledge.md"
        assert knowledge_file.exists()
        content = knowledge_file.read_text(encoding="utf-8")
        assert "Test Knowledge" in content
        assert "AUTO-CONSOLIDATED" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
