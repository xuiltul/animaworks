# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for demo mode onboarding feature.

Tests cover:
- UIConfig.demo_mode field existence and defaults
- Demo preset config_overlay.json files contain demo_mode: true
- Demo preload data integrity (JSONL, JSON, Markdown)
- i18n key coverage for demo.* namespace
- Frontend JS/CSS structural requirements
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.models import AnimaWorksConfig, UIConfig

# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "demo"
EXAMPLES_DIR = DEMO_DIR / "examples"
PRESETS_DIR = DEMO_DIR / "presets"
STATIC_DIR = PROJECT_ROOT / "server" / "static"
I18N_DIR = STATIC_DIR / "i18n"
STYLES_DIR = STATIC_DIR / "styles"
MODULES_DIR = STATIC_DIR / "modules"
CHAT_DIR = STATIC_DIR / "pages" / "chat"


# ── 1. Config Model ────────────────────────────────────────


class TestUIConfigDemoMode:
    def test_demo_mode_field_exists(self):
        cfg = UIConfig()
        assert hasattr(cfg, "demo_mode")

    def test_demo_mode_default_false(self):
        cfg = UIConfig()
        assert cfg.demo_mode is False

    def test_demo_mode_can_be_set_true(self):
        cfg = UIConfig(demo_mode=True)
        assert cfg.demo_mode is True

    def test_demo_mode_in_full_config(self):
        cfg = AnimaWorksConfig()
        assert cfg.ui.demo_mode is False

    def test_demo_mode_json_roundtrip(self):
        cfg = UIConfig(demo_mode=True)
        data = json.loads(cfg.model_dump_json())
        assert data["demo_mode"] is True
        restored = UIConfig(**data)
        assert restored.demo_mode is True


# ── 2. Preset Config Overlays ──────────────────────────────


class TestPresetConfigOverlays:
    PRESETS = ["ja-business", "ja-anime", "en-business", "en-anime"]

    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset_has_demo_mode(self, preset):
        overlay_path = PRESETS_DIR / preset / "config_overlay.json"
        assert overlay_path.exists(), f"Missing {overlay_path}"
        data = json.loads(overlay_path.read_text(encoding="utf-8"))
        assert data.get("ui", {}).get("demo_mode") is True, (
            f"{preset}/config_overlay.json must have ui.demo_mode: true"
        )


# ── 3. Demo Preload Data Integrity ────────────────────────


class TestDemoDataIntegrity:
    JA_ANIMAS = ["kaito", "sora", "hina"]
    EN_ANIMAS = ["alex", "kai", "nova"]

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_activity_log_three_days(self, lang, animas):
        for anima in animas:
            log_dir = EXAMPLES_DIR / lang / anima / "activity_log"
            assert log_dir.exists(), f"Missing {log_dir}"
            jsonl_files = sorted(log_dir.glob("*.jsonl"))
            assert len(jsonl_files) >= 3, (
                f"{lang}/{anima} should have 3+ days of activity_log, got {len(jsonl_files)}"
            )

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_activity_log_rich_entries(self, lang, animas):
        for anima in animas:
            log_dir = EXAMPLES_DIR / lang / anima / "activity_log"
            for f in log_dir.glob("*.jsonl"):
                lines = [line for line in f.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
                assert len(lines) >= 15, (
                    f"{f.relative_to(PROJECT_ROOT)} should have 15+ entries, got {len(lines)}"
                )
                for i, line in enumerate(lines, 1):
                    data = json.loads(line)
                    assert "ts" in data, f"Line {i} in {f.name} missing 'ts'"
                    assert "type" in data, f"Line {i} in {f.name} missing 'type'"

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_knowledge_files_exist(self, lang, animas):
        expected_counts = {"kaito": 2, "sora": 5, "hina": 3, "alex": 2, "kai": 5, "nova": 3}
        for anima in animas:
            k_dir = EXAMPLES_DIR / lang / anima / "knowledge"
            assert k_dir.exists(), f"Missing {k_dir}"
            md_files = list(k_dir.glob("*.md"))
            expected = expected_counts[anima]
            assert len(md_files) >= expected, (
                f"{lang}/{anima}/knowledge should have {expected}+ files, got {len(md_files)}"
            )

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_knowledge_files_have_frontmatter(self, lang, animas):
        for anima in animas:
            k_dir = EXAMPLES_DIR / lang / anima / "knowledge"
            if not k_dir.exists():
                continue
            for f in k_dir.glob("*.md"):
                text = f.read_text(encoding="utf-8")
                assert text.startswith("---"), (
                    f"{f.relative_to(PROJECT_ROOT)} must start with YAML frontmatter"
                )
                assert "confidence:" in text, (
                    f"{f.relative_to(PROJECT_ROOT)} missing 'confidence' in frontmatter"
                )

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_episodes_exist(self, lang, animas):
        for anima in animas:
            e_dir = EXAMPLES_DIR / lang / anima / "episodes"
            assert e_dir.exists(), f"Missing {e_dir}"
            md_files = list(e_dir.glob("*.md"))
            assert len(md_files) >= 2, (
                f"{lang}/{anima}/episodes should have 2+ files, got {len(md_files)}"
            )

    @pytest.mark.parametrize("lang,animas", [("ja", JA_ANIMAS), ("en", EN_ANIMAS)])
    def test_conversation_json_valid(self, lang, animas):
        for anima in animas:
            conv_path = EXAMPLES_DIR / lang / anima / "state" / "conversation.json"
            assert conv_path.exists(), f"Missing {conv_path}"
            data = json.loads(conv_path.read_text(encoding="utf-8"))
            assert "turns" in data
            assert "anima_name" in data
            assert len(data["turns"]) >= 2

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_channels_general_rich(self, lang):
        general = EXAMPLES_DIR / lang / "channels" / "general.jsonl"
        assert general.exists()
        lines = [line for line in general.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        assert len(lines) >= 15, f"{lang}/channels/general.jsonl needs 15+ entries"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_channels_ops_exists(self, lang):
        ops = EXAMPLES_DIR / lang / "channels" / "ops.jsonl"
        assert ops.exists()
        lines = [line for line in ops.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        assert len(lines) >= 8, f"{lang}/channels/ops.jsonl needs 8+ entries"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_user_profile_exists(self, lang):
        profile = EXAMPLES_DIR / lang / "users" / "ceo" / "index.md"
        assert profile.exists()
        text = profile.read_text(encoding="utf-8")
        assert len(text) > 50


# ── 4. i18n Keys ──────────────────────────────────────────


class TestDemoI18nKeys:
    REQUIRED_KEYS = [
        "demo.splash_subtitle",
        "demo.splash_role_engineer",
        "demo.splash_role_assistant",
        "demo.splash_role_pm",
        "demo.splash_engineer_desc",
        "demo.splash_assistant_desc",
        "demo.splash_pm_desc",
        "demo.splash_cta_chat",
        "demo.splash_cta_activity",
        "demo.suggest_card_title",
        "demo.suggest_card_subtitle",
    ]

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_common_demo_keys_exist(self, lang):
        i18n_path = I18N_DIR / f"{lang}.json"
        data = json.loads(i18n_path.read_text(encoding="utf-8"))
        for key in self.REQUIRED_KEYS:
            assert key in data, f"Missing i18n key '{key}' in {lang}.json"
            assert data[key].strip(), f"Empty i18n value for '{key}' in {lang}.json"

    def test_ja_prompt_keys(self):
        data = json.loads((I18N_DIR / "ja.json").read_text(encoding="utf-8"))
        for anima in ["kaito", "sora", "hina"]:
            for i in range(1, 5):
                key = f"demo.prompts.{anima}.{i}"
                assert key in data, f"Missing '{key}' in ja.json"

    def test_en_prompt_keys(self):
        data = json.loads((I18N_DIR / "en.json").read_text(encoding="utf-8"))
        for anima in ["alex", "kai", "nova"]:
            for i in range(1, 5):
                key = f"demo.prompts.{anima}.{i}"
                assert key in data, f"Missing '{key}' in en.json"


# ── 5. Frontend Structure ────────────────────────────────


class TestDemoFrontendStructure:
    def test_state_has_demo_mode(self):
        text = (MODULES_DIR / "state.js").read_text(encoding="utf-8")
        assert "demoMode" in text, "state.js must contain demoMode"

    def test_app_has_init_demo_mode(self):
        text = (MODULES_DIR / "app.js").read_text(encoding="utf-8")
        assert "initDemoMode" in text
        assert "showDemoSplashIfNeeded" in text
        assert "aw-demo-splash-seen" in text

    def test_chat_renderer_has_suggest_cards(self):
        text = (CHAT_DIR / "chat-renderer.js").read_text(encoding="utf-8")
        assert "renderDemoSuggestedCards" in text
        assert "state.demoMode" in text
        assert "demo-suggest-card" in text

    def test_pane_host_has_chips_container(self):
        text = (CHAT_DIR / "pane-host.js").read_text(encoding="utf-8")
        assert "chatPromptChips" in text
        assert "chat-prompt-chips" in text

    def test_events_controller_has_demo_handlers(self):
        text = (CHAT_DIR / "events-controller.js").read_text(encoding="utf-8")
        assert "demo-suggest-card" in text
        assert "demo-prompt" in text or "demoPrompt" in text

    def test_base_css_has_splash_styles(self):
        text = (STYLES_DIR / "base.css").read_text(encoding="utf-8")
        assert ".demo-splash" in text
        assert ".demo-splash-btn" in text
        assert ".demo-splash-card" in text

    def test_chat_css_has_suggest_styles(self):
        text = (STYLES_DIR / "chat.css").read_text(encoding="utf-8")
        assert ".demo-suggest-card" in text
        assert ".demo-suggest-container" in text
        assert ".chat-prompt-chip" in text
        assert ".chat-prompt-chips" in text


# ── 6. adjust_dates.sh Coverage ──────────────────────────


class TestAdjustDatesScript:
    def test_handles_all_date_patterns(self):
        text = (DEMO_DIR / "adjust_dates.sh").read_text(encoding="utf-8")
        assert "2026-03-01" in text
        assert "2026-03-02" in text
        assert "2026-03-03" in text

    def test_handles_knowledge_episodes(self):
        text = (DEMO_DIR / "adjust_dates.sh").read_text(encoding="utf-8")
        assert "knowledge" in text
        assert "episodes" in text

    def test_handles_conversation_task_queue(self):
        text = (DEMO_DIR / "adjust_dates.sh").read_text(encoding="utf-8")
        assert "conversation.json" in text
        assert "task_queue.jsonl" in text
