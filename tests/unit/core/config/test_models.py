"""Unit tests for core/config/models.py — Pydantic config models & load/save."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.model_mode import CANONICAL_MODES
from core.config.models import (
    DEFAULT_LOCAL_LLM_MODEL,
    AnimaDefaults,
    AnimaModelConfig,
    AnimaWorksConfig,
    CommandsPermission,
    CredentialConfig,
    ExternalToolsPermission,
    ImageGenConfig,
    LocalLLMConfig,
    PermissionsConfig,
    PromptConfig,
    RAGConfig,
    SkillPromotionConfig,
    SkillsConfig,
    SystemConfig,
    ToolCreationPermission,
    _format_permissions_for_prompt,
    _match_pattern_table,
    _normalise_mode,
    _pattern_specificity,
    get_config_path,
    invalidate_cache,
    load_config,
    load_model_config,
    load_permissions,
    parse_fallback_entry,
    read_anima_supervisor,
    register_anima_in_config,
    resolve_anima_config,
    resolve_execution_mode,
    save_config,
)

# ── Pydantic model defaults ──────────────────────────────


class TestSystemConfig:
    def test_defaults(self):
        sc = SystemConfig()
        assert sc.mode == "server"
        assert not hasattr(sc, "log_level")
        assert not hasattr(sc, "gateway")
        assert not hasattr(sc, "worker")

    def test_prompt_identity_business_exclusions_default(self):
        assert PromptConfig().identity_business_exclude_headings == [
            "外見",
            "基本プロフィール",
            "Appearance",
            "Basic Profile",
        ]


class TestCredentialConfig:
    def test_defaults(self):
        cc = CredentialConfig()
        assert cc.api_key == ""
        assert cc.base_url is None

    def test_custom(self):
        cc = CredentialConfig(api_key="sk-123", base_url="http://localhost")
        assert cc.api_key == "sk-123"
        assert cc.base_url == "http://localhost"


class TestAnimaModelConfig:
    def test_all_none_by_default(self):
        pmc = AnimaModelConfig()
        assert pmc.supervisor is None
        assert pmc.speciality is None


class TestAnimaDefaults:
    def test_defaults(self):
        pd = AnimaDefaults()
        assert pd.model == "claude-sonnet-5-5"
        assert pd.max_tokens == 8192
        assert pd.credential == "anthropic"
        assert pd.context_threshold == 0.50
        assert pd.conversation_history_threshold == 0.30


class TestAnimaWorksConfig:
    def test_defaults(self):
        config = AnimaWorksConfig()
        assert config.version == 1
        assert isinstance(config.system, SystemConfig)
        assert "anthropic" in config.credentials
        assert config.animas == {}
        assert config.local_llm.default_model == DEFAULT_LOCAL_LLM_MODEL
        assert config.prompt.skill_catalog_router_enabled is True
        assert isinstance(config.skills, SkillsConfig)
        assert isinstance(config.skills.promotion, SkillPromotionConfig)
        assert config.skills.promotion.success_count_threshold == 3
        assert config.skills.promotion.confidence_threshold == 0.8

    def test_roundtrip_json(self):
        config = AnimaWorksConfig()
        config.animas["alice"] = AnimaModelConfig(supervisor="bob", speciality="engineer")
        data = config.model_dump(mode="json")
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.animas["alice"].supervisor == "bob"
        assert restored.animas["alice"].speciality == "engineer"

    def test_legacy_throttle_keys_are_ignored_when_loading_existing_config(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "heartbeat": {
                        "msg_heartbeat_cooldown_s": 300,
                        "cascade_window_s": 1800,
                        "cascade_threshold": 3,
                        "depth_window_s": 600,
                        "max_depth": 6,
                        "actionable_intents": ["report", "question"],
                        "channel_post_cooldown_s": 300,
                        "outbound_limit_enabled": False,
                    },
                    "anima_defaults": {
                        "max_outbound_per_hour": 60,
                        "max_outbound_per_day": 300,
                        "max_recipients_per_run": 10,
                    },
                }
            ),
            encoding="utf-8",
        )
        invalidate_cache()
        try:
            config = load_config(config_path)
        finally:
            invalidate_cache()

        serialized = config.model_dump(mode="json")
        assert config.heartbeat.interval_minutes == 30
        assert (
            not {
                "msg_heartbeat_cooldown_s",
                "cascade_window_s",
                "cascade_threshold",
                "depth_window_s",
                "max_depth",
                "actionable_intents",
                "channel_post_cooldown_s",
                "outbound_limit_enabled",
            }
            & serialized["heartbeat"].keys()
        )
        assert (
            not {
                "max_outbound_per_hour",
                "max_outbound_per_day",
                "max_recipients_per_run",
            }
            & serialized["anima_defaults"].keys()
        )


class TestLocalLLMConfig:
    def test_defaults(self):
        config = LocalLLMConfig()
        assert config.base_url == "http://127.0.0.1:11434"
        assert config.default_model == "ollama/qwen2.5-coder:14b"
        assert config.presets["coding"] == "ollama/qwen2.5-coder:14b"
        assert config.presets["reasoning"] == "ollama/deepseek-r1:8b"
        assert config.role_presets["engineer"] == "coding"
        assert config.role_presets["manager"] == "reasoning"

    def test_missing_presets_are_backfilled(self):
        config = LocalLLMConfig(
            presets={"coding": "ollama/gemma3:12b"},
            role_presets={"engineer": "coding", "writer": "general"},
        )
        assert config.presets["coding"] == "ollama/gemma3:12b"
        assert config.presets["general"] == "ollama/glm4:9b"
        assert config.role_presets["engineer"] == "coding"
        assert config.role_presets["researcher"] == "reasoning"


class TestImageGenConfig:
    def test_defaults(self):
        ic = ImageGenConfig()
        assert ic.style_reference is None
        assert ic.style_prefix == ""
        assert ic.style_suffix == ""
        assert ic.negative_prompt_extra == ""
        assert ic.vibe_strength == 0.6
        assert ic.vibe_info_extracted == 0.8

    def test_custom_values(self):
        ic = ImageGenConfig(
            style_reference="/path/to/ref.png",
            style_prefix="anime coloring, ",
            style_suffix=", high quality",
            negative_prompt_extra="realistic",
            vibe_strength=0.5,
            vibe_info_extracted=0.9,
        )
        assert ic.style_reference == "/path/to/ref.png"
        assert ic.style_prefix == "anime coloring, "
        assert ic.style_suffix == ", high quality"
        assert ic.negative_prompt_extra == "realistic"
        assert ic.vibe_strength == 0.5
        assert ic.vibe_info_extracted == 0.9

    def test_animaworks_config_has_image_gen(self):
        config = AnimaWorksConfig()
        assert isinstance(config.image_gen, ImageGenConfig)
        assert config.image_gen.style_reference is None


class TestFormatPermissionsForPrompt:
    def test_file_roots_denied_defaults_empty(self):
        assert PermissionsConfig().file_roots_denied == []

    def test_file_roots_denied_requires_absolute_paths(self):
        with pytest.raises(ValueError, match="absolute paths"):
            PermissionsConfig(file_roots_denied=["relative/private"])

    @pytest.mark.parametrize("root", ["/home/user/*/private", "/home/user/private?.txt"])
    def test_file_roots_denied_rejects_globs(self, root: str):
        with pytest.raises(ValueError, match="glob patterns"):
            PermissionsConfig(file_roots_denied=[root])

    def test_file_roots_denied_is_normalized(self, tmp_path: Path):
        root = tmp_path / "missing" / ".." / "private"
        config = PermissionsConfig(file_roots_denied=[str(root)])
        assert config.file_roots_denied == [str(root.resolve())]

    def test_invalid_explicit_deny_config_fails_closed(self, tmp_path: Path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "permissions.json").write_text(
            json.dumps({"file_roots_denied": ["relative/private"]}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="absolute paths"):
            load_permissions(anima_dir)

    def test_invalid_json_permissions_fails_closed(self, tmp_path: Path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "permissions.json").write_text('{"file_roots": [', encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_permissions(anima_dir)

    def test_invalid_schema_without_explicit_deny_fails_closed(self, tmp_path: Path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "permissions.json").write_text(
            json.dumps({"commands": "not-an-object"}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            load_permissions(anima_dir)

    def test_unreadable_existing_permissions_fails_closed(self, tmp_path: Path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        permissions_path = anima_dir / "permissions.json"
        permissions_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(Path, "read_text", side_effect=OSError("permission denied")),
            pytest.raises(OSError, match="permission denied"),
        ):
            load_permissions(anima_dir)

    def test_missing_permissions_keeps_open_default(self, tmp_path: Path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        config = load_permissions(anima_dir)

        assert config == PermissionsConfig()

    def test_prompt_omits_default_permissions(self):
        assert _format_permissions_for_prompt(PermissionsConfig(), "sora") == ""

    def test_prompt_displays_only_non_default_permission_constraints(self):
        prompt = _format_permissions_for_prompt(
            PermissionsConfig(
                commands=CommandsPermission(allow_all=False, allow=["git status"], deny=["rm -rf"]),
                tool_creation=ToolCreationPermission(personal=False, shared=True),
            ),
            "sora",
        )
        assert "Allowed commands: git status" in prompt
        assert "Additionally denied commands: rm -rf" in prompt
        assert "Personal tool creation: not allowed" in prompt
        assert "Shared tool creation: allowed" in prompt
        assert "External tools: all allowed" not in prompt

    def test_prompt_displays_denied_roots(self, tmp_path: Path):
        denied = tmp_path / "private"
        prompt = _format_permissions_for_prompt(
            PermissionsConfig(file_roots_denied=[str(denied)]),
            "sora",
        )
        assert "Denied file access" in prompt
        assert str(denied.resolve()) in prompt
        assert "overrides all grants" in prompt

    def test_windows_prompt_mentions_native_runtime(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("core.config.schemas.sys.platform", "win32")
        config = PermissionsConfig(
            file_roots=[r"E:\Projects\Tools\General"],
            file_roots_readonly=[r"E:\Projects\Tools\abconfig\Cnct_Env.py"],
            commands=CommandsPermission(allow_all=True),
            external_tools=ExternalToolsPermission(allow_all=True),
            tool_creation=ToolCreationPermission(personal=True, shared=False),
        )

        prompt = _format_permissions_for_prompt(config, "sora")

        assert "native Windows environment" in prompt
        assert "read_file for absolute paths" in prompt
        assert "PowerShell-compatible shell via execute_command" in prompt


class TestRAGConfig:
    """RAGConfig defaults and overrides."""

    def test_min_retrieval_score_default(self) -> None:
        """RAGConfig.min_retrieval_score defaults to 0.3."""
        rag = RAGConfig()
        assert rag.min_retrieval_score == 0.3

    def test_min_retrieval_score_override(self) -> None:
        """RAGConfig.min_retrieval_score can be overridden."""
        rag = RAGConfig(min_retrieval_score=0.5)
        assert rag.min_retrieval_score == 0.5

    def test_entity_registry_remains_enabled_for_fact_features(self) -> None:
        assert RAGConfig().entity_registry_enabled is True

    def test_removed_graph_retrieval_settings_are_ignored(self, tmp_path: Path) -> None:
        retired_settings = {
            "enable_spreading_activation": True,
            "graph_cache_enabled": True,
            "implicit_link_threshold": 0.75,
            "spreading_memory_types": ["knowledge", "episodes"],
            "entity_aware_graph_enabled": True,
            "graph_entity_edge_cap": 8,
            "graph_inverse_fan_enabled": True,
            "graph_recency_weight_enabled": True,
            "max_graph_hops": 2,
        }
        config_path = get_config_path(tmp_path)
        config_path.write_text(json.dumps({"rag": retired_settings}), encoding="utf-8")
        invalidate_cache()

        try:
            config = load_config(config_path)
        finally:
            invalidate_cache()

        assert not retired_settings.keys() & config.rag.model_dump().keys()

    def test_legacy_boost_config_keys_are_ignored(self) -> None:
        rag = RAGConfig.model_validate(
            {
                "entity_boost_enabled": False,
                "entity_boost": 0.9,
                "entity_boost_cap": 1.0,
                "temporal_boost_enabled": False,
                "temporal_boost": 0.9,
                "temporal_boost_max": 1.0,
                "temporal_half_life_days": 1.0,
                "access_boost_enabled": False,
                "access_boost_weight": 0.9,
                "access_boost_cap": 1.0,
                "access_boost_half_life_days": 1.0,
            }
        )

        assert "entity_boost_enabled" not in rag.model_dump()
        assert "temporal_boost_enabled" not in rag.model_dump()
        assert "access_boost_enabled" not in rag.model_dump()

    def test_fact_reconcile_defaults(self) -> None:
        """Fact reconciliation defaults to high-threshold, bounded top-k behavior."""
        rag = RAGConfig()
        assert rag.facts_reconcile_enabled is True
        assert rag.facts_reconcile_similarity_threshold == 0.82
        assert rag.facts_reconcile_top_k == 5

    def test_shared_check_timing_defaults(self) -> None:
        rag = RAGConfig()
        assert rag.shared_check_ttl_seconds == 30.0
        assert rag.shared_check_backoff_initial_seconds == 5.0
        assert rag.shared_check_backoff_max_seconds == 300.0


# ── Cache management ──────────────────────────────────────


class TestInvalidateCache:
    def test_invalidate_clears(self):
        from core.config import io as config_io

        config_io._config = AnimaWorksConfig()
        config_io._config_path = Path("/fake")
        config_io._config_mtime = 99.0
        invalidate_cache()
        assert config_io._config is None
        assert config_io._config_path is None
        assert config_io._config_mtime == 0.0


# ── mtime-based cache reload ─────────────────────────────


class TestLoadConfigMtimeReload:
    """Tests for mtime-based automatic cache invalidation."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    @staticmethod
    def _write_config(path: Path, **overrides) -> None:
        data = {"version": 1, "credentials": {"anthropic": {"api_key": ""}}}
        data.update(overrides)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_cache_hit_when_mtime_unchanged(self, tmp_path):
        """Same mtime → returns the cached object."""
        path = tmp_path / "config.json"
        self._write_config(path)
        c1 = load_config(path)
        c2 = load_config(path)
        assert c1 is c2

    def test_reload_when_mtime_changed(self, tmp_path):
        """Changed mtime → reloads from disk, returns a new object."""
        path = tmp_path / "config.json"
        self._write_config(path)
        c1 = load_config(path)

        # Bump mtime by rewriting with different content
        import time

        time.sleep(0.05)
        self._write_config(path, animas={"bob": {"supervisor": "alice"}})
        os.utime(path)

        c2 = load_config(path)
        assert c1 is not c2
        assert "bob" in c2.animas

    def test_stat_oserror_returns_cached(self, tmp_path):
        """If stat() fails after initial load, return cached config."""
        path = tmp_path / "config.json"
        self._write_config(path)
        c1 = load_config(path)

        # Remove file — stat() will fail
        path.unlink()
        from core.config import io as config_io

        # Set mtime to 0.0 to simulate OSError path matching
        config_io._config_mtime = 0.0
        c2 = load_config(path)
        # disk_mtime=0.0 == _config_mtime=0.0 → cache hit
        assert c1 is c2

    def test_save_config_records_mtime(self, tmp_path):
        """save_config() stores the file's mtime in _config_mtime."""
        from core.config import io as config_io

        path = tmp_path / "config.json"
        save_config(AnimaWorksConfig(), path)
        assert config_io._config_mtime > 0.0
        assert config_io._config_mtime == path.stat().st_mtime


# ── get_config_path ───────────────────────────────────────


class TestGetConfigPath:
    def test_with_data_dir(self, tmp_path):
        result = get_config_path(tmp_path)
        assert result == tmp_path / "config.json"

    def test_without_data_dir(self, data_dir):
        result = get_config_path()
        assert result == data_dir / "config.json"


# ── load_config ───────────────────────────────────────────


class TestLoadConfig:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_load_existing(self, data_dir):
        config = load_config(data_dir / "config.json")
        assert isinstance(config, AnimaWorksConfig)
        assert config.version == 1

    def test_load_missing_returns_default(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.json")
        assert isinstance(config, AnimaWorksConfig)

    def test_caching(self, data_dir):
        path = data_dir / "config.json"
        c1 = load_config(path)
        c2 = load_config(path)
        assert c1 is c2  # same object due to cache

    def test_invalid_json_raises(self, tmp_path):
        from core.exceptions import ConfigError

        bad = tmp_path / "bad.json"
        bad.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            load_config(bad)

    def test_load_with_animas(self, data_dir):
        # Write a config with an anima (supervisor/speciality only; model in status.json)
        config_data = {
            "version": 1,
            "system": {"mode": "server"},
            "credentials": {"anthropic": {"api_key": ""}},
            "anima_defaults": {"model": "claude-sonnet-4-6", "credential": "anthropic"},
            "animas": {"alice": {"supervisor": "bob", "speciality": "engineer"}},
        }
        (data_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        invalidate_cache()
        config = load_config(data_dir / "config.json")
        assert "alice" in config.animas
        assert config.animas["alice"].supervisor == "bob"
        assert config.animas["alice"].speciality == "engineer"

    def test_removed_config_keys_are_ignored_without_breaking_legacy_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "system": {"mode": "server", "log_level": "DEBUG"},
                    "rag": {"quick_check_timeout_seconds": 2.0},
                    "background_task": {"result_retention_hours": 48},
                    "priming": {
                        "profile": "full",
                        "compact_background_recall": {"heartbeat": {"episodes_max_items": 9}},
                    },
                    "channel_company_defaults": {"general": "alpha"},
                }
            ),
            encoding="utf-8",
        )

        config = load_config(path)

        assert config.system.mode == "server"
        assert not hasattr(config.system, "log_level")
        assert not hasattr(config.rag, "quick_check_timeout_seconds")
        assert not hasattr(config.background_task, "result_retention_hours")
        assert not hasattr(config.priming, "profile")
        assert config.priming.compact_background_recall.episodes_max_items == 2
        assert not hasattr(config, "channel_company_defaults")
        assert "channel_company_defaults" not in config.model_dump()

    def test_removed_consolidation_keys_are_ignored(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "consolidation": {
                        "episode_retention_days": 30,
                        "episode_retention_batch_limit": 200,
                    }
                }
            ),
            encoding="utf-8",
        )

        config = load_config(path)

        fields = type(config.consolidation).model_fields
        assert "episode_retention_days" not in fields
        assert "episode_retention_batch_limit" not in fields


# ── save_config ───────────────────────────────────────────


class TestSaveConfig:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_save_creates_file(self, tmp_path):
        config = AnimaWorksConfig()
        path = tmp_path / "config.json"
        save_config(config, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 1

    def test_save_sets_permissions(self, tmp_path):
        config = AnimaWorksConfig()
        path = tmp_path / "config.json"
        save_config(config, path)
        stat = path.stat()
        if os.name == "nt":
            assert os.access(path, os.R_OK)
            assert os.access(path, os.W_OK)
            return
        # 0o600 = owner read/write only
        assert stat.st_mode & 0o777 == 0o600

    def test_save_updates_cache(self, tmp_path):
        config = AnimaWorksConfig()
        path = tmp_path / "config.json"
        save_config(config, path)
        # Should be cached now
        from core.config import io as config_io

        assert config_io._config is config
        assert config_io._config_path == path

    def test_save_creates_parent_dir(self, tmp_path):
        config = AnimaWorksConfig()
        path = tmp_path / "sub" / "dir" / "config.json"
        save_config(config, path)
        assert path.exists()

    def test_save_pretty_prints(self, tmp_path):
        config = AnimaWorksConfig()
        path = tmp_path / "config.json"
        save_config(config, path)
        text = path.read_text(encoding="utf-8")
        assert "\n" in text  # pretty-printed
        assert text.endswith("\n")

    def test_save_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"version": 1, "locale": "ja"}\n', encoding="utf-8")

        config = AnimaWorksConfig(locale="en")
        save_config(config, path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["locale"] == "en"


# ── resolve_anima_config ─────────────────────────────────


class TestResolveAnimaConfig:
    def test_defaults_when_no_anima_entry(self):
        config = AnimaWorksConfig()
        resolved, cred = resolve_anima_config(config, "unknown")
        assert resolved.model == "claude-sonnet-5-5"
        assert resolved.credential == "anthropic"
        assert cred.api_key == ""

    def test_status_json_override(self, tmp_path):
        """status.json overrides anima_defaults for model config."""
        status = {"model": "gpt-4o", "max_tokens": 8192, "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.animas["alice"] = AnimaModelConfig()
        resolved, cred = resolve_anima_config(config, "alice", anima_dir=tmp_path)
        assert resolved.model == "gpt-4o"
        assert resolved.max_tokens == 8192

    def test_legacy_turn_limit_in_status_json_is_ignored(self, tmp_path):
        legacy_key = "max_turns"
        status = {"model": "gpt-4o", legacy_key: 12, "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")

        resolved, _ = resolve_anima_config(AnimaWorksConfig(), "alice", anima_dir=tmp_path)

        assert resolved.model == "gpt-4o"
        assert not hasattr(resolved, legacy_key)

    def test_mode_s_auth_loaded_from_status_json(self, tmp_path):
        """mode_s_auth in status.json is loaded into resolved config."""
        status = {"mode_s_auth": "bedrock", "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.animas["alice"] = AnimaModelConfig()
        resolved, _ = resolve_anima_config(config, "alice", anima_dir=tmp_path)
        assert resolved.mode_s_auth == "bedrock"

    def test_custom_credential(self, tmp_path):
        """status.json credential overrides anima_defaults."""
        status = {"credential": "openai"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.credentials["openai"] = CredentialConfig(api_key="sk-openai")
        config.animas["alice"] = AnimaModelConfig()
        resolved, cred = resolve_anima_config(config, "alice", anima_dir=tmp_path)
        assert resolved.credential == "openai"
        assert cred.api_key == "sk-openai"

    def test_missing_credential_raises(self, tmp_path):
        """KeyError when status.json credential not in config.credentials."""
        status = {"credential": "nonexistent"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.credentials = {}
        config.animas["alice"] = AnimaModelConfig()
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_anima_config(config, "alice", anima_dir=tmp_path)

    def test_partial_overrides(self):
        config = AnimaWorksConfig()
        config.animas["bob"] = AnimaModelConfig(
            supervisor="alice",
        )
        resolved, _ = resolve_anima_config(config, "bob")
        assert resolved.supervisor == "alice"

    def test_status_json_supervisor_overrides_config_animas(self, tmp_path):
        """status.json supervisor takes priority over config.animas."""
        status = {"supervisor": "rin", "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.animas["bob"] = AnimaModelConfig(supervisor="sakura")
        resolved, _ = resolve_anima_config(config, "bob", anima_dir=tmp_path)
        assert resolved.supervisor == "rin"

    def test_status_json_null_supervisor_makes_toplevel(self, tmp_path):
        """Explicit null supervisor in status.json means top-level (no supervisor)."""
        status = {"supervisor": None, "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.animas["bob"] = AnimaModelConfig(supervisor="sakura")
        resolved, _ = resolve_anima_config(config, "bob", anima_dir=tmp_path)
        assert resolved.supervisor is None

    def test_config_animas_supervisor_used_as_fallback(self, tmp_path):
        """config.animas supervisor is fallback when status.json omits it."""
        status = {"model": "gpt-4o", "credential": "anthropic"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        config = AnimaWorksConfig()
        config.animas["bob"] = AnimaModelConfig(supervisor="sakura")
        resolved, _ = resolve_anima_config(config, "bob", anima_dir=tmp_path)
        assert resolved.supervisor == "sakura"

    def test_local_llm_role_preset_applies_when_model_is_not_explicit(self, tmp_path):
        status = {"role": "engineer"}
        (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
        config = AnimaWorksConfig()
        config.credentials["ollama"] = CredentialConfig(type="ollama", base_url="http://127.0.0.1:11434")
        config.anima_defaults.credential = "ollama"
        config.local_llm.default_model = "ollama/qwen2.5-coder:14b"
        resolved, credential = resolve_anima_config(config, "bob", anima_dir=tmp_path)
        assert resolved.model == "ollama/qwen2.5-coder:14b"
        assert resolved.credential == "ollama"
        assert credential.base_url == "http://127.0.0.1:11434"


# ── Pattern specificity ───────────────────────────────────


class TestPatternSpecificity:
    def test_exact_match_beats_wildcard(self):
        exact = _pattern_specificity("ollama/qwen3:14b")
        wildcard = _pattern_specificity("ollama/qwen3:*")
        assert exact < wildcard  # lower = higher priority

    def test_longer_prefix_beats_shorter(self):
        long_prefix = _pattern_specificity("ollama/qwen3-coder:*")
        short_prefix = _pattern_specificity("ollama/*")
        assert long_prefix < short_prefix

    def test_exact_matches_equal_tier(self):
        a = _pattern_specificity("claude-sonnet-4-6")
        b = _pattern_specificity("openai/gpt-4o")
        # Both are exact — tier 0
        assert a[0] == 0
        assert b[0] == 0


# ── Pattern table matching ────────────────────────────────


class TestMatchPatternTable:
    def test_exact_match(self):
        table = {"ollama/qwen3:14b": "A2", "ollama/*": "A"}
        assert _match_pattern_table("ollama/qwen3:14b", table) == "A2"

    def test_prefix_wildcard(self):
        table = {"claude-*": "A1"}
        assert _match_pattern_table("claude-sonnet-4-6", table) == "A1"

    def test_suffix_wildcard(self):
        table = {"ollama/llama4:*": "A2"}
        assert _match_pattern_table("ollama/llama4:scout", table) == "A2"

    def test_partial_wildcard(self):
        table = {"ollama/glm-4.7*": "A2"}
        assert _match_pattern_table("ollama/glm-4.7:9b", table) == "A2"

    def test_no_match_returns_none(self):
        table = {"claude-*": "A1", "openai/*": "A2"}
        assert _match_pattern_table("unknown/model", table) is None

    def test_specific_pattern_beats_catchall(self):
        table = {
            "ollama/qwen3:14b": "A2",
            "ollama/*": "A",
        }
        assert _match_pattern_table("ollama/qwen3:14b", table) == "A2"
        assert _match_pattern_table("ollama/some-other", table) == "A"

    def test_empty_table_returns_none(self):
        assert _match_pattern_table("claude-sonnet-4-6", {}) is None

    def test_case_normalization(self):
        table = {"claude-*": "a1"}
        assert _match_pattern_table("claude-opus-4-6", table) == "A1"


# ── resolve_execution_mode (wildcard) ─────────────────────


class TestResolveExecutionModeWildcard:
    @pytest.fixture(autouse=True)
    def _clear_models_json(self, monkeypatch):
        """Ensure models.json cache does not leak between tests.

        Also patches _load_models_json to return {} so the real
        ~/.animaworks/models.json does not interfere with assertions
        about config.json model_modes and code-default priorities.
        """
        import core.config.model_mode as _mode
        import core.config.models as _m
        from core.config.models import invalidate_models_json_cache

        invalidate_models_json_cache()
        monkeypatch.setattr(_m, "_load_models_json", lambda: {})
        monkeypatch.setattr(_mode, "_load_models_json", lambda: {})
        yield
        invalidate_models_json_cache()

    def test_explicit_override_autonomous(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "any-model", "autonomous") == "A"

    def test_explicit_override_assisted(self):
        config = AnimaWorksConfig()
        # The legacy 'assisted' alias maps to the canonical autonomous mode.
        assert resolve_execution_mode(config, "any-model", "assisted") == "A"

    def test_explicit_override_legacy_a1(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "any-model", "A1") == "S"

    def test_explicit_override_legacy_a1f(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "any-model", "A1F") == "A"

    def test_explicit_override_legacy_a2(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "any-model", "A2") == "A"

    def test_explicit_override_canonical_and_legacy_modes(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "any-model", "S") == "S"
        assert resolve_execution_mode(config, "any-model", "A") == "A"
        assert resolve_execution_mode(config, "any-model", "B") == "A"
        assert resolve_execution_mode(config, "any-model", "b") == "A"
        assert resolve_execution_mode(config, "any-model", "basic") == "A"
        assert _normalise_mode("B") == "A"
        assert _normalise_mode("b") == "A"
        assert _normalise_mode("basic") == "A"

    def test_unknown_mode_defaults_to_a_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="animaworks.config"):
            assert _normalise_mode("Z") == "A"
        assert "Unrecognised execution mode 'Z'" in caplog.text

    def test_fallback_legacy_b_mode_normalizes_to_a(self):
        config = AnimaWorksConfig()
        assert parse_fallback_entry("b:ollama/qwen3:14b", config) == ("a", "ollama/qwen3:14b")

    def test_resolved_b_mode_is_canonical_a(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "x", "B") == "A"

    def test_claude_wildcard_s(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "claude-sonnet-4-6") == "S"
        assert resolve_execution_mode(config, "claude-opus-4-6") == "S"
        # Future Claude model also matches
        assert resolve_execution_mode(config, "claude-haiku-5-20260101") == "S"

    def test_cloud_api_providers_a(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "openai/gpt-4.1") == "A"
        assert resolve_execution_mode(config, "google/gemini-2.5-pro") == "A"
        assert resolve_execution_mode(config, "zai/zai-model") == "A"
        assert resolve_execution_mode(config, "minimax/some-model") == "A"
        assert resolve_execution_mode(config, "moonshot/kimi") == "A"

    def test_openai_codex_provider_routes_to_c(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "openai-codex/gpt-5.3-codex") == "C"

    def test_grok_model_routes_to_x(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "grok/grok-4.5") == "X"

    def test_grok_wildcard_routes_future_models_to_x(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "grok/future-model") == "X"

    def test_explicit_x_override_takes_priority(self):
        config = AnimaWorksConfig(model_modes={"grok/*": "A"})
        assert resolve_execution_mode(config, "grok/grok-4.5", "X") == "X"

    def test_normalise_lowercase_x(self):
        assert _normalise_mode("x") == "X"

    def test_agy_model_routes_to_y(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "agy/gemini-4-argon") == "Y"

    def test_agy_wildcard_routes_future_models_to_y(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "agy/whatever") == "Y"

    def test_explicit_y_override_takes_priority(self):
        config = AnimaWorksConfig(model_modes={"agy/*": "A"})
        assert resolve_execution_mode(config, "agy/gemini-4-argon", "Y") == "Y"

    def test_canonical_modes_include_y(self):
        assert "Y" in CANONICAL_MODES
        assert _normalise_mode("y") == "Y"

    def test_ollama_a_models(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "ollama/qwen3:14b") == "A"
        assert resolve_execution_mode(config, "ollama/qwen3:30b") == "A"
        assert resolve_execution_mode(config, "ollama/llama4:scout") == "A"

    def test_ollama_small_models_default_a(self):
        config = AnimaWorksConfig()
        # Mode B was removed; small local models now default to A
        assert resolve_execution_mode(config, "ollama/qwen3:8b") == "A"
        assert resolve_execution_mode(config, "ollama/deepseek-r1:14b") == "A"
        assert resolve_execution_mode(config, "ollama/gemma3:27b") == "A"

    def test_unknown_model_fallback_a(self):
        config = AnimaWorksConfig()
        assert resolve_execution_mode(config, "totally/unknown-model") == "A"

    def test_config_legacy_values_normalised(self):
        # config.json model_modes with legacy A2 value → normalised to A
        # Patch models.json lookup so config.model_modes is used (models.json takes priority)
        with patch("core.config.model_mode._match_models_json", return_value=None):
            config = AnimaWorksConfig(model_modes={"ollama/qwen3:8b": "A2"})
            assert resolve_execution_mode(config, "ollama/qwen3:8b") == "A"

    def test_config_wildcard_overrides_code_default(self):
        with patch("core.config.model_mode._match_models_json", return_value=None):
            config = AnimaWorksConfig(model_modes={"ollama/gemma3*": "A"})
            # Code default says A for gemma3*, but config.json overrides
            assert resolve_execution_mode(config, "ollama/gemma3:27b") == "A"

    def test_empty_config_falls_through_to_code(self):
        config = AnimaWorksConfig(model_modes={})
        assert resolve_execution_mode(config, "claude-sonnet-4-6") == "S"
        assert resolve_execution_mode(config, "openai/gpt-4.1") == "A"
        assert resolve_execution_mode(config, "ollama/qwen3:8b") == "A"


# ── load_model_config ─────────────────────────────────────


class TestLoadModelConfig:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_returns_model_config(self, data_dir):
        from core.schemas import ModelConfig

        anima_dir = data_dir / "animas" / "test-anima"
        anima_dir.mkdir(parents=True, exist_ok=True)

        mc = load_model_config(anima_dir)
        assert isinstance(mc, ModelConfig)
        # Should use default model from anima_defaults
        assert mc.model == "claude-sonnet-5-5"

    def test_inherits_defaults(self, data_dir):
        anima_dir = data_dir / "animas" / "unknown"
        anima_dir.mkdir(parents=True, exist_ok=True)

        mc = load_model_config(anima_dir)
        # Values come from DEFAULT_TEST_CONFIG, not AnimaDefaults class defaults
        assert mc.max_tokens == 1024
        assert mc.context_threshold == 0.50
        assert mc.conversation_history_threshold == 0.30

    def test_anima_override(self, data_dir):
        import json as _json

        # Model override now in status.json
        config_data = {
            "version": 1,
            "credentials": {"anthropic": {"api_key": "sk-test"}},
            "anima_defaults": {"model": "claude-sonnet-4-6", "credential": "anthropic"},
            "animas": {"alice": {}},
        }
        (data_dir / "config.json").write_text(
            _json.dumps(config_data),
            encoding="utf-8",
        )
        invalidate_cache()

        anima_dir = data_dir / "animas" / "alice"
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / "status.json").write_text(
            _json.dumps({"model": "openai/gpt-4o", "max_tokens": 8192}),
        )

        mc = load_model_config(anima_dir)
        assert mc.model == "openai/gpt-4o"
        assert mc.max_tokens == 8192

    def test_resolves_credential(self, data_dir):
        import json as _json

        config_data = {
            "version": 1,
            "credentials": {
                "anthropic": {"api_key": ""},
                "openai": {"api_key": "sk-openai", "base_url": "https://api.openai.com"},
            },
            "anima_defaults": {"model": "claude-sonnet-4-6", "credential": "anthropic"},
            "animas": {"bob": {}},
        }
        (data_dir / "config.json").write_text(
            _json.dumps(config_data),
            encoding="utf-8",
        )
        invalidate_cache()

        anima_dir = data_dir / "animas" / "bob"
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / "status.json").write_text(
            _json.dumps({"credential": "openai"}),
        )

        mc = load_model_config(anima_dir)
        assert mc.api_key == "sk-openai"
        assert mc.api_base_url == "https://api.openai.com"
        assert mc.credential == "openai"
        assert mc.credential_type == "api_key"

    def test_resolves_codex_azure_credential_metadata(self, data_dir):
        import json as _json

        config_data = {
            "version": 1,
            "credentials": {
                "anthropic": {"api_key": ""},
                "azure_codex": {
                    "type": "codex_azure",
                    "api_key": "az-key",
                    "base_url": "https://example-resource.openai.azure.com",
                    "keys": {
                        "api_version": "2025-04-01-preview",
                        "codex_model": "gpt-55-prod",
                    },
                },
            },
            "anima_defaults": {"model": "claude-sonnet-4-6", "credential": "anthropic"},
            "animas": {"bob": {}},
        }
        (data_dir / "config.json").write_text(
            _json.dumps(config_data),
            encoding="utf-8",
        )
        invalidate_cache()

        anima_dir = data_dir / "animas" / "bob"
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / "status.json").write_text(
            _json.dumps({"model": "codex/gpt-5.5", "credential": "azure_codex"}),
        )

        mc = load_model_config(anima_dir)
        assert mc.credential == "azure_codex"
        assert mc.credential_type == "codex_azure"
        assert mc.api_key == "az-key"
        assert mc.api_base_url == "https://example-resource.openai.azure.com"
        assert mc.extra_keys["api_version"] == "2025-04-01-preview"
        assert mc.extra_keys["codex_model"] == "gpt-55-prod"


# ── read_anima_supervisor ────────────────────────────────


class TestReadAnimaSupervisor:
    """Tests for read_anima_supervisor helper."""

    def test_from_status_json(self, tmp_path: Path) -> None:
        """Reads supervisor from status.json."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "bob"}),
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "bob"

    def test_from_identity_md(self, tmp_path: Path) -> None:
        """Falls back to IDENTITY.md table when no status.json."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "| 上司 | charlie |\n",
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "charlie"

    def test_status_json_takes_priority(self, tmp_path: Path) -> None:
        """status.json supervisor wins over IDENTITY.md."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "bob"}),
            encoding="utf-8",
        )
        (anima_dir / "IDENTITY.md").write_text(
            "| 上司 | charlie |\n",
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "bob"

    def test_nashi_returns_none(self, tmp_path: Path) -> None:
        """Supervisor value 'なし' is treated as no supervisor."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "なし"}),
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) is None

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        """Returns None when anima dir has no status.json or IDENTITY.md."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        assert read_anima_supervisor(anima_dir) is None

    def test_japanese_name_with_parens(self, tmp_path: Path) -> None:
        """Japanese name with parenthesised English name is resolved."""
        anima_dir = tmp_path / "hinata"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "| 上司 | 琴葉（kotoha） |\n",
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "kotoha"

    def test_invalid_json_falls_back(self, tmp_path: Path) -> None:
        """Falls back to IDENTITY.md when status.json is invalid JSON."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text("bad json", encoding="utf-8")
        (anima_dir / "IDENTITY.md").write_text(
            "| 上司 | bob |\n",
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "bob"

    def test_empty_supervisor_in_status(self, tmp_path: Path) -> None:
        """Empty supervisor in status.json falls back to IDENTITY.md."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": ""}),
            encoding="utf-8",
        )
        (anima_dir / "IDENTITY.md").write_text(
            "| 上司 | bob |\n",
            encoding="utf-8",
        )
        assert read_anima_supervisor(anima_dir) == "bob"


# ── register_anima_in_config ─────────────────────────────


class TestRegisterAnimaInConfig:
    """Tests for register_anima_in_config helper."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_registers_new_anima_with_supervisor(self, tmp_path: Path) -> None:
        """New anima is added to config.json with supervisor from status.json."""
        data_dir = tmp_path
        config_path = data_dir / "config.json"
        save_config(AnimaWorksConfig(), config_path)
        invalidate_cache()

        anima_dir = data_dir / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "bob"}),
            encoding="utf-8",
        )

        register_anima_in_config(data_dir, "alice")

        invalidate_cache()
        cfg = load_config(config_path)
        assert "alice" in cfg.animas
        assert cfg.animas["alice"].supervisor == "bob"

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        """Existing anima entry is not overwritten."""
        data_dir = tmp_path
        config_path = data_dir / "config.json"
        cfg = AnimaWorksConfig()
        cfg.animas["alice"] = AnimaModelConfig(
            supervisor="charlie",
            speciality="engineer",
        )
        save_config(cfg, config_path)
        invalidate_cache()

        anima_dir = data_dir / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"supervisor": "bob"}),
            encoding="utf-8",
        )

        register_anima_in_config(data_dir, "alice")

        invalidate_cache()
        cfg = load_config(config_path)
        assert cfg.animas["alice"].supervisor == "charlie"  # unchanged
        assert cfg.animas["alice"].speciality == "engineer"  # unchanged

    def test_no_config_file_noop(self, tmp_path: Path) -> None:
        """Does nothing when config.json does not exist."""
        register_anima_in_config(tmp_path, "alice")
        # Should not raise

    def test_no_anima_dir_registers_none_supervisor(self, tmp_path: Path) -> None:
        """Anima dir doesn't exist — registers with supervisor=None."""
        data_dir = tmp_path
        config_path = data_dir / "config.json"
        save_config(AnimaWorksConfig(), config_path)
        invalidate_cache()

        register_anima_in_config(data_dir, "alice")

        invalidate_cache()
        cfg = load_config(config_path)
        assert "alice" in cfg.animas
        assert cfg.animas["alice"].supervisor is None
