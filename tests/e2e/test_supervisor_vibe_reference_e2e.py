# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for supervisor image as Vibe Transfer reference.

Tests the full flow from dispatch handler through ImageGenPipeline config
resolution, verifying that supervisor's avatar image is correctly resolved
and applied as the Vibe Transfer reference.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import AnimaWorksConfig, ImageGenConfig


@pytest.fixture
def animas_dir(data_dir: Path) -> Path:
    """Return the animas directory within the test data_dir."""
    d = data_dir / "animas"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def supervisor_with_image(animas_dir: Path) -> Path:
    """Create a supervisor anima with a fullbody avatar image."""
    supervisor_dir = animas_dir / "sakura"
    assets_dir = supervisor_dir / "assets"
    assets_dir.mkdir(parents=True)
    fullbody = assets_dir / "avatar_fullbody.png"
    fullbody.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # Minimal PNG header
    return supervisor_dir


@pytest.fixture
def supervisor_without_image(animas_dir: Path) -> Path:
    """Create a supervisor anima without avatar assets."""
    supervisor_dir = animas_dir / "mio"
    supervisor_dir.mkdir(parents=True)
    return supervisor_dir


@pytest.fixture
def subordinate_dir(animas_dir: Path) -> Path:
    """Create a subordinate anima directory."""
    d = animas_dir / "rin"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock_pipeline_result():
    """Create a mock PipelineResult."""
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "fullbody_path": "assets/avatar_fullbody.png",
        "errors": [],
    }
    return mock_result


def _dispatch_generate(args: dict, config: AnimaWorksConfig, animas_dir: Path):
    """Call dispatch() with proper patches for generate_character_assets."""
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "fullbody_path": "assets/avatar_fullbody.png",
        "errors": [],
    }

    captured_configs: list[ImageGenConfig] = []

    original_init = None

    def capture_pipeline_init(self, anima_dir, config=None):
        captured_configs.append(config)
        self._anima_dir = anima_dir
        self._assets_dir = anima_dir / "assets"
        self._config = config or ImageGenConfig()

    def mock_generate_all(self, **kwargs):
        return mock_result

    with (
        patch("core.config.models.load_config", return_value=config),
        patch("core.tools.image_gen.ImageGenPipeline.__init__", capture_pipeline_init),
        patch("core.tools.image_gen.ImageGenPipeline.generate_all", mock_generate_all),
        patch("core.tools.image_gen.logger"),
    ):
        from core.tools.image_gen import dispatch
        result = dispatch("generate_character_assets", dict(args))

    return result, captured_configs, mock_result


class TestSupervisorVibeReferenceE2E:
    """End-to-end tests for supervisor image -> Vibe Transfer reference flow."""

    def test_full_flow_supervisor_image_applied(
        self,
        data_dir: Path,
        supervisor_with_image: Path,
        subordinate_dir: Path,
    ):
        """Full flow: supervisor has image -> subordinate gets it as vibe reference."""
        config = AnimaWorksConfig(
            image_gen=ImageGenConfig(
                style_prefix="anime, ",
                vibe_strength=0.8,
                vibe_info_extracted=0.7,
            ),
        )

        result, captured_configs, _ = _dispatch_generate(
            {
                "anima_dir": str(subordinate_dir),
                "prompt": "1girl, short hair, blue eyes",
                "negative_prompt": "lowres",
                "supervisor_name": "sakura",
            },
            config,
            data_dir / "animas",
        )

        assert len(captured_configs) == 1
        config_used = captured_configs[0]
        expected_path = str(
            supervisor_with_image / "assets" / "avatar_fullbody.png"
        )
        assert config_used.style_reference == expected_path
        # Vibe transfer settings preserved from global config
        assert config_used.vibe_strength == 0.8
        assert config_used.vibe_info_extracted == 0.7
        assert config_used.style_prefix == "anime, "

        assert result == {"fullbody_path": "assets/avatar_fullbody.png", "errors": []}

    def test_full_flow_no_supervisor_image_fallback(
        self,
        data_dir: Path,
        supervisor_without_image: Path,
        subordinate_dir: Path,
    ):
        """Full flow: supervisor exists but no image -> falls back to global config."""
        config = AnimaWorksConfig(
            image_gen=ImageGenConfig(
                style_reference="/global/org_style.png",
                vibe_strength=0.6,
            ),
        )

        _, captured_configs, _ = _dispatch_generate(
            {
                "anima_dir": str(subordinate_dir),
                "prompt": "1girl",
                "supervisor_name": "mio",
            },
            config,
            data_dir / "animas",
        )

        assert len(captured_configs) == 1
        config_used = captured_configs[0]
        # Global style_reference preserved (supervisor has no image)
        assert config_used.style_reference == "/global/org_style.png"

    def test_full_flow_no_supervisor_specified(
        self,
        data_dir: Path,
        subordinate_dir: Path,
    ):
        """Full flow: no supervisor_name -> global config used as-is."""
        config = AnimaWorksConfig(
            image_gen=ImageGenConfig(style_reference="/global/style.png"),
        )

        _, captured_configs, _ = _dispatch_generate(
            {
                "anima_dir": str(subordinate_dir),
                "prompt": "1girl",
            },
            config,
            data_dir / "animas",
        )

        assert len(captured_configs) == 1
        config_used = captured_configs[0]
        assert config_used is config.image_gen

    def test_supervisor_image_overrides_global_reference(
        self,
        data_dir: Path,
        supervisor_with_image: Path,
        subordinate_dir: Path,
    ):
        """Supervisor's image takes priority over global style_reference."""
        config = AnimaWorksConfig(
            image_gen=ImageGenConfig(
                style_reference="/global/org_style.png",
            ),
        )

        _, captured_configs, _ = _dispatch_generate(
            {
                "anima_dir": str(subordinate_dir),
                "prompt": "1girl",
                "supervisor_name": "sakura",
            },
            config,
            data_dir / "animas",
        )

        assert len(captured_configs) == 1
        config_used = captured_configs[0]
        # Supervisor image should win over global style_reference
        expected_path = str(
            supervisor_with_image / "assets" / "avatar_fullbody.png"
        )
        assert config_used.style_reference == expected_path
        assert config_used.style_reference != "/global/org_style.png"

    def test_original_config_not_mutated(
        self,
        data_dir: Path,
        supervisor_with_image: Path,
        subordinate_dir: Path,
    ):
        """Applying supervisor image must not mutate the original ImageGenConfig."""
        original_ref = "/global/org_style.png"
        config = AnimaWorksConfig(
            image_gen=ImageGenConfig(style_reference=original_ref),
        )

        _dispatch_generate(
            {
                "anima_dir": str(subordinate_dir),
                "prompt": "1girl",
                "supervisor_name": "sakura",
            },
            config,
            data_dir / "animas",
        )

        # Original config object must remain unchanged
        assert config.image_gen.style_reference == original_ref

    def test_schema_includes_supervisor_name_param(self):
        """The generate_character_assets schema should include supervisor_name."""
        from core.tools.image_gen import get_tool_schemas

        schemas = get_tool_schemas()
        gen_schema = next(s for s in schemas if s["name"] == "generate_character_assets")
        props = gen_schema["input_schema"]["properties"]
        assert "supervisor_name" in props
        assert props["supervisor_name"]["type"] == "string"

    def test_multiple_subordinates_get_same_supervisor_reference(
        self,
        data_dir: Path,
        supervisor_with_image: Path,
        animas_dir: Path,
    ):
        """Multiple subordinates created by the same supervisor get same vibe reference."""
        config = AnimaWorksConfig()
        all_captured_configs: list[ImageGenConfig] = []

        subordinates = ["rin", "yui", "hana"]
        for name in subordinates:
            sub_dir = animas_dir / name
            sub_dir.mkdir(parents=True, exist_ok=True)

            _, captured, _ = _dispatch_generate(
                {
                    "anima_dir": str(sub_dir),
                    "prompt": "1girl",
                    "supervisor_name": "sakura",
                },
                config,
                animas_dir,
            )
            all_captured_configs.extend(captured)

        expected_path = str(
            supervisor_with_image / "assets" / "avatar_fullbody.png"
        )
        assert len(all_captured_configs) == 3
        for cfg in all_captured_configs:
            assert cfg.style_reference == expected_path
