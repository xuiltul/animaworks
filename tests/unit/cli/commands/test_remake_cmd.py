"""Unit tests for cli/commands/remake_cmd.py — remake-assets subcommand."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.remake_cmd import ALL_STEPS, _run, register


# ── register() ─────────────────────────────────────────────


class TestRegister:
    """Tests for the register() function that adds the subparser."""

    def test_register_adds_subcommand(self):
        """register() should add a 'remake-assets' subparser."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        register(subparsers)

        # Parsing 'remake-assets' with required args should succeed
        args = parser.parse_args([
            "remake-assets", "rin", "--style-from", "miku",
        ])
        assert args.command == "remake-assets"
        assert args.anima == "rin"
        assert args.style_from == "miku"
        assert args.dry_run is False
        assert args.no_backup is False
        assert args.vibe_strength == 0.6
        assert args.vibe_info_extracted == 0.8
        assert args.seed is None
        assert args.steps is None
        assert args.prompt is None
        assert hasattr(args, "func")


# ── Validation tests ─────────────────────────────────────────


def _make_namespace(**overrides) -> argparse.Namespace:
    """Build a Namespace with all remake-assets defaults, applying overrides."""
    defaults = {
        "anima": "rin",
        "style_from": "miku",
        "steps": None,
        "prompt": None,
        "vibe_strength": 0.6,
        "vibe_info_extracted": 0.8,
        "seed": None,
        "no_backup": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidateTargetAnima:
    """Target anima must exist in data_dir."""

    def test_validate_target_anima_missing(self, data_dir, make_anima, capsys):
        """Error when the target anima directory does not exist."""
        # Create style-from anima but NOT the target
        style_dir = make_anima("miku")
        assets = style_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "avatar_fullbody.png").write_bytes(b"PNG_FAKE")

        args = _make_namespace(anima="nonexistent", style_from="miku")
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "nonexistent" in captured.out


class TestValidateStyleFrom:
    """Style-from anima must exist and have a fullbody image."""

    def test_validate_style_from_missing(self, data_dir, make_anima, capsys):
        """Error when style-from anima does not exist."""
        make_anima("rin")
        args = _make_namespace(anima="rin", style_from="nonexistent")
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "nonexistent" in captured.out

    def test_validate_style_from_no_fullbody(self, data_dir, make_anima, capsys):
        """Error when style-from anima exists but lacks avatar_fullbody.png."""
        make_anima("rin")
        style_dir = make_anima("miku")
        # assets dir exists but no fullbody image
        (style_dir / "assets").mkdir(exist_ok=True)

        args = _make_namespace(anima="rin", style_from="miku")
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "avatar_fullbody.png" in captured.out


class TestValidatePrompt:
    """Prompt must be available from --prompt or prompt.txt."""

    def test_validate_no_prompt(self, data_dir, make_anima, capsys):
        """Error when no prompt.txt and no --prompt flag."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        # Target has no prompt.txt
        (target_dir / "assets").mkdir(exist_ok=True)

        # Style-from has the required fullbody image
        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_FAKE")

        args = _make_namespace(anima="rin", style_from="miku", prompt=None)
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "prompt" in captured.out.lower()


class TestValidateSteps:
    """Invalid step names should be rejected."""

    def test_validate_invalid_steps(self, data_dir, make_anima, capsys):
        """Error on unrecognised step names."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, test prompt")

        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_FAKE")

        args = _make_namespace(
            anima="rin", style_from="miku", steps="fullbody,bogus_step",
        )
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "bogus_step" in captured.out


class TestValidateVibeStrength:
    """Vibe strength/info parameters must be in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        ("strength", "info"),
        [
            (-0.1, 0.8),
            (1.5, 0.8),
            (0.6, -0.1),
            (0.6, 1.5),
        ],
    )
    def test_validate_vibe_strength_range(
        self, data_dir, make_anima, capsys, strength, info,
    ):
        """Error when vibe-strength or vibe-info-extracted is out of range."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, test prompt")

        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_FAKE")

        args = _make_namespace(
            anima="rin",
            style_from="miku",
            vibe_strength=strength,
            vibe_info_extracted=info,
        )
        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out


# ── Dry-run ─────────────────────────────────────────────────


class TestDryRun:
    """--dry-run prints a plan without calling the pipeline."""

    @patch("cli.commands.remake_cmd.ImageGenPipeline", create=True)
    def test_dry_run_output(self, mock_pipeline_cls, data_dir, make_anima, capsys):
        """Dry-run prints the plan and does not invoke the pipeline."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        # Set up target assets with prompt
        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, test prompt")
        (assets / "avatar_fullbody.png").write_bytes(b"PNG_FAKE_DATA")

        # Set up style reference
        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_STYLE_DATA")

        args = _make_namespace(anima="rin", style_from="miku", dry_run=True)
        _run(args)

        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "No API calls were made" in captured.out
        # Pipeline should NOT have been instantiated
        mock_pipeline_cls.assert_not_called()


# ── Backup behaviour ────────────────────────────────────────


class TestBackup:
    """Backup is created before pipeline runs unless --no-backup."""

    def test_backup_created(self, data_dir, make_anima, capsys):
        """Existing assets are backed up when running the pipeline."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        # Create target assets
        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, test prompt")
        (assets / "avatar_fullbody.png").write_bytes(b"PNG_ORIGINAL")

        # Create style reference
        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_STYLE")

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.fullbody_path = None
        mock_result.bustup_paths = {}
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.skipped = []
        mock_result.errors = []
        mock_pipeline.generate_all.return_value = mock_result

        mock_pipeline_cls = MagicMock(return_value=mock_pipeline)
        args = _make_namespace(anima="rin", style_from="miku")

        with patch("core.tools.image_gen.ImageGenPipeline", mock_pipeline_cls):
            _run(args)

        captured = capsys.readouterr()
        assert "Backup created" in captured.out

        # Verify a backup directory was created
        backup_dirs = [
            d for d in target_dir.iterdir()
            if d.is_dir() and d.name.startswith("assets_backup_")
        ]
        assert len(backup_dirs) == 1

        # Original file is preserved in backup
        backup_dir = backup_dirs[0]
        assert (backup_dir / "avatar_fullbody.png").read_bytes() == b"PNG_ORIGINAL"

    def test_no_backup_flag(self, data_dir, make_anima, capsys):
        """--no-backup skips creating a backup directory."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, test prompt")
        (assets / "avatar_fullbody.png").write_bytes(b"PNG_ORIGINAL")

        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_STYLE")

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.fullbody_path = None
        mock_result.bustup_paths = {}
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.skipped = []
        mock_result.errors = []
        mock_pipeline.generate_all.return_value = mock_result

        mock_pipeline_cls = MagicMock(return_value=mock_pipeline)
        args = _make_namespace(anima="rin", style_from="miku", no_backup=True)

        with patch("core.tools.image_gen.ImageGenPipeline", mock_pipeline_cls):
            _run(args)

        # No backup directory should have been created
        backup_dirs = [
            d for d in target_dir.iterdir()
            if d.is_dir() and d.name.startswith("assets_backup_")
        ]
        assert len(backup_dirs) == 0


# ── Pipeline invocation ─────────────────────────────────────


class TestPipelineInvocation:
    """Verify ImageGenPipeline.generate_all is called with correct arguments."""

    def test_pipeline_called_with_correct_args(self, data_dir, make_anima, capsys):
        """Mock ImageGenPipeline and verify generate_all kwargs."""
        target_dir = make_anima("rin")
        style_dir = make_anima("miku")

        assets = target_dir / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "prompt.txt").write_text("1girl, silver hair, red eyes")

        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(b"PNG_VIBE_IMAGE")

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.fullbody_path = None
        mock_result.bustup_paths = {}
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.skipped = []
        mock_result.errors = []
        mock_pipeline.generate_all.return_value = mock_result

        mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

        args = _make_namespace(
            anima="rin",
            style_from="miku",
            vibe_strength=0.7,
            vibe_info_extracted=0.9,
            seed=42,
            steps="fullbody,bustup",
        )

        with patch("core.tools.image_gen.ImageGenPipeline", mock_pipeline_cls):
            _run(args)

        # Pipeline was instantiated with the target anima directory
        mock_pipeline_cls.assert_called_once()
        init_kwargs = mock_pipeline_cls.call_args
        assert init_kwargs[0][0] == target_dir

        # generate_all was called with correct parameters
        mock_pipeline.generate_all.assert_called_once()
        call_kwargs = mock_pipeline.generate_all.call_args[1]

        assert call_kwargs["prompt"] == "1girl, silver hair, red eyes"
        assert call_kwargs["skip_existing"] is False
        assert call_kwargs["steps"] == ["fullbody", "bustup"]
        assert call_kwargs["vibe_image"] == b"PNG_VIBE_IMAGE"
        assert call_kwargs["vibe_strength"] == 0.7
        assert call_kwargs["vibe_info_extracted"] == 0.9
        assert call_kwargs["seed"] == 42
        assert call_kwargs["progress_callback"] is not None
