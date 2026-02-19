# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the remake-assets workflow.

Tests the complete remake flow:
  - POST /api/animas/{name}/assets/remake-preview  (preview generation)
  - POST /api/animas/{name}/assets/remake-confirm   (cascade rebuild)
  - DELETE /api/animas/{name}/assets/remake-preview  (cancel & restore)
  - CLI remake-assets --dry-run

All external API calls (NovelAI, fal.ai, Meshy) are mocked via
``unittest.mock.patch`` on ``ImageGenPipeline.generate_all``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _make_test_app(animas_dir: Path):
    """Create a test FastAPI app with the assets router and mock ws_manager."""
    from fastapi import FastAPI

    from server.routes.assets import create_assets_router

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_assets_router()
    app.include_router(router, prefix="/api")
    return app


def _make_pipeline_result(
    *,
    fullbody_path: Path | None = None,
    bustup_paths: dict[str, Path] | None = None,
    chibi_path: Path | None = None,
    model_path: Path | None = None,
    rigged_model_path: Path | None = None,
    animation_paths: dict[str, Path] | None = None,
    errors: list[str] | None = None,
):
    """Create a mock PipelineResult."""
    mock = MagicMock()
    mock.fullbody_path = fullbody_path
    mock.bustup_path = None
    mock.bustup_paths = bustup_paths or {}
    mock.chibi_path = chibi_path
    mock.model_path = model_path
    mock.rigged_model_path = rigged_model_path
    mock.animation_paths = animation_paths or {}
    mock.errors = errors or []
    mock.skipped = []
    mock.to_dict.return_value = {
        "fullbody": str(fullbody_path) if fullbody_path else None,
        "errors": mock.errors,
    }
    return mock


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _setup_anima_with_assets(
    animas_dir: Path,
    name: str,
    *,
    with_prompt: bool = True,
    with_fullbody: bool = True,
) -> Path:
    """Create an anima directory with optional assets and prompt.txt."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(
        f"# {name}\nA test anima.\n", encoding="utf-8",
    )
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    if with_prompt:
        (assets_dir / "prompt.txt").write_text(
            "1girl, black hair, red eyes, school uniform, full body",
            encoding="utf-8",
        )
    if with_fullbody:
        (assets_dir / "avatar_fullbody.png").write_bytes(_FAKE_PNG)
    return anima_dir


# ── E2E: Remake Preview ─────────────────────────────────


class TestRemakePreview:
    """E2E tests for POST /api/animas/{name}/assets/remake-preview."""

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_preview_returns_preview_url(
        self, mock_pipeline_cls, tmp_path,
    ):
        """Successful preview returns preview_url, seed_used, and backup_id."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        # Target anima (assets to remake)
        target_dir = _setup_anima_with_assets(
            animas_dir, "target", with_prompt=True, with_fullbody=True,
        )
        # Style-from anima (reference)
        _setup_anima_with_assets(
            animas_dir, "style-ref", with_fullbody=True,
        )

        # Mock pipeline
        result_path = target_dir / "assets" / "avatar_fullbody.png"
        mock_result = _make_pipeline_result(fullbody_path=result_path)
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={
                    "style_from": "style-ref",
                    "vibe_strength": 0.7,
                    "seed": 42,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "preview_url" in data
        assert data["preview_url"] == "/api/animas/target/assets/avatar_fullbody.png"
        assert data["seed_used"] == 42
        assert "backup_id" in data
        assert data["backup_id"].startswith("assets_backup_")

        # Verify pipeline was called with correct args
        mock_pipeline.generate_all.assert_called_once()
        call_kwargs = mock_pipeline.generate_all.call_args.kwargs
        assert call_kwargs["steps"] == ["fullbody"]
        assert call_kwargs["skip_existing"] is False
        assert call_kwargs["seed"] == 42
        assert call_kwargs["vibe_strength"] == 0.7
        assert call_kwargs["vibe_image"] is not None  # style-ref bytes loaded

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_preview_creates_backup(
        self, mock_pipeline_cls, tmp_path,
    ):
        """Preview endpoint creates a backup directory of existing assets."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        target_dir = _setup_anima_with_assets(animas_dir, "target")
        _setup_anima_with_assets(animas_dir, "style-ref")

        # Add an extra asset to verify backup completeness
        (target_dir / "assets" / "avatar_bustup.png").write_bytes(_FAKE_PNG)

        mock_result = _make_pipeline_result(
            fullbody_path=target_dir / "assets" / "avatar_fullbody.png",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )

        assert resp.status_code == 200
        backup_id = resp.json()["backup_id"]

        # Verify backup directory was created
        backup_dir = target_dir / backup_id
        assert backup_dir.exists()
        assert backup_dir.is_dir()

        # Verify backup contains the original files
        assert (backup_dir / "avatar_fullbody.png").exists()
        assert (backup_dir / "avatar_bustup.png").exists()
        assert (backup_dir / "prompt.txt").exists()

    async def test_remake_preview_missing_style_anima(self, tmp_path):
        """Returns 404 when style_from anima does not exist."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        _setup_anima_with_assets(animas_dir, "target")

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "nonexistent"},
            )

        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]

    async def test_remake_preview_missing_fullbody(self, tmp_path):
        """Returns 404 when style-from anima has no avatar_fullbody.png."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        _setup_anima_with_assets(animas_dir, "target")
        # Style-ref exists but has NO fullbody
        _setup_anima_with_assets(
            animas_dir, "style-ref", with_fullbody=False,
        )

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )

        assert resp.status_code == 404
        assert "avatar_fullbody.png" in resp.json()["detail"]

    async def test_remake_preview_target_not_found(self, tmp_path):
        """Returns 404 when target anima does not exist."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/ghost/assets/remake-preview",
                json={"style_from": "anyone"},
            )

        assert resp.status_code == 404
        assert "ghost" in resp.json()["detail"]

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_preview_emits_websocket_event(
        self, mock_pipeline_cls, tmp_path,
    ):
        """Preview broadcasts anima.remake_preview_ready via WebSocket."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        target_dir = _setup_anima_with_assets(animas_dir, "target")
        _setup_anima_with_assets(animas_dir, "style-ref")

        mock_result = _make_pipeline_result(
            fullbody_path=target_dir / "assets" / "avatar_fullbody.png",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )

        assert resp.status_code == 200

        ws = app.state.ws_manager
        preview_events = []
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if (
                isinstance(payload, dict)
                and payload.get("type") == "anima.remake_preview_ready"
            ):
                preview_events.append(payload)

        assert len(preview_events) == 1
        event_data = preview_events[0]["data"]
        assert event_data["name"] == "target"
        assert "preview_url" in event_data
        assert "backup_id" in event_data

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_preview_restores_backup_on_failure(
        self, mock_pipeline_cls, tmp_path,
    ):
        """On generation failure, assets are restored from backup."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        target_dir = _setup_anima_with_assets(animas_dir, "target")
        _setup_anima_with_assets(animas_dir, "style-ref")

        # Mark original assets with known content
        original_content = b"original-fullbody-content"
        (target_dir / "assets" / "avatar_fullbody.png").write_bytes(original_content)

        # Mock pipeline to return errors
        mock_result = _make_pipeline_result(errors=["fullbody: 500 Internal Server Error"])
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )

        assert resp.status_code == 500
        assert "Preview generation failed" in resp.json()["detail"]

        # Verify assets were restored (original content preserved)
        restored = (target_dir / "assets" / "avatar_fullbody.png").read_bytes()
        assert restored == original_content

        # Verify no orphaned backup directories remain
        backup_dirs = list(target_dir.glob("assets_backup_*"))
        assert len(backup_dirs) == 0

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_preview_no_orphan_backups_on_pipeline_error(
        self, mock_pipeline_cls, tmp_path,
    ):
        """Multiple failed attempts should not leave orphaned backup directories."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        target_dir = _setup_anima_with_assets(animas_dir, "target")
        _setup_anima_with_assets(animas_dir, "style-ref")

        mock_result = _make_pipeline_result(errors=["fullbody: timeout"])
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)

        # Two failed attempts
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )
            await client.post(
                "/api/animas/target/assets/remake-preview",
                json={"style_from": "style-ref"},
            )

        # No orphaned backups should remain
        backup_dirs = list(target_dir.glob("assets_backup_*"))
        assert len(backup_dirs) == 0


# ── E2E: Remake Confirm ──────────────────────────────────


class TestRemakeConfirm:
    """E2E tests for POST /api/animas/{name}/assets/remake-confirm."""

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_remake_confirm_starts_cascade(
        self, mock_pipeline_cls, tmp_path,
    ):
        """Confirm returns started status with remaining steps list."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        target_dir = _setup_anima_with_assets(animas_dir, "target")

        # Create a backup dir to simulate having done preview first
        backup_dir = target_dir / "assets_backup_20260216_120000"
        shutil.copytree(target_dir / "assets", backup_dir)

        # Mock the pipeline for background cascade
        mock_result = _make_pipeline_result(
            bustup_paths={"neutral": target_dir / "assets" / "avatar_bustup.png"},
            chibi_path=target_dir / "assets" / "avatar_chibi.png",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-confirm",
                json={"backup_id": "assets_backup_20260216_120000"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert isinstance(data["steps"], list)
        assert "bustup" in data["steps"]
        assert "chibi" in data["steps"]
        assert "3d" in data["steps"]
        assert "rigging" in data["steps"]
        assert "animations" in data["steps"]

    async def test_remake_confirm_missing_backup(self, tmp_path):
        """Returns 404 when the specified backup_id does not exist."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        _setup_anima_with_assets(animas_dir, "target")

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-confirm",
                json={"backup_id": "assets_backup_20260101_000000"},
            )

        assert resp.status_code == 404
        assert "Backup not found" in resp.json()["detail"]

    async def test_remake_confirm_no_fullbody_preview(self, tmp_path):
        """Returns 400 when fullbody image is missing (preview not done)."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        target_dir = _setup_anima_with_assets(
            animas_dir, "target", with_fullbody=False,
        )

        # Create backup dir but no fullbody in assets
        backup_dir = target_dir / "assets_backup_20260216_120000"
        backup_dir.mkdir()

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/target/assets/remake-confirm",
                json={"backup_id": "assets_backup_20260216_120000"},
            )

        assert resp.status_code == 400
        assert "fullbody" in resp.json()["detail"].lower()


# ── E2E: Cancel Remake Preview ───────────────────────────


class TestCancelRemakePreview:
    """E2E tests for DELETE /api/animas/{name}/assets/remake-preview."""

    async def test_cancel_remake_restores_backup(self, tmp_path):
        """Cancel restores assets from the most recent backup and removes it."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        target_dir = _setup_anima_with_assets(animas_dir, "target")

        # Create a backup with original content
        backup_dir = target_dir / "assets_backup_20260216_120000"
        shutil.copytree(target_dir / "assets", backup_dir)

        # Write different content to the original backup to mark it
        (backup_dir / "prompt.txt").write_text(
            "original prompt from backup", encoding="utf-8",
        )

        # Now overwrite the current assets to simulate preview having changed them
        (target_dir / "assets" / "avatar_fullbody.png").write_bytes(
            b"new-preview-content",
        )

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(
                "/api/animas/target/assets/remake-preview",
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "restored"
        assert data["backup_used"] == "assets_backup_20260216_120000"

        # Verify assets were restored from backup
        restored_prompt = (target_dir / "assets" / "prompt.txt").read_text(
            encoding="utf-8",
        )
        assert restored_prompt == "original prompt from backup"

        # Backup directory should no longer exist (it was renamed to assets)
        assert not backup_dir.exists()

    async def test_cancel_remake_no_backup_returns_404(self, tmp_path):
        """Returns 404 when there is no backup to restore."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        _setup_anima_with_assets(animas_dir, "target")

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(
                "/api/animas/target/assets/remake-preview",
            )

        assert resp.status_code == 404
        assert "No backup found" in resp.json()["detail"]

    async def test_cancel_remake_uses_most_recent_backup(self, tmp_path):
        """When multiple backups exist, the most recent one is used."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        target_dir = _setup_anima_with_assets(animas_dir, "target")

        # Create two backups with different timestamps
        old_backup = target_dir / "assets_backup_20260216_100000"
        old_backup.mkdir()
        (old_backup / "prompt.txt").write_text("old backup", encoding="utf-8")

        new_backup = target_dir / "assets_backup_20260216_120000"
        new_backup.mkdir()
        (new_backup / "prompt.txt").write_text("new backup", encoding="utf-8")
        (new_backup / "avatar_fullbody.png").write_bytes(_FAKE_PNG)

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(
                "/api/animas/target/assets/remake-preview",
            )

        assert resp.status_code == 200
        assert resp.json()["backup_used"] == "assets_backup_20260216_120000"

        # Verify the newer backup was used
        restored_prompt = (target_dir / "assets" / "prompt.txt").read_text(
            encoding="utf-8",
        )
        assert restored_prompt == "new backup"


# ── E2E: CLI Dry-Run ─────────────────────────────────────


class TestCLIDryRun:
    """E2E tests for the CLI remake-assets --dry-run command."""

    def test_cli_dry_run(self, data_dir: Path, make_anima, capsys):
        """Dry-run prints planned actions without making API calls."""
        # Create target anima
        target_dir = make_anima("cli-target")
        assets_dir = target_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "prompt.txt").write_text(
            "1girl, test character", encoding="utf-8",
        )
        (assets_dir / "avatar_fullbody.png").write_bytes(_FAKE_PNG)

        # Create style-from anima
        style_dir = make_anima("cli-style")
        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(_FAKE_PNG)

        from cli.commands.remake_cmd import _run

        # Build a namespace that mirrors CLI args
        import argparse

        args = argparse.Namespace(
            anima="cli-target",
            style_from="cli-style",
            steps=None,
            prompt=None,
            vibe_strength=0.6,
            vibe_info_extracted=0.8,
            seed=None,
            no_backup=False,
            dry_run=True,
        )

        _run(args)

        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "No API calls were made" in captured.out
        assert "cli-target" in captured.out
        assert "cli-style" in captured.out
        assert "Backup:" in captured.out
        assert "fullbody" in captured.out.lower()

    def test_cli_dry_run_missing_style_anima(self, data_dir: Path, make_anima, capsys):
        """Dry-run exits early if style-from anima does not exist."""
        make_anima("cli-target")

        from cli.commands.remake_cmd import _run

        import argparse

        args = argparse.Namespace(
            anima="cli-target",
            style_from="ghost",
            steps=None,
            prompt=None,
            vibe_strength=0.6,
            vibe_info_extracted=0.8,
            seed=None,
            no_backup=False,
            dry_run=True,
        )

        _run(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "ghost" in captured.out

    def test_cli_dry_run_with_custom_steps(self, data_dir: Path, make_anima, capsys):
        """Dry-run respects --steps filter and shows only selected steps."""
        target_dir = make_anima("cli-target2")
        assets_dir = target_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "prompt.txt").write_text("test prompt", encoding="utf-8")
        (assets_dir / "avatar_fullbody.png").write_bytes(_FAKE_PNG)

        style_dir = make_anima("cli-style2")
        style_assets = style_dir / "assets"
        style_assets.mkdir(exist_ok=True)
        (style_assets / "avatar_fullbody.png").write_bytes(_FAKE_PNG)

        from cli.commands.remake_cmd import _run

        import argparse

        args = argparse.Namespace(
            anima="cli-target2",
            style_from="cli-style2",
            steps="fullbody,bustup",
            prompt=None,
            vibe_strength=0.6,
            vibe_info_extracted=0.8,
            seed=123,
            no_backup=False,
            dry_run=True,
        )

        _run(args)

        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "fullbody" in captured.out.lower()
        assert "bustup" in captured.out.lower()
        assert "Seed:" in captured.out
