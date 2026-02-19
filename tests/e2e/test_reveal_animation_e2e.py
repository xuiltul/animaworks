# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Anima birth reveal animation.

Tests the complete flow from asset generation API → WebSocket event emission,
verifying the contract that the frontend reveal animation relies on:
1. POST /animas/{name}/assets/generate emits anima.assets_updated event
2. The event payload includes asset filenames (avatar_*) for trigger detection
3. The workspace HTML serves correctly with reveal overlay elements
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _make_test_app(animas_dir: Path | None = None):
    """Create a test FastAPI app with mock supervisor and ws_manager."""
    from fastapi import FastAPI
    from server.routes.assets import create_assets_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_assets_router()
    app.include_router(router, prefix="/api")
    return app


def _make_pipeline_result(
    *,
    fullbody: bool = True,
    bustup: bool = True,
    chibi: bool = False,
):
    """Create a mock PipelineResult with specified asset availability."""
    mock = MagicMock()
    mock.fullbody_path = Path("/tmp/avatar_fullbody.png") if fullbody else None
    mock.bustup_path = Path("/tmp/avatar_bustup.png") if bustup else None
    mock.chibi_path = Path("/tmp/avatar_chibi.png") if chibi else None
    mock.model_path = None
    mock.rigged_model_path = None
    mock.animation_paths = {}
    mock.errors = []
    mock.to_dict.return_value = {"status": "done", "fullbody": fullbody, "bustup": bustup}
    return mock


# ── E2E: WebSocket event emission on asset generation ───


class TestRevealWebSocketEventE2E:
    """E2E: Verify anima.assets_updated WebSocket event structure
    that the frontend reveal animation handler depends on."""

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_assets_updated_event_emitted_on_generate(
        self, mock_pipeline_cls, tmp_path
    ):
        """Asset generation should broadcast anima.assets_updated event."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_result = _make_pipeline_result(fullbody=True, bustup=True)
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate",
                json={"prompt": "anime character"},
            )

        assert resp.status_code == 200

        # Verify WebSocket broadcast was called
        ws = app.state.ws_manager
        assert ws.broadcast.call_count >= 1

        # Find the anima.assets_updated event
        assets_events = []
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.assets_updated":
                assets_events.append(payload)

        assert len(assets_events) == 1
        event = assets_events[0]
        assert event["data"]["name"] == "alice"
        assert isinstance(event["data"]["assets"], list)

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_assets_updated_event_contains_avatar_filenames(
        self, mock_pipeline_cls, tmp_path
    ):
        """Event payload assets list should include avatar_* filenames
        so the frontend can detect when to trigger reveal animation."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_result = _make_pipeline_result(fullbody=True, bustup=True)
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/assets/generate",
                json={"prompt": "anime character"},
            )

        ws = app.state.ws_manager
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.assets_updated":
                assets = payload["data"]["assets"]
                # At least one avatar_* file should be in the list
                avatar_files = [a for a in assets if a.startswith("avatar_")]
                assert len(avatar_files) >= 1, (
                    f"Expected avatar_* files in assets list, got: {assets}"
                )
                break
        else:
            pytest.fail("anima.assets_updated event not found")

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_event_includes_errors_field(
        self, mock_pipeline_cls, tmp_path
    ):
        """Event should include errors list for frontend error handling."""
        anima_dir = tmp_path / "bob"
        anima_dir.mkdir()

        mock_result = _make_pipeline_result(fullbody=True, bustup=False)
        mock_result.errors = ["Bustup generation failed"]
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/bob/assets/generate",
                json={"prompt": "anime character"},
            )

        ws = app.state.ws_manager
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.assets_updated":
                assert "errors" in payload["data"]
                break


# ── E2E: Workspace page serves reveal overlay ───────────


class TestRevealWorkspacePageE2E:
    """E2E: Verify the workspace HTML page includes the reveal overlay
    elements that the JS reveal module depends on."""

    def test_workspace_html_has_reveal_overlay(self):
        """The served workspace page must contain the reveal overlay div."""
        html_path = (
            Path(__file__).resolve().parents[2]
            / "server"
            / "static"
            / "workspace"
            / "index.html"
        )
        html = html_path.read_text(encoding="utf-8")

        # Required element IDs for reveal.js
        assert 'id="wsRevealOverlay"' in html
        assert 'id="wsRevealAvatar"' in html
        assert 'id="wsRevealText"' in html

    def test_workspace_css_has_reveal_styles(self):
        """The workspace stylesheet must contain reveal animation styles."""
        css_path = (
            Path(__file__).resolve().parents[2]
            / "server"
            / "static"
            / "workspace"
            / "style.css"
        )
        css = css_path.read_text(encoding="utf-8")

        assert ".ws-reveal-overlay" in css
        assert "@keyframes ws-reveal-flash" in css
        assert "@keyframes ws-reveal-content" in css

    def test_reveal_js_module_exists(self):
        """The reveal.js module must exist for the import in app.js."""
        reveal_path = (
            Path(__file__).resolve().parents[2]
            / "server"
            / "static"
            / "workspace"
            / "modules"
            / "reveal.js"
        )
        assert reveal_path.exists()
        content = reveal_path.read_text(encoding="utf-8")
        assert "export async function playReveal" in content

    def test_app_js_imports_reveal(self):
        """app.js must import playReveal from reveal.js."""
        app_js_path = (
            Path(__file__).resolve().parents[2]
            / "server"
            / "static"
            / "workspace"
            / "modules"
            / "app.js"
        )
        content = app_js_path.read_text(encoding="utf-8")
        assert 'import { playReveal } from "./reveal.js"' in content
